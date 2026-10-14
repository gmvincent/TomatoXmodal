import os
import bisect
import torch
import numpy as np
import pandas as pd
from PIL import Image

from torchvision import tv_tensors
from torchvision.transforms import v2

classes = {
    0: 'control',
    1: 'bacterial_spot',
    2: 'septoria_leaf_spot',
    3: 'early_blight'
}

seg_classes = {
    0: 'background',
    1: 'canopy',
    2: 'lesion',
}

reg_scale = [
    1.0,  # <1%,    score=1
    3.0,  # 1-3%,   score=2
    6.0,  # 3-6%,   score=3
    12.0, # 6-12%,  score=4
    25.0, # 12-25%, score=5
    50.0, # 25-50%, score=6
    75.0, # 50-75%, score=7
    87.0, # 75-87%, score=8
    94.0, # 87-94%, score=9
    97.0, # 94-97%, score=10
]


class TomatoTask2D(torch.utils.data.Dataset):
    def __init__(self, 
                 root: str, 
                 split: str,
                 transform=None,
                 depth: bool=True,
                 num_tasks: str="1", 
                 task: str= "classification",
                 num_views: int=-1, 
                 reg_target: str="hb", # "hb" or "pct"
                 ):
        """
        Args:
            root (str): directory containing image files
            split (str): train-val-test split subdirectory
            transform (callable, optional): Optional transform to apply to each voxel
            depth (Bool, optional): Indicates if depth maps should be included in feature set
            num_tasks (str, optional): indicates what number of tasks 
            task (str, optional): indicates the type of task (classification, segmentation, both)
            num_view (int, optional): indicates the `view` of the plant used. -1 is all four views and 0-3 point
                to specific rotations of the plant
        """
        self.path = os.path.join(root, split)
        self.transform = transform
        self.include_depth = depth
        self.reg_target = reg_target
        self.num_tasks = num_tasks
        self.tasks = [task] if isinstance(task, str) else list(task)
        
        self.depth_global_min = 353.6300
        self.depth_global_max = 1219.0477
        
        exclude_days=("DAI3") #"DAI3", "DAI22", "DAI25", "DAI28")
        
        all_files = sorted(os.listdir(self.path))
        self.img_files = [
            f for f in all_files
            if f.endswith(".png") and f.split("_")[2] not in exclude_days
        ]
        
        if num_views >= 0:
            self.img_files = [f for f in self.img_files if int(f.split("_")[0])%10 == num_views]
        
        if self.include_depth:
            self.depth_files = {}
            valid_img_files = []
            for f in self.img_files:
                base_name = os.path.splitext(f)[0]
                depth_name = base_name + ".npy"
                if os.path.exists(os.path.join(self.path, depth_name)):
                    self.depth_files[base_name] = depth_name
                    valid_img_files.append(f)
            self.img_files = valid_img_files
        
        
        self._seg_count = self.tasks.count("segmentation")
        if self._seg_count > 2:
            raise ValueError("At most 2 'segmentation' entries are supported (raw, or canopy+lesion).")
        self._needs_mask = self._seg_count > 0 or "regression" in self.tasks
        
        if self._needs_mask:
            self.mask_files = {}
            for f in self.img_files:
                base_name = os.path.splitext(f)[0]
                mask_name = base_name + "_mask.png"
                if os.path.exists(os.path.join(self.path, "masks", mask_name)):
                    self.mask_files[base_name] = mask_name
            
        self.labels = [int(f.split("_")[1].lstrip("T")) for f in self.img_files]
        
    def __len__(self):
        return len(self.img_files)

    def get_label(self, idx: int) -> int:
        # TODO: set up other task and multi-task get_label()
        return int(os.path.basename(self.img_files[idx]).split("_")[1].lstrip("T"))
    
    def _load_image(self, img_name: str) -> np.ndarray:
        img_path = os.path.join(self.path, img_name)
        
        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0
        
        # Swap R and B to fix channel order
        img = img[:, :, [2, 1, 0]]
        return img
 
    def _load_depth(self, base_name: str, target_hw) -> np.ndarray:
        depth_file = self.depth_files[base_name] 
        
        depth_path = os.path.join(self.path, depth_file)
        d = np.load(depth_path).astype(np.float32)
        
        # Normalize depth
        valid = np.isfinite(d)
        d_norm = np.zeros_like(d, dtype=np.float32)
        d_norm[valid] = (d[valid] - self.depth_global_min) / (self.depth_global_max - self.depth_global_min)
        d_norm = np.clip(d_norm, 0, 1)
        d_norm[~valid] = 1.0
        
        d_norm = np.expand_dims(d_norm, axis=2)
        
        if target_hw != d_norm.shape[:2]:
            raise ValueError(f"Shape mismatch: {target_hw} vs {d_norm.shape}")
        return d_norm
        
    def _load_raw_mask(self, base_name: str) -> np.ndarray:
        mask_path = os.path.join(self.path, "masks", self.mask_files[base_name])
        
        mask = Image.open(mask_path).convert("L")
        return np.array(mask, dtype=np.int64)
 
    @staticmethod
    def _decompose_mask(raw_mask: np.ndarray):
        """raw_mask holds values from `seg_classes` (0=background, 1=canopy, 2=lesion)."""
        canopy_mask = np.isin(raw_mask, (1, 2)).astype(np.float32)
        lesion_mask = (raw_mask == 2).astype(np.float32)
        return canopy_mask, lesion_mask
 
    def _severity(self, canopy_mask: np.ndarray, lesion_mask: np.ndarray) -> float:
        canopy_px = float(canopy_mask.sum())
        lesion_px = float(lesion_mask.sum())
        pct = min(100.0 * lesion_px / canopy_px, 100.0) if canopy_px > 0 else 0.0
        
        if self.reg_target == "pct":
            return pct
        
        elif self.reg_target == "hb":
            if pct <= 0:
                return 0.0
            elif pct >= 100:
                return 12.0
            else:
                grade = bisect.bisect_right(reg_scale, pct) +1

            return float(grade)
            
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        base_name = os.path.splitext(img_name)[0]
        
        img_np = self._load_image(img_name)
        rgb = tv_tensors.Image(torch.from_numpy(img_np).permute(2, 0, 1))
        
        depth = None
        if self.include_depth:
            d = self._load_depth(base_name, img_np.shape[:2])
            depth = tv_tensors.Mask(torch.from_numpy(d).permute(2, 0, 1).float())
        
        raw_mask, regression_score = None, None
        if self._needs_mask:
            raw_mask_np = self._load_raw_mask(base_name)
            
            if "regression" in self.tasks:
                canopy_orig, lesion_orig = self._decompose_mask(raw_mask_np)
                regression_score = self._severity(canopy_orig, lesion_orig)   
                
            raw_mask = tv_tensors.Mask(torch.from_numpy(raw_mask_np))
        
        # transforms    
        if self.transform:
            # None entries pass through untouched
            rgb, depth, raw_mask = self.transform(rgb, depth, raw_mask)
        else:
            rgb = v2.functional.to_dtype(rgb, torch.float32, scale=True)

        img = torch.cat([rgb, depth.as_subclass(torch.Tensor)], dim=0) if depth is not None else rgb
        
        canopy_mask = lesion_mask = None
        if raw_mask is not None:
            raw_mask = raw_mask.as_subclass(torch.Tensor).long()
            if self._seg_count == 2:
                canopy_mask, lesion_mask = self._decompose_mask(raw_mask.numpy())
        
        output = [img.float()]
        seg_emitted = 0
        for t in self.tasks:
            if t == "classification":
                output.append(self.labels[idx])
            elif t == "segmentation":
                if self._seg_count == 1:
                    output.append(torch.as_tensor(raw_mask))
                else:
                    m = canopy_mask if seg_emitted == 0 else lesion_mask
                    output.append(torch.as_tensor(m))
                    seg_emitted += 1
 
            elif t == "regression":
                output.append(torch.tensor(regression_score, dtype=torch.float32))
        
        return tuple(output)