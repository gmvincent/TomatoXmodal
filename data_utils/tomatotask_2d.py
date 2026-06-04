import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.preprocessing import MinMaxScaler

classes = {
    0: 'control',
    1: 'bacterial_spot',
    2: 'septoria_leaf_spot',
    3: 'early_blight'
}

class TomatoTask2D(torch.utils.data.Dataset):
    def __init__(self, 
                 root: str, 
                 split: str,
                 transform=None,
                 depth: bool=True,
                 num_views: int=-1
                 ):
        """
        Args:
            root (str): directory containing image files
            split (str): train-val-test split subdirectory
            transform (callable, optional): Optional transform to apply to each voxel
            depth (Bool, optional): Indicates if depth maps should be included in feature set
            num_view (int, optional): indicates the `view` of the plant used. -1 is all four views and 0-3 point
                to specific rotations of the plant
        """
        self.path = os.path.join(root, split)
        self.transform = transform
        self.include_depth = depth
        
        self.depth_global_min = 353.6300
        self.depth_global_max = 1219.0477
        
        exclude_days=("DAI3", "DAI22", "DAI25", "DAI28")
        
        all_files = os.listdir(self.path)
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
            
        self.labels = [int(f.split("_")[1].lstrip("T")) for f in self.img_files]
        
    def __len__(self):
        return len(self.img_files)

    def get_label(self, idx: int) -> int:
        return int(os.path.basename(self.img_files[idx]).split("_")[1].lstrip("T"))
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.path, img_name)
        
        img = Image.open(img_path).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0
        
        # Swap R and B to fix channel order
        img = img[:, :, [2, 1, 0]]
    
        if self.include_depth:
            base_name = os.path.splitext(img_name)[0]
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
            
            if img.shape[:2] != d_norm.shape[:2]:
                raise ValueError(f"Shape mismatch: {img.shape} vs {d_norm.shape}")
            img = np.concatenate([img, d_norm], axis=2)
            
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        return img.float(), label