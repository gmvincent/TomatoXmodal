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

class TomatoSideViewDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, depth=True):
        """
        Args:
            root (str): Directory containing .xlsx file
            transform (callable, optional): Optional transform to apply to each voxel
            depth (Bool, optional): Indicates if depth maps should be included in feature set
        """
        self.root = root
        self.transform = transform
        self.include_depth = depth
        
        self.depth_global_min = 353.6300
        self.depth_global_max = 1218.8663
        
        exclude_days=("DAI22", "DAI25", "DAI28")
        
        all_files = os.listdir(root)
        self.img_files = [
            f for f in all_files
            if f.endswith(".png")
            and not any(tag in f for tag in exclude_days)
        ]
        
        self.img_files = [f for f in self.img_files if int(f.split("_")[0])%10 == 0]
        
        if self.include_depth:
            self.depth_files = {
                os.path.splitext(f)[0]: os.path.splitext(f)[0] + ".npy"
                for f in self.img_files
                if os.path.exists(os.path.join(root, os.path.splitext(f)[0] + ".npy"))
            }
            
            self._pre_load_data()
        else:
            self.depth_files = {}
            
        self.labels = [int(f.split("_")[1].lstrip("T")) for f in self.img_files]
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.root, img_name)
        
        img = Image.open(img_path).convert("RGB")
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
        img = img.permute(2, 0, 1)
                
        if self.include_depth:
            d = self.depth_files[os.path.splitext(img_name)[0]]  # tensor
            img = torch.cat([img, d], dim=0)
            
        if self.transform:
            img = self.transform(img)
            
        label = self.labels[idx]

        return img.float(), label
    
    def _pre_load_data(self):
        
        for img_name, depth_file in self.depth_files.items():
            depth_path = os.path.join(self.root, depth_file)
            d = np.load(depth_path).astype(np.float32)
            
            valid = np.isfinite(d)
            d_norm = np.zeros_like(d, dtype=np.float32)
            d_norm[valid] = (d[valid] - self.depth_global_min) / (self.depth_global_max - self.depth_global_min)
            d_norm = np.clip(d_norm, 0, 1)
            d_norm[~valid] = 1.0
            
            d_norm = torch.from_numpy(d_norm).unsqueeze(0)  # [1, H, W]

            self.depth_files[img_name] = d_norm