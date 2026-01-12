import os
import torch
import random
import numpy as np
import glob

import scipy

classes = {
    0: 'control',
    1: 'bacterial_spot',
    2: 'septoria_leaf_spot',
    3: 'early_blight'
}

class VoxelDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_size=(320, 320, 384), include_scalars=False):
        """
        Args:
            root (str): Directory to search for .npy voxel files
            transform (callable, optional): Optional transform to apply to each voxel
            target_size (tuple): The desired output size for the voxel grid
            include_scalars (bool): If True, also load associated scalar dictionary
        """
        self.target_size = target_size
        self.include_scalars = include_scalars
        self.transform = transform

        # Collect all voxel files
        all_files = glob.glob(os.path.join(root, "**", "*.npy"), recursive=True)
        voxel_files = [
            f for f in all_files
            if '3DRGBN' in os.path.basename(f)
            and not f.endswith("_dict.npy")
            and not any(d in f for d in ["DAI22", "DAI25", "DAI28"])
        ]

        self.voxel_grids = []
        self.labels = []
        self.scalar_paths = []

        # Load all voxels into RAM
        for f in voxel_files:
            label = int(os.path.basename(f).split("_")[1].lstrip("T"))
            if label == 1:
                continue

            self.labels.append(label)
            voxel_grid = np.load(f).astype(np.float32)
            self.voxel_grids.append(voxel_grid)

            if include_scalars:
                scalar_path = f.replace(".npy", "_dict.npz")
                self.scalar_paths.append(scalar_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        voxel_grid = self.voxel_grids[idx]

        if self.transform:
            voxel_grid = self.transform(voxel_grid)

        # Transpose to (C,W,H,D)
        voxel_grid = np.transpose(voxel_grid, (3,0,1,2))
        voxel_grid = torch.from_numpy(voxel_grid).float().clone()

        if self.include_scalars:
            scalar_data = np.load(self.scalar_paths[idx])
            scalar_dict = {k: torch.as_tensor(v).clone() for k,v in scalar_data.items()}
            return voxel_grid, scalar_dict, self.labels[idx]

        return voxel_grid, self.labels[idx]
    

def augment_voxel(voxel_grid):
    """
    Applies 3D data augmentation similar to 2D image augmentation.
    
    Args:
        voxel_grid (np.array): The voxel grid as a NumPy array.
    
    Returns:
        np.array: The augmented voxel grid.
    """
    # Randomly rotate the grid in 90-degree increments
    rot_axes = random.choice([0, 1, 2])
    rot_k = random.randint(0, 3)
    if rot_axes == 0:
        voxel_grid = np.rot90(voxel_grid, k=rot_k, axes=(1, 2))
    elif rot_axes == 1:
        voxel_grid = np.rot90(voxel_grid, k=rot_k, axes=(0, 2))
    else:
        voxel_grid = np.rot90(voxel_grid, k=rot_k, axes=(0, 1))

    # Random color jitter
    brightness_factor = np.random.uniform(0.8, 1.2)
    contrast_factor = np.random.uniform(0.8, 1.2)
    
    # Apply to each channel
    voxel_grid = voxel_grid * brightness_factor
    voxel_grid = voxel_grid * contrast_factor
    voxel_grid = np.clip(voxel_grid, 0, 255)

    return voxel_grid