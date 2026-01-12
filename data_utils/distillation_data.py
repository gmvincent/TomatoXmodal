import numpy as np
import os
import glob
import torch
from PIL import Image

from plyfile import PlyData
from torch_geometric.data import Data, DataLoader

from utils.mesh_helpers import mesh_downsample, compute_downsample_matrix, preprocess_spiral, to_sparse

classes = {
    0: 'control',
    1: 'bacterial_spot',
    2: 'septoria_leaf_spot',
    3: 'early_blight'
}

def load_ply(path):
    """
    Loads a PLY mesh, applies a z > 5 mask, remaps faces, 
    and returns masked vertices, faces, and per-vertex features.
    """
    try:
        ply = PlyData.read(path)

        vertex_data = ply['vertex'].data
        x = vertex_data['x']
        y = vertex_data['y']
        z = vertex_data['z']

        mask = z > 5
        orig_idx = np.arange(len(z))
        new_idx = np.cumsum(mask) - 1

        # Filter vertices
        vertices = np.stack([x[mask], y[mask], z[mask]], axis=-1).astype(np.float32)

        face_data = ply['face'].data
        if 'vertex_indices' in face_data.dtype.names:
            faces_raw = np.vstack(face_data['vertex_indices'])
        elif 'vertex_index' in face_data.dtype.names:
            faces_raw = np.vstack(face_data['vertex_index'])
        else:
            raise ValueError("No face index field in PLY.")

        # Keep faces where *all* referenced vertices survived masking
        face_mask = mask[faces_raw].all(axis=1)
        faces_filtered = faces_raw[face_mask]

        # Remap indices to new masked vertex order
        faces = new_idx[faces_filtered]

        colors = [
            vertex_data[c][mask]
            for c in ['red','green','blue','alpha']
            if c in vertex_data.dtype.names
        ]
        if colors:
            colors = np.stack(colors, axis=-1).astype(np.float32)
            # Normalize RGB
            if colors[..., :3].max() > 1.0:
                colors[..., :3] /= 255.0
        else:
            colors = None

        if "scalar_nir" in vertex_data.dtype.names:
            nir = vertex_data["scalar_nir"][mask].astype(np.float32)
            nir /= 255.0
        else:
            nir = None

        features = np.concatenate(
            [f for f in [colors, nir[:, None] if nir is not None else None] if f is not None],
            axis=-1
        )

        return vertices, faces, features

    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None, None, None

def mesh_to_graph(verts, faces, features):
    """
    Convert mesh vertices/faces to PyG graph (Data object).
    """
    row, col = [], []

    for a, b, c in faces:
        row += [a, b, b, c, c, a]
        col += [b, a, c, b, a, c]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index)

class XModalDataset(torch.utils.data.Dataset):
    """
    Dataset for loading PLY meshes and preparing SpiralNet structures.
    """

    def __init__(
        self,
        root,
         transform=None,
    ):
        self.transform = transform

        all_files = glob.glob(os.path.join(root, "**", "*.ply"), recursive=True)
        
        candidate_files = [
            f for f in all_files
            if 'trian' not in os.path.basename(f)
            and not any(d in f for d in ["DAI22", "DAI25", "DAI28"])
        ]
        
        # Keep only valid PLYs
        self.mesh_files = []
        self.img_files = []
        for f in candidate_files:
            verts, faces, feats = load_ply(f)
            if verts is None:
                continue
            
            
            base = "_".join(os.path.basename(f).split("_")[:-1])
            base = base.replace("X", "0")
            try:
                img_candidates = glob.glob(os.path.join("/home/gmvincen/cmwilli5_drive/gmvincen_data/tomato_diseases/cropped_rgbd",
                                                    base + "*.png"))
                
                if len(img_candidates) == 0:
                    raise FileNotFoundError

                with Image.open(img_candidates[0]) as im:
                    im.verify() 
            
                self.mesh_files.append(f)
                self.img_files.append(img_candidates[0])
                
            except (FileNotFoundError, OSError, ValueError) as e:
                print(f"Skipping mesh {f}: image issue ({e})")
                continue
        
        
    def __len__(self):
        return len(self.mesh_files)
    
    def __getitem__(self, idx):
        fp = self.mesh_files[idx]
        img = self.img_files[idx]
        
        img = Image.open(img).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]
        img = torch.from_numpy(img)
        if img.shape[0] != 3:  # convert HWC to CHW
            img = img.permute(2, 0, 1)
        
        verts, faces, feats = load_ply(fp)
        label = int(os.path.basename(fp).split("_")[1].lstrip("T"))

        data = mesh_to_graph(verts, faces, feats)       
        
        if self.transform:
            img = self.transform(img)
        
        return data, img, label