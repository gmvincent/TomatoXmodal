import numpy as np
import os
import glob
import torch

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

class MeshDataset(torch.utils.data.Dataset):
    """
    Dataset for loading PLY meshes and preparing SpiralNet structures.
    """

    def __init__(
        self,
        root,
        seq_lengths=[21,21,21],
        ds_factors=[4, 4, 4],
        graph=False,
    ):
        self.graph = graph
        self.seq_lengths = seq_lengths
        self.ds_factors = ds_factors
        
        all_files = glob.glob(os.path.join(root, "**", "*.ply"), recursive=True)
        
        candidate_files = [
            f for f in all_files
            if 'trian' not in os.path.basename(f)
            and not any(d in f for d in ["DAI22", "DAI25", "DAI28"])
        ]

        # Keep only valid PLYs
        self.mesh_files = []
        for f in candidate_files:
            verts, faces, feats = load_ply(f)
            if verts is not None:
                self.mesh_files.append(f)

        
    def __len__(self):
        return len(self.mesh_files)
    
    def build_mesh_pyramid(self, verts, faces):
        vertices = [verts]
        face_list = [faces]
        down_transforms = []

        # Create hierarchy
        for factor in self.ds_factors:
            v_hi, f_hi = vertices[-1], face_list[-1]
            v_lo, f_lo = mesh_downsample(v_hi, f_hi, factor)
            D = compute_downsample_matrix(v_hi, v_lo)

            vertices.append(v_lo)
            face_list.append(f_lo)
            down_transforms.append(D)

        # Spirals (only up to second-to-last level)
        spirals = [
            preprocess_spiral(face_list[i], self.seq_lengths[i], vertices[i])
            for i in range(len(self.seq_lengths))
        ]

        return vertices, face_list, down_transforms, spirals

    def __getitem__(self, idx):
        fp = self.mesh_files[idx]

        verts, faces, feats = load_ply(fp)
        label = int(os.path.basename(fp).split("_")[1].lstrip("T"))
    
        if self.graph:
            data = mesh_to_graph(verts, faces, feats)
            data.y = torch.tensor(label, dtype=torch.long)
            return data
        
        try:
            vertices, face_list, down_transforms, spirals = self.build_mesh_pyramid(verts, faces)
        except Exception as e:
            print(f"Skipping mesh {fp} due to error: {e}")
            return self.__getitem__((idx+1) % len(self.mesh_files))  # skip to next
            
        
        return {
            "features": torch.tensor(feats, dtype=torch.float32),
            "spiral_indices": [torch.tensor(s, dtype=torch.long) for s in spirals],
            "down_transform": [to_sparse(D) for D in down_transforms],
            "label": label
        }