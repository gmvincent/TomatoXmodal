import os
import glob
import torch
import numpy as np

import open3d as o3d 
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

from data_utils.spiral_helpers import compute_downsample_matrix, preprocess_spiral, sparse_to_torch

classes = {
    0: 'control',
    1: 'bacterial_spot',
    2: 'septoria_leaf_spot',
    3: 'early_blight'
}
   
def mesh_to_graph(verts, faces, features):
    """
    Convert mesh vertices/faces to PyG graph (Data object).
    """
    faces = torch.as_tensor(faces, dtype=torch.long)
    
    edges = torch.cat([
        faces[:, [0,1]],
        faces[:, [1,0]],
        faces[:, [1,2]],
        faces[:, [2,1]],
        faces[:, [2,0]],
        faces[:, [0,2]],
    ], dim=0)

    edge_index, _ = coalesce(edges.t().contiguous(), None)
    x = torch.as_tensor(features, dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index)

class TomatoTask3D(torch.utils.data.Dataset):
    """
    PLY mesh dataset supporting four representations:
 
      "pcd"    — point cloud  [N, 3 (+ C) + 3]
      "mesh"   — face-centric dict (centers, normals, face-rings)
      "graph"  — PyG Data with edge_index
      "spiral" — SpiralNet++ pyramid (multi-level spiral indices + pool matrices)
    """

    def __init__(
        self,
        root: str,
        split: str,
        representation: str="mesh", # "mesh", "spiral", "graph", "pcd",
        include_spectral: bool=True,
        seq_lengths=[21,21,21],
        ds_factors=[4, 4, 4],
        target_faces=None,
        num_tasks: str="1", 
        seed: int = 42
    ):
        self.representation = representation
        self.include_spectral = include_spectral
        self.seq_lengths = seq_lengths
        self.ds_factors = ds_factors
        self.num_tasks = num_tasks
        self.seed = seed
        
        if isinstance(target_faces, str):
            target_faces = None if target_faces.lower() == "none" else int(target_faces)
        self.target_faces = target_faces
            
        exclude_days=("DAI3",) #, "DAI22", "DAI25", "DAI28")
        
        all_files = sorted(glob.glob(os.path.join(root, split, "**", "*.ply"), recursive=True))

        self.mesh_files = [
            f for f in all_files
            if 'trian' not in os.path.basename(f)
            and f.split("_")[2] not in exclude_days
        ]

    @staticmethod
    def load_ply(path):
        """
        Load a PLY mesh, apply a z > 6 mask, remap faces, and return
        (vertices, faces, per-vertex features).  Returns (None, None, None)
        on failure.
        """
        try:
            ply = PlyData.read(path)

            vertex_data = ply["vertex"].data
            x, y, z = vertex_data["x"], vertex_data["y"], vertex_data["z"]
            
            # Filter vertices
            mask = z > 6
            new_idx = np.cumsum(mask) - 1
            vertices = np.stack([x[mask], y[mask], z[mask]], axis=-1).astype(np.float32)

            face_data = ply["face"].data
            if "vertex_indices" in face_data.dtype.names:
                faces_raw = np.vstack(face_data["vertex_indices"])
            elif "vertex_index" in face_data.dtype.names:
                faces_raw = np.vstack(face_data["vertex_index"])
            else:
                raise ValueError("No face index field found in PLY.")

            # Keep faces where all referenced vertices survived masking
            face_mask = mask[faces_raw].all(axis=1)
            faces = new_idx[faces_raw[face_mask]]

            colors = [vertex_data[c][mask] for c in ("red", "green", "blue") if c in vertex_data.dtype.names]
            if colors:
                colors = np.stack(colors, axis=-1, dtype=np.float32)
                # Normalize RGB
                if colors[..., :3].max() > 1.0: colors[..., :3] /= 255.0
            else:
                colors = None

            if "scalar_nir" in vertex_data.dtype.names:
                nir = vertex_data["scalar_nir"][mask].astype(np.float32)[:, None] 
                if nir.max() > 1.0: nir / 255.0
            else:
                nir = None

            feature_list = [f for f in (colors, nir) if f is not None]
            features = (
                np.concatenate(feature_list, axis=-1)
                if feature_list
                else np.empty((vertices.shape[0], 0), dtype=np.float32)
            )

            return vertices, faces, features

        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return None, None, None
    
    def __len__(self):
        return len(self.mesh_files)
    
    def __getitem__(self, idx):
        fp = self.mesh_files[idx]
        verts, faces, feats = self.load_ply(fp)
        
        label = int(os.path.basename(fp).split("_")[1].lstrip("T"))
        label = torch.tensor(label, dtype=torch.long)
        
        if self.target_faces is not None:
            verts, faces, feats = self.simplify_mesh(verts, faces, feats, self.target_faces)
 
        if self.representation == "pcd":
            return self._build_pcd(verts, faces, feats, label)
        elif self.representation == "mesh":
            return self._build_mesh(verts, faces, feats, label)
        elif self.representation == "graph":
            return self._build_graph(verts, faces, feats, label)
        elif self.representation == "spiral":
            try:
                return self._build_spiral(verts, faces, feats, label)
            except Exception as exc:
                print(f"Skipping {fp} (spiral build failed): {exc}")
                return self.__getitem__((idx + 1) % len(self.mesh_files))
        else:
            raise ValueError(f"Unknown representation: {self.representation!r}")
    
    def _build_pcd(self, verts, faces, feats, label):
        """Point-cloud representation: [N, 3 (+ C) + 3]."""
        _, normals = self._compute_normals(verts, faces)
 
        if self.target_faces is not None:
            verts, feats, normals = self._normalize_point_count(
                verts, feats, normals, target_n=self.target_faces, seed=self.seed
            )
 
        points = [torch.as_tensor(verts, dtype=torch.float32)]
        if self.include_spectral and feats.shape[1] > 0:
            points.append(torch.as_tensor(feats, dtype=torch.float32))
        points.append(torch.as_tensor(normals, dtype=torch.float32))
 
        return torch.cat(points, dim=1), label
    
    def _build_mesh(self, verts, faces, feats, label):
        """
        Mesh representation.  Spectral features are always included in
        vert_feats / face_feats; callers that don't want them can simply
        ignore those keys.
        """
        centers = self._compute_face_centers(verts, faces)
        normals, _ = self._compute_normals(verts, faces)
        rings = self._compute_face_rings(faces)
        face_feats = feats[faces].mean(axis=1)
 
        return {
            "verts":      torch.from_numpy(verts).float(),
            "vert_feats": torch.from_numpy(feats).float().T,       # [C, V]
            "face_feats": torch.from_numpy(face_feats).float().T,  # [C, F]
            "faces":      torch.from_numpy(faces).long(),
            "centers":    torch.from_numpy(centers).float().T,     # [3, F]
            "normals":    torch.from_numpy(normals).float().T,     # [3, F]
            "ring_1":     torch.from_numpy(rings[0]).long(),
            "ring_2":     torch.from_numpy(rings[1]).long(),
            "ring_3":     torch.from_numpy(rings[2]).long(),
        }, label
    
    def _build_graph(self, verts, faces, feats, label):
        """Graph representation: node features are XYZ (+ spectral if requested)."""
        if self.include_spectral and feats.shape[1] > 0:
            x = np.concatenate([verts, feats], axis=-1)
        else:
            x = verts
 
        data = mesh_to_graph(verts, faces, x)
        data.y = label
        return data
    
    def _build_spiral(self, verts, faces, feats, label):
        """
        SpiralNet++ pyramid.
 
        Builds a coarse-to-fine (actually fine-to-coarse) mesh hierarchy:
          Level 0  — original mesh (after optional decimation)
          Level 1  — mesh decimated by ds_factors[0]
          Level 2  — level-1 mesh decimated by ds_factors[1]
          …
 
        At each level l we precompute:
          • ``spiral_indices[l]``   int64  [V_l, seq_lengths[l]]
          • ``down_transform[l]``   sparse float32 [V_{l+1}, V_l]
 
        The feature at level l is:
          ``feats_l = X (XYZ + optional spectral)``   propagated via D.
 
        Returns
        -------
        dict with keys:
          "features"        list[Tensor [1, V_l, C]]   one per level
          "spiral_indices"  list[Tensor [V_l, seq_l]]  one per level
          "down_transform"  list[sparse Tensor]         len = num_levels - 1
          "label"           Tensor scalar
        """
        
        verts_levels = [verts]
        faces_levels = [faces]
        down_matrices: list[csr_matrix] = []
 
        for factor in self.ds_factors:
            v_hi, f_hi = verts_levels[-1], faces_levels[-1]
            target = max(4, len(f_hi) // factor)
 
            v_lo, f_lo, _ = self.simplify_mesh(v_hi, f_hi,
                                                np.zeros((len(v_hi), 0), dtype=np.float32),
                                                target)
            if len(f_lo) == 0 or len(v_lo) < 4:
                # Pyramid has collapsed — stop here
                break
 
            D = compute_downsample_matrix(v_hi, v_lo)
            verts_levels.append(v_lo)
            faces_levels.append(f_lo)
            down_matrices.append(D)
 
        num_levels = len(verts_levels)
 
        # Base feature: XYZ + spectral (if requested)
        if self.include_spectral and feats.shape[1] > 0:
            base_feats = np.concatenate([verts, feats], axis=-1)   # [V0, 3+C]
        else:
            base_feats = verts.copy()                               # [V0, 3]
 
        feat_levels = [base_feats]
        for D in down_matrices:
            feat_levels.append(D @ feat_levels[-1])                # [V_l, C]
 
        # seq_lengths and dilations are cycled if shorter than num_levels
        spiral_indices = []
        for l in range(num_levels):
            seq = self.seq_lengths[l % len(self.seq_lengths)]
            dil = self.dilations[l % len(self.dilations)]
            sp  = preprocess_spiral(verts_levels[l], faces_levels[l],
                                    seq_len=seq, dilation=dil)
            spiral_indices.append(torch.as_tensor(sp, dtype=torch.long))
 
        features = [
            torch.as_tensor(f, dtype=torch.float32).unsqueeze(0)   # [1, V_l, C]
            for f in feat_levels
        ]
        down_transforms = [sparse_to_torch(D) for D in down_matrices]
 
        return {
            "features":       features,          # list[Tensor], len = num_levels
            "spiral_indices": spiral_indices,    # list[Tensor], len = num_levels
            "down_transform": down_transforms,   # list[sparse Tensor], len = num_levels - 1
            "label":          label,
        }

    @staticmethod
    def _compute_normals(verts, faces):
        """
        Compute per-vertex normals by averaging adjacent face normals.
        verts: [V,3]
        faces: [F,3]
        """
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0, axis=1)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8

        vertex_normals = np.zeros_like(verts)

        # accumulate face normals onto vertices
        np.add.at(vertex_normals, faces[:, 0], face_normals)
        np.add.at(vertex_normals, faces[:, 1], face_normals)
        np.add.at(vertex_normals, faces[:, 2], face_normals)
        
        vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-8
        return face_normals.astype(np.float32), vertex_normals.astype(np.float32)
    
    @staticmethod
    def _compute_face_centers(verts, faces):
        return verts[faces].mean(axis=1).astype(np.float32)

    @staticmethod
    def _compute_face_rings(faces):
        F = faces.shape[0]
        
        # Build edge from face incidence using all three directed edges per face
        edges = np.concatenate([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ], axis=0)
        
        edges = np.sort(edges, axis=1)        # canonical ordering
        face_ids = np.repeat(np.arange(F), 3) # corresponding face ids
        
        # Assign a unique id to each distinct edge via lexsort
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges = edges[order]
        face_ids = face_ids[order]

        # Find edge group boundaries
        diff = np.any(np.diff(edges, axis=0), axis=1)
        edge_id = np.concatenate([[0], np.cumsum(diff)]) # group label per half-edge
        
        # Two faces sharing an edge are adjacent
        E = edge_id[-1] + 1
        
        # Build edge x face incidence (E x F), then adj = inc @ inc.T - diag
        inc = coo_matrix(
            (np.ones(len(face_ids), dtype=np.uint8), (edge_id, face_ids)),
            shape=(E, F),
        ).tocsr()
        adj = (inc.T @ inc).tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()
        adj = (adj > 0).astype(np.uint8)
        
        def extract_ring(a, k):
            """Pad / truncate each row of sparse adjacency to exactly k neighbours."""
            rings = np.full((F, k), fill_value=np.arange(F)[:, None], dtype=np.int64)

            cx = a.tocoo()
            # Group cols by row using np.split after sorting
            order_ = np.argsort(cx.row, kind="stable")
            rows_, cols_ = cx.row[order_], cx.col[order_]
            splits = np.searchsorted(rows_, np.arange(F + 1))
            for fi in range(F):
                nbrs = cols_[splits[fi]:splits[fi + 1]]
                take = min(len(nbrs), k)
                if take > 0:
                    rings[fi, :take] = nbrs[:take]
            return rings

        ring1 = extract_ring(adj, 3)
 
        adj2 = (adj @ adj).astype(bool).astype(np.uint8)
        adj2 = ((adj + adj2) > 0).astype(np.uint8)
        ring2 = extract_ring(adj2, 6)
 
        adj3 = (adj2 @ adj).astype(bool).astype(np.uint8)
        adj3 = ((adj3 + adj2) > 0).astype(np.uint8)
        ring3 = extract_ring(adj3, 12)
 
        return ring1, ring2, ring3
    
    @staticmethod
    def simplify_mesh(verts, faces, features, target_faces: int):
        """
        Simplify mesh to target face count while preserving vertex features.
        """
        if len(faces) <= int(target_faces):
            return verts, faces, features

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts),
            triangles=o3d.utility.Vector3iVector(faces),
        )

        simplified = mesh.simplify_quadric_decimation(int(target_faces))
        new_verts = np.asarray(simplified.vertices).astype(np.float32)
        new_faces = np.asarray(simplified.triangles).astype(np.int64)

        used_verts = np.unique(new_faces)
        new_verts = new_verts[used_verts]
        
        # Map old vertex indices to new via nearest neighbor, k=1
        _, idx = cKDTree(verts).query(new_verts, k=1)
        new_features = features[idx]

        # remap faces
        mapping = np.empty(used_verts.max() + 1, dtype=np.int64)
        mapping[used_verts] = np.arange(len(used_verts))
        new_faces = mapping[new_faces]
        
        # Enforce exact count by trimming or padding with repeats
        if len(new_faces) > int(target_faces):
            new_faces = new_faces[:int(target_faces)]
 
        return new_verts, new_faces, new_features
    
    @staticmethod
    def _normalize_point_count(verts, feats, normals, target_n: int, seed: int = 42):
        """Subsample or pad to exactly `target_n` points (reproducible)."""
        actual = verts.shape[0]
        rng = np.random.default_rng(seed)
        
        if actual >= int(target_n):
            # Random subsample
            idx = rng.choice(actual, int(target_n), replace=False)
        else:
            # Pad with repeats
            idx = np.concatenate([
                np.arange(actual),
                rng.choice(actual, int(target_n) - actual, replace=True),
            ])
            
        return verts[idx], feats[idx], normals[idx]
    