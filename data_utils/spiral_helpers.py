"""
SpiralNet++ utilities for meshes with per-sample topology (PlantEye plant scans).
 
Differences from the original SpiralNet++ pipeline, which assumes one shared
template mesh:
 
  * Spirals, pooling matrices and pyramid are built per sample.
  * 1-rings are ordered CCW around the *oriented* vertex normal (from the
    dataset's dual-view orientation step), so spiral direction is consistent
    across the mesh. The previous SVD-based normal had an arbitrary sign,
    which flipped CW/CCW vertex by vertex.
  * Rings are computed once per mesh from sparse adjacency instead of scanning
    every face for every visited vertex.
  * Dilation follows SpiralNet++: build a spiral of length seq_length*dilation,
    then keep every `dilation`-th entry.
  * Downsampling mean-pools every fine vertex into its nearest coarse vertex
    (default) instead of copying one fine vertex per coarse vertex.
  * Every sample gets the same pyramid depth: if decimation collapses, the
    level is repeated with an identity transform.
  * `spiral_collate` batches samples as one disconnected mesh (B = 1 along the
    batch axis), with offset spiral indices, block-diagonal pooling matrices
    and per-level `batch` vectors for per-sample global pooling.
 
Note: the spiral *starting point* (nearest neighbour) is still arbitrary across
samples. That is inherent to SpiralNet++ on non-registered meshes.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix, diags, identity



def vertex_adjacency(faces: np.ndarray, n_verts: int) -> csr_matrix:
    """Symmetric vertex adjacency (CSR, no self loops) from triangle faces."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    
    A = coo_matrix(
        (np.ones(len(e), dtype=np.int32), (e[:, 0], e[:, 1])),
        shape=(n_verts, n_verts),
    ).tocsr()
    
    A.sum_duplicates()
    A.setdiag(0)
    A.eliminate_zeros()
    A.sort_indices()
    
    return A

def ordered_rings(verts: np.ndarray, faces: np.ndarray, vnormals: np.ndarray) -> list:
    """
    1-ring neighbours of every vertex, ordered CCW around its oriented normal,
    starting from the nearest neighbour. Returns a list of Python int lists.
    """
    V = len(verts)
    A = vertex_adjacency(faces, V)
    
    verts = verts.astype(np.float64)
    up = np.array([0.0, 0.0, 1.0])
    rings = []
    
    for v in range(V):
        nb = A.indices[A.indptr[v]:A.indptr[v + 1]]
        if len(nb) < 2:
            rings.append(nb.tolist())
            continue
        n = vnormals[v].astype(np.float64)
        n_len = np.linalg.norm(n)
        n = n / n_len if n_len > 1e-8 else up
        
        d = verts[nb] - verts[v];                # [K, 3]
        p = d - (d @ n)[:, None] * n             # tangent-plane projection
        
        p_len = np.linalg.norm(p, axis=1)
        
        valid = np.where(p_len > 1e-12)[0]
        if len(valid) == 0:
            rings.append(nb.tolist())
            continue
        
        
        ref = p[valid[np.argmin(np.linalg.norm(d[valid], axis=1))]]
        ref = ref / np.linalg.norm(ref)
        perp = np.cross(n, ref)

        # Angle in [0, 2*pi) so the reference neighbour comes first, then CCW
        ang = np.mod(np.arctan2(p @ perp, p @ ref), 2 * np.pi)
        
        rings.append(nb[np.argsort(ang, kind="stable")].tolist())
    return rings

def preprocess_spiral(verts, faces, vnormals, seq_length: int, dilation: int = 1, rings=None):
    """
    Produces spiral sequences for each vertex.
    
    Grows ring by ring from the vertex, visiting each ring's vertices in the
    CCW order of the previous ring. Short spirals (boundaries, small
    components) are padded with the centre vertex.
    """
    if rings is None:
        rings = ordered_rings(verts, faces, vnormals)
    
    V = len(verts)
    full = seq_length * dilation
    spirals = np.empty((V, seq_length), dtype=np.int64)
    n_pad = 0

    for v in range(V):
        visited, queue, spiral = {v}, [v], [v]
        
        while len(spiral) < full and queue:
            new_queue = []
            for u in queue:
                for w in rings[u]:
                    if w not in visited:
                        visited.add(w)
                        spiral.append(w)
                        new_queue.append(w)
                        if len(spiral) >= full: 
                            break
                        
                if len(spiral) >= full: 
                    break
            
            queue = new_queue

        spiral = spiral[::dilation][:seq_length]
        n_pad += seq_length - len(spiral)
        spiral.extend([v] * (seq_length - len(spiral)))
        spirals[v] = spiral
    
    # pad fraction
    # p_frac = n_pad / max(V * seq_length, 1)
    
    return spirals

def compute_downsample_matrix(verts_hi, verts_lo, mode: str = "mean"):
    """
    Build sparse [n_lo, n_hi] downsample matrix D mapping high-res → low-res vertices.
    
    mode="mean":    each fine vertex is assigned to its nearest coarse vertex
                    and each coarse vertex averages its assigned fine vertices.
                    Coarse vertices with no assignment fall back to their
                    nearest fine vertex.
    mode="nearest": each coarse vertex copies its nearest fine vertex
                    (original SpiralNet++/COMA convention).
    """
    

    n_hi, n_lo = len(verts_hi), len(verts_lo)
    if mode == "nearest":
        _, idx  = cKDTree(verts_hi).query(verts_lo, k=1)   # nearest high-res vertex index
        rows = np.arange(n_lo, dtype=np.int32)
        cols = idx.astype(np.int32)
        data = np.ones(n_lo, dtype=np.float32)

        return csr_matrix((data, (rows, cols)), shape=(n_lo, n_hi))

    elif mode == "mean":
        _, assign = cKDTree(verts_lo).query(verts_hi, k=1)
        rows, cols = [assign], [np.arange(n_hi)]
        
        empty = np.where(np.bincount(assign, minlength=n_lo) == 0)[0]
        
        if len(empty):
            _, nn = cKDTree(verts_hi).query(verts_lo[empty], k=1)
            rows.append(empty)
            cols.append(nn)
    
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        D = coo_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n_lo, n_hi)
        ).tocsr()
        D.sum_duplicates()
        row_sums = np.asarray(D.sum(axis=1)).ravel()
        return (diags(1.0 / row_sums) @ D).tocsr().astype(np.float32)
    else:
        raise ValueError(f"Unknown pooling mode: {mode!r}")

def identity_downsample(n: int) -> csr_matrix:
    """Identity pooling, used when a pyramid level cannot be decimated further."""
    return identity(n, dtype=np.float32, format="csr")

def build_spiral_pyramid(
    verts, faces, base_feats,
    ds_factors, seq_lengths, dilations,
    simplify_fn, orient_fn, normals_fn,
    pool_mode: str = "mean",
    min_verts: int = 4,
):
    """
    Fixed-depth spiral pyramid: len(ds_factors) + 1 levels for every sample.
 
    simplify_fn(v, f, feats, target) -> (v, f, feats)   e.g. TomatoTask3D.simplify_mesh
    orient_fn(v, f)                  -> f               oriented faces
    normals_fn(v, f)                 -> vertex normals  [V, 3]
 
    If decimation cannot reduce a level (mesh already at/below target) or
    collapses it, that level is repeated with an identity pooling matrix so
    the depth stays fixed.
 
    Returns dict:
      "features"       Tensor [1, V0, C]       level-0 input features only
      "spiral_indices" list[Tensor [V_l, S_l]] one per level
      "down_transform" list[sparse Tensor]     len = num_levels - 1
      "pad_frac"       list[float]             padded-entry fraction per level
      "real_levels"    int                     levels actually decimated + 1
    """
    v_levels, f_levels, downs = [verts], [faces], []
    real_levels = 1
 
    for factor in ds_factors:
        v_hi, f_hi = v_levels[-1], f_levels[-1]
        target = max(4, len(f_hi) // factor)
        decimated = False
 
        if len(f_hi) > target:
            v_lo, f_lo, _ = simplify_fn(
                v_hi, f_hi, np.zeros((len(v_hi), 0), dtype=np.float32), target
            )
            if len(f_lo) > 0 and len(v_lo) >= min_verts and len(v_lo) < len(v_hi):
                f_lo = orient_fn(v_lo, f_lo)
                downs.append(compute_downsample_matrix(v_hi, v_lo, mode=pool_mode))
                v_levels.append(v_lo)
                f_levels.append(f_lo)
                decimated = True
                real_levels += 1
 
        if not decimated:
            downs.append(identity_downsample(len(v_hi)))
            v_levels.append(v_hi)
            f_levels.append(f_hi)
 
    spiral_indices, pad_frac = [], []
    for l, (v, f) in enumerate(zip(v_levels, f_levels)):
        seq = seq_lengths[l % len(seq_lengths)]
        dil = dilations[l % len(dilations)]
        sp, pf = preprocess_spiral(v, f, normals_fn(v, f), seq_length=seq, dilation=dil)
        spiral_indices.append(torch.as_tensor(sp, dtype=torch.long))
        pad_frac.append(pf)
 
    return {
        "features": torch.as_tensor(base_feats, dtype=torch.float32).unsqueeze(0),
        "spiral_indices": spiral_indices,
        "down_transform": [sparse_to_torch(D) for D in downs],
        "pad_frac": pad_frac,
        "real_levels": real_levels,
    }

def sparse_to_torch(mat: csr_matrix) -> torch.Tensor:
    """
    Turn scipy CSR matrix into a torch.sparse_coo_tensor (float32).
    """
    cx = mat.tocoo()
    
    indices = torch.from_numpy(np.vstack((cx.row, cx.col)).astype(np.int64))
    values  = torch.from_numpy(cx.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, mat.shape).coalesce()

def block_diag_sparse(mats) -> torch.Tensor:
    """Block-diagonal stack of sparse COO tensors."""
    idx, val, r0, c0 = [], [], 0, 0
    for m in mats:
        m = m.coalesce()
        idx.append(m.indices() + torch.tensor([[r0], [c0]]))
        val.append(m.values())
        r0 += m.shape[0]
        c0 += m.shape[1]
    return torch.sparse_coo_tensor(torch.cat(idx, 1), torch.cat(val), (r0, c0)).coalesce()