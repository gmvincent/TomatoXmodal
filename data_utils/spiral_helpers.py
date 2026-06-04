import torch
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix

def _ordered_one_ring(v: int, verts: np.ndarray, faces: np.ndarray) -> list[int]:
    """
    Return the 1-ring neighbours of vertex *v* in CCW angular order around v's
    local tangent plane.
 
    The tangent-plane normal is estimated from the least-variance direction of
    the neighbour cloud (last right-singular vector).  A fixed reference
    direction (the first neighbour's projection) gives a consistent winding
    across the whole mesh, which is what SpiralNet++ requires for translation-
    invariant convolution.
    """
    touching = np.where(np.any(faces == v, axis=1))[0]
    if len(touching) == 0:
        return []
 
    nbrs = np.unique(faces[touching].ravel())
    nbrs = nbrs[nbrs != v]
    if len(nbrs) == 0:
        return []
 
    deltas = verts[nbrs] - verts[v]          # [K, 3]
 
    # Estimate surface normal as the least-variance direction
    if len(deltas) >= 2:
        _, _, vh = np.linalg.svd(deltas, full_matrices=False)
        normal   = vh[-1]                    # [3]
    else:
        normal   = np.array([0.0, 0.0, 1.0], dtype=np.float32)
 
    # Project neighbours onto the tangent plane
    proj = deltas - (deltas @ normal)[:, None] * normal   # [K, 3]
    norms = np.linalg.norm(proj, axis=1, keepdims=True)
    proj  = proj / (norms + 1e-8)
 
    # Stable reference: first neighbour (after NN sort for consistency)
    # We pre-sort by distance so the reference is the geographically closest
    # neighbour, making the ordering as stable as possible across similar meshes.
    dist_order = np.argsort(np.linalg.norm(deltas, axis=1))
    ref_idx    = dist_order[0]
    ref  = proj[ref_idx]
    perp = np.cross(normal, ref)
    perp = perp / (np.linalg.norm(perp) + 1e-8)
 
    angles = np.arctan2(proj @ perp, proj @ ref)   # [K]
    return nbrs[np.argsort(angles)].tolist()

def compute_downsample_matrix(verts_hi, verts_lo):
    """
    Build sparse [n_lo, n_hi] downsample matrix D mapping high-res → low-res vertices.
    
    Each coarse vertex is mapped to its nearest fine vertex (NN projection),
    which matches the convention in the original SpiralNet++ codebase.
    """
    _, idx  = cKDTree(verts_hi).query(verts_lo, k=1)   # nearest high-res vertex index

    n_lo = verts_lo.shape[0]
    n_hi = verts_hi.shape[0]

    rows = np.arange(n_lo, dtype=np.int32)
    cols = idx.astype(np.int32)
    data = np.ones(n_lo, dtype=np.float32)

    return csr_matrix((data, (rows, cols)), shape=(n_lo, n_hi))

def preprocess_spiral(verts, faces, seq_length, dilation: int = 1):
    """
    Produces spiral sequences for each vertex.
    """
    N = verts.shape[0]
    
    spirals = np.empty((N, seq_length), dtype=np.int64)

    for v in range(N):
        visited = {v}
        queue = [v]
        spiral = [v]

        while len(spiral) < seq_length:
            new_queue = []
            for u in queue:
                for w in _ordered_one_ring(u, verts, faces)[::dilation]:
                    if w not in visited:
                        visited.add(w)
                        spiral.append(w)
                        new_queue.append(w)
                        if len(spiral) >= seq_length: 
                            break
                        
                if len(spiral) >= seq_length: 
                    break
            if not new_queue: 
                break
            queue = new_queue

        while len(spiral) < seq_length:
            spiral.append(v)
        spirals[v] = spiral[:seq_length]

    return spirals

def sparse_to_torch(mat: csr_matrix):
    """
    Turn scipy CSR matrix into a torch.sparse_coo_tensor (float32).
    """
    cx = mat.tocoo()
    
    indices = torch.from_numpy(np.vstack((cx.row, cx.col)).astype(np.int64))
    values  = torch.from_numpy(cx.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, mat.shape).coalesce()