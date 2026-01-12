import torch
import trimesh
import open3d as o3d
import numpy as np
import scipy.spatial
import scipy.sparse as sp

def mesh_downsample(vertices, faces, factor=4):
    """
    Downsample mesh to ~N/factor vertices using QEM simplification.
    """
    #mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    #target = vertices.shape[0] // factor
    #simp = mesh.simplify_quadratic_decimation(target)
    #return np.asarray(simp.vertices), np.asarray(simp.faces)
    
    if vertices.shape[0] == 0 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid vertices shape: {vertices.shape}")
    
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    mesh.compute_vertex_normals()

    target = max(10, int(len(vertices)/factor))
    target = min(target, len(vertices))

    mesh_s = mesh.simplify_quadric_decimation(target)

    v = np.asarray(mesh_s.vertices)
    f = np.asarray(mesh_s.triangles)
    return v, f

def compute_downsample_matrix(verts_hi, verts_lo):
    """
    Build sparse downsample matrix D mapping high-res → low-res vertices.
    """
    tree = scipy.spatial.KDTree(verts_hi)
    _, idx = tree.query(verts_lo)   # nearest high-res vertex index

    n_lo = verts_lo.shape[0]
    n_hi = verts_hi.shape[0]

    rows = np.arange(n_lo)
    cols = idx
    data = np.ones(n_lo)

    return sp.coo_matrix((data, (rows, cols)), shape=(n_lo, n_hi))

def preprocess_spiral(faces, seq_length, vertices):
    """
    Produces spiral sequences for each vertex.
    """
    N = vertices.shape[0]
    
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for f in faces:
        a, b, c = f
        adj[a].extend([b, c])
        adj[b].extend([a, c])
        adj[c].extend([a, b])

    adj = [list(set(a)) for a in adj]

    spiral_indices = np.zeros((N, seq_length), dtype=np.int64)

    for v in range(N):
        queue = [v]
        visited = {v}
        spiral = [v]

        while len(spiral) < seq_length and queue:
            new_queue = []
            for u in queue:
                for w in adj[u]:
                    if w not in visited:
                        visited.add(w)
                        spiral.append(w)
                        new_queue.append(w)
                        if len(spiral) == seq_length:
                            break
                if len(spiral) == seq_length:
                    break
            queue = new_queue

        if len(spiral) < seq_length:
            spiral.extend([v] * (seq_length - len(spiral)))

        spiral_indices[v] = np.array(spiral)

    return spiral_indices

def to_sparse(mat):
    """
    Turn scipy sparse into a torch sparse tensor.
    """
    mat = mat.tocoo()
    indices_np = np.vstack((mat.row, mat.col)).astype(np.int64)
    indices = torch.from_numpy(indices_np)
    values = torch.tensor(mat.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, mat.shape)
