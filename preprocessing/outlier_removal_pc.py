import os
import glob
import numpy as np
from tqdm import tqdm
from plyfile import PlyData
from scipy.spatial import cKDTree

from data_utils.mesh_data import classes as mesh_classes

import matplotlib.pyplot as plt

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

        mask = z > 6
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

        feature_names = []
        feature_arrays = []
        for name in vertex_data.dtype.names:
            if name not in ['x','y','z']:
                feature_arrays.append(vertex_data[name][mask].astype(np.float32))
                feature_names.append(name)

        if feature_arrays:
            features = np.stack(feature_arrays, axis=-1)
        else:
            features = None

        return vertices, faces, features, feature_names

    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None, None, None, None
    
    
def local_plane(neighbor_points):
    """
    Fit local plane via PCA.
    
    Args:
        neighbor_points: (k, 3) array
    
    Returns:
        normal: approximation of normal vector at the vertex pt
        p_bar: center point of k nearest neighbood of point pt; centroid
    """
    
    if len(neighbor_points) < 3:
        return None, None
    
    # centroid
    p_bar = np.mean(neighbor_points, axis=0)
    
    # compute covariance matrix
    centered = neighbor_points - p_bar
    M = centered.T @ centered / len(neighbor_points)
    
    # eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    
    # smallest eigenvalue
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    
    # unit normal
    normal /= np.linalg.norm(normal)

    return normal, p_bar

def sparse_outliers(in_pcd, delta=0.0035, k=20):
    """
    Sparse outlier and isolated outlier removal algorithm
    
    Args:
        in_pcd: three dimensional scanned point cloud with outliers
        delta: probability threshold
        k: number of nearest neighbors

    Returns:
        out_pcd: filtered point cloud
    """
    in_pcd = np.array(in_pcd)
    
    # search the k nearest neighborhood of pt, KNN(pt)
    tree = cKDTree(in_pcd)
    distances, indices = tree.query(in_pcd, k=k+1)
    
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    # calculate the local density, LD(pt) of pt
    d_bar = np.mean(distances, axis=1) # average distance
    d_bar[d_bar == 0] = 1e-8 # avoid divide by zero
    
    # local density
    ld = (1/k) * np.sum(np.exp(-distances/d_bar[:,None]), axis=1)

    # compute the porbability, P(pt)
    pro = 1 - ld
    
    delta = 0.1 * d_bar  # adaptive threshold from paper
    mask = pro > delta
    
    out_pcd = in_pcd[mask]

    return out_pcd, mask

def nonisolated_outliers(in_pcd, k=20):
    """
    Non-isolated outlier removal alogrithm
    Removes outliers near the surface of the object. Avoiding removing points related to the object.
    
    Args:
        in_pcd: three dimensional scanned point cloud with outliers
        k: number of nearest neighbors

    Returns:
        out_pcd: filtered point cloud
    """
    in_pcd = np.array(in_pcd)
    
    # search the k nearest neighborhood of pt, KNN(pt)
    tree = cKDTree(in_pcd)

    _, indices = tree.query(in_pcd, k=k+1)
    indices = indices[:, 1:]
    
    projected_pcd = in_pcd.copy()
    
    for i, pt in enumerate(in_pcd):
        neighbor_points = in_pcd[indices[i]]
        
        # fit a local plane for pt and qj neighbors
        normal, p_bar = local_plane(neighbor_points)
        
        if normal is None:
            continue
        
        # deviation from plane
        dis = np.dot(pt - p_bar, normal)
        
        # project qj onto the corresponding fited plane
        projected_pcd[i] = pt - dis * normal
        
    return projected_pcd

def sparse_plus_nonisolated(points, k_sparse=20, k_plane=20):
    filtered, mask = sparse_outliers(points, k=k_sparse)
    projected = nonisolated_outliers(filtered, k=k_plane)
    return projected, mask

def filter_mesh(verts, faces, feats, mask):
    """
    Remove vertices using mask and update faces accordingly.
    """
    # Map old indices to new indices
    old_to_new = -np.ones(len(mask), dtype=int)
    old_to_new[mask] = np.arange(np.sum(mask))

    # Filter vertices and features
    new_verts = verts[mask]
    new_feats = feats[mask] if feats is not None else None

    # Update faces
    new_faces = []
    for face in faces:
        if mask[face].all():  # keep only faces where all 3 vertices remain
            new_faces.append(old_to_new[face])

    new_faces = np.array(new_faces)

    return new_verts, new_faces, new_feats

def save_ply(path, verts, faces, features=None, feature_names=None):
    """Saves .ply files after sparse and isolated outlier removal

    Args:
        path (str): file name and path
        verts (np.array): vertices of the point clouds
        faces (_type_): faces of the mesh structure
        features (_type_, optional): Color channels. Defaults to None.
    """
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")

        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if features is not None and feature_names is not None:
            for name in feature_names:
                f.write(f"property float {name}\n")

        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i in range(len(verts)):
            line = " ".join(map(str, verts[i]))
            if features is not None:
                line += " " + " ".join(map(str, features[i]))
            f.write(line + "\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def visualize_pointcloud(points, title, save_path=None):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(
        points[:,0],
        points[:,1],
        points[:,2],
        s=0.75,
        c=points[:,2],
        cmap="Spectral"
    )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    root = "/home/gmvincen/cmwilli5_drive/gmvincen_data/tomato_diseases"
    data_path = os.path.join(root, "phenospex_polygons")
    os.makedirs("images", exist_ok=True)
    os.makedirs(os.path.join(root, "cleaned_phenospex_polygons"), exist_ok=True)


    all_files = glob.glob(os.path.join(data_path, "**", "*.ply"), recursive=True)
    mesh_files = [f for f in all_files if 'trian' not in os.path.basename(f)]
    
    # visualize first 3 meshes
    labels_used = []
    for fp in tqdm(mesh_files, desc="Preprocessing Files", total=len(mesh_files), unit="file"):
        verts, faces, feats, feat_names = load_ply(fp)
        
        if verts is None:
            continue

        label_idx = int(os.path.basename(fp).split("_")[1].lstrip("T"))
        label_name = mesh_classes[label_idx]        
        
        _, mask = sparse_outliers(verts)
        new_verts, new_faces, new_feats = filter_mesh(verts, faces, feats, mask)
        new_verts = nonisolated_outliers(new_verts)
        
        save_ply(
            os.path.join(root, "cleaned_phenospex_polygons", os.path.basename(fp)),
            new_verts,
            new_faces,
            new_feats,
            feat_names,
        )
        
        if label_idx not in labels_used:
 
            labels_used.append(label_idx)
            
            print(f"Visualizing {label_name}")
            sparse_only, _ = sparse_outliers(verts)
            nonisolated_only = nonisolated_outliers(verts)
            both, _ = sparse_plus_nonisolated(verts)

            visualize_pointcloud(
                verts,
                f"Original - {label_name}",
                f"images/original_{label_name}.png"
            )

            visualize_pointcloud(
                sparse_only,
                f"Sparse Only - {label_name}",
                f"images/sparse_{label_name}.png"
            )

            visualize_pointcloud(
                nonisolated_only,
                f"Non-Isolated Only - {label_name}",
                f"images/nonisolated_{label_name}.png"
            )

            visualize_pointcloud(
                both,
                f"Sparse + Non-Isolated - {label_name}",
                f"images/both_{label_name}.png"
            )
    
