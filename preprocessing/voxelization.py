import os
import glob
import numpy as np
from tqdm import tqdm

import torch
import trimesh
from plyfile import PlyData

import scipy.ndimage

def load_ply(ply_path):
    plydata = PlyData.read(ply_path)
    data = plydata['vertex'].data

    xyz = np.stack([data['x'], data['y'], data['z']], axis=1)
    rgb = np.stack([data['red'], data['green'], data['blue']], axis=1)

    if 'scalar_nir' in data.dtype.names:
        nir = data['scalar_nir']
    elif 'scalar_wvl1' in data.dtype.names:
        nir = data['scalar_wvl1']
    else:
        raise ValueError(f"No NIR channel found in {ply_path}")

    rgbn = np.concatenate([rgb, nir[:, None]], axis=1)

    scalar_fields = {
        name: data[name] for name in data.dtype.names
        if name not in ['x', 'y', 'z', 'red', 'green', 'blue', 'scalar_nir']
    }

    return xyz, rgbn, scalar_fields

def create_voxel_grid(ply_path, voxel_size=1.0):
    try:
        mesh = trimesh.load(ply_path)
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        return None
    
    points_np = mesh.vertices.astype(np.float32)
    colors_np = mesh.visual.vertex_colors[:, :4].astype(np.float32)

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    points = torch.from_numpy(points_np).to(device)
    colors = torch.from_numpy(colors_np).to(device)

    # 1. Normalize and shift coordinates to a positive range
    min_coords = points.min(dim=0).values
    shifted_points = points - min_coords

    # 2. Map coordinates to integer voxel indices
    voxel_indices = torch.floor(shifted_points / voxel_size).long()

    # 3. Determine the grid dimensions
    max_indices = voxel_indices.max(dim=0).values
    grid_dim = max_indices + 1
    width, height, depth = grid_dim.tolist()
    #print(f"Output 3D image dimensions: ({width}, {height}, {depth})")

    # Initialize tensors for accumulating colors and point counts
    accumulated_colors = torch.zeros((width, height, depth, 4), dtype=torch.float32, device=device)
    point_counts = torch.zeros((width, height, depth), dtype=torch.int32, device=device)

    # 4. Filter out-of-bounds indices (should not happen with this method but is a good practice)
    valid_indices_mask = (voxel_indices[:, 0] < width) & (voxel_indices[:, 1] < height) & (voxel_indices[:, 2] < depth)
    valid_indices = voxel_indices[valid_indices_mask]
    valid_colors = colors[valid_indices_mask]

    # 5. Accumulate colors and counts
    for i in range(valid_indices.shape[0]):
        x, y, z = valid_indices[i]
        accumulated_colors[x, y, z] += valid_colors[i]
        point_counts[x, y, z] += 1
    
    # 6. Calculate the average color
    point_counts_expanded = point_counts.unsqueeze(3).expand(-1, -1, -1, 4)
    averaged_colors = torch.div(accumulated_colors, point_counts_expanded + 1e-8)
    
    # 7. Convert back to uint8 for final output
    output_image = averaged_colors.to(torch.uint8)

    return output_image

def process_ply(ply_path):
    _, _, scalar_fields = load_ply(ply_path)
    voxel_grid = create_voxel_grid(ply_path)
    
    return voxel_grid, scalar_fields

def rename_files(base_name):
    parts = base_name.split('_')
    if len(parts) < 4:
        raise ValueError(f"Filename {base_name} doesn't follow expected format.")
    prefix = '_'.join(parts[:3])
    suffix = '_'.join(parts[3:])
    voxel_name = f"{prefix}_3DRGBN_{suffix.replace('.ply', '.npy')}"
    scalar_name = f"{prefix}_3DRGBN_{suffix.replace('.ply', '_dict.npz')}"
    return voxel_name, scalar_name



def resize_with_contrast(voxel_grid, target_size):
    current_shape = voxel_grid.shape
    
    zoom_factors = (
        target_size[0] / current_shape[0],
        target_size[1] / current_shape[1],
        target_size[2] / current_shape[2],
        1.0  # don’t scale channels
    )
    
    # Save original min/max
    orig_min, orig_max = voxel_grid.min(), voxel_grid.max()
    
    # Resize
    voxel_resized = scipy.ndimage.zoom(voxel_grid, zoom_factors, order=1)
    
    # Contrast-preserving rescale
    """new_min, new_max = voxel_resized.min(), voxel_resized.max()
    if new_max > new_min:  # avoid divide by zero
        voxel_resized = (voxel_resized - new_min) / (new_max - new_min)  # scale to [0,1]
        voxel_resized = voxel_resized * (orig_max - orig_min) + orig_min  # scale back
    else:
        voxel_resized = np.full_like(voxel_resized, orig_min)
    """
    return voxel_resized


def main():
    root_dir = '/home/gmvincen/cmwilli5_drive/gmvincen_data/tomato_diseases/voxel_data'
    out_dir = '/home/gmvincen/cmwilli5_drive/gmvincen_data/tomato_diseases/scaled_voxel_data'
    
    all_files = glob.glob(os.path.join(root_dir, "**", "*.npy"), recursive=True)
    voxel_files = [f for f in all_files if 'trian' not in os.path.basename(f)]
    
    for fp in tqdm(voxel_files, desc="Scaling Voxels", total=len(voxel_files), unit="file"):
        if fp.endswith('.npy') and 'X' in fp:

            try:
                #voxel, scalars = process_ply(full_path)
                voxel_grid = np.load(fp)
                voxel_resized = resize_with_contrast(voxel_grid, (200, 212, 248))
                
                #voxel_out, scalar_out = rename_files(fname)
                voxel_path = os.path.join(out_dir,  os.path.basename(fp))
                #scalar_path = os.path.join(out_dir, dirpath.split("/")[-1], scalar_out)

                np.save(voxel_path, voxel_resized)
                #np.savez(scalar_path, **scalars)

                #print(f"Saved voxel to: {voxel_path}")
                #print(f"Saved scalars to: {scalar_path}")
            except Exception as e:
                print(f"Failed on {fp}: {e}")

if __name__ == '__main__':
    main()
