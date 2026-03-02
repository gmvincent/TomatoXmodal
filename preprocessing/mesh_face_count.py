import glob
import numpy as np
from plyfile import PlyData
from tqdm import tqdm

root = "/home/gmvincen/cmwilli5_drive/gmvincen_data/tomato_diseases/cleaned_phenospex_polygons/"
files = glob.glob(root + "/*.ply")

face_counts = []

for f in tqdm(files):
    try:
        ply = PlyData.read(f)
        if 'face' in ply:
            face_counts.append(len(ply['face'].data))
        else:
            face_counts.append(0)
    except Exception as e:
        print(f"Failed to load {f}: {e}")

face_counts = np.array(face_counts)

print(f"Total meshes: {len(face_counts)}")
print(f"Min faces: {face_counts.min()}")
print(f"Max faces: {face_counts.max()}")
print(f"Average faces: {face_counts.mean():.2f}")
print(f"Std faces: {face_counts.std():.2f}")