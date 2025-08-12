import os
import argparse
import numpy as np
import trimesh
import cubvh

parser = argparse.ArgumentParser(description="Merge close vertices in triangular meshes (CPU)")
parser.add_argument('path', type=str, help='Path to a mesh file or a directory containing meshes')
parser.add_argument('--workspace', type=str, default='output_merged', help='Output directory for merged meshes')
parser.add_argument('--pattern', type=str, default='*', help='Glob pattern when input is a directory')
parser.add_argument('--threshold', type=float, default=1e-5, help='Distance threshold for merging')
args = parser.parse_args()

valid_exts = {'.obj', '.ply', '.stl', '.off', '.glb', '.gltf', '.fbx'}

def is_mesh_file(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in valid_exts

def process_mesh(path: str):
    print(f"Processing: {path}")
    mesh = trimesh.load(path, process=False, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Not a triangular mesh: {path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    v_before = vertices.shape[0]
    f_before = faces.shape[0]

    v_new, f_new = cubvh.merge_vertices(vertices, faces, float(args.threshold))
    v_after = v_new.shape[0]
    f_after = f_new.shape[0]

    print(f"Merged vertices: {v_before} -> {v_after} (removed {v_before - v_after})")
    print(f"Faces (after degenerates removed & duplicates filtered): {f_before} -> {f_after}")

    merged = trimesh.Trimesh(vertices=v_new, faces=f_new, process=False)
    base = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]
    out_path = os.path.join(args.workspace, f"{base}{ext}")
    os.makedirs(args.workspace, exist_ok=True)
    merged.export(out_path)
    print(f"Saved: {out_path}")

if os.path.isdir(args.path):
    import glob
    files = [p for p in glob.glob(os.path.join(args.path, args.pattern)) if is_mesh_file(p)]
    for p in files:
        try:
            process_mesh(p)
        except Exception as e:
            print(f"[WARN] {p} failed: {e}")
else:
    process_mesh(args.path)
