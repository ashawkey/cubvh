import os
import glob
import argparse
import tqdm
import time
import numpy as np
import trimesh

import cubvh

parser = argparse.ArgumentParser(description="Repair small holes in triangular meshes using ear clipping (CPU)")
parser.add_argument('path', type=str, help='Path to a mesh file or a directory containing meshes')
parser.add_argument('--workspace', type=str, default='output', help='Output directory for repaired meshes')
parser.add_argument('--pattern', type=str, default='*', help='Glob pattern when input is a directory')
parser.add_argument('--return_added', action='store_true', help='If set, only triangulate holes and append to original faces')
parser.add_argument('--no_containment', action='store_true', help='Disable containment check for ear clipping (faster, less robust)')
parser.add_argument('--eps', type=float, default=1e-7, help='Numeric epsilon for robustness')
parser.add_argument('--verbose', action='store_true', help='Print per-file progress and number of faces added')
args = parser.parse_args()

os.makedirs(args.workspace, exist_ok=True)

valid_exts = {'.obj', '.ply', '.stl', '.off', '.glb', '.gltf', '.fbx'}


def is_mesh_file(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in valid_exts


def process_mesh(path: str):
    start_time = time.time()
    
    print(f"Processing: {path}")
    
    mesh = trimesh.load(path, process=False, force='mesh')
    # merge close vertices
    mesh.merge_vertices(digits_vertex=3)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Not a triangular mesh: {path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    original_faces = faces.shape[0]

    # Call CPU hole filling (NumPy binding)
    check_containment = not args.no_containment
    result = cubvh.fill_holes(vertices, faces, return_added=args.return_added, check_containment=check_containment, eps=float(args.eps), verbose=args.verbose)

    if args.return_added:
        # result holds only added triangles
        added = int(result.shape[0])
    else:
        # result holds full face list
        added = int(result.shape[0] - original_faces)

    new_faces = result.astype(np.int32, copy=False)

    if args.verbose:
        print(f"Added faces: {added} (from {original_faces} to {new_faces.shape[0]})")

    repaired = trimesh.Trimesh(vertices=vertices, faces=new_faces, process=False)
    repaired.merge_vertices(digits_vertex=4)

    base = os.path.splitext(os.path.basename(path))[0]
    ext = os.path.splitext(path)[1]
    out_path = os.path.join(args.workspace, f"{base}{ext}")
    repaired.export(out_path)
    end_time = time.time()
    
    print(f"Saved: {out_path} in {end_time - start_time:.2f} seconds")


if os.path.isdir(args.path):
    files = [p for p in glob.glob(os.path.join(args.path, args.pattern)) if is_mesh_file(p)]
    for p in tqdm.tqdm(files):
        try:
            process_mesh(p)
        except Exception as e:
            print(f"[WARN] {p} failed: {e}")
else:
    process_mesh(args.path)
