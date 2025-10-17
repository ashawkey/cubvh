import os
import time
import trimesh
import argparse
import numpy as np

from cubvh import parallel_decimate, decimate

# reference implementation
from kiui.mesh_utils import decimate_mesh

parser = argparse.ArgumentParser()
parser.add_argument("mesh", type=str)
parser.add_argument("--target_faces", type=int, default=100000)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--ref", action="store_true")
args = parser.parse_args()

mesh = trimesh.load(args.mesh)
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate(mesh.geometry.values())
# clean mesh and merge close vertices
# merge close vertices using rounding-based tolerance relative to scale
extents = np.asarray(mesh.extents, dtype=np.float64)
scale = float(np.linalg.norm(extents)) if extents.size == 3 else 1.0
if not np.isfinite(scale) or scale == 0.0:
    scale = 1.0
epsilon = scale * 1e-6
decimals = max(0, int(-np.floor(np.log10(epsilon)))) if epsilon > 0 else 6
mesh.vertices = np.round(mesh.vertices, decimals=decimals)
mesh.merge_vertices()
# final cleanup to drop any unreferenced vertices introduced
mesh.remove_unreferenced_vertices()

# decimate
t0 = time.time()
V, F = mesh.vertices, mesh.faces
print(f"Original mesh has {len(V)} vertices and {len(F)} faces")
V2, F2 = parallel_decimate(V, F, int(args.target_faces / 2))
print(f"Decimated mesh has {len(V2)} vertices and {len(F2)} faces")
print(f"Decimation took {time.time() - t0} seconds")

# export
mesh2 = trimesh.Trimesh(V2, F2)
output = args.output if args.output is not None else os.path.basename(args.mesh).split('.')[0] + '_decimated.glb'
mesh2.export(output)

# reference implementation
if args.ref:
    t0 = time.time()
    V2, F2 = decimate_mesh(V, F, args.target_faces, verbose=False)
    print(f"Reference decimated mesh has {len(V2)} vertices and {len(F2)} faces")
    print(f"Reference decimation took {time.time() - t0} seconds")

    # export
    mesh2 = trimesh.Trimesh(V2, F2)
    mesh2.export(output.replace('.glb', '_reference.glb'))