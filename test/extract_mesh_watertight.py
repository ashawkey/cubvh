import os
import zlib
import glob
import tqdm
import time
import mcubes
import trimesh
import argparse
import numpy as np

import torch
import cubvh

import kiui

"""
Extract watertight mesh from a arbitrary mesh by UDF expansion and floodfill.
"""
parser = argparse.ArgumentParser()
parser.add_argument('test_path', type=str)
parser.add_argument('--res', type=int, default=512)
parser.add_argument('--workspace', type=str, default='output')
opt = parser.parse_args()

device = torch.device('cuda')

points = torch.stack(
    torch.meshgrid(
        torch.linspace(-1, 1, opt.res, device=device),
        torch.linspace(-1, 1, opt.res, device=device),
        torch.linspace(-1, 1, opt.res, device=device),
        indexing="ij",
    ), dim=-1,
) # [N, N, N, 3]


def sphere_normalize(vertices):
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    radius = np.linalg.norm(vertices - bcenter, axis=-1).max()
    vertices = (vertices - bcenter) / (radius)  # to [-1, 1]
    return vertices


def box_normalize(vertices, bound=0.95):
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    vertices = bound * (vertices - bcenter) / (bmax - bmin).max()
    return vertices


def run(path):

    mesh = trimesh.load(path, process=False, force='mesh')
    mesh.vertices = sphere_normalize(mesh.vertices)
    vertices = torch.from_numpy(mesh.vertices).float().to(device)
    triangles = torch.from_numpy(mesh.faces).long().to(device)

    t0 = time.time()
    BVH = cubvh.cuBVH(vertices, triangles)
    print(f'BVH build time: {time.time() - t0:.4f}s')
    eps = 2 / opt.res

    # naive sdf
    # sdf, _, _ = BVH.signed_distance(points.view(-1, 3), return_uvw=False, mode='raystab') # some mesh may not be watertight...
    # sdf = sdf.cpu().numpy()
    # occ = (sdf < 0)

    # udf
    t0 = time.time()
    udf, _, _ = BVH.unsigned_distance(points.view(-1, 3), return_uvw=False)
    print(f'UDF time: {time.time() - t0:.4f}s')

    # floodfill
    t0 = time.time()
    udf = udf.view(opt.res, opt.res, opt.res).contiguous()
    occ = udf < eps
    floodfill_mask = cubvh.floodfill(occ)
    empty_label = floodfill_mask[0, 0, 0].item()
    empty_mask = (floodfill_mask == empty_label)
    print(f'Floodfill time: {time.time() - t0:.4f}s')

    # binary occupancy
    occ_mask = ~empty_mask

    # truncated SDF
    sdf = udf - eps  # inner is negative
    inner_mask = occ_mask & (sdf > 0)
    sdf[inner_mask] *= -1

    sdf = sdf.cpu().numpy()
    print(f'SDF occupancy ratio: {np.sum(sdf < 0) / sdf.size:.4f}')

    # # packbits and compress
    # occ = occ.astype(np.uint8).reshape(-1)
    # occ = np.packbits(occ)
    # occ = zlib.compress(occ.tobytes())

    # # save to disk
    # with open(os.path.basename(path).split('.')[0] + '.bin', 'wb') as f:
    #     f.write(occ)

    # # uncompress and unpack
    # occ = zlib.decompress(occ)
    # occ = np.frombuffer(occ, dtype=np.uint8)
    # occ = np.unpackbits(occ, count=opt.res**3).reshape(opt.res, opt.res, opt.res)

    # marching cubes
    t0 = time.time()
    vertices, triangles = mcubes.marching_cubes(sdf, 0)
    vertices = vertices / (sdf.shape[-1] - 1.0) * 2 - 1
    vertices = vertices.astype(np.float32)
    triangles = triangles.astype(np.int32)
    watertight_mesh = trimesh.Trimesh(vertices, triangles)
    print(f'MC time: {time.time() - t0:.4f}s, vertices: {len(watertight_mesh.vertices)}, triangles: {len(watertight_mesh.faces)}')

    name = os.path.splitext(os.path.basename(path))[0]
    watertight_mesh.export(f'{opt.workspace}/{name}.obj')

os.makedirs(opt.workspace, exist_ok=True)

if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
    for path in tqdm.tqdm(file_paths):
        try:
            run(path)
        except Exception as e:
            print(f'[WARN] {path} failed: {e}')
else:
    run(opt.test_path)