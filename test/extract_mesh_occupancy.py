import os
import zlib
import time
import numpy as np
import torch

import cubvh
import mcubes
import trimesh
import argparse

import kiui
from kiui.mesh import Mesh

from skimage import morphology

parser = argparse.ArgumentParser()
parser.add_argument('mesh', type=str)
parser.add_argument('--res', type=int, default=512)
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

def run(path):
    
    mesh = Mesh.load(path, wotex=True, bound=0.95, device=device)

    t0 = time.time()
    BVH = cubvh.cuBVH(mesh.v, mesh.f)
    print('BVH build time:', time.time() - t0)

    # naive sdf
    # sdf, _, _ = BVH.signed_distance(points.view(-1, 3), return_uvw=False, mode='raystab') # some mesh may not be watertight...
    # sdf = sdf.cpu().numpy()
    # occ = (sdf < 0)

    # udf floodfill
    t0 = time.time()
    udf, _, _ = BVH.unsigned_distance(points.view(-1, 3), return_uvw=False)
    print('UDF time:', time.time() - t0)
    udf = udf.cpu().numpy().reshape(opt.res, opt.res, opt.res)
    occ = udf < 2 / opt.res # tolerance 2 voxels

    t0 = time.time()
    empty_mask = morphology.flood(occ, (0, 0, 0), connectivity=1) # flood from the corner, which is for sure empty
    print('Floodfill time:', time.time() - t0)
    occ = ~empty_mask

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
    occ = occ.reshape(opt.res, opt.res, opt.res)
    t0 = time.time()
    verts, faces = mcubes.marching_cubes(occ, 0.5)
    # smoothed_occ = mcubes.smooth(occ, method='constrained')  # very slow
    # verts, faces = mcubes.marching_cubes(smoothed_occ, 0)
    print('MC time:', time.time() - t0)

    verts = verts / (opt.res - 1) * 2 - 1
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(os.path.basename(path).split('.')[0] + '.ply')

run(opt.mesh)