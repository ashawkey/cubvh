import os
import glob
import tqdm
import trimesh
import argparse
import numpy as np
import torch
import cubvh

import kiui
from cubvh.sparse_voxel_extractor import SparseVoxelExtractor, box_normalize, save_quantized, load_quantized, extract_mesh

"""
Block-sparse marching cubes implementation.
"""
parser = argparse.ArgumentParser()
parser.add_argument('test_path', type=str)
parser.add_argument('--res', type=int, default=512)
parser.add_argument('--res_fine', type=int, default=1024)
parser.add_argument('--workspace', type=str, default='output')
parser.add_argument('--target_faces', type=int, default=1000000)
parser.add_argument('--save', action='store_true')
parser.add_argument('--extract_mesh', action='store_true')
parser.add_argument('--cpu_mc', action='store_true', help='Use CPU sparse marching cubes instead of CUDA')
opt = parser.parse_args()

device = torch.device('cuda')

extractor = SparseVoxelExtractor(opt.res, opt.res_fine, device, verbose=True)

### main function
def run(path):

    name = os.path.splitext(os.path.basename(path))[0] + "_" + str(opt.res) + "_" + str(opt.res_fine)
    mesh = trimesh.load(path, process=False, force='mesh')
    vertices = mesh.vertices
    triangles = mesh.faces
    vertices = box_normalize(vertices, bound=0.95)

    vertices = torch.from_numpy(vertices).float().to(device)
    triangles = torch.from_numpy(triangles).long().to(device)
    extractor.build_bvh(vertices, triangles)

    out = extractor.extract_sparse_voxels()
    coords = out['coords']
    sdfs = out['sdfs']

    if opt.save:
        kiui.lo(coords, sdfs)
        out_npz = f'{opt.workspace}/{name}.npz'
        _ = save_quantized(coords, sdfs, out_npz, resolution=opt.res_fine)
        # round-trip load to match prior behavior
        coords, sdfs = load_quantized(out_npz, device=device, resolution=opt.res_fine, normalized_tsdf=True)
        kiui.lo(coords, sdfs) # the same
    
    if opt.extract_mesh:
        vertices, triangles = extract_mesh(
            coords,
            sdfs,
            resolution=opt.res_fine,
            cpu_mc=opt.cpu_mc,
            target_faces=opt.target_faces,
            verbose=True,
        )
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(f'{opt.workspace}/{name}.glb')

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