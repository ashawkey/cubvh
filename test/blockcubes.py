import os
import glob
import tqdm
import trimesh
import argparse
import time
import math
import numpy as np

import torch
import cubvh

import kiui
from kiui.mesh_utils import decimate_mesh

"""
Sparcubes implementation.
"""
parser = argparse.ArgumentParser()
parser.add_argument('test_path', type=str)
parser.add_argument('--min_res', type=int, default=256)
parser.add_argument('--res', type=int, default=1024)
parser.add_argument('--workspace', type=str, default='output2')
parser.add_argument('--target_faces', type=int, default=-1)
opt = parser.parse_args()

device = torch.device('cuda')

res = opt.min_res
res_fine = opt.res
res_block = res_fine // res
# eps = 2 / res
eps_fine = 2 / res_fine

grid_points = torch.stack(
    torch.meshgrid(
        torch.linspace(-1, 1, res + 1, device=device),
        torch.linspace(-1, 1, res + 1, device=device),
        torch.linspace(-1, 1, res + 1, device=device),
        indexing="ij",
    ), dim=-1,
) # [min_res + 1, min_res + 1, min_res + 1, 3]

subgrid_offsets = torch.stack(
    torch.meshgrid(
        torch.linspace(-1/res, 1/res, res_block + 1, device=device),
        torch.linspace(-1/res, 1/res, res_block + 1, device=device),
        torch.linspace(-1/res, 1/res, res_block + 1, device=device),
        indexing="ij",
    ), dim=-1,
) # [res_block + 1, res_block + 1, res_block + 1, 3]

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
    vertices = 2 * bound * (vertices - bcenter) / (bmax - bmin).max()
    return vertices

def run(path):

    name = os.path.splitext(os.path.basename(path))[0]
    mesh = trimesh.load(path, process=False, force='mesh')
    mesh.vertices = box_normalize(mesh.vertices, bound=0.95)
    # mesh.vertices = sphere_normalize(mesh.vertices)
    vertices = torch.from_numpy(mesh.vertices).float().to(device)
    triangles = torch.from_numpy(mesh.faces).long().to(device)

    start_time = time.time()
    BVH = cubvh.cuBVH(vertices, triangles)
    print(f'BVH construction time: {time.time() - start_time:.2f}s')

    ### coarsest resolution: dense grid query
    start_time = time.time()
    # UDF
    udf, _, _ = BVH.unsigned_distance(grid_points.contiguous().view(-1, 3), return_uvw=False)

    # flood fill
    udf = udf.view(res + 1, res + 1, res + 1).contiguous()
    occ = udf < eps_fine # directly use eps_fine here!
    floodfill_mask = cubvh.floodfill(occ)
    empty_label = floodfill_mask[0, 0, 0].item()
    empty_mask = (floodfill_mask == empty_label)
    occ_mask = ~empty_mask

    # truncated SDF
    sdf = udf - eps_fine  # inner is negative
    inner_mask = occ_mask & (sdf > 0)
    sdf[inner_mask] *= -1  # [res + 1, res + 1, res + 1], numpy float32

    # convert to sparse voxels
    sdf_000 = sdf[:-1, :-1, :-1]
    sdf_001 = sdf[:-1, :-1, 1:]
    sdf_010 = sdf[:-1, 1:, :-1]
    sdf_011 = sdf[:-1, 1:, 1:]
    sdf_100 = sdf[1:, :-1, :-1]
    sdf_101 = sdf[1:, :-1, 1:]
    sdf_110 = sdf[1:, 1:, :-1]
    sdf_111 = sdf[1:, 1:, 1:]
    
    # keep voxels where the 8 corners have different sign
    active_cell_mask = torch.sign(sdf_000) != torch.sign(sdf_001)
    active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_010)
    active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_011)
    active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_100)
    active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_101)
    active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_110)
    active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_111)

    # also keep voxels where the true border lies in
    # TODO: this is not correct algorithm... have false positives...
    cube_diagonal_length = math.sqrt(3) / res # half of the cube diagonal length
    border_cell_mask =  torch.minimum(udf[:-1, :-1, :-1], udf[:-1, :-1, 1:])
    border_cell_mask = torch.minimum(border_cell_mask, udf[:-1, 1:, :-1])
    border_cell_mask = torch.minimum(border_cell_mask, udf[:-1, 1:, 1:])
    border_cell_mask = torch.minimum(border_cell_mask, udf[1:, :-1, :-1])
    border_cell_mask = torch.minimum(border_cell_mask, udf[1:, :-1, 1:])
    border_cell_mask = torch.minimum(border_cell_mask, udf[1:, 1:, :-1])
    border_cell_mask = torch.minimum(border_cell_mask, udf[1:, 1:, 1:])
    border_cell_mask = border_cell_mask <= cube_diagonal_length
    # center-check
    udf_center = torch.nn.functional.avg_pool3d(udf[None, None], kernel_size=2, stride=1).squeeze()
    border_cell_mask &= (udf_center <= cube_diagonal_length)
    # sign-check
    border_cell_mask &= (sdf_000 > 0) & (sdf_001 > 0) & (sdf_010 > 0) & (sdf_011 > 0) & (sdf_100 > 0) & (sdf_101 > 0) & (sdf_110 > 0) & (sdf_111 > 0)

    active_cell_mask |= border_cell_mask

    active_cells_index = torch.nonzero(active_cell_mask, as_tuple=True) # ([N], [N], [N])
    active_cells = torch.stack(active_cells_index, dim=-1) # [N, 3]

    active_cells_sdf = torch.stack([
        # order matters! this is the standard marching cubes order of 8 corners
        sdf_000[active_cells_index], 
        sdf_100[active_cells_index],
        sdf_110[active_cells_index],
        sdf_010[active_cells_index],
        sdf_001[active_cells_index],
        sdf_101[active_cells_index],
        sdf_111[active_cells_index],
        sdf_011[active_cells_index],
    ], dim=-1) # [N, 8]

    active_cells_center = (active_cells + 0.5) / (res) * 2 - 1 # [N, 3] in [-1, 1]
    N = active_cells.shape[0]

    print(f'Coarse level time: {time.time() - start_time:.2f}s, active cells: {N} / {res ** 3} = {N / res ** 3 * 100:.2f}%')

    # construct fine grid points only for the active cells
    start_time = time.time()
    active_cells_fine_grid_points = active_cells_center.view(N, 1, 1, 1, 3) + subgrid_offsets.view(1, res_block + 1, res_block + 1, res_block + 1, 3) # [N, res_block + 1, res_block + 1, res_block + 1, 3]
    active_cells_fine_grid_points = active_cells_fine_grid_points.view(-1, 3) # [N * (res_block + 1) ** 3, 3]

    # query new SDF values
    udf_fine, _, _ = BVH.unsigned_distance(active_cells_fine_grid_points, return_uvw=False)
    
    # batched flood fill
    udf_fine = udf_fine.view(N, res_block + 1, res_block + 1, res_block + 1).contiguous()
    occ_fine = udf_fine < eps_fine
    fine_floodfill_mask = cubvh.floodfill(occ_fine) # [N, res_block + 1, res_block + 1, res_block + 1]

    # we need to find out the empty label for each batch (any grid with positive sdf)
    active_cells_first_pos_idx = (active_cells_sdf > 0).float().argmax(dim=1) # [N], 0-7
    # get the corners of the fine_floodfill_mask
    fine_floodfill_corners = torch.stack([
        fine_floodfill_mask[:, 0, 0, 0],
        fine_floodfill_mask[:, -1, 0, 0],
        fine_floodfill_mask[:, -1, -1, 0],
        fine_floodfill_mask[:, 0, -1, 0],
        fine_floodfill_mask[:, 0, 0, -1],
        fine_floodfill_mask[:, -1, 0, -1],
        fine_floodfill_mask[:, -1, -1, -1],
        fine_floodfill_mask[:, 0, -1, -1],
    ], dim=1) # [N, 8]
    # gather the empty label for each batch
    empty_labels = torch.gather(fine_floodfill_corners, dim=1, index=active_cells_first_pos_idx.unsqueeze(1)) # [N]
    fine_empty_mask = (fine_floodfill_mask == empty_labels.view(N, 1, 1, 1)) # [N, res_block + 1, res_block + 1, res_block + 1]
    fine_occ_mask = ~fine_empty_mask

    # truncated SDF
    sdf_fine = udf_fine - eps_fine  # inner is negative
    fine_inner_mask = fine_occ_mask & (sdf_fine > 0)
    sdf_fine[fine_inner_mask] *= -1  # [N, res_block + 1, res_block + 1, res_block + 1]

    # convert to sparse voxels again
    sdf_fine_000 = sdf_fine[:, :-1, :-1, :-1]
    sdf_fine_001 = sdf_fine[:, :-1, :-1, 1:]
    sdf_fine_010 = sdf_fine[:, :-1, 1:, :-1]
    sdf_fine_011 = sdf_fine[:, :-1, 1:, 1:]
    sdf_fine_100 = sdf_fine[:, 1:, :-1, :-1]
    sdf_fine_101 = sdf_fine[:, 1:, :-1, 1:]
    sdf_fine_110 = sdf_fine[:, 1:, 1:, :-1]
    sdf_fine_111 = sdf_fine[:, 1:, 1:, 1:]

    # this time we only keep the active cells
    fine_active_cells_mask = torch.sign(sdf_fine_000) != torch.sign(sdf_fine_001)
    fine_active_cells_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_010)
    fine_active_cells_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_011)
    fine_active_cells_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_100)
    fine_active_cells_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_101)
    fine_active_cells_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_110)
    fine_active_cells_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_111)

    fine_active_cells_index_local = torch.nonzero(fine_active_cells_mask, as_tuple=True) # ([M], [M], [M], [M])
    fine_active_cells_local = torch.stack(fine_active_cells_index_local[1:], dim=-1) # [M, 3]
    M = fine_active_cells_local.shape[0]

    fine_active_cells_sdf = torch.stack([
        sdf_fine_000[fine_active_cells_index_local],
        sdf_fine_100[fine_active_cells_index_local],
        sdf_fine_110[fine_active_cells_index_local],
        sdf_fine_010[fine_active_cells_index_local],
        sdf_fine_001[fine_active_cells_index_local],
        sdf_fine_101[fine_active_cells_index_local],
        sdf_fine_111[fine_active_cells_index_local],
        sdf_fine_011[fine_active_cells_index_local],
    ], dim=-1) # [M, 8]

    # convert to global fine index
    fine_active_cells_global = res_block * active_cells[fine_active_cells_index_local[0]] + fine_active_cells_local # [M, 3]

    print(f'Fine level time: {time.time() - start_time:.2f}s, active cells: {M} / {(res_fine + 1) ** 3} = {M / (res_fine + 1) ** 3 * 100:.2f}%')

    ### now, convert them back to the mesh!
    start_time = time.time()
    vertices, triangles = cubvh.sparse_marching_cubes(fine_active_cells_global, fine_active_cells_sdf, 0)
    vertices = vertices / (res_fine - 1.0) * 2 - 1
    vertices = vertices.detach().cpu().numpy()
    triangles = triangles.detach().cpu().numpy()
    kiui.lo(vertices, triangles)
    print(f'Mesh extraction time: {time.time() - start_time:.2f}s')

    if opt.target_faces > 0:
        start_time = time.time()
        vertices, triangles = decimate_mesh(vertices, triangles, 1e6, optimalplacement=False)
        print(f'Decimation time: {time.time() - start_time:.2f}s')

    start_time = time.time()
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(f'{opt.workspace}/{name}.ply')

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