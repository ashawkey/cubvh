import os
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
Sparcubes implementation.
"""
parser = argparse.ArgumentParser()
parser.add_argument('test_path', type=str)
parser.add_argument('--min_res', type=int, default=512)
parser.add_argument('--max_res', type=int, default=1024) # at least 8x larger than min_res
# parser.add_argument('--min_res', type=int, default=32)
# parser.add_argument('--max_res', type=int, default=256) # at least 8x larger than min_res
parser.add_argument('--workspace', type=str, default='output2')
opt = parser.parse_args()

device = torch.device('cuda')

res = opt.min_res
res_fine = opt.max_res
res_sub = res_fine // res

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
        torch.linspace(-1/res, 1/res, res_sub + 1, device=device),
        torch.linspace(-1/res, 1/res, res_sub + 1, device=device),
        torch.linspace(-1/res, 1/res, res_sub + 1, device=device),
        indexing="ij",
    ), dim=-1,
) # [res_sub + 1, res_sub + 1, res_sub + 1, 3]

def box_normalize(vertices, bound=0.95):
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    vertices = 2 * bound * (vertices - bcenter) / (bmax - bmin).max()
    return vertices

def run(path):

    mesh = trimesh.load(path, process=False, force='mesh')
    mesh.vertices = box_normalize(mesh.vertices, bound=0.95)

    vertices = torch.from_numpy(mesh.vertices).float().to(device)
    triangles = torch.from_numpy(mesh.faces).long().to(device)

    BVH = cubvh.cuBVH(vertices, triangles)

    ### coarsest resolution: dense grid query
    # UDF
    eps = 2 / res
    udf, _, _ = BVH.unsigned_distance(grid_points.contiguous().view(-1, 3), return_uvw=False)
    kiui.lo(grid_points.view(-1, 3), udf)

    # flood fill
    udf = udf.view(res + 1, res + 1, res + 1).contiguous()
    occ = udf < eps
    floodfill_mask = cubvh.floodfill(occ)
    empty_label = floodfill_mask[0, 0, 0].item()
    empty_mask = (floodfill_mask == empty_label)
    occ_mask = ~empty_mask

    # truncated SDF
    sdf = udf - eps  # inner is negative
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
    
    # only keep voxels where the 8 corners have different sign
    sdf_mask = torch.sign(sdf_000) != torch.sign(sdf_001)
    sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_010)
    sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_011)
    sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_100)
    sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_101)
    sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_110)
    sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_111)

    active_cells_index = torch.nonzero(sdf_mask, as_tuple=True) # ([N], [N], [N])
    active_cells = torch.stack(active_cells_index, dim=-1) # [N, 3]

    active_cells_sdf = torch.stack([
        sdf_000[active_cells_index],
        sdf_001[active_cells_index],
        sdf_010[active_cells_index],
        sdf_011[active_cells_index],
        sdf_100[active_cells_index],
        sdf_101[active_cells_index],
        sdf_110[active_cells_index],
        sdf_111[active_cells_index],
    ], dim=-1) # [N, 8]

    active_cells_center = (active_cells + 0.5) / (res) * 2 - 1 # [N, 3] in [-1, 1]
    N = active_cells.shape[0]

    print(f'coarsest resolution: {res}, active cells: {N} / {res ** 3} = {N / res ** 3 * 100:.2f}%')

    # construct fine grid points only for the active cells
    active_cells_fine_grid_points = active_cells_center.view(N, 1, 1, 1, 3) + subgrid_offsets.view(1, res_sub + 1, res_sub + 1, res_sub + 1, 3) # [N, res_sub + 1, res_sub + 1, res_sub + 1, 3]
    active_cells_fine_grid_points = active_cells_fine_grid_points.view(-1, 3) # [N * (res_sub + 1) ** 3, 3]

    # query new SDF values
    eps_fine = 2 / res_fine
    udf_fine, _, _ = BVH.unsigned_distance(active_cells_fine_grid_points, return_uvw=False)
    kiui.lo(active_cells_fine_grid_points, udf_fine)
    
    # batched flood fill
    udf_fine = udf_fine.view(N, res_sub + 1, res_sub + 1, res_sub + 1).contiguous()
    occ_fine = udf_fine < eps
    fine_floodfill_mask = cubvh.floodfill(occ_fine) # [N, res_sub + 1, res_sub + 1, res_sub + 1]

    # we need to find out the empty label for each batch (any grid with positive sdf)
    active_cells_first_pos_idx = (active_cells_sdf > 0).float().argmax(dim=1) # [N], 0-7
    # get the corners of the fine_floodfill_mask
    fine_floodfill_corners = torch.stack([
        fine_floodfill_mask[:, 0, 0, 0],
        fine_floodfill_mask[:, 0, 0, -1],
        fine_floodfill_mask[:, 0, -1, 0],
        fine_floodfill_mask[:, 0, -1, -1],
        fine_floodfill_mask[:, -1, 0, 0],
        fine_floodfill_mask[:, -1, 0, -1],
        fine_floodfill_mask[:, -1, -1, 0],
        fine_floodfill_mask[:, -1, -1, -1],
    ], dim=1) # [N, 8]
    # gather the empty label for each batch
    empty_labels = torch.gather(fine_floodfill_corners, dim=1, index=active_cells_first_pos_idx.unsqueeze(1)) # [N]
    fine_empty_mask = (fine_floodfill_mask == empty_labels.view(N, 1, 1, 1)) # [N, res_sub + 1, res_sub + 1, res_sub + 1]
    fine_occ_mask = ~fine_empty_mask

    # truncated SDF
    sdf_fine = udf_fine - eps  # inner is negative
    fine_inner_mask = fine_occ_mask & (sdf_fine > 0)
    sdf_fine[fine_inner_mask] *= -1  # [N, res_sub + 1, res_sub + 1, res_sub + 1]

    # convert to sparse voxels again
    sdf_fine_000 = sdf_fine[:, :-1, :-1, :-1]
    sdf_fine_001 = sdf_fine[:, :-1, :-1, 1:]
    sdf_fine_010 = sdf_fine[:, :-1, 1:, :-1]
    sdf_fine_011 = sdf_fine[:, :-1, 1:, 1:]
    sdf_fine_100 = sdf_fine[:, 1:, :-1, :-1]
    sdf_fine_101 = sdf_fine[:, 1:, :-1, 1:]
    sdf_fine_110 = sdf_fine[:, 1:, 1:, :-1]
    sdf_fine_111 = sdf_fine[:, 1:, 1:, 1:]

    sdf_fine_mask = torch.sign(sdf_fine_000) != torch.sign(sdf_fine_001)
    sdf_fine_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_010)
    sdf_fine_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_011)
    sdf_fine_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_100)
    sdf_fine_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_101)
    sdf_fine_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_110)
    sdf_fine_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_111)

    kiui.lo(sdf_fine_mask, sdf_fine)

    fine_active_cells_index_local = torch.nonzero(sdf_fine_mask, as_tuple=True) # ([M], [M], [M], [M])
    fine_active_cells_local = torch.stack(fine_active_cells_index_local[1:], dim=-1) # [M, 3]
    M = fine_active_cells_local.shape[0]

    fine_active_cells_sdf = torch.stack([
        sdf_fine_000[fine_active_cells_index_local],
        sdf_fine_001[fine_active_cells_index_local],
        sdf_fine_010[fine_active_cells_index_local],
        sdf_fine_011[fine_active_cells_index_local],
        sdf_fine_100[fine_active_cells_index_local],
        sdf_fine_101[fine_active_cells_index_local],
        sdf_fine_110[fine_active_cells_index_local],
        sdf_fine_111[fine_active_cells_index_local],
    ], dim=-1) # [M, 8]

    print(f'fine resolution: {res_fine}, active cells: {M} / {(res_fine + 1) ** 3} = {M / (res_fine + 1) ** 3 * 100:.2f}%')

    # convert to global fine index
    fine_active_cells_global = res_sub * active_cells[fine_active_cells_index_local[0]] + fine_active_cells_local # [M, 3]

    # TODO: save fine_active_cells_sdf and fine_active_cells_global 
    kiui.lo(fine_active_cells_sdf, fine_active_cells_global)

    ### now, convert them back to the mesh!
    # we need to divide the sparse indices & voxels back to dense blocks
    # submeshes = []
    # for i in range(res):
    #     for j in range(res):
    #         for k in range(res):
    #             # get the active cells in this block.
    #             active_cells_block_mask = (fine_active_cells_global[:, 0] >= i * res_sub) & (fine_active_cells_global[:, 0] < (i + 1) * res_sub)
    #             active_cells_block_mask &= (fine_active_cells_global[:, 1] >= j * res_sub) & (fine_active_cells_global[:, 1] < (j + 1) * res_sub)
    #             active_cells_block_mask &= (fine_active_cells_global[:, 2] >= k * res_sub) & (fine_active_cells_global[:, 2] < (k + 1) * res_sub)
    #             if active_cells_block_mask.sum() == 0:
    #                 continue
    #             active_cells_in_block = fine_active_cells_global[active_cells_block_mask]
    #             inds = active_cells_in_block - torch.tensor([i * res_sub, j * res_sub, k * res_sub], device=device)
    #             # scatter the sdf values back to the dense grid
    #             # TODO: how to handle the default values? all positive?
    #             sdf_block = torch.ones(res_sub + 1, res_sub + 1, res_sub + 1, device=device)
    #             # there are 8 corners, but values should be the same at the same corner.
    #             sdf_block[inds[:, 0], inds[:, 1], inds[:, 2]] = fine_active_cells_sdf[active_cells_block_mask, 0]
    #             sdf_block[inds[:, 0], inds[:, 1], inds[:, 2] + 1] = fine_active_cells_sdf[active_cells_block_mask, 1]
    #             sdf_block[inds[:, 0], inds[:, 1] + 1, inds[:, 2]] = fine_active_cells_sdf[active_cells_block_mask, 2]
    #             sdf_block[inds[:, 0], inds[:, 1] + 1, inds[:, 2] + 1] = fine_active_cells_sdf[active_cells_block_mask, 3]
    #             sdf_block[inds[:, 0] + 1, inds[:, 1], inds[:, 2]] = fine_active_cells_sdf[active_cells_block_mask, 4]
    #             sdf_block[inds[:, 0] + 1, inds[:, 1], inds[:, 2] + 1] = fine_active_cells_sdf[active_cells_block_mask, 5]
    #             sdf_block[inds[:, 0] + 1, inds[:, 1] + 1, inds[:, 2]] = fine_active_cells_sdf[active_cells_block_mask, 6]
    #             sdf_block[inds[:, 0] + 1, inds[:, 1] + 1, inds[:, 2] + 1] = fine_active_cells_sdf[active_cells_block_mask, 7]
    #             # kiui.lo(sdf_block, inds)
    #             # convert to mesh block
    #             sdf_block = sdf_block.cpu().numpy()
    #             block_center = np.array([i + 0.5, j + 0.5, k + 0.5]) * 2 / res_sub - 1
    #             vertices, triangles = mcubes.marching_cubes(sdf_block, 0)
    #             vertices = vertices / (sdf_block.shape[-1] - 1.0) * 2 - 1
    #             vertices = vertices / res_sub + block_center
    #             vertices = vertices.astype(np.float32)
    #             triangles = triangles.astype(np.int32)
    #             print(f'block {i}, {j}, {k}: {vertices.shape}, {triangles.shape}', end='\r')
    #             submeshes.append(trimesh.Trimesh(vertices, triangles))

    # mesh = trimesh.util.concatenate(submeshes)
    # kiui.lo(mesh.vertices, mesh.faces)

    # import sparsecubes as sc
    # mesh = sc.marching_cubes(fine_active_cells_global.detach().cpu().numpy())
    # kiui.lo(mesh.vertices, mesh.faces)

    # from cubvh import sparse_marching_cubes
    # verts, faces = sparse_marching_cubes(fine_active_cells_global, fine_active_cells_sdf, 0)
    # mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces.detach().cpu().numpy())
    # kiui.lo(mesh.vertices, mesh.faces)

    # from cubvh import SpMC
    # verts, faces = SpMC(fine_active_cells_global.detach().cpu().numpy(), fine_active_cells_sdf.detach().cpu().numpy(), 0)
    # mesh = trimesh.Trimesh(verts, faces)
    # kiui.lo(mesh.vertices, mesh.faces)

    # flexicube
    from sparseflex.flexicube import FlexiCubes
    flexicube = FlexiCubes(device)

    

    # tmp debugging: just use a dense mc
    # sdf = torch.ones(res_fine + 1, res_fine + 1, res_fine + 1, device=device)
    # sdf[fine_active_cells_global[:, 0], fine_active_cells_global[:, 1], fine_active_cells_global[:, 2]] = fine_active_cells_sdf[:, 0]
    # sdf[fine_active_cells_global[:, 0], fine_active_cells_global[:, 1], fine_active_cells_global[:, 2] + 1] = fine_active_cells_sdf[:, 1]
    # sdf[fine_active_cells_global[:, 0], fine_active_cells_global[:, 1] + 1, fine_active_cells_global[:, 2]] = fine_active_cells_sdf[:, 2]
    # sdf[fine_active_cells_global[:, 0], fine_active_cells_global[:, 1] + 1, fine_active_cells_global[:, 2] + 1] = fine_active_cells_sdf[:, 3]
    # sdf[fine_active_cells_global[:, 0] + 1, fine_active_cells_global[:, 1], fine_active_cells_global[:, 2]] = fine_active_cells_sdf[:, 4]
    # sdf[fine_active_cells_global[:, 0] + 1, fine_active_cells_global[:, 1], fine_active_cells_global[:, 2] + 1] = fine_active_cells_sdf[:, 5]
    # sdf[fine_active_cells_global[:, 0] + 1, fine_active_cells_global[:, 1] + 1, fine_active_cells_global[:, 2]] = fine_active_cells_sdf[:, 6]
    # sdf[fine_active_cells_global[:, 0] + 1, fine_active_cells_global[:, 1] + 1, fine_active_cells_global[:, 2] + 1] = fine_active_cells_sdf[:, 7]
    # sdf = sdf.cpu().numpy()
    # vertices, triangles = mcubes.marching_cubes(sdf, 0)
    # vertices = vertices / (sdf.shape[-1] - 1.0) * 2 - 1
    # vertices = vertices.astype(np.float32)
    # triangles = triangles.astype(np.int32)
    # mesh = trimesh.Trimesh(vertices, triangles)
    # kiui.lo(mesh.vertices, mesh.faces)

    # we must decimate to visualize...
    # from kiui.mesh_utils import decimate_mesh
    # verts, faces = decimate_mesh(mesh.vertices, mesh.faces, 5e5)
    # verts = box_normalize(verts, bound=0.95)
    # mesh = trimesh.Trimesh(verts, faces)
    # kiui.lo(mesh.vertices, mesh.faces)

    name = os.path.splitext(os.path.basename(path))[0]
    mesh.export(f'{opt.workspace}/{name}.obj')

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