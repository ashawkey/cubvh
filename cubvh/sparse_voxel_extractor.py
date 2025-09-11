import os
import math
from tabnanny import verbose
import time
from typing import Optional, Tuple

import trimesh

import kiui
import numpy as np
import torch
import torch.nn.functional as F

import cubvh


def sphere_normalize(vertices: np.ndarray) -> np.ndarray:
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    radius = np.linalg.norm(vertices - bcenter, axis=-1).max()
    vertices = (vertices - bcenter) / (radius)
    return vertices


def box_normalize(vertices: np.ndarray, bound: float = 0.95) -> np.ndarray:
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    vertices = 2 * bound * (vertices - bcenter) / (bmax - bmin).max()
    return vertices


def compress_coords(coords: np.ndarray, res: int) -> np.ndarray:
    # Flatten [N, 3] coords to a single uint64 (since 2048^3 exceeds uint32)
    coords = coords.astype(np.uint64)
    return coords[:, 0] * res * res + coords[:, 1] * res + coords[:, 2]


def decompress_coords(compressed: np.ndarray, res: int) -> np.ndarray:
    # Decompress a single uint64 to [3] coords
    x = compressed // (res * res)
    y = (compressed % (res * res)) // res
    z = compressed % res
    return np.stack([x, y, z], axis=-1)


def quantize_sdf(sdf: np.ndarray, resolution: int) -> Tuple[np.ndarray, float]:
    """
    Quantize 8-corner SDF values per voxel into packed uint64.

    Args:
        sdf: (N, 8) float32 signed distances in [-1, 1] normalization.
        resolution: fine resolution used to scale delta.

    Returns:
        packed: (N,) uint64 packed 8x8-bit codes (little endian, corner 0 in LSB).
        delta: float quantization step for de-quantization.
    """
    if sdf.ndim != 2 or sdf.shape[1] != 8:
        raise ValueError("sdf must have shape [N, 8]")
    delta = 2 / resolution * math.sqrt(3) / 127.0
    mag = np.rint(np.clip(np.abs(sdf) / delta, 0, 127)).astype(np.uint8)
    sign = (sdf < 0).astype(np.uint8) << 7
    codes = (sign | mag).astype(np.uint8)  # (N, 8)
    shifts = (np.arange(8, dtype=np.uint64) * 8)[None, :]
    packed = np.sum((codes.astype(np.uint64) << shifts), axis=1, dtype=np.uint64)
    return packed, delta


def dequantize_sdf(packed: np.ndarray, resolution: Optional[int] = None) -> np.ndarray:
    if packed.dtype != np.uint64:
        raise ValueError("packed must be dtype uint64")
    shifts = (np.arange(8, dtype=np.uint64) * 8)[None, :]
    bytes_ = ((packed[:, None] >> shifts) & 0xFF).astype(np.uint8)  # (N, 8)
    sign = ((bytes_ >> 7) & 1).astype(np.int8)
    sdf = (bytes_ & 0x7F).astype(np.float32) / 127.0
    if resolution is not None:
        sdf = sdf * (2 / resolution * math.sqrt(3))
    sdf[sign == 1] *= -1.0
    return sdf


def save_quantized(coords: torch.Tensor, sdf: torch.Tensor, out_path: str, resolution: int) -> float:
    """
    Save quantized sparse voxel data to NPZ.

    Args:
        coords: (M, 3) int grid coords
        sdf: (M, 8) float truncated SDF per corner (normalized as produced)
        out_path: path without extension or with .npz
        resolution: fine grid resolution used for compression/quantization

    Returns:
        quant_delta: float delta used during quantization
    """
    coords_np = coords.detach().cpu().numpy()
    sdf_np = sdf.detach().cpu().numpy()

    coords_q = compress_coords(coords_np, int(resolution))
    sdf_q, delta = quantize_sdf(sdf_np, int(resolution))

    if not out_path.endswith('.npz'):
        out_path = out_path + '.npz'
    dirpath = os.path.dirname(out_path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    np.savez_compressed(out_path, coords=coords_q, sdfs=sdf_q)
    return delta


def load_quantized(in_path: str, device: torch.device, resolution: int, normalized_tsdf: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load quantized sparse voxel data from NPZ and dequantize.

    Args:
        in_path: path to .npz file
        device: torch device for returned tensors
        resolution: fine grid resolution used for decompression/dequantization
        normalized_tsdf: if True, return normalized tsdf in [-1, 1] scale (no resolution scaling)
                         matching prior behavior. If False, scale using resolution.

    Returns:
        coords: (M, 3) int32 tensor on device
        sdf: (M, 8) float32 tensor on device
    """
    data = np.load(in_path)
    coords_q = data['coords']
    sdf_q = data['sdfs']

    coords_np = decompress_coords(coords_q, int(resolution))
    if normalized_tsdf:
        sdf_np = dequantize_sdf(sdf_q)
    else:
        sdf_np = dequantize_sdf(sdf_q, int(resolution))
    dev = torch.device(device)
    coords = torch.from_numpy(coords_np).int().to(dev)
    sdf = torch.from_numpy(sdf_np).float().to(dev)
    return coords, sdf


def extract_mesh(
    coords: torch.Tensor,
    sdf: torch.Tensor,
    resolution: int,
    cpu_mc: bool = False,
    target_faces: int = 0,
    ensure_consistency: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mesh from sparse voxels and post-process.

    Args:
        coords: (M,3) int32 tensor on device
        sdf: (M,8) float32 tensor on device
        cpu_mc: if True, use CPU marching cubes; else CUDA kernel
        target_faces: if > 0, decimate afterwards
        ensure_consistency: if True, average shared corners across voxels before extraction
        verbose: print timing

    Returns:
        vertices: (V,3) float32 numpy in [-1,1]
        triangles: (F,3) int32 numpy
    """
    t0 = time.time()
    if cpu_mc:
        coords_np = coords.detach().cpu().numpy().astype(np.int32)
        corners_np = sdf.detach().cpu().numpy().astype(np.float32)
        vertices, triangles = cubvh.sparse_marching_cubes_cpu(coords_np, corners_np, 0, ensure_consistency)
        vertices = vertices.astype(np.float32)
        triangles = triangles.astype(np.int32)
    else:
        vertices_t, triangles_t = cubvh.sparse_marching_cubes(coords, sdf, 0, ensure_consistency)
        vertices = vertices_t.detach().cpu().numpy()
        triangles = triangles_t.detach().cpu().numpy()

    # map to [-1, 1]
    vertices = vertices / int(resolution) * 2 - 1
    if verbose:
        print(f'Mesh extraction time: {time.time() - t0:.2f}s')

    # Optional decimation (defer importing kiui/mesh_utils until needed)
    if target_faces and target_faces > 0:
        from kiui.mesh_utils import decimate_mesh
        t0 = time.time()
        vertices, triangles = decimate_mesh(vertices, triangles, 1e6, backend="omo", optimalplacement=False)
        if verbose:
            print(f'Decimation time: {time.time() - t0:.2f}s')

    return vertices, triangles


class SparseVoxelExtractor:

    def __init__(self, min_res: int, res_fine: int, device: torch.device, verbose: bool = False):
        self.device = device
        # hierarchical querying resolutions [res, res*2, res*4, ... res_fine]
        self.res = int(min_res)
        self.res_fine = int(res_fine)
        self.num_levels = int(np.log2(self.res_fine // self.res))
        # res_fine should be 2^res
        assert self.res_fine == 2 ** self.num_levels * self.res
        # fixed sub-block resolution
        self.res_block = 2
        self.verbose = verbose

        # Precompute coarse grid points in NDC [-1, 1]
        self.grid_points = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, self.res + 1, device=self.device),
                torch.linspace(-1, 1, self.res + 1, device=self.device),
                torch.linspace(-1, 1, self.res + 1, device=self.device),
                indexing="ij",
            ),
            dim=-1,
        ).contiguous()  # [res+1, res+1, res+1, 3]

        self.bvh = None
    
    def build_bvh(self, vertices, triangles):
        # Build BVH once, then run coarse and fine extractions sequentially
        t0 = time.time()
        self.bvh = cubvh.cuBVH(vertices, triangles)
        if self.verbose:
            print(f'BVH construction time: {time.time() - t0:.2f}s')


    def extract_sparse_voxels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract sparse active voxels at fine resolution from a triangle mesh.

        Args:
            vertices: (V, 3) float32 in [-1, 1], on self.device
            triangles: (F, 3) int64, on self.device
            verbose: print timing/stats

        Returns:
            a dict contains:
            
            - coords: (M, 3) int32 grid coords in [0, res_fine-1]
            - sdfs: (M, 8) float32 truncated SDF per voxel corner
        """

        assert self.bvh is not None
    
        # Coarse extraction
        res = self.res
        out = self.extract_coarse_voxels(res, last_level=self.num_levels == 0)

        vertices, triangles = extract_mesh(
            out['coords'],
            out['sdfs'],
            resolution=res,
            target_faces=0,
            ensure_consistency=False,
            verbose=True,
        )
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(f'{res}.glb')

        # Hierarchical Fine extraction
        for level in range(self.num_levels):
            res *= 2
            last_level = level == self.num_levels - 1
            out = self.extract_fine_voxels(res, out['coords'], out['sdfs'], last_level=last_level)

            vertices, triangles = extract_mesh(
                out['coords'],
                out['sdfs'],
                resolution=res,
                target_faces=0,
                ensure_consistency=False,
                verbose=True,
            )
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(f'{res}.glb')

        return out

    def extract_coarse_voxels(self, res, last_level: bool = False):
        """
        Perform coarse resolution extraction.

        Returns a dict with keys:
        - occ_ratio: float
        - coords: (N, 3) int32
        - sdfs: (N, 8) float32
        """
        # eps for next level (res * 2)
        eps = 2 / res
        eps_fine = 1 / res

        t0 = time.time()
        udf, _, _ = self.bvh.unsigned_distance(self.grid_points.view(-1, 3), return_uvw=False)
        udf = udf.view(res + 1, res + 1, res + 1).contiguous()

        # mark occupied voxels at coarse band-width
        occ = udf < eps_fine if not last_level else udf < eps

        # floodfill
        floodfill_mask = cubvh.floodfill(occ)
        empty_label = floodfill_mask[0, 0, 0].item()
        empty_mask = (floodfill_mask == empty_label)
        print('empty mask: ', empty_mask.sum().item())

        # get the fine empty_mask (at fine band-width) by contracting the outer band.
        # some negative (inner) mask will be corrected as positive iff. their udf > eps_fine and they have a positive neighbor.
        # delta_mask = (udf > eps_fine) & (udf < eps)
        # print('delta mask: ', delta_mask.sum().item())
        # # empty_mask |= delta_mask
        # # loop until no more empty mask is added
        # num_empty = empty_mask.sum().item()
        # while True:
        #     has_empty_neighbor_mask = torch.zeros_like(delta_mask, dtype=torch.bool)
        #     has_empty_neighbor_mask[:, :, 1:] |= empty_mask[:, :, :-1]
        #     has_empty_neighbor_mask[:, :, :-1] |= empty_mask[:, :, 1:]
        #     has_empty_neighbor_mask[:, 1:, :] |= empty_mask[:, :-1, :]
        #     has_empty_neighbor_mask[:, :-1, :] |= empty_mask[:, 1:, :]
        #     has_empty_neighbor_mask[1:, :, :] |= empty_mask[:-1, :, :]
        #     has_empty_neighbor_mask[:-1, :, :] |= empty_mask[1:, :, :]
        #     corrected_empty_mask = delta_mask & has_empty_neighbor_mask
        #     empty_mask |= corrected_empty_mask
        #     new_num_empty = empty_mask.sum().item()
        #     print('corrected empty mask: ', new_num_empty)
        #     if new_num_empty == num_empty:
        #         break
        #     num_empty = new_num_empty

        # invert to get the occ_mask
        occ_mask = ~empty_mask
        occ_ratio = occ_mask.sum().item() / (res + 1) ** 3

        # Truncated SDF (inner negative)
        sdf = udf - eps_fine if not last_level else udf - eps
        inner_mask = occ_mask & (sdf > 0)
        sdf[inner_mask] *= -1
        kiui.lo(sdf, verbose=1)
  
        # Coarse active voxels (sign change or near-surface border)
        sdf_000 = sdf[:-1, :-1, :-1]
        sdf_001 = sdf[:-1, :-1, 1:]
        sdf_010 = sdf[:-1, 1:, :-1]
        sdf_011 = sdf[:-1, 1:, 1:]
        sdf_100 = sdf[1:, :-1, :-1]
        sdf_101 = sdf[1:, :-1, 1:]
        sdf_110 = sdf[1:, 1:, :-1]
        sdf_111 = sdf[1:, 1:, 1:]

        active_voxel_mask = torch.sign(sdf_000) != torch.sign(sdf_001)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_010)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_011)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_100)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_101)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_110)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_111)
        print('coarse num active voxels: ', active_voxel_mask.sum().item())

        # add more voxels to make sure no fine voxels are left out
        # if not last_level:
        #     border_voxel_mask = torch.minimum(udf[:-1, :-1, :-1], udf[:-1, :-1, 1:])
        #     border_voxel_mask = torch.minimum(border_voxel_mask, udf[:-1, 1:, :-1])
        #     border_voxel_mask = torch.minimum(border_voxel_mask, udf[:-1, 1:, 1:])
        #     border_voxel_mask = torch.minimum(border_voxel_mask, udf[1:, :-1, :-1])
        #     border_voxel_mask = torch.minimum(border_voxel_mask, udf[1:, :-1, 1:])
        #     border_voxel_mask = torch.minimum(border_voxel_mask, udf[1:, 1:, :-1])
        #     border_voxel_mask = torch.minimum(border_voxel_mask, udf[1:, 1:, 1:])
        #     border_voxel_mask = border_voxel_mask <= eps
        #     active_voxel_mask |= border_voxel_mask

        coords_indices = torch.nonzero(active_voxel_mask, as_tuple=True)
        coords = torch.stack(coords_indices, dim=-1)  # [N,3]

        sdfs = torch.stack([
            # order matters! this is the standard marching cubes order of 8 corners
            sdf_000[coords_indices],
            sdf_100[coords_indices],
            sdf_110[coords_indices],
            sdf_010[coords_indices],
            sdf_001[coords_indices],
            sdf_101[coords_indices],
            sdf_111[coords_indices],
            sdf_011[coords_indices],
        ], dim=-1)

        N = coords.shape[0]

        if self.verbose:
            print(f'Coarse Res: {res}, time: {time.time() - t0:.2f}s, active cells: {N} / {res ** 3} = {N / res ** 3 * 100:.2f}%, occ: {occ_ratio:.4f}')

        return {
            'occ_ratio': occ_ratio,
            'coords': coords.int(),
            'sdfs': sdfs.float(),
        }

    def extract_fine_voxels(self, res: int, coarse_coords: torch.Tensor, coarse_sdfs: torch.Tensor, last_level: bool = False):
        """
        Perform fine resolution extraction given coarse outputs.

        Args:
            coarse_coords: (N,3) int32 coarse grid coords
            coarse_sdfs: (N,8) float32 coarse SDFs per voxel corner

        Returns a dict with keys:
        - occ_ratio: float (fine res)
        - coords: (M, 3) int32 fine grid coords
        - sdfs: (M, 8) float32 fine SDFs
        """
        res_coarse = res // 2 # previous level's resolution
        eps = 2 / res
        eps_fine = 1 / res

        # Ensure tensors are on device and expected dtype
        coarse_coords = coarse_coords.to(device=self.device).int()
        coarse_sdfs = coarse_sdfs.to(device=self.device).float()

        # Fine sampling over active cells only
        t0 = time.time()
        coarse_voxel_centers = (coarse_coords + 0.5) / res_coarse * 2 - 1
        N = coarse_coords.shape[0]

        # Subgrid offsets within each coarse cell for the fine sampling
        subgrid_offsets = torch.stack(
            torch.meshgrid(
                torch.linspace(-1 / res_coarse, 1 / res_coarse, self.res_block + 1, device=self.device),
                torch.linspace(-1 / res_coarse, 1 / res_coarse, self.res_block + 1, device=self.device),
                torch.linspace(-1 / res_coarse, 1 / res_coarse, self.res_block + 1, device=self.device),
                indexing="ij",
            ),
            dim=-1,
        ).contiguous()  # [res_block+1, res_block+1, res_block+1, 3]

        grid_points = coarse_voxel_centers.view(N, 1, 1, 1, 3) + subgrid_offsets.view(1, self.res_block + 1, self.res_block + 1, self.res_block + 1, 3)
        grid_points = grid_points.view(-1, 3) # TODO: lots of duplication here, maybe optimize?

        # batched flood fill
        udf, _, _ = self.bvh.unsigned_distance(grid_points, return_uvw=False)
        udf = udf.view(N, self.res_block + 1, self.res_block + 1, self.res_block + 1).contiguous()
        occ = udf < eps_fine if not last_level else udf < eps
        ff_mask = cubvh.floodfill(occ) # [N, self.res_block + 1, self.res_block + 1, self.res_block + 1]

        _num = (occ.sum(dim=(1, 2, 3)) > 0).sum().item()
        print('num occupied coarse voxels: ', _num, "all: ", N)
        # assert _num == N, 'all coarse voxels should be occupied'

        # we need to find out the empty label for each batch (take care of ALL corners with positive sdf!)
        # get the corners of the ff_mask
        ff_corners = torch.stack([
            ff_mask[:, 0, 0, 0],
            ff_mask[:, -1, 0, 0],
            ff_mask[:, -1, -1, 0],
            ff_mask[:, 0, -1, 0],
            ff_mask[:, 0, 0, -1],
            ff_mask[:, -1, 0, -1],
            ff_mask[:, -1, -1, -1],
            ff_mask[:, 0, -1, -1],
        ], dim=1)
        empty_mask = torch.zeros_like(ff_mask, dtype=torch.bool)
        empty_mask |= (ff_corners[:, 0].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 0] > 0).view(-1, 1, 1, 1)
        empty_mask |= (ff_corners[:, 1].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 1] > 0).view(-1, 1, 1, 1)
        empty_mask |= (ff_corners[:, 2].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 2] > 0).view(-1, 1, 1, 1)
        empty_mask |= (ff_corners[:, 3].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 3] > 0).view(-1, 1, 1, 1)
        empty_mask |= (ff_corners[:, 4].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 4] > 0).view(-1, 1, 1, 1)
        empty_mask |= (ff_corners[:, 5].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 5] > 0).view(-1, 1, 1, 1)
        empty_mask |= (ff_corners[:, 6].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 6] > 0).view(-1, 1, 1, 1)
        empty_mask |= (ff_corners[:, 7].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 7] > 0).view(-1, 1, 1, 1)
        occ_mask = ~empty_mask

        sdf = udf - eps_fine if not last_level else udf - eps # [N, self.res_block+1, self.res_block+1, self.res_block+1]
        inner_mask = occ_mask & (sdf > 0)
        sdf[inner_mask] *= -1
        kiui.lo(sdf, verbose=1)
        print('eps = ', eps_fine if not last_level else eps)

        # convert to sparse voxels again
        sdf_000 = sdf[:, :-1, :-1, :-1]
        sdf_001 = sdf[:, :-1, :-1, 1:]
        sdf_010 = sdf[:, :-1, 1:, :-1]
        sdf_011 = sdf[:, :-1, 1:, 1:]
        sdf_100 = sdf[:, 1:, :-1, :-1]
        sdf_101 = sdf[:, 1:, :-1, 1:]
        sdf_110 = sdf[:, 1:, 1:, :-1]
        sdf_111 = sdf[:, 1:, 1:, 1:]

        # keep the active voxels
        active_voxel_mask = torch.sign(sdf_000) != torch.sign(sdf_001)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_010)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_011)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_100)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_101)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_110)
        active_voxel_mask |= torch.sign(sdf_000) != torch.sign(sdf_111)
        print('num active voxels: ', active_voxel_mask.sum().item())

        # if not the final level, add more voxels to make sure no finer voxels are left out
        if not last_level:
            border_voxel_mask = torch.minimum(udf[:, :-1, :-1, :-1], udf[:, :-1, :-1, 1:])
            border_voxel_mask = torch.minimum(border_voxel_mask, udf[:, :-1, 1:, :-1])
            border_voxel_mask = torch.minimum(border_voxel_mask, udf[:, :-1, 1:, 1:])
            border_voxel_mask = torch.minimum(border_voxel_mask, udf[:, 1:, :-1, :-1])
            border_voxel_mask = torch.minimum(border_voxel_mask, udf[:, 1:, :-1, 1:])
            border_voxel_mask = torch.minimum(border_voxel_mask, udf[:, 1:, 1:, :-1])
            border_voxel_mask = torch.minimum(border_voxel_mask, udf[:, 1:, 1:, 1:])
            border_voxel_mask = border_voxel_mask <= eps
            active_voxel_mask |= border_voxel_mask
            
        coords_indices_local = torch.nonzero(active_voxel_mask, as_tuple=True)
        coords_local = torch.stack(coords_indices_local[1:], dim=-1)
        M = coords_local.shape[0]

        sdfs = torch.stack([
            sdf_000[coords_indices_local],
            sdf_100[coords_indices_local],
            sdf_110[coords_indices_local],
            sdf_010[coords_indices_local],
            sdf_001[coords_indices_local],
            sdf_101[coords_indices_local],
            sdf_111[coords_indices_local],
            sdf_011[coords_indices_local],
        ], dim=-1)

        coords = self.res_block * coarse_coords[coords_indices_local[0]] + coords_local

        out = {
            'coords': coords.int(),
            'sdfs': sdfs.float(),
        }

        if self.verbose:
            print(f'Fine Res: {res}, time: {time.time() - t0:.2f}s, active cells: {M} / {res ** 3} = {M / res ** 3 * 100:.2f}%')

        return out