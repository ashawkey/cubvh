import os
import math
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import cubvh


class SparseVoxelExtractor:
    """
    Block-sparse marching cubes pipeline.
    - Hold resolution-dependent constants (grid_points, subgrid_offsets, eps_fine).
    - Extract sparse voxels from a mesh surface using a coarse-to-fine BVH UDF query.
    - Save/load quantized sparse voxel data (coords + 8-corner SDFs).
    - Extract a mesh from sparse voxels using sparse marching cubes (CPU or CUDA).
    """

    def __init__(self, min_res: int, res_fine: int, device: torch.device, verbose: bool = False):
        self.res = int(min_res)
        self.res_fine = int(res_fine)
        self.device = device
        self.res_block = self.res_fine // self.res
        # use fine epsilon as in original implementation
        self.eps_fine = 2 / self.res_fine
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

        # Subgrid offsets within each coarse cell for the fine sampling
        self.subgrid_offsets = torch.stack(
            torch.meshgrid(
                torch.linspace(-1 / self.res, 1 / self.res, self.res_block + 1, device=self.device),
                torch.linspace(-1 / self.res, 1 / self.res, self.res_block + 1, device=self.device),
                torch.linspace(-1 / self.res, 1 / self.res, self.res_block + 1, device=self.device),
                indexing="ij",
            ),
            dim=-1,
        ).contiguous()  # [res_block+1, res_block+1, res_block+1, 3]

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
        coarse = self.extract_sparse_voxels_coarse()

        # Fine extraction
        fine = self.extract_sparse_voxels_fine(coarse['coarse_coords'], coarse['coarse_sdfs'])

        # Merge outputs for backward compatibility
        out = {
            **coarse,
            **fine,
        }
        return out

    def extract_sparse_voxels_coarse(self):
        """
        Perform coarse resolution extraction.

        Returns a dict with keys:
        - coarse_occ_ratio: float
        - coarse_coords: (N, 3) int32
        - coarse_sdfs: (N, 8) float32
        """
        res = self.res
        eps_fine = self.eps_fine

        t0 = time.time()
        udf, _, _ = self.bvh.unsigned_distance(self.grid_points.view(-1, 3), return_uvw=False)
        udf = udf.view(res + 1, res + 1, res + 1).contiguous()

        # mark occupied voxels
        occ = udf < eps_fine
        cube_diagonal_length = math.sqrt(3) / res
        border_voxel_mask = torch.minimum(udf[:-1, :-1, :-1], udf[:-1, :-1, 1:])
        border_voxel_mask = torch.minimum(border_voxel_mask, udf[:-1, 1:, :-1])
        border_voxel_mask = torch.minimum(border_voxel_mask, udf[:-1, 1:, 1:])
        border_voxel_mask = torch.minimum(border_voxel_mask, udf[1:, :-1, :-1])
        border_voxel_mask = torch.minimum(border_voxel_mask, udf[1:, :-1, 1:])
        border_voxel_mask = torch.minimum(border_voxel_mask, udf[1:, 1:, :-1])
        border_voxel_mask = torch.minimum(border_voxel_mask, udf[1:, 1:, 1:])
        border_voxel_mask = border_voxel_mask <= cube_diagonal_length
        # simple center-check
        udf_center = F.avg_pool3d(udf[None, None], kernel_size=2, stride=1).squeeze()
        border_voxel_mask &= (udf_center <= cube_diagonal_length)
        # mark all 8 corners of border voxels as occupied
        occ[:-1, :-1, :-1] |= border_voxel_mask
        occ[:-1, :-1, 1:] |= border_voxel_mask
        occ[:-1, 1:, :-1] |= border_voxel_mask
        occ[:-1, 1:, 1:] |= border_voxel_mask
        occ[1:, :-1, :-1] |= border_voxel_mask
        occ[1:, :-1, 1:] |= border_voxel_mask
        occ[1:, 1:, :-1] |= border_voxel_mask
        occ[1:, 1:, 1:] |= border_voxel_mask

        # floodfill
        floodfill_mask = cubvh.floodfill(occ)
        empty_label = floodfill_mask[0, 0, 0].item()
        empty_mask = (floodfill_mask == empty_label)
        occ_mask = ~empty_mask
        occ_ratio = occ_mask.sum().item() / (res + 1) ** 3

        # Truncated SDF (inner negative)
        sdf = udf - eps_fine
        inner_mask = occ_mask & (sdf > 0)
        sdf[inner_mask] *= -1

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

        # double confirm that border voxels are also active
        active_voxel_mask |= border_voxel_mask

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
            print(f'Res: {res}, time: {time.time() - t0:.2f}s, active cells: {N} / {res ** 3} = {N / res ** 3 * 100:.2f}%, occ: {occ_ratio:.4f}')

        return {
            'coarse_occ_ratio': occ_ratio,
            'coarse_coords': coords.int(),
            'coarse_sdfs': sdfs.float(),
        }

    def extract_sparse_voxels_fine(self, coarse_coords: torch.Tensor, coarse_sdfs: torch.Tensor, return_diff: bool = True):
        """
        Perform fine resolution extraction given coarse outputs.

        Args:
            coarse_coords: (N,3) int32 coarse grid coords
            coarse_sdfs: (N,8) float32 coarse SDFs per voxel corner
            return_diff: bool, if return the diff_detail metric.

        Returns a dict with keys:
        - occ_ratio: float (fine res)
        - diff_detail: float, the difference between interpolated coarse sdf and actual fine sdf, measures how the fine mesh is different from coarse mesh.
        - coords: (M, 3) int32 fine grid coords
        - sdfs: (M, 8) float32 fine SDFs
        """
        res = self.res
        res_fine = self.res_fine
        res_block = self.res_block
        eps_fine = self.eps_fine

        # Ensure tensors are on device and expected dtype
        coarse_coords = coarse_coords.to(device=self.device).int()
        coarse_sdfs = coarse_sdfs.to(device=self.device).float()

        # Fine sampling over active cells only
        t0 = time.time()
        coarse_voxel_centers = (coarse_coords + 0.5) / (res) * 2 - 1
        N = coarse_coords.shape[0]
        fine_grid = coarse_voxel_centers.view(N, 1, 1, 1, 3) + self.subgrid_offsets.view(1, res_block + 1, res_block + 1, res_block + 1, 3)
        fine_grid = fine_grid.view(-1, 3)

        # batched flood fill
        udf_fine, _, _ = self.bvh.unsigned_distance(fine_grid, return_uvw=False)
        udf_fine = udf_fine.view(N, res_block + 1, res_block + 1, res_block + 1).contiguous()
        occ_fine = udf_fine < eps_fine
        ff_mask = cubvh.floodfill(occ_fine) # [N, res_block + 1, res_block + 1, res_block + 1]

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
        fine_empty_mask = torch.zeros_like(ff_mask, dtype=torch.bool)
        fine_empty_mask |= (ff_corners[:, 0].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 0] > 0).view(-1, 1, 1, 1)
        fine_empty_mask |= (ff_corners[:, 1].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 1] > 0).view(-1, 1, 1, 1)
        fine_empty_mask |= (ff_corners[:, 2].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 2] > 0).view(-1, 1, 1, 1)
        fine_empty_mask |= (ff_corners[:, 3].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 3] > 0).view(-1, 1, 1, 1)
        fine_empty_mask |= (ff_corners[:, 4].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 4] > 0).view(-1, 1, 1, 1)
        fine_empty_mask |= (ff_corners[:, 5].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 5] > 0).view(-1, 1, 1, 1)
        fine_empty_mask |= (ff_corners[:, 6].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 6] > 0).view(-1, 1, 1, 1)
        fine_empty_mask |= (ff_corners[:, 7].view(-1, 1, 1, 1) == ff_mask) & (coarse_sdfs[:, 7] > 0).view(-1, 1, 1, 1)
        fine_occ_mask = ~fine_empty_mask

        sdf_fine = udf_fine - eps_fine # [N, res_block+1, res_block+1, res_block+1]
        fine_inner_mask = fine_occ_mask & (sdf_fine > 0)
        sdf_fine[fine_inner_mask] *= -1

        # convert to sparse voxels again
        sdf_fine_000 = sdf_fine[:, :-1, :-1, :-1]
        sdf_fine_001 = sdf_fine[:, :-1, :-1, 1:]
        sdf_fine_010 = sdf_fine[:, :-1, 1:, :-1]
        sdf_fine_011 = sdf_fine[:, :-1, 1:, 1:]
        sdf_fine_100 = sdf_fine[:, 1:, :-1, :-1]
        sdf_fine_101 = sdf_fine[:, 1:, :-1, 1:]
        sdf_fine_110 = sdf_fine[:, 1:, 1:, :-1]
        sdf_fine_111 = sdf_fine[:, 1:, 1:, 1:]

        # this time we only keep the active voxels
        fine_active_voxel_mask = torch.sign(sdf_fine_000) != torch.sign(sdf_fine_001)
        fine_active_voxel_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_010)
        fine_active_voxel_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_011)
        fine_active_voxel_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_100)
        fine_active_voxel_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_101)
        fine_active_voxel_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_110)
        fine_active_voxel_mask |= torch.sign(sdf_fine_000) != torch.sign(sdf_fine_111)

        coords_indices_local = torch.nonzero(fine_active_voxel_mask, as_tuple=True)
        coords_local = torch.stack(coords_indices_local[1:], dim=-1)
        M = coords_local.shape[0]

        sdfs = torch.stack([
            sdf_fine_000[coords_indices_local],
            sdf_fine_100[coords_indices_local],
            sdf_fine_110[coords_indices_local],
            sdf_fine_010[coords_indices_local],
            sdf_fine_001[coords_indices_local],
            sdf_fine_101[coords_indices_local],
            sdf_fine_111[coords_indices_local],
            sdf_fine_011[coords_indices_local],
        ], dim=-1)

        coords = res_block * coarse_coords[coords_indices_local[0]] + coords_local

        out = {
            'coords': coords.int(),
            'sdfs': sdfs.float(),
        }

        # calculate diff between sdf_fine and sdf_coarse
        if return_diff:
            # convert 8 corners to 2x2x2 grid
            coarse_sdfs_grid = torch.zeros(N, 2, 2, 2, device=self.device, dtype=torch.float32)
            coarse_sdfs_grid[:, 0, 0, 0] = coarse_sdfs[:, 0]
            coarse_sdfs_grid[:, 1, 0, 0] = coarse_sdfs[:, 1]
            coarse_sdfs_grid[:, 1, 1, 0] = coarse_sdfs[:, 2]
            coarse_sdfs_grid[:, 0, 1, 0] = coarse_sdfs[:, 3]
            coarse_sdfs_grid[:, 0, 0, 1] = coarse_sdfs[:, 4]
            coarse_sdfs_grid[:, 1, 0, 1] = coarse_sdfs[:, 5]
            coarse_sdfs_grid[:, 1, 1, 1] = coarse_sdfs[:, 6]
            coarse_sdfs_grid[:, 0, 1, 1] = coarse_sdfs[:, 7]
            # interpolate from 2x2x2 to res_block+1 x res_block+1 x res_block+1
            interp_sdfs = F.interpolate(coarse_sdfs_grid.unsqueeze(0), size=(res_block + 1, res_block + 1, res_block + 1), mode='trilinear', align_corners=True).squeeze(0) # [N, res_block+1, res_block+1, res_block+1]
            diff_detail = torch.abs(interp_sdfs - sdf_fine).mean()
            out['diff_detail'] = diff_detail
            if self.verbose:
                print(f'Diff detail: {diff_detail:.4f}')

        if self.verbose:
            print(f'Res: {res_fine}, time: {time.time() - t0:.2f}s, active cells: {M} / {(res_fine + 1) ** 3} = {M / (res_fine + 1) ** 3 * 100:.2f}%')

        return out


    @staticmethod
    def sphere_normalize(vertices: np.ndarray) -> np.ndarray:
        bmin = vertices.min(axis=0)
        bmax = vertices.max(axis=0)
        bcenter = (bmax + bmin) / 2
        radius = np.linalg.norm(vertices - bcenter, axis=-1).max()
        vertices = (vertices - bcenter) / (radius)
        return vertices

    @staticmethod
    def box_normalize(vertices: np.ndarray, bound: float = 0.95) -> np.ndarray:
        bmin = vertices.min(axis=0)
        bmax = vertices.max(axis=0)
        bcenter = (bmax + bmin) / 2
        vertices = 2 * bound * (vertices - bcenter) / (bmax - bmin).max()
        return vertices

    @staticmethod
    def compress_coords(coords: np.ndarray, res: int) -> np.ndarray:
        # Flatten [N, 3] coords to a single uint64 (since 2048^3 exceeds uint32)
        coords = coords.astype(np.uint64)
        return coords[:, 0] * res * res + coords[:, 1] * res + coords[:, 2]

    @staticmethod
    def decompress_coords(compressed: np.ndarray, res: int) -> np.ndarray:
        # Decompress a single uint64 to [3] coords
        x = compressed // (res * res)
        y = (compressed % (res * res)) // res
        z = compressed % res
        return np.stack([x, y, z], axis=-1)

    @staticmethod
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

    @staticmethod
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
    
    def save_quantized(self, coords: torch.Tensor, sdf: torch.Tensor, out_path: str) -> float:
        """
        Save quantized sparse voxel data to NPZ.

        Args:
            coords: (M, 3) int grid coords
            sdf: (M, 8) float truncated SDF per corner (normalized as produced)
            out_path: path without extension or with .npz

        Returns:
            quant_delta: float delta used during quantization
        """
        coords_np = coords.detach().cpu().numpy()
        sdf_np = sdf.detach().cpu().numpy()

        coords_q = self.compress_coords(coords_np, self.res_fine)
        sdf_q, delta = self.quantize_sdf(sdf_np, self.res_fine)

        if not out_path.endswith('.npz'):
            out_path = out_path + '.npz'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, coords=coords_q, sdfs=sdf_q)
        return delta

    def load_quantized(self, in_path: str, normalized_tsdf: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load quantized sparse voxel data from NPZ and dequantize.

        Args:
            in_path: path to .npz file
            normalized_tsdf: if True, return normalized tsdf in [-1, 1] scale (no resolution scaling)
                              matching prior behavior. If False, scale using resolution.

        Returns:
            coords: (M, 3) int32 tensor on self.device
            sdf: (M, 8) float32 tensor on self.device
        """
        data = np.load(in_path)
        coords_q = data['coords']
        sdf_q = data['sdfs']

        coords = torch.from_numpy(self.decompress_coords(coords_q, self.res_fine)).int().to(self.device)
        if normalized_tsdf:
            sdf_np = self.dequantize_sdf(sdf_q)
        else:
            sdf_np = self.dequantize_sdf(sdf_q, self.res_fine)
        sdf = torch.from_numpy(sdf_np).float().to(self.device)
        return coords, sdf
    
    @staticmethod
    def extract_mesh(
        coords: torch.Tensor,
        sdf: torch.Tensor,
        resolution: int,
        cpu_mc: bool = False,
        target_faces: int = 0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract mesh from sparse voxels and post-process.

        Args:
            coords: (M,3) int32 tensor on device
            sdf: (M,8) float32 tensor on device
            cpu_mc: if True, use CPU marching cubes; else CUDA kernel
            target_faces: if > 0, decimate afterwards
            verbose: print timing

        Returns:
            vertices: (V,3) float32 numpy in [-1,1]
            triangles: (F,3) int32 numpy
        """
        t0 = time.time()
        if cpu_mc:
            coords_np = coords.detach().cpu().numpy().astype(np.int32)
            corners_np = sdf.detach().cpu().numpy().astype(np.float32)
            vertices, triangles = cubvh.sparse_marching_cubes_cpu(coords_np, corners_np, 0, True)
            vertices = vertices.astype(np.float32)
            triangles = triangles.astype(np.int32)
        else:
            vertices_t, triangles_t = cubvh.sparse_marching_cubes(coords, sdf, 0, True)
            vertices = vertices_t.detach().cpu().numpy()
            triangles = triangles_t.detach().cpu().numpy()

        # map to [-1, 1]
        vertices = vertices / resolution * 2 - 1
        if verbose:
            print(f'Mesh extraction time: {time.time() - t0:.2f}s')

        # merge and fill
        # t0 = time.time()
        # vertices, triangles = cubvh.merge_vertices(vertices, triangles, threshold=1 / resolution)
        # if verbose:
        #     print(f'Merge close vertices time: {time.time() - t0:.2f}s')
        # t0 = time.time()
        # triangles = cubvh.fill_holes(vertices, triangles)
        # if verbose:
        #     print(f'Fill holes time: {time.time() - t0:.2f}s')

        # Optional decimation (defer importing kiui/mesh_utils until needed)
        if target_faces and target_faces > 0:
            from kiui.mesh_utils import decimate_mesh
            t0 = time.time()
            vertices, triangles = decimate_mesh(vertices, triangles, 1e6, backend="omo", optimalplacement=False)
            if verbose:
                print(f'Decimation time: {time.time() - t0:.2f}s')

        return vertices, triangles
