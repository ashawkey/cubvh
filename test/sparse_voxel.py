import io
import time
import math
import argparse
import numpy as np
import torch
import trimesh
import cubvh
from typing import Optional
from functools import wraps
from rich.console import Console
import os

# fixed parameters
res = 1024
device = torch.device('cuda')
eps = 2 / res

grid_points = torch.stack(
    torch.meshgrid(
        torch.linspace(-1, 1, res + 1, device=device),
        torch.linspace(-1, 1, res + 1, device=device),
        torch.linspace(-1, 1, res + 1, device=device),
        indexing="ij",
    ),
    dim=-1,
)  # [min_res + 1, min_res + 1, min_res + 1, 3]


class sync_timer:
    """
    Synchronized timer to count the inference time and GPU memory usage of `nn.Module.forward` or else.
    set env var TIMER=1 to enable timing logging!
    set env var MEM=1 to enable memory logging!

    Example as context manager:
    ```python
    with timer('name'):
        run()
    ```

    Example as decorator:
    ```python
    @timer('name')
    def run():
        pass
    ```
    """

    def __init__(self, name, flag_env="TIMER", memory_flag_env="MEM"):
        self.name = name
        self.flag_env = flag_env
        self.memory_flag_env = memory_flag_env
        self.memory_start = None
        self.memory_end = None
        self.memory_peak = None
        self.console = Console()

    def __enter__(self):
        if os.environ.get(self.flag_env, "0") == "1":
            # Timing setup
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

            # Memory setup
            if os.environ.get(self.memory_flag_env, "0") == "1" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()  # Reset peak memory counter
                self.memory_start = torch.cuda.memory_allocated()

            self.start.record(stream=torch.cuda.current_stream())
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        if os.environ.get(self.flag_env, "0") == "1":
            self.end.record(stream=torch.cuda.current_stream())
            torch.cuda.synchronize()

            # Calculate timing
            self.time = self.start.elapsed_time(self.end)

            # Calculate memory usage
            if (
                os.environ.get(self.memory_flag_env, "0") == "1"
                and torch.cuda.is_available()
                and self.memory_start is not None
            ):
                self.memory_end = torch.cuda.memory_allocated()
                self.memory_peak = torch.cuda.max_memory_allocated()
                memory_used = (self.memory_end - self.memory_start) / 1024**2  # Convert to MB
                peak_memory = self.memory_peak / 1024**2  # Convert to MB

                self.console.print(
                    f"[red][bold]{self.name}[/bold][/red]: "
                    f"[green]time =[/green] {self.time:.2f} ms, "
                    f"[green]delta memory =[/green] {memory_used:.2f} MB, "
                    f"[green]peak memory =[/green] {peak_memory:.2f} MB"
                )
            else:
                self.console.print(f"[red][bold]{self.name}[/bold][/red]: " f"[green]time =[/green] {self.time:.2f} ms")

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper
    
def box_normalize(vertices: np.ndarray, bound: float = 0.95) -> np.ndarray:
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    vertices = 2 * bound * (vertices - bcenter) / (bmax - bmin).max()
    return vertices


def compress_coords(coords: np.ndarray, res: int) -> np.ndarray:
    # flatten [N, 3] coords to a single uint64 (since 2048^3 exceeds uint32)
    coords = coords.astype(np.uint64)
    return coords[:, 0] * res * res + coords[:, 1] * res + coords[:, 2]


def decompress_coords(compressed: np.ndarray, res: int) -> np.ndarray:
    # decompress a single uint64 to [3] coords
    x = compressed // (res * res)
    y = (compressed % (res * res)) // res
    z = compressed % res  # [N]
    return np.stack([x, y, z], axis=-1)  # [N, 3]


def quantize_sdf(sdf: np.ndarray, resolution: int) -> tuple[np.ndarray, float]:
    """
    Parameters
    ----------
    sdf : (N, 8) float32
        Signed-distance values at the 8 cube corners of every voxel.
    resolution : int
        Resolution of the grid. assume normalized sdf to [-1, 1].

    Returns
    -------
    packed : (N,) uint64
        Eight 8-bit codes per voxel packed little-endian into a 64-bit word.
        Corner 0 ends up in the least-significant byte, corner 7 in the MSB.
    delta : float
        Quantisation step so that   reconstructed_f = code * delta.
        Needed by `dequantize_sdf`.
    """
    if sdf.ndim != 2 or sdf.shape[1] != 8:
        raise ValueError("sdf must have shape [N, 8]")
    delta = 2 / resolution * math.sqrt(3) / 127.0
    mag = np.rint(np.clip(np.abs(sdf) / delta, 0, 127)).astype(np.uint8)
    sign = (sdf < 0).astype(np.uint8) << 7  # 0x80 if inside
    codes = (sign | mag).astype(np.uint8)  # shape (N, 8), dtype uint8
    shifts = (np.arange(8, dtype=np.uint64) * 8)[None, :]  # (1, 8)
    packed = np.sum((codes.astype(np.uint64) << shifts), axis=1, dtype=np.uint64)
    return packed, delta


def dequantize_sdf(packed: np.ndarray, resolution: Optional[int] = None) -> np.ndarray:  # type: ignore
    """
    Inverse of quantize_sdf.

    Parameters
    ----------
    packed : (N,) uint64
        Array that came out of quantize_sdf.
    resolution : int
        Resolution of the grid, if provided, will be used to compute the delta and reconstruct the exact sdf.
        Otherwise, we'll return truncated & normalized sdf to [-1, 1].

    Returns
    -------
    sdf : (N, 8) float32
        Reconstructed signed-distance samples.
    """
    if packed.dtype != np.uint64:
        raise ValueError("packed must be dtype uint64")
    shifts = (np.arange(8, dtype=np.uint64) * 8)[None, :]  # (1, 8)
    bytes_ = ((packed[:, None] >> shifts) & 0xFF).astype(np.uint8)  # (N, 8)
    sign = ((bytes_ >> 7) & 1).astype(np.int8)  # 0 outside, 1 inside
    sdf = (bytes_ & 0x7F).astype(np.float32) / 127.0  # 0 … 1
    if resolution is not None:
        sdf = sdf * (2 / resolution * math.sqrt(3))
    sdf[sign == 1] *= -1.0
    return sdf

def extract_spvox(path: str) -> tuple[torch.Tensor, torch.Tensor, float, int]:

    # load mesh
    with sync_timer('Load mesh'):
        mesh: trimesh.Trimesh = trimesh.load(path, process=False, force="mesh")
        mesh.vertices = box_normalize(mesh.vertices, bound=0.95)
        vertices = torch.from_numpy(mesh.vertices).float()
        triangles = torch.from_numpy(mesh.faces).long()

    # construct BVH
    with sync_timer('Build BVH'):
        BVH = cubvh.cuBVH(vertices, triangles)

    # coarsest resolution: dense grid query
    with sync_timer('Dense grid query'):
        udf, _, _ = BVH.unsigned_distance(grid_points.contiguous().view(-1, 3), return_uvw=False)
        udf = udf.view(res + 1, res + 1, res + 1).contiguous()
        occ = udf < eps  # use fine eps even at coarsest level!

    with sync_timer('Floodfill'):
        floodfill_mask = cubvh.floodfill(occ)
        empty_label = floodfill_mask[0, 0, 0].item()
        empty_mask = floodfill_mask == empty_label
        # empty_mask = morphology.flood(occ.cpu().numpy(), (0, 0, 0), connectivity=1)
        # empty_mask = torch.from_numpy(empty_mask).bool().to(self.device)
        occ_mask = ~empty_mask
        occ_ratio = occ_mask.sum().item() / (res**3)

    with sync_timer('Extract spvox'):
        sdf = udf - eps  # inner is negative
        inner_mask = occ_mask & (sdf > 0)
        sdf[inner_mask] *= -1  # [res + 1, res + 1, res + 1]
        sdf_000 = sdf[:-1, :-1, :-1]
        sdf_001 = sdf[:-1, :-1, 1:]
        sdf_010 = sdf[:-1, 1:, :-1]
        sdf_011 = sdf[:-1, 1:, 1:]
        sdf_100 = sdf[1:, :-1, :-1]
        sdf_101 = sdf[1:, :-1, 1:]
        sdf_110 = sdf[1:, 1:, :-1]
        sdf_111 = sdf[1:, 1:, 1:]
        # keep active cells with different sign at corners
        active_cell_mask = torch.sign(sdf_000) != torch.sign(sdf_001)
        active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_010)
        active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_011)
        active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_100)
        active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_101)
        active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_110)
        active_cell_mask |= torch.sign(sdf_000) != torch.sign(sdf_111)
        
        # get active cells and corner sdf values
        active_cells_index = torch.nonzero(active_cell_mask, as_tuple=True)  # ([N], [N], [N])
        active_cells = torch.stack(active_cells_index, dim=-1)  # [N, 3]
        active_cells_sdf = torch.stack(
            [
                # order matters! this is the standard marching cubes order of 8 corners
                sdf_000[active_cells_index],
                sdf_100[active_cells_index],
                sdf_110[active_cells_index],
                sdf_010[active_cells_index],
                sdf_001[active_cells_index],
                sdf_101[active_cells_index],
                sdf_111[active_cells_index],
                sdf_011[active_cells_index],
            ],
            dim=-1,
        )  # [N, 8]
        
        num_voxels = active_cells.shape[0]

    return active_cells, active_cells_sdf, occ_ratio, num_voxels

def process_data(mesh_path: str, workspace: str) -> None:

    print(f"Processing {mesh_path}...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream=torch.cuda.current_stream())
    active_cells, active_cells_sdf, occ_ratio, num_voxels = extract_spvox(mesh_path)

    # quantize
    active_cells_np = active_cells.cpu().numpy()
    active_cells_sdf_np = active_cells_sdf.cpu().numpy()
    active_cells_np = compress_coords(active_cells_np, res)
    active_cells_sdf_np, _ = quantize_sdf(active_cells_sdf_np, res)

   # save
    np.savez_compressed(
        os.path.join(workspace, os.path.basename(mesh_path).replace('.glb', '.npz')),
        active_cells=active_cells_np,
        active_cells_sdf=active_cells_sdf_np,
    )
    torch.cuda.synchronize()
    time = start.elapsed_time(end) / 1000.0 
    return time

if __name__ == "__main__":
    import argparse
    import os
    import glob
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('test_path', type=str)
    parser.add_argument('--workspace', type=str, default='output')
    args = parser.parse_args()

    os.makedirs(args.workspace, exist_ok=True)

    if os.path.isdir(args.test_path):
        file_paths = glob.glob(os.path.join(args.test_path, "*"))
        all_time = []
        for path in tqdm.tqdm(file_paths):
            try:
                t = process_data(path, args.workspace)
                all_time.append(t)
            except Exception as e:
                print(f'[WARN] {path} failed: {e}')
        print(f'Mean process time: {np.mean(all_time):.4f}s')
    else:
        t = process_data(args.test_path, args.workspace)
        print(f'Process time: {t:.4f}s')