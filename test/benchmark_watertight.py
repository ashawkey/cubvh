import os
import json
import time
import math
import argparse
import numpy as np

import torch
import trimesh
import mcubes

from skimage.morphology import flood

import cubvh


def box_normalize(vertices: np.ndarray, bound: float = 0.95) -> np.ndarray:
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    bcenter = (bmax + bmin) / 2
    vertices = 2 * bound * (vertices - bcenter) / (bmax - bmin).max()
    return vertices


def make_linspace_grid(res: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(-1, 1, res, device=device)
    ys = torch.linspace(-1, 1, res, device=device)
    zs = torch.linspace(-1, 1, res, device=device)
    return xs, ys, zs


@torch.no_grad()
def compute_udf_dense(BVH: cubvh.cuBVH, res: int, device: torch.device, z_chunk: int = 64, verbose: bool = False) -> torch.Tensor:
    xs, ys, zs = make_linspace_grid(res, device)
    udf = torch.empty((res, res, res), device=device, dtype=torch.float32)
    num_chunks = math.ceil(res / z_chunk)
    chunk_idx = 0
    t0 = time.perf_counter()
    for z_start in range(0, res, z_chunk):
        z_end = min(res, z_start + z_chunk)
        X, Y, Z = torch.meshgrid(xs, ys, zs[z_start:z_end], indexing="ij")
        pts = torch.stack([X, Y, Z], dim=-1).contiguous().view(-1, 3)
        d, _, _ = BVH.unsigned_distance(pts, return_uvw=False)
        udf[:, :, z_start:z_end] = d.view(res, res, z_end - z_start)
        chunk_idx += 1
        if verbose:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(f"[{time.strftime('%H:%M:%S')}] UDF chunks {chunk_idx}/{num_chunks} done; elapsed {elapsed:.3f}s", flush=True)
    return udf


def time_op(callable_op):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = callable_op()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def run_benchmark(mesh_path: str, res_list: list[int], repeats: int, output_json: str | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}", flush=True)
    print(f"Loading mesh: {mesh_path}", flush=True)
    dt_load, mesh = time_op(lambda: trimesh.load(mesh_path, process=False, force='mesh'))
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces in {dt_load:.3f}s", flush=True)

    t0 = time.perf_counter()
    mesh.vertices = box_normalize(mesh.vertices, bound=0.95)
    t_norm = time.perf_counter() - t0
    print(f"Normalized vertices in {t_norm:.3f}s", flush=True)

    dt_to_dev, out = time_op(lambda: (torch.from_numpy(mesh.vertices).float().to(device), torch.from_numpy(mesh.faces).long().to(device)))
    vertices, triangles = out
    print(f"Moved tensors to {device} in {dt_to_dev:.3f}s", flush=True)

    results = {"meta": {"mesh": mesh_path, "repeats": repeats}, "ff": {}, "mc": {}}

    # Build BVH once per run; its cost is not measured here
    print("Building BVH...", flush=True)
    dt_bvh, BVH = time_op(lambda: cubvh.cuBVH(vertices, triangles))
    print(f"BVH built in {dt_bvh:.3f}s", flush=True)

    for res in res_list:
        eps = 2.0 / float(res)
        print(f"\n=== Resolution {res} (eps={eps:.6f}) ===", flush=True)

        # Compute UDF once per resolution
        print("Computing UDF...", flush=True)
        dt_udf, udf = time_op(lambda: compute_udf_dense(BVH, res, device, verbose=True))
        print(f"UDF computed in {dt_udf:.3f}s", flush=True)

        # Occupancy mask
        dt_occ, occ = time_op(lambda: udf < eps)
        print(f"Occupancy computed in {dt_occ:.3f}s", flush=True)

        # Flood fill benchmarks
        results["ff"].setdefault(str(res), {})

        # cubvh floodfill (GPU)
        cubvh_times = []
        for i in range(repeats):
            def _ff_cubvh():
                return cubvh.floodfill(occ)
            dt, floodfill_mask = time_op(_ff_cubvh)
            cubvh_times.append(dt)
            print(f"cubvh floodfill run {i+1}/{repeats}: {dt:.3f}s", flush=True)
        # compute empty mask from last run
        empty_label = floodfill_mask[0, 0, 0].item()
        empty_mask_cubvh = (floodfill_mask == empty_label)
        results["ff"][str(res)]["cubvh"] = {
            "times": cubvh_times,
            "mean": float(np.mean(cubvh_times)),
            "std": float(np.std(cubvh_times)),
        }

        # skimage flood_fill (CPU)
        sk_times = []
        dt_occ_np, occ_np = time_op(lambda: occ.detach().cpu().numpy())
        print(f"Converted occupancy to numpy in {dt_occ_np:.3f}s", flush=True)
        for i in range(repeats):
            def _ff_sk():
                return flood(occ_np, (0, 0, 0), connectivity=1)
            dt, empty_mask_np = time_op(_ff_sk)
            sk_times.append(dt)
            print(f"skimage flood run {i+1}/{repeats}: {dt:.3f}s", flush=True)
        empty_mask_sk = torch.from_numpy(empty_mask_np).to(device)
        results["ff"][str(res)]["skimage"] = {
            "times": sk_times,
            "mean": float(np.mean(sk_times)),
            "std": float(np.std(sk_times)),
        }

        # Build SDF (shared by all MC variants)
        def _build_sdf():
            occ_mask_local = ~empty_mask_cubvh  # consistent with main script default
            sdf_local = udf - eps
            inner_mask_local = occ_mask_local & (sdf_local > 0)
            sdf_local[inner_mask_local] *= -1
            return sdf_local, occ_mask_local, inner_mask_local

        dt_sdf, sdf_out = time_op(_build_sdf)
        sdf, occ_mask, inner_mask = sdf_out
        print(f"SDF built in {dt_sdf:.3f}s", flush=True)

        # MC benchmarks
        results["mc"].setdefault(str(res), {})

        # Prepare sparse MC inputs (shared)
        def _prepare_sparse_inputs():
            sdf_000 = sdf[:-1, :-1, :-1]
            sdf_001 = sdf[:-1, :-1, 1:]
            sdf_010 = sdf[:-1, 1:, :-1]
            sdf_011 = sdf[:-1, 1:, 1:]
            sdf_100 = sdf[1:, :-1, :-1]
            sdf_101 = sdf[1:, :-1, 1:]
            sdf_110 = sdf[1:, 1:, :-1]
            sdf_111 = sdf[1:, 1:, 1:]
            sdf_mask = torch.sign(sdf_000) != torch.sign(sdf_001)
            sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_010)
            sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_011)
            sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_100)
            sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_101)
            sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_110)
            sdf_mask |= torch.sign(sdf_000) != torch.sign(sdf_111)
            active_cells_index = torch.nonzero(sdf_mask, as_tuple=True)
            active_cells_local = torch.stack(active_cells_index, dim=-1)
            active_cells_sdf_local = torch.stack([
                sdf_000[active_cells_index],
                sdf_100[active_cells_index],
                sdf_110[active_cells_index],
                sdf_010[active_cells_index],
                sdf_001[active_cells_index],
                sdf_101[active_cells_index],
                sdf_111[active_cells_index],
                sdf_011[active_cells_index],
            ], dim=-1)
            return active_cells_local, active_cells_sdf_local

        dt_sparse, sparse_out = time_op(_prepare_sparse_inputs)
        active_cells, active_cells_sdf = sparse_out
        print(f"Prepared sparse MC inputs in {dt_sparse:.3f}s (active cells: {active_cells.shape[0]})", flush=True)

        # 1) pymcubes (CPU dense)
        pymc_times = []
        pymc_counts = []
        dt_sdf_np, sdf_np = time_op(lambda: sdf.detach().cpu().numpy())
        print(f"Converted SDF to numpy in {dt_sdf_np:.3f}s", flush=True)
        for i in range(repeats):
            def _mc_pymcubes():
                v, f = mcubes.marching_cubes(sdf_np, 0)
                return v, f
            dt, out = time_op(_mc_pymcubes)
            v, f = out
            pymc_times.append(dt)
            pymc_counts.append((int(v.shape[0]), int(f.shape[0])))
            print(f"pymcubes run {i+1}/{repeats}: {dt:.3f}s (V={v.shape[0]}, F={f.shape[0]})", flush=True)
        results["mc"][str(res)]["pymcubes"] = {
            "times": pymc_times,
            "mean": float(np.mean(pymc_times)),
            "std": float(np.std(pymc_times)),
            "last_counts": {
                "vertices": pymc_counts[-1][0],
                "triangles": pymc_counts[-1][1],
            },
        }

        # 2) spcumc (CUDA sparse)
        spcu_times = []
        spcu_counts = []
        try:
            from cubvh import sparse_marching_cubes as spmc_cuda
            for i in range(repeats):
                def _mc_spcu():
                    v, f = spmc_cuda(active_cells, active_cells_sdf, 0)
                    return v, f
                dt, out = time_op(_mc_spcu)
                v, f = out
                spcu_times.append(dt)
                spcu_counts.append((int(v.shape[0]), int(f.shape[0])))
                print(f"spcumc run {i+1}/{repeats}: {dt:.3f}s (V={v.shape[0]}, F={f.shape[0]})", flush=True)
            results["mc"][str(res)]["spcumc"] = {
                "times": spcu_times,
                "mean": float(np.mean(spcu_times)),
                "std": float(np.std(spcu_times)),
                "last_counts": {
                    "vertices": spcu_counts[-1][0] if spcu_counts else 0,
                    "triangles": spcu_counts[-1][1] if spcu_counts else 0,
                },
            }
        except Exception as e:
            print(f"spcumc error: {e}", flush=True)
            results["mc"][str(res)]["spcumc"] = {"error": str(e)}

        # 3) spmc (CPU sparse)
        spmc_times = []
        spmc_counts = []
        try:
            from cubvh import sparse_marching_cubes_cpu as spmc_cpu
            dt_ac_np, active_cells_np = time_op(lambda: active_cells.detach().cpu().numpy())
            dt_ac_sdf_np, active_cells_sdf_np = time_op(lambda: active_cells_sdf.detach().cpu().numpy())
            print(f"Converted sparse inputs to numpy in {dt_ac_np + dt_ac_sdf_np:.3f}s", flush=True)
            for i in range(repeats):
                def _mc_spmc():
                    v, f = spmc_cpu(active_cells_np, active_cells_sdf_np, 0)
                    return v, f
                dt, out = time_op(_mc_spmc)
                v, f = out
                spmc_times.append(dt)
                spmc_counts.append((int(v.shape[0]), int(f.shape[0])))
                print(f"spmc (CPU) run {i+1}/{repeats}: {dt:.3f}s (V={v.shape[0]}, F={f.shape[0]})", flush=True)
            results["mc"][str(res)]["spmc"] = {
                "times": spmc_times,
                "mean": float(np.mean(spmc_times)),
                "std": float(np.std(spmc_times)),
                "last_counts": {
                    "vertices": spmc_counts[-1][0] if spmc_counts else 0,
                    "triangles": spmc_counts[-1][1] if spmc_counts else 0,
                },
            }
        except Exception as e:
            print(f"spmc (CPU) error: {e}", flush=True)
            results["mc"][str(res)]["spmc"] = {"error": str(e)}

        # 4) disomc (CUDA dense via diso)
        disomc_times = []
        disomc_counts = []
        try:
            import diso
            print("Initializing diso DiffMC...", flush=True)
            dt_init_mc, diffmc = time_op(lambda: diso.DiffMC(dtype=torch.float32))
            if device.type == 'cuda':
                dt_to_cuda, diffmc = time_op(lambda: diffmc.cuda())
                print(f"DiffMC initialized in {dt_init_mc:.3f}s and moved to CUDA in {dt_to_cuda:.3f}s", flush=True)
            else:
                print(f"DiffMC initialized in {dt_init_mc:.3f}s", flush=True)
            for i in range(repeats):
                def _mc_disomc():
                    v, f = diffmc(sdf, normalize=True)
                    return v, f
                dt, out = time_op(_mc_disomc)
                v, f = out
                disomc_times.append(dt)
                disomc_counts.append((int(v.shape[0]), int(f.shape[0])))
                print(f"disomc run {i+1}/{repeats}: {dt:.3f}s (V={v.shape[0]}, F={f.shape[0]})", flush=True)
            results["mc"][str(res)]["disomc"] = {
                "times": disomc_times,
                "mean": float(np.mean(disomc_times)),
                "std": float(np.std(disomc_times)),
                "last_counts": {
                    "vertices": disomc_counts[-1][0] if disomc_counts else 0,
                    "triangles": disomc_counts[-1][1] if disomc_counts else 0,
                },
            }
        except Exception as e:
            print(f"disomc error: {e}", flush=True)
            results["mc"][str(res)]["disomc"] = {"error": str(e)}

        # 5) disodmc (CUDA dense via diso, dual)
        disodmc_times = []
        disodmc_counts = []
        try:
            import diso
            print("Initializing diso DiffDMC...", flush=True)
            dt_init_dmc, diffdmc = time_op(lambda: diso.DiffDMC(dtype=torch.float32))
            if device.type == 'cuda':
                dt_to_cuda_dmc, diffdmc = time_op(lambda: diffdmc.cuda())
                print(f"DiffDMC initialized in {dt_init_dmc:.3f}s and moved to CUDA in {dt_to_cuda_dmc:.3f}s", flush=True)
            else:
                print(f"DiffDMC initialized in {dt_init_dmc:.3f}s", flush=True)
            for i in range(repeats):
                def _mc_disodmc():
                    v, f = diffdmc(sdf, normalize=True, return_quads=False)
                    return v, f
                dt, out = time_op(_mc_disodmc)
                v, f = out
                disodmc_times.append(dt)
                disodmc_counts.append((int(v.shape[0]), int(f.shape[0])))
                print(f"disodmc run {i+1}/{repeats}: {dt:.3f}s (V={v.shape[0]}, F={f.shape[0]})", flush=True)
            results["mc"][str(res)]["disodmc"] = {
                "times": disodmc_times,
                "mean": float(np.mean(disodmc_times)),
                "std": float(np.std(disodmc_times)),
                "last_counts": {
                    "vertices": disodmc_counts[-1][0] if disodmc_counts else 0,
                    "triangles": disodmc_counts[-1][1] if disodmc_counts else 0,
                },
            }
        except Exception as e:
            print(f"disodmc error: {e}", flush=True)
            results["mc"][str(res)]["disodmc"] = {"error": str(e)}

    # Persist results
    if output_json is not None:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

    # Pretty print summary
    def fmt(mean, std):
        return f"{mean:.4f} +/- {std:.4f} s"

    print("\nFlood fill benchmark (mean +/- std):")
    for res in res_list:
        r = str(res)
        if r in results["ff"]:
            row = results["ff"][r]
            cubvh_row = row.get("cubvh", {})
            sk_row = row.get("skimage", {})
            print(f"  res={res}: cubvh={fmt(cubvh_row.get('mean', float('nan')), cubvh_row.get('std', float('nan')))}; skimage={fmt(sk_row.get('mean', float('nan')), sk_row.get('std', float('nan')))}")

    print("\nMarching cubes benchmark (mean +/- std):")
    for res in res_list:
        r = str(res)
        if r in results["mc"]:
            row = results["mc"][r]
            def f(k):
                v = row.get(k, {})
                return fmt(v.get('mean', float('nan')), v.get('std', float('nan'))) if 'mean' in v else f"error: {v.get('error', 'unknown')}"
            print(f"  res={res}: pymcubes={f('pymcubes')}; spcumc={f('spcumc')}; spmc={f('spmc')}; disomc={f('disomc')}; disodmc={f('disodmc')}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark flood fill and marching cubes variants on a mesh")
    parser.add_argument("mesh", type=str, nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "data", "seahourse3.glb"))
    parser.add_argument("--res", type=int, nargs="*", default=[512, 1024], help="Resolutions to benchmark, e.g., --res 512 1024")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per variant")
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "..", "output", "bench_watertight.json"))
    args = parser.parse_args()

    mesh_path = os.path.abspath(args.mesh)
    res_list = list(dict.fromkeys(args.res))
    output_json = os.path.abspath(args.output) if args.output else None

    print(f"Benchmarking on: {mesh_path}")
    print(f"Resolutions: {res_list}, repeats: {args.repeats}")
    run_benchmark(mesh_path, res_list, args.repeats, output_json)


if __name__ == "__main__":
    main()


