import argparse
import logging
import math
import time
from typing import Callable, Dict, Tuple, List, Optional

import numpy as np
import torch
from scipy import ndimage

import cubvh


def _now() -> float:
    return time.perf_counter()


def generate_random(shape: Tuple[int, int, int], p: float, rng: np.random.Generator, **_: dict) -> np.ndarray:
    vol = rng.random(shape, dtype=np.float32) < p
    return vol.astype(np.bool_)


def generate_sphere(shape: Tuple[int, int, int], radius_frac: float = 0.25, **_: dict) -> np.ndarray:
    z, y, x = [np.arange(s, dtype=np.float32) for s in shape]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    cz, cy, cx = [(s - 1) / 2.0 for s in shape]
    r = radius_frac * min(shape)
    dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return (dist2 <= r * r).astype(np.bool_)


def generate_hollow_sphere(shape: Tuple[int, int, int], radius_frac: float = 0.25, thickness: int = 1, **_: dict) -> np.ndarray:
    z, y, x = [np.arange(s, dtype=np.float32) for s in shape]
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    cz, cy, cx = [(s - 1) / 2.0 for s in shape]
    r = radius_frac * min(shape)
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    return ((dist <= r) & (dist >= r - float(max(1, thickness)))).astype(np.bool_)


def generate_checkerboard(shape: Tuple[int, int, int], period: int = 2, **_: dict) -> np.ndarray:
    z, y, x = np.indices(shape, dtype=np.int32)
    return (((z + y + x) // period) % 2 == 0)


def generate_plane(shape: Tuple[int, int, int], axis: str = "z", index: int | None = None, thickness: int = 1, **_: dict) -> np.ndarray:
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be one of x,y,z")
    sx, sy, sz = shape[2], shape[1], shape[0]
    if index is None:
        index = {"x": sx // 2, "y": sy // 2, "z": sz // 2}[axis]
    vol = np.zeros(shape, dtype=np.bool_)
    if axis == "z":
        z0 = max(0, index - thickness // 2)
        z1 = min(shape[0], z0 + max(1, thickness))
        vol[z0:z1, :, :] = True
    elif axis == "y":
        y0 = max(0, index - thickness // 2)
        y1 = min(shape[1], y0 + max(1, thickness))
        vol[:, y0:y1, :] = True
    else:  # x
        x0 = max(0, index - thickness // 2)
        x1 = min(shape[2], x0 + max(1, thickness))
        vol[:, :, x0:x1] = True
    return vol


def generate_cross(shape: Tuple[int, int, int], thickness: int = 1, **_: dict) -> np.ndarray:
    cz, cy, cx = [s // 2 for s in shape]
    vol = np.zeros(shape, dtype=np.bool_)
    # three orthogonal lines crossing the center
    vol[:, cy - thickness // 2 : cy - thickness // 2 + max(1, thickness), cx] = True
    vol[cz, :, cx - thickness // 2 : cx - thickness // 2 + max(1, thickness)] = True
    vol[cz - thickness // 2 : cz - thickness // 2 + max(1, thickness), cy, :] = True
    return vol


def generate_small_cube(shape: Tuple[int, int, int], side_frac: float = 0.25, **_: dict) -> np.ndarray:
    side = max(1, int(side_frac * min(shape)))
    start = [(s - side) // 2 for s in shape]
    vol = np.zeros(shape, dtype=np.bool_)
    vol[start[0] : start[0] + side, start[1] : start[1] + side, start[2] : start[2] + side] = True
    return vol


PATTERN_FUNCS: Dict[str, Callable[..., np.ndarray]] = {
    "random": generate_random,
    "sphere": generate_sphere,
    "hollow_sphere": generate_hollow_sphere,
    "checkerboard": generate_checkerboard,
    "plane": generate_plane,
    "cross": generate_cross,
    "small_cube": generate_small_cube,
}


def run_single_test(
    pattern: str,
    shape: Tuple[int, int, int],
    args: argparse.Namespace,
    logger: logging.Logger,
    overrides: Optional[dict] = None,
) -> tuple[bool, int, int]:
    rng = np.random.default_rng(args.seed)
    gen = PATTERN_FUNCS[pattern]

    logger.info(f"Pattern={pattern} | shape={shape}")
    t0 = _now()
    params = dict(
        p=args.prob,
        rng=rng,
        radius_frac=args.radius_frac,
        thickness=args.thickness,
        period=args.period,
        axis=args.axis,
        side_frac=args.side_frac,
    )
    if overrides:
        params.update(overrides)
    vol = gen(shape, **params)
    t1 = _now()

    n_ones = int(vol.sum())
    logger.info(f"Generated volume in {t1 - t0:.3f}s, ones={n_ones} ({n_ones/np.prod(shape):.6f} density)")

    if n_ones == 0:
        logger.warning("No occupied voxels; skipping erosion; trivially passing.")
        return True, 0, 0

    if n_ones > args.max_voxels:
        logger.error(
            f"Too many occupied voxels for sparse test: {n_ones} > max_voxels={args.max_voxels}. "
            "Reduce --res or --prob or choose a sparser pattern."
        )
        return False, 0, n_ones

    # Extract coords [N,3]
    coords_np = np.argwhere(vol).astype(np.int32)
    N = coords_np.shape[0]
    logger.info(f"Sparse coords: N={N}")

    # SciPy ground truth (6-neighbour erosion)
    struct = ndimage.generate_binary_structure(rank=3, connectivity=1)
    t2 = _now()
    eroded = ndimage.binary_erosion(vol, structure=struct, iterations=1, border_value=0)
    t3 = _now()
    eroded_ones = int(eroded.sum())
    logger.info(f"SciPy erosion: {t3 - t2:.3f}s, kept={eroded_ones}, ratio={eroded_ones/max(1, n_ones):.4f}")

    # CUDA sparse erosion
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but cubvh.sparse_erode requires CUDA.")

    coords_t = torch.from_numpy(coords_np).int().cuda()
    torch.cuda.synchronize()
    t4 = _now()
    mask_t = cubvh.sparse_erode(coords_t)  # bool [N] on CUDA
    torch.cuda.synchronize()
    t5 = _now()
    kept = int(mask_t.sum().item())
    logger.info(f"CUDA sparse_erode: {t5 - t4:.3f}s, kept={kept}, ratio={kept/max(1, n_ones):.4f}")

    # Compare results at occupied coords
    expected_mask_np = eroded[coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]]
    mismatches = int((mask_t.cpu().numpy() != expected_mask_np).sum())
    acc = 1.0 - mismatches / max(1, N)
    logger.info(f"Accuracy: {acc*100:.4f}%  mismatches={mismatches}/{N}")

    ok = mismatches == 0
    if not ok and args.fail_fast:
        raise AssertionError(f"Mismatch found for pattern={pattern}: {mismatches} of {N}")
    return ok, mismatches, N


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test cubvh.sparse_erode against SciPy for various occupancy patterns.")
    # Preset suite controls
    parser.add_argument("--suite", type=str, default="quick", choices=["quick", "full"], help="Preset suite to run. Default: quick")
    parser.add_argument("--filter", type=str, default="", help="Substring to filter preset case names/patterns.")
    parser.add_argument("--list-cases", action="store_true", help="List cases of the selected suite and exit.")
    # Custom mode controls (used only if --patterns provided)
    parser.add_argument("--res", "-r", type=int, default=64, help="(Custom mode) Grid resolution for each axis.")
    parser.add_argument("--prob", "-p", type=float, default=0.1, help="(Custom) Occupancy probability for random pattern.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for random pattern).")
    parser.add_argument(
        "--patterns",
        type=str,
        default="",
        help=f"(Custom mode) Comma-separated list of patterns to test. Available: {','.join(PATTERN_FUNCS.keys())}. Use 'all' for all.",
    )
    parser.add_argument("--radius-frac", dest="radius_frac", type=float, default=0.25, help="(Custom) Radius as fraction of min(res) for spheres.")
    parser.add_argument("--thickness", type=int, default=1, help="(Custom) Thickness for shells/planes/cross (in voxels).")
    parser.add_argument("--period", type=int, default=2, help="(Custom) Period for checkerboard.")
    parser.add_argument("--axis", type=str, default="z", choices=["x", "y", "z"], help="(Custom) Axis for plane pattern.")
    parser.add_argument("--side-frac", dest="side_frac", type=float, default=0.25, help="(Custom) Side fraction for small_cube.")
    parser.add_argument("--max-voxels", type=int, default=5_000_000, help="Safety cap on number of occupied voxels.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failing pattern.")
    return parser.parse_args()


def build_presets(suite: str) -> List[dict]:
    """Return a list of preset cases. Each case is a dict with keys:
    name, res, pattern, and optional overrides for generator params.
    """
    cases: List[dict] = []

    def C(name: str, res: int, pattern: str, **overrides):
        cases.append({"name": name, "res": res, "pattern": pattern, "overrides": overrides})

    # Quick suite: fast coverage across patterns
    C("rand_r16_p005", 16, "random", p=0.05, seed=0)
    C("rand_r32_p02", 32, "random", p=0.2, seed=1)
    C("checker_r32_per2", 32, "checkerboard", period=2)
    C("checker_r24_per3", 24, "checkerboard", period=3)
    C("sphere_r32_rf30", 32, "sphere", radius_frac=0.30)
    C("hsphere_r32_rf35_t2", 32, "hollow_sphere", radius_frac=0.35, thickness=2)
    C("planeZ_r24_t1", 24, "plane", axis="z", thickness=1)
    C("planeX_r24_t2", 24, "plane", axis="x", thickness=2)
    C("cross_r24_t1", 24, "cross", thickness=1)
    C("smallcube_r24_s30", 24, "small_cube", side_frac=0.30)

    if suite == "full":
        # Add denser randoms and larger shapes within safety bound
        C("rand_r40_p01", 40, "random", p=0.10, seed=2)
        C("rand_r48_p005", 48, "random", p=0.05, seed=3)
        C("sphere_r40_rf35", 40, "sphere", radius_frac=0.35)
        C("hsphere_r48_rf30_t1", 48, "hollow_sphere", radius_frac=0.30, thickness=1)
        C("planeY_r32_t3", 32, "plane", axis="y", thickness=3)
        C("cross_r32_t3", 32, "cross", thickness=3)
        C("smallcube_r32_s40", 32, "small_cube", side_frac=0.40)

    return cases


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("sparse_erode_test")

    # Determine mode: preset suite (default) or custom
    custom_mode = bool(args.patterns.strip())

    if not custom_mode:
        cases = build_presets(args.suite)
        if args.filter:
            filt = args.filter.lower()
            cases = [c for c in cases if (filt in c["name"].lower() or filt in c["pattern"].lower())]
        if args.list_cases:
            print("Preset cases (suite=%s):" % args.suite)
            for i, c in enumerate(cases):
                print(f"  [{i:02d}] {c['name']}: res={c['res']} pattern={c['pattern']} overrides={c['overrides']}")
            return

        logger.info(
            f"Running preset suite='{args.suite}' with {len(cases)} case(s) | seed={args.seed} | max_voxels={args.max_voxels}"
        )
    else:
        # Custom mode
        patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
        if len(patterns) == 1 and patterns[0].lower() == "all":
            patterns = list(PATTERN_FUNCS.keys())

        unknown = [p for p in patterns if p not in PATTERN_FUNCS]
        if unknown:
            raise ValueError(f"Unknown patterns: {unknown}. Available: {list(PATTERN_FUNCS.keys())}")

        logger.info(
            f"Running custom tests | res={args.res} | patterns={patterns} | seed={args.seed} | max_voxels={args.max_voxels}"
        )

    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(torch.cuda.current_device())
        logger.info(f"CUDA device: {dev}")
    else:
        logger.warning("CUDA is not available; tests will fail to run.")

    total = 0
    failed = 0
    mismatches_total = 0
    t_start = _now()

    if not custom_mode:
        for case in cases:
            total += 1
            res = case["res"]
            shape = (res, res, res)
            pat = case["pattern"]
            logger.info(f"\n=== Case {case['name']} ===")
            try:
                ok, mismatches, n = run_single_test(pat, shape, args, logger, overrides=case.get("overrides", {}))
            except Exception as e:
                logger.exception(f"Error while testing case={case['name']} pattern={pat}: {e}")
                failed += 1
                if args.fail_fast:
                    break
                continue
            mismatches_total += mismatches
            if not ok:
                failed += 1
                if args.fail_fast:
                    break
    else:
        shape = (args.res, args.res, args.res)
        for pat in patterns:
            total += 1
            try:
                ok, mismatches, n = run_single_test(pat, shape, args, logger)
            except Exception as e:
                logger.exception(f"Error while testing pattern={pat}: {e}")
                failed += 1
                if args.fail_fast:
                    break
                continue
            mismatches_total += mismatches
            if not ok:
                failed += 1
                if args.fail_fast:
                    break

    t_end = _now()
    logger.info(
        f"Summary: ran={total}, failed={failed}, total_mismatches={mismatches_total}, elapsed={t_end - t_start:.3f}s"
    )

    assert failed == 0, f"{failed} pattern(s) failed. See logs above."
    logger.info("[OK] All patterns match SciPy ground truth.")


if __name__ == "__main__":
    main()
