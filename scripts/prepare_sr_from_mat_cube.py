from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio


def _choose_cube(payload: dict[str, object], key: str | None) -> np.ndarray:
    if key is not None:
        return np.asarray(payload[key])
    candidates: list[tuple[int, str, np.ndarray]] = []
    for name, value in payload.items():
        if name.startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.ndim == 3:
            score = int(np.prod(arr.shape))
            candidates.append((score, name, arr))
    if not candidates:
        raise ValueError("No 3D cube found in source .mat file.")
    _, name, arr = max(candidates, key=lambda item: item[0])
    print(f"selected_cube_key={name} shape={arr.shape}")
    return arr


def _to_hwc(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got {cube.shape}")
    if cube.shape[-1] <= 512 and cube.shape[0] > 16 and cube.shape[1] > 16:
        return cube.astype(np.float32, copy=False)
    if cube.shape[0] <= 512 and cube.shape[1] > 16 and cube.shape[2] > 16:
        return np.transpose(cube, (1, 2, 0)).astype(np.float32, copy=False)
    return cube.astype(np.float32, copy=False)


def _resize_hwc(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC).astype(np.float32, copy=False)


def _make_pair(gt: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w, _ = gt.shape
    lr = _resize_hwc(gt, (w // scale, h // scale))
    lms = _resize_hwc(lr, (w, h))
    return gt.astype(np.float32, copy=False), lr, lms


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HyperSIGMA-style SR mats from an available HSI cube.")
    parser.add_argument("--source-mat", type=Path, required=True)
    parser.add_argument("--source-key", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/Data/SpectralData/Aware_open/data/super_resolution"))
    parser.add_argument("--dataset-name", type=str, default="houston_proxy")
    parser.add_argument("--scale", type=int, choices=(4, 8), required=True)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--max-train-patches", type=int, default=2048)
    args = parser.parse_args()

    payload = sio.loadmat(args.source_mat)
    cube = _to_hwc(_choose_cube(payload, args.source_key))
    cube = cube.astype(np.float32)
    cube = cube - float(np.nanmin(cube))
    denom = float(np.nanmax(cube))
    if denom > 0:
        cube = cube / denom
    cube = np.nan_to_num(cube, nan=0.0, posinf=1.0, neginf=0.0)

    h, w, c = cube.shape
    if h < args.test_size or w < args.test_size:
        raise ValueError(f"Cube {cube.shape} is too small for test_size={args.test_size}")
    out_dir = args.output_root / f"{args.dataset_name}_x{args.scale}"
    train_dir = out_dir / "trains"
    train_dir.mkdir(parents=True, exist_ok=True)

    test_tiles = []
    for left in range(0, w - args.test_size + 1, args.test_size):
        test_tiles.append(cube[: args.test_size, left : left + args.test_size, :])
    if not test_tiles:
        test_tiles.append(cube[: args.test_size, : args.test_size, :])
    gt_list, ms_list, lms_list = [], [], []
    for tile in test_tiles:
        gt, ms, lms = _make_pair(tile, args.scale)
        gt_list.append(gt)
        ms_list.append(ms)
        lms_list.append(lms)
    sio.savemat(out_dir / f"{args.dataset_name}_test.mat", {"gt": np.stack(gt_list), "ms": np.stack(ms_list), "ms_bicubic": np.stack(lms_list)})

    train_region = cube[args.test_size :, :, :]
    if train_region.shape[0] < args.patch_size:
        train_region = cube[:, args.test_size :, :] if cube.shape[1] - args.test_size >= args.patch_size else cube
    count = 0
    for top in range(0, train_region.shape[0] - args.patch_size + 1, args.stride):
        for left in range(0, train_region.shape[1] - args.patch_size + 1, args.stride):
            patch = train_region[top : top + args.patch_size, left : left + args.patch_size, :]
            gt, ms, lms = _make_pair(patch, args.scale)
            count += 1
            sio.savemat(train_dir / f"block_{args.dataset_name}_{count:05d}.mat", {"gt": gt, "ms": ms, "ms_bicubic": lms}, do_compression=False)
            if count >= args.max_train_patches:
                break
        if count >= args.max_train_patches:
            break
    if count == 0:
        raise RuntimeError(f"No train patches created from source cube {cube.shape}.")

    meta = {
        "source_mat": str(args.source_mat),
        "source_key": args.source_key,
        "source_shape_hwc": list(cube.shape),
        "dataset_name": args.dataset_name,
        "scale": args.scale,
        "train_patches": count,
        "test_tiles": len(test_tiles),
        "note": "Proxy SR dataset generated from an available local cube; use official Houston2018 to reproduce HyperSIGMA Houston protocol exactly.",
    }
    (out_dir / "source_manifest.json").write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    main()
