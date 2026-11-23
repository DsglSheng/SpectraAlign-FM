from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from skimage.transform import resize


def load_houston18(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "ori_data" not in handle:
            raise KeyError(f"{path} does not contain variable 'ori_data'.")
        raw = np.asarray(handle["ori_data"], dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError(f"Expected 3-D cube, got shape {raw.shape}.")
    # MATLAB v7.3 stores this public Houston18 as (C, W, H).  We transpose to
    # H, W, C, then rotate so HyperSIGMA's 256-row test split has training area.
    cube_hwc = raw.transpose(2, 1, 0)
    return cube_hwc.transpose(1, 0, 2)


def pad_to_minimum(cube: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    h, w, _ = cube.shape
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return cube
    return np.pad(cube, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")


def imresize_cube(cube: np.ndarray, factor: float) -> np.ndarray:
    h, w, c = cube.shape
    out_h = max(1, int(round(h * factor)))
    out_w = max(1, int(round(w * factor)))
    return resize(
        cube,
        (out_h, out_w, c),
        order=3,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)


def write_test_file(cube: np.ndarray, out_dir: Path, dataset_name: str, scale: int, test_size: int) -> int:
    factor = 1.0 / float(scale)
    test_pic_num = cube.shape[1] // test_size
    if test_pic_num <= 0:
        raise ValueError(f"Cube width {cube.shape[1]} is too small for test_size={test_size}.")
    gt = np.zeros((test_pic_num, test_size, test_size, cube.shape[2]), dtype=np.float32)
    ms = np.zeros((test_pic_num, test_size // scale, test_size // scale, cube.shape[2]), dtype=np.float32)
    ms_bicubic = np.zeros_like(gt)
    for i in range(test_pic_num):
        left = i * test_size
        test = cube[:test_size, left : left + test_size, :]
        low = imresize_cube(test, factor)
        bicubic = imresize_cube(low, float(scale))
        gt[i] = test
        ms[i] = low
        ms_bicubic[i] = bicubic[:test_size, :test_size, :]
    sio.savemat(out_dir / f"{dataset_name}_test.mat", {"gt": gt, "ms": ms, "ms_bicubic": ms_bicubic}, do_compression=True)
    return test_pic_num


def write_train_patches(
    train_cube: np.ndarray,
    out_dir: Path,
    scale: int,
    patch_size: int,
    stride: int,
) -> int:
    factor = 1.0 / float(scale)
    train_dir = out_dir / "trains"
    train_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    max_row = train_cube.shape[0] - patch_size
    max_col = train_cube.shape[1] - patch_size
    if max_row < 0 or max_col < 0:
        raise ValueError(f"Training cube {train_cube.shape} is smaller than patch_size={patch_size}.")
    for top in range(0, max_row + 1, stride):
        for left in range(0, max_col + 1, stride):
            gt = train_cube[top : top + patch_size, left : left + patch_size, :].astype(np.float32)
            ms = imresize_cube(gt, factor)
            ms_bicubic = imresize_cube(ms, float(scale))[:patch_size, :patch_size, :]
            sio.savemat(
                train_dir / f"block_public_houston18_{index:05d}.mat",
                {"gt": gt, "ms": ms, "ms_bicubic": ms_bicubic},
                do_compression=True,
            )
            index += 1
    return index - 1


def prepare_scale(args: argparse.Namespace, cube: np.ndarray, scale: int) -> dict[str, object]:
    out_dir = args.output_root / f"{args.dataset_name}_x{scale}"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_count = write_test_file(cube, out_dir, args.dataset_name, scale, args.test_size)
    train_cube = cube[args.test_size :, :, :]
    patch_count = write_train_patches(train_cube, out_dir, scale, args.patch_size, args.stride)
    manifest = {
        "source": "YuxiangZhang-BIT/Data-CSHSI public Houston18 Google Drive",
        "source_file": str(args.input),
        "source_variable": "ori_data",
        "original_h5_shape_cwh": list(args.original_h5_shape),
        "prepared_cube_shape_hwc": list(cube.shape),
        "dataset_name": args.dataset_name,
        "scale": scale,
        "test_size": args.test_size,
        "test_tiles": test_count,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "train_patches": patch_count,
        "note": "Public Houston18 classification cube converted to HyperSIGMA SR file structure; not the full HyperSIGMA Houston SR raw scene.",
    }
    (out_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, default="houston")
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--scales", type=int, nargs="+", default=[4, 8])
    args = parser.parse_args()

    with h5py.File(args.input, "r") as handle:
        args.original_h5_shape = tuple(int(v) for v in handle["ori_data"].shape)
    cube = load_houston18(args.input)
    cube = cube.astype(np.float32)
    cube /= max(float(np.nanmax(cube)), 1.0e-12)
    cube = pad_to_minimum(cube, args.test_size + args.patch_size, args.test_size)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = [prepare_scale(args, cube, scale) for scale in args.scales]
    (args.output_root / "prepare_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
