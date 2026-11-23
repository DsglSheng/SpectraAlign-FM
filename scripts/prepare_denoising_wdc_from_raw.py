from __future__ import annotations

import argparse
import random
from pathlib import Path

import lmdb
import numpy as np
import torch
from scipy.io import savemat
from scipy.ndimage import zoom
from skimage import io


class BandMinMaxQuantileStateful:
    def __init__(self, low: float = 0.02, up: float = 0.98, epsilon: float = 0.001) -> None:
        self.low = low
        self.up = up
        self.epsilon = epsilon
        self.q: torch.Tensor | None = None

    def fit(self, images: list[torch.Tensor]) -> None:
        flattened = torch.cat([image.flatten(start_dim=1) for image in images], dim=1)
        bands = int(flattened.shape[0])
        quantiles = np.zeros((bands, 2), dtype=np.float32)
        for band in range(bands):
            quantiles[band] = np.percentile(flattened[band].cpu().numpy(), q=100 * np.array([self.low, self.up]))
        self.q = torch.tensor(quantiles, dtype=torch.float32).T[..., None, None]

    def transform(self, image: torch.Tensor) -> torch.Tensor:
        if self.q is None:
            raise RuntimeError("Normalizer must be fitted before transform().")
        image = torch.minimum(image, self.q[1])
        image = torch.maximum(image, self.q[0])
        return (image - self.q[0]) / (self.epsilon + (self.q[1] - self.q[0]))


def data_augmentation(image: np.ndarray, mode: int | None = None) -> np.ndarray:
    axes = (-2, -1)
    if mode is None:
        mode = random.randint(0, 7)
    if mode == 1:
        image = image[:, ::-1, :]
    elif mode == 2:
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        image = np.rot90(image, axes=axes)[:, ::-1, :]
    elif mode == 4:
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        image = np.rot90(image, k=2, axes=axes)[:, ::-1, :]
    elif mode == 6:
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        image = np.rot90(image, k=3, axes=axes)[:, ::-1, :]
    if random.random() < 0.5:
        image = image[::-1, :, :]
    return np.ascontiguousarray(image)


def ensure_chw(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected 3D WDC TIFF, got shape {image.shape}.")
    if image.shape[0] == 191:
        return image
    if image.shape[-1] == 191:
        return np.transpose(image, (2, 0, 1))
    raise ValueError(f"Cannot infer spectral dimension for WDC TIFF shape {image.shape}.")


def extract_patches(data: np.ndarray, stride_hw: tuple[int, int]) -> list[np.ndarray]:
    channels, height, width = data.shape
    if channels != 191:
        raise ValueError(f"Expected 191 WDC bands, got {channels}.")
    stride_h, stride_w = stride_hw
    patches = []
    for top in range(0, height - 64 + 1, stride_h):
        for left in range(0, width - 64 + 1, stride_w):
            patches.append(data[:, top : top + 64, left : left + 64])
    return patches


def build_training_patches(train_images: list[np.ndarray], seed: int) -> list[np.ndarray]:
    np.random.seed(seed)
    random.seed(seed)
    patches: list[np.ndarray] = []
    for data in train_images:
        for scale, stride in zip((1.0, 0.5, 0.25), ((16, 16), (8, 8), (8, 8))):
            scaled = data if scale == 1.0 else zoom(data, zoom=(1, scale, scale))
            for patch in extract_patches(scaled, stride):
                patches.append(data_augmentation(patch.astype(np.float32, copy=False)))
    return patches


def write_lmdb(patches: list[np.ndarray], output_db: Path) -> None:
    if output_db.exists():
        raise FileExistsError(f"Refusing to overwrite existing LMDB: {output_db}")
    output_db.mkdir(parents=True)
    map_size = int(sum(patch.nbytes for patch in patches) * 1.4)
    env = lmdb.open(str(output_db), map_size=map_size, writemap=True)
    meta_path = output_db / "meta_info.txt"
    with meta_path.open("w", encoding="utf-8") as meta, env.begin(write=True) as txn:
        for index, patch in enumerate(patches):
            patch = np.ascontiguousarray(patch.astype(np.float32, copy=False))
            key = f"{index:08}"
            channels, height, width = patch.shape
            meta.write(f"{key} ({height},{width},{channels})\n")
            txn.put(key.encode("ascii"), patch.tobytes())
    env.sync()
    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild HyperSIGMA WDC denoising LMDB from raw dc.tif.")
    parser.add_argument("--raw-tif", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2017)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    raw = ensure_chw(io.imread(args.raw_tif)).astype(np.float32, copy=False)
    print(f"loaded raw WDC {raw.shape} from {args.raw_tif}", flush=True)
    tensor = torch.tensor(raw, dtype=torch.float32)
    test = tensor[:, 600:800, 50:250].clone()
    train_0 = tensor[:, :600, :].clone()
    train_1 = tensor[:, 800:, :].clone()
    val = tensor[:, 600:656, 251:].clone()

    normalizer = BandMinMaxQuantileStateful()
    normalizer.fit([train_0, train_1])
    train_0_np = normalizer.transform(train_0).cpu().numpy()
    train_1_np = normalizer.transform(train_1).cpu().numpy()

    normalizer.fit([test])
    test_np = normalizer.transform(test).cpu().numpy()
    normalizer.fit([val])
    val_np = normalizer.transform(val).cpu().numpy()

    split_root = args.output_root / "splits"
    (split_root / "train").mkdir(parents=True, exist_ok=True)
    (split_root / "test").mkdir(parents=True, exist_ok=True)
    (split_root / "val").mkdir(parents=True, exist_ok=True)
    savemat(split_root / "train" / "train_0.mat", {"data": train_0_np})
    savemat(split_root / "train" / "train_1.mat", {"data": train_1_np})
    savemat(split_root / "test" / "test.mat", {"data": test_np})
    savemat(split_root / "val" / "val.mat", {"data": val_np})

    patches = build_training_patches([train_0_np, train_1_np], seed=args.seed)
    print(f"writing {len(patches)} patches to LMDB", flush=True)
    write_lmdb(patches, args.output_root / "Training" / "wdc.db")
    print("done", flush=True)


if __name__ == "__main__":
    main()
