from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SPECAWARE_EXAMPLE = REPO_ROOT / "external" / "SpecAware" / "example"
if SPECAWARE_EXAMPLE.exists() and str(SPECAWARE_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(SPECAWARE_EXAMPLE))

from aware_open.downstream import AwareFeatureExtractor, build_downstream_metadata  # noqa: E402
from UperNet.MultiLevelNeck import MultiLevelNeck  # noqa: E402
from UperNet.UperNet_decoder import UPerNetDecoder  # noqa: E402


QINGPU_CLASS_NAMES = (
    "Asphalt road",
    "Greenhouse",
    "Cement road",
    "Lotus",
    "Water bamboo 1",
    "Farmland",
    "Water bamboo 2",
    "Water",
    "Buildings",
    "Bulrush",
    "Oak",
    "Ligustrum",
    "Elaeocarpus sylvestris",
    "Camptotheca acuminata",
    "Soapberry",
    "Salix",
    "Goldenrain tree",
    "Cedar",
    "Camphor tree",
    "Zelkova schneideriana",
)

AERORIT_CLASS_COLORS = (
    (0, 255, 0),
    (0, 0, 255),
    (153, 0, 0),
    (255, 0, 0),
    (255, 127, 80),
    (0, 255, 255),
)

AERORIT_CLASS_NAMES = tuple(f"class_{i+1}_rgb_{r}_{g}_{b}" for i, (r, g, b) in enumerate(AERORIT_CLASS_COLORS))

WHU_H2SR_CLASS_NAMES = (
    "Paddy field",
    "Dry farmland",
    "Forest land",
    "Grassland",
    "Building",
    "Highway",
    "Greenhouse",
    "Water body",
)
WHU_H2SR_LABEL_CODE_TO_DENSE = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    15: 8,
}


ENVI_DTYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
}


def parse_envi_header(path: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip().lower()] = value.strip().strip("{}").strip()
    return info


def parse_envi_float_list(path: Path, key: str) -> list[float]:
    import re

    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(rf"(?is)\b{re.escape(key)}\s*=\s*\{{(.*?)\}}", text)
    if not match:
        return []
    return [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", match.group(1))]


class SpectralImageView:
    def __init__(
        self,
        data: np.memmap,
        *,
        layout: str,
        bands: int,
        lines: int,
        samples: int,
        band_indices: np.ndarray | None = None,
    ) -> None:
        self.data = data
        self.layout = layout
        self.source_bands = int(bands)
        self.band_indices = None if band_indices is None else np.asarray(band_indices, dtype=np.int64)
        visible_bands = int(self.band_indices.size) if self.band_indices is not None else self.source_bands
        self.shape = (visible_bands, int(lines), int(samples))

    def read_patch(self, top: int, left: int, size: int) -> np.ndarray:
        if self.layout == "bsq":
            band_sel = slice(None) if self.band_indices is None else self.band_indices
            return np.asarray(self.data[band_sel, top : top + size, left : left + size], dtype=np.float32)
        if self.layout == "bil":
            band_sel = slice(None) if self.band_indices is None else self.band_indices
            patch = np.asarray(self.data[top : top + size, band_sel, left : left + size], dtype=np.float32)
            return np.transpose(patch, (1, 0, 2))
        raise ValueError(f"Unsupported ENVI interleave {self.layout!r}.")

    def read_points(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        if self.layout == "bsq":
            values = np.asarray(self.data[:, rows, cols], dtype=np.float32)
            if self.band_indices is not None:
                values = values[self.band_indices]
            return values
        if self.layout == "bil":
            values = np.asarray(self.data[rows, :, cols], dtype=np.float32)
            if self.band_indices is not None:
                values = values[:, self.band_indices]
            return values.T
        raise ValueError(f"Unsupported ENVI interleave {self.layout!r}.")

    def with_band_indices(self, band_indices: np.ndarray) -> "SpectralImageView":
        return SpectralImageView(
            self.data,
            layout=self.layout,
            bands=self.source_bands,
            lines=self.shape[1],
            samples=self.shape[2],
            band_indices=band_indices,
        )


def open_envi_image(data_path: Path, header_path: Path) -> tuple[SpectralImageView, dict[str, Any]]:
    header = parse_envi_header(header_path)
    samples = int(header["samples"])
    lines = int(header["lines"])
    bands = int(header["bands"])
    dtype = ENVI_DTYPES[int(header["data type"])]
    interleave = header.get("interleave", "bsq").lower()
    if interleave == "bsq":
        array = np.memmap(data_path, mode="r", dtype=dtype, shape=(bands, lines, samples))
    elif interleave == "bil":
        array = np.memmap(data_path, mode="r", dtype=dtype, shape=(lines, bands, samples))
    else:
        raise ValueError(f"Only BSQ/BIL ENVI files are supported here, got {interleave!r}.")
    wavelengths = parse_envi_float_list(header_path, "wavelength")
    fwhm = parse_envi_float_list(header_path, "fwhm")
    return SpectralImageView(array, layout=interleave, bands=bands, lines=lines, samples=samples), {
        "samples": samples,
        "lines": lines,
        "bands": bands,
        "dtype": str(dtype),
        "interleave": interleave,
        "wavelengths": wavelengths,
        "fwhm": fwhm,
    }


def apply_aerorit_band_subset(
    image: SpectralImageView,
    image_info: dict[str, Any],
    mode: str,
) -> tuple[SpectralImageView, dict[str, Any], str]:
    if mode == "full":
        return image, image_info, ""
    if mode != "specaware_305":
        raise ValueError(f"Unsupported AeroRIT band subset mode {mode!r}.")
    wavelengths = list(image_info.get("wavelengths", []))
    if not wavelengths:
        raise ValueError("AeroRIT specaware_305 band subset requires ENVI wavelengths.")
    start = next((idx for idx, value in enumerate(wavelengths) if float(value) >= 400.0), None)
    if start is None:
        raise ValueError("Could not find AeroRIT wavelength >= 400 nm.")
    end = min(start + 305, len(wavelengths))
    if end - start < 305:
        raise ValueError(f"AeroRIT has only {end - start} bands available after 400 nm; expected 305.")
    band_indices = np.arange(start, end, dtype=np.int64)
    subset_info = dict(image_info)
    subset_info["source_bands"] = int(image.source_bands)
    subset_info["source_wavelengths"] = wavelengths
    subset_info["band_subset_mode"] = mode
    subset_info["band_indices"] = band_indices.astype(int).tolist()
    subset_info["bands"] = int(band_indices.size)
    subset_info["wavelengths"] = [float(wavelengths[int(i)]) for i in band_indices]
    fwhm = list(image_info.get("fwhm", []))
    if len(fwhm) >= end:
        subset_info["fwhm"] = [float(fwhm[int(i)]) for i in band_indices]
    note = (
        " AeroRIT band subset: first 305 bands from the first wavelength >= 400 nm, "
        f"covering {subset_info['wavelengths'][0]:.3f}-{subset_info['wavelengths'][-1]:.3f} nm."
    )
    return image.with_band_indices(band_indices), subset_info, note


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(
    labels: np.ndarray,
    *,
    class_count: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int,
    output_path: Path,
) -> dict[str, np.ndarray]:
    if output_path.exists():
        payload = np.load(output_path)
        return {name: payload[name] for name in ("train", "val", "test")}
    rng = np.random.default_rng(seed)
    flat = labels.reshape(-1)
    train: list[np.ndarray] = []
    val: list[np.ndarray] = []
    test: list[np.ndarray] = []
    for class_id in range(1, class_count + 1):
        idx = np.flatnonzero(flat == class_id)
        rng.shuffle(idx)
        n_train = min(train_per_class, idx.size)
        n_val = min(val_per_class, max(idx.size - n_train, 0))
        if test_per_class <= 0:
            n_test = max(idx.size - n_train - n_val, 0)
        else:
            n_test = min(test_per_class, max(idx.size - n_train - n_val, 0))
        train.append(idx[:n_train])
        val.append(idx[n_train : n_train + n_val])
        test.append(idx[n_train + n_val : n_train + n_val + n_test])
    result = {
        "train": np.concatenate(train).astype(np.int64),
        "val": np.concatenate(val).astype(np.int64),
        "test": np.concatenate(test).astype(np.int64),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    return result


def split_indices_spatial_blocks(
    labels: np.ndarray,
    *,
    class_count: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int,
    output_path: Path,
    block_size: int,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, np.ndarray]:
    if output_path.exists():
        payload = np.load(output_path)
        return {name: payload[name] for name in ("train", "val", "test")}
    rng = np.random.default_rng(seed)
    height, width = labels.shape
    block_size = max(int(block_size), 1)
    block_cols = int(math.ceil(width / block_size))
    flat_labels = labels.reshape(-1)
    labeled = np.flatnonzero(flat_labels > 0)
    rows = labeled // width
    cols = labeled % width
    block_ids = (rows // block_size) * block_cols + (cols // block_size)
    unique_blocks = np.unique(block_ids)
    rng.shuffle(unique_blocks)
    n_train = max(1, int(round(unique_blocks.size * train_fraction)))
    n_val = max(1, int(round(unique_blocks.size * val_fraction)))
    n_train = min(n_train, unique_blocks.size)
    n_val = min(n_val, max(unique_blocks.size - n_train, 0))
    train_blocks = set(int(v) for v in unique_blocks[:n_train])
    val_blocks = set(int(v) for v in unique_blocks[n_train : n_train + n_val])
    test_blocks = set(int(v) for v in unique_blocks[n_train + n_val :])
    split_block_sets = {"train": train_blocks, "val": val_blocks, "test": test_blocks}
    result: dict[str, np.ndarray] = {}
    for split_name, block_set in split_block_sets.items():
        split_parts: list[np.ndarray] = []
        for class_id in range(1, class_count + 1):
            mask = flat_labels[labeled] == class_id
            class_indices = labeled[mask]
            class_blocks = block_ids[mask]
            class_indices = class_indices[np.isin(class_blocks, list(block_set))]
            rng.shuffle(class_indices)
            if split_name == "train":
                limit = train_per_class
            elif split_name == "val":
                limit = val_per_class
            else:
                limit = test_per_class
            if limit > 0:
                class_indices = class_indices[: min(limit, class_indices.size)]
            split_parts.append(class_indices.astype(np.int64, copy=False))
        result[split_name] = np.concatenate(split_parts).astype(np.int64) if split_parts else np.empty((0,), dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train=result["train"],
        val=result["val"],
        test=result["test"],
        block_size=np.array([block_size], dtype=np.int64),
        train_blocks=np.array(sorted(train_blocks), dtype=np.int64),
        val_blocks=np.array(sorted(val_blocks), dtype=np.int64),
        test_blocks=np.array(sorted(test_blocks), dtype=np.int64),
    )
    return result


def split_indices_aerorit_patch_201070(
    labels: np.ndarray,
    *,
    seed: int,
    output_path: Path,
    patch_size: int = 128,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if output_path.exists():
        payload = np.load(output_path)
        indices = {name: payload[name] for name in ("train", "val", "test")}
        patches = {name: payload[f"{name}_patches"] for name in ("train", "val", "test")}
        return indices, patches
    rng = np.random.default_rng(seed)
    height, width = labels.shape
    patch_size = int(patch_size)
    rows = height // patch_size
    cols = width // patch_size
    if rows <= 0 or cols <= 0:
        raise ValueError(f"AeroRIT image {height}x{width} is smaller than patch size {patch_size}.")
    patches = np.array(
        [(r * patch_size, c * patch_size) for r in range(rows) for c in range(cols)],
        dtype=np.int64,
    )
    order = np.arange(patches.shape[0], dtype=np.int64)
    rng.shuffle(order)
    train_count = max(1, int(round(order.size * 0.20)))
    val_count = max(1, int(round(order.size * 0.10)))
    train_count = min(train_count, order.size)
    val_count = min(val_count, max(order.size - train_count, 0))
    split_patch_ids = {
        "train": order[:train_count],
        "val": order[train_count : train_count + val_count],
        "test": order[train_count + val_count :],
    }
    flat_indices: dict[str, np.ndarray] = {}
    patch_splits: dict[str, np.ndarray] = {}
    for name, ids in split_patch_ids.items():
        coords = patches[ids]
        patch_splits[name] = coords
        parts: list[np.ndarray] = []
        for top, left in coords:
            patch = labels[int(top) : int(top) + patch_size, int(left) : int(left) + patch_size]
            valid = np.flatnonzero(patch.reshape(-1) > 0)
            if valid.size:
                row = valid // patch_size + int(top)
                col = valid % patch_size + int(left)
                parts.append((row * width + col).astype(np.int64, copy=False))
        flat_indices[name] = np.concatenate(parts).astype(np.int64) if parts else np.empty((0,), dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train=flat_indices["train"],
        val=flat_indices["val"],
        test=flat_indices["test"],
        train_patches=patch_splits["train"],
        val_patches=patch_splits["val"],
        test_patches=patch_splits["test"],
        patch_size=np.array([patch_size], dtype=np.int64),
        grid_shape=np.array([rows, cols], dtype=np.int64),
        dropped_border=np.array([height - rows * patch_size, width - cols * patch_size], dtype=np.int64),
        fractions=np.array([0.20, 0.10, 0.70], dtype=np.float32),
    )
    return flat_indices, patch_splits


def split_indices_qingpu_train_areas(
    labels: np.ndarray,
    *,
    class_count: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int,
    output_path: Path,
    area_size: int,
    area_count: int,
) -> dict[str, np.ndarray]:
    if output_path.exists():
        payload = np.load(output_path)
        return {name: payload[name] for name in ("train", "val", "test")}
    rng = np.random.default_rng(seed)
    height, width = labels.shape
    area_size = max(int(area_size), 1)
    area_count = max(int(area_count), 1)
    row_count = int(math.ceil(height / area_size))
    col_count = int(math.ceil(width / area_size))
    flat_labels = labels.reshape(-1)
    block_counts: list[tuple[int, int, int, int, np.ndarray]] = []
    for br in range(row_count):
        top = br * area_size
        bottom = min(height, top + area_size)
        for bc in range(col_count):
            left = bc * area_size
            right = min(width, left + area_size)
            patch = labels[top:bottom, left:right]
            counts = np.bincount(patch[patch > 0].reshape(-1), minlength=class_count + 1)[1 : class_count + 1].astype(np.float64)
            if counts.sum() > 0:
                block_counts.append((br, bc, top, left, counts))
    if not block_counts:
        raise ValueError("No labeled Qingpu blocks were found for area split.")
    selected: list[int] = []
    selected_counts = np.zeros((class_count,), dtype=np.float64)
    remaining = set(range(len(block_counts)))
    for _ in range(min(area_count, len(block_counts))):
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            counts = block_counts[idx][4]
            # Prefer blocks that add pixels for still-underrepresented classes.
            need_weight = 1.0 / np.sqrt(selected_counts + 1.0)
            scarcity_bonus = (selected_counts < max(train_per_class, 1)).astype(np.float64) + 1.0
            score = float((np.log1p(counts) * need_weight * scarcity_bonus).sum())
            if score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        selected_counts += block_counts[best_idx][4]
        remaining.remove(best_idx)

    block_id_map = np.full(labels.shape, -1, dtype=np.int32)
    for idx, (_, _, top, left, _) in enumerate(block_counts):
        bottom = min(height, top + area_size)
        right = min(width, left + area_size)
        block_id_map[top:bottom, left:right] = idx
    labeled = np.flatnonzero(flat_labels > 0)
    selected_set = set(selected)
    block_for_labeled = block_id_map.reshape(-1)[labeled]
    in_train_area = np.isin(block_for_labeled, np.array(selected, dtype=np.int32))
    result_parts: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    supplements: dict[int, int] = {}
    for class_id in range(1, class_count + 1):
        class_all = labeled[flat_labels[labeled] == class_id]
        class_blocks = block_id_map.reshape(-1)[class_all]
        train_pool = class_all[np.isin(class_blocks, np.array(selected, dtype=np.int32))]
        rest_pool = class_all[~np.isin(class_blocks, np.array(selected, dtype=np.int32))]
        rng.shuffle(train_pool)
        rng.shuffle(rest_pool)
        train_take = train_pool[: min(train_per_class, train_pool.size)]
        if train_take.size < min(train_per_class, class_all.size):
            needed = min(train_per_class, class_all.size) - train_take.size
            supplement = rest_pool[:needed]
            train_take = np.concatenate([train_take, supplement])
            rest_pool = rest_pool[needed:]
            supplements[class_id] = int(supplement.size)
        val_take = rest_pool[: min(val_per_class, rest_pool.size)]
        remaining_pool = rest_pool[val_take.size :]
        if test_per_class <= 0:
            test_take = remaining_pool
        else:
            test_take = remaining_pool[: min(test_per_class, remaining_pool.size)]
        result_parts["train"].append(train_take.astype(np.int64, copy=False))
        result_parts["val"].append(val_take.astype(np.int64, copy=False))
        result_parts["test"].append(test_take.astype(np.int64, copy=False))
    result = {
        key: np.concatenate(parts).astype(np.int64) if parts else np.empty((0,), dtype=np.int64)
        for key, parts in result_parts.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        train=result["train"],
        val=result["val"],
        test=result["test"],
        area_size=np.array([area_size], dtype=np.int64),
        selected_blocks=np.array(selected, dtype=np.int64),
        selected_block_rc=np.array([[block_counts[i][0], block_counts[i][1]] for i in selected], dtype=np.int64),
        selected_class_counts=selected_counts.astype(np.int64),
        supplements=np.array(sorted(supplements.items()), dtype=np.int64) if supplements else np.empty((0, 2), dtype=np.int64),
    )
    return result


def make_sparse_label_map(labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    flat_labels = labels.reshape(-1)
    out = np.zeros_like(flat_labels, dtype=np.uint8)
    out[indices] = flat_labels[indices].astype(np.uint8, copy=False)
    return out.reshape(labels.shape)


def estimate_band_stats(
    image: SpectralImageView,
    labels: np.ndarray,
    *,
    sample_count: int,
    seed: int,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    bands = int(image.shape[0])
    if sample_count <= 0:
        mean = np.zeros((bands,), dtype=np.float32)
        std = np.full((bands,), 10000.0, dtype=np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, mean=mean, std=std, note=np.array(["fixed_zero_mean_scale_10000"]))
        return mean, std
    if output_path.exists():
        data = np.load(output_path)
        return data["mean"].astype(np.float32), data["std"].astype(np.float32)
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(labels > 0)
    if rows.size == 0:
        raise ValueError("Qingpu label map has no labeled pixels.")
    choice = rng.choice(rows.size, size=min(sample_count, rows.size), replace=False)
    rows = rows[choice]
    cols = cols[choice]
    total = np.zeros((bands,), dtype=np.float64)
    total_sq = np.zeros((bands,), dtype=np.float64)
    seen = 0
    chunk = 2048
    for start in range(0, rows.size, chunk):
        r = rows[start : start + chunk]
        c = cols[start : start + chunk]
        values = image.read_points(r, c)
        total += values.sum(axis=1)
        total_sq += np.square(values, dtype=np.float32).sum(axis=1)
        seen += int(values.shape[1])
    mean = total / max(seen, 1)
    var = np.maximum(total_sq / max(seen, 1) - mean * mean, 1.0)
    std = np.sqrt(var)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return mean.astype(np.float32), std.astype(np.float32)


class SpectralCropDataset(Dataset):
    def __init__(
        self,
        *,
        image: SpectralImageView,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        crop_size: int,
        train: bool,
        length: int | None = None,
        jitter: int = 0,
        seed: int = 0,
        center_only_labels: bool = False,
    ) -> None:
        self.image = image
        self.labels = labels
        self.indices = indices.astype(np.int64, copy=False)
        self.mean = mean.astype(np.float32, copy=False)
        self.std = np.maximum(std.astype(np.float32, copy=False), 1.0)
        self.crop_size = int(crop_size)
        self.train = bool(train)
        self.length = int(length) if length is not None else int(self.indices.size)
        self.jitter = int(jitter)
        self.rng = np.random.default_rng(seed)
        self.height = int(labels.shape[0])
        self.width = int(labels.shape[1])
        self.center_only_labels = bool(center_only_labels)

    def __len__(self) -> int:
        return self.length

    def _center_for_index(self, index: int) -> tuple[int, int]:
        if self.train:
            flat = int(self.indices[int(self.rng.integers(0, self.indices.size))])
        else:
            flat = int(self.indices[index % self.indices.size])
        row = flat // self.width
        col = flat % self.width
        if self.train and self.jitter > 0:
            row = int(np.clip(row + self.rng.integers(-self.jitter, self.jitter + 1), 0, self.height - 1))
            col = int(np.clip(col + self.rng.integers(-self.jitter, self.jitter + 1), 0, self.width - 1))
        return row, col

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row, col = self._center_for_index(index)
        half = self.crop_size // 2
        top = int(np.clip(row - half, 0, max(self.height - self.crop_size, 0)))
        left = int(np.clip(col - half, 0, max(self.width - self.crop_size, 0)))
        image = self.image.read_patch(top, left, self.crop_size)
        image = (image - self.mean[:, None, None]) / self.std[:, None, None]
        label = np.asarray(self.labels[top : top + self.crop_size, left : left + self.crop_size], dtype=np.int64)
        if self.center_only_labels:
            center_label = int(self.labels[row, col])
            label_center_only = np.zeros_like(label, dtype=np.int64)
            if center_label > 0:
                label_center_only[int(row - top), int(col - left)] = center_label
            label = label_center_only
        if self.train:
            if random.random() < 0.5:
                image = image[:, ::-1, :]
                label = label[::-1, :]
            if random.random() < 0.5:
                image = image[:, :, ::-1]
                label = label[:, ::-1]
            k = random.randint(0, 3)
            if k:
                image = np.rot90(image, k=k, axes=(1, 2))
                label = np.rot90(label, k=k, axes=(0, 1))
        return torch.from_numpy(image.copy()).float(), torch.from_numpy(label.copy()).long()


class SpectralPatchDataset(Dataset):
    def __init__(
        self,
        *,
        image: SpectralImageView,
        labels: np.ndarray,
        patch_coords: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        patch_size: int,
        train: bool,
        length: int | None = None,
        seed: int = 0,
    ) -> None:
        self.image = image
        self.labels = labels
        self.patch_coords = np.asarray(patch_coords, dtype=np.int64)
        self.mean = mean.astype(np.float32, copy=False)
        self.std = np.maximum(std.astype(np.float32, copy=False), 1.0)
        self.patch_size = int(patch_size)
        self.train = bool(train)
        self.length = int(length) if length is not None else int(self.patch_coords.shape[0])
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def _patch_coord(self, index: int) -> tuple[int, int]:
        if self.train:
            patch_index = int(self.rng.integers(0, self.patch_coords.shape[0]))
        else:
            patch_index = int(index % self.patch_coords.shape[0])
        top, left = self.patch_coords[patch_index]
        return int(top), int(left)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        top, left = self._patch_coord(index)
        image = self.image.read_patch(top, left, self.patch_size)
        image = (image - self.mean[:, None, None]) / self.std[:, None, None]
        label = np.asarray(self.labels[top : top + self.patch_size, left : left + self.patch_size], dtype=np.int64)
        if self.train:
            if random.random() < 0.5:
                image = image[:, ::-1, :]
                label = label[::-1, :]
            if random.random() < 0.5:
                image = image[:, :, ::-1]
                label = label[:, ::-1]
            k = random.randint(0, 3)
            if k:
                image = np.rot90(image, k=k, axes=(1, 2))
                label = np.rot90(label, k=k, axes=(0, 1))
        return torch.from_numpy(image.copy()).float(), torch.from_numpy(label.copy()).long()


def normalize_whu_labels(label: np.ndarray, class_count: int) -> np.ndarray:
    arr = np.asarray(label)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.int64, copy=False)
    known_codes = set(WHU_H2SR_LABEL_CODE_TO_DENSE) | {0}
    present_codes = set(int(v) for v in np.unique(arr).tolist())
    if present_codes.issubset(known_codes):
        mapped = np.zeros(arr.shape, dtype=np.uint8)
        for raw_code, dense_code in WHU_H2SR_LABEL_CODE_TO_DENSE.items():
            if dense_code <= class_count:
                mapped[arr == raw_code] = dense_code
        return mapped
    values = np.unique(arr)
    values = values[values >= 0]
    if values.size == 0:
        return arr.astype(np.uint8)
    max_value = int(values.max())
    min_value = int(values.min())
    if min_value >= 0 and max_value <= class_count - 1:
        # Dense labels are commonly encoded as 0..7; masked_ce expects 1..C.
        return (arr + 1).astype(np.uint8)
    if min_value >= 1 and max_value <= class_count:
        return arr.astype(np.uint8)
    arr = arr.copy()
    arr[(arr < 1) | (arr > class_count)] = 0
    return arr.astype(np.uint8)


def parse_class_boosts(spec: str, class_count: int) -> np.ndarray:
    boosts = np.ones((class_count,), dtype=np.float32)
    spec = str(spec or "").strip()
    if not spec:
        return boosts
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Class boost entry must be CLASS:WEIGHT, got {item!r}.")
        key, value = item.split(":", 1)
        class_id = int(key)
        if class_id < 1 or class_id > class_count:
            raise ValueError(f"Class boost id {class_id} is outside 1..{class_count}.")
        boosts[class_id - 1] = float(value)
    return boosts


def build_whu_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    image_files = sorted(p for p in data_dir.rglob("*.tif") if p.is_file() and "label" not in str(p).lower())
    label_files = sorted(p for p in data_dir.glob("H2SR_label_*/*.tif") if p.is_file())
    labels_by_name = {p.name: p for p in label_files}
    pairs = [(image, labels_by_name[image.name]) for image in image_files if image.name in labels_by_name]
    if not pairs:
        raise FileNotFoundError(f"No WHU-H2SR image/label TIFF pairs found under {data_dir}.")
    return pairs


def split_tile_pairs(
    pairs: list[tuple[Path, Path]],
    *,
    seed: int,
    output_path: Path,
    train_fraction: float = 0.20,
    val_fraction: float = 0.10,
) -> dict[str, np.ndarray]:
    if output_path.exists():
        payload = np.load(output_path)
        return {name: payload[name] for name in ("train", "val", "test")}
    rng = np.random.default_rng(seed)
    order = np.arange(len(pairs), dtype=np.int64)
    rng.shuffle(order)
    train_fraction = float(train_fraction)
    val_fraction = float(val_fraction)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(f"train_fraction + val_fraction must be < 1, got {train_fraction + val_fraction}.")
    train_count = max(1, int(round(len(order) * train_fraction)))
    val_count = int(round(len(order) * val_fraction))
    if val_fraction > 0:
        val_count = max(1, val_count)
    val_count = min(val_count, max(len(order) - train_count - 1, 0))
    result = {
        "train": order[:train_count],
        "val": order[train_count : train_count + val_count],
        "test": order[train_count + val_count :],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    return result


def _read_tiff_array(path: Path) -> np.ndarray:
    import tifffile

    try:
        return np.asarray(tifffile.memmap(path))
    except Exception:
        return np.asarray(tifffile.imread(path))


def estimate_tile_band_stats(
    pairs: list[tuple[Path, Path]],
    indices: np.ndarray,
    *,
    sample_count: int,
    seed: int,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if output_path.exists():
        data = np.load(output_path)
        return data["mean"].astype(np.float32), data["std"].astype(np.float32)
    first = _read_tiff_array(pairs[int(indices[0])][0])
    if first.ndim != 3:
        raise ValueError(f"Expected WHU image shaped HxWxC, got {first.shape}.")
    bands = int(first.shape[-1])
    if sample_count <= 0:
        mean = np.zeros((bands,), dtype=np.float32)
        std = np.ones((bands,), dtype=np.float32)
    else:
        rng = np.random.default_rng(seed)
        total = np.zeros((bands,), dtype=np.float64)
        total_sq = np.zeros((bands,), dtype=np.float64)
        seen = 0
        per_tile = max(64, min(1024, int(math.ceil(sample_count / max(len(indices), 1)))))
        shuffled = np.array(indices, dtype=np.int64, copy=True)
        rng.shuffle(shuffled)
        for idx in shuffled:
            image = _read_tiff_array(pairs[int(idx)][0])
            if image.ndim != 3:
                raise ValueError(f"Expected WHU image shaped HxWxC, got {image.shape} for {pairs[int(idx)][0]}.")
            h, w, _ = image.shape
            n = min(per_tile, h * w, max(sample_count - seen, 0))
            if n <= 0:
                break
            flat = rng.choice(h * w, size=n, replace=False)
            rows = flat // w
            cols = flat % w
            values = np.asarray(image[rows, cols, :], dtype=np.float32)
            total += values.sum(axis=0)
            total_sq += np.square(values, dtype=np.float32).sum(axis=0)
            seen += int(values.shape[0])
            if seen >= sample_count:
                break
        mean = total / max(seen, 1)
        var = np.maximum(total_sq / max(seen, 1) - mean * mean, 1.0e-8)
        std = np.sqrt(var)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_whu_class_counts(pairs: list[tuple[Path, Path]], indices: np.ndarray, class_count: int) -> np.ndarray:
    counts = np.zeros((class_count,), dtype=np.float64)
    for idx in indices:
        label = normalize_whu_labels(_read_tiff_array(pairs[int(idx)][1]), class_count)
        vals = label.reshape(-1)
        bincount = np.bincount(vals[vals > 0] - 1, minlength=class_count)
        counts += bincount[:class_count]
    counts[counts <= 0] = 1.0
    return counts


class WhuTileDataset(Dataset):
    def __init__(
        self,
        *,
        pairs: list[tuple[Path, Path]],
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        crop_size: int,
        class_count: int,
        train: bool,
        length: int | None = None,
        seed: int = 0,
    ) -> None:
        self.pairs = pairs
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = mean.astype(np.float32, copy=False)
        self.std = np.maximum(std.astype(np.float32, copy=False), 1.0e-6)
        self.crop_size = int(crop_size)
        self.class_count = int(class_count)
        self.train = bool(train)
        self.length = int(length) if length is not None else int(self.indices.size)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def _pair_index(self, index: int) -> int:
        if self.train:
            return int(self.indices[int(self.rng.integers(0, self.indices.size))])
        return int(self.indices[index % self.indices.size])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.pairs[self._pair_index(index)]
        image = _read_tiff_array(image_path)
        label = normalize_whu_labels(_read_tiff_array(label_path), self.class_count)
        if image.ndim != 3:
            raise ValueError(f"Expected WHU image shaped HxWxC, got {image.shape} for {image_path}.")
        height, width, _ = image.shape
        max_top = max(height - self.crop_size, 0)
        max_left = max(width - self.crop_size, 0)
        if self.train:
            top = int(self.rng.integers(0, max_top + 1)) if max_top > 0 else 0
            left = int(self.rng.integers(0, max_left + 1)) if max_left > 0 else 0
        else:
            top = max_top // 2
            left = max_left // 2
        image_crop = np.asarray(image[top : top + self.crop_size, left : left + self.crop_size, :], dtype=np.float32)
        label_crop = np.asarray(label[top : top + self.crop_size, left : left + self.crop_size], dtype=np.int64)
        image_crop = np.transpose(image_crop, (2, 0, 1))
        image_crop = (image_crop - self.mean[:, None, None]) / self.std[:, None, None]
        if self.train:
            if random.random() < 0.5:
                image_crop = image_crop[:, ::-1, :]
                label_crop = label_crop[::-1, :]
            if random.random() < 0.5:
                image_crop = image_crop[:, :, ::-1]
                label_crop = label_crop[:, ::-1]
        return torch.from_numpy(image_crop.copy()).float(), torch.from_numpy(label_crop.copy()).long()


def group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class AwareUPerNetSeg(nn.Module):
    def __init__(
        self,
        extractor: AwareFeatureExtractor,
        num_classes: int,
        *,
        input_channels: int = 0,
        decoder_channels: int = 256,
        segmentation_channels: int = 64,
        raw_shortcut: bool = False,
        raw_shortcut_channels: int = 128,
        head_channels: int = 320,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.raw_shortcut = bool(raw_shortcut)
        self.embed_dim = extractor.model_dim
        self.neck = MultiLevelNeck(
            in_channels=[self.embed_dim] * 4,
            out_channels=decoder_channels,
            scales=[4, 2, 1, 0.5],
        )
        self.decoder = UPerNetDecoder(
            encoder_channels=[decoder_channels] * 4,
            encoder_depth=4,
            pyramid_channels=decoder_channels,
            segmentation_channels=segmentation_channels,
            target_upsample=max(1, extractor.local_patch_size // 4),
        )
        if self.raw_shortcut:
            if int(input_channels) <= 0:
                raise ValueError("input_channels must be positive when raw_shortcut is enabled.")
            self.local_proj = nn.Sequential(
                nn.Conv2d(int(input_channels), raw_shortcut_channels, 3, padding=1),
                nn.GroupNorm(group_count(raw_shortcut_channels), raw_shortcut_channels),
                nn.GELU(),
                nn.Conv2d(raw_shortcut_channels, raw_shortcut_channels, 3, padding=1),
                nn.GroupNorm(group_count(raw_shortcut_channels), raw_shortcut_channels),
                nn.GELU(),
            )
            head_in = segmentation_channels + int(input_channels) + raw_shortcut_channels
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(head_in, head_channels, 3, padding=1),
                nn.GroupNorm(group_count(head_channels), head_channels),
                nn.GELU(),
                nn.Dropout2d(float(head_dropout)),
                nn.Conv2d(head_channels, head_channels, 3, padding=1),
                nn.GroupNorm(group_count(head_channels), head_channels),
                nn.GELU(),
                nn.Conv2d(head_channels, num_classes, kernel_size=1),
            )
        else:
            self.segmentation_head = nn.Conv2d(segmentation_channels, num_classes, kernel_size=1)

    @staticmethod
    def _token_map(tokens: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        h, w = grid_size
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor.forward_features(x)
        intermediate = list(features["intermediate_tokens"])[-4:]
        maps = [self._token_map(tokens, features["grid_size"]) for tokens in intermediate]
        neck_features = self.neck(maps)
        decoded = self.decoder(*neck_features)
        if self.raw_shortcut:
            if decoded.shape[-2:] != x.shape[-2:]:
                decoded = F.interpolate(decoded, size=x.shape[-2:], mode="bilinear", align_corners=False)
            local = self.local_proj(x)
            logits = self.segmentation_head(torch.cat([x, local, decoded], dim=1))
        else:
            logits = self.segmentation_head(decoded)
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def masked_ce(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    valid = labels > 0
    if int(valid.sum().item()) == 0:
        return None, logits.new_empty((0,), dtype=torch.long), logits.new_empty((0,), dtype=torch.long)
    target = labels[valid].long() - 1
    pred_logits = logits.permute(0, 2, 3, 1)[valid]
    return F.cross_entropy(pred_logits, target, weight=class_weights), pred_logits.argmax(dim=-1), target


def masked_segmentation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    *,
    dice_loss_weight: float = 0.0,
    focal_loss_weight: float = 0.0,
    focal_gamma: float = 1.5,
    label_smoothing: float = 0.0,
    soft_iou_loss_weight: float = 0.0,
    lovasz_loss_weight: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    valid = labels > 0
    if int(valid.sum().item()) == 0:
        return None, logits.new_empty((0,), dtype=torch.long), logits.new_empty((0,), dtype=torch.long)
    target = labels[valid].long() - 1
    pred_logits = logits.permute(0, 2, 3, 1)[valid]
    ce = F.cross_entropy(pred_logits, target, weight=class_weights, label_smoothing=float(label_smoothing))
    loss = ce
    if focal_loss_weight > 0:
        ce_each = F.cross_entropy(
            pred_logits,
            target,
            weight=class_weights,
            reduction="none",
            label_smoothing=float(label_smoothing),
        )
        pt = torch.softmax(pred_logits, dim=-1).gather(1, target[:, None]).squeeze(1).clamp_min(1.0e-6)
        focal = torch.pow(1.0 - pt, float(focal_gamma)) * ce_each
        loss = loss + float(focal_loss_weight) * focal.mean()
    if dice_loss_weight > 0:
        probs = torch.softmax(pred_logits, dim=-1)
        one_hot = F.one_hot(target, num_classes=pred_logits.shape[-1]).to(dtype=probs.dtype)
        intersection = (probs * one_hot).sum(dim=0)
        denominator = probs.sum(dim=0) + one_hot.sum(dim=0)
        dice = 1.0 - (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)
        if class_weights is not None:
            dice_weights = class_weights.to(dtype=dice.dtype)
            dice_weights = dice_weights / dice_weights.mean().clamp_min(1.0e-6)
            dice = dice * dice_weights
        loss = loss + float(dice_loss_weight) * dice.mean()
    if soft_iou_loss_weight > 0 or lovasz_loss_weight > 0:
        probs = torch.softmax(pred_logits, dim=-1)
        one_hot = F.one_hot(target, num_classes=pred_logits.shape[-1]).to(dtype=probs.dtype)
    if soft_iou_loss_weight > 0:
        intersection = (probs * one_hot).sum(dim=0)
        union = (probs + one_hot - probs * one_hot).sum(dim=0)
        soft_iou = 1.0 - (intersection + 1.0e-6) / (union + 1.0e-6)
        if class_weights is not None:
            iou_weights = class_weights.to(dtype=soft_iou.dtype)
            iou_weights = iou_weights / iou_weights.mean().clamp_min(1.0e-6)
            soft_iou = soft_iou * iou_weights
        loss = loss + float(soft_iou_loss_weight) * soft_iou.mean()
    if lovasz_loss_weight > 0:
        loss = loss + float(lovasz_loss_weight) * lovasz_softmax_flat(probs.float(), target.long(), class_weights)
    return loss, pred_logits.argmax(dim=-1), target


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gt_sorted = gt_sorted.float()
    p = int(gt_sorted.numel())
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1.0e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
    return jaccard


def lovasz_softmax_flat(
    probs: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    class_count = int(probs.shape[-1])
    for class_id in range(class_count):
        fg = (labels == class_id).float()
        if int(fg.sum().item()) == 0:
            continue
        class_pred = probs[:, class_id]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        class_loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        if class_weights is not None:
            class_loss = class_loss * class_weights[class_id].detach().to(device=class_loss.device, dtype=class_loss.dtype)
        losses.append(class_loss)
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def compute_metrics(pred: np.ndarray, target: np.ndarray, class_count: int) -> dict[str, Any]:
    mask = target >= 0
    pred = pred[mask].astype(np.int64, copy=False)
    target = target[mask].astype(np.int64, copy=False)
    if target.size == 0:
        return {"oa": 0.0, "aa": 0.0, "kappa": 0.0, "miou": 0.0, "per_class_acc": [], "per_class_iou": []}
    conf = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(conf, (target, pred), 1)
    oa = float(np.trace(conf) / max(conf.sum(), 1))
    acc = []
    iou = []
    for c in range(class_count):
        tp = conf[c, c]
        denom_acc = conf[c, :].sum()
        denom_iou = conf[c, :].sum() + conf[:, c].sum() - tp
        acc.append(float(tp / denom_acc) if denom_acc > 0 else float("nan"))
        iou.append(float(tp / denom_iou) if denom_iou > 0 else float("nan"))
    aa = float(np.nanmean(acc))
    miou = float(np.nanmean(iou))
    kappa = float(cohen_kappa_score(target, pred, labels=list(range(class_count))))
    return {"oa": oa, "aa": aa, "kappa": kappa, "miou": miou, "per_class_acc": acc, "per_class_iou": iou}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_count: int, class_weights: torch.Tensor | None, precision: str) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    counts: list[int] = []
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    autocast_enabled = device.type == "cuda" and precision == "bf16"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(x)
            loss, pred, target = masked_ce(logits, y, class_weights)
        if loss is None:
            continue
        n = int(target.numel())
        losses.append(float(loss.item()) * n)
        counts.append(n)
        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
    if preds:
        metrics = compute_metrics(np.concatenate(preds), np.concatenate(targets), class_count)
    else:
        metrics = compute_metrics(np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), class_count)
    metrics["loss"] = float(sum(losses) / max(sum(counts), 1))
    metrics["pixels"] = int(sum(counts))
    return metrics


@torch.no_grad()
def evaluate_qingpu_sliding_scene(
    model: nn.Module,
    image: SpectralImageView,
    target_labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    crop_size: int,
    stride: int,
    device: torch.device,
    class_count: int,
    precision: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    height, width = target_labels.shape
    crop_size = int(crop_size)
    stride = max(1, int(stride))
    positions_y = list(range(0, max(height - crop_size, 0) + 1, stride))
    positions_x = list(range(0, max(width - crop_size, 0) + 1, stride))
    if not positions_y or positions_y[-1] != max(height - crop_size, 0):
        positions_y.append(max(height - crop_size, 0))
    if not positions_x or positions_x[-1] != max(width - crop_size, 0):
        positions_x.append(max(width - crop_size, 0))
    pred_map = np.zeros((height, width), dtype=np.uint8)
    border = max((crop_size - stride) // 2, 0)
    autocast_enabled = device.type == "cuda" and precision == "bf16"
    std = np.maximum(std.astype(np.float32, copy=False), 1.0)
    mean = mean.astype(np.float32, copy=False)
    total = len(positions_y) * len(positions_x)
    done = 0
    for top in positions_y:
        for left in positions_x:
            patch = image.read_patch(top, left, crop_size)
            patch = (patch - mean[:, None, None]) / std[:, None, None]
            x = torch.from_numpy(patch.copy()).unsqueeze(0).float().to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = model(x)
            pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8) + 1
            fill_top = top if top == 0 else min(top + border, height)
            fill_left = left if left == 0 else min(left + border, width)
            fill_bottom = min(top + crop_size, height) if top + crop_size >= height else min(top + crop_size - border, height)
            fill_right = min(left + crop_size, width) if left + crop_size >= width else min(left + crop_size - border, width)
            local_top = fill_top - top
            local_left = fill_left - left
            local_bottom = local_top + (fill_bottom - fill_top)
            local_right = local_left + (fill_right - fill_left)
            if fill_bottom > fill_top and fill_right > fill_left:
                pred_map[fill_top:fill_bottom, fill_left:fill_right] = pred[local_top:local_bottom, local_left:local_right]
            done += 1
            if done % 200 == 0:
                print(json.dumps({"event": "sliding_eval_progress", "done": done, "total": total}, ensure_ascii=True), flush=True)
    valid = target_labels > 0
    metrics = compute_metrics(pred_map[valid].astype(np.int64) - 1, target_labels[valid].astype(np.int64) - 1, class_count)
    metrics["pixels"] = int(valid.sum())
    metrics["crop_size"] = crop_size
    metrics["stride"] = stride
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, pred_map=pred_map, target_labels=target_labels.astype(np.uint8, copy=False))
    return metrics


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp_{int(time.time() * 1000)}")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_compatible_model_state(model: nn.Module, path: Path) -> dict[str, int | str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_state = payload.get("model", payload) if isinstance(payload, dict) else payload
    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
    }
    target_state.update(compatible)
    model.load_state_dict(target_state)
    return {
        "path": str(path),
        "loaded": len(compatible),
        "source": len(source_state),
        "target": len(target_state),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpecAware-style single-GPU segmentation fine-tuning for Aware checkpoints.")
    parser.add_argument("--dataset", choices=("qingpu", "aerorit", "whu"), default="qingpu")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype-bank-eo1", type=Path, required=True)
    parser.add_argument("--prototype-bank-gf5", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--train-per-class", type=int, default=2000)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--jitter", type=int, default=32)
    parser.add_argument("--stats-samples", type=int, default=80000)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--split-mode", choices=("random_per_class", "spatial_blocks", "qingpu_train_areas", "aerorit_patch_201070"), default="random_per_class")
    parser.add_argument("--spatial-block-size", type=int, default=1024)
    parser.add_argument("--spatial-train-fraction", type=float, default=0.60)
    parser.add_argument("--spatial-val-fraction", type=float, default=0.20)
    parser.add_argument("--aerorit-band-subset", choices=("full", "specaware_305"), default="full")
    parser.add_argument("--train-area-size", type=int, default=1024)
    parser.add_argument("--train-area-count", type=int, default=16)
    parser.add_argument("--center-only-labels", action="store_true")
    parser.add_argument("--stats-from-train-only", action="store_true")
    parser.add_argument("--final-qingpu-sliding-eval", action="store_true")
    parser.add_argument("--full-scene-stride", type=int, default=128)
    parser.add_argument("--decoder-channels", type=int, default=256)
    parser.add_argument("--segmentation-channels", type=int, default=64)
    parser.add_argument("--raw-shortcut", action="store_true")
    parser.add_argument("--raw-shortcut-channels", type=int, default=128)
    parser.add_argument("--head-channels", type=int, default=320)
    parser.add_argument("--head-dropout", type=float, default=0.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--class-boosts", type=str, default="")
    parser.add_argument("--dice-loss-weight", type=float, default=0.0)
    parser.add_argument("--focal-loss-weight", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--soft-iou-loss-weight", type=float, default=0.0)
    parser.add_argument("--lovasz-loss-weight", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "qingpu":
        image, image_info = open_envi_image(args.data_dir / "Qingpu_HSI_image.dat", args.data_dir / "Qingpu_HSI_image.hdr")
        gt_mem, gt_info = open_envi_image(args.data_dir / "Qingpu_HSI_gt.dat", args.data_dir / "Qingpu_HSI_gt.hdr")
        if gt_mem.layout == "bsq":
            labels = np.asarray(gt_mem.data[0], dtype=np.uint8)
        elif gt_mem.layout == "bil":
            labels = np.asarray(gt_mem.data[:, 0, :], dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported Qingpu label layout {gt_mem.layout!r}")
        class_names = QINGPU_CLASS_NAMES
        metadata_sensor = "eo1"
        metadata_level = "L1"
        metadata_gsd = 0.75
        wavelengths = torch.linspace(400.0, 1000.0, int(image_info["bands"]))
        protocol_note = "Qingpu SpecAware-style run: UPerNet decoder, weighted CE, LR 5e-5, 100 epochs. Since the released package has no explicit training-area masks, the default run uses fixed-seed sampled train/val/test pixels per class. The optional spatial_blocks split is a conservative recheck that holds out disjoint spatial blocks."
    elif args.dataset == "aerorit":
        import tifffile

        refl = args.data_dir / "full_reflectance" / "2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine-001_"
        hdr = args.data_dir / "headers" / "2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine.hdr"
        if not hdr.exists():
            hdr = args.data_dir / "headers_extracted" / "2017-08-01_14-28-56_000011__061158-066258_radiance_fwd_proj_EmpLine.hdr"
        image, image_info = open_envi_image(refl, hdr)
        rgb = np.asarray(tifffile.memmap(args.data_dir / "sample_10band" / "image_labels.tif"), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"AeroRIT labels should be HxWx3 RGB, got {rgb.shape}")
        labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for class_id, color in enumerate(AERORIT_CLASS_COLORS, start=1):
            labels[np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)] = class_id
        if int((labels == 0).sum()) > 0:
            raise ValueError(f"AeroRIT labels contain {int((labels == 0).sum())} unmapped RGB pixels.")
        image, image_info, band_note = apply_aerorit_band_subset(image, image_info, args.aerorit_band_subset)
        gt_info = {"samples": labels.shape[1], "lines": labels.shape[0], "bands": 1, "dtype": "rgb_label_tif", "interleave": "rgb"}
        class_names = AERORIT_CLASS_NAMES
        metadata_sensor = "others"
        metadata_level = "L1"
        metadata_gsd = 0.4
        if len(image_info.get("wavelengths", [])) == int(image_info["bands"]):
            wavelengths = torch.tensor(image_info["wavelengths"], dtype=torch.float32)
        else:
            wavelengths = torch.linspace(400.0, 897.0, int(image_info["bands"]))
        protocol_note = "AeroRIT SpecAware-style single-GPU run: full reflectance BIL scene plus RGB label TIFF, UPerNet decoder, weighted CE, fixed-seed sampled train/val/test pixels per class. Effective batch is emulated by batch_size * grad_accum."
        protocol_note += band_note
    elif args.dataset == "whu":
        pairs = build_whu_pairs(args.data_dir)
        image_info = {"samples": 300, "lines": 300, "bands": 249, "dtype": "float32_tiff", "interleave": "tif_yxs"}
        gt_info = {"samples": 300, "lines": 300, "bands": 1, "dtype": "label_tif", "interleave": "tif_yx"}
        class_names = WHU_H2SR_CLASS_NAMES
        metadata_sensor = "others"
        metadata_level = "surface"
        metadata_gsd = 1.0
        wavelengths = torch.linspace(400.0, 1000.0, int(image_info["bands"]))
        protocol_note = "WHU-H2SR SpecAware-style run: 2531 300x300x249 reflectance tiles, fixed-seed tile split 20/10/70, random 256x256 training crop, center 256x256 validation/test crop, UPerNet decoder, weighted CE, LR 5e-5, 100 epochs. This follows the SpecAware paper setting where WHU-H2SR uses 256x256 inputs and batch size 16; effective batch is emulated by batch_size * grad_accum."
    else:
        raise ValueError(args.dataset)

    class_count = len(class_names)
    patch_splits: dict[str, np.ndarray] | None = None
    if args.dataset == "whu":
        split_path = args.output_dir / f"tile_splits_seed{args.seed}_tr20_va10_te70.npz"
        splits = split_tile_pairs(pairs, seed=args.seed, output_path=split_path)
        mean, std = estimate_tile_band_stats(
            pairs,
            splits["train"],
            sample_count=args.stats_samples,
            seed=args.seed,
            output_path=args.output_dir / f"tile_band_stats_{args.stats_samples}_seed{args.seed}.npz",
        )
        class_counts = compute_whu_class_counts(pairs, splits["train"], class_count)
        weights = 1.0 / np.power(class_counts, float(args.class_weight_power))
        weights *= parse_class_boosts(args.class_boosts, class_count)
        weights = weights / max(weights.mean(), 1.0e-12)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        train_dataset = WhuTileDataset(
            pairs=pairs,
            indices=splits["train"],
            mean=mean,
            std=std,
            crop_size=args.crop_size,
            class_count=class_count,
            train=True,
            length=args.steps_per_epoch * args.batch_size,
            seed=args.seed + 17,
        )
        val_dataset = WhuTileDataset(
            pairs=pairs,
            indices=splits["val"],
            mean=mean,
            std=std,
            crop_size=args.crop_size,
            class_count=class_count,
            train=False,
            seed=args.seed + 29,
        )
        test_dataset = WhuTileDataset(
            pairs=pairs,
            indices=splits["test"],
            mean=mean,
            std=std,
            crop_size=args.crop_size,
            class_count=class_count,
            train=False,
            seed=args.seed + 41,
        )
    else:
        if args.split_mode == "aerorit_patch_201070":
            if args.dataset != "aerorit":
                raise ValueError("split-mode aerorit_patch_201070 is only valid for --dataset aerorit.")
            split_path = args.output_dir / f"splits_aerorit_patch128_seed{args.seed}_tr20_va10_te70.npz"
            splits, patch_splits = split_indices_aerorit_patch_201070(
                labels,
                seed=args.seed,
                output_path=split_path,
                patch_size=args.crop_size,
            )
            protocol_note = (
                "AeroRIT SpecAware-paper recheck: full scene is tiled into non-overlapping "
                f"{args.crop_size}x{args.crop_size} patches, then split by patch into 20% train, "
                "10% validation, and 70% test. Training, validation, and test loaders use the "
                "exact assigned patches with dense patch labels."
            ) + band_note
        elif args.split_mode == "spatial_blocks":
            split_path = args.output_dir / (
                f"splits_spatial_b{args.spatial_block_size}_seed{args.seed}"
                f"_tr{args.train_per_class}_va{args.val_per_class}_te{args.test_per_class}.npz"
            )
            splits = split_indices_spatial_blocks(
                labels,
                class_count=class_count,
                train_per_class=args.train_per_class,
                val_per_class=args.val_per_class,
                test_per_class=args.test_per_class,
                seed=args.seed,
                output_path=split_path,
                block_size=args.spatial_block_size,
                train_fraction=args.spatial_train_fraction,
                val_fraction=args.spatial_val_fraction,
            )
            protocol_note += (
                f" Recheck split: disjoint {args.spatial_block_size}x{args.spatial_block_size} spatial blocks,"
                f" train/val/test block fractions approximately {args.spatial_train_fraction:.2f}/"
                f"{args.spatial_val_fraction:.2f}/{1.0 - args.spatial_train_fraction - args.spatial_val_fraction:.2f}."
            )
        elif args.split_mode == "qingpu_train_areas":
            split_path = args.output_dir / (
                f"splits_qingpu_auto{args.train_area_count}areas_a{args.train_area_size}_seed{args.seed}"
                f"_tr{args.train_per_class}_va{args.val_per_class}_te{args.test_per_class}.npz"
            )
            splits = split_indices_qingpu_train_areas(
                labels,
                class_count=class_count,
                train_per_class=args.train_per_class,
                val_per_class=args.val_per_class,
                test_per_class=args.test_per_class,
                seed=args.seed,
                output_path=split_path,
                area_size=args.train_area_size,
                area_count=args.train_area_count,
            )
            protocol_note += (
                f" Official-like Qingpu split: approximate PatchOut/FreeNet protocol using"
                f" {args.train_area_count} automatically selected {args.train_area_size}x{args.train_area_size}"
                f" training areas, 2000 pixels/class for training, and non-training regions for validation/test."
            )
        else:
            split_path = args.output_dir / f"splits_seed{args.seed}_tr{args.train_per_class}_va{args.val_per_class}_te{args.test_per_class}.npz"
            splits = split_indices(
                labels,
                class_count=class_count,
                train_per_class=args.train_per_class,
                val_per_class=args.val_per_class,
                test_per_class=args.test_per_class,
                seed=args.seed,
                output_path=split_path,
            )
        if args.center_only_labels:
            protocol_note += " Center-only supervision/evaluation is enabled to prevent repeated counting from overlapping crops."
        train_labels = make_sparse_label_map(labels, splits["train"])
        val_labels = make_sparse_label_map(labels, splits["val"])
        test_labels = make_sparse_label_map(labels, splits["test"])
        stats_labels = train_labels if args.stats_from_train_only else labels
        mean, std = estimate_band_stats(
            image,
            stats_labels,
            sample_count=args.stats_samples,
            seed=args.seed,
            output_path=args.output_dir / f"band_stats_{args.stats_samples}_seed{args.seed}.npz",
        )
        class_counts = np.bincount(train_labels.reshape(-1), minlength=class_count + 1)[1:].astype(np.float64)
        class_counts[class_counts <= 0] = 1.0
        weights = 1.0 / np.power(class_counts, float(args.class_weight_power))
        weights *= parse_class_boosts(args.class_boosts, class_count)
        weights = weights / max(weights.mean(), 1.0e-12)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        if patch_splits is not None:
            train_dataset = SpectralPatchDataset(
                image=image,
                labels=train_labels,
                patch_coords=patch_splits["train"],
                mean=mean,
                std=std,
                patch_size=args.crop_size,
                train=True,
                length=args.steps_per_epoch * args.batch_size,
                seed=args.seed + 17,
            )
            val_dataset = SpectralPatchDataset(
                image=image,
                labels=val_labels,
                patch_coords=patch_splits["val"],
                mean=mean,
                std=std,
                patch_size=args.crop_size,
                train=False,
                seed=args.seed + 29,
            )
            test_dataset = SpectralPatchDataset(
                image=image,
                labels=test_labels,
                patch_coords=patch_splits["test"],
                mean=mean,
                std=std,
                patch_size=args.crop_size,
                train=False,
                seed=args.seed + 41,
            )
        else:
            train_dataset = SpectralCropDataset(
                image=image,
                labels=train_labels,
                indices=splits["train"],
                mean=mean,
                std=std,
                crop_size=args.crop_size,
                train=True,
                length=args.steps_per_epoch * args.batch_size,
                jitter=args.jitter,
                seed=args.seed + 17,
                center_only_labels=args.center_only_labels,
            )
            val_dataset = SpectralCropDataset(
                image=image,
                labels=val_labels,
                indices=splits["val"],
                mean=mean,
                std=std,
                crop_size=args.crop_size,
                train=False,
                seed=args.seed + 29,
                center_only_labels=args.center_only_labels,
            )
            test_dataset = SpectralCropDataset(
                image=image,
                labels=test_labels,
                indices=splits["test"],
                mean=mean,
                std=std,
                crop_size=args.crop_size,
                train=False,
                seed=args.seed + 41,
                center_only_labels=args.center_only_labels,
            )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    metadata = build_downstream_metadata(
        num_bands=int(image_info["bands"]),
        sensor=metadata_sensor,
        level=metadata_level,
        gsd=metadata_gsd,
        wavelengths_nm=wavelengths,
        fwhm_nm=None,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        metadata=metadata,
        freeze_encoder=args.freeze_encoder,
        device=device,
    )
    model = AwareUPerNetSeg(
        extractor,
        class_count,
        input_channels=int(wavelengths.numel()),
        decoder_channels=args.decoder_channels,
        segmentation_channels=args.segmentation_channels,
        raw_shortcut=args.raw_shortcut,
        raw_shortcut_channels=args.raw_shortcut_channels,
        head_channels=args.head_channels,
        head_dropout=args.head_dropout,
    ).to(device)
    init_model_stats = None
    if args.init_model is not None:
        init_model_stats = load_compatible_model_state(model, args.init_model)
        print(json.dumps({"event": "init_model", "stats": init_model_stats}, ensure_ascii=True), flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    autocast_enabled = device.type == "cuda" and args.precision == "bf16"

    config = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "image_info": image_info,
        "gt_info": gt_info,
        "dataset": args.dataset,
        "class_names": class_names,
        "split_counts": {k: int(v.size) for k, v in splits.items()},
        "patch_split_counts": None if patch_splits is None else {k: int(v.shape[0]) for k, v in patch_splits.items()},
        "class_weights": weights.astype(float).tolist(),
        "class_boost_values": parse_class_boosts(args.class_boosts, class_count).astype(float).tolist(),
        "init_model_stats": init_model_stats,
        "protocol_note": protocol_note,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    latest_path = args.output_dir / "latest.pt"
    best_path = args.output_dir / "best_val_miou.pt"
    start_epoch = 1
    best_miou = -math.inf
    if args.auto_resume and latest_path.exists():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_miou = float(payload.get("best_miou", best_miou))
        print(json.dumps({"event": "resume", "start_epoch": start_epoch, "best_miou": best_miou}, ensure_ascii=True), flush=True)

    if args.dry_run:
        x, y = next(iter(train_loader))
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(x)
            loss, _, _ = masked_segmentation_loss(
                logits,
                y,
                class_weights,
                dice_loss_weight=args.dice_loss_weight,
                focal_loss_weight=args.focal_loss_weight,
                focal_gamma=args.focal_gamma,
                soft_iou_loss_weight=args.soft_iou_loss_weight,
                lovasz_loss_weight=args.lovasz_loss_weight,
                label_smoothing=args.label_smoothing,
            )
        print(json.dumps({"event": "dry_run", "input": list(x.shape), "logits": list(logits.shape), "loss": None if loss is None else float(loss.item())}, ensure_ascii=True), flush=True)
        return

    log_path = args.output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_pixels = 0
        total_correct = 0
        step_count = 0
        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = model(x)
                loss, pred, target = masked_segmentation_loss(
                    logits,
                    y,
                    class_weights,
                    dice_loss_weight=args.dice_loss_weight,
                    focal_loss_weight=args.focal_loss_weight,
                    focal_gamma=args.focal_gamma,
                    soft_iou_loss_weight=args.soft_iou_loss_weight,
                    lovasz_loss_weight=args.lovasz_loss_weight,
                    label_smoothing=args.label_smoothing,
                )
                if loss is None:
                    continue
                scaled_loss = loss / max(args.grad_accum, 1)
            scaler.scale(scaled_loss).backward()
            if step % max(args.grad_accum, 1) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            pixels = int(target.numel())
            total_loss += float(loss.item()) * pixels
            total_pixels += pixels
            total_correct += int((pred == target).sum().item())
            step_count += 1
        if step_count % max(args.grad_accum, 1) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, class_count, class_weights, args.precision)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_pixels, 1),
            "train_oa": total_correct / max(total_pixels, 1),
            "train_pixels": total_pixels,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(record, ensure_ascii=True), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou,
            "record": record,
            "config": config,
        }
        atomic_save(payload, latest_path)
        if float(val_metrics["miou"]) > best_miou:
            best_miou = float(val_metrics["miou"])
            payload["best_miou"] = best_miou
            atomic_save(payload, best_path)
    final: dict[str, Any] = {}
    for name, path in {"best_val_miou": best_path, "latest": latest_path}.items():
        payload = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        test_metrics = evaluate(model, test_loader, device, class_count, class_weights, args.precision)
        record_payload = {
            "epoch": int(payload["epoch"]),
            "val": payload["record"]["val"],
            "test": test_metrics,
        }
        if args.dataset == "qingpu" and args.final_qingpu_sliding_eval:
            slide_metrics = evaluate_qingpu_sliding_scene(
                model,
                image,
                test_labels,
                mean,
                std,
                crop_size=args.crop_size,
                stride=args.full_scene_stride,
                device=device,
                class_count=class_count,
                precision=args.precision,
                output_path=args.output_dir / f"{name}_qingpu_sliding_prediction_stride{args.full_scene_stride}.npz",
            )
            record_payload["sliding_test"] = slide_metrics
        final[name] = record_payload
    (args.output_dir / "test_metrics.json").write_text(json.dumps(final, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "final", "metrics": final}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
