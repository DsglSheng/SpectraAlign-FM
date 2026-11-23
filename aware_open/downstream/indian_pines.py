from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset


INDIAN_PINES_CLASS_NAMES = [
    "Alfalfa",
    "Corn-notill",
    "Corn-mintill",
    "Corn",
    "Grass-pasture",
    "Grass-trees",
    "Grass-pasture-mowed",
    "Hay-windrowed",
    "Oats",
    "Soybean-notill",
    "Soybean-mintill",
    "Soybean-clean",
    "Wheat",
    "Woods",
    "Buildings-Grass-Trees-Drives",
    "Stone-Steel-Towers",
]


@dataclass(frozen=True)
class IndianPinesProtocol:
    name: str
    pca_components: int
    sensor: str
    level: str
    wavelengths: torch.Tensor
    fwhm: torch.Tensor
    metadata_note: str


class IndianPinesPatchDataset(Dataset):
    def __init__(
        self,
        cube: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        window_size: int,
        input_size: int,
        augment: bool = False,
        spectral_jitter_std: float = 0.0,
    ) -> None:
        if window_size % 2 != 1:
            raise ValueError(f"Indian Pines patch extraction needs an odd window_size, got {window_size}.")
        if input_size < window_size:
            raise ValueError(f"input_size={input_size} must be >= window_size={window_size}.")
        self.cube = cube.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64).reshape(-1)
        self.targets = self.labels - 1
        self.indices = indices.astype(np.int64).reshape(-1)
        self.window_size = int(window_size)
        self.input_size = int(input_size)
        self.augment = bool(augment)
        self.spectral_jitter_std = float(spectral_jitter_std)
        self.height, self.width, _ = self.cube.shape

        margin = (self.window_size - 1) // 2
        self.padded = np.pad(self.cube, ((margin, margin), (margin, margin), (0, 0)), mode="constant")
        pad_total = self.input_size - self.window_size
        self.input_pad_before = pad_total // 2
        self.input_pad_after = pad_total - self.input_pad_before

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        flat_index = int(self.indices[item])
        row = flat_index // self.width
        col = flat_index % self.width
        patch = self.padded[row : row + self.window_size, col : col + self.window_size, :]
        if self.augment:
            if random.random() < 0.5:
                patch = np.flip(patch, axis=0)
            if random.random() < 0.5:
                patch = np.flip(patch, axis=1)
            rotations = random.randint(0, 3)
            if rotations:
                patch = np.rot90(patch, k=rotations, axes=(0, 1))
            if self.spectral_jitter_std > 0.0:
                scale = np.random.normal(1.0, self.spectral_jitter_std, size=(1, 1, patch.shape[-1])).astype(np.float32)
                patch = patch * scale
        if self.input_size != self.window_size:
            patch = np.pad(
                patch,
                (
                    (self.input_pad_before, self.input_pad_after),
                    (self.input_pad_before, self.input_pad_after),
                    (0, 0),
                ),
                mode="reflect",
            )
        label = int(self.targets[flat_index])
        if label < 0:
            raise ValueError("IndianPinesPatchDataset received an unlabeled center pixel.")
        return torch.from_numpy(patch.transpose(2, 0, 1).copy()), torch.tensor(label, dtype=torch.long)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_indian_pines(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    cube_path = data_root / "Indian_pines_corrected.mat"
    gt_path = data_root / "Indian_pines_gt.mat"
    if not cube_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Missing Indian Pines files under {data_root}.")
    cube = sio.loadmat(cube_path)["data"].astype(np.float32)
    labels = sio.loadmat(gt_path)["groundT"].astype(np.int64)
    return cube, labels


def normalize_cube(cube: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, object]]:
    data = cube.astype(np.float32, copy=False)
    if mode == "none":
        return data, {"mode": "none"}
    if mode == "mean_std":
        center = float(data.mean())
        scale = float(data.std())
        if not np.isfinite(scale) or scale < 1.0e-6:
            scale = 1.0
        return ((data - center) / scale).astype(np.float32), {"mode": mode, "mean": center, "std": scale}
    if mode == "minmax":
        data_min = float(data.min())
        data_max = float(data.max())
        scale = data_max - data_min
        if not np.isfinite(scale) or scale < 1.0e-6:
            scale = 1.0
        return ((data - data_min) / scale).astype(np.float32), {"mode": mode, "min": data_min, "max": data_max}
    raise ValueError(f"Unsupported normalization mode {mode!r}.")


def pca_whiten(cube: np.ndarray, components: int) -> tuple[np.ndarray, dict[str, object]]:
    if components <= 0:
        return cube.astype(np.float32), {"enabled": False, "components": int(cube.shape[-1])}
    height, width, channels = cube.shape
    if components > channels:
        raise ValueError(f"pca-components={components} exceeds input band count {channels}.")

    flat = cube.reshape(-1, channels).astype(np.float64)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:components].T
    projected *= math.sqrt(max(flat.shape[0] - 1, 1))
    projected /= np.maximum(singular_values[:components], 1.0e-12)
    return projected.reshape(height, width, components).astype(np.float32), {
        "enabled": True,
        "components": int(components),
        "source_bands": int(channels),
        "whiten": True,
    }


def indian_pines_aviris_wavelengths(num_bands: int) -> torch.Tensor:
    if num_bands != 200:
        return torch.linspace(400.0, 2500.0, int(num_bands), dtype=torch.float32)
    removed = set(range(104, 108)) | set(range(150, 164)) | {220}
    full = [400.0 + (2500.0 - 400.0) * idx / 219.0 for idx in range(220) if (idx + 1) not in removed]
    if len(full) != 200:
        return torch.linspace(400.0, 2500.0, int(num_bands), dtype=torch.float32)
    return torch.tensor(full, dtype=torch.float32)


def fwhm_from_wavelengths(wavelengths_nm: torch.Tensor) -> torch.Tensor:
    wavelengths = wavelengths_nm.float().flatten()
    if wavelengths.numel() == 1:
        return torch.ones_like(wavelengths)
    diffs = torch.diff(wavelengths).abs()
    left = torch.cat([diffs[:1], diffs])
    right = torch.cat([diffs, diffs[-1:]])
    return ((left + right) * 0.5).clamp_min(1.0)


def resolve_indian_pines_protocol(
    *,
    protocol_name: str,
    raw_band_count: int,
    pca_components: int | None,
    sensor: str | None = None,
    level: str | None = None,
    wavelength_start_nm: float = 400.0,
    wavelength_end_nm: float = 2500.0,
) -> IndianPinesProtocol:
    resolved_pca = pca_components
    if resolved_pca is None:
        resolved_pca = 30 if protocol_name == "hypersigma_pca" else 0
    if protocol_name == "hypersigma_pca" and resolved_pca <= 0:
        raise ValueError("hypersigma_pca protocol requires pca-components > 0.")
    if protocol_name != "hypersigma_pca" and resolved_pca > 0:
        raise ValueError(f"{protocol_name} must use raw bands; set pca-components to 0 or omit it.")

    if sensor is None:
        resolved_sensor = "avc" if protocol_name == "hypersigma_pca" else "eo1"
    else:
        resolved_sensor = str(sensor).lower()
    if level is None:
        resolved_level = "surface" if protocol_name == "hypersigma_pca" else "L1"
    else:
        resolved_level = str(level)
    if protocol_name == "hypersigma_pca":
        wavelengths = torch.linspace(float(wavelength_start_nm), float(wavelength_end_nm), int(resolved_pca))
        metadata_note = "PCA channels use synthetic evenly-spaced wavelengths for HyperSIGMA protocol matching."
    else:
        wavelengths = indian_pines_aviris_wavelengths(raw_band_count)
        metadata_note = "Raw Indian Pines bands use approximate AVIRIS wavelengths after common water-band removal."
    return IndianPinesProtocol(
        name=str(protocol_name),
        pca_components=int(resolved_pca),
        sensor=resolved_sensor,
        level=resolved_level,
        wavelengths=wavelengths,
        fwhm=fwhm_from_wavelengths(wavelengths),
        metadata_note=metadata_note,
    )


def split_labeled_indices(
    gt_flat: np.ndarray,
    class_count: int,
    *,
    train_ratio: float,
    val_ratio: float,
    train_num: int,
    val_num: int,
    split_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_index: list[np.ndarray] = []
    val_index: list[np.ndarray] = []
    test_index: list[np.ndarray] = []
    for class_id in range(1, class_count + 1):
        idx = np.where(gt_flat == class_id)[0]
        np.random.shuffle(idx)
        if split_type == "ratio":
            current_train = int(np.ceil(idx.shape[0] * train_ratio))
            current_val = int(np.ceil(idx.shape[0] * val_ratio))
        else:
            current_train = int(train_num) if int(train_num) <= idx.shape[0] + 1 else 10
            current_val = int(val_num)
        train_index.append(idx[:current_train])
        val_index.append(idx[current_train : current_train + current_val])
        test_index.append(idx[current_train + current_val :])
    return np.concatenate(train_index), np.concatenate(val_index), np.concatenate(test_index)


def build_indian_pines_loaders(
    *,
    data_root: Path,
    protocol: IndianPinesProtocol,
    window_size: int,
    input_size: int,
    batch_size: int,
    num_workers: int,
    train_ratio: float,
    val_ratio: float,
    train_num: int,
    val_num: int,
    split_type: str,
    augment: bool,
    spectral_jitter_std: float,
    normalization_mode: str = "auto",
) -> tuple[DataLoader, DataLoader, DataLoader, int, int, dict[str, object]]:
    cube, labels = load_indian_pines(data_root)
    selected_normalization = normalization_mode
    if selected_normalization == "auto":
        selected_normalization = "minmax" if protocol.pca_components > 0 else "mean_std"
    cube, normalization_config = normalize_cube(cube, selected_normalization)
    if protocol.pca_components > 0:
        cube, pca_config = pca_whiten(cube, protocol.pca_components)
    else:
        pca_config = {"enabled": False, "components": int(cube.shape[-1])}

    _, _, channels = cube.shape
    if channels != int(protocol.wavelengths.numel()):
        raise ValueError(f"Input has {channels} channels but protocol metadata has {protocol.wavelengths.numel()} wavelengths.")

    class_count = int(labels.max())
    gt_flat = labels.reshape(-1)
    train_index, val_index, test_index = split_labeled_indices(
        gt_flat,
        class_count,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        train_num=train_num,
        val_num=val_num,
        split_type=split_type,
    )
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    dataset_kwargs = dict(window_size=window_size, input_size=input_size)
    split_summary = {
        "train_samples": int(train_index.shape[0]),
        "val_samples": int(val_index.shape[0]),
        "test_samples": int(test_index.shape[0]),
        "pca": pca_config,
        "normalization": normalization_config,
        "protocol": protocol.name,
        "metadata_note": protocol.metadata_note,
    }
    train_loader = DataLoader(
        IndianPinesPatchDataset(
            cube,
            gt_flat,
            train_index,
            augment=augment,
            spectral_jitter_std=spectral_jitter_std if augment else 0.0,
            **dataset_kwargs,
        ),
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(IndianPinesPatchDataset(cube, gt_flat, val_index, **dataset_kwargs), shuffle=False, **common)
    test_loader = DataLoader(IndianPinesPatchDataset(cube, gt_flat, test_index, **dataset_kwargs), shuffle=False, **common)
    return train_loader, val_loader, test_loader, class_count, channels, split_summary


def confusion_matrix_np(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels.astype(np.int64), predictions.astype(np.int64), strict=False):
        if 0 <= label < num_classes and 0 <= pred < num_classes:
            conf[label, pred] += 1
    return conf


def kappa_from_confusion(conf: np.ndarray) -> float:
    total = float(conf.sum())
    if total <= 0:
        return 0.0
    observed = float(np.trace(conf)) / total
    expected = float((conf.sum(axis=0) * conf.sum(axis=1)).sum()) / (total * total)
    denom = 1.0 - expected
    if abs(denom) < 1.0e-12:
        return 0.0
    return (observed - expected) / denom


def metrics_from_predictions(labels: np.ndarray, predictions: np.ndarray, class_count: int, loss: float) -> dict[str, Any]:
    conf = confusion_matrix_np(labels, predictions, class_count)
    support = conf.sum(axis=1)
    per_class = conf.diagonal() / np.maximum(support, 1)
    class_metrics = [
        {
            "class_id": idx + 1,
            "name": INDIAN_PINES_CLASS_NAMES[idx] if idx < len(INDIAN_PINES_CLASS_NAMES) else str(idx + 1),
            "accuracy": float(per_class[idx]),
            "support": int(support[idx]),
        }
        for idx in range(class_count)
    ]
    return {
        "loss": float(loss),
        "oa": float(np.mean(labels == predictions)),
        "aa": float(np.mean(per_class)),
        "kappa": float(kappa_from_confusion(conf)),
        "per_class_accuracy": class_metrics,
        "confusion_matrix": conf.tolist(),
    }
