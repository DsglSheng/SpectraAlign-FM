from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .classification_scene import fwhm_from_wavelengths, load_mat_array, normalize_cube, seed_everything


CHANGE_DATASET_ALIASES: dict[str, str] = {
    "hermiston": "hermiston",
    "farmland": "farmland",
    "ba": "bayarea",
    "bay": "bayarea",
    "bayarea": "bayarea",
    "bay_area": "bayarea",
    "sb": "barbara",
    "santa_barbara": "barbara",
    "santabarbara": "barbara",
    "barbara": "barbara",
}


@dataclass(frozen=True)
class ChangeDetectionSceneConfig:
    dataset: str
    display_name: str
    sensor: str
    level: str
    gsd_m: float
    wavelengths: torch.Tensor
    fwhm: torch.Tensor
    metadata_note: str
    t1_filename: str
    t1_key: str
    t2_filename: str
    t2_key: str
    gt_filename: str
    gt_key: str
    channels: int
    height: int
    width: int
    patch_size: int
    train_number: int
    changed_value: int
    unchanged_value: int
    zero_policy: str = "ignore"


class ChangePatchDataset(Dataset):
    def __init__(
        self,
        t1_cube: np.ndarray,
        t2_cube: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        *,
        patch_size: int,
        augment: bool = False,
    ) -> None:
        if t1_cube.shape != t2_cube.shape:
            raise ValueError(f"t1/t2 shapes are inconsistent: {t1_cube.shape} vs {t2_cube.shape}.")
        if t1_cube.ndim != 3:
            raise ValueError(f"Expected HWC cubes, got {t1_cube.shape}.")
        if labels.shape != t1_cube.shape[:2]:
            raise ValueError(f"Label shape {labels.shape} is inconsistent with cube {t1_cube.shape}.")
        self.t1_cube = t1_cube.astype(np.float32, copy=False)
        self.t2_cube = t2_cube.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.patch_size = int(patch_size)
        self.augment = bool(augment)
        self.height, self.width, _ = self.t1_cube.shape
        self.margin = self.patch_size // 2
        pad_width = ((self.margin, self.margin), (self.margin, self.margin), (0, 0))
        self.padded_t1 = np.pad(self.t1_cube, pad_width, mode="reflect")
        self.padded_t2 = np.pad(self.t2_cube, pad_width, mode="reflect")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        flat_index = int(self.indices[index])
        row = flat_index // self.width
        col = flat_index % self.width
        t1 = self.padded_t1[row : row + self.patch_size, col : col + self.patch_size, :]
        t2 = self.padded_t2[row : row + self.patch_size, col : col + self.patch_size, :]
        label = int(self.labels[row, col])
        if self.augment:
            if random.random() < 0.5:
                t1 = np.flip(t1, axis=0)
                t2 = np.flip(t2, axis=0)
            if random.random() < 0.5:
                t1 = np.flip(t1, axis=1)
                t2 = np.flip(t2, axis=1)
            rotations = random.randint(0, 3)
            if rotations:
                t1 = np.rot90(t1, k=rotations, axes=(0, 1))
                t2 = np.rot90(t2, k=rotations, axes=(0, 1))
        return {
            "t1": torch.from_numpy(t1.transpose(2, 0, 1).copy()).float(),
            "t2": torch.from_numpy(t2.transpose(2, 0, 1).copy()).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "index": torch.tensor(flat_index, dtype=torch.long),
        }


def canonicalize_change_name(dataset: str) -> str:
    normalized = str(dataset).strip().lower().replace("-", "_")
    return CHANGE_DATASET_ALIASES.get(normalized, normalized)


def change_default_wavelengths(num_bands: int) -> torch.Tensor:
    return torch.linspace(400.0, 2500.0, int(num_bands), dtype=torch.float32)


def resolve_change_scene_config(dataset: str) -> ChangeDetectionSceneConfig:
    normalized = canonicalize_change_name(dataset)
    if normalized == "hermiston":
        wavelengths = change_default_wavelengths(154)
        return ChangeDetectionSceneConfig(
            dataset=normalized,
            display_name="Hermiston",
            sensor="others",
            level="surface",
            gsd_m=1.0,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="HyperSIGMA Hermiston CD scene with approximate AVIRIS-like wavelengths.",
            t1_filename="Sa1.mat",
            t1_key="T1",
            t2_filename="Sa2.mat",
            t2_key="T2",
            gt_filename="SaGT.mat",
            gt_key="GT",
            channels=154,
            height=307,
            width=241,
            patch_size=5,
            train_number=500,
            changed_value=1,
            unchanged_value=2,
        )
    if normalized == "farmland":
        wavelengths = change_default_wavelengths(155)
        return ChangeDetectionSceneConfig(
            dataset=normalized,
            display_name="Farmland",
            sensor="others",
            level="surface",
            gsd_m=1.0,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="HyperSIGMA Farmland CD scene with approximate AVIRIS-like wavelengths.",
            t1_filename="Farm1.mat",
            t1_key="imgh",
            t2_filename="Farm2.mat",
            t2_key="imghl",
            gt_filename="GTChina1.mat",
            gt_key="label",
            channels=155,
            height=450,
            width=140,
            patch_size=5,
            train_number=500,
            changed_value=2,
            unchanged_value=1,
            zero_policy="changed",
        )
    if normalized == "bayarea":
        wavelengths = change_default_wavelengths(224)
        return ChangeDetectionSceneConfig(
            dataset=normalized,
            display_name="BayArea",
            sensor="avng",
            level="surface",
            gsd_m=1.0,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="HyperSIGMA BayArea CD scene with approximate AVIRIS wavelengths.",
            t1_filename="BayArea.mat",
            t1_key="T1",
            t2_filename="BayArea.mat",
            t2_key="T2",
            gt_filename="BayArea.mat",
            gt_key="GT",
            channels=224,
            height=600,
            width=500,
            patch_size=15,
            train_number=500,
            changed_value=1,
            unchanged_value=2,
            zero_policy="ignore",
        )
    if normalized == "barbara":
        wavelengths = change_default_wavelengths(224)
        return ChangeDetectionSceneConfig(
            dataset=normalized,
            display_name="Santa Barbara",
            sensor="avng",
            level="surface",
            gsd_m=1.0,
            wavelengths=wavelengths,
            fwhm=fwhm_from_wavelengths(wavelengths),
            metadata_note="HyperSIGMA Santa Barbara CD scene with approximate AVIRIS wavelengths.",
            t1_filename="Barbara.mat",
            t1_key="T1",
            t2_filename="Barbara.mat",
            t2_key="T2",
            gt_filename="Barbara.mat",
            gt_key="GT",
            channels=224,
            height=984,
            width=740,
            patch_size=15,
            train_number=500,
            changed_value=1,
            unchanged_value=2,
            zero_policy="ignore",
        )
    raise ValueError(f"Unsupported change detection dataset {dataset!r}.")


def resolve_change_file(data_root: Path, config: ChangeDetectionSceneConfig, filename: str) -> Path:
    candidates = [
        data_root / config.dataset / filename,
        data_root / config.display_name / filename,
        data_root / filename,
    ]
    if config.dataset == "barbara":
        candidates.append(data_root / "sb" / filename)
    if config.dataset == "bayarea":
        candidates.append(data_root / "ba" / filename)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def to_hwc_cube(array: np.ndarray) -> np.ndarray:
    cube = np.asarray(array)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got {cube.shape}.")
    if cube.shape[-1] <= 512 and cube.shape[0] > 16 and cube.shape[1] > 16:
        return cube.astype(np.float32, copy=False)
    if cube.shape[0] <= 512 and cube.shape[1] > 16 and cube.shape[2] > 16:
        cube = np.transpose(cube, (1, 2, 0))
    return cube.astype(np.float32, copy=False)


def normalize_change_labels(raw: np.ndarray, config: ChangeDetectionSceneConfig) -> np.ndarray:
    values = np.asarray(raw)
    while values.ndim > 2 and 1 in values.shape:
        values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D label map, got {values.shape}.")
    labels = np.full(values.shape, -1, dtype=np.int64)
    labels[values == config.unchanged_value] = 0
    labels[values == config.changed_value] = 1
    if config.zero_policy == "changed":
        labels[values == 0] = 1
    elif config.zero_policy == "unchanged":
        labels[values == 0] = 0
    elif config.zero_policy != "ignore":
        raise ValueError(f"Unsupported zero_policy={config.zero_policy!r}.")
    return labels


def load_change_scene(
    *,
    data_root: Path,
    config: ChangeDetectionSceneConfig,
    normalization: str = "paired_minmax",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    t1_path = resolve_change_file(data_root, config, config.t1_filename)
    t2_path = resolve_change_file(data_root, config, config.t2_filename)
    gt_path = resolve_change_file(data_root, config, config.gt_filename)
    missing = [str(path) for path in (t1_path, t2_path, gt_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing change detection data files: " + ", ".join(missing))

    t1 = to_hwc_cube(load_mat_array(t1_path, config.t1_key, expected_ndim=3))
    t2 = to_hwc_cube(load_mat_array(t2_path, config.t2_key, expected_ndim=3))
    labels = normalize_change_labels(load_mat_array(gt_path, config.gt_key, expected_ndim=(2, 3)), config)
    if t1.shape != t2.shape:
        raise ValueError(f"t1/t2 shapes are inconsistent: {t1.shape} vs {t2.shape}.")
    if t1.shape[:2] != labels.shape:
        raise ValueError(f"Cube/label shapes are inconsistent: cube={t1.shape}, labels={labels.shape}.")

    mode = str(normalization).strip().lower()
    if mode == "paired_minmax":
        first = t1.astype(np.float32, copy=False)
        second = t2.astype(np.float32, copy=False)
        mins = np.minimum(first.min(axis=(0, 1), keepdims=True), second.min(axis=(0, 1), keepdims=True))
        maxs = np.maximum(first.max(axis=(0, 1), keepdims=True), second.max(axis=(0, 1), keepdims=True))
        scale = np.maximum(maxs - mins, 1.0e-6)
        t1 = ((first - mins) / scale).astype(np.float32, copy=False)
        t2 = ((second - mins) / scale).astype(np.float32, copy=False)
        normalization_info = {"mode": mode}
    else:
        t1, t1_info = normalize_cube(t1, mode)
        t2, t2_info = normalize_cube(t2, mode)
        normalization_info = {"mode": mode, "t1": t1_info, "t2": t2_info}
    return t1, t2, labels, normalization_info


def make_change_splits(
    labels: np.ndarray,
    *,
    train_number: int,
    split_seed: int,
    split_rng: str = "default",
) -> dict[str, np.ndarray | dict[str, int]]:
    changed = np.flatnonzero(labels.reshape(-1) == 1)
    unchanged = np.flatnonzero(labels.reshape(-1) == 0)
    if changed.size < train_number or unchanged.size < train_number:
        raise ValueError(
            f"Not enough labeled pixels for train_number={train_number}: "
            f"changed={changed.size}, unchanged={unchanged.size}."
        )
    rng_name = str(split_rng).strip().lower()
    if rng_name == "legacy":
        # HyperSIGMA uses np.random.seed(1) followed by np.random.choice on
        # unchanged pixels first and changed pixels second.
        rng = np.random.RandomState(int(split_seed))
        train_unchanged = unchanged[rng.choice(unchanged.size, size=int(train_number), replace=False)]
        train_changed = changed[rng.choice(changed.size, size=int(train_number), replace=False)]
        train_indices = np.concatenate([np.sort(train_changed), np.sort(train_unchanged)]).astype(np.int64)
    elif rng_name == "default":
        rng = np.random.default_rng(int(split_seed))
        train_changed = rng.choice(changed, size=int(train_number), replace=False)
        train_unchanged = rng.choice(unchanged, size=int(train_number), replace=False)
        train_indices = np.concatenate([train_unchanged, train_changed]).astype(np.int64)
        rng.shuffle(train_indices)
    else:
        raise ValueError(f"Unsupported split_rng={split_rng!r}.")
    eval_indices = np.flatnonzero(labels.reshape(-1) >= 0).astype(np.int64)
    return {
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "summary": {
            "train_changed": int(train_changed.size),
            "train_unchanged": int(train_unchanged.size),
            "eval_changed": int(changed.size),
            "eval_unchanged": int(unchanged.size),
            "eval_valid": int(eval_indices.size),
            "ignored": int(np.sum(labels.reshape(-1) < 0)),
            "split_seed": int(split_seed),
            "split_rng": rng_name,
        },
    }


def build_change_detection_scene(
    *,
    data_root: Path,
    config: ChangeDetectionSceneConfig,
    batch_size: int,
    num_workers: int,
    train_number: int | None = None,
    split_seed: int = 1,
    split_rng: str = "default",
    augment: bool = True,
    normalization: str = "paired_minmax",
) -> dict[str, Any]:
    t1, t2, labels, normalization_info = load_change_scene(
        data_root=data_root,
        config=config,
        normalization=normalization,
    )
    resolved_train_number = int(config.train_number if train_number is None else train_number)
    splits = make_change_splits(
        labels,
        train_number=resolved_train_number,
        split_seed=split_seed,
        split_rng=split_rng,
    )
    common = dict(batch_size=int(batch_size), num_workers=int(num_workers), pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        ChangePatchDataset(
            t1,
            t2,
            labels,
            np.asarray(splits["train_indices"]),
            patch_size=config.patch_size,
            augment=augment,
        ),
        shuffle=True,
        drop_last=False,
        **common,
    )
    eval_loader = DataLoader(
        ChangePatchDataset(
            t1,
            t2,
            labels,
            np.asarray(splits["eval_indices"]),
            patch_size=config.patch_size,
            augment=False,
        ),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return {
        "config": config,
        "t1": t1,
        "t2": t2,
        "labels": labels,
        "channels": int(t1.shape[-1]),
        "height": int(t1.shape[0]),
        "width": int(t1.shape[1]),
        "patch_size": int(config.patch_size),
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "train_indices": np.asarray(splits["train_indices"], dtype=np.int64),
        "eval_indices": np.asarray(splits["eval_indices"], dtype=np.int64),
        "summary": {
            "dataset": config.dataset,
            "display_name": config.display_name,
            "sensor": config.sensor,
            "level": config.level,
            "gsd_m": float(config.gsd_m),
            "metadata_note": config.metadata_note,
            "normalization": normalization_info,
            "patch_size": int(config.patch_size),
            "train_number": resolved_train_number,
            **splits["summary"],
        },
    }


__all__ = [
    "CHANGE_DATASET_ALIASES",
    "ChangeDetectionSceneConfig",
    "ChangePatchDataset",
    "build_change_detection_scene",
    "canonicalize_change_name",
    "load_change_scene",
    "resolve_change_scene_config",
    "seed_everything",
]
