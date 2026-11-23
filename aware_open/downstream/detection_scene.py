from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader, Dataset

from .classification_scene import (
    fwhm_from_wavelengths,
    load_mat_array,
    normalize_cube,
    seed_everything,
    tile_scene,
)


DETECTION_DATASET_ALIASES: dict[str, str] = {
    "mosaic": "mosaic",
    "aviris": "aviris",
    "cri": "cri",
    "pavia": "pavia",
    "reno_urban": "renourban",
    "renourban": "renourban",
    "viareggio": "viareggio",
}


def _placeholder_wavelengths() -> torch.Tensor:
    return torch.tensor([550.0], dtype=torch.float32)


def _viareggio_binned_wavelengths() -> torch.Tensor:
    return torch.linspace(388.0248, 530.4873, 127, dtype=torch.float32)


@dataclass(frozen=True)
class DetectionSceneConfig:
    task: str
    dataset: str
    sensor: str
    level: str
    gsd_m: float
    wavelengths: torch.Tensor
    fwhm: torch.Tensor
    metadata_note: str
    cube_filename: str
    cube_key: str
    gt_filename: str
    gt_key: str
    coarse_filename: str | None
    coarse_key: str | None
    img_size: int
    overlap_size: int


class DetectionTileDataset(Dataset):
    def __init__(
        self,
        tiles: np.ndarray,
        labels: np.ndarray,
        *,
        prior_tiles: np.ndarray | None = None,
        target_spectrum: np.ndarray | None = None,
        augment: bool = False,
    ) -> None:
        if tiles.ndim != 4:
            raise ValueError(f"Expected tiles shaped (N, H, W, C), got {tiles.shape}.")
        if labels.ndim != 3:
            raise ValueError(f"Expected labels shaped (N, H, W), got {labels.shape}.")
        if tiles.shape[0] != labels.shape[0] or tiles.shape[1:3] != labels.shape[1:3]:
            raise ValueError("Tile and label shapes are inconsistent.")
        if prior_tiles is not None:
            if prior_tiles.ndim == 3:
                prior_tiles = prior_tiles[..., None]
            if prior_tiles.ndim != 4:
                raise ValueError(f"Expected prior tiles shaped (N, H, W, P), got {prior_tiles.shape}.")
            if prior_tiles.shape[:3] != tiles.shape[:3] or prior_tiles.shape[0] != tiles.shape[0]:
                raise ValueError("Prior tile shapes are inconsistent with image tiles.")
        self.tiles = tiles.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.prior_tiles = None if prior_tiles is None else prior_tiles.astype(np.float32, copy=False)
        self.target_spectrum = None if target_spectrum is None else np.asarray(target_spectrum, dtype=np.float32).reshape(-1)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return int(self.tiles.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tile = self.tiles[index]
        label = self.labels[index]
        prior = None if self.prior_tiles is None else self.prior_tiles[index]
        if self.augment:
            if np.random.rand() < 0.5:
                tile = np.flip(tile, axis=0)
                label = np.flip(label, axis=0)
                if prior is not None:
                    prior = np.flip(prior, axis=0)
            if np.random.rand() < 0.5:
                tile = np.flip(tile, axis=1)
                label = np.flip(label, axis=1)
                if prior is not None:
                    prior = np.flip(prior, axis=1)
            rotations = int(np.random.randint(0, 4))
            if rotations:
                tile = np.rot90(tile, k=rotations, axes=(0, 1))
                label = np.rot90(label, k=rotations, axes=(0, 1))
                if prior is not None:
                    prior = np.rot90(prior, k=rotations, axes=(0, 1))

        sample = {
            "image": torch.from_numpy(tile.transpose(2, 0, 1).copy()).float(),
            "label": torch.from_numpy(label.copy()).long(),
        }
        if prior is not None:
            sample["prior"] = torch.from_numpy(prior.transpose(2, 0, 1).copy()).float()
        if self.target_spectrum is not None:
            sample["target_spectrum"] = torch.from_numpy(self.target_spectrum.copy()).float()
        return sample


def canonicalize_detection_name(dataset: str) -> str:
    normalized = str(dataset).strip().lower().replace("-", "_")
    return DETECTION_DATASET_ALIASES.get(normalized, normalized)


def detection_default_wavelengths(task: str, dataset: str, num_bands: int) -> torch.Tensor:
    normalized_task = str(task).strip().lower()
    normalized_dataset = canonicalize_detection_name(dataset)
    if normalized_task == "htd":
        if normalized_dataset == "aviris":
            return torch.linspace(400.0, 2500.0, int(num_bands), dtype=torch.float32)
        return torch.linspace(400.0, 2450.0, int(num_bands), dtype=torch.float32)
    if normalized_task == "had":
        if normalized_dataset == "pavia":
            return torch.linspace(430.0, 860.0, int(num_bands), dtype=torch.float32)
        return torch.linspace(400.0, 1000.0, int(num_bands), dtype=torch.float32)
    raise ValueError(f"Unsupported detection task {task!r}.")


def resolve_detection_scene_config(task: str, dataset: str) -> DetectionSceneConfig:
    normalized_task = str(task).strip().lower()
    normalized_dataset = canonicalize_detection_name(dataset)
    if normalized_task == "htd":
        if normalized_dataset == "mosaic":
            wavelengths = detection_default_wavelengths(normalized_task, normalized_dataset, 128)
            return DetectionSceneConfig(
                task=normalized_task,
                dataset=normalized_dataset,
                sensor="others",
                level="surface",
                gsd_m=1.0,
                wavelengths=wavelengths,
                fwhm=fwhm_from_wavelengths(wavelengths),
                metadata_note="SHARE2012 Mosaic-style target detection scene with approximate SpecTIR wavelengths.",
                cube_filename="mosaic.mat",
                cube_key="X",
                gt_filename="mosaic.mat",
                gt_key="TL",
                coarse_filename="coarse_det/Mosaic.mat",
                coarse_key="Mosaic",
                img_size=32,
                overlap_size=16,
            )
        if normalized_dataset == "aviris":
            wavelengths = detection_default_wavelengths(normalized_task, normalized_dataset, 189)
            return DetectionSceneConfig(
                task=normalized_task,
                dataset=normalized_dataset,
                sensor="avng",
                level="surface",
                gsd_m=3.5,
                wavelengths=wavelengths,
                fwhm=fwhm_from_wavelengths(wavelengths),
                metadata_note="AVIRIS target detection scene with approximate AVIRIS wavelengths.",
                cube_filename="aviris.mat",
                cube_key="X",
                gt_filename="aviris.mat",
                gt_key="TL",
                coarse_filename="coarse_det/AVIRIS.mat",
                coarse_key="AVIRIS",
                img_size=32,
                overlap_size=16,
            )
        if normalized_dataset == "renourban":
            wavelengths = _placeholder_wavelengths()
            return DetectionSceneConfig(
                task=normalized_task,
                dataset=normalized_dataset,
                sensor="others",
                level="surface",
                gsd_m=1.0,
                wavelengths=wavelengths,
                fwhm=fwhm_from_wavelengths(wavelengths),
                metadata_note=(
                    "RenoUrban target detection scene expecting a standardized renourban.mat "
                    "with cube/label keys X and TL."
                ),
                cube_filename="renourban.mat",
                cube_key="X",
                gt_filename="renourban.mat",
                gt_key="TL",
                coarse_filename="coarse_det/RenoUrban.mat",
                coarse_key="RenoUrban",
                img_size=32,
                overlap_size=16,
            )
    if normalized_task == "had":
        if normalized_dataset == "cri":
            wavelengths = detection_default_wavelengths(normalized_task, normalized_dataset, 46)
            return DetectionSceneConfig(
                task=normalized_task,
                dataset=normalized_dataset,
                sensor="others",
                level="unknown",
                gsd_m=1.0,
                wavelengths=wavelengths,
                fwhm=fwhm_from_wavelengths(wavelengths),
                metadata_note="Cri anomaly detection scene with approximate Nuance visible-range wavelengths.",
                cube_filename="Nuance_Cri_400_400_46_1254.mat",
                cube_key="hsi",
                gt_filename="Nuance_Cri_400_400_46_1254.mat",
                gt_key="hsi_gt",
                coarse_filename="coarse_det/Cri_coarse_det_map.mat",
                coarse_key="show",
                img_size=32,
                overlap_size=16,
            )
        if normalized_dataset == "pavia":
            wavelengths = detection_default_wavelengths(normalized_task, normalized_dataset, 102)
            return DetectionSceneConfig(
                task=normalized_task,
                dataset=normalized_dataset,
                sensor="rosis",
                level="surface",
                gsd_m=1.3,
                wavelengths=wavelengths,
                fwhm=fwhm_from_wavelengths(wavelengths),
                metadata_note="Pavia anomaly detection scene with approximate ROSIS wavelengths.",
                cube_filename="Paiva_108_120_102_43.mat",
                cube_key="hsi",
                gt_filename="Paiva_108_120_102_43.mat",
                gt_key="hsi_gt",
                coarse_filename="coarse_det/Pavia_coarse_det_map.mat",
                coarse_key="show",
                img_size=32,
                overlap_size=16,
            )
        if normalized_dataset == "viareggio":
            wavelengths = _viareggio_binned_wavelengths()
            return DetectionSceneConfig(
                task=normalized_task,
                dataset=normalized_dataset,
                sensor="others",
                level="surface",
                gsd_m=1.0,
                wavelengths=wavelengths,
                fwhm=fwhm_from_wavelengths(wavelengths),
                metadata_note=(
                    "Viareggio anomaly detection scene expecting a standardized viareggio.mat "
                    "with cube/label keys hsi and hsi_gt."
                ),
                cube_filename="viareggio.mat",
                cube_key="hsi",
                gt_filename="viareggio.mat",
                gt_key="hsi_gt",
                coarse_filename="coarse_det/Viareggio_coarse_det_map.mat",
                coarse_key="show",
                img_size=32,
                overlap_size=16,
            )
    raise ValueError(f"Unsupported detection pair task={task!r}, dataset={dataset!r}.")


def _resolve_path(root: Path, override: Path | None, default_name: str) -> Path:
    if override is not None:
        return override if override.is_absolute() else (root / override)
    return root / default_name


def _resolve_optional_path(root: Path, override: Path | None, default_name: str | None) -> Path | None:
    if override is not None:
        return override if override.is_absolute() else (root / override)
    if default_name is None:
        return None
    return root / default_name


def _squeeze_map(array: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(array)
    while value.ndim > 2 and 1 in value.shape:
        value = np.squeeze(value)
    if value.ndim != 2:
        raise ValueError(f"{name} must be 2D after squeezing, got shape {value.shape}.")
    return value


def _to_hwc_cube(array: np.ndarray) -> np.ndarray:
    cube = np.asarray(array)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D hyperspectral cube, got shape {cube.shape}.")
    if cube.shape[-1] <= 256 and cube.shape[0] > 16 and cube.shape[1] > 16:
        return cube.astype(np.float32, copy=False)
    if cube.shape[0] <= 256 and cube.shape[1] > 16 and cube.shape[2] > 16:
        cube = np.transpose(cube, (1, 2, 0))
    return cube.astype(np.float32, copy=False)


def load_detection_scene(
    *,
    data_root: Path,
    config: DetectionSceneConfig,
    cube_path: Path | None = None,
    gt_path: Path | None = None,
    coarse_path: Path | None = None,
    allow_missing_coarse: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    resolved_cube = _resolve_path(data_root, cube_path, config.cube_filename)
    resolved_gt = _resolve_path(data_root, gt_path, config.gt_filename)
    resolved_coarse = _resolve_optional_path(data_root, coarse_path, config.coarse_filename)
    for path in (resolved_cube, resolved_gt):
        if not path.exists():
            raise FileNotFoundError(f"Missing detection data file: {path}")

    cube = _to_hwc_cube(load_mat_array(resolved_cube, config.cube_key, expected_ndim=3))
    gt = _squeeze_map(load_mat_array(resolved_gt, config.gt_key, expected_ndim=(2, 3)), name="gt")
    gt = (gt > 0).astype(np.int64, copy=False)

    coarse = None
    if resolved_coarse is not None and resolved_coarse.exists():
        coarse_key = config.coarse_key if config.coarse_key is not None else None
        coarse = _squeeze_map(load_mat_array(resolved_coarse, coarse_key, expected_ndim=(2, 3)), name="coarse")
        coarse = coarse.astype(np.float32, copy=False)
    elif not allow_missing_coarse:
        raise FileNotFoundError(f"Missing detection coarse map: {resolved_coarse}")

    if cube.shape[:2] != gt.shape:
        raise ValueError(f"Detection scene shapes are inconsistent: cube={cube.shape}, gt={gt.shape}.")
    if coarse is not None and cube.shape[:2] != coarse.shape:
        raise ValueError(
            "Detection scene shapes are inconsistent: "
            f"cube={cube.shape}, gt={gt.shape}, coarse={coarse.shape}."
        )
    return cube, gt, coarse


def standardize_score_map(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float32)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if not np.isfinite(lower) or not np.isfinite(upper) or abs(upper - lower) < 1.0e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lower) / (upper - lower)).astype(np.float32, copy=False)


def compute_auc_scores(output: np.ndarray, target: np.ndarray) -> dict[str, float]:
    y_true = np.reshape(target, (-1, 1), order="F")
    y_score = np.reshape(output, (-1, 1), order="F")
    fpr, tpr, threshold = sk_metrics.roc_curve(y_true, y_score, drop_intermediate=False)
    fpr = fpr[1:]
    tpr = tpr[1:]
    threshold = threshold[1:]
    auc1 = float(round(sk_metrics.auc(fpr, tpr), 4))
    auc2 = float(round(sk_metrics.auc(threshold, fpr), 4))
    auc3 = float(round(sk_metrics.auc(threshold, tpr), 4))
    auc4 = float(round(auc1 + auc3 - auc2, 4))
    auc5 = float(round(auc3 / max(auc2, 1.0e-12), 4))
    return {
        "auc": auc1,
        "auc_threshold_fpr": auc2,
        "auc_threshold_tpr": auc3,
        "auc_composite": auc4,
        "auc_ratio": auc5,
    }


def build_pseudo_label_map(
    coarse_map: np.ndarray,
    *,
    bg_ratio: float = 0.30,
    fg_ratio: float = 0.0015,
    ignore_value: int = 255,
) -> np.ndarray:
    flattened = np.asarray(coarse_map, dtype=np.float32).reshape(-1)
    total = int(flattened.shape[0])
    if total <= 1:
        raise ValueError("coarse_map must contain at least two pixels.")
    if not 0.0 <= bg_ratio < 1.0:
        raise ValueError(f"bg_ratio must be in [0, 1), got {bg_ratio}.")
    if not 0.0 <= fg_ratio < 1.0:
        raise ValueError(f"fg_ratio must be in [0, 1), got {fg_ratio}.")
    bg_count = max(1, int(round(float(bg_ratio) * float(total)))) if bg_ratio > 0.0 else 0
    bg_count = min(bg_count, total - 1)
    remaining = max(total - bg_count, 1)
    fg_count = max(1, int(round(float(fg_ratio) * float(total)))) if fg_ratio > 0.0 else 0
    fg_count = min(fg_count, remaining)

    order = np.argsort(flattened)
    labels = np.full((total,), int(ignore_value), dtype=np.int64)
    if bg_count > 0:
        labels[order[:bg_count]] = 0
    if fg_count > 0:
        labels[order[-fg_count:]] = 1
    return labels.reshape(coarse_map.shape)


def cosin_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    norm_x = np.sqrt(np.sum(x**2, axis=-1))
    norm_y = np.sqrt(np.sum(y**2, axis=-1))
    x_y = np.sum(np.multiply(x, y), axis=-1)
    similarity = np.clip(x_y / np.maximum(norm_x * norm_y, 1.0e-12), -1.0, 1.0)
    return np.arccos(similarity)


def target_spectrum_from_map(cube: np.ndarray, gt_map: np.ndarray, *, spectrum_type: int = 0) -> np.ndarray:
    positives = np.asarray(cube, dtype=np.float32)[np.asarray(gt_map) == 1]
    if positives.size == 0:
        raise ValueError("Ground-truth map does not contain any positive target pixel.")
    avg_target = np.mean(positives, axis=0)
    if spectrum_type == 0:
        return avg_target.astype(np.float32, copy=False)
    if spectrum_type == 1:
        distance = np.sqrt(np.sum((positives - avg_target[None, :]) ** 2, axis=-1))
        keep = max(1, int(positives.shape[0] * 0.8))
        return np.mean(positives[np.argsort(distance)[:keep]], axis=0).astype(np.float32, copy=False)
    if spectrum_type == 2:
        distance = cosin_similarity(positives, avg_target[None, :])
        keep = max(1, int(positives.shape[0] * 0.8))
        return np.mean(positives[np.argsort(distance)[:keep]], axis=0).astype(np.float32, copy=False)
    if spectrum_type == 3:
        norms = np.sqrt(np.sum(positives**2, axis=-1))
        return positives[np.argsort(norms)[positives.shape[0] // 2]].astype(np.float32, copy=False)
    if spectrum_type == 4:
        median_idx = positives.shape[0] // 2
        distance = np.mean(cosin_similarity(positives[:, None, :], positives[None, :, :]), axis=1)
        return positives[np.argsort(distance)[median_idx]].astype(np.float32, copy=False)
    if spectrum_type == 5:
        distance = np.mean(np.sqrt(np.sum((positives[:, None, :] - positives[None, :, :]) ** 2, axis=-1)), axis=1)
        return positives[np.argmin(distance)].astype(np.float32, copy=False)
    if spectrum_type == 6:
        distance = np.mean(cosin_similarity(positives[:, None, :], positives[None, :, :]), axis=1)
        return positives[np.argmin(distance)].astype(np.float32, copy=False)
    if spectrum_type == 7:
        distance = np.sqrt(np.sum((positives - avg_target[None, :]) ** 2, axis=-1))
        return positives[np.argmin(distance)].astype(np.float32, copy=False)
    if spectrum_type == 8:
        distance = cosin_similarity(positives, avg_target[None, :])
        return positives[np.argmin(distance)].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported target spectrum type {spectrum_type}.")


def _ridge_inverse(matrix: np.ndarray, ridge: float = 1.0e-3) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}.")
    dimension = int(matrix.shape[0])
    trace = float(np.trace(matrix)) / max(dimension, 1)
    ridge_value = max(float(ridge) * max(trace, 1.0), 1.0e-6)
    return np.linalg.pinv(matrix + ridge_value * np.eye(dimension, dtype=np.float64))


def compute_cem_map(cube: np.ndarray, target_spectrum: np.ndarray, *, ridge: float = 1.0e-3) -> np.ndarray:
    flat = np.asarray(cube, dtype=np.float64).reshape(-1, cube.shape[-1])
    target = np.asarray(target_spectrum, dtype=np.float64).reshape(-1)
    covariance = (flat.T @ flat) / max(int(flat.shape[0]), 1)
    precision = _ridge_inverse(covariance, ridge=ridge)
    weight = precision @ target
    denominator = float(target.T @ weight)
    weight = weight / max(abs(denominator), 1.0e-12)
    scores = flat @ weight
    return scores.reshape(cube.shape[:2]).astype(np.float32, copy=False)


def compute_sam_map(cube: np.ndarray, target_spectrum: np.ndarray) -> np.ndarray:
    flat = np.asarray(cube, dtype=np.float32).reshape(-1, cube.shape[-1])
    target = np.asarray(target_spectrum, dtype=np.float32).reshape(1, -1)
    score = 1.0 - (cosin_similarity(flat, target) / np.pi)
    return score.reshape(cube.shape[:2]).astype(np.float32, copy=False)


def compute_rx_map(cube: np.ndarray, *, ridge: float = 1.0e-3) -> np.ndarray:
    flat = np.asarray(cube, dtype=np.float64).reshape(-1, cube.shape[-1])
    mean = np.mean(flat, axis=0, keepdims=True)
    centered = flat - mean
    covariance = (centered.T @ centered) / max(int(flat.shape[0]) - 1, 1)
    precision = _ridge_inverse(covariance, ridge=ridge)
    scores = np.einsum("nc,cd,nd->n", centered, precision, centered, optimize=True)
    return scores.reshape(cube.shape[:2]).astype(np.float32, copy=False)


def compute_local_residual_map(cube: np.ndarray, *, window_size: int = 9) -> np.ndarray:
    if window_size <= 1 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer > 1, got {window_size}.")
    tensor = torch.from_numpy(np.asarray(cube, dtype=np.float32).transpose(2, 0, 1)).unsqueeze(0)
    local_mean = F.avg_pool2d(tensor, kernel_size=window_size, stride=1, padding=window_size // 2)
    residual = (tensor - local_mean).pow(2).mean(dim=1)
    return residual.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def build_auto_detection_priors(
    task: str,
    cube: np.ndarray,
    *,
    target_spectrum: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normalized_task = str(task).strip().lower()
    if normalized_task == "htd":
        if target_spectrum is None:
            raise ValueError("target_spectrum is required for HTD automatic priors.")
        cem = standardize_score_map(compute_cem_map(cube, target_spectrum))
        sam = standardize_score_map(compute_sam_map(cube, target_spectrum))
        coarse = standardize_score_map(0.65 * cem + 0.35 * sam)
        priors = np.stack([cem, sam, coarse], axis=-1).astype(np.float32, copy=False)
        return coarse, priors, {"coarse_source": "auto", "methods": ["cem", "sam", "blend"]}
    if normalized_task == "had":
        rx = standardize_score_map(compute_rx_map(cube))
        residual = standardize_score_map(compute_local_residual_map(cube))
        coarse = standardize_score_map(0.7 * rx + 0.3 * residual)
        priors = np.stack([rx, residual, coarse], axis=-1).astype(np.float32, copy=False)
        return coarse, priors, {"coarse_source": "auto", "methods": ["rx", "local_residual", "blend"]}
    raise ValueError(f"Unsupported detection task {task!r}.")


def tile_auxiliary_map(values: np.ndarray, *, img_size: int, overlap_size: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Auxiliary map must be 2D or 3D, got shape {array.shape}.")
    dummy = np.zeros(array.shape[:2], dtype=np.int64)
    tiles, _, starts = tile_scene(array, dummy, img_size=img_size, overlap_size=overlap_size)
    return tiles, starts


def _padded_extent(starts: list[tuple[int, int]], img_size: int, fallback_shape: tuple[int, int]) -> tuple[int, int]:
    if not starts:
        return int(fallback_shape[0]), int(fallback_shape[1])
    return (
        int(max(row for row, _ in starts) + img_size),
        int(max(col for _, col in starts) + img_size),
    )


def build_detection_scene(
    *,
    data_root: Path,
    config: DetectionSceneConfig,
    cube_path: Path | None = None,
    gt_path: Path | None = None,
    coarse_path: Path | None = None,
    img_size: int | None = None,
    overlap_size: int | None = None,
    batch_size: int,
    num_workers: int,
    augment: bool,
    normalization: str = "mean_std",
    pseudo_bg_ratio: float = 0.30,
    pseudo_fg_ratio: float = 0.0015,
    target_spectrum_type: int = 0,
    target_spectrum_source: str = "gt",
    allow_missing_coarse: bool = True,
    use_auto_priors: bool = True,
) -> dict[str, Any]:
    cube, labels, external_coarse = load_detection_scene(
        data_root=data_root,
        config=config,
        cube_path=cube_path,
        gt_path=gt_path,
        coarse_path=coarse_path,
        allow_missing_coarse=allow_missing_coarse,
    )
    cube, normalization_config = normalize_cube(cube, normalization)
    resolved_img_size = int(config.img_size if img_size is None else img_size)
    resolved_overlap = int(config.overlap_size if overlap_size is None else overlap_size)

    target_spectrum = None
    if config.task == "htd":
        target_spectrum = target_spectrum_from_map(cube, labels, spectrum_type=int(target_spectrum_type))

    auto_coarse = None
    auto_priors = None
    coarse_info = {"coarse_source": "file" if external_coarse is not None else "none", "methods": []}
    if use_auto_priors:
        auto_coarse, auto_priors, coarse_info = build_auto_detection_priors(
            config.task,
            cube,
            target_spectrum=target_spectrum,
        )

    if external_coarse is None:
        if auto_coarse is None:
            raise FileNotFoundError("No coarse detection map was found and automatic prior generation is disabled.")
        coarse = auto_coarse
        prior_maps = auto_priors
        coarse_info["coarse_source"] = "auto"
    else:
        coarse = standardize_score_map(external_coarse)
        prior_list = [coarse[..., None]]
        if auto_priors is not None:
            prior_list.append(auto_priors)
            coarse_info = {
                "coarse_source": "file+auto",
                "methods": ["file_coarse", *coarse_info["methods"]],
            }
        prior_maps = np.concatenate(prior_list, axis=-1).astype(np.float32, copy=False)

    train_gt = build_pseudo_label_map(
        coarse,
        bg_ratio=float(pseudo_bg_ratio),
        fg_ratio=float(pseudo_fg_ratio),
    )

    source_key = str(target_spectrum_source).strip().lower()
    if config.task == "htd":
        if source_key == "gt":
            target_map = labels
        elif source_key == "pseudo":
            target_map = (train_gt == 1).astype(np.int64, copy=False)
        else:
            raise ValueError(f"Unsupported target spectrum source {target_spectrum_source!r}.")
        target_spectrum = target_spectrum_from_map(
            cube,
            target_map,
            spectrum_type=int(target_spectrum_type),
        )

    train_tiles, train_labels, starts = tile_scene(
        cube,
        train_gt,
        img_size=resolved_img_size,
        overlap_size=resolved_overlap,
    )
    eval_tiles, eval_labels, eval_starts = tile_scene(
        cube,
        labels,
        img_size=resolved_img_size,
        overlap_size=resolved_overlap,
    )
    if starts != eval_starts:
        raise RuntimeError("Train and evaluation tiling produced inconsistent start indices.")
    prior_tiles, prior_starts = tile_auxiliary_map(
        prior_maps,
        img_size=resolved_img_size,
        overlap_size=resolved_overlap,
    )
    if starts != prior_starts:
        raise RuntimeError("Auxiliary prior tiling is inconsistent with image tiling.")

    padded_height, padded_width = _padded_extent(starts, resolved_img_size, labels.shape)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        DetectionTileDataset(
            train_tiles,
            train_labels,
            prior_tiles=prior_tiles,
            target_spectrum=target_spectrum,
            augment=augment,
        ),
        shuffle=True,
        drop_last=False,
        **common,
    )
    eval_loader = DataLoader(
        DetectionTileDataset(
            eval_tiles,
            eval_labels,
            prior_tiles=prior_tiles,
            target_spectrum=target_spectrum,
            augment=False,
        ),
        shuffle=False,
        drop_last=False,
        **common,
    )

    valid_train = train_gt != 255
    pseudo_pos = int(np.sum(train_gt == 1))
    pseudo_neg = int(np.sum(train_gt == 0))
    return {
        "config": config,
        "cube": cube,
        "labels": labels,
        "coarse": coarse,
        "prior_maps": prior_maps,
        "pseudo_labels": train_gt,
        "target_spectrum": target_spectrum,
        "height": int(labels.shape[0]),
        "width": int(labels.shape[1]),
        "padded_height": int(padded_height),
        "padded_width": int(padded_width),
        "channels": int(cube.shape[-1]),
        "prior_channels": int(prior_maps.shape[-1]),
        "img_size": resolved_img_size,
        "overlap_size": resolved_overlap,
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "starts": starts,
        "scene_name": f"{config.task}_{config.dataset}",
        "summary": {
            "task": config.task,
            "dataset": config.dataset,
            "sensor": config.sensor,
            "level": config.level,
            "gsd_m": float(config.gsd_m),
            "metadata_note": config.metadata_note,
            "normalization": normalization_config,
            "img_size": resolved_img_size,
            "overlap_size": resolved_overlap,
            "pseudo_bg_ratio": float(pseudo_bg_ratio),
            "pseudo_fg_ratio": float(pseudo_fg_ratio),
            "pseudo_valid_pixels": int(np.sum(valid_train)),
            "pseudo_positive_pixels": pseudo_pos,
            "pseudo_negative_pixels": pseudo_neg,
            "gt_positive_pixels": int(np.sum(labels == 1)),
            "gt_negative_pixels": int(np.sum(labels == 0)),
            "target_spectrum_type": int(target_spectrum_type),
            "target_spectrum_source": source_key if config.task == "htd" else None,
            "coarse_info": coarse_info,
            "prior_channels": int(prior_maps.shape[-1]),
        },
    }


__all__ = [
    "DetectionSceneConfig",
    "DetectionTileDataset",
    "build_auto_detection_priors",
    "build_detection_scene",
    "build_pseudo_label_map",
    "canonicalize_detection_name",
    "compute_auc_scores",
    "compute_cem_map",
    "compute_local_residual_map",
    "compute_rx_map",
    "compute_sam_map",
    "detection_default_wavelengths",
    "load_detection_scene",
    "resolve_detection_scene_config",
    "seed_everything",
    "standardize_score_map",
    "target_spectrum_from_map",
    "tile_auxiliary_map",
]
