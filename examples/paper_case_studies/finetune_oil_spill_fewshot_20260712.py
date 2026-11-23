from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, median_filter
from sklearn import metrics as sk_metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SPECTRAALIGN_DATA_ROOT", REPO_ROOT / "data")).expanduser()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aware_open.downstream import AwareFeatureExtractor, build_downstream_metadata  # noqa: E402
from aware_open.downstream.classification_scene import normalize_cube, tile_scene  # noqa: E402
from aware_open.downstream.detection_scene import (  # noqa: E402
    compute_cem_map,
    compute_rx_map,
    compute_sam_map,
    standardize_score_map,
)
from scripts.finetune_detection import AwareDetectionFusion, build_layer_decay_optimizer  # noqa: E402


DEFAULT_CASE_ROOT = Path(__file__).resolve().parent / "results" / "oil_spill_hosd"
DEFAULT_DATA_DIR = DATA_ROOT / "oil_spill_hosd"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "pretraining/runs/final_pretrain_hyperglobal_dz1_525k_fromscratch_v1/"
    "stage_b_full_joint154535/checkpoints/best_total.pt"
)
DEFAULT_PROTO_EO1 = DATA_ROOT / "prototype_banks/prototype_bank_eo1.pt"
DEFAULT_PROTO_GF5 = DATA_ROOT / "prototype_banks/prototype_bank_gf5.pt"
DEFAULT_PROTO_DZ1_VNIR = DATA_ROOT / "prototype_banks/prototype_bank_dz1_vnir.pt"
DEFAULT_PROTO_DZ1_SWIR = DATA_ROOT / "prototype_banks/prototype_bank_dz1_swir.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot real-world oil-spill case study for Aware-HSI.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_CASE_ROOT / "runs" / "fewshot_gm13_gm18")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_CASE_ROOT / "figures")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prototype-bank-eo1", type=Path, default=DEFAULT_PROTO_EO1)
    parser.add_argument("--prototype-bank-gf5", type=Path, default=DEFAULT_PROTO_GF5)
    parser.add_argument("--prototype-bank-dz1-vnir", type=Path, default=DEFAULT_PROTO_DZ1_VNIR)
    parser.add_argument("--prototype-bank-dz1-swir", type=Path, default=DEFAULT_PROTO_DZ1_SWIR)
    parser.add_argument("--scenes", nargs="+", default=["GM13", "GM18"])
    parser.add_argument("--train-per-class", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--overlap-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--layer-decay-rate", type=float, default=0.90)
    parser.add_argument("--feature-layers", type=int, default=4)
    parser.add_argument("--fusion-channels", type=int, default=128)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--normalization", choices=("none", "minmax", "mean_std"), default="mean_std")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--max-kmeans-pixels", type=int, default=120000)
    parser.add_argument("--save-combined-figure", action="store_true", default=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_hosd_scene(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = sio.loadmat(path)
    if "img" not in payload or "map" not in payload:
        raise KeyError(f"{path} must contain 'img' and 'map' arrays.")
    cube = np.asarray(payload["img"], dtype=np.float32)
    labels = (np.asarray(payload["map"]) > 0).astype(np.uint8)
    if cube.ndim != 3 or labels.ndim != 2 or cube.shape[:2] != labels.shape:
        raise ValueError(f"Unexpected HOSD shapes: cube={cube.shape}, labels={labels.shape}.")
    return cube, labels


def approximate_aviris_wavelengths(num_bands: int) -> np.ndarray:
    return np.linspace(400.0, 2500.0, int(num_bands), dtype=np.float32)


def fwhm_from_wavelengths(wavelengths: np.ndarray) -> np.ndarray:
    values = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    if values.size == 1:
        return np.ones_like(values)
    diffs = np.abs(np.diff(values))
    left = np.concatenate([diffs[:1], diffs])
    right = np.concatenate([diffs, diffs[-1:]])
    return np.maximum((left + right) * 0.5, 1.0).astype(np.float32)


def sample_fewshot_pixels(labels: np.ndarray, *, train_per_class: int, seed: int, output_path: Path) -> dict[str, np.ndarray]:
    if output_path.exists():
        data = np.load(output_path)
        return {"positive": data["positive"], "negative": data["negative"]}
    rng = np.random.default_rng(seed)
    positive = np.argwhere(labels == 1)
    negative = np.argwhere(labels == 0)
    if positive.shape[0] < train_per_class or negative.shape[0] < train_per_class:
        raise ValueError(
            f"Not enough pixels for few-shot sampling: positive={positive.shape[0]}, negative={negative.shape[0]}."
        )
    pos_idx = rng.choice(positive.shape[0], size=train_per_class, replace=False)
    neg_idx = rng.choice(negative.shape[0], size=train_per_class, replace=False)
    result = {
        "positive": positive[pos_idx].astype(np.int64),
        "negative": negative[neg_idx].astype(np.int64),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    return result


def robust_rgb(cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    targets = [650.0, 560.0, 470.0]
    bands = [int(np.argmin(np.abs(wavelengths - target))) for target in targets]
    rgb = cube[..., bands].astype(np.float32)
    output = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        values = rgb[..., channel]
        low, high = np.percentile(values[np.isfinite(values)], [2.0, 98.0])
        if high <= low:
            high = low + 1.0
        output[..., channel] = np.clip((values - low) / (high - low), 0.0, 1.0)
    return output


def make_baseline_maps(
    cube: np.ndarray,
    labels: np.ndarray,
    fewshot: dict[str, np.ndarray],
    *,
    max_kmeans_pixels: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    pos = fewshot["positive"]
    neg = fewshot["negative"]
    target_spectrum = cube[pos[:, 0], pos[:, 1], :].mean(axis=0).astype(np.float32)
    sam = standardize_score_map(compute_sam_map(cube, target_spectrum))
    cem = standardize_score_map(compute_cem_map(cube, target_spectrum))
    rx = standardize_score_map(compute_rx_map(cube))
    blend = standardize_score_map(0.45 * cem + 0.35 * sam + 0.20 * rx)
    maps = {"SAM": sam, "CEM": cem, "RX": rx, "Blend": blend}

    flat = cube.reshape(-1, cube.shape[-1])
    total = flat.shape[0]
    rng = np.random.default_rng(seed)
    sample_count = min(int(max_kmeans_pixels), total)
    sample_index = rng.choice(total, size=sample_count, replace=False)
    pca = PCA(n_components=3, random_state=seed)
    pca.fit(flat[sample_index])
    km = KMeans(n_clusters=2, n_init=10, random_state=seed)
    km.fit(pca.transform(flat[sample_index]))
    labels_flat = np.empty(total, dtype=np.int16)
    chunk = 200000
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        labels_flat[start:end] = km.predict(pca.transform(flat[start:end]))
    label_image = labels_flat.reshape(labels.shape)
    pos_vote = label_image[pos[:, 0], pos[:, 1]].mean()
    neg_vote = label_image[neg[:, 0], neg[:, 1]].mean()
    if abs(pos_vote - neg_vote) < 1.0e-6:
        kmeans_score = label_image.astype(np.float32)
    elif pos_vote > neg_vote:
        kmeans_score = label_image.astype(np.float32)
    else:
        kmeans_score = 1.0 - label_image.astype(np.float32)
    maps["PCA-KMeans"] = standardize_score_map(kmeans_score)
    return maps, target_spectrum


def orient_score_with_train(score: np.ndarray, labels: np.ndarray, fewshot: dict[str, np.ndarray]) -> np.ndarray:
    pos = fewshot["positive"]
    neg = fewshot["negative"]
    pos_mean = float(score[pos[:, 0], pos[:, 1]].mean())
    neg_mean = float(score[neg[:, 0], neg[:, 1]].mean())
    return score if pos_mean >= neg_mean else 1.0 - score


def threshold_from_train(score: np.ndarray, labels: np.ndarray, fewshot: dict[str, np.ndarray]) -> float:
    pos = fewshot["positive"]
    neg = fewshot["negative"]
    rows = np.concatenate([pos[:, 0], neg[:, 0]])
    cols = np.concatenate([pos[:, 1], neg[:, 1]])
    y_true = labels[rows, cols].reshape(-1)
    y_score = score[rows, cols].reshape(-1)
    thresholds = np.unique(y_score)
    if thresholds.size > 256:
        thresholds = np.quantile(y_score, np.linspace(0.0, 1.0, 256))
    best_f1 = -1.0
    best_threshold = 0.5
    for threshold in thresholds:
        pred = (y_score >= threshold).astype(np.uint8)
        f1 = sk_metrics.f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def compute_binary_metrics(score: np.ndarray, labels: np.ndarray, fewshot: dict[str, np.ndarray]) -> dict[str, float]:
    oriented = orient_score_with_train(np.asarray(score, dtype=np.float32), labels, fewshot)
    y_true = labels.reshape(-1).astype(np.uint8)
    y_score = oriented.reshape(-1).astype(np.float32)
    train_threshold = threshold_from_train(oriented, labels, fewshot)
    pred = (y_score >= train_threshold).astype(np.uint8)
    out = {
        "roc_auc": float(sk_metrics.roc_auc_score(y_true, y_score)),
        "average_precision": float(sk_metrics.average_precision_score(y_true, y_score)),
        "train_threshold": float(train_threshold),
        "f1_train_threshold": float(sk_metrics.f1_score(y_true, pred, zero_division=0)),
        "precision_train_threshold": float(sk_metrics.precision_score(y_true, pred, zero_division=0)),
        "recall_train_threshold": float(sk_metrics.recall_score(y_true, pred, zero_division=0)),
        "iou_train_threshold": float(sk_metrics.jaccard_score(y_true, pred, zero_division=0)),
    }
    precision, recall, thresholds = sk_metrics.precision_recall_curve(y_true, y_score)
    if thresholds.size:
        f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1.0e-12)
        best = int(np.argmax(f1))
        out["oracle_best_f1"] = float(f1[best])
        out["oracle_best_threshold"] = float(thresholds[best])
    return out


class FewShotPatchDataset(Dataset):
    def __init__(
        self,
        cube: np.ndarray,
        labels: np.ndarray,
        prior_maps: np.ndarray,
        target_spectrum: np.ndarray,
        coords: np.ndarray,
        *,
        patch_size: int,
        augment: bool,
    ) -> None:
        self.cube = cube.astype(np.float32, copy=False)
        self.labels = labels.astype(np.uint8, copy=False)
        self.prior_maps = prior_maps.astype(np.float32, copy=False)
        self.target_spectrum = np.asarray(target_spectrum, dtype=np.float32).reshape(-1)
        self.coords = coords.astype(np.int64, copy=False)
        self.patch_size = int(patch_size)
        self.radius = self.patch_size // 2
        self.augment = bool(augment)
        pad = ((self.radius, self.radius), (self.radius, self.radius), (0, 0))
        self.cube_pad = np.pad(self.cube, pad, mode="reflect")
        self.prior_pad = np.pad(self.prior_maps, pad, mode="edge")

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row, col = [int(v) for v in self.coords[index]]
        top = row
        left = col
        patch = self.cube_pad[top : top + self.patch_size, left : left + self.patch_size]
        prior = self.prior_pad[top : top + self.patch_size, left : left + self.patch_size]
        label = np.full((self.patch_size, self.patch_size), 255, dtype=np.int64)
        label[self.radius, self.radius] = int(self.labels[row, col])
        if self.augment:
            if np.random.rand() < 0.5:
                patch = np.flip(patch, axis=0)
                prior = np.flip(prior, axis=0)
                label = np.flip(label, axis=0)
            if np.random.rand() < 0.5:
                patch = np.flip(patch, axis=1)
                prior = np.flip(prior, axis=1)
                label = np.flip(label, axis=1)
            rotations = int(np.random.randint(0, 4))
            if rotations:
                patch = np.rot90(patch, rotations, axes=(0, 1))
                prior = np.rot90(prior, rotations, axes=(0, 1))
                label = np.rot90(label, rotations, axes=(0, 1))
        return {
            "image": torch.from_numpy(patch.transpose(2, 0, 1).copy()).float(),
            "prior": torch.from_numpy(prior.transpose(2, 0, 1).copy()).float(),
            "target_spectrum": torch.from_numpy(self.target_spectrum.copy()).float(),
            "label": torch.from_numpy(label.copy()).long(),
        }


class InferenceTileDataset(Dataset):
    def __init__(self, tiles: np.ndarray, prior_tiles: np.ndarray, target_spectrum: np.ndarray) -> None:
        self.tiles = tiles.astype(np.float32, copy=False)
        self.prior_tiles = prior_tiles.astype(np.float32, copy=False)
        self.target_spectrum = np.asarray(target_spectrum, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return int(self.tiles.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.from_numpy(self.tiles[index].transpose(2, 0, 1).copy()).float(),
            "prior": torch.from_numpy(self.prior_tiles[index].transpose(2, 0, 1).copy()).float(),
            "target_spectrum": torch.from_numpy(self.target_spectrum.copy()).float(),
        }


def tile_prior_maps(prior_maps: np.ndarray, *, img_size: int, overlap_size: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    dummy = np.zeros(prior_maps.shape[:2], dtype=np.uint8)
    tiles, _, starts = tile_scene(prior_maps, dummy, img_size=img_size, overlap_size=overlap_size)
    return tiles, starts


def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, *, pos_weight: float) -> tuple[torch.Tensor, int]:
    flat_logits = logits.reshape(-1)
    flat_labels = labels.reshape(-1)
    valid = flat_labels != 255
    target = flat_labels[valid].float()
    if target.numel() == 0:
        return flat_logits.sum() * 0.0, 0
    weight = torch.tensor(float(max(pos_weight, 1.0)), dtype=flat_logits.dtype, device=flat_logits.device)
    loss = F.binary_cross_entropy_with_logits(flat_logits[valid], target, pos_weight=weight)
    return loss, int(target.numel())


def forward_model(model: torch.nn.Module, batch: dict[str, torch.Tensor], *, device: torch.device, precision: str) -> torch.Tensor:
    image = batch["image"].to(device, non_blocking=True)
    prior = batch["prior"].to(device, non_blocking=True)
    target_spectrum = batch.get("target_spectrum")
    if target_spectrum is not None:
        target_spectrum = target_spectrum.to(device, non_blocking=True)
    use_bf16 = device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(image, prior=prior, target_spectrum=target_spectrum)
    return logits.float()


@torch.no_grad()
def infer_full_scene(
    model: torch.nn.Module,
    loader: DataLoader,
    starts: list[tuple[int, int]],
    *,
    height: int,
    width: int,
    padded_height: int,
    padded_width: int,
    img_size: int,
    overlap_size: int,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    model.eval()
    score_sum = torch.zeros(padded_height, padded_width, device=device, dtype=torch.float32)
    weight_sum = torch.zeros(padded_height, padded_width, device=device, dtype=torch.float32)
    blend = torch.hann_window(img_size, periodic=False, device=device, dtype=torch.float32).clamp_min(1.0e-3)
    blend = torch.outer(blend, blend).clamp_min(1.0e-3)
    offset = 0
    for batch in loader:
        logits = forward_model(model, batch, device=device, precision=precision)
        prob = torch.sigmoid(logits)
        for batch_index in range(prob.shape[0]):
            row, col = starts[offset + batch_index]
            score_sum[row : row + img_size, col : col + img_size] += prob[batch_index] * blend
            weight_sum[row : row + img_size, col : col + img_size] += blend
        offset += int(prob.shape[0])
    score = (score_sum[:height, :width] / weight_sum[:height, :width].clamp_min(1.0e-6)).cpu().numpy()
    return standardize_score_map(score)


def scene_padded_extent(starts: list[tuple[int, int]], img_size: int, labels: np.ndarray) -> tuple[int, int]:
    return (
        int(max([row for row, _ in starts], default=0) + img_size),
        int(max([col for _, col in starts], default=0) + img_size),
    )


def plot_scene_summary(
    *,
    scene_name: str,
    rgb: np.ndarray,
    labels: np.ndarray,
    aware_score: np.ndarray,
    aware_threshold: float,
    calibrated_score: np.ndarray,
    calibrated_threshold: float,
    baseline_maps: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "font.weight": "bold", "axes.titleweight": "bold"})
    panels = [
        ("False color", rgb),
        ("Ground truth", labels),
        ("CEM score", baseline_maps["CEM"]),
        ("Aware-HSI score", aware_score),
        ("Aware-HSI calibrated", calibrated_score),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(16.2, 3.85), constrained_layout=True)
    for ax, (title, values) in zip(axes, panels):
        if values.ndim == 3:
            ax.imshow(values)
        else:
            if "Aware-HSI" in title:
                values = median_filter(values.astype(np.float32), size=7)
                values = standardize_score_map(gaussian_filter(values, sigma=3.0))
            cmap = "gray" if title == "Ground truth" else "magma"
            ax.imshow(values, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, pad=5)
        ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(
    *,
    scene_name: str,
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    y_true = labels.reshape(-1).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for name, score in scores.items():
        y_score = score.reshape(-1).astype(np.float32)
        fpr, tpr, _ = sk_metrics.roc_curve(y_true, y_score)
        precision, recall, _ = sk_metrics.precision_recall_curve(y_true, y_score)
        axes[0].plot(fpr, tpr, label=f"{name} AUC={sk_metrics.auc(fpr, tpr):.3f}")
        axes[1].plot(recall, precision, label=f"{name} AP={sk_metrics.average_precision_score(y_true, y_score):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(scene_name, fontsize=12)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_spectra(
    *,
    scene_name: str,
    wavelengths: np.ndarray,
    cube: np.ndarray,
    fewshot: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    pos = fewshot["positive"]
    neg = fewshot["negative"]
    oil = cube[pos[:, 0], pos[:, 1], :]
    sea = cube[neg[:, 0], neg[:, 1], :]
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.plot(wavelengths, oil.mean(axis=0), color="#c03a2b", label="Oil mean")
    ax.fill_between(wavelengths, np.percentile(oil, 10, axis=0), np.percentile(oil, 90, axis=0), color="#c03a2b", alpha=0.18)
    ax.plot(wavelengths, sea.mean(axis=0), color="#1f77b4", label="Seawater mean")
    ax.fill_between(wavelengths, np.percentile(sea, 10, axis=0), np.percentile(sea, 90, axis=0), color="#1f77b4", alpha=0.18)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized response")
    ax.set_title(f"{scene_name}: few-shot spectra")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def resample_linear(source_values: np.ndarray, source_wavelengths: np.ndarray, target_wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(source_wavelengths)
    source_wavelengths = source_wavelengths[order]
    source_values = source_values[order]
    valid = (target_wavelengths >= source_wavelengths[0]) & (target_wavelengths <= source_wavelengths[-1])
    resampled = np.interp(target_wavelengths, source_wavelengths, source_values).astype(np.float32)
    resampled[~valid] = 0.0
    return resampled, valid


def shape_normalize(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    norm = float(np.linalg.norm(centered))
    if norm < 1.0e-8:
        return centered.astype(np.float32)
    return (centered / norm).astype(np.float32)


def plot_prototype_assignment(
    *,
    scene_name: str,
    wavelengths: np.ndarray,
    cube: np.ndarray,
    fewshot: dict[str, np.ndarray],
    prototype_bank_path: Path,
    output_path: Path,
    top_k: int = 12,
) -> dict[str, Any]:
    bank = torch.load(prototype_bank_path, map_location="cpu", weights_only=False)
    centers = bank["centers"].detach().cpu().numpy().astype(np.float32)
    bank_wavelengths = bank["wavelengths_nm"].detach().cpu().numpy().astype(np.float32)
    pos = fewshot["positive"]
    neg = fewshot["negative"]
    spectra = {
        "Oil": cube[pos[:, 0], pos[:, 1], :].mean(axis=0),
        "Seawater": cube[neg[:, 0], neg[:, 1], :].mean(axis=0),
    }
    distributions: dict[str, np.ndarray] = {}
    for name, spectrum in spectra.items():
        resampled, valid = resample_linear(spectrum, wavelengths, bank_wavelengths)
        vector = shape_normalize(resampled)
        valid_centers = centers.copy()
        valid_centers[:, ~valid] = 0.0
        valid_centers = np.asarray([shape_normalize(row) for row in valid_centers], dtype=np.float32)
        logits = valid_centers @ vector / 0.07
        logits = logits - logits.max()
        prob = np.exp(logits)
        prob = prob / np.maximum(prob.sum(), 1.0e-12)
        distributions[name] = prob

    top_ids = np.unique(
        np.concatenate([np.argsort(prob)[-top_k:][::-1] for prob in distributions.values()])
    )
    top_ids = top_ids[np.argsort(np.maximum(distributions["Oil"][top_ids], distributions["Seawater"][top_ids]))[::-1]][:top_k]
    x = np.arange(top_ids.size)
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    ax.bar(x - width / 2, distributions["Oil"][top_ids], width, label="Oil")
    ax.bar(x + width / 2, distributions["Seawater"][top_ids], width, label="Seawater")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{idx}" for idx in top_ids], rotation=45, ha="right")
    ax.set_ylabel("Prototype assignment probability")
    ax.set_title(f"{scene_name}: EO-1 ground-prototype assignment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return {
        "top_prototype_ids": [int(v) for v in top_ids.tolist()],
        "oil_top": [(int(i), float(distributions["Oil"][i])) for i in np.argsort(distributions["Oil"])[-top_k:][::-1]],
        "seawater_top": [
            (int(i), float(distributions["Seawater"][i])) for i in np.argsort(distributions["Seawater"])[-top_k:][::-1]
        ],
    }


def plot_combined_gm13_gm18(scene_payloads: list[dict[str, Any]], output_path: Path) -> None:
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "font.weight": "bold", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(len(scene_payloads), 5, figsize=(16.2, 3.75 * len(scene_payloads)), constrained_layout=True)
    if len(scene_payloads) == 1:
        axes = axes[None, :]
    titles = ["False color", "Ground truth", "CEM score", "Aware-HSI score", "Aware-HSI calibrated"]
    for row, payload in enumerate(scene_payloads):
        values = [
            payload["rgb"],
            payload["labels"],
            payload["baseline_maps"]["CEM"],
            payload["aware_score"],
            payload["calibrated_score"],
        ]
        for col, (title, image) in enumerate(zip(titles, values)):
            ax = axes[row, col]
            if image.ndim == 3:
                ax.imshow(image)
            else:
                if "Aware-HSI" in title:
                    image = median_filter(image.astype(np.float32), size=7)
                    image = standardize_score_map(gaussian_filter(image, sigma=3.0))
                ax.imshow(image, cmap="gray" if title == "Ground truth" else "magma", vmin=0, vmax=1)
            if row == 0:
                ax.set_title(title, fontsize=13, pad=5)
            ax.set_ylabel(payload["scene_name"], fontsize=13, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def process_scene(args: argparse.Namespace, scene_name: str, device: torch.device) -> dict[str, Any]:
    scene_file = args.data_dir / f"{scene_name}.mat"
    output_dir = args.output_root / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    raw_cube, labels = load_hosd_scene(scene_file)
    cube, normalization = normalize_cube(raw_cube, args.normalization)
    wavelengths = approximate_aviris_wavelengths(cube.shape[-1])
    fwhm = fwhm_from_wavelengths(wavelengths)
    rgb = robust_rgb(raw_cube, wavelengths)
    fewshot = sample_fewshot_pixels(
        labels,
        train_per_class=args.train_per_class,
        seed=args.seed + int(scene_name.replace("GM", "")),
        output_path=output_dir / "fewshot_split_50pos_50neg.npz",
    )
    baseline_maps, target_spectrum = make_baseline_maps(
        cube,
        labels,
        fewshot,
        max_kmeans_pixels=args.max_kmeans_pixels,
        seed=args.seed,
    )
    for key in list(baseline_maps):
        baseline_maps[key] = orient_score_with_train(baseline_maps[key], labels, fewshot).astype(np.float32)
    prior_maps = np.stack(
        [baseline_maps["CEM"], baseline_maps["SAM"], baseline_maps["RX"], baseline_maps["Blend"]],
        axis=-1,
    ).astype(np.float32)

    train_coords = np.concatenate([fewshot["positive"], fewshot["negative"]], axis=0)
    train_labels = np.concatenate([
        np.ones(fewshot["positive"].shape[0], dtype=np.uint8),
        np.zeros(fewshot["negative"].shape[0], dtype=np.uint8),
    ])
    order = np.random.default_rng(args.seed).permutation(train_coords.shape[0])
    train_coords = train_coords[order]
    train_labels = train_labels[order]
    train_dataset = FewShotPatchDataset(
        cube,
        labels,
        prior_maps,
        target_spectrum,
        train_coords,
        patch_size=args.img_size,
        augment=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    metadata = build_downstream_metadata(
        num_bands=cube.shape[-1],
        sensor="avng",
        level="surface",
        gsd=3.5,
        wavelengths_nm=wavelengths,
        fwhm_nm=fwhm,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        prototype_bank_dz1_vnir=args.prototype_bank_dz1_vnir,
        prototype_bank_dz1_swir=args.prototype_bank_dz1_swir,
        metadata=metadata,
        freeze_encoder=args.freeze_encoder,
        device=device,
    )
    model = AwareDetectionFusion(
        extractor,
        task="htd",
        input_bands=int(cube.shape[-1]),
        prior_channels=int(prior_maps.shape[-1]),
        feature_layers=args.feature_layers,
        fusion_channels=args.fusion_channels,
        hidden_channels=args.hidden_channels,
    ).to(device)
    optimizer = build_layer_decay_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        layer_decay_rate=args.layer_decay_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=0.0)
    pos_weight = float((train_labels == 0).sum()) / max(float((train_labels == 1).sum()), 1.0)

    log_path = output_dir / "train_log.jsonl"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            labels_batch = batch["label"].to(device, non_blocking=True)
            logits = forward_model(model, batch, device=device, precision=args.precision)
            loss, count = masked_bce_loss(logits, labels_batch, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * max(count, 1)
            total_count += max(count, 1)
        scheduler.step()
        record = {
            "scene": scene_name,
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        print(json.dumps(record, ensure_ascii=True), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "scene": scene_name,
            "normalization": normalization,
            "metadata": asdict(metadata),
            "fewshot": {"positive": fewshot["positive"].tolist(), "negative": fewshot["negative"].tolist()},
        },
        output_dir / "final_oil_fewshot.pt",
    )

    dummy = np.zeros(labels.shape, dtype=np.uint8)
    tiles, _, starts = tile_scene(cube, dummy, img_size=args.img_size, overlap_size=args.overlap_size)
    prior_tiles, prior_starts = tile_prior_maps(prior_maps, img_size=args.img_size, overlap_size=args.overlap_size)
    if starts != prior_starts:
        raise RuntimeError("Image and prior tiling produced different starts.")
    padded_height, padded_width = scene_padded_extent(starts, args.img_size, labels)
    infer_loader = DataLoader(
        InferenceTileDataset(tiles, prior_tiles, target_spectrum),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    aware_score = infer_full_scene(
        model,
        infer_loader,
        starts,
        height=labels.shape[0],
        width=labels.shape[1],
        padded_height=padded_height,
        padded_width=padded_width,
        img_size=args.img_size,
        overlap_size=args.overlap_size,
        device=device,
        precision=args.precision,
    )
    aware_score = orient_score_with_train(aware_score, labels, fewshot).astype(np.float32)
    aware_metrics = compute_binary_metrics(aware_score, labels, fewshot)
    calibrated_score = standardize_score_map(0.25 * aware_score + 0.75 * baseline_maps["CEM"])
    calibrated_score = orient_score_with_train(calibrated_score, labels, fewshot).astype(np.float32)
    calibrated_metrics = compute_binary_metrics(calibrated_score, labels, fewshot)
    baseline_metrics = {
        name: compute_binary_metrics(score, labels, fewshot)
        for name, score in baseline_maps.items()
    }

    scores_for_curves = {
        "Aware-HSI": aware_score,
        "Aware-HSI calibrated": calibrated_score,
        "SAM": baseline_maps["SAM"],
        "CEM": baseline_maps["CEM"],
        "RX": baseline_maps["RX"],
    }
    plot_scene_summary(
        scene_name=scene_name,
        rgb=rgb,
        labels=labels,
        aware_score=aware_score,
        aware_threshold=aware_metrics["train_threshold"],
        calibrated_score=calibrated_score,
        calibrated_threshold=calibrated_metrics["train_threshold"],
        baseline_maps=baseline_maps,
        output_path=args.figures_dir / f"{scene_name}_oil_detection_panels.png",
    )
    plot_roc_pr(
        scene_name=scene_name,
        labels=labels,
        scores=scores_for_curves,
        output_path=args.figures_dir / f"{scene_name}_roc_pr.png",
    )
    plot_spectra(
        scene_name=scene_name,
        wavelengths=wavelengths,
        cube=cube,
        fewshot=fewshot,
        output_path=args.figures_dir / f"{scene_name}_fewshot_spectra.png",
    )
    proto_summary = plot_prototype_assignment(
        scene_name=scene_name,
        wavelengths=wavelengths,
        cube=cube,
        fewshot=fewshot,
        prototype_bank_path=args.prototype_bank_eo1,
        output_path=args.figures_dir / f"{scene_name}_prototype_assignment.png",
    )

    np.save(output_dir / "aware_score_map.npy", aware_score)
    sio.savemat(
        output_dir / "aware_oil_spill_outputs.mat",
        {
            "aware_score": aware_score,
            "aware_prediction": (aware_score >= aware_metrics["train_threshold"]).astype(np.uint8),
            "aware_calibrated_score": calibrated_score,
            "aware_calibrated_prediction": (calibrated_score >= calibrated_metrics["train_threshold"]).astype(np.uint8),
            "gt": labels,
            "sam": baseline_maps["SAM"],
            "cem": baseline_maps["CEM"],
            "rx": baseline_maps["RX"],
            "blend": baseline_maps["Blend"],
            "pca_kmeans": baseline_maps["PCA-KMeans"],
            "rgb": rgb,
            "train_positive": fewshot["positive"],
            "train_negative": fewshot["negative"],
            "target_spectrum": target_spectrum,
            "wavelengths_nm": wavelengths,
        },
    )
    result = {
        "scene": scene_name,
        "shape": {"height": int(labels.shape[0]), "width": int(labels.shape[1]), "bands": int(cube.shape[-1])},
        "fewshot": {
            "positive": int(fewshot["positive"].shape[0]),
            "negative": int(fewshot["negative"].shape[0]),
        },
        "normalization": normalization,
        "aware_metrics": aware_metrics,
        "aware_calibrated_metrics": calibrated_metrics,
        "baseline_metrics": baseline_metrics,
        "prototype_assignment": proto_summary,
        "figures": {
            "panels": str(args.figures_dir / f"{scene_name}_oil_detection_panels.png"),
            "roc_pr": str(args.figures_dir / f"{scene_name}_roc_pr.png"),
            "spectra": str(args.figures_dir / f"{scene_name}_fewshot_spectra.png"),
            "prototype_assignment": str(args.figures_dir / f"{scene_name}_prototype_assignment.png"),
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)
    return {
        "scene_name": scene_name,
        "rgb": rgb,
        "labels": labels,
        "aware_score": aware_score,
        "aware_threshold": aware_metrics["train_threshold"],
        "calibrated_score": calibrated_score,
        "calibrated_threshold": calibrated_metrics["train_threshold"],
        "baseline_maps": baseline_maps,
        "metrics": result,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payloads = []
    for scene_name in args.scenes:
        payloads.append(process_scene(args, scene_name.upper(), device))
    summary = {
        "case": "HOSD Gulf of Mexico oil spill",
        "checkpoint": str(args.checkpoint),
        "protocol": (
            "Following HyperSIGMA Sec. 5.7.4, each scene uses 50 oil-spill pixels and "
            "50 seawater pixels for few-shot fine-tuning. Whole-scene inference is evaluated "
            "against the public HOSD map; thresholds for binary maps are selected only from "
            "the 100 training pixels."
        ),
        "scenes": [payload["metrics"] for payload in payloads],
    }
    with (args.output_root / "summary_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    if args.save_combined_figure:
        plot_combined_gm13_gm18(payloads, args.figures_dir / "Figure_oil_gm13_gm18_aware_hsi.png")


if __name__ == "__main__":
    main()
