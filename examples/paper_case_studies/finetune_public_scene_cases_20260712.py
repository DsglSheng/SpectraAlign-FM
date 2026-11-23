#!/usr/bin/env python3
"""Few-shot public-scene case studies for Aware-HSI.

This script is deliberately lighter than the full benchmark fine-tuning recipe:
the Aware-HSI encoder is frozen and only the dense segmentation/classification
head is optimized. The goal is to support real-world applicability figures on
public scenes that are not part of the main comparison tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SPECTRAALIGN_DATA_ROOT", PROJECT / "data")).expanduser()
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from aware_open.downstream import AwareFeatureExtractor, build_downstream_metadata, build_segmentation_scene, resolve_scene_config
from aware_open.downstream.classification_scene import CenteredLabelCropDataset, metrics_from_predictions
from scripts.finetune_classification import (
    AwareSSFusionSeg,
    build_blend_window,
    build_class_spectral_prototypes,
    build_class_weights_from_labels,
    infer_tile_logits,
    masked_cross_entropy,
)


DEFAULT_CHECKPOINT = (
    PROJECT / "pretraining/runs/final_pretrain_hyperglobal_dz1_525k_fromscratch_v1/"
    "stage_b_full_joint154535/checkpoints/best_total.pt"
)
DEFAULT_PROTO_EO1 = DATA_ROOT / "prototype_banks/prototype_bank_eo1.pt"
DEFAULT_PROTO_GF5 = DATA_ROOT / "prototype_banks/prototype_bank_gf5.pt"
DEFAULT_PROTO_DZ1_VNIR = DATA_ROOT / "prototype_banks/prototype_bank_dz1_vnir.pt"
DEFAULT_PROTO_DZ1_SWIR = DATA_ROOT / "prototype_banks/prototype_bank_dz1_swir.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot frozen-encoder public-scene fine-tuning.")
    parser.add_argument("--dataset", choices=("pavia_center", "salinas"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prototype-bank-eo1", type=Path, default=DEFAULT_PROTO_EO1)
    parser.add_argument("--prototype-bank-gf5", type=Path, default=DEFAULT_PROTO_GF5)
    parser.add_argument("--prototype-bank-dz1-vnir", type=Path, default=DEFAULT_PROTO_DZ1_VNIR)
    parser.add_argument("--prototype-bank-dz1-swir", type=Path, default=DEFAULT_PROTO_DZ1_SWIR)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--train-num", type=int, default=50)
    parser.add_argument("--val-num", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--overlap-size", type=int, default=32)
    parser.add_argument("--fusion-channels", type=int, default=128)
    parser.add_argument("--class-prototype-weight", type=float, default=0.15)
    parser.add_argument("--class-prototype-temperature", type=float, default=0.10)
    parser.add_argument("--normalization", choices=("minmax", "mean_std", "none"), default="minmax")
    parser.add_argument("--seed", type=int, default=53111)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def palette(num_classes: int) -> np.ndarray:
    base = np.array(
        [
            [0, 0, 0],
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
            [128, 128, 0],
            [255, 215, 180],
            [0, 0, 128],
        ],
        dtype=np.uint8,
    )
    if num_classes + 1 <= len(base):
        return base[: num_classes + 1]
    extra = np.random.default_rng(20260531).integers(0, 255, size=(num_classes + 1 - len(base), 3), dtype=np.uint8)
    return np.vstack([base, extra])


def label_to_rgb(labels: np.ndarray, num_classes: int) -> np.ndarray:
    pal = palette(num_classes)
    clipped = np.clip(labels.astype(np.int64), 0, num_classes)
    return pal[clipped]


def robust_rgb(cube: np.ndarray, dataset: str) -> np.ndarray:
    if dataset == "pavia_center":
        bands = (60, 35, 15)
    else:
        bands = (45, 25, 8)
    channels = []
    for idx in bands:
        idx = min(max(int(idx), 0), cube.shape[-1] - 1)
        band = cube[..., idx].astype(np.float32)
        valid = np.isfinite(band)
        lo, hi = np.percentile(band[valid], [1.0, 99.0])
        channels.append(np.clip((band - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    return np.dstack(channels)


def pca_kmeans_map(cube: np.ndarray, num_classes: int, valid_mask: np.ndarray) -> np.ndarray:
    h, w, c = cube.shape
    flat = cube.reshape(-1, c).astype(np.float32)
    rng = np.random.default_rng(20260531)
    valid_idx = np.flatnonzero(valid_mask.reshape(-1))
    sample = rng.choice(valid_idx, size=min(50000, valid_idx.size), replace=False)
    pca = PCA(n_components=min(8, c), random_state=20260531)
    train = pca.fit_transform(flat[sample])
    km = MiniBatchKMeans(n_clusters=num_classes, random_state=20260531, batch_size=4096, n_init=10)
    km.fit(train[:, : min(6, train.shape[1])])
    labels = km.predict(pca.transform(flat)[:, : min(6, train.shape[1])]).reshape(h, w) + 1
    labels[~valid_mask] = 0
    return labels.astype(np.int16)


def evaluate_center_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_count: int,
    precision: str,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    model.eval()
    model.extractor.encoder.eval()
    all_pred: list[np.ndarray] = []
    all_label: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    autocast_enabled = device.type == "cuda" and precision == "bf16"
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = model(x)
                loss, selected_logits, selected_target = masked_cross_entropy(logits, y)
            if loss is None:
                continue
            total_loss += float(loss.item()) * int(selected_target.numel())
            total_count += int(selected_target.numel())
            all_pred.append(selected_logits.argmax(dim=1).cpu().numpy())
            all_label.append(selected_target.cpu().numpy())
    pred = np.concatenate(all_pred) if all_pred else np.empty((0,), dtype=np.int64)
    label = np.concatenate(all_label) if all_label else np.empty((0,), dtype=np.int64)
    return metrics_from_predictions(label, pred, class_count, total_loss / max(total_count, 1), class_names=class_names)


@torch.no_grad()
def infer_full_scene(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    class_count: int,
    precision: str,
    starts: list[tuple[int, int]],
    height: int,
    width: int,
    padded_height: int,
    padded_width: int,
    img_size: int,
    overlap_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    model.extractor.encoder.eval()
    score_sum = torch.zeros(padded_height, padded_width, class_count, device=device, dtype=torch.float32)
    weight_sum = torch.zeros(padded_height, padded_width, device=device, dtype=torch.float32)
    blend = build_blend_window(img_size, overlap_size, device=device)
    offset = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits_map = infer_tile_logits(model, x, precision=precision, eval_tta=False)
        batch_size = int(logits_map.shape[0])
        for batch_index in range(batch_size):
            row, col = starts[offset + batch_index]
            score_sum[row : row + img_size, col : col + img_size] += logits_map[batch_index] * blend.unsqueeze(-1)
            weight_sum[row : row + img_size, col : col + img_size] += blend
        offset += batch_size
    logits = score_sum[:height, :width] / weight_sum[:height, :width].unsqueeze(-1).clamp_min(1.0e-6)
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1).cpu().numpy().astype(np.int16) + 1
    conf = probs.max(dim=-1).values.cpu().numpy().astype(np.float32)
    return pred, conf


def full_metrics(pred: np.ndarray, label_map: np.ndarray, class_count: int, class_names: tuple[str, ...]) -> dict[str, Any]:
    valid = label_map.reshape(-1) > 0
    y_true = label_map.reshape(-1)[valid].astype(np.int64) - 1
    y_pred = pred.reshape(-1)[valid].astype(np.int64) - 1
    return metrics_from_predictions(y_true, y_pred, class_count, 0.0, class_names=class_names)


def save_case_figure(
    *,
    dataset: str,
    output_dir: Path,
    cube: np.ndarray,
    labels: np.ndarray,
    test_gt: np.ndarray,
    prediction: np.ndarray,
    confidence: np.ndarray,
    baseline: np.ndarray,
    class_count: int,
    metrics: dict[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "font.weight": "bold",
            "axes.titleweight": "bold",
        }
    )
    rgb = robust_rgb(cube, dataset)
    error = np.zeros((*prediction.shape, 3), dtype=np.float32)
    mask = test_gt > 0
    correct = (prediction == test_gt) & mask
    wrong = (prediction != test_gt) & mask
    error[correct] = np.array([1.0, 1.0, 1.0])
    error[wrong] = np.array([0.95, 0.05, 0.05])
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.75), dpi=260)
    panels = [
        (rgb, "False color / pseudo RGB", None),
        (label_to_rgb(labels, class_count), "Public ground truth", None),
        (label_to_rgb(baseline, class_count), "PCA+k-means baseline", None),
        (label_to_rgb(prediction, class_count), "Aware-HSI few-shot prediction", None),
        (confidence, "Prediction confidence", "viridis"),
        (error, "Test-mask correctness", None),
    ]
    for ax, (image, name, cmap) in zip(axes.ravel(), panels, strict=True):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1 if cmap else None)
        ax.set_title(name, fontsize=11, pad=5)
        ax.set_axis_off()
    fig.tight_layout(pad=0.35, h_pad=0.65, w_pad=0.35)
    fig.savefig(output_dir / f"Figure_{dataset}_case.png", bbox_inches="tight")
    fig.savefig(output_dir / f"Figure_{dataset}_case.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = resolve_scene_config(args.dataset)
    scene = build_segmentation_scene(
        data_root=args.data_root,
        config=config,
        img_size=args.img_size,
        overlap_size=args.overlap_size,
        train_num=args.train_num,
        val_num=args.val_num,
        batch_size=args.batch_size,
        num_workers=0,
        augment=True,
        normalization=args.normalization,
        train_repeat_factor=1,
        train_spatial_jitter=16,
        centered_training=True,
        train_shuffle=True,
        train_sampling="inverse_sqrt",
        train_drop_last=False,
        eval_drop_last=False,
    )
    class_count = int(scene["class_count"])
    class_names = tuple(config.class_names)
    metadata = build_downstream_metadata(
        num_bands=int(scene["channels"]),
        sensor=config.sensor,
        level=config.level,
        gsd=config.gsd_m,
        wavelengths_nm=config.wavelengths if int(config.wavelengths.numel()) == int(scene["channels"]) else None,
        fwhm_nm=config.fwhm if int(config.fwhm.numel()) == int(scene["channels"]) else None,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        prototype_bank_dz1_vnir=args.prototype_bank_dz1_vnir,
        prototype_bank_dz1_swir=args.prototype_bank_dz1_swir,
        metadata=metadata,
        freeze_encoder=True,
        device=device,
    )
    class_prototypes, proto_info = build_class_spectral_prototypes(scene["cube"], scene["train_gt"], class_count)
    class_weights, weight_info = build_class_weights_from_labels(scene["train_gt"], class_count, "inverse_sqrt")
    if class_weights is not None:
        class_weights = class_weights.to(device)
    model = AwareSSFusionSeg(
        extractor,
        class_count,
        feature_layers=4,
        fusion_channels=args.fusion_channels,
        class_prototypes=class_prototypes.to(device),
        class_prototype_weight=args.class_prototype_weight,
        class_prototype_temperature=args.class_prototype_temperature,
    ).to(device)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    val_indices = np.flatnonzero(scene["val_gt"].reshape(-1) > 0)
    val_dataset = CenteredLabelCropDataset(
        scene["cube"],
        scene["val_gt"],
        val_indices,
        crop_size=int(scene["img_size"]),
        augment=False,
        spatial_jitter=0,
        repeat_factor=1,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    autocast_enabled = device.type == "cuda" and args.precision == "bf16"
    best_state = None
    best_val = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.extractor.encoder.eval()
        total_loss = 0.0
        total_count = 0
        total_correct = 0
        for x, y in scene["train_loader"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = model(x)
                loss, selected_logits, selected_target = masked_cross_entropy(logits, y, class_weights=class_weights)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(selected_target.numel())
            total_count += int(selected_target.numel())
            total_correct += int((selected_logits.argmax(dim=1) == selected_target).sum().item())
        scheduler.step()
        val_metrics = evaluate_center_loader(model, val_loader, device, class_count, args.precision, class_names)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "train_acc": total_correct / max(total_count, 1),
            "val_oa": float(val_metrics["oa"]),
            "val_aa": float(val_metrics["aa"]),
            "val_kappa": float(val_metrics["kappa"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if float(val_metrics["oa"]) > best_val:
            best_val = float(val_metrics["oa"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    prediction, confidence = infer_full_scene(
        model,
        scene["test_loader"],
        device,
        class_count=class_count,
        precision=args.precision,
        starts=scene["test_starts"],
        height=int(scene["height"]),
        width=int(scene["width"]),
        padded_height=int(scene["padded_height"]),
        padded_width=int(scene["padded_width"]),
        img_size=int(scene["img_size"]),
        overlap_size=int(scene["overlap_size"]),
    )
    test_metrics = full_metrics(prediction, scene["test_gt"], class_count, class_names)
    baseline = pca_kmeans_map(scene["cube"], class_count, scene["labels"] > 0)

    np.savez_compressed(
        args.output_dir / "case_outputs.npz",
        prediction=prediction,
        confidence=confidence,
        labels=scene["labels"].astype(np.int16),
        train_gt=scene["train_gt"].astype(np.int16),
        val_gt=scene["val_gt"].astype(np.int16),
        test_gt=scene["test_gt"].astype(np.int16),
        baseline_pca_kmeans=baseline,
    )
    torch.save({"model": best_state, "history": history, "config": vars(args), "split": scene["split_summary"]}, args.output_dir / "best_lightweight_head.pt")
    report = {
        "dataset": args.dataset,
        "protocol": "frozen HyperGlobal-DZ1 Aware-HSI encoder, train dense head only, 50 train and 20 validation pixels per class",
        "checkpoint": str(args.checkpoint),
        "split": scene["split_summary"],
        "class_prototypes": proto_info,
        "class_weights": weight_info,
        "history": history,
        "test_metrics": test_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_case_figure(
        dataset=args.dataset,
        output_dir=args.output_dir,
        cube=scene["cube"],
        labels=scene["labels"],
        test_gt=scene["test_gt"],
        prediction=prediction,
        confidence=confidence,
        baseline=baseline,
        class_count=class_count,
        metrics=test_metrics,
    )


if __name__ == "__main__":
    main()
