from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from aware_open.downstream.classification_scene import (
    CenteredLabelCropDataset,
    build_segmentation_scene,
    metrics_from_predictions,
    resolve_scene_config,
    seed_everything,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "HyperSIGMA-main" / "ImageClassification" / "data"
DEFAULT_SPECTRALGPT_REPO = (
    REPO_ROOT
    / "external"
    / "SpectralGPT_IEEE_TPAMI_20260516"
    / "IEEE_TPAMI_SpectralGPT"
)
DEFAULT_SPECTRALGPT_CKPT = (
    REPO_ROOT
    / "pretrained"
    / "spectralgpt_zenodo_10533809_20260516"
    / "SpectralGPT+.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune SpectralGPT on HyperSIGMA HSI classification datasets. "
            "HSI cubes are adapted to the 12-frame SpectralGPT input using PCA-12."
        )
    )
    parser.add_argument("--spectralgpt-repo", type=Path, default=DEFAULT_SPECTRALGPT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_SPECTRALGPT_CKPT)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--overlap-size", type=int, default=32)
    parser.add_argument("--train-num", type=int, default=None)
    parser.add_argument("--val-num", type=int, default=None)
    parser.add_argument("--split-mode", choices=("number", "official", "ratio"), default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--official-val-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--scheduler-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3704)
    parser.add_argument("--normalization", choices=("none", "minmax", "mean_std"), default="mean_std")
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--pca-whiten", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-weighting", choices=("none", "inverse", "inverse_sqrt"), default="none")
    parser.add_argument("--train-repeat-factor", type=int, default=4)
    parser.add_argument("--train-spatial-jitter", type=int, default=8)
    parser.add_argument("--train-sampling", choices=("shuffle", "inverse", "inverse_sqrt"), default="shuffle")
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-tta", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def import_spectralgpt(repo: Path):
    repo = repo.expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"SpectralGPT repo not found: {repo}")
    sys.path.insert(0, str(repo))
    import models_vit_tensor  # type: ignore

    return models_vit_tensor


def resize_spatial_pos_embed(pos: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if pos.ndim != 3:
        return pos
    source_tokens = int(pos.shape[1])
    if source_tokens == target_tokens:
        return pos
    source_hw = int(round(math.sqrt(source_tokens)))
    target_hw = int(round(math.sqrt(target_tokens)))
    if source_hw * source_hw != source_tokens or target_hw * target_hw != target_tokens:
        raise ValueError(f"Cannot resize spatial position embedding {source_tokens} -> {target_tokens}.")
    resized = pos.reshape(1, source_hw, source_hw, pos.shape[-1]).permute(0, 3, 1, 2)
    resized = F.interpolate(resized, size=(target_hw, target_hw), mode="bicubic", align_corners=False)
    return resized.permute(0, 2, 3, 1).reshape(1, target_tokens, pos.shape[-1])


def build_spectralgpt_model(models_vit_tensor: Any, *, img_size: int, num_classes: int, drop_path: float, dropout: float) -> nn.Module:
    return models_vit_tensor.VisionTransformer(
        img_size=int(img_size),
        in_chans=1,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_frames=12,
        t_patch_size=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=int(num_classes),
        drop_path_rate=float(drop_path),
        dropout=float(dropout),
        sep_pos_embed=True,
        cls_embed=False,
    )


def load_spectralgpt_checkpoint(model: nn.Module, checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"SpectralGPT checkpoint not found: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model_state = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    dropped: dict[str, str] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key in {"head.weight", "head.bias"}:
            dropped[key] = "task_head_reinitialized"
            continue
        if key == "pos_embed_spatial" and key in model_state and value.shape != model_state[key].shape:
            value = resize_spatial_pos_embed(value, int(model_state[key].shape[1]))
        if key not in model_state:
            dropped[key] = "not_in_model"
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            dropped[key] = f"shape {tuple(value.shape)} != {tuple(model_state[key].shape)}"
            continue
        filtered[key] = value
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)
    return {
        "checkpoint": str(checkpoint),
        "loaded_keys": len(filtered),
        "dropped_keys": dropped,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def nonzero_indices(label_map: np.ndarray) -> np.ndarray:
    return np.where(label_map.reshape(-1) > 0)[0].astype(np.int64, copy=False)


def build_center_loaders(scene: dict[str, Any], args: argparse.Namespace) -> dict[str, DataLoader]:
    cube = scene["cube"]
    img_size = int(scene["img_size"])
    common = dict(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    train_idx = nonzero_indices(scene["train_gt"])
    val_idx = nonzero_indices(scene["val_gt"])
    test_idx = nonzero_indices(scene["test_gt"])
    if train_idx.size == 0:
        raise ValueError(f"{args.dataset} has no training labels after split.")
    if test_idx.size == 0:
        raise ValueError(f"{args.dataset} has no test labels after split.")

    train_dataset = CenteredLabelCropDataset(
        cube,
        scene["train_gt"],
        train_idx,
        crop_size=img_size,
        augment=not bool(args.no_augment),
        spatial_jitter=int(args.train_spatial_jitter),
        repeat_factor=int(args.train_repeat_factor),
    )
    val_dataset = CenteredLabelCropDataset(cube, scene["val_gt"], val_idx, crop_size=img_size) if val_idx.size else None
    test_dataset = CenteredLabelCropDataset(cube, scene["test_gt"], test_idx, crop_size=img_size)

    sampler = None
    if args.train_sampling != "shuffle":
        center_labels = scene["train_gt"].reshape(-1)[train_idx]
        counts = np.bincount(center_labels.astype(np.int64), minlength=int(scene["class_count"]) + 1).astype(np.float64)
        counts[counts <= 0.0] = 1.0
        if args.train_sampling == "inverse":
            weights = 1.0 / counts[center_labels.astype(np.int64)]
        else:
            weights = 1.0 / np.sqrt(counts[center_labels.astype(np.int64)])
        weights = np.repeat(weights.astype(np.float32, copy=False), int(args.train_repeat_factor))
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(train_dataset, shuffle=sampler is None, sampler=sampler, drop_last=False, **common),
        "test": DataLoader(test_dataset, shuffle=False, drop_last=False, **common),
    }
    if val_dataset is not None:
        loaders["val"] = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return loaders


def build_class_weights(label_map: np.ndarray, class_count: int, mode: str) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if mode == "none":
        return None, {"mode": "none"}
    labels = label_map.reshape(-1)
    valid = labels[labels > 0].astype(np.int64, copy=False)
    counts = np.bincount(valid, minlength=class_count + 1)[1:].astype(np.float64)
    counts[counts <= 0.0] = 1.0
    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unsupported class weighting mode: {mode}")
    weights = weights / max(float(weights.mean()), 1.0e-12)
    return torch.as_tensor(weights, dtype=torch.float32), {
        "mode": mode,
        "counts": counts.astype(np.int64).tolist(),
        "weights": weights.astype(np.float32).tolist(),
    }


def center_targets(label_maps: torch.Tensor) -> torch.Tensor:
    center = int(label_maps.shape[-1]) // 2
    return label_maps[:, center, center].long() - 1


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type=device.type, enabled=False)


def forward_with_tta(model: nn.Module, x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return model(x)
    variants = [
        x,
        torch.flip(x, dims=(-1,)),
        torch.flip(x, dims=(-2,)),
        torch.flip(x, dims=(-2, -1)),
    ]
    return torch.stack([model(v) for v in variants], dim=0).mean(dim=0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    precision: str,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    eval_tta: bool = False,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    for x, labels in loader:
        x = x.to(device, non_blocking=True).float()
        target = center_targets(labels).to(device, non_blocking=True)
        valid = target >= 0
        if int(valid.sum().item()) == 0:
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, precision):
            logits = forward_with_tta(model, x, eval_tta and not training)
            loss = F.cross_entropy(
                logits[valid],
                target[valid],
                weight=class_weights,
                label_smoothing=float(label_smoothing) if training else 0.0,
            )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        preds_all.append(logits.detach().argmax(dim=1)[valid].cpu().numpy().astype(np.int64))
        labels_all.append(target.detach()[valid].cpu().numpy().astype(np.int64))
    labels_np = np.concatenate(labels_all) if labels_all else np.empty((0,), dtype=np.int64)
    preds_np = np.concatenate(preds_all) if preds_all else np.empty((0,), dtype=np.int64)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "labels": labels_np,
        "predictions": preds_np,
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "best": best,
            "args": vars(args),
        },
        path,
    )


def main() -> int:
    args = parse_args()
    args.scheduler_epochs = int(args.epochs if args.scheduler_epochs is None else args.scheduler_epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    models_vit_tensor = import_spectralgpt(args.spectralgpt_repo)
    config = resolve_scene_config(args.dataset)
    scene = build_segmentation_scene(
        data_root=args.data_root,
        config=config,
        split_mode=args.split_mode,
        img_size=args.img_size,
        overlap_size=args.overlap_size,
        train_num=args.train_num,
        val_num=args.val_num,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=max(1, int(args.batch_size)),
        num_workers=0,
        augment=False,
        normalization=args.normalization,
        pca_components=int(args.pca_components),
        pca_whiten=bool(args.pca_whiten),
        train_repeat_factor=1,
        train_spatial_jitter=0,
        centered_training=True,
        official_val_ratio=args.official_val_ratio,
    )
    loaders = build_center_loaders(scene, args)
    class_count = int(scene["class_count"])
    model = build_spectralgpt_model(
        models_vit_tensor,
        img_size=int(scene["img_size"]),
        num_classes=class_count,
        drop_path=float(args.drop_path),
        dropout=float(args.dropout),
    )
    load_info = load_spectralgpt_checkpoint(model, args.checkpoint)
    model.to(device)

    class_weights, class_weight_info = build_class_weights(scene["train_gt"], class_count, args.class_weighting)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(args.scheduler_epochs)),
        eta_min=max(1.0e-7, float(args.lr) * 0.01),
    )

    start_epoch = 1
    best = {"val_oa": -1.0, "val_loss": float("inf"), "train_loss": float("inf")}
    resume_path = args.resume_from
    if resume_path is None and args.auto_resume:
        latest = args.output_dir / "latest_classification.pt"
        if latest.exists():
            resume_path = latest
    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best.update({k: float(v) for k, v in ckpt.get("best", {}).items()})

    config_payload = {
        **vars(args),
        "checkpoint": str(args.checkpoint),
        "spectralgpt_repo": str(args.spectralgpt_repo),
        "device": str(device),
        "class_count": class_count,
        "split_summary": scene["split_summary"],
        "spectralgpt_adapter": "PCA-12 HSI-to-Sentinel-style tensor input",
        "load_info": load_info,
        "class_weight_info": class_weight_info,
    }
    save_json(args.output_dir / "config.json", config_payload)

    log_path = args.output_dir / "classification_log.jsonl"
    class_names = tuple(scene["config"].class_names)
    for epoch in range(start_epoch, int(args.epochs) + 1):
        tic = time.time()
        train_raw = run_epoch(
            model,
            loaders["train"],
            device,
            optimizer=optimizer,
            precision=args.precision,
            class_weights=class_weights,
            label_smoothing=float(args.label_smoothing),
        )
        scheduler.step()
        train_metrics = metrics_from_predictions(
            train_raw["labels"], train_raw["predictions"], class_count, train_raw["loss"], class_names=class_names
        )
        val_metrics = None
        if "val" in loaders:
            with torch.no_grad():
                val_raw = run_epoch(
                    model,
                    loaders["val"],
                    device,
                    optimizer=None,
                    precision=args.precision,
                    class_weights=None,
                    label_smoothing=0.0,
                    eval_tta=bool(args.eval_tta),
                )
            val_metrics = metrics_from_predictions(
                val_raw["labels"], val_raw["predictions"], class_count, val_raw["loss"], class_names=class_names
            )

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - tic,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": {k: v for k, v in train_metrics.items() if k != "confusion_matrix"},
            "val": None if val_metrics is None else {k: v for k, v in val_metrics.items() if k != "confusion_matrix"},
        }
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
        print(json.dumps(record, ensure_ascii=False, allow_nan=True), flush=True)

        save_checkpoint(
            args.output_dir / "latest_classification.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best=best,
            args=args,
        )
        if train_metrics["loss"] < best["train_loss"]:
            best["train_loss"] = float(train_metrics["loss"])
            save_checkpoint(
                args.output_dir / "best_train_loss.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best=best,
                args=args,
            )
        if val_metrics is not None:
            if val_metrics["loss"] < best["val_loss"]:
                best["val_loss"] = float(val_metrics["loss"])
                save_checkpoint(
                    args.output_dir / "best_val_loss.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best=best,
                    args=args,
                )
            if val_metrics["oa"] > best["val_oa"]:
                best["val_oa"] = float(val_metrics["oa"])
                save_checkpoint(
                    args.output_dir / "best_val_oa.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best=best,
                    args=args,
                )

    with torch.no_grad():
        test_raw = run_epoch(
            model,
            loaders["test"],
            device,
            optimizer=None,
            precision=args.precision,
            class_weights=None,
            label_smoothing=0.0,
            eval_tta=bool(args.eval_tta),
        )
    test_metrics = metrics_from_predictions(
        test_raw["labels"], test_raw["predictions"], class_count, test_raw["loss"], class_names=class_names
    )
    save_json(args.output_dir / "test_metrics.json", test_metrics)
    print("[SpectralGPT] Test metrics:", json.dumps(test_metrics, ensure_ascii=False, allow_nan=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
