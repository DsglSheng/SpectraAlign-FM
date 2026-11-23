from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aware_open.downstream import (
    build_indian_pines_loaders,
    metrics_from_predictions,
    resolve_indian_pines_protocol,
    seed_everything,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYPERSIGMA_ROOT = REPO_ROOT.parent / "HyperSIGMA-main"
HYPERSIGMA_IMAGE_CLS_ROOT = DEFAULT_HYPERSIGMA_ROOT / "ImageClassification"

if str(HYPERSIGMA_IMAGE_CLS_ROOT) not in sys.path:
    sys.path.insert(0, str(HYPERSIGMA_IMAGE_CLS_ROOT))

from model import ss_fusion_cls  # noqa: E402


@contextmanager
def allow_argparse_namespace_unpickle():
    import argparse as _argparse

    with torch.serialization.safe_globals([_argparse.Namespace]):
        yield


class HyperSIGMAClassifier(nn.Module):
    def __init__(
        self,
        *,
        img_size: int,
        in_channels: int,
        patch_size: int,
        num_classes: int,
        model_size: str,
        spatial_checkpoint: Path,
        spectral_checkpoint: Path,
    ) -> None:
        super().__init__()
        self.model = ss_fusion_cls.SSFusionFramework(
            img_size=img_size,
            in_channels=in_channels,
            patch_size=patch_size,
            classes=num_classes,
            model_size=model_size,
        )
        self._load_pretrained_weights(spatial_checkpoint=spatial_checkpoint, spectral_checkpoint=spectral_checkpoint)

    def _load_pretrained_weights(self, *, spatial_checkpoint: Path, spectral_checkpoint: Path) -> None:
        if not spatial_checkpoint.exists():
            raise FileNotFoundError(f"Missing HyperSIGMA spatial checkpoint: {spatial_checkpoint}")
        if not spectral_checkpoint.exists():
            raise FileNotFoundError(f"Missing HyperSIGMA spectral checkpoint: {spectral_checkpoint}")
        # HyperSIGMA's own init path already handles channel-mismatch patch embedding,
        # spatial pos_embed interpolation, and the spectral spat_map mismatch caused by
        # non-64 downstream input sizes. Reuse that logic instead of hard-merging tensors.
        with allow_argparse_namespace_unpickle():
            self.model.spat_encoder.init_weights(str(spatial_checkpoint))
        with allow_argparse_namespace_unpickle():
            self.model.spec_encoder.init_weights(str(spectral_checkpoint))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune HyperSIGMA on Indian Pines with the unified Aware protocol.")
    parser.add_argument("--data-root", type=Path, default=HYPERSIGMA_IMAGE_CLS_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "finetune_indian_pines_hypersigma")
    parser.add_argument("--hypersigma-root", type=Path, default=DEFAULT_HYPERSIGMA_ROOT)
    parser.add_argument("--spatial-checkpoint", type=Path, default=None)
    parser.add_argument("--spectral-checkpoint", type=Path, default=None)
    parser.add_argument("--protocol", choices=("hypersigma_pca",), default="hypersigma_pca")
    parser.add_argument("--window-size", type=int, default=33)
    parser.add_argument("--input-size", type=int, default=33)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--train-num", type=int, default=10)
    parser.add_argument("--val-num", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--split-type", choices=("number", "ratio"), default="number")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--scheduler-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay-rate", type=float, default=0.9)
    parser.add_argument("--model-size", choices=("base", "large", "huge"), default="base")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_count: int, precision: str) -> dict[str, Any]:
    model.eval()
    all_pred: list[np.ndarray] = []
    all_label: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    autocast_enabled = device.type == "cuda" and precision == "bf16"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        total_loss += float(loss.item()) * int(y.numel())
        total_count += int(y.numel())
        all_pred.append(pred.cpu().numpy())
        all_label.append(y.cpu().numpy())
    pred_np = np.concatenate(all_pred)
    label_np = np.concatenate(all_label)
    return metrics_from_predictions(label_np, pred_np, class_count, total_loss / max(total_count, 1))


def get_num_layer_for_hypersigma(var_name: str, num_max_layer: int) -> int:
    if ".blocks." in var_name:
        try:
            return int(var_name.split(".blocks.")[1].split(".")[0]) + 1
        except (IndexError, ValueError):
            return num_max_layer - 1
    if "patch_embed" in var_name or "pos_embed" in var_name:
        return 0
    return num_max_layer - 1


def build_layer_decay_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    layer_decay_rate: float,
    depth: int,
) -> torch.optim.Optimizer:
    groups: dict[tuple[int, float], dict[str, object]] = {}
    num_layers = int(depth) + 2
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_num_layer_for_hypersigma(name, num_layers)
        decay = 0.0 if param.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name else weight_decay
        key = (layer_id, decay)
        if key not in groups:
            scale = layer_decay_rate ** (num_layers - layer_id - 1)
            groups[key] = {"params": [], "lr": lr * scale, "weight_decay": decay}
        groups[key]["params"].append(param)
    return torch.optim.AdamW(list(groups.values()), lr=lr, betas=(0.9, 0.999))


def compact_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "loss": float(metrics["loss"]),
        "oa": float(metrics["oa"]),
        "aa": float(metrics["aa"]),
        "kappa": float(metrics["kappa"]),
    }


def resolve_default_checkpoint(args: argparse.Namespace, kind: str) -> Path:
    if kind == "spatial":
        return args.hypersigma_root / "Pretrain" / "spat-vit-base-ultra-checkpoint-1599.pth"
    if kind == "spectral":
        return args.hypersigma_root / "Pretrain" / "spec-vit-base-ultra-checkpoint-1599.pth"
    raise ValueError(f"Unsupported checkpoint kind {kind!r}.")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.window_size != args.input_size:
        raise ValueError("HyperSIGMA protocol expects input-size == window-size.")

    resolved_protocol = resolve_indian_pines_protocol(
        protocol_name=args.protocol,
        raw_band_count=200,
        pca_components=args.pca_components,
        sensor=None,
        level=None,
    )
    train_loader, val_loader, test_loader, class_count, channels, split_summary = build_indian_pines_loaders(
        data_root=args.data_root,
        protocol=resolved_protocol,
        window_size=args.window_size,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        train_num=args.train_num,
        val_num=args.val_num,
        split_type=args.split_type,
        augment=False,
        spectral_jitter_std=0.0,
    )

    spatial_checkpoint = args.spatial_checkpoint or resolve_default_checkpoint(args, "spatial")
    spectral_checkpoint = args.spectral_checkpoint or resolve_default_checkpoint(args, "spectral")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HyperSIGMAClassifier(
        img_size=args.input_size,
        in_channels=channels,
        patch_size=args.patch_size,
        num_classes=class_count,
        model_size=args.model_size,
        spatial_checkpoint=spatial_checkpoint,
        spectral_checkpoint=spectral_checkpoint,
    ).to(device)

    depth_map = {"base": 12, "large": 24, "huge": 32}
    optimizer = build_layer_decay_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        layer_decay_rate=args.layer_decay_rate,
        depth=depth_map[args.model_size],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.scheduler_epochs, eta_min=0.0)
    autocast_enabled = device.type == "cuda" and args.precision == "bf16"

    checkpoint_paths = {
        "best_train_loss": args.output_dir / "best_train_loss.pt",
        "best_val_loss": args.output_dir / "best_val_loss.pt",
        "best_val_oa": args.output_dir / "best_val_oa.pt",
        "latest": args.output_dir / "latest_classification.pt",
    }
    best_values = {"best_train_loss": math.inf, "best_val_loss": math.inf, "best_val_oa": -math.inf}
    log_path = args.output_dir / "classification_log.jsonl"

    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        payload = vars(args).copy()
        payload["resolved_protocol"] = {
            "name": resolved_protocol.name,
            "pca_components": resolved_protocol.pca_components,
            "sensor": resolved_protocol.sensor,
            "level": resolved_protocol.level,
            "wavelengths": resolved_protocol.wavelengths.tolist(),
            "fwhm": resolved_protocol.fwhm.tolist(),
            "metadata_note": resolved_protocol.metadata_note,
        }
        payload["splits"] = split_summary
        payload["resolved_spatial_checkpoint"] = str(spatial_checkpoint)
        payload["resolved_spectral_checkpoint"] = str(spectral_checkpoint)
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)

    for epoch in range(args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_count = 0
        running_correct = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = model(x)
                ce_loss = F.cross_entropy(logits, y)
            ce_loss.backward()
            optimizer.step()

            batch_count = int(y.numel())
            running_loss += float(ce_loss.item()) * batch_count
            running_ce += float(ce_loss.item()) * batch_count
            running_count += batch_count
            running_correct += int((logits.argmax(dim=1) == y).sum().item())
        scheduler.step()

        train_loss = running_loss / max(running_count, 1)
        val_metrics = evaluate(model, val_loader, device, class_count, args.precision)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ce": running_ce / max(running_count, 1),
            "train_acc": running_correct / max(running_count, 1),
            "val_loss": val_metrics["loss"],
            "val_oa": val_metrics["oa"],
            "val_aa": val_metrics["aa"],
            "val_kappa": val_metrics["kappa"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(record, ensure_ascii=True))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val": compact_metrics(val_metrics),
            "args": vars(args),
        }
        torch.save(checkpoint, checkpoint_paths["latest"])
        if train_loss < best_values["best_train_loss"]:
            best_values["best_train_loss"] = train_loss
            torch.save(checkpoint, checkpoint_paths["best_train_loss"])
        if float(val_metrics["loss"]) < best_values["best_val_loss"]:
            best_values["best_val_loss"] = float(val_metrics["loss"])
            torch.save(checkpoint, checkpoint_paths["best_val_loss"])
        if float(val_metrics["oa"]) > best_values["best_val_oa"]:
            best_values["best_val_oa"] = float(val_metrics["oa"])
            torch.save(checkpoint, checkpoint_paths["best_val_oa"])

    test_results: dict[str, Any] = {}
    for key in ("best_train_loss", "best_val_loss", "best_val_oa", "latest"):
        payload = torch.load(checkpoint_paths[key], map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        metrics = evaluate(model, test_loader, device, class_count, args.precision)
        test_results[key] = {
            "epoch": int(payload["epoch"]),
            "test": metrics,
            "selection_val": payload.get("val", {}),
            "selection_train_loss": float(payload.get("train_loss", float("nan"))),
        }

    print(json.dumps(test_results, ensure_ascii=True))
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(test_results, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
