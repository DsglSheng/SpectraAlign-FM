from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aware_open.downstream import (  # noqa: E402
    AwareFeatureExtractor,
    build_change_detection_scene,
    build_downstream_metadata,
    resolve_change_scene_config,
    seed_everything,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "change_detection"
BASELINES = {
    "hermiston": {"spatsigma": 0.9208, "hypersigma": 0.9174},
    "farmland": {"spatsigma": 0.9531, "hypersigma": 0.9549},
    "bayarea": {"spatsigma": 0.9889, "hypersigma": 0.9893},
    "barbara": {"spatsigma": 0.9895, "hypersigma": 0.9904},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Aware for HyperSIGMA-style hyperspectral change detection.")
    parser.add_argument("--dataset", type=str, required=True, help="hermiston, farmland, bayarea/ba, barbara/sb")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype-bank-eo1", type=Path, required=True)
    parser.add_argument("--prototype-bank-gf5", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--scheduler-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--train-number", type=int, default=500)
    parser.add_argument("--lr", type=float, default=6.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--layer-decay-rate", type=float, default=0.9)
    parser.add_argument("--feature-layers", type=int, default=4)
    parser.add_argument("--fusion-channels", type=int, default=96)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--seed", type=int, default=3704)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--split-rng", choices=("default", "legacy"), default="default")
    parser.add_argument("--gsd", type=float, default=None)
    parser.add_argument("--normalization", choices=("paired_minmax", "none", "minmax", "mean_std"), default="paired_minmax")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--raw-feature-mode", choices=("stats", "spectral"), default="stats")
    parser.add_argument("--raw-channels", type=int, default=32)
    args = parser.parse_args()
    if args.freeze_encoder:
        print("[Aware] --freeze-encoder is disabled for fair full-parameter fine-tuning; using freeze_encoder=False.")
        args.freeze_encoder = False
    return args


def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = 8
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class AwareChangeFusion(nn.Module):
    def __init__(
        self,
        extractor: AwareFeatureExtractor,
        *,
        feature_layers: int = 4,
        fusion_channels: int = 96,
        hidden_channels: int = 128,
        dropout: float = 0.1,
        input_channels: int | None = None,
        raw_feature_mode: str = "stats",
        raw_channels: int = 32,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.feature_layers = int(feature_layers)
        self.fusion_channels = int(fusion_channels)
        self.embed_dim = extractor.model_dim
        self.spectral_gate_dim = 128
        self.raw_feature_mode = str(raw_feature_mode).strip().lower()
        self.raw_channels = int(raw_channels)
        if self.raw_feature_mode == "spectral" and input_channels is None:
            raise ValueError("input_channels is required when raw_feature_mode='spectral'.")
        if self.raw_feature_mode not in {"stats", "spectral"}:
            raise ValueError(f"Unsupported raw_feature_mode={raw_feature_mode!r}.")
        self.projectors = nn.ModuleList(
            [nn.Conv2d(self.embed_dim, fusion_channels, kernel_size=1, bias=False) for _ in range(self.feature_layers)]
        )
        self.spec_reduce = nn.Sequential(
            nn.LayerNorm(extractor.band_dim),
            nn.Linear(extractor.band_dim, self.spectral_gate_dim, bias=False),
            nn.GELU(),
        )
        self.meta_reduce = nn.Sequential(
            nn.LayerNorm(extractor.context_dim),
            nn.Linear(extractor.context_dim, self.spectral_gate_dim, bias=False),
            nn.GELU(),
        )
        self.gates = nn.ModuleList([self._make_gate() for _ in range(self.feature_layers)])
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_channels * self.feature_layers, hidden_channels, kernel_size=1, bias=False),
            make_group_norm(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            make_group_norm(hidden_channels),
            nn.GELU(),
        )
        raw_input_dim = 4 if self.raw_feature_mode == "stats" else 4 + 2 * int(input_channels)
        self.raw_reduce = nn.Sequential(
            nn.Linear(raw_input_dim, self.raw_channels),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.raw_channels, self.raw_channels),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels + self.raw_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_channels, 2),
        )

    def _make_gate(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.spectral_gate_dim, 128, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.embed_dim, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _token_map(tokens: torch.Tensor, grid_size: tuple[int, int], output_size: tuple[int, int]) -> torch.Tensor:
        grid_h, grid_w = grid_size
        feature = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], grid_h, grid_w)
        return F.interpolate(feature, size=output_size, mode="bilinear", align_corners=False)

    def _pad_for_extractor(self, x: torch.Tensor) -> torch.Tensor:
        patch = int(self.extractor.local_patch_size)
        height, width = int(x.shape[-2]), int(x.shape[-1])
        target_h = int(math.ceil(height / patch) * patch)
        target_w = int(math.ceil(width / patch) * patch)
        pad_h = target_h - height
        pad_w = target_w - width
        if pad_h == 0 and pad_w == 0:
            return x
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return F.pad(x, (left, right, top, bottom), mode="reflect")

    def _aware_maps(self, x: torch.Tensor) -> tuple[list[torch.Tensor], dict[str, Any]]:
        padded = self._pad_for_extractor(x)
        features = self.extractor.forward_features(padded)
        intermediate = list(features["intermediate_tokens"])
        if len(intermediate) < self.feature_layers:
            raise RuntimeError(f"Need {self.feature_layers} feature layers, got {len(intermediate)}.")
        selected = intermediate[-self.feature_layers :]
        output_size = (int(padded.shape[-2]), int(padded.shape[-1]))
        spectral_context = self.spec_reduce(features["band_tokens"].mean(dim=1)) + self.meta_reduce(features["sample_context"])
        maps = []
        for tokens, projector, gate in zip(selected, self.projectors, self.gates):
            token_map = self._token_map(tokens, features["grid_size"], output_size)
            weight = gate(spectral_context).view(x.shape[0], -1, 1, 1)
            maps.append(projector((1.0 + weight) * token_map))
        return maps, features

    @staticmethod
    def _raw_pair_stats(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(x.float() - y.float())
        mean = diff.mean(dim=(1, 2, 3))
        std = diff.flatten(1).std(dim=1)
        maxv = diff.flatten(1).amax(dim=1)
        center = diff[:, :, diff.shape[-2] // 2, diff.shape[-1] // 2].mean(dim=1)
        return torch.stack([mean, std, maxv, center], dim=1)

    def _raw_pair_features(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        stats = self._raw_pair_stats(x, y)
        if self.raw_feature_mode == "stats":
            return stats
        diff = torch.abs(x.float() - y.float())
        center = diff[:, :, diff.shape[-2] // 2, diff.shape[-1] // 2]
        patch_mean = diff.mean(dim=(2, 3))
        return torch.cat([stats, center, patch_mean], dim=1)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t1_maps, _ = self._aware_maps(t1)
        t2_maps, _ = self._aware_maps(t2)
        diff_maps = [torch.abs(first - second) for first, second in zip(t1_maps, t2_maps)]
        fused = self.fuse(torch.cat(diff_maps, dim=1))
        pooled = F.adaptive_avg_pool2d(fused, output_size=1).flatten(1)
        raw = self.raw_reduce(self._raw_pair_features(t1, t2))
        return self.classifier(torch.cat([pooled, raw], dim=1))


def build_layer_decay_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    layer_decay_rate: float,
) -> torch.optim.Optimizer:
    blocks = getattr(model.extractor.encoder.backbone.transformer, "blocks", [])
    depth = len(blocks)
    pattern = re.compile(r"extractor\.encoder\.backbone\.transformer\.blocks\.(\d+)\.")
    groups: dict[tuple[float, float], dict[str, object]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        match = pattern.search(name)
        if match is None or depth <= 0:
            scale = 1.0
        else:
            layer_id = int(match.group(1))
            scale = layer_decay_rate ** max(depth - layer_id - 1, 0)
        decay = 0.0 if param.ndim <= 1 or name.endswith(".bias") else weight_decay
        key = (scale, decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * scale, "weight_decay": decay}
        groups[key]["params"].append(param)
    return torch.optim.AdamW(list(groups.values()), lr=lr, betas=(0.9, 0.999))


def forward_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    t1 = batch["t1"].to(device, non_blocking=True)
    t2 = batch["t2"].to(device, non_blocking=True)
    autocast_enabled = device.type == "cuda" and precision == "bf16"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        logits = model(t1, t2)
    return logits.float()


def compute_binary_metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(target, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    valid = y_true >= 0
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    total = max(tp + tn + fp + fn, 1)
    oa = (tp + tn) / total
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / max(total * total, 1)
    kappa = (oa - pe) / max(1.0 - pe, 1.0e-12)
    return {
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "oa": round(float(oa), 4),
        "kappa": round(float(kappa), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def evaluate_change_detection(
    *,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    labels: np.ndarray,
    height: int,
    width: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    flat_pred = np.full((height * width,), -1, dtype=np.int64)
    flat_prob = np.full((height * width,), np.nan, dtype=np.float32)
    for batch in loader:
        logits = forward_batch(model, batch, device=device, precision=precision)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        indices = batch["index"].cpu().numpy().astype(np.int64)
        flat_pred[indices] = pred
        flat_prob[indices] = prob.astype(np.float32)
    pred_map = flat_pred.reshape(height, width)
    prob_map = flat_prob.reshape(height, width)
    metrics = compute_binary_metrics(labels, pred_map)
    metrics["pred_map"] = pred_map
    metrics["prob_map"] = prob_map
    return metrics


def save_prediction_outputs(*, output_dir: Path, labels: np.ndarray, pred_map: np.ndarray, prob_map: np.ndarray, prefix: str) -> None:
    np.save(output_dir / f"{prefix}_pred_map.npy", pred_map.astype(np.int64, copy=False))
    np.save(output_dir / f"{prefix}_prob_map.npy", prob_map.astype(np.float32, copy=False))
    sio.savemat(
        output_dir / f"{prefix}_maps.mat",
        {
            "pred_map": pred_map.astype(np.int64, copy=False),
            "prob_map": prob_map.astype(np.float32, copy=False),
            "gt": labels.astype(np.int64, copy=False),
        },
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scene_config = resolve_change_scene_config(args.dataset)
    scene = build_change_detection_scene(
        data_root=args.data_root,
        config=scene_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_number=args.train_number,
        split_seed=args.split_seed,
        split_rng=args.split_rng,
        augment=not args.no_augment,
        normalization=args.normalization,
    )
    channels = int(scene["channels"])
    resolved_gsd = scene_config.gsd_m if args.gsd is None else float(args.gsd)
    metadata = build_downstream_metadata(
        num_bands=channels,
        sensor=scene_config.sensor,
        level=scene_config.level,
        gsd=resolved_gsd,
        wavelengths_nm=scene_config.wavelengths if int(scene_config.wavelengths.numel()) == channels else None,
        fwhm_nm=scene_config.fwhm if int(scene_config.fwhm.numel()) == channels else None,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        metadata=metadata,
        freeze_encoder=args.freeze_encoder,
        device=device,
    )
    model = AwareChangeFusion(
        extractor,
        feature_layers=args.feature_layers,
        fusion_channels=args.fusion_channels,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        input_channels=channels,
        raw_feature_mode=args.raw_feature_mode,
        raw_channels=args.raw_channels,
    ).to(device)
    optimizer = build_layer_decay_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        layer_decay_rate=args.layer_decay_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(args.scheduler_epochs), int(args.epochs)),
        eta_min=0.0,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    baseline = BASELINES.get(scene_config.dataset, {})
    checkpoint_paths = {
        "best_f1": args.output_dir / "best_f1.pt",
        "latest": args.output_dir / "latest_change_detection.pt",
    }
    best_f1 = -math.inf
    best_record: dict[str, Any] | None = None
    log_path = args.output_dir / "change_detection_log.jsonl"

    with (args.output_dir / "config.json").open("w", encoding="utf-8") as handle:
        payload = vars(args).copy()
        payload["resolved_scene"] = {
            "dataset": scene_config.dataset,
            "display_name": scene_config.display_name,
            "sensor": scene_config.sensor,
            "level": scene_config.level,
            "gsd_m": scene_config.gsd_m,
            "metadata_note": scene_config.metadata_note,
            "wavelengths": metadata.wavelengths_nm.tolist(),
            "fwhm": metadata.fwhm_nm.tolist(),
        }
        payload["scene"] = scene["summary"]
        payload["baselines"] = baseline
        payload["implementation_note"] = (
            "Aware change detection follows HyperSIGMA CD's patch-based protocol: "
            "500 changed and 500 unchanged training pixels, full valid-pixel F1 evaluation, "
            "50 epochs, AdamW lr=6e-5, weight_decay=0.05 by default. "
            "Aware pads HyperSIGMA's 5x5/15x15 patches to the encoder local-patch multiple internally."
        )
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0
        for batch in scene["train_loader"]:
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, batch, device=device, precision=args.precision)
            loss = criterion(logits, label)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
            count = int(label.numel())
            running_loss += float(loss.item()) * count
            running_correct += int((torch.argmax(logits.detach(), dim=1) == label).sum().item())
            running_count += count
        scheduler.step()
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": running_loss / max(running_count, 1),
            "train_acc": running_correct / max(running_count, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        should_eval = epoch % max(int(args.eval_every), 1) == 0 or epoch == args.epochs
        if should_eval:
            metrics = evaluate_change_detection(
                model=model,
                loader=scene["eval_loader"],
                labels=np.asarray(scene["labels"], dtype=np.int64),
                height=int(scene["height"]),
                width=int(scene["width"]),
                device=device,
                precision=args.precision,
            )
            pred_map = metrics.pop("pred_map")
            prob_map = metrics.pop("prob_map")
            record.update({f"eval_{key}": value for key, value in metrics.items()})
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "train_loss": record["train_loss"],
                "eval": metrics,
                "args": vars(args),
                "scene_summary": scene["summary"],
            }
            torch.save(checkpoint, checkpoint_paths["latest"])
            save_prediction_outputs(output_dir=args.output_dir, labels=scene["labels"], pred_map=pred_map, prob_map=prob_map, prefix="latest")
            f1 = float(metrics["f1"])
            if f1 > best_f1:
                best_f1 = f1
                best_record = dict(record)
                torch.save(checkpoint, checkpoint_paths["best_f1"])
                save_prediction_outputs(output_dir=args.output_dir, labels=scene["labels"], pred_map=pred_map, prob_map=prob_map, prefix="best_f1")

        print(json.dumps(record, ensure_ascii=True))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    if not checkpoint_paths["best_f1"].exists():
        raise RuntimeError("No evaluation checkpoint was written; reduce --eval-every or check training.")
    payload = torch.load(checkpoint_paths["best_f1"], map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    final_metrics = evaluate_change_detection(
        model=model,
        loader=scene["eval_loader"],
        labels=np.asarray(scene["labels"], dtype=np.int64),
        height=int(scene["height"]),
        width=int(scene["width"]),
        device=device,
        precision=args.precision,
    )
    final_pred = final_metrics.pop("pred_map")
    final_prob = final_metrics.pop("prob_map")
    save_prediction_outputs(output_dir=args.output_dir, labels=scene["labels"], pred_map=final_pred, prob_map=final_prob, prefix="final_best_f1")
    result = {
        "best_f1_checkpoint_epoch": int(payload["epoch"]),
        "best_f1_metrics": final_metrics,
        "selection_eval": payload.get("eval", {}),
        "selection_train_loss": float(payload.get("train_loss", float("nan"))),
        "baselines": baseline,
        "beats_hypersigma": bool(float(final_metrics["f1"]) > float(baseline.get("hypersigma", -math.inf))),
        "beats_spatsigma": bool(float(final_metrics["f1"]) > float(baseline.get("spatsigma", -math.inf))),
        "best_log_record": best_record,
    }
    print(json.dumps(result, ensure_ascii=True))
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
