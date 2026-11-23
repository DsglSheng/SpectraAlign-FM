from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aware_open.downstream.change_detection_scene import (
    ChangePatchDataset,
    load_change_scene,
    make_change_splits,
    resolve_change_scene_config,
)
from aware_open.downstream.detection_scene import (
    build_detection_scene,
    compute_auc_scores,
    resolve_detection_scene_config,
)
from finetune_spectralgpt_classification import (
    autocast_context,
    build_spectralgpt_model,
    import_spectralgpt,
    load_spectralgpt_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight SpectralGPT adapters for binary HyperSIGMA-style downstream screens."
    )
    parser.add_argument("--mode", choices=("detection", "change"), required=True)
    parser.add_argument("--spectralgpt-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", choices=("htd", "had"), default="htd")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--overlap-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=3704)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--train-number", type=int, default=500)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--normalization", type=str, default=None)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def adapt_channels_to_12(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 12:
        return x
    b, c, h, w = x.shape
    y = x.permute(0, 2, 3, 1).reshape(-1, 1, c)
    y = F.interpolate(y, size=12, mode="linear", align_corners=False)
    return y.reshape(b, h, w, 12).permute(0, 3, 1, 2).contiguous()


class SpectralGPTBinary(nn.Module):
    def __init__(self, models_vit_tensor: Any, *, img_size: int, drop_path: float = 0.1) -> None:
        super().__init__()
        self.model = build_spectralgpt_model(
            models_vit_tensor,
            img_size=img_size,
            num_classes=2,
            drop_path=drop_path,
            dropout=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = adapt_channels_to_12(x)
        return self.model(x)


class SpectralGPTPairBinary(nn.Module):
    def __init__(self, models_vit_tensor: Any, *, img_size: int) -> None:
        super().__init__()
        self.model = build_spectralgpt_model(
            models_vit_tensor,
            img_size=img_size,
            num_classes=2,
            drop_path=0.1,
            dropout=0.0,
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return self.model(adapt_channels_to_12(t2 - t1))


def binary_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray | None = None) -> dict[str, float]:
    labels = labels.reshape(-1).astype(np.int64)
    preds = preds.reshape(-1).astype(np.int64)
    valid = labels >= 0
    labels = labels[valid]
    preds = preds[valid]
    tp = float(np.sum((labels == 1) & (preds == 1)))
    tn = float(np.sum((labels == 0) & (preds == 0)))
    fp = float(np.sum((labels == 0) & (preds == 1)))
    fn = float(np.sum((labels == 1) & (preds == 0)))
    total = max(1.0, tp + tn + fp + fn)
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    oa = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(1.0, total * total)
    kappa = (oa - pe) / max(1.0e-12, 1.0 - pe)
    out = {
        "oa": float(oa),
        "kappa": float(kappa),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }
    if probs is not None:
        try:
            from sklearn.metrics import roc_auc_score

            out["auc"] = float(roc_auc_score(labels, probs.reshape(-1)[valid]))
        except Exception:
            out["auc"] = float("nan")
    return out


def detection_targets(label_maps: torch.Tensor) -> torch.Tensor:
    valid = label_maps != 255
    positive = (label_maps == 1).flatten(1).any(dim=1)
    has_valid = valid.flatten(1).any(dim=1)
    target = positive.long()
    target[~has_valid] = 0
    return target


def run_detection(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    models_vit_tensor = import_spectralgpt(args.spectralgpt_repo)
    config = resolve_detection_scene_config(args.task, args.dataset)
    scene = build_detection_scene(
        data_root=args.data_root,
        config=config,
        img_size=args.img_size,
        overlap_size=args.overlap_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        normalization=args.normalization or "mean_std",
    )
    model = SpectralGPTBinary(models_vit_tensor, img_size=int(scene["img_size"]))
    load_info = load_spectralgpt_checkpoint(model.model, args.checkpoint)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    latest = args.output_dir / "latest.pt"
    start_epoch = 1
    if args.auto_resume and latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0)) + 1
    save_json(args.output_dir / "config.json", {**vars(args), "load_info": load_info, "scene": scene["summary"]})
    log_path = args.output_dir / "train_log.jsonl"
    best_auc = -1.0
    for epoch in range(start_epoch, args.epochs + 1):
        tic = time.time()
        model.train()
        losses: list[float] = []
        for batch in scene["train_loader"]:
            image = batch["image"].to(device)
            target = detection_targets(batch["label"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                logits = model(image)
                loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {"epoch": epoch, "elapsed_sec": time.time() - tic, "loss": float(np.mean(losses) if losses else 0.0)}
        if epoch % max(1, args.eval_every) == 0 or epoch == args.epochs:
            metrics, score_map = evaluate_detection(model, scene, device, args.precision)
            record["eval"] = metrics
            if metrics.get("auc", -1.0) > best_auc:
                best_auc = float(metrics.get("auc", -1.0))
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, args.output_dir / "best_auc.pt")
                np.save(args.output_dir / "best_auc_score_map.npy", score_map.astype(np.float32))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, latest)
    final_metrics, final_score = evaluate_detection(model, scene, device, args.precision)
    np.save(args.output_dir / "final_score_map.npy", final_score.astype(np.float32))
    save_json(args.output_dir / "test_metrics.json", final_metrics)
    return final_metrics


@torch.no_grad()
def evaluate_detection(
    model: SpectralGPTBinary,
    scene: dict[str, Any],
    device: torch.device,
    precision: str,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    h, w = int(scene["padded_height"]), int(scene["padded_width"])
    score_sum = np.zeros((h, w), dtype=np.float64)
    score_count = np.zeros((h, w), dtype=np.float64)
    offset = 0
    for batch in scene["eval_loader"]:
        image = batch["image"].to(device)
        with autocast_context(device, precision):
            prob = torch.softmax(model(image), dim=1)[:, 1].float().cpu().numpy()
        for local_idx, score in enumerate(prob):
            row, col = scene["starts"][offset + local_idx]
            size = int(scene["img_size"])
            score_sum[row : row + size, col : col + size] += float(score)
            score_count[row : row + size, col : col + size] += 1.0
        offset += len(prob)
    score = score_sum / np.maximum(score_count, 1.0)
    score = score[: int(scene["height"]), : int(scene["width"])]
    try:
        metrics = compute_auc_scores(score, scene["labels"])
    except Exception as exc:
        labels = np.asarray(scene["labels"]).reshape(-1)
        scores = score.reshape(-1)
        valid = labels >= 0
        metrics = {"auc": 0.5, "auc_error": str(exc)}
        try:
            from sklearn.metrics import roc_auc_score

            if np.unique(labels[valid]).size >= 2 and np.unique(scores[valid]).size >= 2:
                metrics["auc"] = float(roc_auc_score(labels[valid], scores[valid]))
        except Exception as inner:
            metrics["auc_error"] = f"{exc}; fallback={inner}"
    return metrics, score.astype(np.float32)


def run_change(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    models_vit_tensor = import_spectralgpt(args.spectralgpt_repo)
    config = resolve_change_scene_config(args.dataset)
    t1, t2, labels, norm_info = load_change_scene(
        data_root=args.data_root,
        config=config,
        normalization=args.normalization or "paired_minmax",
    )
    splits = make_change_splits(
        labels,
        train_number=args.train_number,
        split_seed=args.split_seed,
        split_rng="default",
    )
    train_loader = DataLoader(
        ChangePatchDataset(t1, t2, labels, np.asarray(splits["train_indices"]), patch_size=args.img_size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        ChangePatchDataset(t1, t2, labels, np.asarray(splits["eval_indices"]), patch_size=args.img_size, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model = SpectralGPTPairBinary(models_vit_tensor, img_size=args.img_size)
    load_info = load_spectralgpt_checkpoint(model.model, args.checkpoint)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    latest = args.output_dir / "latest.pt"
    start_epoch = 1
    if args.auto_resume and latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0)) + 1
    save_json(
        args.output_dir / "config.json",
        {**vars(args), "load_info": load_info, "normalization": norm_info, "split_summary": splits["summary"]},
    )
    log_path = args.output_dir / "train_log.jsonl"
    best_f1 = -1.0
    for epoch in range(start_epoch, args.epochs + 1):
        tic = time.time()
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            t1_batch = batch["t1"].to(device)
            t2_batch = batch["t2"].to(device)
            target = batch["label"].long().to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                logits = model(t1_batch, t2_batch)
                loss = F.cross_entropy(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - tic,
            "loss": float(np.mean(losses) if losses else 0.0),
        }
        should_eval = epoch % max(1, int(args.eval_every)) == 0 or epoch == args.epochs
        if should_eval:
            metrics, pred_map, prob_map = evaluate_change(model, eval_loader, labels, device, args.precision)
            record["eval"] = metrics
            if metrics["f1"] > best_f1:
                best_f1 = float(metrics["f1"])
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, args.output_dir / "best_f1.pt")
                np.save(args.output_dir / "best_f1_pred_map.npy", pred_map.astype(np.int64))
                np.save(args.output_dir / "best_f1_prob_map.npy", prob_map.astype(np.float32))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, latest)
    metrics, pred_map, prob_map = evaluate_change(model, eval_loader, labels, device, args.precision)
    np.save(args.output_dir / "final_pred_map.npy", pred_map.astype(np.int64))
    np.save(args.output_dir / "final_prob_map.npy", prob_map.astype(np.float32))
    save_json(args.output_dir / "test_metrics.json", metrics)
    return metrics


@torch.no_grad()
def evaluate_change(
    model: SpectralGPTPairBinary,
    loader: DataLoader,
    labels: np.ndarray,
    device: torch.device,
    precision: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    pred_map = np.full(labels.shape, -1, dtype=np.int64)
    prob_map = np.zeros(labels.shape, dtype=np.float32)
    for batch in loader:
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        flat = batch["index"].cpu().numpy()
        with autocast_context(device, precision):
            prob = torch.softmax(model(t1, t2), dim=1)[:, 1].float().cpu().numpy()
        pred = (prob >= 0.5).astype(np.int64)
        rows = flat // labels.shape[1]
        cols = flat % labels.shape[1]
        pred_map[rows, cols] = pred
        prob_map[rows, cols] = prob
    valid = pred_map >= 0
    metrics = binary_metrics(labels[valid], pred_map[valid], prob_map[valid])
    return metrics, pred_map, prob_map


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "detection":
        metrics = run_detection(args, device)
    else:
        metrics = run_change(args, device)
    print("[SpectralGPT downstream] Test metrics:", json.dumps(metrics, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
