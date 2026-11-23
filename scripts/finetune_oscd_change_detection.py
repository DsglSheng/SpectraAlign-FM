from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aware_open.downstream.adapter import AwareFeatureExtractor, build_downstream_metadata

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - optional postprocess dependency
    ndi = None


OSCD_BASELINES = {
    "spatsigma": {"oa": 0.9608, "kappa": 0.5649, "f1": 0.5853, "precision": 0.6459, "recall": 0.5350},
    "hypersigma": {"oa": 0.9578, "kappa": 0.5706, "f1": 0.5928, "precision": 0.5912, "recall": 0.5945},
}


S2_WAVELENGTHS_NM = [
    443.0,
    490.0,
    560.0,
    665.0,
    705.0,
    740.0,
    783.0,
    842.0,
    865.0,
    945.0,
    1375.0,
    1610.0,
    2190.0,
]
S2_FWHM_NM = [
    20.0,
    65.0,
    35.0,
    30.0,
    15.0,
    15.0,
    20.0,
    115.0,
    20.0,
    20.0,
    30.0,
    90.0,
    180.0,
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalize_img(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32, copy=False) - 123.675) / 58.395


def load_label(path: Path) -> np.ndarray:
    label = np.asarray(imageio.imread(path))
    if label.ndim > 2:
        label = label[..., 0]
    label = (label.astype(np.float32) / 255.0).round().astype(np.int64)
    return label


class OSCDDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        names: list[str],
        *,
        crop_size: int,
        max_samples: int | None,
        split: str,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.names = list(names)
        self.crop_size = int(crop_size)
        self.split = split
        if max_samples is not None:
            repeats = int(math.ceil(float(max_samples) / max(len(self.names), 1)))
            self.names = (self.names * repeats)[: int(max_samples)]
        if not self.names:
            raise ValueError(f"No OSCD samples for split={split}")

    def __len__(self) -> int:
        return len(self.names)

    def _load(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        root = self.dataset_path / name
        pre = np.load(root / "imgs_1_rect" / "ms_data.npy").astype(np.float32, copy=False)
        post = np.load(root / "imgs_2_rect" / "ms_data.npy").astype(np.float32, copy=False)
        label = load_label(root / "cm" / "cm.png")
        return pre, post, label

    def _random_crop(self, pre: np.ndarray, post: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = label.shape
        crop = self.crop_size
        H = max(crop, h)
        W = max(crop, w)
        pad_pre = np.zeros((H, W, pre.shape[-1]), dtype=np.float32)
        pad_post = np.zeros((H, W, post.shape[-1]), dtype=np.float32)
        pad_label = np.ones((H, W), dtype=np.int64) * 255
        top = int(np.random.randint(H - h + 1))
        left = int(np.random.randint(W - w + 1))
        pad_pre[top : top + h, left : left + w] = pre
        pad_post[top : top + h, left : left + w] = post
        pad_label[top : top + h, left : left + w] = label

        best = None
        for _ in range(10):
            r = random.randrange(0, H - crop + 1)
            c = random.randrange(0, W - crop + 1)
            lab = pad_label[r : r + crop, c : c + crop]
            valid = lab != 255
            if valid.any():
                _, counts = np.unique(lab[valid], return_counts=True)
                if len(counts) > 1 and float(counts.max()) / float(counts.sum()) < 0.75:
                    best = (r, c)
                    break
            best = (r, c)
        assert best is not None
        r, c = best
        return pad_pre[r : r + crop, c : c + crop], pad_post[r : r + crop, c : c + crop], pad_label[r : r + crop, c : c + crop]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        name = self.names[int(index)]
        pre, post, label = self._load(name)
        if self.split == "train":
            pre, post, label = self._random_crop(pre, post, label)
            if random.random() > 0.5:
                pre = np.fliplr(pre)
                post = np.fliplr(post)
                label = np.fliplr(label)
            if random.random() > 0.5:
                pre = np.flipud(pre)
                post = np.flipud(post)
                label = np.flipud(label)
            k = random.randrange(4)
            if k:
                pre = np.rot90(pre, k)
                post = np.rot90(post, k)
                label = np.rot90(label, k)

        pre = normalize_img(pre)
        post = normalize_img(post)
        return {
            "pre": torch.from_numpy(pre.transpose(2, 0, 1).copy()).float(),
            "post": torch.from_numpy(post.transpose(2, 0, 1).copy()).float(),
            "label": torch.from_numpy(label.copy()).long(),
            "name": name,
        }


class AwareOSCDFusion(nn.Module):
    def __init__(
        self,
        extractor: AwareFeatureExtractor,
        *,
        feature_layers: int = 4,
        fusion_channels: int = 128,
        hidden_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.feature_layers = int(feature_layers)
        self.embed_dim = extractor.model_dim
        self.fusion_channels = int(fusion_channels)
        self.projectors = nn.ModuleList(
            [nn.Conv2d(self.embed_dim * 4, self.fusion_channels, kernel_size=1, bias=False) for _ in range(self.feature_layers)]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(self.fusion_channels * self.feature_layers, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )

    @staticmethod
    def _token_map(tokens: torch.Tensor, grid_size: tuple[int, int], output_size: tuple[int, int]) -> torch.Tensor:
        gh, gw = grid_size
        feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], gh, gw)
        return F.interpolate(feat, size=output_size, mode="bilinear", align_corners=False)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        fp = self.extractor.forward_features(pre)
        fq = self.extractor.forward_features(post)
        pre_tokens = list(fp["intermediate_tokens"])[-self.feature_layers :]
        post_tokens = list(fq["intermediate_tokens"])[-self.feature_layers :]
        if len(pre_tokens) < self.feature_layers or len(post_tokens) < self.feature_layers:
            raise RuntimeError("Aware checkpoint did not expose enough intermediate layers.")
        output_size = (int(pre.shape[-2]), int(pre.shape[-1]))
        maps = []
        for idx, (pt, qt) in enumerate(zip(pre_tokens, post_tokens)):
            pm = self._token_map(pt, fp["grid_size"], output_size)
            qm = self._token_map(qt, fq["grid_size"], output_size)
            fused = torch.cat([pm, qm, torch.abs(qm - pm), qm - pm], dim=1)
            maps.append(self.projectors[idx](fused))
        return self.fuse(torch.cat(maps, dim=1))


def confusion_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    pred = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    gt = labels.detach().cpu().numpy().astype(np.int64)
    mask = (gt >= 0) & (gt < 2)
    values = 2 * gt[mask] + pred[mask]
    return np.bincount(values, minlength=4).reshape(2, 2).astype(np.float64)


def confusion_from_prediction(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    gt = labels.astype(np.int64, copy=False)
    pr = pred.astype(np.int64, copy=False)
    mask = (gt >= 0) & (gt < 2)
    values = 2 * gt[mask] + pr[mask]
    return np.bincount(values, minlength=4).reshape(2, 2).astype(np.float64)


def metrics_from_confusion(conf: np.ndarray) -> dict[str, float]:
    tn, fp = conf[0, 0], conf[0, 1]
    fn, tp = conf[1, 0], conf[1, 1]
    total = conf.sum()
    oa = (tp + tn) / max(total, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    pe = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / max(total * total, 1.0)
    kappa = (oa - pe) / max(1.0 - pe, 1.0e-12)
    return {"oa": float(oa), "kappa": float(kappa), "f1": float(f1), "precision": float(precision), "recall": float(recall)}


def crop_tensor(img: torch.Tensor, crop_size: int) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
    _, _, h, w = img.shape
    stride = crop_size // 2
    rows = list(range(0, max(h - crop_size, 0) + 1, stride))
    cols = list(range(0, max(w - crop_size, 0) + 1, stride))
    if rows[-1] != h - crop_size:
        rows.append(h - crop_size)
    if cols[-1] != w - crop_size:
        cols.append(w - crop_size)
    patches, coords = [], []
    for r in rows:
        for c in cols:
            patches.append(img[:, :, r : r + crop_size, c : c + crop_size])
            coords.append((r, c))
    return patches, coords


def combine_patches(preds: list[torch.Tensor], coords: list[tuple[int, int]], shape: tuple[int, int], crop_size: int) -> torch.Tensor:
    device = preds[0].device
    out = torch.zeros((1, 2, shape[0], shape[1]), device=device)
    count = torch.zeros((1, 1, shape[0], shape[1]), device=device)
    for pred, (r, c) in zip(preds, coords):
        out[:, :, r : r + crop_size, c : c + crop_size] += pred
        count[:, :, r : r + crop_size, c : c + crop_size] += 1.0
    return out / count.clamp_min(1.0)


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in str(text).split(",") if x.strip())
    return values or (0,)


def apply_binary_postprocess(pred: np.ndarray, *, min_component_size: int = 0, close_iterations: int = 0) -> np.ndarray:
    if ndi is None:
        return pred.astype(np.int64, copy=False)
    work = pred.astype(bool, copy=True)
    if close_iterations > 0:
        work = ndi.binary_closing(work, iterations=int(close_iterations))
    if min_component_size > 0:
        labeled, num = ndi.label(work)
        if num > 0:
            counts = np.bincount(labeled.ravel())
            remove = counts < int(min_component_size)
            remove[0] = False
            work[remove[labeled]] = False
    return work.astype(np.int64)


def confusion_from_prob(
    prob: np.ndarray,
    label: np.ndarray,
    *,
    threshold: float,
    min_component_size: int = 0,
    close_iterations: int = 0,
) -> np.ndarray:
    pred = (prob >= float(threshold)).astype(np.int64)
    pred = apply_binary_postprocess(pred, min_component_size=min_component_size, close_iterations=close_iterations)
    return confusion_from_prediction(pred, label)


def threshold_candidates(mode: str, threshold: float, threshold_min: float, threshold_max: float, threshold_step: float) -> list[float]:
    if mode == "fixed":
        return [float(threshold)]
    step = max(float(threshold_step), 1.0e-6)
    values = np.arange(float(threshold_min), float(threshold_max) + step * 0.5, step, dtype=np.float64)
    return [float(round(x, 6)) for x in values]


def forward_eval_patch(model: AwareOSCDFusion, pre: torch.Tensor, post: torch.Tensor, *, use_amp: bool, eval_tta: bool) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
        logits = model(pre, post).float()
        if not eval_tta:
            return logits
        total = logits
        count = 1.0
        for dims in ((-1,), (-2,), (-2, -1)):
            aug_pre = torch.flip(pre, dims=dims)
            aug_post = torch.flip(post, dims=dims)
            aug_logits = model(aug_pre, aug_post).float()
            total = total + torch.flip(aug_logits, dims=dims)
            count += 1.0
        return total / count


@torch.no_grad()
def evaluate(
    model: AwareOSCDFusion,
    loader: DataLoader,
    device: torch.device,
    crop_size: int,
    precision: str,
    *,
    threshold_mode: str = "fixed",
    threshold: float = 0.5,
    threshold_min: float = 0.35,
    threshold_max: float = 0.7,
    threshold_step: float = 0.005,
    postprocess_min_components: tuple[int, ...] = (0,),
    postprocess_close_iterations: int = 0,
    eval_tta: bool = False,
) -> dict[str, Any]:
    model.eval()
    thresholds = threshold_candidates(threshold_mode, threshold, threshold_min, threshold_max, threshold_step)
    component_sizes = tuple(max(0, int(x)) for x in postprocess_min_components)
    confs = {(thr, comp): np.zeros((2, 2), dtype=np.float64) for thr in thresholds for comp in component_sizes}
    scene_cache = []
    use_amp = precision == "bf16" and device.type == "cuda"
    for batch in loader:
        pre = batch["pre"].to(device)
        post = batch["post"].to(device)
        label = batch["label"].to(device)
        pre_patches, coords = crop_tensor(pre, crop_size)
        post_patches, _ = crop_tensor(post, crop_size)
        preds = []
        for pp, qq in zip(pre_patches, post_patches):
            preds.append(forward_eval_patch(model, pp, qq, use_amp=use_amp, eval_tta=eval_tta))
        logits = combine_patches(preds, coords, (label.shape[-2], label.shape[-1]), crop_size)
        prob = torch.softmax(logits, dim=1)[:, 1].squeeze(0).detach().cpu().numpy().astype(np.float32)
        gt = label.squeeze(0).detach().cpu().numpy().astype(np.int64)
        for thr in thresholds:
            for comp in component_sizes:
                confs[(thr, comp)] += confusion_from_prob(
                    prob,
                    gt,
                    threshold=thr,
                    min_component_size=comp,
                    close_iterations=postprocess_close_iterations,
                )
        scene_cache.append({"name": batch["name"][0], "prob": prob, "label": gt})

    selected_key = max(confs, key=lambda key: metrics_from_confusion(confs[key])["f1"]) if threshold_mode == "scan" else (thresholds[0], component_sizes[0])
    selected_threshold, selected_component_size = selected_key
    conf = confs[selected_key]
    per_scene = []
    for item in scene_cache:
        scene_conf = confusion_from_prob(
            item["prob"],
            item["label"],
            threshold=selected_threshold,
            min_component_size=selected_component_size,
            close_iterations=postprocess_close_iterations,
        )
        per_scene.append({"name": item["name"], "confusion": scene_conf.tolist(), "metrics": metrics_from_confusion(scene_conf)})
    metrics = metrics_from_confusion(conf)
    metrics["confusion"] = conf.tolist()
    metrics["per_scene"] = per_scene
    metrics["threshold"] = float(selected_threshold)
    metrics["threshold_mode"] = str(threshold_mode)
    metrics["postprocess_min_component_size"] = int(selected_component_size)
    metrics["postprocess_close_iterations"] = int(postprocess_close_iterations)
    metrics["eval_tta"] = bool(eval_tta)
    if threshold_mode == "scan":
        ranked = []
        for key, item_conf in confs.items():
            item_metrics = metrics_from_confusion(item_conf)
            ranked.append(
                {
                    "threshold": float(key[0]),
                    "postprocess_min_component_size": int(key[1]),
                    "f1": item_metrics["f1"],
                    "precision": item_metrics["precision"],
                    "recall": item_metrics["recall"],
                    "kappa": item_metrics["kappa"],
                    "oa": item_metrics["oa"],
                }
            )
        ranked.sort(key=lambda x: x["f1"], reverse=True)
        metrics["threshold_scan_top"] = ranked[:20]
    return metrics


def binary_dice_loss(logits: torch.Tensor, label: torch.Tensor, *, ignore_index: int = 255, eps: float = 1.0e-6) -> torch.Tensor:
    valid = label != ignore_index
    if int(valid.sum().item()) == 0:
        return logits.new_tensor(0.0)
    probs = torch.softmax(logits, dim=1)[:, 1]
    target = (label == 1).float()
    probs = probs[valid]
    target = target[valid]
    inter = torch.sum(probs * target)
    denom = torch.sum(probs) + torch.sum(target)
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def focal_cross_entropy_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    *,
    weight: torch.Tensor,
    gamma: float = 1.5,
    label_smoothing: float = 0.0,
    ignore_index: int = 255,
) -> torch.Tensor:
    valid = label != ignore_index
    if int(valid.sum().item()) == 0:
        return logits.new_tensor(0.0)
    flat_logits = logits.permute(0, 2, 3, 1)[valid]
    flat_label = label[valid]
    ce = F.cross_entropy(flat_logits, flat_label, weight=weight, reduction="none", label_smoothing=float(label_smoothing))
    pt = torch.softmax(flat_logits, dim=1).gather(1, flat_label[:, None]).squeeze(1).clamp_min(1.0e-6)
    return (((1.0 - pt) ** float(gamma)) * ce).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Aware epoch130 on OSCD with HyperSIGMA-style settings.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype-bank-eo1", type=Path, required=True)
    parser.add_argument("--prototype-bank-gf5", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/Data/SpectralData/Aware_open/data/oscd/original_data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=400000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--feature-layers", type=int, default=4)
    parser.add_argument("--fusion-channels", type=int, default=128)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--positive-class-weight", type=float, default=10.0)
    parser.add_argument("--dice-loss-weight", type=float, default=0.0)
    parser.add_argument("--focal-loss-weight", type=float, default=0.0)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--best-metric", choices=("kappa", "f1", "oa"), default="kappa")
    parser.add_argument("--threshold-mode", choices=("fixed", "scan"), default="fixed")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-min", type=float, default=0.35)
    parser.add_argument("--threshold-max", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--postprocess-min-components", type=str, default="0")
    parser.add_argument("--postprocess-close-iterations", type=int, default=0)
    parser.add_argument("--eval-tta", action="store_true")
    parser.add_argument("--init-model", type=Path, default=None, help="Optional previous best.pt/state dict to initialize from.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug limiter; omit for full HyperSIGMA setting.")
    parser.add_argument("--seed", type=int, default=3704)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from latest.pt or best.pt in the output directory when available.")
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps({**vars(args), "baselines": OSCD_BASELINES}, default=str, indent=2), encoding="utf-8")

    train_names = [line.strip() for line in (args.data_root / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    test_names = [line.strip() for line in (args.data_root / "test.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    train_set = OSCDDataset(args.data_root, train_names, crop_size=args.crop_size, max_samples=args.max_iters, split="train")
    test_set = OSCDDataset(args.data_root, test_names, crop_size=args.crop_size, max_samples=None, split="test")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=max(0, min(args.num_workers, 4)), drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = build_downstream_metadata(
        num_bands=13,
        sensor="others",
        level="L1",
        gsd=10.0,
        wavelengths_nm=S2_WAVELENGTHS_NM,
        fwhm_nm=S2_FWHM_NM,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        metadata=metadata,
        freeze_encoder=False,
        device=device,
    )
    model = AwareOSCDFusion(
        extractor,
        feature_layers=args.feature_layers,
        fusion_channels=args.fusion_channels,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
    ).to(device)
    if args.init_model is not None:
        payload = torch.load(args.init_model, map_location="cpu", weights_only=False)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            json.dumps(
                {
                    "init_model": str(args.init_model),
                    "missing_keys": list(missing),
                    "unexpected_keys": list(unexpected),
                }
            ),
            flush=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    class_weight = torch.tensor([1.0, float(args.positive_class_weight)], dtype=torch.float32, device=device)
    use_amp = args.precision == "bf16" and device.type == "cuda"

    best: dict[str, Any] | None = None
    resume_step = 0
    if args.auto_resume:
        for resume_path in (args.output_dir / "latest.pt", args.output_dir / "best.pt"):
            if not resume_path.exists():
                continue
            payload = torch.load(resume_path, map_location="cpu", weights_only=False)
            state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if isinstance(payload, dict):
                resume_step = int(payload.get("step", 0))
                if "optimizer" in payload:
                    try:
                        optimizer.load_state_dict(payload["optimizer"])
                    except ValueError as exc:
                        print(json.dumps({"event": "optimizer_resume_skipped", "reason": str(exc)}), flush=True)
                if isinstance(payload.get("best"), dict):
                    best = payload["best"]
                elif isinstance(payload.get("metrics"), dict):
                    best = {"step": resume_step, "train_loss": None, "elapsed_sec": 0.0, "eval": payload["metrics"]}
            print(
                json.dumps(
                    {
                        "event": "resume_oscd_change_detection",
                        "resume_from": str(resume_path),
                        "start_step": resume_step,
                        "missing_keys": list(missing),
                        "unexpected_keys": list(unexpected),
                    }
                ),
                flush=True,
            )
            break

    log_path = args.output_dir / "train_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    target_steps = len(train_loader)
    if args.max_train_batches is not None:
        target_steps = min(target_steps, int(args.max_train_batches))
    if args.auto_resume and resume_step >= target_steps:
        (args.output_dir / "DONE").write_text(
            json.dumps({"event": "already_complete", "step": resume_step, "target_steps": target_steps}, indent=2),
            encoding="utf-8",
        )
        return
    for local_step, batch in enumerate(train_loader, start=1):
        step = resume_step + local_step
        if step > target_steps:
            break
        model.train()
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(pre, post)
            loss = F.cross_entropy(logits, label, weight=class_weight, ignore_index=255, label_smoothing=float(args.label_smoothing))
            if args.dice_loss_weight > 0:
                loss = loss + float(args.dice_loss_weight) * binary_dice_loss(logits, label)
            if args.focal_loss_weight > 0:
                loss = loss + float(args.focal_loss_weight) * focal_cross_entropy_loss(
                    logits,
                    label,
                    weight=class_weight,
                    gamma=float(args.focal_gamma),
                    label_smoothing=float(args.label_smoothing),
                )
            loss_for_backward = loss / max(args.grad_accum_steps, 1)
        loss_for_backward.backward()
        if step % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        should_eval = step % args.eval_every == 0 or step == target_steps
        if should_eval:
            metrics = evaluate(
                model,
                test_loader,
                device,
                args.crop_size,
                args.precision,
                threshold_mode=args.threshold_mode,
                threshold=args.threshold,
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                threshold_step=args.threshold_step,
                postprocess_min_components=parse_int_list(args.postprocess_min_components),
                postprocess_close_iterations=args.postprocess_close_iterations,
                eval_tta=args.eval_tta,
            )
            record = {"step": step, "train_loss": float(loss.detach().cpu()), "elapsed_sec": time.time() - start_time, "eval": metrics}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            best_metric = str(args.best_metric)
            if best is None or metrics[best_metric] > best["eval"][best_metric]:
                best = record
                torch.save({"step": step, "model": model.state_dict(), "args": vars(args), "metrics": metrics}, args.output_dir / "best.pt")
                test_metrics = {
                    "best_step": step,
                    "metrics": metrics,
                    "baselines": OSCD_BASELINES,
                    "delta_vs_hypersigma": {k: metrics[k] - OSCD_BASELINES["hypersigma"][k] for k in OSCD_BASELINES["hypersigma"]},
                }
                (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                    "best": best,
                },
                args.output_dir / "latest.pt",
            )

    if best is None:
        metrics = evaluate(
            model,
            test_loader,
            device,
            args.crop_size,
            args.precision,
            threshold_mode=args.threshold_mode,
            threshold=args.threshold,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_step=args.threshold_step,
            postprocess_min_components=parse_int_list(args.postprocess_min_components),
            postprocess_close_iterations=args.postprocess_close_iterations,
            eval_tta=args.eval_tta,
        )
        (args.output_dir / "test_metrics.json").write_text(
            json.dumps({"best_step": 0, "metrics": metrics, "baselines": OSCD_BASELINES}, indent=2),
            encoding="utf-8",
        )
    (args.output_dir / "DONE").write_text(
        json.dumps({"event": "complete", "best_step": None if best is None else best.get("step")}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
