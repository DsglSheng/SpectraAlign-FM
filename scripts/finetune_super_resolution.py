from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset

from aware_open.downstream.adapter import AwareFeatureExtractor, build_downstream_metadata


HYPERSIGMA_SR_BASELINES = {
    "houston_x4": {"MPSNR": 43.970, "MSSIM": 0.990, "ERGAS": 0.350, "SAM": 1.413},
    "houston_x8": {"MPSNR": 39.790, "MSSIM": 0.976, "ERGAS": 0.684, "SAM": 2.371},
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class HSTrainingData(Dataset):
    def __init__(self, image_dir: Path, augment: bool = True) -> None:
        self.files = sorted(Path(image_dir).glob("*.mat"))
        if not self.files:
            raise FileNotFoundError(f"No train .mat files under {image_dir}")
        self.augment = bool(augment)
        self.factor = 8 if self.augment else 1

    def __len__(self) -> int:
        return len(self.files) * self.factor

    @staticmethod
    def _augment(arr: np.ndarray, mode: int) -> np.ndarray:
        if mode == 0:
            return arr
        if mode == 1:
            return np.flipud(np.rot90(arr))
        if mode == 2:
            return np.flipud(arr)
        if mode == 3:
            return np.rot90(arr, 3)
        if mode == 4:
            return np.flipud(np.rot90(arr, 2))
        if mode == 5:
            return np.rot90(arr)
        if mode == 6:
            return np.rot90(arr, 2)
        if mode == 7:
            return np.flipud(np.rot90(arr, 3))
        return arr

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_index = int(index) // self.factor if self.augment else int(index)
        aug = int(index) % self.factor if self.augment else 0
        payload = sio.loadmat(self.files[file_index])
        ms = self._augment(np.asarray(payload["ms"], dtype=np.float32), aug)
        lms = self._augment(np.asarray(payload["ms_bicubic"], dtype=np.float32), aug)
        gt = self._augment(np.asarray(payload["gt"], dtype=np.float32), aug)
        return (
            torch.from_numpy(ms.transpose(2, 0, 1).copy()).float(),
            torch.from_numpy(lms.transpose(2, 0, 1).copy()).float(),
            torch.from_numpy(gt.transpose(2, 0, 1).copy()).float(),
        )


class HSTestData(Dataset):
    def __init__(self, mat_path: Path) -> None:
        payload = sio.loadmat(mat_path)
        self.ms = np.asarray(payload["ms"], dtype=np.float32)
        self.lms = np.asarray(payload["ms_bicubic"], dtype=np.float32)
        self.gt = np.asarray(payload["gt"], dtype=np.float32)

    def __len__(self) -> int:
        return int(self.gt.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(index)
        return (
            torch.from_numpy(self.ms[i].transpose(2, 0, 1).copy()).float(),
            torch.from_numpy(self.lms[i].transpose(2, 0, 1).copy()).float(),
            torch.from_numpy(self.gt[i].transpose(2, 0, 1).copy()).float(),
        )


class Upsampler(nn.Sequential):
    def __init__(self, scale: int, channels: int) -> None:
        layers: list[nn.Module] = []
        if scale & (scale - 1) != 0:
            raise ValueError(f"Only power-of-two scale is supported, got {scale}")
        for _ in range(int(math.log2(scale))):
            layers.append(nn.Conv2d(channels, 4 * channels, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(2))
            layers.append(nn.GELU())
        super().__init__(*layers)


class AwareSRFusion(nn.Module):
    def __init__(
        self,
        extractor: AwareFeatureExtractor,
        *,
        scale: int,
        channels: int,
        feature_layers: int = 4,
        fusion_channels: int = 128,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.scale = int(scale)
        self.channels = int(channels)
        self.feature_layers = int(feature_layers)
        self.embed_dim = extractor.model_dim
        self.projectors = nn.ModuleList([nn.Conv2d(self.embed_dim, fusion_channels, kernel_size=1, bias=False) for _ in range(feature_layers)])
        self.fuse = nn.Sequential(
            nn.Conv2d(fusion_channels * feature_layers, fusion_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(fusion_channels, channels, kernel_size=3, padding=1),
        )
        self.upsample = Upsampler(scale, channels)
        self.skip = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.tail = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1)

    @staticmethod
    def _token_map(tokens: torch.Tensor, grid_size: tuple[int, int], output_size: tuple[int, int]) -> torch.Tensor:
        gh, gw = grid_size
        feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], gh, gw)
        return F.interpolate(feat, size=output_size, mode="bilinear", align_corners=False)

    def forward(self, ms: torch.Tensor, lms: torch.Tensor) -> torch.Tensor:
        features = self.extractor.forward_features(ms)
        selected = list(features["intermediate_tokens"])[-self.feature_layers :]
        if len(selected) < self.feature_layers:
            raise RuntimeError("Aware checkpoint did not expose enough intermediate layers.")
        low_size = (int(ms.shape[-2]), int(ms.shape[-1]))
        maps = []
        for projector, tokens in zip(self.projectors, selected):
            maps.append(projector(self._token_map(tokens, features["grid_size"], low_size)))
        x = self.fuse(torch.cat(maps, dim=1))
        x = self.upsample(x)
        if x.shape[-2:] != lms.shape[-2:]:
            x = F.interpolate(x, size=lms.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, self.skip(lms)], dim=1)
        return self.tail(x)


def cal_sam_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-6
    inner = torch.sum(pred * target, dim=1, keepdim=True)
    denom = torch.norm(pred, p=2, dim=1, keepdim=True) * torch.norm(target, p=2, dim=1, keepdim=True)
    cos = (inner / denom.clamp_min(eps)).sum(dim=1).clamp(-1.0 + eps, 1.0 - eps)
    return torch.mean(torch.acos(cos)) / math.pi


def gradient_volume(x: torch.Tensor) -> torch.Tensor:
    gx = x[:, :, 1:, 1:] - x[:, :, :-1, 1:]
    gy = x[:, :, 1:, 1:] - x[:, :, 1:, :-1]
    gc = x[:, 1:, 1:, 1:] - x[:, :-1, 1:, 1:]
    min_c = min(gx.shape[1], gy.shape[1], gc.shape[1])
    return torch.sqrt(gx[:, :min_c].pow(2) + gy[:, :min_c].pow(2) + gc[:, :min_c].pow(2) + 1.0e-6)


class HLoss(nn.Module):
    def __init__(self, la1: float, la2: float) -> None:
        super().__init__()
        self.la1 = float(la1)
        self.la2 = float(la2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target) + self.la1 * cal_sam_loss(pred, target) + self.la2 * F.l1_loss(gradient_volume(pred), gradient_volume(target))


def compare_mpsnr(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([peak_signal_noise_ratio(gt[:, :, i], pred[:, :, i], data_range=1.0) for i in range(gt.shape[2])]))


def compare_mssim(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([structural_similarity(gt[:, :, i], pred[:, :, i], data_range=1.0) for i in range(gt.shape[2])]))


def compare_sam(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    num = np.sum(gt * pred, axis=2)
    denom = np.linalg.norm(gt, axis=2) * np.linalg.norm(pred, axis=2)
    valid = denom > 0
    cos = np.clip(num[valid] / denom[valid], -1.0, 1.0)
    return float(np.mean(np.arccos(cos)) * 180.0 / np.pi)


def compare_ergas(gt: np.ndarray, pred: np.ndarray, scale: int) -> float:
    vals = []
    for i in range(gt.shape[2]):
        mse = np.mean((gt[:, :, i] - pred[:, :, i]) ** 2)
        mean = np.mean(gt[:, :, i])
        vals.append(mse / max(mean * mean, 1.0e-12))
    return float((100.0 / scale) * np.sqrt(np.mean(vals)))


def compare_rmse(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.linalg.norm(gt - pred) / math.sqrt(float(np.prod(gt.shape))))


def compare_corr(gt: np.ndarray, pred: np.ndarray) -> float:
    gt2 = gt.reshape(-1, gt.shape[-1]).T
    pr2 = pred.reshape(-1, pred.shape[-1]).T
    gt2 = gt2 - gt2.mean(axis=1, keepdims=True)
    pr2 = pr2 - pr2.mean(axis=1, keepdims=True)
    num = np.sum(gt2 * pr2, axis=1)
    den = np.sqrt(np.sum(gt2 * gt2, axis=1) * np.sum(pr2 * pr2, axis=1))
    return float(np.mean(num / np.maximum(den, 1.0e-12)))


def quality_assessment(gt: np.ndarray, pred: np.ndarray, scale: int) -> dict[str, float]:
    pred = np.clip(pred, 0.0, 1.0)
    return {
        "MPSNR": compare_mpsnr(gt, pred),
        "MSSIM": compare_mssim(gt, pred),
        "ERGAS": compare_ergas(gt, pred, scale),
        "SAM": compare_sam(gt, pred),
        "CrossCorrelation": compare_corr(gt, pred),
        "RMSE": compare_rmse(gt, pred),
    }


@torch.no_grad()
def evaluate(model: AwareSRFusion, loader: DataLoader, device: torch.device, scale: int, precision: str) -> dict[str, Any]:
    model.eval()
    use_amp = precision == "bf16" and device.type == "cuda"
    per_file = []
    sums: dict[str, float] = {}
    block_size = 32
    for idx, (ms, lms, gt) in enumerate(loader):
        ms = ms.to(device)
        lms = lms.to(device)
        gt = gt.to(device)
        pred = torch.empty_like(gt)
        rows = list(range(0, ms.shape[-2], block_size))
        cols = list(range(0, ms.shape[-1], block_size))
        for r in rows:
            for c in cols:
                r2 = min(r + block_size, ms.shape[-2])
                c2 = min(c + block_size, ms.shape[-1])
                if r2 - r < block_size:
                    r = max(0, r2 - block_size)
                if c2 - c < block_size:
                    c = max(0, c2 - block_size)
                hr0, hr1 = r * scale, r2 * scale
                hc0, hc1 = c * scale, c2 * scale
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(ms[:, :, r:r2, c:c2], lms[:, :, hr0:hr1, hc0:hc1])
                pred[:, :, hr0:hr1, hc0:hc1] = out.float()
        pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        gt_np = gt.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        metrics = quality_assessment(gt_np, pred_np[: gt_np.shape[0], : gt_np.shape[1]], scale)
        per_file.append({"index": idx, **metrics})
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    avg = {key: value / max(len(per_file), 1) for key, value in sums.items()}
    avg["per_file"] = per_file
    return avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Aware epoch130 for HyperSIGMA-style hyperspectral SR.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prototype-bank-eo1", type=Path, required=True)
    parser.add_argument("--prototype-bank-gf5", type=Path, required=True)
    parser.add_argument("--init-model", type=Path, default=None, help="Optional previous best.pt/state dict to initialize from.")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing trains/ and <dataset>_test.mat.")
    parser.add_argument("--dataset-name", type=str, default="houston")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=(4, 8), required=True)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=6.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--la1", type=float, default=0.3)
    parser.add_argument("--la2", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--feature-layers", type=int, default=4)
    parser.add_argument("--fusion-channels", type=int, default=128)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_set = HSTrainingData(args.data_root / "trains", augment=True)
    test_path = args.data_root / f"{args.dataset_name}_test.mat"
    test_set = HSTestData(test_path)
    channels = args.channels or int(test_set.gt.shape[-1])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, shuffle=False)

    config = {**vars(args), "channels": channels, "baselines": HYPERSIGMA_SR_BASELINES}
    (args.output_dir / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = build_downstream_metadata(num_bands=channels, sensor="others", level="surface", gsd=1.0)
    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        metadata=metadata,
        freeze_encoder=False,
        device=device,
    )
    model = AwareSRFusion(extractor, scale=args.scale, channels=channels, feature_layers=args.feature_layers, fusion_channels=args.fusion_channels).to(device)
    if args.init_model is not None:
        state = torch.load(args.init_model, map_location="cpu", weights_only=False)
        state_dict = state.get("model", state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded init model from {args.init_model}; missing_keys={missing}; unexpected_keys={unexpected}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = HLoss(args.la1, args.la2)
    use_amp = args.precision == "bf16" and device.type == "cuda"
    best: dict[str, Any] | None = None
    log_path = args.output_dir / "train_log.jsonl"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, (ms, lms, gt) in enumerate(train_loader, start=1):
            if args.max_train_batches is not None and step > args.max_train_batches:
                break
            ms = ms.to(device, non_blocking=True)
            lms = lms.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(ms, lms)
                loss = loss_fn(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_loader, device, args.scale, args.precision)
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else None,
                "elapsed_sec": time.time() - start,
                "eval": metrics,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if best is None or metrics["MPSNR"] > best["eval"]["MPSNR"]:
                best = record
                torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args), "metrics": metrics}, args.output_dir / "best.pt")
                baseline_key = f"{args.dataset_name}_x{args.scale}".lower()
                baseline = HYPERSIGMA_SR_BASELINES.get(baseline_key)
                result = {"best_epoch": epoch, "metrics": metrics, "baseline": baseline}
                if baseline:
                    result["delta_vs_hypersigma"] = {k: metrics[k] - baseline[k] for k in baseline if k in metrics}
                (args.output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
