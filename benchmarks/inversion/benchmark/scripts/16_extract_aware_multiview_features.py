from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model_adapters import AwareBackbone


DEFAULTS = {
    "aware_repo": Path("/mnt/Data/SpectralData/Aware_open"),
    "aware_checkpoint": Path(
        "/mnt/Data/SpectralData/Aware_open/runs/pretrain_ours_v3wa_kl003_full450k_20260505/"
        "branches/ours_v3wa_kl003_full450k/checkpoints/best_total.pt"
    ),
    "prototype_bank_eo1": Path(
        "/mnt/Data/SpectralData/Aware_open/artifacts/prototypes_v3_core_20260503/prototype_bank_eo1.pt"
    ),
    "prototype_bank_gf5": Path(
        "/mnt/Data/SpectralData/Aware_open/artifacts/prototypes_v3_core_20260503/prototype_bank_gf5.pt"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen Aware-HSI single-token and four-token spectroscopy features."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--force", action="store_true")
    for name, value in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=value)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_mean_std(spectrum: torch.Tensor) -> torch.Tensor:
    mean = spectrum.mean(dim=1, keepdim=True)
    std = spectrum.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return (spectrum - mean) / std


def smooth_spectrum(spectrum: torch.Tensor, kernel_size: int) -> torch.Tensor:
    pad = kernel_size // 2
    padded = F.pad(spectrum[:, None, :], (pad, pad), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1).squeeze(1)


def difference(spectrum: torch.Tensor, order: int) -> torch.Tensor:
    output = spectrum
    for _ in range(order):
        output = F.pad(output[:, 1:] - output[:, :-1], (1, 0))
    return output


def build_multiview_cube(spectrum: torch.Tensor, patch_size: int) -> torch.Tensor:
    native = sample_mean_std(spectrum)
    smooth = sample_mean_std(smooth_spectrum(native, kernel_size=9))
    first = sample_mean_std(difference(native, order=1))
    second = sample_mean_std(difference(native, order=2))
    cells = torch.stack((native, smooth, first, second), dim=-1).reshape(
        spectrum.shape[0], spectrum.shape[1], 2, 2
    )
    return cells.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)


def feature_tokens(features: dict[str, Any]) -> torch.Tensor:
    intermediate = features["intermediate_tokens"]
    patch_tokens = features["patch_tokens"]
    cls_token = features["cls_token"]
    if not isinstance(intermediate, list) or not isinstance(patch_tokens, torch.Tensor):
        raise TypeError("Aware feature output has an unexpected structure.")
    return torch.cat([*intermediate, patch_tokens, cls_token.unsqueeze(1)], dim=1)


def main() -> int:
    args = parse_args()
    if args.output_npz.exists() and not args.force:
        print(f"[skip] exists: {args.output_npz}", flush=True)
        return 0
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(args.dataset_npz, allow_pickle=True) as payload:
        spectra = np.asarray(payload["spectra"], dtype=np.float32)
        wavelengths_nm = np.asarray(payload["wavelengths_nm"], dtype=np.float32)
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        dataset_name = str(np.asarray(payload["dataset"]).item())

    backbone = AwareBackbone(
        aware_repo=args.aware_repo,
        checkpoint=args.aware_checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        wavelengths_nm=torch.from_numpy(wavelengths_nm),
        sensor=str(metadata.get("sensor", "field")),
        level=str(metadata.get("level", "field")),
        gsd=float(metadata.get("gsd", 1.0)),
        device=device,
    ).to(device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    loader = DataLoader(
        TensorDataset(torch.from_numpy(spectra)),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )
    single_outputs: list[np.ndarray] = []
    multiview_outputs: list[np.ndarray] = []
    started = time.time()
    with torch.no_grad():
        for batch_index, (batch,) in enumerate(loader, start=1):
            batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
            native = sample_mean_std(batch)
            single_cube = native[:, :, None, None].expand(
                -1, -1, backbone.patch_size, backbone.patch_size
            ).contiguous()
            multiview_cube = build_multiview_cube(batch, backbone.patch_size)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and args.precision == "bf16",
            ):
                single = feature_tokens(backbone.extractor.forward_features(single_cube))
                multiview = feature_tokens(backbone.extractor.forward_features(multiview_cube))
            single_outputs.append(single.float().cpu().numpy().astype(np.float16))
            multiview_outputs.append(multiview.float().cpu().numpy().astype(np.float16))
            if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
                print(f"[extract] batch {batch_index}/{len(loader)}", flush=True)

    single_tokens = np.concatenate(single_outputs)
    multiview_tokens = np.concatenate(multiview_outputs)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        dataset=np.array(dataset_name),
        single_tokens=single_tokens,
        multiview_tokens=multiview_tokens,
        normalization=np.array("per-sample global mean/std, matching Aware-HSI pretraining"),
        multiview=np.array("2x2 cells: native spectrum, 9-band smooth, first difference, second difference"),
    )
    summary = {
        "dataset": dataset_name,
        "samples": int(spectra.shape[0]),
        "single_tokens": list(single_tokens.shape),
        "multiview_tokens": list(multiview_tokens.shape),
        "dtype": str(multiview_tokens.dtype),
        "seconds": time.time() - started,
    }
    args.output_npz.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
