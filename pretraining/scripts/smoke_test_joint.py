#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from aware_open.config import AwareEncoderConfig, FoundationConfig, FoundationDecoderConfig
from aware_open.data.pretrain import Sigma450KShardDataset, collate_modality_samples
from aware_open.models.foundation import MetadataConditionedFoundationModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate legacy count and a four-sensor mixed forward pass.")
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--joint-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-legacy-count", type=int, default=447_072)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def first_index_by_sensor(dataset: Sigma450KShardDataset) -> dict[str, int]:
    output: dict[str, int] = {}
    for shard_index, shard in enumerate(dataset.shards):
        output.setdefault(shard.sensor, dataset.sample_offsets[shard_index])
    return output


def finite_scalars(values: dict[str, torch.Tensor]) -> bool:
    return all(torch.isfinite(value).all().item() for value in values.values() if torch.is_tensor(value))


def main() -> None:
    args = parse_args()
    legacy = Sigma450KShardDataset(args.legacy_root, crop_size=64, random_crop=False, normalization="mean_std")
    if len(legacy) != args.expected_legacy_count:
        raise AssertionError("Legacy sample count changed: {}".format(len(legacy)))
    joint = Sigma450KShardDataset(
        args.joint_root,
        crop_size=64,
        random_crop=False,
        normalization="mean_std",
        manifest_path=args.joint_manifest,
    )
    indices = first_index_by_sensor(joint)
    required = {"eo1", "gf5", "dz1_vnir", "dz1_swir"}
    missing = sorted(required - set(indices))
    if missing:
        raise AssertionError("Joint manifest is missing sensors: {}".format(missing))
    samples = [joint[indices[sensor]] for sensor in ("eo1", "gf5", "dz1_vnir", "dz1_swir")]
    batch = collate_modality_samples(samples)
    if tuple(batch.values.shape[2:]) != (64, 64):
        raise AssertionError("Unexpected mixed spatial shape: {}".format(tuple(batch.values.shape)))

    prototype_count = 16
    generator = torch.Generator().manual_seed(20260705)
    banks = {
        sensor: (
            torch.randn(prototype_count, sample.values.shape[0], generator=generator),
            sample.wavelengths.clone(),
        )
        for sensor, sample in zip(("eo1", "gf5", "dz1_vnir", "dz1_swir"), samples)
    }
    config = FoundationConfig(
        encoder=AwareEncoderConfig(
            model_dim=128,
            band_dim=32,
            context_dim=64,
            spectral_dim=32,
            spatial_dim=32,
            num_prototypes=4,
            local_patch_size=8,
            prototype_heads=4,
            prototype_depth=1,
            token_encoder_depth=2,
            token_encoder_heads=4,
            backbone_out_indices=(0, 1),
            drop_rate=0.0,
        ),
        decoder=FoundationDecoderConfig(
            decoder_dim=64,
            decoder_depth=1,
            decoder_heads=4,
            decoder_dropout=0.0,
            tdfa_depth=1,
            frequency_groups=4,
        ),
        prototype_bank_size=prototype_count,
    )
    device = torch.device(args.device)
    model = MetadataConditionedFoundationModel(
        config,
        eo1_bank_centers=banks["eo1"][0],
        gf5_bank_centers=banks["gf5"][0],
        eo1_bank_wavelengths=banks["eo1"][1],
        gf5_bank_wavelengths=banks["gf5"][1],
        extra_prototype_banks={
            "dz1_vnir": banks["dz1_vnir"],
            "dz1_swir": banks["dz1_swir"],
        },
    ).to(device)
    batch.values = batch.values.to(device)
    batch.wavelengths = batch.wavelengths.to(device)
    batch.valid_mask = None if batch.valid_mask is None else batch.valid_mask.to(device)
    batch.fwhm = None if batch.fwhm is None else batch.fwhm.to(device)
    batch.metadata.gsd = batch.metadata.gsd.to(device)
    with torch.no_grad():
        output = model(space=batch)
    if not torch.isfinite(output["loss"]).item() or not finite_scalars(output["losses"]):
        raise AssertionError("Mixed forward produced a non-finite loss")

    counts = Counter()
    for shard in joint.shards:
        counts[shard.sensor] += shard.count
    report = {
        "legacy_count": len(legacy),
        "joint_count": len(joint),
        "sensor_counts": dict(counts),
        "mixed_batch_shape": list(batch.values.shape),
        "loss": float(output["loss"].item()),
        "losses": {key: float(value.item()) for key, value in output["losses"].items()},
        "success": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
