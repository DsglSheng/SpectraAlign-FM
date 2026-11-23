#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact final DZ1 validation report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temperature_quotas(counts: Dict[str, int], total: int, temperature: float) -> Dict[str, int]:
    weights = {sensor: float(count) ** temperature for sensor, count in counts.items()}
    raw = {sensor: total * weight / sum(weights.values()) for sensor, weight in weights.items()}
    quotas = {sensor: int(math.floor(value)) for sensor, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(raw, key=lambda sensor: (raw[sensor] - quotas[sensor], sensor), reverse=True)
    for sensor in order[:remaining]:
        quotas[sensor] += 1
    return quotas


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    reports = root / "reports"
    extraction = load_json(reports / "extraction_summary.json")
    inventory = load_json(reports / "inventory_summary.json")
    joint = load_json(reports / "joint_manifest_summary.json")
    smoke = load_json(reports / "mixed_forward_smoke.json")
    prototypes = load_json(root / "artifacts" / "dz1_prototypes_v1" / "prototype_build_summary.json")
    migration = load_json(reports / "checkpoint_migration.json")
    distributions = load_json(reports / "patch_distributions.json")

    pair_path = root / "data" / "pairing" / "vnir_swir_pairs.jsonl"
    pairs: List[Dict[str, Any]] = [json.loads(line) for line in pair_path.read_text(encoding="utf-8").splitlines() if line]
    distances = [float(item["center_distance_km"]) for item in pairs]
    overlaps = [float(item["overlap_over_smaller"]) for item in pairs]

    checkpoint_path = root / "artifacts" / "checkpoints" / "neutral_epoch0045_dz1.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"]
    sensor_key = next(key for key in state if key.endswith("sample_encoder.sensor_embed.weight"))
    sensor_embedding = state[sensor_key]
    zero_centers = {
        sensor: int((state["extra_dz1_{}_bank_centers".format(sensor)].norm(dim=-1) < 1.0e-8).sum().item())
        for sensor in ("vnir", "swir")
    }

    epoch_size = int(joint["total_samples"]) // 4 * 4
    quotas = temperature_quotas({key: int(value) for key, value in joint["sensor_counts"].items()}, epoch_size, 0.5)
    source_files = [
        root / "code" / "aware_open" / "config.py",
        root / "code" / "aware_open" / "dz1_profiles.py",
        root / "code" / "aware_open" / "data" / "pretrain.py",
        root / "code" / "aware_open" / "models" / "foundation.py",
        root / "scripts" / "prepare_dz1_dataset.py",
        root / "scripts" / "build_joint_manifest.py",
        root / "scripts" / "build_dz1_prototypes.py",
        root / "scripts" / "migrate_checkpoint.py",
        root / "scripts" / "train_foundation_dz1.py",
    ]
    source_hashes = {str(path.relative_to(root)): sha256_file(path) for path in source_files}
    run_files = [path for path in (root / "runs").rglob("*") if path.is_file()]

    success = (
        inventory["core_product_count"] == 160
        and extraction["accepted_patches"] == 77907
        and joint["total_samples"] == 524979
        and joint["subset_samples"] == 100000
        and smoke.get("success") is True
        and migration.get("success") is True
        and not migration.get("unexpected_changed_tensors")
        and zero_centers == {"vnir": 0, "swir": 0}
        and len(run_files) == 0
    )
    summary = {
        "success": success,
        "data": {
            "products": inventory["product_count"],
            "core_products": inventory["core_product_count"],
            "candidate_patches": extraction["candidate_patches"],
            "rejected_nodata": extraction["rejected_nodata"],
            "accepted_patches": extraction["accepted_patches"],
            "accepted_by_sensor": distributions["sensor"],
            "dz1_shards": extraction["shard_count"],
            "checksums": extraction["dataset_checksums"],
        },
        "joint_corpus": {
            "samples": joint["total_samples"],
            "shards": joint["total_shards"],
            "sensor_counts": joint["sensor_counts"],
            "manifest_sha256": joint["combined_manifest_sha256"],
            "subset_samples": joint["subset_samples"],
            "subset_sensor_counts": joint["subset_sensor_counts"],
            "temperature_0_5_epoch_size": epoch_size,
            "temperature_0_5_quotas": quotas,
            "temperature_0_5_proportions": {key: value / epoch_size for key, value in quotas.items()},
        },
        "pairing": {
            "pair_count": len(pairs),
            "center_distance_km_p50": statistics.median(distances),
            "center_distance_km_max": max(distances),
            "overlap_over_smaller_min": min(overlaps),
            "base_pretraining_uses_pairs": False,
        },
        "prototypes": {
            "ground_sample_count": prototypes["ground_sample_count"],
            "banks": prototypes["banks"],
            "zero_prototype_centers": zero_centers,
        },
        "checkpoint": {
            **migration,
            "embedding_shape": list(sensor_embedding.shape),
            "new_rows_equal_others": bool(
                torch.equal(sensor_embedding[11], sensor_embedding[10])
                and torch.equal(sensor_embedding[12], sensor_embedding[10])
            ),
        },
        "mixed_forward": smoke,
        "training": {"stage_c_or_d_started": len(run_files) > 0, "run_file_count": len(run_files)},
        "source_sha256": source_hashes,
        "paths": {
            "remote_root": str(root),
            "joint_manifest": joint["combined_manifest"],
            "checkpoint": str(checkpoint_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"success": success, "output": str(args.output), "samples": joint["total_samples"]}))


if __name__ == "__main__":
    main()
