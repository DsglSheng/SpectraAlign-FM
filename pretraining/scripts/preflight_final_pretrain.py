#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only preflight for the final HyperGlobal+DZ1 run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    repo_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ.get("SPECTRAALIGN_DATA_ROOT", repo_root / "data")).expanduser().resolve()
    data_config = config["data"]
    manifest_path = data_root / data_config["joint_manifest"]
    records = read_jsonl(manifest_path)
    counts: Counter[str] = Counter()
    missing_values = []
    for record in records:
        counts[str(record["sensor"])] += int(record["count"])
        values_path = Path(record["values_path"])
        if not values_path.exists():
            missing_values.append(str(values_path))

    bank_config = config["ground_spectral_prior"]
    bank_paths = {
        "eo1": data_root / bank_config["eo1_bank"],
        "gf5": data_root / bank_config["gf5_bank"],
        "dz1_vnir": data_root / bank_config["dz1_vnir_bank"],
        "dz1_swir": data_root / bank_config["dz1_swir_bank"],
    }
    banks: Dict[str, Any] = {}
    for sensor, path in bank_paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        centers = torch.as_tensor(payload["centers"]).float()
        wavelengths = torch.as_tensor(payload["wavelengths_nm"]).float()
        banks[sensor] = {
            "path": str(path),
            "sensor_field": str(payload["sensor"]),
            "centers_shape": list(centers.shape),
            "wavelength_shape": list(wavelengths.shape),
            "zero_centers": int((centers.norm(dim=-1) < 1.0e-8).sum().item()),
        }

    compute = config["compute"]
    project_root = args.config.resolve().parents[1]
    run_root = project_root / "runs" / str(config["run_name"])
    existing_checkpoints = list(run_root.rglob("*.pt")) if run_root.exists() else []
    expected_counts = {key: int(value) for key, value in data_config["sensor_counts"].items()}
    sampling_alpha = float(data_config["sensor_sampling_temperature"])
    nominal_counts = {key: int(value) for key, value in data_config["sampler_epoch_nominal_counts"].items()}
    total_steps = int(config["stage_a_neutral"]["max_optimizer_steps"]) + int(
        config["stage_b_full"]["max_optimizer_steps"]
    )
    reference = data_root / config["scientific_control"]["reference_checkpoint"]
    smoke_path = data_root / "reports" / "mixed_forward_smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else None
    checks = {
        "config_is_authorized": config.get("status") == "authorized_for_execution_2026-07-06",
        "manifest_exists": manifest_path.exists(),
        "manifest_sample_count": sum(counts.values()) == int(data_config["samples"]),
        "manifest_shard_count": len(records) == int(data_config["shards"]),
        "sensor_counts": dict(counts) == expected_counts,
        "natural_sampling_alpha": sampling_alpha == 1.0,
        "natural_shard_local_sampling": data_config.get("sensor_sampling_mode") == "natural_shard_local",
        "sampler_nominal_counts": nominal_counts == expected_counts,
        "sampler_drop_last": sum(expected_counts.values()) - int(data_config["sampler_epoch_size"])
        == int(data_config["sampler_drop_last_samples"])
        == 3,
        "all_values_exist": not missing_values,
        "all_banks_exist": all(path.exists() for path in bank_paths.values()),
        "bank_sensor_fields": all(item["sensor_field"] == sensor for sensor, item in banks.items()),
        "bank_prototype_counts": all(item["centers_shape"][0] == 256 for item in banks.values()),
        "bank_wavelength_dimensions": all(
            item["centers_shape"][1] == item["wavelength_shape"][0] for item in banks.values()
        ),
        "bank_zero_centers": all(item["zero_centers"] == 0 for item in banks.values()),
        "global_batch": int(compute["world_size"]) * int(compute["batch_size_per_gpu"]) == 448,
        "fixed_total_steps": total_steps == int(config["scientific_control"]["total_optimizer_steps"]),
        "mixed_forward_smoke": bool(smoke and smoke.get("success")),
    }
    report = {
        "ready": all(checks.values()),
        "checks": checks,
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "samples": sum(counts.values()),
        "shards": len(records),
        "sensor_counts": dict(counts),
        "sensor_sampling_temperature": sampling_alpha,
        "sensor_sampling_mode": data_config["sensor_sampling_mode"],
        "sampler_epoch_nominal_counts": nominal_counts,
        "sampler_drop_last_samples": int(data_config["sampler_drop_last_samples"]),
        "prototype_banks": banks,
        "reference_checkpoint_available": reference.is_file(),
        "optimizer_steps": {
            "stage_a": int(config["stage_a_neutral"]["max_optimizer_steps"]),
            "stage_b": int(config["stage_b_full"]["max_optimizer_steps"]),
            "total": total_steps,
        },
        "missing_values_preview": missing_values[:10],
        "training_started": bool(existing_checkpoints),
        "existing_checkpoint_count": len(existing_checkpoints),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
