#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a zero-copy HyperGlobal + DZ1 shard manifest.")
    parser.add_argument("--hyperglobal-root", type=Path, required=True)
    parser.add_argument("--dz1-shard-root", type=Path, required=True)
    parser.add_argument("--dz1-manifest", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--subset-count", type=int, default=100_000)
    parser.add_argument("--subset-manifest", type=Path, default=None)
    parser.add_argument("--sampling-temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(str(temporary), str(path))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def absolutize(record: Dict[str, Any], root: Path, source: str) -> Dict[str, Any]:
    output = dict(record)
    for key in ("values_path", "metadata_path", "samples_path"):
        value = output.get(key)
        if value is None:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        output[key] = str(path.resolve())
    output["dataset_source"] = source
    return output


def sensor_counts(records: Iterable[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for record in records:
        counts[str(record["sensor"])] += int(record["count"])
    return counts


def allocate_quotas(counts: Counter, total: int, temperature: float) -> Dict[str, int]:
    target = min(max(0, int(total)), sum(int(count) for count in counts.values()))
    quotas = {sensor: 0 for sensor in counts}
    remaining = target
    # Allocate iteratively so a small sensor is capped at its available sample
    # count and its unused quota is redistributed to the other sensors.  This
    # keeps the requested fixed-size ablation subset exact, including pilots.
    while remaining > 0:
        active = [sensor for sensor in counts if quotas[sensor] < int(counts[sensor])]
        if not active:
            break
        weights = {sensor: float(counts[sensor]) ** float(temperature) for sensor in active}
        weight_sum = sum(weights.values())
        raw = {sensor: remaining * weights[sensor] / weight_sum for sensor in active}
        assigned = 0
        for sensor in active:
            capacity = int(counts[sensor]) - quotas[sensor]
            increment = min(capacity, int(math.floor(raw[sensor])))
            quotas[sensor] += increment
            assigned += increment
        remaining -= assigned
        if remaining <= 0:
            break
        order = sorted(
            active,
            key=lambda sensor: (raw[sensor] - math.floor(raw[sensor]), sensor),
            reverse=True,
        )
        for sensor in order:
            if remaining <= 0:
                break
            if quotas[sensor] < int(counts[sensor]):
                quotas[sensor] += 1
                remaining -= 1
    return quotas


def build_subset(records: List[Dict[str, Any]], total: int, temperature: float, seed: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sensor"])].append(record)
    counts = sensor_counts(records)
    quotas = allocate_quotas(counts, min(total, sum(counts.values())), temperature)
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for sensor in sorted(grouped):
        shards = list(grouped[sensor])
        rng.shuffle(shards)
        remaining = quotas.get(sensor, 0)
        for record in shards:
            if remaining <= 0:
                break
            keep = min(int(record["count"]), remaining)
            item = dict(record)
            item["count"] = keep
            item["subset_prefix_only"] = keep < int(record["count"])
            selected.append(item)
            remaining -= keep
    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    hyper_manifest = args.hyperglobal_root / "manifest.jsonl"
    dz_manifest = args.dz1_manifest or (args.dz1_shard_root / "manifest.jsonl")
    hyper_records = [absolutize(item, args.hyperglobal_root, "hyperglobal450k") for item in read_jsonl(hyper_manifest)]
    dz_records = [absolutize(item, args.dz1_shard_root, "dz1_202607") for item in read_jsonl(dz_manifest)]
    combined = hyper_records + dz_records
    write_jsonl(args.output_manifest, combined)

    counts = sensor_counts(combined)
    summary: Dict[str, Any] = {
        "hyperglobal_manifest": str(hyper_manifest.resolve()),
        "dz1_manifest": str(dz_manifest.resolve()),
        "combined_manifest": str(args.output_manifest.resolve()),
        "total_samples": int(sum(counts.values())),
        "total_shards": len(combined),
        "sensor_counts": dict(counts),
        "sampling_temperature": args.sampling_temperature,
        "hyperglobal_manifest_sha256": sha256_file(hyper_manifest),
        "dz1_manifest_sha256": sha256_file(dz_manifest),
        "combined_manifest_sha256": sha256_file(args.output_manifest),
    }
    if args.subset_manifest is not None and args.subset_count > 0:
        subset = build_subset(combined, args.subset_count, args.sampling_temperature, args.seed)
        write_jsonl(args.subset_manifest, subset)
        subset_counts = sensor_counts(subset)
        summary["subset_manifest"] = str(args.subset_manifest.resolve())
        summary["subset_samples"] = int(sum(subset_counts.values()))
        summary["subset_sensor_counts"] = dict(subset_counts)
        summary["subset_manifest_sha256"] = sha256_file(args.subset_manifest)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_suffix(args.summary.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(str(temporary), str(args.summary))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
