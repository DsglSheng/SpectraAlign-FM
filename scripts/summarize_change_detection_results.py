from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DATASETS: dict[str, dict[str, Any]] = {
    "hermiston": {
        "files": ["Sa1.mat", "Sa2.mat", "SaGT.mat"],
        "result_dir": "hermiston_aware_e50_seed1",
        "spatsigma_f1": 0.9208,
        "hypersigma_f1": 0.9174,
        "gpu": "GPU0",
        "patch_size": 5,
    },
    "farmland": {
        "files": ["Farm1.mat", "Farm2.mat", "GTChina1.mat"],
        "result_dir": "farmland_aware_e50_seed1",
        "spatsigma_f1": 0.9531,
        "hypersigma_f1": 0.9549,
        "gpu": "GPU1",
        "patch_size": 5,
    },
    "bayarea": {
        "files": ["BayArea.mat"],
        "result_dir": "bayarea_aware_e50_seed1",
        "spatsigma_f1": 0.9889,
        "hypersigma_f1": 0.9893,
        "gpu": "GPU2",
        "patch_size": 15,
    },
    "barbara": {
        "files": ["Barbara.mat"],
        "result_dir": "barbara_aware_e50_seed1",
        "spatsigma_f1": 0.9895,
        "hypersigma_f1": 0.9904,
        "gpu": "GPU3",
        "patch_size": 15,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Aware change-detection fine-tuning results.")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/Data/SpectralData/Aware_open/data/change_detection"))
    parser.add_argument("--result-root", type=Path, default=Path("/mnt/Data/SpectralData/Aware_open/result/change_detection"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_data_file(data_root: Path, dataset: str, filename: str) -> Path | None:
    for candidate in (data_root / dataset / filename, data_root / filename):
        if candidate.exists():
            return candidate
    return None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {"_error": f"Invalid JSON: {path}"}


def extract_best_metrics(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload:
        return None
    metrics = payload.get("best_f1_metrics")
    if isinstance(metrics, dict):
        return metrics
    return None


def read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                last = {"_error": f"Invalid JSONL line in {path}"}
    return last


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def process_running(pid: int | None) -> bool | None:
    if pid is None:
        return None
    try:
        return Path(f"/proc/{pid}").exists()
    except OSError:
        return None


def summarize_dataset(dataset: str, spec: dict[str, Any], data_root: Path, result_root: Path) -> dict[str, Any]:
    missing = [name for name in spec["files"] if resolve_data_file(data_root, dataset, name) is None]
    out_dir = result_root / str(spec["result_dir"])
    metrics_path = out_dir / "test_metrics.json"
    config_path = out_dir / "config.json"
    log_path = out_dir / "change_detection_log.jsonl"
    pid = read_pid(out_dir / "pid")
    last_log = read_last_jsonl(log_path)
    metrics_payload = read_json(metrics_path)
    best_metrics = extract_best_metrics(metrics_payload)
    f1 = None if best_metrics is None else float(best_metrics.get("f1", 0.0))
    hypersigma = float(spec["hypersigma_f1"])
    spatsigma = float(spec["spatsigma_f1"])
    return {
        "dataset": dataset,
        "data_ready": len(missing) == 0,
        "missing_files": missing,
        "assigned_gpu": spec["gpu"],
        "result_dir": str(out_dir),
        "config_exists": config_path.exists(),
        "log_exists": log_path.exists(),
        "metrics_exists": metrics_path.exists(),
        "pid": pid,
        "running": process_running(pid),
        "last_log": last_log,
        "best_f1": f1,
        "hypersigma_f1": hypersigma,
        "spatsigma_f1": spatsigma,
        "delta_vs_hypersigma": None if f1 is None else round(f1 - hypersigma, 4),
        "delta_vs_spatsigma": None if f1 is None else round(f1 - spatsigma, 4),
        "beats_hypersigma": None if f1 is None else bool(f1 > hypersigma),
        "beats_spatsigma": None if f1 is None else bool(f1 > spatsigma),
        "best_f1_metrics": best_metrics,
    }


def main() -> None:
    args = parse_args()
    output = args.output or (args.result_root / "comparison_summary.json")
    summaries = [
        summarize_dataset(dataset, spec, args.data_root, args.result_root)
        for dataset, spec in DATASETS.items()
    ]
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(args.data_root),
        "result_root": str(args.result_root),
        "experiment_plan": {
            "protocol": "HyperSIGMA-like change detection fine-tuning.",
            "epochs": 50,
            "batch_size": 32,
            "train_number_per_class": 500,
            "optimizer": "AdamW",
            "lr": 6.0e-5,
            "weight_decay": 0.05,
            "layer_decay_rate": 0.9,
            "eval_metric": "F1 on all valid pixels",
            "patch_size": {dataset: spec["patch_size"] for dataset, spec in DATASETS.items()},
            "gpu_assignment": {dataset: spec["gpu"] for dataset, spec in DATASETS.items()},
        },
        "baselines": {
            dataset: {
                "spatsigma_f1": spec["spatsigma_f1"],
                "hypersigma_f1": spec["hypersigma_f1"],
            }
            for dataset, spec in DATASETS.items()
        },
        "results": summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    tmp.replace(output)
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
