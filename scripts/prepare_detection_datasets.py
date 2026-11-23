from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import requests
import scipy.io as sio


DATASET_SOURCES: dict[str, dict[str, Any]] = {
    "cri": {
        "task": "had",
        "source_url": "https://raw.githubusercontent.com/Ayushma00/DMS2F-HAD/main/Data/Cri.mat",
        "source_filename": "Cri.mat",
        "converted_filename": "Nuance_Cri_400_400_46_1254.mat",
        "source_cube_key": "data",
        "source_gt_key": "map",
        "target_cube_key": "hsi",
        "target_gt_key": "hsi_gt",
        "reference": ["https://github.com/Ayushma00/DMS2F-HAD", "https://github.com/QYo-Liu/UEML"],
    },
    "pavia": {
        "task": "had",
        "source_url": "https://raw.githubusercontent.com/jeline0110/GT-HAD/main/data/pavia.mat",
        "source_filename": "pavia.mat",
        "converted_filename": "Paiva_108_120_102_43.mat",
        "source_cube_key": "data",
        "source_gt_key": "map",
        "target_cube_key": "hsi",
        "target_gt_key": "hsi_gt",
        "reference": ["https://github.com/jeline0110/GT-HAD", "https://github.com/NanWangAC/GT-HAD"],
    },
    "aviris": {
        "task": "htd",
        "source_url": "https://raw.githubusercontent.com/Fegg-DuXingzhi/hyperspectral-target-detection-dataset/main/AVIRIS-I.mat",
        "source_filename": "AVIRIS-I.mat",
        "converted_filename": "aviris.mat",
        "source_cube_key": "data",
        "source_gt_key": "gt",
        "target_cube_key": "X",
        "target_gt_key": "TL",
        "reference": ["https://github.com/Fegg-DuXingzhi/hyperspectral-target-detection-dataset"],
    },
    "mosaic": {
        "task": "htd",
        "source_url": None,
        "reference": ["HyperSIGMA target detection benchmark", "Public direct download not yet resolved"],
    },
    "renourban": {
        "task": "htd",
        "source_url": None,
        "reference": ["RenoUrban target detection benchmark", "Public direct download not yet resolved"],
    },
    "viareggio": {
        "task": "had",
        "source_url": None,
        "reference": ["https://github.com/meiqihu/ACDA", "Viareggio public share link appears to be Baidu-only"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert public hyperspectral detection datasets.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["mosaic", "aviris", "renourban", "pavia", "cri", "viareggio"],
        help="Dataset names to prepare.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def summarize_array(name: str, value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": None if array.size == 0 else float(np.min(array)),
        "max": None if array.size == 0 else float(np.max(array)),
    }


def convert_dataset(config: dict[str, Any], source_path: Path, destination_path: Path) -> dict[str, Any]:
    payload = sio.loadmat(source_path)
    cube = np.asarray(payload[config["source_cube_key"]])
    gt = np.asarray(payload[config["source_gt_key"]])
    while gt.ndim > 2 and 1 in gt.shape:
        gt = np.squeeze(gt)
    sio.savemat(
        destination_path,
        {
            config["target_cube_key"]: cube,
            config["target_gt_key"]: gt,
        },
    )
    return {
        "cube": summarize_array(config["target_cube_key"], cube),
        "gt": summarize_array(config["target_gt_key"], gt),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "output_root": str(output_root),
        "datasets": {},
    }

    for dataset_name in args.datasets:
        key = str(dataset_name).strip().lower().replace("-", "_")
        if key not in DATASET_SOURCES:
            manifest["datasets"][key] = {"status": "unknown_dataset"}
            continue
        config = DATASET_SOURCES[key]
        task = config["task"]
        dataset_dir = output_root / "detection" / task / key
        dataset_dir.mkdir(parents=True, exist_ok=True)

        entry: dict[str, Any] = {
            "task": task,
            "references": config.get("reference", []),
        }
        if config.get("source_url") is None:
            entry["status"] = "missing_public_download"
            manifest["datasets"][key] = entry
            continue

        raw_dir = dataset_dir / "raw"
        source_path = raw_dir / config["source_filename"]
        converted_path = dataset_dir / config["converted_filename"]
        if not source_path.exists():
            download_file(config["source_url"], source_path)
        stats = convert_dataset(config, source_path, converted_path)
        entry.update(
            {
                "status": "ready",
                "source_url": config["source_url"],
                "raw_path": str(source_path),
                "converted_path": str(converted_path),
                "stats": stats,
            }
        )
        manifest["datasets"][key] = entry

    manifest_path = output_root / "detection" / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
    print(json.dumps(manifest, ensure_ascii=True))


if __name__ == "__main__":
    main()
