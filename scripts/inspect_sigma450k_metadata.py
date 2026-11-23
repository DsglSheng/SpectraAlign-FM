from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scipy.io import loadmat, whosmat

try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency
    h5py = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect sigma450k .mat files for hidden metadata fields."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/Data/SpectralData/sigma450k"),
        help="sigma450k root directory",
    )
    parser.add_argument(
        "--samples-per-partition",
        type=int,
        default=3,
        help="How many .mat files to inspect per partition",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    return parser.parse_args()


def choose_evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return items

    last_index = len(items) - 1
    seen: set[int] = set()
    selected: list[Path] = []
    for i in range(count):
        index = round(i * last_index / (count - 1))
        if index in seen:
            continue
        seen.add(index)
        selected.append(items[index])
    return selected


def summarize_value(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python_type": type(value).__name__,
    }
    shape = getattr(value, "shape", None)
    if shape is not None:
        summary["shape"] = list(shape)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        summary["dtype"] = str(dtype)
    return summary


def inspect_with_scipy(mat_path: Path) -> dict[str, Any]:
    variables = [
        {
            "name": name,
            "shape": list(shape),
            "dtype": dtype,
        }
        for name, shape, dtype in whosmat(mat_path)
    ]
    loaded = loadmat(mat_path)
    keys = [key for key in loaded.keys() if not key.startswith("__")]
    return {
        "reader": "scipy.io",
        "variables": variables,
        "keys": keys,
        "loaded": {key: summarize_value(loaded[key]) for key in keys},
    }


def inspect_with_h5py(mat_path: Path) -> dict[str, Any]:
    if h5py is None:
        raise RuntimeError("h5py is not installed")
    with h5py.File(mat_path, "r") as handle:
        keys = list(handle.keys())
        loaded = {}
        for key in keys:
            value = handle[key]
            loaded[key] = {
                "python_type": type(value).__name__,
                "shape": list(getattr(value, "shape", ()) or ()),
                "dtype": str(getattr(value, "dtype", "")),
            }
    return {
        "reader": "h5py",
        "variables": [],
        "keys": keys,
        "loaded": loaded,
    }


def inspect_mat(mat_path: Path) -> dict[str, Any]:
    try:
        detail = inspect_with_scipy(mat_path)
    except Exception as exc:
        detail = {
            "reader": "scipy.io",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if h5py is not None:
            try:
                detail = inspect_with_h5py(mat_path)
            except Exception as h5_exc:  # pragma: no cover - fallback path
                detail["h5py_error"] = f"{type(h5_exc).__name__}: {h5_exc}"

    return {
        "file": str(mat_path),
        **detail,
    }


def inspect_partition(partition_dir: Path, samples_per_partition: int) -> dict[str, Any]:
    files = sorted(partition_dir.glob("*.mat"))
    sampled = choose_evenly_spaced(files, samples_per_partition)
    samples = [inspect_mat(path) for path in sampled]

    union_keys: Counter[str] = Counter()
    img_shapes: Counter[str] = Counter()
    files_with_only_img = 0
    files_with_extra_keys = 0

    for sample in samples:
        keys = sample.get("keys", [])
        if keys == ["img"]:
            files_with_only_img += 1
        elif keys:
            files_with_extra_keys += 1

        for key in keys:
            union_keys[key] += 1

        loaded = sample.get("loaded", {})
        img_summary = loaded.get("img")
        if img_summary and "shape" in img_summary:
            img_shapes[str(tuple(img_summary["shape"]))] += 1

    return {
        "partition": partition_dir.name,
        "total_mat_files": len(files),
        "sampled_files": len(sampled),
        "files_with_only_img": files_with_only_img,
        "files_with_extra_keys": files_with_extra_keys,
        "union_keys": {key: union_keys[key] for key in sorted(union_keys)},
        "img_shapes": {key: img_shapes[key] for key in sorted(img_shapes)},
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    partitions = sorted(
        path for path in args.root.glob("*")
        if path.is_dir() and not path.name.startswith(".")
    )

    report = {
        "root": str(args.root),
        "samples_per_partition": args.samples_per_partition,
        "partitions": [
            inspect_partition(partition_dir, args.samples_per_partition)
            for partition_dir in partitions
        ],
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nWrote report to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
