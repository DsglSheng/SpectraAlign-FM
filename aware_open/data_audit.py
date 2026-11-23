from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from aware_open.sigma450k_profiles import resolve_sigma450k_profile

try:
    from scipy.io import whosmat
except ImportError:  # pragma: no cover - optional dependency
    whosmat = None


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _infer_aviris_sensor(source_name: str) -> str:
    upper = source_name.upper()
    if upper.startswith("AV3"):
        return "AVIRIS-3"
    if upper.startswith("ANG"):
        return "AVIRIS-NG"
    if upper.startswith("F"):
        return "AVIRIS-C"
    return "unknown"


def _infer_level(source_name: str) -> str:
    upper = source_name.upper()
    if "L2" in upper or "RFL" in upper or "CORR" in upper:
        return "L2"
    if "L1" in upper or "RDN" in upper:
        return "L1"
    return "unknown"


def audit_ground(root: Path) -> dict[str, Any]:
    native_root = root / "native"
    json_files = sorted(native_root.rglob("*.json"))
    npz_files = sorted(native_root.rglob("*.npz"))

    paired = 0
    missing_npz = 0
    for json_path in json_files:
        if json_path.with_suffix(".npz").exists():
            paired += 1
        else:
            missing_npz += 1

    return {
        "root": str(root),
        "json_files": len(json_files),
        "npz_files": len(npz_files),
        "paired_samples": paired,
        "missing_npz_for_json": missing_npz,
    }


def audit_400k(root: Path) -> dict[str, Any]:
    patches_root = root / "patches"
    scene_dirs = sorted(path for path in patches_root.glob("*") if path.is_dir())

    sensor_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()

    metadata_files = 0
    missing_gsd = 0
    missing_wavelength = 0
    missing_fwhm = 0
    missing_source = 0
    examples: list[str] = []

    for scene_dir in scene_dirs:
        metadata_path = scene_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata_files += 1
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_file = str(metadata.get("source_file", "")).strip()

        if not source_file:
            missing_source += 1
            continue

        sensor_counts[_infer_aviris_sensor(source_file)] += 1
        level_counts[_infer_level(source_file)] += 1
        extension_counts[Path(source_file).suffix or "<none>"] += 1

        if len(examples) < 8:
            examples.append(source_file)

        if metadata.get("gsd") is None:
            missing_gsd += 1
        if not metadata.get("wavelength"):
            missing_wavelength += 1
        if not metadata.get("fwhm"):
            missing_fwhm += 1

    return {
        "root": str(root),
        "scene_dirs": len(scene_dirs),
        "metadata_files": metadata_files,
        "missing_source_file": missing_source,
        "missing_gsd": missing_gsd,
        "missing_wavelength": missing_wavelength,
        "missing_fwhm": missing_fwhm,
        "sensor_counts": _sorted_counter(sensor_counts),
        "level_counts": _sorted_counter(level_counts),
        "source_extensions": _sorted_counter(extension_counts),
        "example_source_files": examples,
    }


def audit_sigma450k(root: Path) -> dict[str, Any]:
    part_dirs = sorted(path for path in root.glob("*") if path.is_dir() and not path.name.startswith("."))
    mat_files = sorted(root.rglob("*.mat"))
    sidecars = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".txt", ".npy", ".npz", ".yaml", ".yml"}
    )

    partition_counts: Counter[str] = Counter(path.parent.name for path in mat_files)
    eo1_preserved = 0
    gf5_numeric_only = 0
    band_samples: dict[str, int | None] = {}
    resolved_profiles: dict[str, str] = {}
    unresolved_partitions: dict[str, str] = {}

    for mat_path in mat_files:
        stem = mat_path.stem
        if stem.startswith("EO1"):
            eo1_preserved += 1
        if stem.isdigit():
            gf5_numeric_only += 1

    if whosmat is not None:
        for part_dir in part_dirs:
            first_mat = next(iter(sorted(part_dir.glob("*.mat"))), None)
            if first_mat is None:
                continue
            bands = None
            try:
                shapes = whosmat(first_mat)
            except Exception:  # pragma: no cover - invalid or partial MAT files
                shapes = []
            for name, shape, _dtype in shapes:
                if name == "img" and len(shape) == 3:
                    bands = int(shape[0])
                    break
            band_samples[part_dir.name] = bands
            if bands is None:
                unresolved_partitions[part_dir.name] = "Could not read img band count from the sampled .mat file."
                continue
            try:
                profile = resolve_sigma450k_profile(part_dir.name, bands)
            except ValueError as exc:
                unresolved_partitions[part_dir.name] = str(exc)
            else:
                resolved_profiles[part_dir.name] = profile.profile_id
    else:  # pragma: no cover - exercised only without scipy
        for part_dir in part_dirs:
            band_samples[part_dir.name] = None
            unresolved_partitions[part_dir.name] = "scipy is not installed, so band counts could not be inspected."

    return {
        "root": str(root),
        "partitions": [part_dir.name for part_dir in part_dirs],
        "mat_files": len(mat_files),
        "sidecar_files": len(sidecars),
        "partition_counts": _sorted_counter(partition_counts),
        "eo1_scene_id_like_files": eo1_preserved,
        "gf5_numeric_only_files": gf5_numeric_only,
        "band_samples_by_partition": band_samples,
        "resolved_profiles": resolved_profiles,
        "unresolved_partitions": unresolved_partitions,
    }


def audit_real_datasets(data_root: Path) -> dict[str, Any]:
    return {
        "data_root": str(data_root),
        "ground": audit_ground(data_root / "ground"),
        "400k": audit_400k(data_root / "400k"),
        "sigma450k": audit_sigma450k(data_root / "sigma450k"),
    }
