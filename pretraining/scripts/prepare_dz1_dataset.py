#!/usr/bin/env python3
"""Build scale-aligned DZ01 shards without modifying the legacy Aware project.

The public training sample contract is compatible with the existing Sigma450K
shards: values are float32 arrays shaped (N, C, 64, 64), with one JSONL record
per shard. Full-scene GeoTIFFs are an internal cache only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import rasterio
    from rasterio.transform import xy
    from rasterio.windows import Window
    from rasterio.warp import transform, transform_bounds
except ImportError as exc:  # pragma: no cover
    raise SystemExit("rasterio is required for DZ1 preparation") from exc


BAND_PATTERN = re.compile(r"_B(\d+)\.TIF$", re.IGNORECASE)
MTL_LINE_PATTERN = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$")
PRODUCT_SENSOR_PATTERN = re.compile(r"^DZ01([VS])_")
ACQUISITION_PATTERN = re.compile(r"(\d{14})")
UINT16_SATURATION = 65535.0

VNIR_EXPECTED_BANDS = 16
SWIR_EXPECTED_BANDS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DZ01 scale-aligned pretraining shards.")
    parser.add_argument("stage", choices=("inventory", "stack", "extract", "all"))
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--classification-csv", type=Path, default=None)
    parser.add_argument("--acquisition", action="append", default=[])
    parser.add_argument("--cloud-cover-max", type=float, default=20.0)
    parser.add_argument("--saturation-fraction-max", type=float, default=0.005)
    parser.add_argument("--robust-scale-min", type=float, default=1.0e-6)
    parser.add_argument("--samples-per-shard", type=int, default=256)
    parser.add_argument("--vnir-mode", choices=("context128", "native64"), default="context128")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-stacks", action="store_true", help="Extract from existing stacks only.")
    parser.add_argument("--max-products", type=int, default=None)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
    os.replace(str(temporary), str(path))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_mtl(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = MTL_LINE_PATTERN.match(line)
            if match is None:
                continue
            key, raw = match.groups()
            if key in {"GROUP", "END_GROUP", "END"}:
                continue
            values[key] = raw.strip().strip('"')
    return values


def require_float(metadata: Dict[str, str], key: str) -> float:
    if key not in metadata:
        raise KeyError(key)
    value = float(metadata[key])
    if not math.isfinite(value):
        raise ValueError("Non-finite {}".format(key))
    return value


def optional_float(metadata: Dict[str, str], key: str) -> Optional[float]:
    try:
        return require_float(metadata, key)
    except (KeyError, TypeError, ValueError):
        return None


def band_values(metadata: Dict[str, str], prefix: str, band_count: int) -> List[float]:
    return [require_float(metadata, "{}_BAND_{}".format(prefix, index)) for index in range(1, band_count + 1)]


def sensor_spec(sensor_code: str, vnir_mode: str) -> Dict[str, Any]:
    if sensor_code == "V":
        native64 = vnir_mode == "native64"
        return {
            "sensor": "dz1_vnir",
            "sensor_name": "DZ01 VNIR",
            "band_count": VNIR_EXPECTED_BANDS,
            "native_gsd_m": 14.0,
            "effective_gsd_m": 14.0 if native64 else 28.0,
            "source_window_size": 64 if native64 else 128,
            "output_size": 64,
            "resampling_factor": 1 if native64 else 2,
        }
    if sensor_code == "S":
        return {
            "sensor": "dz1_swir",
            "sensor_name": "DZ01 SWIR",
            "band_count": SWIR_EXPECTED_BANDS,
            "native_gsd_m": 30.0,
            "effective_gsd_m": 30.0,
            "source_window_size": 64,
            "output_size": 64,
            "resampling_factor": 1,
        }
    raise ValueError("Unknown sensor code: {}".format(sensor_code))


def natural_band_paths(product_dir: Path) -> List[Path]:
    indexed: List[Tuple[int, Path]] = []
    for path in product_dir.glob("*.TIF"):
        match = BAND_PATTERN.search(path.name)
        if match is not None:
            indexed.append((int(match.group(1)), path))
    indexed.sort(key=lambda item: item[0])
    if indexed and [item[0] for item in indexed] != list(range(1, len(indexed) + 1)):
        raise ValueError("Non-contiguous band numbering under {}".format(product_dir))
    return [item[1] for item in indexed]


def load_classification_rows(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            product_id = row.get("产品编号")
            if product_id:
                rows[product_id] = row
    return rows


def acquisition_ids(input_root: Path, selected: Sequence[str]) -> List[str]:
    available = sorted(path.name for path in input_root.iterdir() if path.is_dir())
    if not selected:
        return available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise FileNotFoundError("Missing acquisitions: {}".format(missing))
    return [item for item in available if item in set(selected)]


def raster_signature(path: Path) -> Tuple[Any, ...]:
    with rasterio.open(str(path)) as dataset:
        return (
            dataset.width,
            dataset.height,
            None if dataset.crs is None else dataset.crs.to_wkt(),
            tuple(dataset.transform),
            dataset.dtypes[0],
            dataset.nodata,
        )


def build_product_record(
    input_root: Path,
    acquisition_id: str,
    product_dir: Path,
    classification: Dict[str, Dict[str, str]],
    cloud_cover_max: float,
    vnir_mode: str,
) -> Dict[str, Any]:
    product_id = product_dir.name
    match = PRODUCT_SENSOR_PATTERN.match(product_id)
    if match is None:
        raise ValueError("Unrecognized product directory: {}".format(product_dir))
    sensor_code = match.group(1)
    spec = sensor_spec(sensor_code, vnir_mode)
    mtl_paths = sorted(product_dir.glob("*_MTL.txt"))
    if len(mtl_paths) != 1:
        raise ValueError("Expected one MTL under {}, found {}".format(product_dir, len(mtl_paths)))
    mtl_path = mtl_paths[0]
    metadata = parse_mtl(mtl_path)
    band_paths = natural_band_paths(product_dir)
    reasons: List[str] = []
    structural_ok = True
    expected_bands = int(spec["band_count"])
    if len(band_paths) != expected_bands:
        reasons.append("band_count_{}_expected_{}".format(len(band_paths), expected_bands))
        structural_ok = False

    signatures: List[Tuple[Any, ...]] = []
    for path in band_paths:
        try:
            signatures.append(raster_signature(path))
        except Exception as exc:
            reasons.append("raster_open_error:{}:{}".format(path.name, type(exc).__name__))
            structural_ok = False
    if signatures and len(set(signatures)) != 1:
        reasons.append("band_grid_mismatch")
        structural_ok = False

    calibration_ok = True
    try:
        wavelengths = band_values(metadata, "MEASURED_CENTER_WAVELENGTH", expected_bands)
        wavelength_min = band_values(metadata, "MEASURED_MIN_WAVELENGTH", expected_bands)
        wavelength_max = band_values(metadata, "MEASURED_MAX_WAVELENGTH", expected_bands)
        radiance_mult = band_values(metadata, "RADIANCE_MULT", expected_bands)
        radiance_add = band_values(metadata, "RADIANCE_ADD", expected_bands)
    except (KeyError, TypeError, ValueError) as exc:
        wavelengths, wavelength_min, wavelength_max, radiance_mult, radiance_add = [], [], [], [], []
        reasons.append("metadata_band_error:{}".format(exc))
        calibration_ok = False

    unit_swir = bool(sensor_code == "S" and radiance_mult and all(abs(value - 1.0) <= 1.0e-12 for value in radiance_mult))
    if unit_swir:
        reasons.append("swir_identity_radiance_scale")
        calibration_ok = False
    if radiance_mult and any(value <= 0.0 or not math.isfinite(value) for value in radiance_mult):
        reasons.append("invalid_radiance_scale")
        calibration_ok = False

    image_quality = optional_float(metadata, "IMAGE_QUALITY")
    cloud_cover = optional_float(metadata, "CLOUD_COVER")
    if image_quality != 1.0:
        reasons.append("image_quality_not_1")
    if cloud_cover is None:
        reasons.append("missing_cloud_cover")
    elif cloud_cover > cloud_cover_max:
        reasons.append("cloud_cover_gt_{}".format(cloud_cover_max))

    width = height = epsg = None
    transform_values: Optional[List[float]] = None
    bounds_wgs84: Optional[List[float]] = None
    dtype = nodata = None
    if signatures:
        width, height, crs_wkt, affine, dtype, nodata = signatures[0]
        transform_values = list(affine)
        with rasterio.open(str(band_paths[0])) as dataset:
            epsg = None if dataset.crs is None else dataset.crs.to_epsg()
            if dataset.crs is not None:
                bounds_wgs84 = list(transform_bounds(dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21))

    class_row = classification.get(product_id, {})
    acquired = metadata.get("DATE_ACQUIRED")
    center_time = metadata.get("SCENE_CENTER_TIME")
    timestamp_iso = None
    if acquired:
        timestamp_iso = acquired + ("T" + center_time if center_time else "")

    core_eligible = bool(
        structural_ok
        and calibration_ok
        and len(band_paths) == expected_bands
        and image_quality == 1.0
        and cloud_cover is not None
        and cloud_cover <= cloud_cover_max
    )
    theoretical_patch_count = 0
    if width is not None and height is not None:
        source_size = int(spec["source_window_size"])
        theoretical_patch_count = int(width // source_size) * int(height // source_size)

    record: Dict[str, Any] = {
        "acquisition_id": acquisition_id,
        "product_id": product_id,
        "sensor_code": sensor_code,
        **spec,
        "relative_product_path": str(product_dir.relative_to(input_root)).replace("\\", "/"),
        "relative_mtl_path": str(mtl_path.relative_to(input_root)).replace("\\", "/"),
        "relative_band_paths": [str(path.relative_to(input_root)).replace("\\", "/") for path in band_paths],
        "wavelength_nm": wavelengths,
        "wavelength_min_nm": wavelength_min,
        "wavelength_max_nm": wavelength_max,
        "effective_bandwidth_nm": [upper - lower for lower, upper in zip(wavelength_min, wavelength_max)],
        "bandwidth_source": "MTL_MEASURED_MAX_MINUS_MIN_PROXY",
        "response_model": "boxcar_measured_span",
        "radiance_mult": radiance_mult,
        "radiance_add": radiance_add,
        "level": metadata.get("DATA_TYPE", "L2"),
        "image_quality": image_quality,
        "cloud_cover_pct": cloud_cover,
        "sun_azimuth_deg": optional_float(metadata, "SUN_AZIMUTH"),
        "sun_zenith_deg": optional_float(metadata, "SUN_ZENITH"),
        "sat_azimuth_deg": optional_float(metadata, "SAT_AZIMUTH"),
        "sat_zenith_deg": optional_float(metadata, "SAT_ZENITH"),
        "roll_angle_deg": optional_float(metadata, "ROLL_ANGLE"),
        "pitch_angle_deg": optional_float(metadata, "PITCH_ANGLE"),
        "yaw_angle_deg": optional_float(metadata, "YAW_ANGLE"),
        "timestamp_iso": timestamp_iso,
        "processing_software_version": metadata.get("PROCESSING_SOFTWARE_VERSION"),
        "width": width,
        "height": height,
        "epsg": epsg,
        "transform": transform_values,
        "bounds_wgs84": bounds_wgs84,
        "dtype": dtype,
        "nodata": nodata,
        "structural_ok": structural_ok,
        "calibration_ok": calibration_ok,
        "core_eligible": core_eligible,
        "quarantine_reasons": reasons,
        "theoretical_patch_count": theoretical_patch_count,
        "province": class_row.get("主要省级行政区"),
        "county": class_row.get("主要县级行政区"),
        "scene_environment": class_row.get("场景分类"),
    }
    return record


def build_inventory(args: argparse.Namespace) -> List[Dict[str, Any]]:
    input_root = args.input_root.resolve()
    classifications = load_classification_rows(args.classification_csv)
    records: List[Dict[str, Any]] = []
    acquisition_list = acquisition_ids(input_root, args.acquisition)
    for acquisition_id in acquisition_list:
        acquisition_dir = input_root / acquisition_id
        for product_dir in sorted(path for path in acquisition_dir.iterdir() if path.is_dir()):
            records.append(
                build_product_record(
                    input_root,
                    acquisition_id,
                    product_dir,
                    classifications,
                    args.cloud_cover_max,
                    args.vnir_mode,
                )
            )
            if args.max_products is not None and len(records) >= args.max_products:
                break
        if args.max_products is not None and len(records) >= args.max_products:
            break

    manifest_root = args.output_root / "data" / "manifests"
    write_jsonl(manifest_root / "products.jsonl", records)
    write_jsonl(manifest_root / "dz1_quarantine.jsonl", [item for item in records if not item["core_eligible"]])
    build_pairing(records, args.output_root / "data" / "pairing" / "vnir_swir_pairs.jsonl")

    sensor_counts = Counter(item["sensor"] for item in records)
    core_counts = Counter(item["sensor"] for item in records if item["core_eligible"])
    reason_counts = Counter(reason for item in records for reason in item["quarantine_reasons"])
    summary = {
        "input_root": str(input_root),
        "vnir_mode": args.vnir_mode,
        "cloud_cover_max": args.cloud_cover_max,
        "acquisition_count": len(set(item["acquisition_id"] for item in records)),
        "product_count": len(records),
        "sensor_product_counts": dict(sensor_counts),
        "core_product_count": sum(core_counts.values()),
        "core_sensor_product_counts": dict(core_counts),
        "quarantine_reason_counts": dict(reason_counts),
        "theoretical_core_patch_count": int(sum(item["theoretical_patch_count"] for item in records if item["core_eligible"])),
    }
    write_json(args.output_root / "reports" / "inventory_summary.json", summary)
    return records


def bbox_overlap_metrics(first: Sequence[float], second: Sequence[float]) -> Dict[str, float]:
    aw, asouth, ae, an = first
    bw, bsouth, be, bn = second
    intersection = max(0.0, min(ae, be) - max(aw, bw)) * max(0.0, min(an, bn) - max(asouth, bsouth))
    area_a = max(0.0, ae - aw) * max(0.0, an - asouth)
    area_b = max(0.0, be - bw) * max(0.0, bn - bsouth)
    union = area_a + area_b - intersection
    return {
        "overlap_over_smaller": intersection / max(min(area_a, area_b), 1.0e-12),
        "bbox_iou": intersection / max(union, 1.0e-12),
    }


def haversine_km(first: Sequence[float], second: Sequence[float]) -> float:
    lon1, lat1 = first
    lon2, lat2 = second
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    value = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 6371.0088 * 2.0 * math.asin(math.sqrt(value))


def build_pairing(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["acquisition_id"]].append(record)
    pairs: List[Dict[str, Any]] = []
    for acquisition_id, items in sorted(grouped.items()):
        vnir = [item for item in items if item["sensor"] == "dz1_vnir" and item.get("bounds_wgs84")]
        swir = [item for item in items if item["sensor"] == "dz1_swir" and item.get("bounds_wgs84")]
        for left in swir:
            for right in vnir:
                first_bounds = left["bounds_wgs84"]
                second_bounds = right["bounds_wgs84"]
                first_center = ((first_bounds[0] + first_bounds[2]) / 2.0, (first_bounds[1] + first_bounds[3]) / 2.0)
                second_center = ((second_bounds[0] + second_bounds[2]) / 2.0, (second_bounds[1] + second_bounds[3]) / 2.0)
                pairs.append(
                    {
                        "acquisition_id": acquisition_id,
                        "vnir_product_id": right["product_id"],
                        "swir_product_id": left["product_id"],
                        "vnir_core_eligible": right["core_eligible"],
                        "swir_core_eligible": left["core_eligible"],
                        "center_distance_km": haversine_km(first_center, second_center),
                        **bbox_overlap_metrics(first_bounds, second_bounds),
                        "use_for_base_pretraining": False,
                    }
                )
    write_jsonl(output_path, pairs)


def stack_path(output_root: Path, record: Dict[str, Any]) -> Path:
    return output_root / "data" / "stacked_private" / record["sensor"] / (record["product_id"] + ".tif")


def validate_stack(path: Path, expected_count: int) -> bool:
    try:
        with rasterio.open(str(path)) as dataset:
            return dataset.count == expected_count and dataset.dtypes[0] == "float32" and dataset.width > 0 and dataset.height > 0
    except Exception:
        return False


def build_stack(input_root: Path, output_root: Path, record: Dict[str, Any], overwrite: bool) -> Path:
    destination = stack_path(output_root, record)
    if destination.exists() and not overwrite and validate_stack(destination, int(record["band_count"])):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.tif")
    if temporary.exists():
        temporary.unlink()

    band_paths = [input_root / relative for relative in record["relative_band_paths"]]
    with rasterio.open(str(band_paths[0])) as reference:
        profile = reference.profile.copy()
        profile.update(
            driver="GTiff",
            count=len(band_paths),
            dtype="float32",
            nodata=np.nan,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            predictor=3,
            zlevel=6,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(str(temporary), "w", **profile) as target:
            target.update_tags(
                product_id=record["product_id"],
                sensor=record["sensor"],
                level=record["level"],
                native_gsd_m=record["native_gsd_m"],
                effective_gsd_m=record["effective_gsd_m"],
                calibration="MTL_RADIANCE_MULT_ADD",
            )
            for band_index, (path, multiplier, addition, wavelength, bandwidth) in enumerate(
                zip(
                    band_paths,
                    record["radiance_mult"],
                    record["radiance_add"],
                    record["wavelength_nm"],
                    record["effective_bandwidth_nm"],
                ),
                start=1,
            ):
                with rasterio.open(str(path)) as source:
                    raw = source.read(1)
                values = raw.astype(np.float32)
                valid = raw != 0
                values = values * np.float32(multiplier) + np.float32(addition)
                values[~valid] = np.nan
                target.write(values, band_index)
                target.set_band_description(band_index, "B{}_{:.3f}nm".format(band_index, wavelength))
                target.update_tags(
                    band_index,
                    wavelength_nm=wavelength,
                    effective_bandwidth_nm=bandwidth,
                    bandwidth_source=record["bandwidth_source"],
                    radiance_mult=multiplier,
                    radiance_add=addition,
                )
    os.replace(str(temporary), str(destination))
    return destination


def build_stacks(args: argparse.Namespace, records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for record in records:
        if not record["core_eligible"]:
            counts["skipped_noncore"] += 1
            continue
        build_stack(args.input_root.resolve(), args.output_root, record, args.overwrite)
        counts["stacked_or_reused"] += 1
        counts[record["sensor"]] += 1
    write_json(args.output_root / "reports" / "stack_summary.json", dict(counts))
    return dict(counts)


class ShardWriter:
    def __init__(self, output_root: Path, samples_per_shard: int, overwrite: bool) -> None:
        self.root = output_root / "data" / "dz1_shards"
        self.values_root = self.root / "values"
        self.metadata_root = self.root / "metadata"
        self.samples_root = self.root / "samples"
        self.manifest_path = self.root / "manifest.jsonl"
        self.global_samples_path = output_root / "data" / "manifests" / "dz1_samples.jsonl"
        if (self.manifest_path.exists() or self.values_root.exists()) and not overwrite:
            raise FileExistsError("DZ1 shard output already exists; pass --overwrite or choose another output root")
        if overwrite:
            for path in (self.values_root, self.metadata_root, self.samples_root):
                if path.exists():
                    shutil.rmtree(str(path))
            for path in (self.manifest_path, self.global_samples_path):
                if path.exists():
                    path.unlink()
        self.values_root.mkdir(parents=True, exist_ok=True)
        self.metadata_root.mkdir(parents=True, exist_ok=True)
        self.samples_root.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self.buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.metadata_buffers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.shard_indices: Dict[str, int] = defaultdict(int)
        self.manifest_records: List[Dict[str, Any]] = []
        self.global_handle = self.global_samples_path.open("w", encoding="utf-8")

    def add(self, sensor: str, values: np.ndarray, metadata: Dict[str, Any]) -> None:
        self.buffers[sensor].append(np.asarray(values, dtype=np.float32))
        self.metadata_buffers[sensor].append(metadata)
        if len(self.buffers[sensor]) >= self.samples_per_shard:
            self.flush(sensor)

    def flush(self, sensor: str) -> None:
        values = self.buffers[sensor]
        metadata = self.metadata_buffers[sensor]
        if not values:
            return
        shard_index = self.shard_indices[sensor]
        stem = "{}_{:06d}".format(sensor, shard_index)
        values_path = self.values_root / (stem + ".npy")
        metadata_path = self.metadata_root / (stem + ".npz")
        samples_path = self.samples_root / (stem + ".jsonl")
        stacked = np.stack(values, axis=0).astype(np.float32, copy=False)
        np.save(str(values_path), stacked, allow_pickle=False)
        np.savez_compressed(
            str(metadata_path),
            source_product_index=np.asarray([item["source_product_index"] for item in metadata], dtype=np.int32),
            source_row=np.asarray([item["source_row"] for item in metadata], dtype=np.int32),
            source_col=np.asarray([item["source_col"] for item in metadata], dtype=np.int32),
            latitude=np.asarray([item["latitude"] for item in metadata], dtype=np.float32),
            longitude=np.asarray([item["longitude"] for item in metadata], dtype=np.float32),
            footprint_area_m2=np.asarray([item["footprint_area_m2"] for item in metadata], dtype=np.float32),
            valid_fraction=np.asarray([item["valid_fraction"] for item in metadata], dtype=np.float32),
            saturation_fraction=np.asarray([item["saturation_fraction"] for item in metadata], dtype=np.float32),
            brightness_p90=np.asarray([item["brightness_p90"] for item in metadata], dtype=np.float32),
            spectral_flatness=np.asarray([item["spectral_flatness"] for item in metadata], dtype=np.float32),
            robust_scale=np.asarray([item["robust_scale"] for item in metadata], dtype=np.float32),
        )
        for local_index, item in enumerate(metadata):
            item["shard_local_index"] = local_index
            item["values_path"] = str(values_path.relative_to(self.root)).replace("\\", "/")
            self.global_handle.write(json.dumps(item, ensure_ascii=False, default=json_default) + "\n")
        write_jsonl(samples_path, metadata)
        first = metadata[0]
        manifest_record = {
            "values_path": str(values_path.relative_to(self.root)).replace("\\", "/"),
            "metadata_path": str(metadata_path.relative_to(self.root)).replace("\\", "/"),
            "samples_path": str(samples_path.relative_to(self.root)).replace("\\", "/"),
            "count": int(stacked.shape[0]),
            "partition": "DZ1-202607-{}".format(sensor.upper()),
            "sensor": sensor,
            "level": "L2",
            "gsd_m": float(first["effective_gsd_m"]),
            "native_gsd_m": float(first["native_gsd_m"]),
            "resampling_factor": int(first["resampling_factor"]),
            "wavelength": first["wavelength_nm"],
            "fwhm": first["effective_bandwidth_nm"],
            "bandwidth_source": "MTL_MEASURED_MAX_MINUS_MIN_PROXY",
            "response_model": "boxcar_measured_span",
            "sample_shape": list(stacked.shape[1:]),
            "storage_dtype": "float32",
            "values_sha256": sha256_file(values_path),
            "metadata_sha256": sha256_file(metadata_path),
            "samples_sha256": sha256_file(samples_path),
        }
        self.manifest_records.append(manifest_record)
        self.buffers[sensor] = []
        self.metadata_buffers[sensor] = []
        self.shard_indices[sensor] += 1

    def close(self) -> List[Dict[str, Any]]:
        for sensor in sorted(self.buffers):
            self.flush(sensor)
        self.global_handle.close()
        write_jsonl(self.manifest_path, self.manifest_records)
        return self.manifest_records


def area_average(values: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return values
    channels, height, width = values.shape
    if height % factor or width % factor:
        raise ValueError("Area-average input shape is not divisible by factor")
    return values.reshape(channels, height // factor, factor, width // factor, factor).mean(axis=(2, 4))


def timestamp_ordinal(timestamp_iso: Optional[str]) -> Optional[float]:
    if not timestamp_iso:
        return None
    text = timestamp_iso.rstrip("Z")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text[:10], "%Y-%m-%d")
        except ValueError:
            return None
    return float(parsed.date().toordinal())


def patch_center(dataset: Any, source_row: int, source_col: int, source_size: int) -> Tuple[float, float]:
    center_row = source_row + source_size / 2.0
    center_col = source_col + source_size / 2.0
    x_coord, y_coord = xy(dataset.transform, center_row, center_col, offset="center")
    if dataset.crs is None:
        return float("nan"), float("nan")
    longitudes, latitudes = transform(dataset.crs, "EPSG:4326", [x_coord], [y_coord])
    return float(latitudes[0]), float(longitudes[0])


def extract_product(
    output_root: Path,
    record: Dict[str, Any],
    product_index: int,
    writer: ShardWriter,
    saturation_fraction_max: float,
    robust_scale_min: float,
    counters: Counter,
    brightness_values: Dict[str, List[float]],
    qa_values: Dict[str, Dict[str, List[float]]],
    patch_distributions: Dict[str, Counter],
) -> None:
    path = stack_path(output_root, record)
    if not validate_stack(path, int(record["band_count"])):
        raise FileNotFoundError("Missing or invalid stack: {}".format(path))
    source_size = int(record["source_window_size"])
    output_size = int(record["output_size"])
    factor = int(record["resampling_factor"])
    sensor = record["sensor"]
    saturation_values = np.asarray(
        [UINT16_SATURATION * mult + add for mult, add in zip(record["radiance_mult"], record["radiance_add"])],
        dtype=np.float32,
    ).reshape(-1, 1, 1)

    with rasterio.open(str(path)) as dataset:
        usable_width = dataset.width - dataset.width % source_size
        usable_height = dataset.height - dataset.height % source_size
        for source_row in range(0, usable_height, source_size):
            stripe = dataset.read(window=Window(0, source_row, usable_width, source_size)).astype(np.float32, copy=False)
            for source_col in range(0, usable_width, source_size):
                counters["candidate_patches"] += 1
                values = stripe[:, :, source_col : source_col + source_size]
                finite = np.isfinite(values)
                valid_fraction = float(finite.mean())
                if valid_fraction < 1.0:
                    counters["rejected_nodata"] += 1
                    continue
                saturation_fraction = float((values >= saturation_values * np.float32(0.999999)).mean())
                if saturation_fraction > saturation_fraction_max:
                    counters["rejected_saturation"] += 1
                    continue
                output = area_average(values, factor)
                if output.shape != (int(record["band_count"]), output_size, output_size):
                    raise ValueError("Unexpected output shape {} for {}".format(output.shape, record["product_id"]))
                sampled = output[:, ::4, ::4].reshape(-1)
                q25, q75 = np.percentile(sampled, [25.0, 75.0])
                robust_scale = float((q75 - q25) / 1.349)
                if not math.isfinite(robust_scale) or robust_scale < robust_scale_min:
                    counters["rejected_flat"] += 1
                    continue
                brightness_p90 = float(np.percentile(sampled, 90.0))
                per_band_std = output.reshape(output.shape[0], -1).std(axis=1)
                spectral_flatness = float(per_band_std.mean() / max(float(np.abs(output).mean()), 1.0e-6))
                latitude, longitude = patch_center(dataset, source_row, source_col, source_size)
                sample_id = "{}_r{:05d}_c{:05d}".format(record["product_id"], source_row, source_col)
                metadata = {
                    "sample_id": sample_id,
                    "source_product_index": product_index,
                    "source_product_id": record["product_id"],
                    "acquisition_id": record["acquisition_id"],
                    "sensor": sensor,
                    "level": "L2",
                    "source_row": source_row,
                    "source_col": source_col,
                    "source_window_size": source_size,
                    "output_size": output_size,
                    "native_gsd_m": record["native_gsd_m"],
                    "effective_gsd_m": record["effective_gsd_m"],
                    "resampling_factor": factor,
                    "footprint_area_m2": float((source_size * float(record["native_gsd_m"])) ** 2),
                    "latitude": latitude,
                    "longitude": longitude,
                    "timestamp_ordinal": timestamp_ordinal(record.get("timestamp_iso")),
                    "wavelength_nm": record["wavelength_nm"],
                    "effective_bandwidth_nm": record["effective_bandwidth_nm"],
                    "valid_fraction": valid_fraction,
                    "saturation_fraction": saturation_fraction,
                    "brightness_p90": brightness_p90,
                    "spectral_flatness": spectral_flatness,
                    "robust_scale": robust_scale,
                    "scene_cloud_cover_pct": record["cloud_cover_pct"],
                    "scene_image_quality": record["image_quality"],
                    "province": record.get("province"),
                    "county": record.get("county"),
                    "scene_environment": record.get("scene_environment"),
                    "encoded_metadata_fields": [
                        "modality",
                        "sensor",
                        "level",
                        "effective_gsd_m",
                        "footprint_area_m2",
                        "wavelength_nm",
                        "effective_bandwidth_nm",
                    ],
                }
                writer.add(sensor, output, metadata)
                counters["accepted_patches"] += 1
                counters["accepted_{}".format(sensor)] += 1
                brightness_values[sensor].append(brightness_p90)
                qa_values[sensor]["valid_fraction"].append(valid_fraction)
                qa_values[sensor]["saturation_fraction"].append(saturation_fraction)
                qa_values[sensor]["brightness_p90"].append(brightness_p90)
                qa_values[sensor]["spectral_flatness"].append(spectral_flatness)
                qa_values[sensor]["robust_scale"].append(robust_scale)
                province = str(record.get("province") or "<unknown>")
                scene = str(record.get("scene_environment") or "<unknown>")
                patch_distributions["sensor"][sensor] += 1
                patch_distributions["province"][province] += 1
                patch_distributions["scene_environment"][scene] += 1
                patch_distributions["sensor_province"][sensor + "|" + province] += 1
                patch_distributions["sensor_scene_environment"][sensor + "|" + scene] += 1


def build_audit_candidates(output_root: Path, brightness_values: Dict[str, List[float]]) -> Dict[str, Any]:
    thresholds = {
        sensor: float(np.percentile(np.asarray(values, dtype=np.float32), 99.0))
        for sensor, values in brightness_values.items()
        if values
    }
    source = output_root / "data" / "manifests" / "dz1_samples.jsonl"
    destination = output_root / "data" / "manifests" / "dz1_audit_candidates.jsonl"
    candidates: List[Dict[str, Any]] = []
    if source.exists():
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                threshold = thresholds.get(record["sensor"])
                if threshold is not None and float(record["brightness_p90"]) >= threshold:
                    candidates.append({**record, "audit_reason": "sensor_brightness_p99"})
    write_jsonl(destination, candidates)
    montage_paths = build_audit_montages(output_root, candidates)
    return {
        "brightness_p99": thresholds,
        "audit_candidate_count": len(candidates),
        "audit_montages": montage_paths,
    }


def build_audit_montages(output_root: Path, candidates: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        grouped[record["sensor"]].append(record)
    outputs: Dict[str, str] = {}
    shard_root = output_root / "data" / "dz1_shards"
    for sensor, records in grouped.items():
        records = sorted(records, key=lambda item: float(item["brightness_p90"]), reverse=True)[:24]
        if not records:
            continue
        tile_size, text_height, columns = 128, 24, 6
        rows = int(math.ceil(len(records) / columns))
        canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + text_height)), "white")
        draw = ImageDraw.Draw(canvas)
        cache: Dict[str, np.ndarray] = {}
        for index, record in enumerate(records):
            relative = record["values_path"]
            if relative not in cache:
                cache[relative] = np.load(str(shard_root / relative), mmap_mode="r")
            cube = np.asarray(cache[relative][int(record["shard_local_index"])], dtype=np.float32)
            bands = (7, 4, 2) if sensor == "dz1_vnir" else (8, 5, 1)
            rgb = cube[list(bands)]
            image_channels = []
            for channel in rgb:
                low, high = np.percentile(channel, [2.0, 98.0])
                scaled = np.clip((channel - low) / max(high - low, 1.0e-6), 0.0, 1.0)
                image_channels.append((scaled * 255.0).astype(np.uint8))
            image = Image.fromarray(np.stack(image_channels, axis=-1), mode="RGB").resize((tile_size, tile_size))
            x_pos = (index % columns) * tile_size
            y_pos = (index // columns) * (tile_size + text_height)
            canvas.paste(image, (x_pos, y_pos))
            label = "{} {:.2f}".format(record["source_product_id"][-18:-10], float(record["brightness_p90"]))
            draw.text((x_pos + 2, y_pos + tile_size + 3), label, fill="black")
        destination = output_root / "reports" / ("audit_candidates_" + sensor + ".png")
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(destination))
        outputs[sensor] = str(destination)
    return outputs


def summarize_quantiles(qa_values: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    percentiles = [0.0, 1.0, 5.0, 50.0, 95.0, 99.0, 100.0]
    output: Dict[str, Any] = {}
    for sensor, metrics in sorted(qa_values.items()):
        output[sensor] = {}
        for metric, values in sorted(metrics.items()):
            array = np.asarray(values, dtype=np.float64)
            quantiles = np.percentile(array, percentiles) if array.size else np.full(len(percentiles), np.nan)
            output[sensor][metric] = {
                "count": int(array.size),
                **{"p{:g}".format(level): float(value) for level, value in zip(percentiles, quantiles)},
            }
    return output


def build_checksum_report(output_root: Path, manifest_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    manifest_path = output_root / "data" / "dz1_shards" / "manifest.jsonl"
    core_path = output_root / "data" / "manifests" / "dz1_core.jsonl"
    digest = hashlib.sha256()
    for record in manifest_records:
        digest.update(str(record["values_sha256"]).encode("ascii"))
        digest.update(str(record["metadata_sha256"]).encode("ascii"))
        digest.update(str(record["samples_sha256"]).encode("ascii"))
    report = {
        "manifest_sha256": sha256_file(manifest_path),
        "core_manifest_sha256": sha256_file(core_path),
        "ordered_shard_files_digest_sha256": digest.hexdigest(),
        "shard_count": len(manifest_records),
        "verified_file_count": len(manifest_records) * 3,
    }
    write_json(output_root / "reports" / "dataset_checksums.json", report)
    return report


def extract_shards(args: argparse.Namespace, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    core_records = [record for record in records if record["core_eligible"]]
    writer = ShardWriter(args.output_root, args.samples_per_shard, args.overwrite)
    counters: Counter = Counter()
    brightness_values: Dict[str, List[float]] = defaultdict(list)
    qa_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    patch_distributions: Dict[str, Counter] = defaultdict(Counter)
    for product_index, record in enumerate(core_records):
        extract_product(
            args.output_root,
            record,
            product_index,
            writer,
            args.saturation_fraction_max,
            args.robust_scale_min,
            counters,
            brightness_values,
            qa_values,
            patch_distributions,
        )
    manifest_records = writer.close()
    dz1_manifest = args.output_root / "data" / "manifests" / "dz1_core.jsonl"
    write_jsonl(dz1_manifest, manifest_records)
    audit_summary = build_audit_candidates(args.output_root, brightness_values)
    for key in (
        "candidate_patches",
        "rejected_nodata",
        "rejected_saturation",
        "rejected_flat",
        "accepted_patches",
        "accepted_dz1_vnir",
        "accepted_dz1_swir",
    ):
        counters.setdefault(key, 0)
    quantile_summary = summarize_quantiles(qa_values)
    distribution_summary = {
        name: dict(sorted(counts.items())) for name, counts in sorted(patch_distributions.items())
    }
    write_json(args.output_root / "reports" / "qa_numeric_quantiles.json", quantile_summary)
    write_json(args.output_root / "reports" / "patch_distributions.json", distribution_summary)
    checksum_summary = build_checksum_report(args.output_root, manifest_records)
    summary: Dict[str, Any] = {
        **dict(counters),
        "core_product_count": len(core_records),
        "shard_count": len(manifest_records),
        "samples_per_shard": args.samples_per_shard,
        "vnir_mode": args.vnir_mode,
        "qa_quantiles_report": str(args.output_root / "reports" / "qa_numeric_quantiles.json"),
        "patch_distributions_report": str(args.output_root / "reports" / "patch_distributions.json"),
        "checksum_report": str(args.output_root / "reports" / "dataset_checksums.json"),
        "dataset_checksums": checksum_summary,
        **audit_summary,
    }
    write_json(args.output_root / "reports" / "extraction_summary.json", summary)
    return summary


def load_or_build_inventory(args: argparse.Namespace) -> List[Dict[str, Any]]:
    path = args.output_root / "data" / "manifests" / "products.jsonl"
    if path.exists() and not args.overwrite:
        records = read_jsonl(path)
        selected = set(args.acquisition)
        if selected:
            records = [record for record in records if record["acquisition_id"] in selected]
        return records
    return build_inventory(args)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "inventory":
        records = build_inventory(args)
        print(json.dumps({"stage": "inventory", "products": len(records)}, ensure_ascii=False))
        return

    records = load_or_build_inventory(args)
    if args.stage in {"stack", "all"} and not args.skip_stacks:
        stack_summary = build_stacks(args, records)
        if args.stage == "stack":
            print(json.dumps({"stage": "stack", **stack_summary}, ensure_ascii=False))
            return
    if args.stage in {"extract", "all"}:
        extraction = extract_shards(args, records)
        print(json.dumps({"stage": "extract", **extraction}, ensure_ascii=False))


if __name__ == "__main__":
    main()
