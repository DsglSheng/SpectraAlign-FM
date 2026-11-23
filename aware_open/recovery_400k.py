from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RecoveryManifestEntry:
    source_key: str
    source_file: str
    patch_dir: str
    metadata_path: str
    sensor: str
    level: str
    sensor_level: str
    download_url: str
    expected_raw_name: str


def normalize_source_key(source_name: str) -> str:
    return Path(source_name).stem


def construct_download_url(flight: dict[str, str]) -> str:
    url = flight.get("download_url", "").strip()
    sensor = flight.get("sensor", "")
    level = flight.get("level", "")
    flight_id = flight.get("flight_id", "")

    if sensor == "AVIRIS-3":
        if level == "L1":
            return f"https://data.ornldaac.earthdata.nasa.gov/protected/aviris/AV3_L1B_RDN/data/{flight_id}.nc"
        return f"https://data.ornldaac.earthdata.nasa.gov/protected/aviris/AV3_L2A_RFL/data/{flight_id}.nc"

    if url:
        return url

    if sensor == "AVIRIS-C":
        if level == "L1":
            return f"https://daac.ornl.gov/daacdata/aviris/AVIRIS-Classic_L1B_Radiance/data/{flight_id}"
        return f"https://daac.ornl.gov/daacdata/aviris/AVIRIS_L2_Refl/data/{flight_id}"

    if sensor == "AVIRIS-NG":
        if level == "L1":
            return f"https://daac.ornl.gov/daacdata/aviris/AVIRISNG_L1B_RDN/data/{flight_id}"
        return f"https://daac.ornl.gov/daacdata/aviris/AVIRISNG_L2_RFL/data/{flight_id}"

    return url


def load_flight_index(metadata_csv: Path) -> dict[str, dict[str, str]]:
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["flight_id"]: row for row in rows}


def build_recovery_manifest(
    patches_root: Path,
    metadata_csv: Path,
    *,
    only_missing_gsd: bool = True,
) -> list[RecoveryManifestEntry]:
    flight_index = load_flight_index(metadata_csv)
    entries: list[RecoveryManifestEntry] = []

    for patch_dir in sorted(path for path in patches_root.glob("*") if path.is_dir()):
        metadata_path = patch_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if only_missing_gsd and metadata.get("gsd") is not None:
            continue

        source_file = str(metadata.get("source_file", "")).strip()
        if not source_file:
            raise ValueError(f"{metadata_path} is missing source_file")

        source_key = normalize_source_key(source_file)
        flight = flight_index.get(source_key)
        if flight is None:
            raise KeyError(f"Could not match {source_key} in {metadata_csv}")

        download_url = construct_download_url(flight)
        if not download_url:
            raise ValueError(f"No download URL available for {source_key}")

        expected_raw_name = Path(download_url).name or f"{source_key}.nc"
        entries.append(
            RecoveryManifestEntry(
                source_key=source_key,
                source_file=source_file,
                patch_dir=str(patch_dir),
                metadata_path=str(metadata_path),
                sensor=flight.get("sensor", ""),
                level=flight.get("level", ""),
                sensor_level=flight.get("sensor_level", ""),
                download_url=download_url,
                expected_raw_name=expected_raw_name,
            )
        )

    return entries


def write_manifest_csv(entries: list[RecoveryManifestEntry], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(entries[0]).keys()) if entries else list(RecoveryManifestEntry.__dataclass_fields__.keys()))
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def load_manifest_csv(manifest_path: Path) -> list[RecoveryManifestEntry]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [RecoveryManifestEntry(**row) for row in rows]


def resolve_raw_scene_path(raw_root: Path, entry: RecoveryManifestEntry) -> Path | None:
    exact = raw_root / entry.expected_raw_name
    if exact.exists():
        return exact

    source_name = Path(entry.source_file).name
    candidate_names = {
        source_name,
        entry.expected_raw_name,
        f"{entry.source_key}.nc",
        entry.source_key,
    }
    for candidate in sorted(raw_root.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.name in candidate_names or candidate.stem == entry.source_key:
            return candidate
    return None


def _try_rasterio_gsd(scene_path: Path) -> float | None:
    try:
        import rasterio
    except ImportError:  # pragma: no cover - optional dependency
        return None

    try:
        with rasterio.open(scene_path) as dataset:
            x_res = abs(float(dataset.transform.a))
            y_res = abs(float(dataset.transform.e))
    except Exception:
        return None

    if x_res <= 0 or y_res <= 0:
        return None
    return float((x_res + y_res) / 2.0)


def _parse_geotransform(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        parts = [part for part in value.replace(",", " ").split() if part]
        if len(parts) == 6:
            try:
                coeffs = [float(part) for part in parts]
            except ValueError:
                return None
            return abs(coeffs[1]) if coeffs[1] else abs(coeffs[5])
        return None
    return None


def _median_spacing(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    diffs = np.abs(np.diff(finite))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _iter_nc_spaces(dataset: Any) -> list[Any]:
    spaces = [dataset]
    for group in dataset.groups.values():
        spaces.extend(_iter_nc_spaces(group))
    return spaces


def extract_gsd_from_netcdf(scene_path: Path) -> float:
    rasterio_gsd = _try_rasterio_gsd(scene_path)
    if rasterio_gsd is not None:
        return rasterio_gsd

    import netCDF4 as nc

    dataset = nc.Dataset(scene_path, "r")
    try:
        for space in _iter_nc_spaces(dataset):
            for attr_name in ("GeoTransform", "geotransform", "geo_transform"):
                gsd = _parse_geotransform(getattr(space, attr_name, None))
                if gsd is not None:
                    return gsd

            for attr_name in ("spatial_resolution", "spatialresolution", "pixel_size", "pixelsize"):
                value = getattr(space, attr_name, None)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

            variables = getattr(space, "variables", {})
            for x_name, y_name in (
                ("x", "y"),
                ("projection_x_coordinate", "projection_y_coordinate"),
                ("easting", "northing"),
            ):
                if x_name not in variables or y_name not in variables:
                    continue
                x_res = _median_spacing(np.asarray(variables[x_name][:], dtype=float))
                y_res = _median_spacing(np.asarray(variables[y_name][:], dtype=float))
                if x_res is None or y_res is None:
                    continue
                return float((x_res + y_res) / 2.0)
    finally:
        dataset.close()

    raise ValueError(f"Could not extract GSD from NetCDF scene {scene_path}")


def extract_gsd_from_scene(scene_path: Path) -> float:
    if scene_path.suffix.lower() == ".nc":
        return extract_gsd_from_netcdf(scene_path)

    rasterio_gsd = _try_rasterio_gsd(scene_path)
    if rasterio_gsd is None:
        raise ValueError(f"Could not extract GSD from raster scene {scene_path}")
    return rasterio_gsd


def apply_gsd_to_patch_metadata(
    metadata_path: Path,
    *,
    gsd: float,
    scene_path: Path,
) -> None:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["gsd"] = float(gsd)
    payload["gsd_source"] = "raw_georeferencing"
    payload["gsd_scene_file"] = scene_path.name
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def backfill_manifest_entries(
    entries: list[RecoveryManifestEntry],
    raw_root: Path,
) -> dict[str, int]:
    updated = 0
    missing_raw = 0

    for entry in entries:
        scene_path = resolve_raw_scene_path(raw_root, entry)
        if scene_path is None:
            missing_raw += 1
            continue
        gsd = extract_gsd_from_scene(scene_path)
        apply_gsd_to_patch_metadata(Path(entry.metadata_path), gsd=gsd, scene_path=scene_path)
        updated += 1

    return {
        "entries": len(entries),
        "updated": updated,
        "missing_raw": missing_raw,
    }


def load_official_download_utils(getdata_root: Path) -> tuple[type[Any], type[Any]]:
    module_path = getdata_root / "utils" / "download_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find official downloader at {module_path}")

    spec = importlib.util.spec_from_file_location("aware_open_getdata_download_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EarthdataSession, module.ResumableDownloader


def download_manifest_entries(
    entries: list[RecoveryManifestEntry],
    *,
    getdata_root: Path,
    raw_root: Path,
    progress_file: Path,
    username: str = "",
    password: str = "",
    chunk_size_mb: int = 8,
    retry_count: int = 3,
) -> dict[str, int]:
    EarthdataSession, ResumableDownloader = load_official_download_utils(getdata_root)
    session = EarthdataSession(username=username, password=password)
    downloader = ResumableDownloader(
        session=session,
        download_dir=str(raw_root),
        progress_file=str(progress_file),
        chunk_size_mb=chunk_size_mb,
    )

    downloaded = 0
    skipped = 0
    for entry in entries:
        if resolve_raw_scene_path(raw_root, entry) is not None:
            skipped += 1
            continue
        filepath = downloader.download_file(
            entry.download_url,
            filename=entry.expected_raw_name,
            retry_count=retry_count,
        )
        if filepath:
            downloaded += 1

    return {
        "entries": len(entries),
        "downloaded": downloaded,
        "skipped_existing": skipped,
    }
