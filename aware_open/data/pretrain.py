from __future__ import annotations

import bisect
import json
import math
import re
import shutil
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..data_types import ModalityBatch, SampleMetadataBatch
from ..sigma450k_profiles import resolve_sigma450k_profile_from_path

try:
    from scipy.io import loadmat
except ImportError:  # pragma: no cover - runtime dependency in training environments
    loadmat = None


@dataclass(frozen=True)
class SampleMetadata:
    modality: str
    sensor: str
    level: str
    gsd: float
    latitude: float | None = None
    longitude: float | None = None
    timestamp: float | None = None
    footprint_area: float | None = None


@dataclass(frozen=True)
class ModalitySample:
    values: torch.Tensor
    wavelengths: torch.Tensor
    fwhm: torch.Tensor | None
    metadata: SampleMetadata
    valid_mask: torch.Tensor | None = None


_AVIRIS_TIMESTAMP_PATTERN = re.compile(r"^[A-Za-z0-9]{3}(\d{8})t(\d{6})")


def _load_or_build_jsonl_index(cache_path: Path | None, builder: callable) -> list[dict[str, Any]]:
    if cache_path is not None and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    records = builder()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return records


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        value = text
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _parse_measurement_day(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _optional_float(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")

    for candidate in (
        lambda x: datetime.fromisoformat(x),
        lambda x: datetime.strptime(x, "%Y-%m-%d"),
        lambda x: datetime.strptime(x, "%Y/%m/%d"),
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
        lambda x: datetime.strptime(x, "%Y/%m/%d %H:%M:%S"),
    ):
        try:
            parsed = candidate(text)
        except ValueError:
            continue
        return float(parsed.date().toordinal())
    return None


def _parse_aviris_timestamp(source_file: str) -> float | None:
    match = _AVIRIS_TIMESTAMP_PATTERN.match(Path(source_file).stem)
    if match is None:
        return None
    acquisition_day = date.fromisoformat(
        f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:8]}"
    )
    return float(acquisition_day.toordinal())


def _infer_aviris_sensor_token(source_file: str) -> str:
    upper = Path(source_file).stem.upper()
    if upper.startswith("AV3"):
        return "av3"
    if upper.startswith("ANG"):
        return "avng"
    if upper.startswith("F"):
        return "avc"
    return "others"


def _infer_level_token(source_file: str) -> str:
    upper = Path(source_file).stem.upper()
    if "L2" in upper or "RFL" in upper or "CORR" in upper:
        return "L2"
    if "L1" in upper or "RDN" in upper:
        return "L1"
    return "unknown"


def _first_np_array(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as arrays:
        if not arrays.files:
            raise ValueError(f"{npz_path} does not contain any arrays.")
        return np.asarray(arrays[arrays.files[0]])


def _crop_cube(values: np.ndarray, crop_size: int | None, random_crop: bool) -> np.ndarray:
    if crop_size is None:
        return values
    if values.ndim != 3:
        raise ValueError(f"Expected a spectral cube of shape (C, H, W), got {tuple(values.shape)}.")

    bands, height, width = values.shape
    del bands
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"crop_size={crop_size} exceeds input spatial size {(height, width)}."
        )
    if crop_size == height and crop_size == width:
        return values

    if random_crop:
        top = int(torch.randint(0, height - crop_size + 1, (1,)).item())
        left = int(torch.randint(0, width - crop_size + 1, (1,)).item())
    else:
        top = max(0, (height - crop_size) // 2)
        left = max(0, (width - crop_size) // 2)
    return values[:, top : top + crop_size, left : left + crop_size]


def _normalize_values(
    values: np.ndarray,
    mode: str,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    if mode == "none":
        return values

    data = values.astype(np.float32, copy=False)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if data.ndim == 3 and mask.ndim == 1:
            mask = mask[:, None, None]
        selected = data[mask]
    else:
        selected = data.reshape(-1)

    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return data

    if mode == "mean_std":
        center = float(selected.mean())
        scale = float(selected.std())
    elif mode == "robust":
        center = float(np.median(selected))
        q1, q3 = np.percentile(selected, [25.0, 75.0])
        scale = float((q3 - q1) / 1.349)
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    if not np.isfinite(scale) or scale < 1.0e-6:
        scale = 1.0
    return (data - center) / scale


def _stack_optional_scalars(values: Sequence[float | None]) -> torch.Tensor | None:
    if not values or any(value is None for value in values):
        return None
    return torch.tensor(values, dtype=torch.float32)


def _collate_metadata(samples: Sequence[ModalitySample]) -> SampleMetadataBatch:
    metadata = [sample.metadata for sample in samples]
    return SampleMetadataBatch(
        modality=[item.modality for item in metadata],
        sensor=[item.sensor for item in metadata],
        level=[item.level for item in metadata],
        gsd=torch.tensor([item.gsd for item in metadata], dtype=torch.float32),
        latitude=_stack_optional_scalars([item.latitude for item in metadata]),
        longitude=_stack_optional_scalars([item.longitude for item in metadata]),
        timestamp=_stack_optional_scalars([item.timestamp for item in metadata]),
        footprint_area=_stack_optional_scalars([item.footprint_area for item in metadata]),
    )


def collate_modality_samples(samples: Sequence[ModalitySample]) -> ModalityBatch:
    if not samples:
        raise ValueError("collate_modality_samples requires at least one sample.")

    first = samples[0]
    values_dim = first.values.dim()
    batch_size = len(samples)
    max_bands = max(int(sample.values.shape[0]) for sample in samples)

    if values_dim == 3:
        spatial_shapes = {tuple(sample.values.shape[1:]) for sample in samples}
        if len(spatial_shapes) != 1:
            raise ValueError(
                f"All 2D modality samples must share the same spatial size, got {sorted(spatial_shapes)}."
            )
        height, width = next(iter(spatial_shapes))
        values = torch.zeros(batch_size, max_bands, height, width, dtype=first.values.dtype)
    elif values_dim == 1:
        values = torch.zeros(batch_size, max_bands, dtype=first.values.dtype)
    else:
        raise ValueError(
            f"Expected sample tensors shaped as (C,) or (C, H, W), got rank {values_dim}."
        )

    wavelengths = torch.zeros(batch_size, max_bands, dtype=torch.float32)
    valid_mask = torch.zeros(batch_size, max_bands, dtype=torch.bool)

    fwhm_present = [sample.fwhm is not None for sample in samples]
    fwhm = torch.zeros(batch_size, max_bands, dtype=torch.float32) if any(fwhm_present) else None

    for index, sample in enumerate(samples):
        bands = int(sample.values.shape[0])
        if sample.values.dim() != values_dim:
            raise ValueError("All collated samples must have the same tensor rank.")

        values[index, :bands] = sample.values
        wavelengths[index, :bands] = sample.wavelengths.float()

        if sample.valid_mask is None:
            valid_mask[index, :bands] = True
        else:
            valid_mask[index, :bands] = sample.valid_mask.bool()

        if fwhm is not None:
            if sample.fwhm is not None:
                fwhm[index, :bands] = sample.fwhm.float()

    return ModalityBatch(
        values=values,
        wavelengths=wavelengths,
        fwhm=fwhm,
        metadata=_collate_metadata(samples),
        valid_mask=valid_mask,
    )


def move_batch_to_device(batch: ModalityBatch, device: torch.device) -> ModalityBatch:
    return replace(
        batch,
        values=batch.values.to(device, non_blocking=True),
        wavelengths=batch.wavelengths.to(device, non_blocking=True),
        fwhm=None if batch.fwhm is None else batch.fwhm.to(device, non_blocking=True),
        valid_mask=None if batch.valid_mask is None else batch.valid_mask.to(device, non_blocking=True),
        metadata=SampleMetadataBatch(
            modality=batch.metadata.modality,
            sensor=batch.metadata.sensor,
            level=batch.metadata.level,
            gsd=batch.metadata.gsd.to(device, non_blocking=True),
            latitude=None if batch.metadata.latitude is None else batch.metadata.latitude.to(device, non_blocking=True),
            longitude=None if batch.metadata.longitude is None else batch.metadata.longitude.to(device, non_blocking=True),
            timestamp=None if batch.metadata.timestamp is None else batch.metadata.timestamp.to(device, non_blocking=True),
            footprint_area=None
            if batch.metadata.footprint_area is None
            else batch.metadata.footprint_area.to(device, non_blocking=True),
        ),
    )


class GroundSpectralDataset(Dataset[ModalitySample]):
    def __init__(
        self,
        root: str | Path,
        *,
        gsd_proxy_m: float = 0.01,
        normalization: str = "none",
        index_cache_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.native_root = self.root / "native"
        self.gsd_proxy_m = float(gsd_proxy_m)
        self.normalization = normalization
        self.records = _load_or_build_jsonl_index(
            None if index_cache_path is None else Path(index_cache_path),
            self._build_index_records,
        )

    def _build_index_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for json_path in sorted(self.native_root.rglob("*.json")):
            npz_path = json_path.with_suffix(".npz")
            if not npz_path.exists():
                continue
            records.append(
                {
                    "json_path": str(json_path),
                    "npz_path": str(npz_path),
                }
            )
        if not records:
            raise ValueError(f"No paired ground samples were found under {self.native_root}.")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ModalitySample:
        record = self.records[index]
        json_path = Path(record["json_path"])
        npz_path = Path(record["npz_path"])

        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as arrays:
            reflectance = np.asarray(arrays["reflectance"], dtype=np.float32)
            wavelengths = np.asarray(arrays["wavelengths"], dtype=np.float32)
            quality_codes = np.asarray(arrays["quality_mask"], dtype=np.uint8) if "quality_mask" in arrays else None
            instrument_fwhm = np.asarray(arrays["instrument_fwhm"], dtype=np.float32) if "instrument_fwhm" in arrays else None
        good_band_mask = None if quality_codes is None else (quality_codes == 0)
        reflectance = _normalize_values(reflectance, self.normalization, valid_mask=good_band_mask)

        return ModalitySample(
            values=torch.from_numpy(reflectance),
            wavelengths=torch.from_numpy(wavelengths),
            fwhm=None if instrument_fwhm is None else torch.from_numpy(instrument_fwhm),
            valid_mask=None if good_band_mask is None else torch.from_numpy(good_band_mask),
            metadata=SampleMetadata(
                modality="ground",
                sensor="field",
                level="field",
                gsd=self.gsd_proxy_m,
                latitude=_optional_float(metadata.get("latitude")),
                longitude=_optional_float(metadata.get("longitude")),
                timestamp=_parse_measurement_day(metadata.get("measurement_date")),
                footprint_area=None,
            ),
        )


class Hyper400KPatchDataset(Dataset[ModalitySample]):
    def __init__(
        self,
        root: str | Path,
        *,
        crop_size: int | None = None,
        random_crop: bool = True,
        normalization: str = "mean_std",
        strict_gsd: bool = True,
        gsd_proxy_m: float | None = None,
        index_cache_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.patches_root = self.root / "patches"
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.normalization = normalization
        self.strict_gsd = strict_gsd
        self.gsd_proxy_m = gsd_proxy_m
        self.records = _load_or_build_jsonl_index(
            None if index_cache_path is None else Path(index_cache_path),
            self._build_index_records,
        )
        self._metadata_cache: dict[Path, dict[str, Any]] = {}

    def _build_index_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for scene_dir in sorted(path for path in self.patches_root.glob("*") if path.is_dir()):
            metadata_path = scene_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            gsd = metadata.get("gsd")
            if gsd is None and self.strict_gsd and self.gsd_proxy_m is None:
                raise ValueError(
                    f"{metadata_path} is missing exact gsd. Backfill it before training or set a proxy explicitly."
                )

            patch_paths = sorted(scene_dir.glob("*.npz"))
            for patch_path in patch_paths:
                records.append(
                    {
                        "patch_path": str(patch_path),
                        "metadata_path": str(metadata_path),
                    }
                )
        if not records:
            raise ValueError(f"No 400k patch samples were found under {self.patches_root}.")
        return records

    def _load_metadata(self, metadata_path: Path) -> dict[str, Any]:
        cached = self._metadata_cache.get(metadata_path)
        if cached is not None:
            return cached
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._metadata_cache[metadata_path] = metadata
        return metadata

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ModalitySample:
        record = self.records[index]
        patch_path = Path(record["patch_path"])
        metadata_path = Path(record["metadata_path"])
        metadata = self._load_metadata(metadata_path)

        cube = np.asarray(_first_np_array(patch_path), dtype=np.float32)
        cube = _crop_cube(cube, self.crop_size, self.random_crop)
        cube = _normalize_values(cube, self.normalization)

        gsd = metadata.get("gsd")
        if gsd is None:
            if self.gsd_proxy_m is None:
                raise ValueError(f"{metadata_path} is missing gsd and no proxy was configured.")
            gsd_value = float(self.gsd_proxy_m)
        else:
            gsd_value = float(gsd)

        height, width = cube.shape[-2:]
        source_file = str(metadata.get("source_file", ""))
        return ModalitySample(
            values=torch.from_numpy(cube),
            wavelengths=torch.tensor(metadata["wavelength"], dtype=torch.float32),
            fwhm=torch.tensor(metadata["fwhm"], dtype=torch.float32),
            valid_mask=None,
            metadata=SampleMetadata(
                modality="air",
                sensor=_infer_aviris_sensor_token(source_file),
                level=_infer_level_token(source_file),
                gsd=gsd_value,
                latitude=None,
                longitude=None,
                timestamp=_parse_aviris_timestamp(source_file),
                footprint_area=float(height * width) * (gsd_value**2),
            ),
        )


@dataclass(frozen=True)
class Sigma450KShardSpec:
    values_path: Path
    count: int
    sensor: str
    level: str
    gsd_m: float
    wavelengths: torch.Tensor
    fwhm: torch.Tensor | None
    sample_shape: tuple[int, int, int]


def build_sigma450k_shards(
    source_root: str | Path,
    output_root: str | Path,
    *,
    samples_per_shard: int = 256,
    overwrite: bool = False,
) -> dict[str, int]:
    if loadmat is None:
        raise ImportError("scipy is required to convert sigma450k .mat files into shards.")
    if samples_per_shard <= 0:
        raise ValueError(f"samples_per_shard must be positive, got {samples_per_shard}.")

    source_root = Path(source_root)
    output_root = Path(output_root)
    manifest_path = output_root / "manifest.jsonl"
    shards_root = output_root / "shards"

    if manifest_path.exists() or shards_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already contains shard outputs. Pass overwrite=True to rebuild them."
            )
        if output_root.exists():
            shutil.rmtree(output_root)

    shards_root.mkdir(parents=True, exist_ok=True)

    manifest_records: list[dict[str, Any]] = []
    total_samples = 0
    total_shards = 0

    for partition_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        mat_paths = sorted(partition_dir.glob("*.mat"))
        if not mat_paths:
            continue

        pending_cubes: list[np.ndarray] = []
        storage_dtype: np.dtype[Any] | None = None
        sample_shape: tuple[int, int, int] | None = None
        profile = None
        partition_shard_index = 0

        for mat_path in mat_paths:
            content = loadmat(mat_path)
            if "img" not in content:
                raise ValueError(f"{mat_path} does not contain an 'img' array.")

            cube = np.asarray(content["img"])
            if cube.ndim != 3:
                raise ValueError(f"Expected sigma450k cube of shape (C, H, W), got {tuple(cube.shape)} for {mat_path}.")

            if sample_shape is None:
                sample_shape = tuple(int(dim) for dim in cube.shape)
                profile = resolve_sigma450k_profile_from_path(mat_path, int(cube.shape[0]))
            elif tuple(int(dim) for dim in cube.shape) != sample_shape:
                raise ValueError(
                    f"{mat_path} shape {tuple(cube.shape)} does not match prior sigma450k shape {sample_shape} "
                    f"within partition {partition_dir.name}."
                )

            if storage_dtype is None:
                storage_dtype = cube.dtype
            elif cube.dtype != storage_dtype:
                storage_dtype = np.result_type(storage_dtype, cube.dtype)
                pending_cubes = [pending.astype(storage_dtype, copy=False) for pending in pending_cubes]

            pending_cubes.append(np.asarray(cube, dtype=storage_dtype))

            if len(pending_cubes) == samples_per_shard:
                shard_path = shards_root / f"{partition_dir.name}_{partition_shard_index:06d}.npy"
                stacked = np.stack(pending_cubes, axis=0)
                np.save(shard_path, stacked, allow_pickle=False)
                manifest_records.append(
                    {
                        "values_path": str(shard_path.relative_to(output_root)).replace("\\", "/"),
                        "count": int(stacked.shape[0]),
                        "partition": partition_dir.name,
                        "sensor": profile.sensor,
                        "level": profile.level,
                        "gsd_m": float(profile.gsd_m),
                        "wavelength": profile.wavelengths_nm.astype(np.float32).tolist(),
                        "fwhm": None if profile.fwhm_nm is None else profile.fwhm_nm.astype(np.float32).tolist(),
                        "sample_shape": list(sample_shape),
                        "storage_dtype": np.dtype(stacked.dtype).name,
                    }
                )
                total_samples += int(stacked.shape[0])
                total_shards += 1
                partition_shard_index += 1
                pending_cubes.clear()

        if pending_cubes:
            if sample_shape is None or profile is None:
                raise ValueError(f"Failed to resolve sigma450k profile for partition {partition_dir.name}.")
            shard_path = shards_root / f"{partition_dir.name}_{partition_shard_index:06d}.npy"
            stacked = np.stack(pending_cubes, axis=0)
            np.save(shard_path, stacked, allow_pickle=False)
            manifest_records.append(
                {
                    "values_path": str(shard_path.relative_to(output_root)).replace("\\", "/"),
                    "count": int(stacked.shape[0]),
                    "partition": partition_dir.name,
                    "sensor": profile.sensor,
                    "level": profile.level,
                    "gsd_m": float(profile.gsd_m),
                    "wavelength": profile.wavelengths_nm.astype(np.float32).tolist(),
                    "fwhm": None if profile.fwhm_nm is None else profile.fwhm_nm.astype(np.float32).tolist(),
                    "sample_shape": list(sample_shape),
                    "storage_dtype": np.dtype(stacked.dtype).name,
                }
            )
            total_samples += int(stacked.shape[0])
            total_shards += 1

    if not manifest_records:
        raise ValueError(f"No sigma450k .mat samples were found under {source_root}.")

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return {
        "total_samples": total_samples,
        "total_shards": total_shards,
        "samples_per_shard": int(samples_per_shard),
    }


class Sigma450KShardDataset(Dataset[ModalitySample]):
    def __init__(
        self,
        root: str | Path,
        *,
        crop_size: int | None = None,
        random_crop: bool = False,
        normalization: str = "mean_std",
        manifest_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.normalization = normalization
        self.manifest_path = Path(manifest_path) if manifest_path is not None else self.root / "manifest.jsonl"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Sigma450k shard manifest not found at {self.manifest_path}. "
                "Build sigma450k shards before running space pretraining."
            )

        self.shards: list[Sigma450KShardSpec] = []
        self.sample_offsets: list[int] = [0]
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                values_path = Path(record["values_path"])
                if not values_path.is_absolute():
                    values_path = self.root / values_path
                shard = Sigma450KShardSpec(
                    values_path=values_path,
                    count=int(record["count"]),
                    sensor=str(record["sensor"]),
                    level=str(record["level"]),
                    gsd_m=float(record["gsd_m"]),
                    wavelengths=torch.tensor(record["wavelength"], dtype=torch.float32),
                    fwhm=None
                    if record.get("fwhm") is None
                    else torch.tensor(record["fwhm"], dtype=torch.float32),
                    sample_shape=tuple(int(dim) for dim in record["sample_shape"]),
                )
                self.shards.append(shard)
                self.sample_offsets.append(self.sample_offsets[-1] + shard.count)

        if not self.shards:
            raise ValueError(f"No sigma450k shards were listed in {self.manifest_path}.")

        self._cached_shard_index: int | None = None
        self._cached_values: np.ndarray | None = None

    def __len__(self) -> int:
        return self.sample_offsets[-1]

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.sample_offsets, index) - 1
        sample_index = index - self.sample_offsets[shard_index]
        return shard_index, sample_index

    def _load_shard_values(self, shard_index: int) -> np.ndarray:
        if self._cached_shard_index != shard_index or self._cached_values is None:
            self._cached_values = np.load(self.shards[shard_index].values_path, mmap_mode="r")
            self._cached_shard_index = shard_index
        return self._cached_values

    def __getitem__(self, index: int) -> ModalitySample:
        shard_index, sample_index = self._resolve_index(index)
        shard = self.shards[shard_index]
        values = self._load_shard_values(shard_index)
        cube = np.asarray(values[sample_index], dtype=np.float32)
        cube = _crop_cube(cube, self.crop_size, self.random_crop)
        cube = _normalize_values(cube, self.normalization)
        height, width = cube.shape[-2:]

        return ModalitySample(
            values=torch.from_numpy(cube),
            wavelengths=shard.wavelengths,
            fwhm=shard.fwhm,
            valid_mask=None,
            metadata=SampleMetadata(
                modality="space",
                sensor=shard.sensor,
                level=shard.level,
                gsd=shard.gsd_m,
                latitude=None,
                longitude=None,
                timestamp=None,
                footprint_area=float(height * width) * (shard.gsd_m**2),
            ),
        )


class Sigma450KShardDistributedSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Sigma450KShardDataset,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}.")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        if drop_last:
            self.num_samples = len(self.dataset) // self.num_replicas
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        shard_indices = list(range(len(self.dataset.shards)))
        if self.shuffle:
            shard_order = torch.randperm(len(shard_indices), generator=generator).tolist()
            shard_indices = [shard_indices[index] for index in shard_order]

        global_indices: list[int] = []
        for shard_index in shard_indices:
            start = self.dataset.sample_offsets[shard_index]
            end = self.dataset.sample_offsets[shard_index + 1]
            global_indices.extend(range(start, end))

        if self.drop_last:
            global_indices = global_indices[: self.total_size]
        else:
            if len(global_indices) < self.total_size:
                padding = self.total_size - len(global_indices)
                global_indices.extend(global_indices[:padding])

        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(global_indices[start:end])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class SensorTemperatureDistributedSampler(Sampler[int]):
    """Distributed sampler with p(sensor) proportional to n(sensor)**temperature.

    It is intentionally opt-in. The legacy shard sampler remains unchanged so
    old EO1/GF5 experiments retain their exact ordering and behavior.
    """

    def __init__(
        self,
        dataset: Sigma450KShardDataset,
        *,
        num_replicas: int,
        rank: int,
        temperature: float = 0.5,
        epoch_size: int | None = None,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}.")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}.")
        if temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {temperature}.")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        requested = len(dataset) if epoch_size is None else int(epoch_size)
        if requested <= 0:
            raise ValueError(f"epoch_size must be positive, got {requested}.")
        if drop_last:
            self.num_samples = requested // self.num_replicas
        else:
            self.num_samples = math.ceil(requested / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        buckets: dict[str, list[int]] = {}
        for shard_index, shard in enumerate(dataset.shards):
            start = dataset.sample_offsets[shard_index]
            end = dataset.sample_offsets[shard_index + 1]
            buckets.setdefault(shard.sensor, []).extend(range(start, end))
        if not buckets:
            raise ValueError("SensorTemperatureDistributedSampler found no sensor buckets.")
        self.sensor_indices = {
            sensor: torch.tensor(indices, dtype=torch.long)
            for sensor, indices in sorted(buckets.items())
        }

    def _quotas(self) -> dict[str, int]:
        weights = {
            sensor: float(indices.numel()) ** self.temperature
            for sensor, indices in self.sensor_indices.items()
        }
        total_weight = sum(weights.values())
        raw = {sensor: self.total_size * weight / total_weight for sensor, weight in weights.items()}
        quotas = {sensor: int(math.floor(value)) for sensor, value in raw.items()}
        remaining = self.total_size - sum(quotas.values())
        order = sorted(raw, key=lambda sensor: (raw[sensor] - quotas[sensor], sensor), reverse=True)
        for sensor in order[:remaining]:
            quotas[sensor] += 1
        return quotas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        selected: list[torch.Tensor] = []
        for sensor, quota in self._quotas().items():
            bucket = self.sensor_indices[sensor]
            if quota <= bucket.numel():
                positions = torch.randperm(bucket.numel(), generator=generator)[:quota]
            else:
                positions = torch.randint(0, bucket.numel(), (quota,), generator=generator)
            selected.append(bucket[positions])
        global_indices = torch.cat(selected, dim=0)
        shuffle = torch.randperm(global_indices.numel(), generator=generator)
        global_indices = global_indices[shuffle]
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(global_indices[start:end].tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class Sigma450KDataset(Dataset[ModalitySample]):
    def __init__(
        self,
        root: str | Path,
        *,
        crop_size: int | None = None,
        random_crop: bool = False,
        normalization: str = "mean_std",
        index_cache_path: str | Path | None = None,
    ) -> None:
        if loadmat is None:
            raise ImportError("scipy is required to load sigma450k .mat files.")

        self.root = Path(root)
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.normalization = normalization
        self.records = _load_or_build_jsonl_index(
            None if index_cache_path is None else Path(index_cache_path),
            self._build_index_records,
        )

    def _build_index_records(self) -> list[dict[str, Any]]:
        records = [{"mat_path": str(path)} for path in sorted(self.root.rglob("*.mat"))]
        if not records:
            raise ValueError(f"No sigma450k .mat samples were found under {self.root}.")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ModalitySample:
        mat_path = Path(self.records[index]["mat_path"])
        content = loadmat(mat_path)
        if "img" not in content:
            raise ValueError(f"{mat_path} does not contain an 'img' array.")

        cube = np.asarray(content["img"], dtype=np.float32)
        profile = resolve_sigma450k_profile_from_path(mat_path, int(cube.shape[0]))
        cube = _crop_cube(cube, self.crop_size, self.random_crop)
        cube = _normalize_values(cube, self.normalization)
        height, width = cube.shape[-2:]

        wavelengths = torch.from_numpy(profile.wavelengths_nm.astype(np.float32))
        fwhm = None if profile.fwhm_nm is None else torch.from_numpy(profile.fwhm_nm.astype(np.float32))

        return ModalitySample(
            values=torch.from_numpy(cube),
            wavelengths=wavelengths,
            fwhm=fwhm,
            valid_mask=None,
            metadata=SampleMetadata(
                modality="space",
                sensor=profile.sensor,
                level=profile.level,
                gsd=float(profile.gsd_m),
                latitude=None,
                longitude=None,
                timestamp=None,
                footprint_area=float(height * width) * (float(profile.gsd_m) ** 2),
            ),
        )
