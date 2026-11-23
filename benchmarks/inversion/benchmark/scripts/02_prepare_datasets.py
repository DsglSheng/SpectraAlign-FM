from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
COMMON_WAVELENGTHS = np.linspace(400.0, 2450.0, 256, dtype=np.float32)


def spectral_columns(columns: list[str]) -> tuple[list[str], np.ndarray]:
    found: list[tuple[float, str]] = []
    for column in columns:
        text = str(column).strip()
        if not re.fullmatch(r"\d+(?:\.\d+)?", text):
            continue
        wavelength = float(text)
        if 350.0 <= wavelength <= 2600.0:
            found.append((wavelength, column))
    found.sort()
    return [column for _, column in found], np.asarray([wave for wave, _ in found], dtype=np.float32)


def resample_spectra(spectra: np.ndarray, source_wavelengths: np.ndarray) -> np.ndarray:
    output = np.empty((spectra.shape[0], COMMON_WAVELENGTHS.size), dtype=np.float32)
    for index, spectrum in enumerate(spectra):
        valid = np.isfinite(spectrum) & np.isfinite(source_wavelengths)
        if int(valid.sum()) < 20:
            output[index] = np.nan
            continue
        output[index] = np.interp(
            COMMON_WAVELENGTHS,
            source_wavelengths[valid],
            spectrum[valid],
        ).astype(np.float32)
    return output


def stratified_validation_indices(labels: np.ndarray, *, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        indices = rng.permutation(indices)
        if indices.size >= 5:
            val_count = max(1, int(round(indices.size * fraction)))
        else:
            val_count = 0
        val_parts.append(indices[:val_count])
        train_parts.append(indices[val_count:])
    train = np.sort(np.concatenate(train_parts)).astype(np.int64)
    val = np.sort(np.concatenate(val_parts)).astype(np.int64)
    return train, val


def save_dataset(
    output: Path,
    *,
    dataset: str,
    target_name: str,
    spectra: np.ndarray,
    targets: np.ndarray,
    wavelengths_nm: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        dataset=np.asarray(dataset),
        target_name=np.asarray(target_name),
        spectra=spectra.astype(np.float32),
        targets=targets.astype(np.float32),
        wavelengths_nm=wavelengths_nm.astype(np.float32),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        metadata_json=np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
    )
    summary = {
        "dataset": dataset,
        "target": target_name,
        "samples": int(targets.size),
        "bands": int(spectra.shape[1]),
        "wavelength_range_nm": [float(wavelengths_nm.min()), float(wavelengths_nm.max())],
        "target_range": [float(targets.min()), float(targets.max())],
        "splits": {
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
        },
        "metadata": metadata,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def prepare_green() -> None:
    train_path = RAW / "greenhyperspectra_train.csv"
    test_path = RAW / "greenhyperspectra_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("GreenHyperSpectra train/test CSV files are missing.")

    header = pd.read_csv(train_path, nrows=0)
    spec_cols, source_wavelengths = spectral_columns(list(header.columns))
    usecols = spec_cols + ["dataset", "cp"]
    train_frame = pd.read_csv(train_path, usecols=usecols)
    test_frame = pd.read_csv(test_path, usecols=usecols)

    def convert(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        targets = pd.to_numeric(frame["cp"], errors="coerce").to_numpy(dtype=np.float32)
        source_ids = frame["dataset"].astype(str).to_numpy()
        raw_spectra = frame[spec_cols].to_numpy(dtype=np.float32)
        valid = np.isfinite(targets) & (np.isfinite(raw_spectra).sum(axis=1) >= 20)
        resampled = resample_spectra(raw_spectra[valid], source_wavelengths)
        finite = np.isfinite(resampled).all(axis=1)
        return resampled[finite], targets[valid][finite], source_ids[valid][finite]

    train_spectra, train_targets, train_sources = convert(train_frame)
    test_spectra, test_targets, test_sources = convert(test_frame)
    local_train, local_val = stratified_validation_indices(train_sources, fraction=0.2, seed=20260624)

    spectra = np.concatenate([train_spectra, test_spectra], axis=0)
    targets = np.concatenate([train_targets, test_targets], axis=0)
    test_offset = int(train_targets.size)
    test_idx = np.arange(test_offset, targets.size, dtype=np.int64)
    save_dataset(
        PROCESSED / "greenhyperspectra_leaf_nitrogen.npz",
        dataset="greenhyperspectra_leaf_nitrogen",
        target_name="leaf nitrogen content Cp (g/cm2)",
        spectra=spectra,
        targets=targets,
        wavelengths_nm=COMMON_WAVELENGTHS,
        train_idx=local_train,
        val_idx=local_val,
        test_idx=test_idx,
        metadata={
            "scene": "global multi-source vegetation spectroscopy",
            "source": "Avatarr05/GreenHyperSpectra",
            "paper": "GreenHyperSpectra (Cherif et al., 2025)",
            "license": "CC-BY-NC-4.0",
            "source_split": "published labeled_splits train/test",
            "validation": "20% within published train, stratified by source dataset ID",
            "sensor": "field",
            "level": "L2",
            "gsd": 1.0,
            "source_wavelength_count": int(source_wavelengths.size),
            "common_wavelength_count": int(COMMON_WAVELENGTHS.size),
            "test_source_count": int(np.unique(test_sources).size),
        },
    )


def grouped_split(groups: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if unique_groups.size < 5:
        rng = np.random.default_rng(seed)
        order = rng.permutation(groups.size)
        test_count = max(1, int(round(groups.size * 0.2)))
        val_count = max(1, int(round(groups.size * 0.2)))
        return (
            np.sort(order[test_count + val_count :]).astype(np.int64),
            np.sort(order[test_count : test_count + val_count]).astype(np.int64),
            np.sort(order[:test_count]).astype(np.int64),
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    test_group_count = max(1, int(round(unique_groups.size * 0.2)))
    val_group_count = max(1, int(round(unique_groups.size * 0.2)))
    test_groups = set(shuffled[:test_group_count].tolist())
    val_groups = set(shuffled[test_group_count : test_group_count + val_group_count].tolist())
    test_idx = np.flatnonzero(np.asarray([group in test_groups for group in groups]))
    val_idx = np.flatnonzero(np.asarray([group in val_groups for group in groups]))
    train_idx = np.flatnonzero(
        np.asarray([group not in test_groups and group not in val_groups for group in groups])
    )
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def prepare_lithium_kaolinite() -> None:
    powder = RAW / "lithium_powder" / "1-Powder"
    spectrum_path = powder / "Pixel average spectrum[5x5].csv"
    mineral_archive = powder / "2-Mineralogical_information.zip"
    if not spectrum_path.exists() or not mineral_archive.exists():
        raise FileNotFoundError("Lithium powder spectra or mineralogical archive is missing.")

    spectrum_frame = pd.read_csv(spectrum_path)
    id_column = spectrum_frame.columns[0]
    spec_cols, source_wavelengths = spectral_columns(list(spectrum_frame.columns))
    spectrum_frame = spectrum_frame.rename(columns={id_column: "Sample_ID"})

    def normalize_sample_id(value: object) -> str:
        text = str(value).strip().upper()
        text = re.sub(r"\[5X5AVG\]$", "", text)
        match = re.fullmatch(r"(.+S)(\d+)", text)
        if match:
            return f"{match.group(1)}{int(match.group(2)):02d}"
        return text

    spectrum_frame["Sample_ID"] = spectrum_frame["Sample_ID"].map(normalize_sample_id)

    member = "2-Mineralogical_information/QXRD_results_retained_7_minerals.csv"
    with zipfile.ZipFile(mineral_archive) as handle, handle.open(member) as stream:
        minerals = pd.read_csv(stream)
    minerals["Sample_ID"] = minerals["Sample_ID"].map(normalize_sample_id)
    merged = spectrum_frame.merge(minerals[["Sample_ID", "Kaolinite"]], on="Sample_ID", how="inner")
    if len(merged) < 50:
        raise RuntimeError(
            f"Only {len(merged)} spectra matched QXRD samples; "
            f"spectrum examples={spectrum_frame['Sample_ID'].head(8).tolist()}, "
            f"QXRD examples={minerals['Sample_ID'].head(8).tolist()}."
        )

    raw_spectra = merged[spec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    targets = pd.to_numeric(merged["Kaolinite"], errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(targets) & (np.isfinite(raw_spectra).sum(axis=1) >= 20)
    raw_spectra = raw_spectra[valid]
    targets = targets[valid]
    sample_ids = merged.loc[valid, "Sample_ID"].astype(str).to_numpy()

    target_wavelengths = np.linspace(
        float(source_wavelengths.min()),
        min(2450.0, float(source_wavelengths.max())),
        256,
        dtype=np.float32,
    )
    spectra = np.empty((raw_spectra.shape[0], target_wavelengths.size), dtype=np.float32)
    for index, spectrum in enumerate(raw_spectra):
        finite = np.isfinite(spectrum)
        spectra[index] = np.interp(
            target_wavelengths,
            source_wavelengths[finite],
            spectrum[finite],
        ).astype(np.float32)

    groups = np.asarray([re.sub(r"S\d+$", "", sample_id, flags=re.IGNORECASE) for sample_id in sample_ids])
    train_idx, val_idx, test_idx = grouped_split(groups, seed=20260624)
    save_dataset(
        PROCESSED / "clay_lithium_kaolinite_abundance.npz",
        dataset="clay_lithium_kaolinite_abundance",
        target_name="QXRD kaolinite abundance (%)",
        spectra=spectra,
        targets=targets,
        wavelengths_nm=target_wavelengths,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        metadata={
            "scene": "clay-type lithium deposit powder spectroscopy",
            "source": "Zenodo 19142250",
            "doi": "10.5281/zenodo.19142250",
            "license": "CC-BY-4.0",
            "measurement": "5x5 center-pixel mean SWIR reflectance joined to QXRD",
            "split": "60/20/20 by sample group prefix, no group overlap",
            "sensor": "field",
            "level": "L2",
            "gsd": 1.0,
            "source_wavelength_count": int(source_wavelengths.size),
            "common_wavelength_count": int(target_wavelengths.size),
            "group_count": int(np.unique(groups).size),
            "sample_id_examples": sample_ids[:5].tolist(),
        },
    )


def find_powder_tables(extracted: Path) -> list[Path]:
    return sorted(
        path
        for path in extracted.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls", ".txt"}
    )


def inspect_lithium() -> None:
    archive = RAW / "lithium_powder.zip"
    if not archive.exists():
        raise FileNotFoundError(archive)
    extracted = RAW / "lithium_powder"
    if not extracted.exists():
        extracted.mkdir(parents=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extracted)
    rows = []
    for path in find_powder_tables(extracted):
        row: dict[str, Any] = {"path": str(path.relative_to(extracted)), "bytes": path.stat().st_size}
        try:
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path, nrows=5)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                frame = pd.read_excel(path, nrows=5)
            else:
                frame = pd.read_csv(path, sep=None, engine="python", nrows=5)
            row["columns"] = [str(column) for column in frame.columns]
            row["rows_previewed"] = int(len(frame))
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)

    powder = extracted / "1-Powder"
    spectrum_path = powder / "Pixel average spectrum[5x5].csv"
    spectrum_frame = pd.read_csv(spectrum_path, nrows=5)
    rows.append(
        {
            "path": str(spectrum_path.relative_to(extracted)),
            "shape_preview": list(spectrum_frame.shape),
            "columns": [str(column) for column in spectrum_frame.columns],
        }
    )
    for archive_name in ("2-Mineralogical_information.zip", "3-Geochemical_information.zip"):
        archive_path = powder / archive_name
        with zipfile.ZipFile(archive_path) as handle:
            for member in handle.namelist():
                if not member.lower().endswith((".csv", ".xlsx", ".xls")):
                    continue
                item: dict[str, Any] = {"archive": archive_name, "member": member}
                try:
                    with handle.open(member) as stream:
                        if member.lower().endswith(".csv"):
                            frame = pd.read_csv(stream, nrows=5)
                        else:
                            frame = pd.read_excel(stream, nrows=5)
                    item["shape_preview"] = list(frame.shape)
                    item["columns"] = [str(column) for column in frame.columns]
                except Exception as error:
                    item["error"] = str(error)
                rows.append(item)
    output = ROOT / "docs" / "lithium_powder_table_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare public inversion benchmark datasets.")
    parser.add_argument("--dataset", choices=("green", "lithium-inspect", "lithium-kaolinite"), required=True)
    args = parser.parse_args()
    if args.dataset == "green":
        prepare_green()
    elif args.dataset == "lithium-inspect":
        inspect_lithium()
    else:
        prepare_lithium_kaolinite()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
