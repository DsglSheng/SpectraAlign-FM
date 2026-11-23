from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "dataset_registry.json"


def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table file: {path}")


def extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        marker.write_text(zip_path.name + "\n", encoding="utf-8")
    return [p for p in out_dir.rglob("*") if p.suffix.lower() in {".csv", ".txt", ".tsv", ".xlsx", ".xls"}]


def normalize_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def maybe_wavelength(name: object) -> float | None:
    text = str(name).strip().lower()
    patterns = [
        r"(?:^|[^0-9])(\d{3,4}(?:\.\d+)?)\s*nm(?:[^0-9]|$)",
        r"(?:rrs|reflectance|refl|band|wl|wvl|wave)[_\s-]*(\d{3,4}(?:\.\d+)?)",
        r"^(\d{3,4}(?:\.\d+)?)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            if 300.0 <= value <= 2600.0:
                return value
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infer_target_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized_aliases = [normalize_name(a) for a in aliases]
    scored: list[tuple[int, str]] = []
    for col in df.columns:
        ncol = normalize_name(col)
        score = 0
        for alias in normalized_aliases:
            if ncol == alias:
                score = max(score, 100)
            elif alias and alias in ncol:
                score = max(score, 50 + len(alias))
        if score > 0:
            numeric = coerce_numeric(df[col])
            if numeric.notna().sum() > 10:
                scored.append((score, col))
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda item: item[0])
    return scored[0][1]


def infer_spectral_columns(df: pd.DataFrame, wl_min: float, wl_max: float) -> tuple[list[str], np.ndarray]:
    candidates: list[tuple[float, str]] = []
    for col in df.columns:
        wave = maybe_wavelength(col)
        if wave is None or not (wl_min <= wave <= wl_max):
            continue
        values = coerce_numeric(df[col])
        if values.notna().sum() >= max(10, int(0.1 * len(df))):
            candidates.append((wave, col))
    candidates.sort(key=lambda item: item[0])
    columns = [col for _, col in candidates]
    wavelengths = np.asarray([wave for wave, _ in candidates], dtype=np.float32)
    return columns, wavelengths


def choose_join_column(left: pd.DataFrame, right: pd.DataFrame) -> str | None:
    common = [c for c in left.columns if c in set(right.columns)]
    best: tuple[int, str] | None = None
    for col in common:
        lvals = set(left[col].dropna().astype(str))
        rvals = set(right[col].dropna().astype(str))
        overlap = len(lvals & rvals)
        if overlap >= 10 and (best is None or overlap > best[0]):
            best = (overlap, col)
    return None if best is None else best[1]


def split_indices(n: int, *, test_fraction: float, val_fraction_from_trainval: float, seed: int) -> dict[str, np.ndarray]:
    if n < 5:
        raise ValueError(f"Need at least 5 samples, got {n}.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = max(1, min(n - 2, int(round(n * test_fraction))))
    trainval = order[:-n_test]
    test = order[-n_test:]
    n_val = max(1, min(len(trainval) - 1, int(round(len(trainval) * val_fraction_from_trainval))))
    train = trainval[:-n_val]
    val = trainval[-n_val:]
    return {
        "train_idx": np.sort(train).astype(np.int64),
        "val_idx": np.sort(val).astype(np.int64),
        "test_idx": np.sort(test).astype(np.int64),
    }


def save_npz(
    out_path: Path,
    *,
    dataset: str,
    spectra: np.ndarray,
    targets: np.ndarray,
    wavelengths_nm: np.ndarray,
    splits: dict[str, np.ndarray],
    target_name: str,
    source_files: list[Path],
    metadata: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        spectra=spectra.astype(np.float32),
        targets=targets.astype(np.float32),
        wavelengths_nm=wavelengths_nm.astype(np.float32),
        dataset=np.asarray(dataset),
        target_name=np.asarray(target_name),
        source_files=np.asarray([str(p) for p in source_files]),
        metadata_json=np.asarray(json.dumps(metadata, indent=2, sort_keys=True)),
        **splits,
    )
    summary = {
        "dataset": dataset,
        "target": target_name,
        "samples": int(spectra.shape[0]),
        "bands": int(spectra.shape[1]),
        "wavelength_min_nm": float(np.nanmin(wavelengths_nm)),
        "wavelength_max_nm": float(np.nanmax(wavelengths_nm)),
        "target_min": float(np.nanmin(targets)),
        "target_max": float(np.nanmax(targets)),
        "splits": {k: int(v.size) for k, v in splits.items()},
        "source_files": [str(p) for p in source_files],
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_from_single_table(dataset_key: str, cfg: dict[str, Any], table_path: Path) -> None:
    df = read_table(table_path)
    target_col = infer_target_column(df, cfg["target_aliases"])
    if target_col is None:
        raise RuntimeError(f"Could not infer target column for {dataset_key} in {table_path}")
    wl_min, wl_max = [float(v) for v in cfg["wavelength_range_nm"]]
    spectral_cols, wavelengths = infer_spectral_columns(df, wl_min, wl_max)
    if len(spectral_cols) < 20:
        raise RuntimeError(f"Only found {len(spectral_cols)} spectral columns in {table_path}; inspect column names manually.")

    target = coerce_numeric(df[target_col]).to_numpy(dtype=np.float32)
    spectra = df[spectral_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(target) & np.isfinite(spectra).all(axis=1)
    if "target_filter" in cfg:
        f = cfg["target_filter"]
        valid &= target >= float(f["min"])
        valid &= target <= float(f["max"])
    spectra = spectra[valid]
    target = target[valid]
    if spectra.shape[0] < 20:
        raise RuntimeError(f"After filtering, only {spectra.shape[0]} samples remain for {dataset_key}.")
    split_cfg = cfg["default_split"]
    splits = split_indices(
        spectra.shape[0],
        test_fraction=float(split_cfg["test_fraction"]),
        val_fraction_from_trainval=float(split_cfg["val_fraction_from_trainval"]),
        seed=int(split_cfg["seed"]),
    )
    save_npz(
        ROOT / cfg["processed_npz"],
        dataset=dataset_key,
        spectra=spectra,
        targets=target,
        wavelengths_nm=wavelengths,
        splits=splits,
        target_name=str(target_col),
        source_files=[table_path],
        metadata=cfg,
    )


def prepare_karly(registry: dict[str, Any]) -> None:
    cfg = registry["datasets"]["karly_soil_moisture"]
    table_path = ROOT / cfg["raw_file"]
    if not table_path.exists():
        raise FileNotFoundError(f"Missing {table_path}; run scripts/01_fetch_public_data.sh first.")
    build_from_single_table("karly_soil_moisture", cfg, table_path)


def prepare_gloria(registry: dict[str, Any]) -> None:
    cfg = registry["datasets"]["gloria_chlorophyll"]
    zip_path = ROOT / cfg["raw_file"]
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing {zip_path}; run scripts/01_fetch_public_data.sh first.")
    extracted = ROOT / "data" / "raw" / "GLORIA-2022"
    tables = extract_zip(zip_path, extracted)
    if not tables:
        raise RuntimeError(f"No tabular files found after extracting {zip_path}.")

    loaded: list[tuple[Path, pd.DataFrame]] = []
    for table in tables:
        try:
            df = read_table(table)
        except Exception as exc:
            print(f"[warn] skip {table}: {exc}")
            continue
        loaded.append((table, df))

    wl_min, wl_max = [float(v) for v in cfg["wavelength_range_nm"]]
    lab_candidates: list[tuple[Path, pd.DataFrame, str]] = []
    spec_candidates: list[tuple[Path, pd.DataFrame, list[str], np.ndarray]] = []
    for path, df in loaded:
        target_col = infer_target_column(df, cfg["target_aliases"])
        if target_col is not None:
            lab_candidates.append((path, df, target_col))
        spectral_cols, wavelengths = infer_spectral_columns(df, wl_min, wl_max)
        if len(spectral_cols) >= 50:
            spec_candidates.append((path, df, spectral_cols, wavelengths))

    if not spec_candidates:
        raise RuntimeError("Could not find a GLORIA reflectance table with wavelength columns.")

    # Prefer a single table containing both spectra and target.
    for path, df, spectral_cols, wavelengths in spec_candidates:
        target_col = infer_target_column(df, cfg["target_aliases"])
        if target_col is not None:
            return build_from_single_table("gloria_chlorophyll", cfg, path)

    if not lab_candidates:
        raise RuntimeError("Could not find a GLORIA metadata/lab table with chlorophyll target.")

    lab_path, lab_df, target_col = lab_candidates[0]
    spec_path, spec_df, spectral_cols, wavelengths = max(spec_candidates, key=lambda item: len(item[2]))
    join_col = choose_join_column(lab_df, spec_df)
    if join_col is None:
        if len(lab_df) == len(spec_df):
            merged = pd.concat([lab_df.reset_index(drop=True), spec_df[spectral_cols].reset_index(drop=True)], axis=1)
        else:
            raise RuntimeError("Could not infer a GLORIA join column between lab and reflectance tables.")
    else:
        merged = lab_df.merge(spec_df[[join_col] + spectral_cols], on=join_col, how="inner")

    target = coerce_numeric(merged[target_col]).to_numpy(dtype=np.float32)
    spectra = merged[spectral_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(target) & np.isfinite(spectra).all(axis=1)
    f = cfg["target_filter"]
    valid &= target >= float(f["min"])
    valid &= target <= float(f["max"])
    spectra = spectra[valid]
    target = target[valid]
    split_cfg = cfg["default_split"]
    splits = split_indices(
        spectra.shape[0],
        test_fraction=float(split_cfg["test_fraction"]),
        val_fraction_from_trainval=float(split_cfg["val_fraction_from_trainval"]),
        seed=int(split_cfg["seed"]),
    )
    save_npz(
        ROOT / cfg["processed_npz"],
        dataset="gloria_chlorophyll",
        spectra=spectra,
        targets=target,
        wavelengths_nm=wavelengths,
        splits=splits,
        target_name=str(target_col),
        source_files=[lab_path, spec_path],
        metadata=cfg,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare public HySens-style pixel regression datasets.")
    parser.add_argument("--dataset", choices=("gloria_chlorophyll", "karly_soil_moisture"), default=None)
    parser.add_argument("--all", action="store_true", help="Prepare all public datasets.")
    args = parser.parse_args()
    if not args.all and args.dataset is None:
        raise SystemExit("Use --all or --dataset.")
    registry = load_registry()
    todo = ["gloria_chlorophyll", "karly_soil_moisture"] if args.all else [args.dataset]
    for name in todo:
        if name == "gloria_chlorophyll":
            prepare_gloria(registry)
        elif name == "karly_soil_moisture":
            prepare_karly(registry)
        else:
            raise AssertionError(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
