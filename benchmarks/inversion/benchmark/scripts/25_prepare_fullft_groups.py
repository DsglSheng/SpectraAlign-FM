from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare KarLy session and GLORIA Dataset_ID groups.")
    parser.add_argument("--dataset", choices=("karly", "gloria"), required=True)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def encode_groups(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    normalized = values.fillna("unknown").astype(str)
    categories = sorted(normalized.unique().tolist())
    mapping = {value: index for index, value in enumerate(categories)}
    return normalized.map(mapping).to_numpy(dtype=np.int64), categories


def prepare_karly(args: argparse.Namespace, targets: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    table = pd.read_csv(args.raw_root / "soilmoisture_dataset.csv")
    valid = np.isfinite(pd.to_numeric(table["soil_moisture"], errors="coerce").to_numpy())
    spectral_columns = [column for column in table.columns if str(column).isdigit()]
    spectra = table[spectral_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    valid &= np.isfinite(spectra).all(axis=1)
    filtered = table.loc[valid].reset_index(drop=True)
    raw_targets = filtered["soil_moisture"].to_numpy(dtype=np.float32)
    if raw_targets.shape != targets.shape or not np.allclose(raw_targets, targets, atol=1.0e-5):
        raise ValueError("KarLy raw rows do not align with the processed NPZ.")
    timestamps = pd.to_datetime(filtered["datetime"])
    gap_seconds = timestamps.diff().dt.total_seconds().fillna(0.0)
    session_ids = (gap_seconds > 1800.0).cumsum().to_numpy(dtype=np.int64)
    day_ids, days = encode_groups(timestamps.dt.strftime("%Y-%m-%d"))
    return session_ids, {
        "session_count": int(session_ids.max() + 1),
        "day_count": len(days),
        "days": days,
        "group_definition": "new session after a gap greater than 30 minutes",
        "day_ids": day_ids.tolist(),
    }


def prepare_gloria(args: argparse.Namespace, targets: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    base = args.raw_root / "GLORIA-2022" / "GLORIA_2022"
    meta = pd.read_csv(base / "GLORIA_meta_and_lab.csv")
    rrs = pd.read_csv(base / "GLORIA_Rrs.csv")
    spectral_columns = [f"Rrs_{wavelength}" for wavelength in range(350, 901)]
    merged = meta.merge(rrs[["GLORIA_ID", *spectral_columns]], on="GLORIA_ID", how="inner")
    raw_targets = pd.to_numeric(merged["Chla"], errors="coerce").to_numpy(dtype=np.float64)
    spectra = merged[spectral_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(raw_targets) & np.isfinite(spectra).all(axis=1)
    valid &= raw_targets >= 0.0
    valid &= raw_targets <= 100.0
    filtered = merged.loc[valid].reset_index(drop=True)
    raw_targets = raw_targets[valid].astype(np.float32)
    if raw_targets.shape != targets.shape or not np.allclose(raw_targets, targets, atol=1.0e-5):
        raise ValueError("GLORIA raw rows do not align with the processed NPZ.")
    dataset_ids, dataset_names = encode_groups(filtered["Dataset_ID"])
    organizations, organization_names = encode_groups(filtered["Organization_ID"])
    sites, site_names = encode_groups(filtered["Site_name"])
    return dataset_ids, {
        "dataset_id_count": len(dataset_names),
        "dataset_ids": dataset_names,
        "organization_count": len(organization_names),
        "organization_ids": organizations.tolist(),
        "site_count": len(site_names),
        "site_ids": sites.tolist(),
        "group_definition": "GLORIA Dataset_ID",
    }


def main() -> int:
    args = parse_args()
    if args.output_npz.exists() and not args.force:
        print(f"[skip] exists: {args.output_npz}")
        return 0
    with np.load(args.dataset_npz, allow_pickle=True) as payload:
        targets = np.asarray(payload["targets"], dtype=np.float32)
        dataset_name = str(np.asarray(payload["dataset"]).item())
    if args.dataset == "karly":
        groups, metadata = prepare_karly(args, targets)
    else:
        groups, metadata = prepare_gloria(args, targets)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        dataset=np.array(dataset_name),
        groups=groups,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    summary = {
        "dataset": dataset_name,
        "samples": int(targets.size),
        "groups": int(np.unique(groups).size),
        **metadata,
    }
    args.output_npz.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
