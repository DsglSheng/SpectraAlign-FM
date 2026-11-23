from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model_adapters import AwareBackbone


DEFAULTS = {
    "aware_repo": Path("/mnt/Data/SpectralData/Aware_open"),
    "aware_checkpoint": Path(
        "/mnt/Data/SpectralData/Aware_open/runs/pretrain_ours_v3wa_kl003_full450k_20260505/"
        "branches/ours_v3wa_kl003_full450k/checkpoints/best_total.pt"
    ),
    "prototype_bank_eo1": Path(
        "/mnt/Data/SpectralData/Aware_open/artifacts/prototypes_v3_core_20260503/prototype_bank_eo1.pt"
    ),
    "prototype_bank_gf5": Path(
        "/mnt/Data/SpectralData/Aware_open/artifacts/prototypes_v3_core_20260503/prototype_bank_gf5.pt"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen Aware-HSI plus validation-selected chemometric inversion branch."
    )
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--pca-dims", type=int, nargs="+", default=(32, 64, 128))
    parser.add_argument("--pls-components", type=int, nargs="+", default=(4, 8, 12, 16, 24, 32))
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=(0.1, 1.0, 10.0, 100.0))
    parser.add_argument("--smoke", action="store_true")
    for name, value in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=value)
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    difference = y_true - y_pred
    return {
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(np.mean(np.abs(difference))),
        "r2": r2_score(y_true, y_pred),
    }


def fwhm_from_wavelengths(wavelengths_nm: np.ndarray) -> np.ndarray:
    if wavelengths_nm.size == 1:
        return np.ones_like(wavelengths_nm)
    differences = np.abs(np.diff(wavelengths_nm))
    fwhm = np.empty_like(wavelengths_nm)
    fwhm[0] = differences[0]
    fwhm[-1] = differences[-1]
    if wavelengths_nm.size > 2:
        fwhm[1:-1] = 0.5 * (differences[:-1] + differences[1:])
    return np.maximum(fwhm, 1.0e-3)


def normalize_bands(spectra: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = spectra[train_idx].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = spectra[train_idx].std(axis=0, dtype=np.float64).astype(np.float32)
    return ((spectra - mean) / np.maximum(std, 1.0e-6)).astype(np.float32)


def sample_normal_variate(spectra: np.ndarray) -> np.ndarray:
    centered = spectra - spectra.mean(axis=1, keepdims=True)
    scale = spectra.std(axis=1, keepdims=True)
    return (centered / np.maximum(scale, 1.0e-6)).astype(np.float32)


def derivative(spectra: np.ndarray, order: int) -> np.ndarray:
    bands = spectra.shape[1]
    window = min(21, bands if bands % 2 else bands - 1)
    window = max(5, window)
    return savgol_filter(
        spectra,
        window_length=window,
        polyorder=3,
        deriv=order,
        axis=1,
        mode="interp",
    ).astype(np.float32)


def spectral_feature_sets(spectra: np.ndarray, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    band_z = normalize_bands(spectra, train_idx)
    snv = sample_normal_variate(spectra)
    first = derivative(snv, 1)
    second = derivative(snv, 2)
    return {
        "bandz": band_z,
        "bandz_snv_d1": np.concatenate((band_z, snv, first), axis=1).astype(np.float32),
        "bandz_snv_d1_d2": np.concatenate((band_z, snv, first, second), axis=1).astype(np.float32),
    }


@torch.no_grad()
def extract_aware_embeddings(
    normalized_spectra: np.ndarray,
    *,
    wavelengths_nm: np.ndarray,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    backbone = AwareBackbone(
        aware_repo=args.aware_repo,
        checkpoint=args.aware_checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        wavelengths_nm=torch.from_numpy(wavelengths_nm),
        sensor=str(metadata.get("sensor", "field")),
        level=str(metadata.get("level", "field")),
        gsd=float(metadata.get("gsd", 1.0)),
        device=device,
    ).to(device)
    backbone.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(normalized_spectra)),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    outputs: list[np.ndarray] = []
    for (batch,) in loader:
        batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and args.precision == "bf16",
        ):
            embedding = backbone(batch)
        outputs.append(embedding.float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def transform_target(values: np.ndarray, name: str) -> np.ndarray:
    if name == "raw":
        return values.astype(np.float64)
    if name == "log1p":
        return np.log1p(np.maximum(values, 0.0)).astype(np.float64)
    raise ValueError(name)


def inverse_target(values: np.ndarray, name: str) -> np.ndarray:
    if name == "raw":
        return values
    if name == "log1p":
        return np.maximum(np.expm1(values), 0.0)
    raise ValueError(name)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "test_metrics.json"
    if final_path.exists() and not args.smoke:
        print(f"[skip] completed: {final_path}", flush=True)
        return 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with np.load(args.dataset_npz, allow_pickle=True) as payload:
        spectra = np.asarray(payload["spectra"], dtype=np.float32)
        targets = np.asarray(payload["targets"], dtype=np.float64)
        wavelengths_nm = np.asarray(payload["wavelengths_nm"], dtype=np.float32)
        train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
        val_idx = np.asarray(payload["val_idx"], dtype=np.int64)
        test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
        dataset_name = str(np.asarray(payload["dataset"]).item())
        target_name = str(np.asarray(payload["target_name"]).item())
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))

    band_z = normalize_bands(spectra, train_idx)
    embedding_cache = args.output_dir / "aware_embeddings.npz"
    if embedding_cache.exists():
        embeddings = np.load(embedding_cache)["embeddings"].astype(np.float32)
    else:
        embeddings = extract_aware_embeddings(
            band_z,
            wavelengths_nm=wavelengths_nm,
            metadata=metadata,
            args=args,
            device=device,
        )
        np.savez_compressed(embedding_cache, embeddings=embeddings)

    feature_sets = spectral_feature_sets(spectra, train_idx)
    pca_dims = [args.pca_dims[0]] if args.smoke else list(args.pca_dims)
    pls_components = [args.pls_components[0]] if args.smoke else list(args.pls_components)
    ridge_alphas = [args.ridge_alphas[0]] if args.smoke else list(args.ridge_alphas)
    feature_names = [next(iter(feature_sets))] if args.smoke else list(feature_sets)
    target_transforms = ["raw"]
    if not args.smoke and targets.min() >= 0.0:
        target_transforms.append("log1p")

    candidates: list[dict[str, Any]] = []
    fitted: list[tuple[dict[str, Any], Any, PCA]] = []
    for pca_dim in pca_dims:
        actual_pca_dim = min(int(pca_dim), len(train_idx) - 1, embeddings.shape[1])
        pca = Pipeline(
            (
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=actual_pca_dim, random_state=args.seed)),
            )
        )
        aware_train = pca.fit_transform(embeddings[train_idx])
        aware_val = pca.transform(embeddings[val_idx])
        for feature_name in feature_names:
            spectral = feature_sets[feature_name]
            x_train = np.concatenate((aware_train, spectral[train_idx]), axis=1)
            x_val = np.concatenate((aware_val, spectral[val_idx]), axis=1)
            for target_transform in target_transforms:
                y_train = transform_target(targets[train_idx], target_transform)
                for component_count in pls_components:
                    actual_components = min(int(component_count), x_train.shape[1], len(train_idx) - 1)
                    model = Pipeline(
                        (
                            ("scale", StandardScaler()),
                            (
                                "regressor",
                                PLSRegression(
                                    n_components=actual_components,
                                    scale=False,
                                    max_iter=1000,
                                    tol=1.0e-7,
                                ),
                            ),
                        )
                    )
                    model.fit(x_train, y_train)
                    val_prediction = inverse_target(
                        np.asarray(model.predict(x_val)).reshape(-1),
                        target_transform,
                    )
                    row = {
                        "family": "pls",
                        "feature_set": feature_name,
                        "pca_dim": actual_pca_dim,
                        "components": actual_components,
                        "target_transform": target_transform,
                        "val": metrics(targets[val_idx], val_prediction),
                    }
                    candidates.append(row)
                    fitted.append((row, model, pca))
                for alpha in ridge_alphas:
                    model = Pipeline(
                        (
                            ("scale", StandardScaler()),
                            ("regressor", Ridge(alpha=float(alpha))),
                        )
                    )
                    model.fit(x_train, y_train)
                    val_prediction = inverse_target(
                        np.asarray(model.predict(x_val)).reshape(-1),
                        target_transform,
                    )
                    row = {
                        "family": "ridge",
                        "feature_set": feature_name,
                        "pca_dim": actual_pca_dim,
                        "alpha": float(alpha),
                        "target_transform": target_transform,
                        "val": metrics(targets[val_idx], val_prediction),
                    }
                    candidates.append(row)
                    fitted.append((row, model, pca))
        print(f"[search] completed aware PCA dim {actual_pca_dim}", flush=True)

    best_index = min(range(len(candidates)), key=lambda index: candidates[index]["val"]["rmse"])
    selected, selected_model, selected_pca = fitted[best_index]
    selected_features = feature_sets[selected["feature_set"]]
    aware_test = selected_pca.transform(embeddings[test_idx])
    x_test = np.concatenate((aware_test, selected_features[test_idx]), axis=1)
    test_prediction = inverse_target(
        np.asarray(selected_model.predict(x_test)).reshape(-1),
        selected["target_transform"],
    )
    test_result = metrics(targets[test_idx], test_prediction)

    candidates_sorted = sorted(candidates, key=lambda row: row["val"]["rmse"])
    save_json(args.output_dir / "validation_search.json", candidates_sorted)
    joblib.dump(
        {
            "aware_pca": selected_pca,
            "regressor": selected_model,
            "selection": selected,
        },
        args.output_dir / "selected_model.joblib",
    )
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        targets=targets[test_idx].astype(np.float32),
        predictions=test_prediction.astype(np.float32),
        test_idx=test_idx,
    )
    result = {
        "model": "aware_chemometric_fusion_v2",
        "dataset": dataset_name,
        "target": target_name,
        "best_val_rmse": selected["val"]["rmse"],
        "selection": selected,
        "test": test_result,
        "protocol": (
            "frozen Aware-HSI embeddings plus train-fitted spectral chemometrics; "
            "feature preprocessing, PCA size, regressor, complexity, and target transform "
            "selected only by validation RMSE; test evaluated once after selection"
        ),
    }
    save_json(final_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
