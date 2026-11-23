from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train-only OOF nonlinear calibration predictions for Aware specialist branches."
    )
    parser.add_argument("--dataset", choices=("karly", "gloria"), required=True)
    parser.add_argument("--dataset-npz", type=Path, required=True)
    parser.add_argument("--aware-features-npz", type=Path, required=True)
    parser.add_argument("--domain-features-npz", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denominator


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_true - y_pred
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": r2_score(y_true, y_pred),
    }


def target_forward(values: np.ndarray, dataset: str) -> np.ndarray:
    if dataset == "gloria":
        return np.log1p(np.maximum(values, 0.0))
    return values


def target_inverse(values: np.ndarray, dataset: str) -> np.ndarray:
    if dataset == "gloria":
        return np.maximum(np.expm1(values), 0.0)
    return values


def build_design(
    tokens: np.ndarray,
    domain_features: np.ndarray,
    spectral_view: np.ndarray,
) -> np.ndarray:
    token_mean = tokens.astype(np.float32).mean(axis=1)
    token_last = tokens[:, -1].astype(np.float32)
    token_std = tokens.astype(np.float32).std(axis=1)
    return np.concatenate(
        (token_mean, token_last, token_std, domain_features, spectral_view),
        axis=1,
    ).astype(np.float32)


def model_factories(seed: int) -> dict[str, Any]:
    def reduced(regressor: Any, components: int = 64) -> Pipeline:
        return Pipeline(
            (
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=components, random_state=seed)),
                ("regressor", regressor),
            )
        )

    return {
        "ridge_1": lambda dim: reduced(Ridge(alpha=1.0), min(64, dim)),
        "ridge_20": lambda dim: reduced(Ridge(alpha=20.0), min(64, dim)),
        "svr_10": lambda dim: reduced(SVR(C=10.0, epsilon=0.03, gamma="scale"), min(64, dim)),
        "svr_50": lambda dim: reduced(SVR(C=50.0, epsilon=0.02, gamma="scale"), min(64, dim)),
        "knn_3": lambda dim: reduced(KNeighborsRegressor(n_neighbors=3, weights="distance", p=2), min(48, dim)),
        "knn_7": lambda dim: reduced(KNeighborsRegressor(n_neighbors=7, weights="distance", p=2), min(48, dim)),
        "extra_leaf1": lambda dim: reduced(
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=1,
                max_features=0.8,
                random_state=seed,
                n_jobs=-1,
            ),
            min(64, dim),
        ),
        "extra_leaf3": lambda dim: reduced(
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                max_features=0.8,
                random_state=seed,
                n_jobs=-1,
            ),
            min(64, dim),
        ),
    }


def main() -> int:
    args = parse_args()
    if args.output_npz.exists() and not args.force:
        print(f"[skip] exists: {args.output_npz}", flush=True)
        return 0
    with np.load(args.dataset_npz, allow_pickle=True) as payload:
        targets = np.asarray(payload["targets"], dtype=np.float64)
        train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
        val_idx = np.asarray(payload["val_idx"], dtype=np.int64)
        test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
        dataset_name = str(np.asarray(payload["dataset"]).item())
    with np.load(args.aware_features_npz, allow_pickle=True) as payload:
        tokens = np.asarray(payload["multiview_tokens"], dtype=np.float16)
    with np.load(args.domain_features_npz, allow_pickle=True) as payload:
        domain_features = np.asarray(payload["domain_features"], dtype=np.float32)
        spectral_view = np.asarray(payload["spectral_view"], dtype=np.float32)
    design = build_design(tokens, domain_features, spectral_view)
    y_train = target_forward(targets[train_idx], args.dataset)
    folds = KFold(n_splits=int(args.folds), shuffle=True, random_state=int(args.seed))
    factories = model_factories(int(args.seed))
    names = list(factories)
    train_stack = np.empty((len(train_idx), len(names)), dtype=np.float32)
    val_stack = np.empty((len(val_idx), len(names)), dtype=np.float32)
    test_stack = np.empty((len(test_idx), len(names)), dtype=np.float32)
    validation_summary: dict[str, Any] = {}
    saved_models: dict[str, Any] = {}

    for model_column, name in enumerate(names):
        oof = np.empty(len(train_idx), dtype=np.float64)
        for fold_number, (fold_train, fold_holdout) in enumerate(folds.split(train_idx), start=1):
            model = factories[name](min(design.shape[1], len(fold_train) - 1))
            model.fit(design[train_idx[fold_train]], y_train[fold_train])
            oof[fold_holdout] = np.asarray(model.predict(design[train_idx[fold_holdout]])).reshape(-1)
            print(f"[{name}] fold {fold_number}/{args.folds}", flush=True)
        final_model = factories[name](min(design.shape[1], len(train_idx) - 1))
        final_model.fit(design[train_idx], y_train)
        train_prediction = target_inverse(oof, args.dataset)
        val_prediction = target_inverse(
            np.asarray(final_model.predict(design[val_idx])).reshape(-1),
            args.dataset,
        )
        test_prediction = target_inverse(
            np.asarray(final_model.predict(design[test_idx])).reshape(-1),
            args.dataset,
        )
        train_stack[:, model_column] = train_prediction.astype(np.float32)
        val_stack[:, model_column] = val_prediction.astype(np.float32)
        test_stack[:, model_column] = test_prediction.astype(np.float32)
        validation_summary[name] = metrics(targets[val_idx], val_prediction)
        saved_models[name] = final_model
        print(
            f"[{name}] val_rmse={validation_summary[name]['rmse']:.8g}",
            flush=True,
        )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        dataset=np.array(dataset_name),
        model_names=np.asarray(names),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_stack=train_stack,
        val_stack=val_stack,
        test_stack=test_stack,
    )
    args.output_npz.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "models": validation_summary,
                "best_base_model": min(
                    validation_summary,
                    key=lambda name: validation_summary[name]["rmse"],
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    joblib.dump(saved_models, args.output_npz.with_suffix(".joblib"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
