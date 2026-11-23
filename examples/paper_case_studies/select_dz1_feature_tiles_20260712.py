from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select visually interpretable DZ1-VNIR tiles for layer-wise feature-response analysis. "
            "The script is read-only with respect to shards and writes only candidate thumbnails/metadata."
        )
    )
    parser.add_argument("--shard-root", type=Path, default=Path("/mnt/Data/SpectralData/DZ1_joint_pretrain_202607/data/dz1_shards"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sensor", default="dz1_vnir")
    parser.add_argument("--tile-patches", type=int, default=4)
    parser.add_argument(
        "--scan-shards",
        type=int,
        default=0,
        help="0 means all matching shards; otherwise scan this many matching shards from the manifest.",
    )
    parser.add_argument("--max-windows-per-shard", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def robust01(arr: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite, [lower, upper])
    if not np.isfinite(hi - lo) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def band_index(wavelengths: np.ndarray, target: float) -> int:
    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    return int(np.argmin(np.abs(wavelengths - target)))


def pseudo_rgb(cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    indices = [band_index(wavelengths, target) for target in (650.0, 560.0, 480.0)]
    return np.stack([robust01(cube[idx]) for idx in indices], axis=-1)


def false_color(cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    indices = [band_index(wavelengths, target) for target in (814.0, 650.0, 560.0)]
    return np.stack([robust01(cube[idx]) for idx in indices], axis=-1)


def spectral_indices(cube: np.ndarray, wavelengths: np.ndarray) -> dict[str, np.ndarray]:
    nir = cube[band_index(wavelengths, 814.0)].astype(np.float32)
    red = cube[band_index(wavelengths, 650.0)].astype(np.float32)
    green = cube[band_index(wavelengths, 560.0)].astype(np.float32)
    blue = cube[band_index(wavelengths, 480.0)].astype(np.float32)
    eps = 1.0e-6
    ndvi = (nir - red) / (np.abs(nir) + np.abs(red) + eps)
    ndwi = (green - nir) / (np.abs(green) + np.abs(nir) + eps)
    brightness = (red + green + blue) / 3.0
    return {"ndvi": ndvi, "ndwi": ndwi, "brightness": brightness}


def gradient_maps(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = np.asarray(gray, dtype=np.float32)
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
    gy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def block_reduce_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    h, w = arr.shape[:2]
    h2 = h - h % factor
    w2 = w - w % factor
    trimmed = arr[:h2, :w2]
    if arr.ndim == 2:
        return trimmed.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))
    channels = arr.shape[2]
    return trimmed.reshape(h2 // factor, factor, w2 // factor, factor, channels).mean(axis=(1, 3))


def entropy_from_labels(labels: np.ndarray) -> float:
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts.astype(np.float64) / float(counts.sum())
    ent = -np.sum(probs * np.log2(probs + 1.0e-12))
    max_ent = math.log2(max(2, len(counts)))
    return float(ent / max_ent) if max_ent > 0 else 0.0


def candidate_metrics(cube: np.ndarray, wavelengths: np.ndarray) -> dict[str, Any]:
    rgb = pseudo_rgb(cube, wavelengths)
    fcir = false_color(cube, wavelengths)
    spec = spectral_indices(cube, wavelengths)
    gray = rgb.mean(axis=-1)
    gx, gy, mag = gradient_maps(gray)

    coarse_gray = block_reduce_mean(gray, 8)
    _, _, coarse_mag = gradient_maps(coarse_gray)
    edge = float(mag.mean())
    coarse_edge = float(coarse_mag.mean())
    coarse_to_fine = coarse_edge / (edge + 1.0e-6)

    # Directional/linear structures are useful anchors, but pure mountain texture
    # often has high fine-scale edges and low persistent coarse edges.
    theta = np.mod(np.arctan2(gy, gx), np.pi)
    weights = mag.reshape(-1)
    theta_flat = theta.reshape(-1)
    if float(weights.sum()) > 1.0e-8:
        hist, _ = np.histogram(theta_flat, bins=18, range=(0.0, np.pi), weights=weights)
        orientation_concentration = float(hist.max() / (hist.sum() + 1.0e-8))
        cardinal = (np.abs(np.sin(theta_flat)) < 0.22) | (np.abs(np.cos(theta_flat)) < 0.22)
        rectilinear = float(weights[cardinal].sum() / (weights.sum() + 1.0e-8))
    else:
        orientation_concentration = 0.0
        rectilinear = 0.0

    ndvi = spec["ndvi"]
    ndwi = spec["ndwi"]
    brightness01 = robust01(spec["brightness"])
    veg_frac = float(np.mean(ndvi > 0.20))
    water_frac = float(np.mean(ndwi > 0.08))
    bright_frac = float(np.mean(brightness01 > 0.72))
    dark_frac = float(np.mean(brightness01 < 0.08))

    ndvi_cls = np.digitize(ndvi, [-0.10, 0.08, 0.20, 0.35])
    ndwi_cls = np.digitize(ndwi, [-0.12, 0.00, 0.08])
    bright_cls = np.digitize(brightness01, [0.22, 0.48, 0.72])
    spectral_labels = ndvi_cls + 5 * ndwi_cls + 20 * bright_cls
    spectral_entropy = entropy_from_labels(spectral_labels)

    coarse_rgb = block_reduce_mean(rgb, 16)
    rgb_labels = (
        np.digitize(coarse_rgb[..., 0], [0.25, 0.50, 0.75])
        + 4 * np.digitize(coarse_rgb[..., 1], [0.25, 0.50, 0.75])
        + 16 * np.digitize(coarse_rgb[..., 2], [0.25, 0.50, 0.75])
    )
    region_entropy = entropy_from_labels(rgb_labels)

    # Penalize visually unhelpful extremes: nearly pure texture, too dark, too
    # bright, or almost single-material scenes.
    texture_penalty = max(0.0, edge - 0.050) * max(0.0, 0.26 - coarse_to_fine)
    monotone_penalty = max(0.0, 0.28 - spectral_entropy) + max(0.0, 0.25 - region_entropy)
    saturation_penalty = 0.55 * max(0.0, dark_frac - 0.35) + 0.35 * max(0.0, bright_frac - 0.55)
    mixed_anchor = min(1.0, 2.5 * min(max(veg_frac, water_frac, bright_frac), 0.40))

    return {
        "rgb": rgb,
        "false_color": fcir,
        "ndvi": ndvi,
        "edge_map": robust01(mag, 4.0, 99.0),
        "edge": edge,
        "coarse_edge": coarse_edge,
        "coarse_to_fine": float(coarse_to_fine),
        "orientation_concentration": orientation_concentration,
        "rectilinear": rectilinear,
        "veg_frac": veg_frac,
        "water_frac": water_frac,
        "bright_frac": bright_frac,
        "dark_frac": dark_frac,
        "spectral_entropy": spectral_entropy,
        "region_entropy": region_entropy,
        "texture_penalty": float(texture_penalty),
        "monotone_penalty": float(monotone_penalty),
        "saturation_penalty": float(saturation_penalty),
        "mixed_anchor": float(mixed_anchor),
    }


def metric_norm(values: np.ndarray, inverse: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        out = np.zeros_like(values, dtype=np.float32)
    else:
        lo, hi = np.percentile(finite, [5.0, 95.0])
        if hi <= lo:
            lo, hi = float(np.min(finite)), float(np.max(finite))
        if hi <= lo:
            out = np.zeros_like(values, dtype=np.float32)
        else:
            out = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - out if inverse else out


def score_candidates(candidates: list[dict[str, Any]]) -> None:
    if not candidates:
        return
    keys = [
        "coarse_edge",
        "coarse_to_fine",
        "orientation_concentration",
        "rectilinear",
        "spectral_entropy",
        "region_entropy",
        "mixed_anchor",
        "texture_penalty",
        "monotone_penalty",
        "saturation_penalty",
    ]
    norm = {k: metric_norm(np.array([c["metrics"][k] for c in candidates])) for k in keys}
    score = (
        0.20 * norm["coarse_edge"]
        + 0.18 * norm["coarse_to_fine"]
        + 0.12 * norm["orientation_concentration"]
        + 0.10 * norm["rectilinear"]
        + 0.18 * norm["spectral_entropy"]
        + 0.14 * norm["region_entropy"]
        + 0.10 * norm["mixed_anchor"]
        - 0.12 * norm["texture_penalty"]
        - 0.10 * norm["monotone_penalty"]
        - 0.08 * norm["saturation_penalty"]
    )
    for idx, cand in enumerate(candidates):
        cand["semantic_anchor_score"] = float(score[idx])


def load_shards(manifest_path: Path, shard_root: Path, sensor: str) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("sensor") != sensor:
                continue
            values_path = Path(obj["values_path"])
            metadata_path = Path(obj["metadata_path"])
            if not values_path.is_absolute():
                values_path = shard_root / values_path
            if not metadata_path.is_absolute():
                metadata_path = shard_root / metadata_path
            obj = dict(obj)
            obj["values_path"] = str(values_path)
            obj["metadata_path"] = str(metadata_path)
            shards.append(obj)
    return shards


def assemble_tile(values: np.ndarray, indices: list[int], tile_patches: int) -> np.ndarray:
    first = np.asarray(values[indices[0]], dtype=np.float32)
    channels, patch_h, patch_w = first.shape
    tile = np.zeros((channels, tile_patches * patch_h, tile_patches * patch_w), dtype=np.float32)
    for linear, sample_index in enumerate(indices):
        rr = linear // tile_patches
        cc = linear % tile_patches
        tile[:, rr * patch_h : (rr + 1) * patch_h, cc * patch_w : (cc + 1) * patch_w] = np.asarray(
            values[sample_index], dtype=np.float32
        )
    return tile


def iter_windows(
    shard: dict[str, Any],
    shard_index: int,
    tile_patches: int,
    max_windows_per_shard: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    meta_path = Path(shard["metadata_path"])
    if not meta_path.exists():
        return []
    with np.load(meta_path) as meta:
        products = np.asarray(meta["source_product_index"])
        rows = np.asarray(meta["source_row"])
        cols = np.asarray(meta["source_col"])
        lat = np.asarray(meta["latitude"]) if "latitude" in meta else None
        lon = np.asarray(meta["longitude"]) if "longitude" in meta else None
        valid = np.asarray(meta["valid_fraction"]) if "valid_fraction" in meta else None
        sat = np.asarray(meta["saturation_fraction"]) if "saturation_fraction" in meta else None

    by_product: dict[int, dict[tuple[int, int], int]] = {}
    for sample_index, (product, row, col) in enumerate(zip(products, rows, cols)):
        by_product.setdefault(int(product), {})[(int(row), int(col))] = int(sample_index)

    windows: list[dict[str, Any]] = []
    for product, pos_to_index in by_product.items():
        unique_rows = sorted({pos[0] for pos in pos_to_index})
        unique_cols = sorted({pos[1] for pos in pos_to_index})
        product_windows: list[tuple[int, int]] = []
        for ri in range(0, max(0, len(unique_rows) - tile_patches + 1)):
            for ci in range(0, max(0, len(unique_cols) - tile_patches + 1)):
                block_rows = unique_rows[ri : ri + tile_patches]
                block_cols = unique_cols[ci : ci + tile_patches]
                if all((r, c) in pos_to_index for r in block_rows for c in block_cols):
                    product_windows.append((ri, ci))
        if max_windows_per_shard > 0 and len(product_windows) > max_windows_per_shard:
            keep = rng.choice(len(product_windows), size=max_windows_per_shard, replace=False)
            product_windows = [product_windows[int(i)] for i in keep]
        for ri, ci in product_windows:
            block_rows = unique_rows[ri : ri + tile_patches]
            block_cols = unique_cols[ci : ci + tile_patches]
            indices = [pos_to_index[(r, c)] for r in block_rows for c in block_cols]
            record: dict[str, Any] = {
                "shard_index": int(shard_index),
                "values_path": shard["values_path"],
                "metadata_path": shard["metadata_path"],
                "source_product_index": int(product),
                "sample_indices": [int(i) for i in indices],
                "source_rows": [int(r) for r in block_rows],
                "source_cols": [int(c) for c in block_cols],
                "sensor": shard.get("sensor"),
                "level": shard.get("level"),
                "gsd_m": float(shard.get("gsd_m", 28.0)),
                "native_gsd_m": shard.get("native_gsd_m"),
                "resampling_factor": shard.get("resampling_factor"),
                "wavelengths": shard.get("wavelength"),
                "fwhm": shard.get("fwhm"),
                "tile_patches": int(tile_patches),
                "sample_shape": [int(shard["sample_shape"][0]), 64 * tile_patches, 64 * tile_patches],
            }
            if lat is not None and lon is not None:
                record["center_latitude"] = float(np.nanmean(lat[indices]))
                record["center_longitude"] = float(np.nanmean(lon[indices]))
            if valid is not None:
                record["mean_valid_fraction"] = float(np.nanmean(valid[indices]))
            if sat is not None:
                record["mean_saturation_fraction"] = float(np.nanmean(sat[indices]))
            windows.append(record)
    return windows


def strip_arrays(candidate: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in candidate.items() if k not in {"rgb", "false_color", "ndvi", "edge_map", "cube", "_arrays"}}
    if "metrics" in out and isinstance(out["metrics"], dict):
        clean_metrics: dict[str, Any] = {}
        for key, value in out["metrics"].items():
            if isinstance(value, np.ndarray):
                continue
            if isinstance(value, np.generic):
                clean_metrics[key] = value.item()
            else:
                clean_metrics[key] = value
        out["metrics"] = clean_metrics
    return out


def write_candidate_panel(candidate: dict[str, Any], path: Path, dpi: int) -> None:
    metrics = candidate["metrics"]
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "font.weight": "bold"})
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.35), constrained_layout=True)
    axes[0].imshow(metrics["rgb"])
    axes[0].set_title("Pseudo-RGB")
    axes[1].imshow(metrics["false_color"])
    axes[1].set_title("False color")
    axes[2].imshow(metrics["ndvi"], cmap="RdYlGn", vmin=-0.35, vmax=0.55)
    axes[2].set_title("NDVI proxy")
    axes[3].imshow(metrics["edge_map"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[3].set_title("Structure map")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        (
            f"rank {candidate['rank']:02d} | score={candidate['semantic_anchor_score']:.3f} | "
            f"product={candidate['source_product_index']} | row={candidate['source_rows'][0]}-{candidate['source_rows'][-1]} "
            f"col={candidate['source_cols'][0]}-{candidate['source_cols'][-1]} | "
            f"lat={candidate.get('center_latitude', float('nan')):.3f}, lon={candidate.get('center_longitude', float('nan')):.3f}"
        ),
        fontsize=9,
        fontweight="bold",
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_atlas(candidates: list[dict[str, Any]], path: Path, dpi: int) -> None:
    n = len(candidates)
    if n == 0:
        return
    rows = n
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7, "font.weight": "bold"})
    fig, axes = plt.subplots(rows, 4, figsize=(7.8, max(2.0, 1.88 * rows)), constrained_layout=True)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    headers = ["Pseudo-RGB", "False color", "NDVI proxy", "Structure map"]
    for c, header in enumerate(headers):
        axes[0, c].set_title(header, fontsize=8, fontweight="bold")
    for r, cand in enumerate(candidates):
        metrics = cand["metrics"]
        panels = [
            (metrics["rgb"], None, None, None),
            (metrics["false_color"], None, None, None),
            (metrics["ndvi"], "RdYlGn", -0.35, 0.55),
            (metrics["edge_map"], "magma", 0.0, 1.0),
        ]
        for c, (img, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(
                    f"#{cand['rank']}\nS={cand['semantic_anchor_score']:.2f}\nP{cand['source_product_index']}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest = args.manifest or (args.shard_root / "manifest.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    shards = load_shards(manifest, args.shard_root, args.sensor)
    if args.scan_shards > 0:
        shards = shards[: args.scan_shards]
    all_candidates: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards):
        values_path = Path(shard["values_path"])
        if not values_path.exists():
            continue
        windows = iter_windows(shard, shard_index, args.tile_patches, args.max_windows_per_shard, rng)
        if not windows:
            continue
        values = np.load(values_path, mmap_mode="r")
        wavelengths = np.asarray(shard["wavelength"], dtype=np.float32)
        for record in windows:
            cube = assemble_tile(values, record["sample_indices"], args.tile_patches)
            metrics = candidate_metrics(cube, wavelengths)
            record["metrics"] = {k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)}
            record["_arrays"] = {
                "rgb": metrics["rgb"],
                "false_color": metrics["false_color"],
                "ndvi": metrics["ndvi"],
                "edge_map": metrics["edge_map"],
            }
            all_candidates.append(record)

    score_candidates(all_candidates)
    all_candidates.sort(key=lambda item: item.get("semantic_anchor_score", -1.0), reverse=True)
    top = all_candidates[: args.top_k]
    for rank, cand in enumerate(top, start=1):
        cand["rank"] = int(rank)
        arrays = cand.pop("_arrays")
        cand["metrics"].update(arrays)
        write_candidate_panel(cand, args.output_dir / f"candidate_{rank:02d}.png", args.dpi)

    # Two atlas pages are usually easier to inspect than one very tall figure.
    page_size = 12
    for page_start in range(0, len(top), page_size):
        page = top[page_start : page_start + page_size]
        write_atlas(page, args.output_dir / f"candidate_atlas_page_{page_start // page_size + 1:02d}.png", args.dpi)

    serializable = []
    for cand in top:
        serializable.append(strip_arrays(cand))
    (args.output_dir / "candidate_tiles_top.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    summary = {
        "status": "ok",
        "sensor": args.sensor,
        "tile_patches": args.tile_patches,
        "scanned_shards": len(shards),
        "num_candidates_scored": len(all_candidates),
        "top_k": len(top),
        "output_dir": str(args.output_dir),
        "best": serializable[0] if serializable else None,
    }
    (args.output_dir / "candidate_selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
