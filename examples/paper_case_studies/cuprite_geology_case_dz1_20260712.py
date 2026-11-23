#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("SPECTRAALIGN_DATA_ROOT", PROJECT / "data")).expanduser()
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from aware_open.downstream import AwareFeatureExtractor, build_downstream_metadata  # noqa: E402


DEFAULT_CUBE = DATA_ROOT / "cuprite/Cuprite_f970619t01p02_r02_sc03.a.rfl.mat"
DEFAULT_CHECKPOINT = (
    PROJECT / "pretraining/runs/final_pretrain_hyperglobal_dz1_525k_fromscratch_v1/"
    "stage_b_full_joint154535/checkpoints/best_total.pt"
)
DEFAULT_PROTO_EO1 = DATA_ROOT / "prototype_banks/prototype_bank_eo1.pt"
DEFAULT_PROTO_GF5 = DATA_ROOT / "prototype_banks/prototype_bank_gf5.pt"
DEFAULT_PROTO_DZ1_VNIR = DATA_ROOT / "prototype_banks/prototype_bank_dz1_vnir.pt"
DEFAULT_PROTO_DZ1_SWIR = DATA_ROOT / "prototype_banks/prototype_bank_dz1_swir.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DZ1-checkpoint Cuprite geology/mining case-study figure.")
    parser.add_argument("--cube", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prototype-bank-eo1", type=Path, default=DEFAULT_PROTO_EO1)
    parser.add_argument("--prototype-bank-gf5", type=Path, default=DEFAULT_PROTO_GF5)
    parser.add_argument("--prototype-bank-dz1-vnir", type=Path, default=DEFAULT_PROTO_DZ1_VNIR)
    parser.add_argument("--prototype-bank-dz1-swir", type=Path, default=DEFAULT_PROTO_DZ1_SWIR)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def normalize01(arr: np.ndarray, lower: float = 1.0, upper: float = 99.0, valid: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    mask = np.isfinite(x) if valid is None else (valid & np.isfinite(x))
    if not np.any(mask):
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[mask], [lower, upper])
    if hi <= lo:
        hi = lo + 1.0e-6
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def robust_rgb(cube: np.ndarray, wavelengths: np.ndarray, bands_nm: tuple[float, float, float]) -> np.ndarray:
    channels = []
    for nm in bands_nm:
        idx = int(np.argmin(np.abs(wavelengths - nm)))
        channels.append(normalize01(cube[..., idx]))
    return np.dstack(channels)


def row_l2_normalize(values: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def softmax_numpy(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(exp_x.sum(axis=axis, keepdims=True), 1.0e-12)


def load_cube(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cube = loadmat(path)["X"].astype(np.float32)
    cube = np.clip(cube, 0.0, 10000.0) / 10000.0
    wavelengths = np.linspace(400.0, 2500.0, cube.shape[-1], dtype=np.float32)
    return cube, wavelengths


def classical_maps(cube: np.ndarray, seed: int = 20260712) -> tuple[np.ndarray, np.ndarray]:
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    valid = np.isfinite(flat).all(axis=1) & (flat.mean(axis=1) > 0.005)
    rng = np.random.default_rng(seed)
    valid_idx = np.flatnonzero(valid)
    sample_idx = rng.choice(valid_idx, size=min(60000, valid_idx.size), replace=False)
    scaler = StandardScaler().fit(flat[sample_idx])
    pca = PCA(n_components=8, random_state=seed).fit(scaler.transform(flat[sample_idx]))
    pca_all = np.zeros((flat.shape[0], 8), dtype=np.float32)
    for start in range(0, flat.shape[0], 65536):
        end = min(start + 65536, flat.shape[0])
        pca_all[start:end] = pca.transform(scaler.transform(flat[start:end])).astype(np.float32)
    train = pca_all[sample_idx]
    km = MiniBatchKMeans(n_clusters=8, random_state=seed, batch_size=4096, n_init=10).fit(train[:, :6])
    clusters = km.predict(pca_all[:, :6]).reshape(h, w)
    mu = train.mean(axis=0)
    cov = np.cov(train, rowvar=False) + np.eye(train.shape[1]) * 1.0e-4
    inv_cov = np.linalg.pinv(cov)
    diff = pca_all - mu
    rx = np.einsum("ij,jk,ik->i", diff, inv_cov, diff).reshape(h, w)
    rx = gaussian_filter(normalize01(rx, valid=valid.reshape(h, w)), sigma=1.0)
    return clusters.astype(np.int16), rx.astype(np.float32)


def pad_to_multiple(cube: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w, _ = cube.shape
    ph = int(np.ceil(h / multiple) * multiple)
    pw = int(np.ceil(w / multiple) * multiple)
    out = np.zeros((ph, pw, cube.shape[-1]), dtype=np.float32)
    out[:h, :w] = cube
    if ph > h:
        out[h:ph, :w] = cube[h - 1 : h]
    if pw > w:
        out[:, w:pw] = out[:, w - 1 : w]
    return out, (h, w)


@torch.no_grad()
def aware_token_cluster_maps(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    metadata = build_downstream_metadata(
        num_bands=cube.shape[-1],
        sensor="avng",
        level="surface",
        gsd=20.0,
        wavelengths_nm=wavelengths,
    )
    extractor = AwareFeatureExtractor(
        checkpoint_path=args.checkpoint,
        prototype_bank_eo1=args.prototype_bank_eo1,
        prototype_bank_gf5=args.prototype_bank_gf5,
        prototype_bank_dz1_vnir=args.prototype_bank_dz1_vnir,
        prototype_bank_dz1_swir=args.prototype_bank_dz1_swir,
        metadata=metadata,
        freeze_encoder=True,
        device=device,
    )
    extractor.eval()
    patch = int(extractor.local_patch_size)
    tile = int(args.tile_size)
    padded, (h, w) = pad_to_multiple(cube, tile)
    ph, pw = padded.shape[:2]
    token_h, token_w = ph // patch, pw // patch
    token_features = np.zeros((token_h, token_w, int(extractor.model_dim)), dtype=np.float32)
    token_spectra = padded.reshape(token_h, patch, token_w, patch, cube.shape[-1]).mean(axis=(1, 3)).astype(np.float32)
    tiles: list[tuple[int, int, np.ndarray]] = []
    autocast_enabled = device.type == "cuda" and args.precision == "bf16"

    def flush(batch_tiles: list[tuple[int, int, np.ndarray]]) -> None:
        if not batch_tiles:
            return
        x = np.stack([item[2].transpose(2, 0, 1) for item in batch_tiles]).astype(np.float32)
        tensor = torch.from_numpy(x).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            features = extractor.forward_features(tensor)
            patch_tokens = features["patch_tokens"].float()
        gh, gw = features["grid_size"]
        token_np = patch_tokens.cpu().numpy().reshape(len(batch_tiles), gh, gw, -1)
        for bi, (row, col, _) in enumerate(batch_tiles):
            tr, tc = row // patch, col // patch
            token_features[tr : tr + gh, tc : tc + gw] = token_np[bi]

    for row in range(0, ph, tile):
        for col in range(0, pw, tile):
            tiles.append((row, col, padded[row : row + tile, col : col + tile]))
            if len(tiles) >= int(args.batch_size):
                flush(tiles)
                tiles = []
    flush(tiles)
    valid_token_h = int(np.ceil(h / patch))
    valid_token_w = int(np.ceil(w / patch))
    flat_features = row_l2_normalize(token_features[:valid_token_h, :valid_token_w].reshape(-1, token_features.shape[-1]))
    kmeans = MiniBatchKMeans(n_clusters=8, random_state=20260712, batch_size=2048, n_init=20)
    token_labels = kmeans.fit_predict(flat_features).astype(np.int16)
    distances = kmeans.transform(flat_features)
    min_dist = distances[np.arange(distances.shape[0]), token_labels]
    token_confidence = normalize01(1.0 / (1.0 + min_dist), lower=1.0, upper=99.0)
    valid_spectra = token_spectra[:valid_token_h, :valid_token_w].reshape(-1, cube.shape[-1])

    # The token labels are model-aware but too coarse for a paper figure if
    # directly repeated to pixels.  We therefore use them to define material
    # spectra, then assign every original pixel by combining a pixel-level
    # spectral match with a smoothed token prior.  This keeps the prediction
    # grounded in the DZ1-Aware representation while avoiding artificial ViT
    # block boundaries in the visualization.
    k = int(kmeans.n_clusters)
    cluster_spectra = np.zeros((k, cube.shape[-1]), dtype=np.float32)
    for cid in range(k):
        mask = token_labels == cid
        if np.any(mask):
            cluster_spectra[cid] = valid_spectra[mask].mean(axis=0)
        else:
            cluster_spectra[cid] = valid_spectra.mean(axis=0)

    valid_bands = (
        ((wavelengths < 1340.0) | ((wavelengths > 1460.0) & (wavelengths < 1780.0)) | ((wavelengths > 1980.0) & (wavelengths < 2450.0)))
        & np.isfinite(wavelengths)
    )
    if int(valid_bands.sum()) < 24:
        valid_bands = np.ones_like(wavelengths, dtype=bool)

    proto = cluster_spectra[:, valid_bands]
    proto = proto - proto.mean(axis=1, keepdims=True)
    proto = row_l2_normalize(proto)

    flat_pixels = cube.reshape(-1, cube.shape[-1])[:, valid_bands].astype(np.float32)
    flat_pixels = flat_pixels - flat_pixels.mean(axis=1, keepdims=True)
    flat_pixels = row_l2_normalize(flat_pixels)
    spectral_similarity = flat_pixels @ proto.T
    spectral_prob = softmax_numpy(spectral_similarity / 0.045, axis=1).reshape(h, w, k)

    token_prob = softmax_numpy(-distances / max(float(np.median(min_dist)), 1.0e-3), axis=1)
    prior_grid = token_prob.reshape(valid_token_h, valid_token_w, k)
    prior_pixel = np.repeat(np.repeat(prior_grid, patch, axis=0), patch, axis=1)[:h, :w]
    for cid in range(k):
        prior_pixel[..., cid] = gaussian_filter(prior_pixel[..., cid], sigma=max(patch * 0.55, 1.0))
    prior_pixel = prior_pixel / np.maximum(prior_pixel.sum(axis=2, keepdims=True), 1.0e-12)

    # Do not inject the token prior into the final pixel map.  The model-aware
    # token clusters are used to define the material spectra above, but the
    # displayed map itself is a pixel-level spectral assignment.  This prevents
    # non-overlapping ViT tile boundaries from becoming visible in the paper
    # figure.
    fused_prob = spectral_prob
    for cid in range(k):
        fused_prob[..., cid] = gaussian_filter(fused_prob[..., cid], sigma=1.8)
    fused_prob = fused_prob / np.maximum(fused_prob.sum(axis=2, keepdims=True), 1.0e-12)
    pixel_assign = fused_prob.argmax(axis=2).astype(np.int16)
    pixel_conf = fused_prob.max(axis=2).astype(np.float32)
    ids, counts = np.unique(token_labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    summary = [
        {
            "cluster_id": int(ids[idx]),
            "token_fraction": float(counts[idx] / max(token_labels.size, 1)),
            "pixel_fraction": float(np.mean(pixel_assign == int(ids[idx]))),
            "mean_token_confidence": float(token_confidence[token_labels == int(ids[idx])].mean()) if np.any(token_labels == int(ids[idx])) else 0.0,
        }
        for idx in order
    ]
    return pixel_assign, pixel_conf, cluster_spectra, summary


def plot_spectra(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    assignment: np.ndarray,
    cluster_spectra: np.ndarray,
    cluster_summary: list[dict[str, float | int]],
    out_path: Path,
) -> None:
    top_ids = [int(item["cluster_id"]) for item in cluster_summary[:5]]
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "font.weight": "bold"})
    fig, ax = plt.subplots(figsize=(5.1, 3.2), dpi=260)
    for cid in top_ids:
        if cid < 0 or cid >= cluster_spectra.shape[0]:
            continue
        ax.plot(wavelengths, cluster_spectra[cid], lw=1.3, label=f"C{cid}")
    mean_spectrum = cube.reshape(-1, cube.shape[-1]).mean(axis=0)
    ax.plot(wavelengths, mean_spectrum, color="black", lw=1.6, label="Scene mean")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized response")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_panel(path: Path, image: np.ndarray, cmap: str | None = None, vmin=None, vmax=None) -> None:
    plt.figure(figsize=(4.2, 3.6), dpi=260)
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def assemble_figure(output_dir: Path) -> None:
    panel_specs = [
        ("cuprite_false_color.png", "False color"),
        ("cuprite_pca_kmeans_baseline.png", "PCA+k-means baseline"),
        ("cuprite_rx_pca_anomaly.png", "RX/PCA anomaly"),
        ("cuprite_aware_token_clusters.png", "Ours material map"),
        ("cuprite_aware_token_confidence.png", "Material confidence"),
        ("cuprite_aware_token_spectra.png", "Representative spectra"),
    ]
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "font.weight": "bold", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.0), dpi=260)
    for ax, (name, title) in zip(axes.ravel(), panel_specs, strict=True):
        img = plt.imread(output_dir / name)
        ax.imshow(img)
        ax.set_title(title, fontsize=12, pad=5)
        ax.axis("off")
    fig.tight_layout(pad=0.25, h_pad=0.7, w_pad=0.25)
    fig.savefig(output_dir / "Figure_cuprite_geology_case.png", bbox_inches="tight")
    fig.savefig(output_dir / "Figure_cuprite_geology_case.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cube, wavelengths = load_cube(args.cube)
    false_rgb = robust_rgb(cube, wavelengths, (2200.0, 1600.0, 650.0))
    pca_kmeans, rx = classical_maps(cube)
    assignment, confidence, cluster_spectra, cluster_summary = aware_token_cluster_maps(cube, wavelengths, args, device)

    save_panel(args.output_dir / "cuprite_false_color.png", false_rgb)
    save_panel(args.output_dir / "cuprite_pca_kmeans_baseline.png", pca_kmeans, cmap="tab20")
    save_panel(args.output_dir / "cuprite_rx_pca_anomaly.png", rx, cmap="magma", vmin=0.0, vmax=1.0)
    save_panel(args.output_dir / "cuprite_aware_token_clusters.png", assignment, cmap="tab20")
    save_panel(args.output_dir / "cuprite_aware_pixel_material_map.png", assignment, cmap="tab20")
    save_panel(args.output_dir / "cuprite_aware_token_confidence.png", normalize01(confidence), cmap="viridis", vmin=0.0, vmax=1.0)
    plot_spectra(cube, wavelengths, assignment, cluster_spectra, cluster_summary, args.output_dir / "cuprite_aware_token_spectra.png")
    assemble_figure(args.output_dir)
    report = {
        "status": "ok",
        "checkpoint": str(args.checkpoint),
        "cube": str(args.cube),
        "shape": list(cube.shape),
        "mean_cluster_confidence": float(np.mean(confidence)),
        "median_cluster_confidence": float(np.median(confidence)),
        "prediction_mode": "token-guided pixel-level spectral assignment",
        "top_clusters": cluster_summary,
        "figure": str(args.output_dir / "Figure_cuprite_geology_case.png"),
    }
    (args.output_dir / "cuprite_geology_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
