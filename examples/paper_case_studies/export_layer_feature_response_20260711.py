from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _add_project_to_path(project_root: Path) -> None:
    code_root = project_root / "code"
    if code_root.exists():
        sys.path.insert(0, str(code_root))
    if project_root.exists():
        sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export block-wise feature response maps for the final HyperGlobal-DZ1 "
            "Aware-HSI checkpoint. The script is intentionally read-only with respect "
            "to datasets and checkpoints."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/Data/SpectralData/DZ1_joint_pretrain_202607"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/mnt/Data/SpectralData/DZ1_joint_pretrain_202607/"
            "runs/final_pretrain_hyperglobal_dz1_525k_fromscratch_v1/"
            "stage_b_full_joint154535/checkpoints/best_total.pt"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/mnt/Data/SpectralData/DZ1_joint_pretrain_202607/data/manifests/hyperglobal_dz1_core.jsonl"),
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        default=Path("/mnt/Data/SpectralData/DZ1_joint_pretrain_202607/data/dz1_shards"),
    )
    parser.add_argument(
        "--prototype-bank-eo1",
        type=Path,
        default=Path(
            "/mnt/Data/SpectralData/Aware_open/artifacts/"
            "prototypes_v3_core_waterzero_artificialx4_20260503/prototype_bank_eo1.pt"
        ),
    )
    parser.add_argument(
        "--prototype-bank-gf5",
        type=Path,
        default=Path(
            "/mnt/Data/SpectralData/Aware_open/artifacts/"
            "prototypes_v3_core_waterzero_artificialx4_20260503/prototype_bank_gf5.pt"
        ),
    )
    parser.add_argument(
        "--prototype-bank-dz1-vnir",
        type=Path,
        default=Path("/mnt/Data/SpectralData/DZ1_joint_pretrain_202607/artifacts/dz1_prototypes_v1/prototype_bank_dz1_vnir.pt"),
    )
    parser.add_argument(
        "--prototype-bank-dz1-swir",
        type=Path,
        default=Path("/mnt/Data/SpectralData/DZ1_joint_pretrain_202607/artifacts/dz1_prototypes_v1/prototype_bank_dz1_swir.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sensor", default="dz1_vnir", choices=("dz1_vnir", "dz1_swir", "eo1", "gf5"))
    parser.add_argument("--scan-shards", type=int, default=60)
    parser.add_argument("--scan-samples-per-shard", type=int, default=8)
    parser.add_argument(
        "--tile-patches",
        type=int,
        default=4,
        help="For DZ1 shards, assemble a contiguous tile of this many 64x64 patches per side before feature extraction.",
    )
    parser.add_argument("--max-windows-per-shard", type=int, default=220)
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help=(
            "Optional candidate_tiles_top.json produced by select_dz1_feature_tiles_20260712.py. "
            "When provided, the selected DZ1 tile is loaded from this file instead of being auto-selected."
        ),
    )
    parser.add_argument("--selection-rank", type=int, default=1, help="Rank to load from --selection-json.")
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--normalization", default="mean_std", choices=("none", "mean_std", "robust"))
    parser.add_argument(
        "--response-method",
        default="feature_norm",
        choices=("feature_norm", "block_update"),
        help=(
            "feature_norm follows the DINOv3 reference visualization: L2 norm of each patch token after each block. "
            "block_update preserves the earlier diagnostic: patch-token update magnitude between adjacent blocks."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=420)
    return parser.parse_args()


def checkpoint_get(args: dict[str, Any], name: str, default: Any) -> Any:
    return args.get(name, default)


def build_config_from_checkpoint_args(checkpoint_args: dict[str, Any], prototype_count: int):
    from aware_open.config import AwareEncoderConfig, FoundationConfig, FoundationDecoderConfig, FoundationLossConfig

    depth = int(checkpoint_get(checkpoint_args, "token_encoder_depth", 12))
    encoder = AwareEncoderConfig(
        model_dim=int(checkpoint_get(checkpoint_args, "model_dim", 768)),
        band_dim=int(checkpoint_get(checkpoint_args, "band_dim", 192)),
        context_dim=int(checkpoint_get(checkpoint_args, "context_dim", 256)),
        spectral_dim=int(checkpoint_get(checkpoint_args, "spectral_dim", 96)),
        spatial_dim=int(checkpoint_get(checkpoint_args, "spatial_dim", 96)),
        num_prototypes=int(checkpoint_get(checkpoint_args, "num_routing_prototypes", 12)),
        local_patch_size=int(checkpoint_get(checkpoint_args, "local_patch_size", 8)),
        prototype_heads=int(checkpoint_get(checkpoint_args, "prototype_heads", 8)),
        prototype_depth=int(checkpoint_get(checkpoint_args, "prototype_depth", 2)),
        token_encoder_depth=depth,
        token_encoder_heads=int(checkpoint_get(checkpoint_args, "token_encoder_heads", 12)),
        backbone_out_indices=tuple(range(depth)),
        backbone_pretrained=False,
        drop_rate=float(checkpoint_get(checkpoint_args, "drop_rate", 0.1)),
        metadata_ablation=str(checkpoint_get(checkpoint_args, "metadata_ablation", "full")),
    )
    decoder = FoundationDecoderConfig(
        decoder_dim=int(checkpoint_get(checkpoint_args, "decoder_dim", 384)),
        decoder_depth=int(checkpoint_get(checkpoint_args, "decoder_depth", 2)),
        decoder_heads=int(checkpoint_get(checkpoint_args, "decoder_heads", 6)),
        decoder_dropout=float(checkpoint_get(checkpoint_args, "decoder_dropout", 0.0)),
        tdfa_depth=int(checkpoint_get(checkpoint_args, "tdfa_depth", 2)),
        frequency_groups=int(checkpoint_get(checkpoint_args, "frequency_groups", 5)),
        low_cutoff_ratio=float(checkpoint_get(checkpoint_args, "frequency_cutoff_low", 0.15)),
        mid_cutoff_ratio=float(checkpoint_get(checkpoint_args, "frequency_cutoff_mid", 0.40)),
    )
    loss = FoundationLossConfig(
        reconstruction_low=float(checkpoint_get(checkpoint_args, "reconstruction_low_weight", 0.10)),
        reconstruction_mid=float(checkpoint_get(checkpoint_args, "reconstruction_mid_weight", 0.20)),
        reconstruction_full=float(checkpoint_get(checkpoint_args, "reconstruction_full_weight", 0.70)),
        proto_kl=float(checkpoint_get(checkpoint_args, "proto_kl_weight", 0.03)),
        proto_balance=float(checkpoint_get(checkpoint_args, "proto_balance_weight", 0.005)),
    )
    return FoundationConfig(
        encoder=encoder,
        decoder=decoder,
        loss=loss,
        mask_ratio=float(checkpoint_get(checkpoint_args, "mask_ratio", 0.75)),
        prototype_bank_size=int(prototype_count),
        prototype_temperature_teacher=float(checkpoint_get(checkpoint_args, "prototype_temperature_teacher", 0.07)),
        prototype_temperature_student=float(checkpoint_get(checkpoint_args, "prototype_temperature_student", 0.07)),
    )


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
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def pseudo_rgb(cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    cube = np.asarray(cube, dtype=np.float32)
    targets = [650.0, 560.0, 480.0]
    indices = [int(np.argmin(np.abs(wavelengths - target))) for target in targets]
    rgb = np.stack([robust01(cube[idx]) for idx in indices], axis=-1)
    return rgb


def edge_score_from_rgb(rgb: np.ndarray) -> float:
    gray = np.asarray(rgb, dtype=np.float32).mean(axis=-1)
    dx = np.abs(np.diff(gray, axis=1)).mean()
    dy = np.abs(np.diff(gray, axis=0)).mean()
    return float(dx + dy + 0.25 * gray.std())


def downsample_mean(image: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h % grid_h == 0 and w % grid_w == 0:
        return image.reshape(grid_h, h // grid_h, grid_w, w // grid_w).mean(axis=(1, 3))
    ys = np.linspace(0, h, grid_h + 1).round().astype(int)
    xs = np.linspace(0, w, grid_w + 1).round().astype(int)
    out = np.zeros((grid_h, grid_w), dtype=np.float32)
    for i in range(grid_h):
        for j in range(grid_w):
            patch = image[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            out[i, j] = float(patch.mean()) if patch.size else 0.0
    return out


def pca_response(tokens: torch.Tensor, grid_size: tuple[int, int], sign_reference: np.ndarray) -> np.ndarray:
    x = tokens[0].detach().float().cpu().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        score = x @ vh[0]
    except np.linalg.LinAlgError:
        score = x[:, 0]
    response = score.reshape(grid_size)
    ref = sign_reference.reshape(-1)
    val = response.reshape(-1)
    if np.std(ref) > 1.0e-6 and np.std(val) > 1.0e-6:
        corr = np.corrcoef(ref, val)[0, 1]
        if np.isfinite(corr) and corr < 0:
            response = -response
    return robust01(response, lower=5.0, upper=95.0)


def extract_block_update_responses(model, batch) -> tuple[tuple[int, int], list[np.ndarray]]:
    """Return per-block patch-token update magnitude maps.

    Direct token norms are uninformative after LayerNorm, and per-layer PCA can be
    dominated by a stable scene brightness axis.  For interpretability we instead
    visualize how strongly each transformer block changes each spatial token.
    """

    patch_size = model.config.encoder.local_patch_size
    height, width = int(batch.values.shape[-2]), int(batch.values.shape[-1])
    grid_size = (height // patch_size, width // patch_size)
    routing = model._build_routing(batch)
    encoded = model.tokenizer(batch.values, routing)
    sequence, (grid_h, grid_w) = model.backbone._prepare_token_sequence(encoded.local_tokens, grid_size, masks=None)
    rope = model.backbone.transformer.rope_embed(H=grid_h, W=grid_w) if model.backbone.transformer.rope_embed is not None else None

    responses: list[np.ndarray] = []
    x = sequence
    _, previous_patch = model.backbone._normalize_sequence(x)
    for block in model.backbone.transformer.blocks:
        x = block(x, rope)
        _, patch = model.backbone._normalize_sequence(x)
        delta = (patch.float() - previous_patch.float()).norm(dim=-1)[0]
        response = delta.detach().cpu().numpy().reshape(grid_size)
        responses.append(robust01(response, lower=2.0, upper=98.0))
        previous_patch = patch
    return grid_size, responses


def extract_feature_norm_responses(model, batch) -> tuple[tuple[int, int], list[np.ndarray]]:
    """Return DINOv3-style per-block feature-norm response maps.

    This mirrors the reference DINOv3 visualization: after each Transformer block,
    discard the cls token and the four storage tokens, compute the L2 norm of each
    patch token, and reshape the response back to the patch grid.
    """

    patch_size = model.config.encoder.local_patch_size
    height, width = int(batch.values.shape[-2]), int(batch.values.shape[-1])
    grid_size = (height // patch_size, width // patch_size)
    routing = model._build_routing(batch)
    encoded = model.tokenizer(batch.values, routing)
    sequence, (grid_h, grid_w) = model.backbone._prepare_token_sequence(encoded.local_tokens, grid_size, masks=None)
    rope = model.backbone.transformer.rope_embed(H=grid_h, W=grid_w) if model.backbone.transformer.rope_embed is not None else None

    responses: list[np.ndarray] = []
    x = sequence
    offset = 1 + int(model.backbone.n_storage_tokens)
    for block in model.backbone.transformer.blocks:
        x = block(x, rope)
        patch = x[:, offset:].float()
        norm = patch.norm(dim=-1)[0]
        response = norm.detach().cpu().numpy().reshape(grid_size)
        responses.append(robust01(response, lower=2.0, upper=98.0))
    return grid_size, responses


def _metadata_path_for_values(values_path: Path) -> Path:
    text = str(values_path)
    if "/values/" in text:
        return Path(text.replace("/values/", "/metadata/")).with_suffix(".npz")
    return values_path.parent.parent / "metadata" / values_path.with_suffix(".npz").name


def _assemble_tile_from_indices(values: np.ndarray, indices: list[int], tile_patches: int) -> np.ndarray:
    first = np.asarray(values[indices[0]], dtype=np.float32)
    channels, patch_h, patch_w = first.shape
    tile = np.zeros((channels, tile_patches * patch_h, tile_patches * patch_w), dtype=np.float32)
    for linear, sample_index in enumerate(indices):
        row = linear // tile_patches
        col = linear % tile_patches
        tile[:, row * patch_h : (row + 1) * patch_h, col * patch_w : (col + 1) * patch_w] = np.asarray(
            values[sample_index], dtype=np.float32
        )
    return tile


def select_dz1_tile(dataset, sensor: str, scan_shards: int, tile_patches: int, max_windows_per_shard: int) -> dict[str, Any]:
    candidates = [idx for idx, shard in enumerate(dataset.shards) if shard.sensor == sensor]
    if not candidates:
        raise ValueError(f"No shards found for sensor={sensor!r}.")
    if scan_shards > 0:
        candidates = candidates[:scan_shards]

    best: dict[str, Any] | None = None
    rng = np.random.default_rng(20260711)
    for shard_index in candidates:
        shard = dataset.shards[shard_index]
        meta_path = _metadata_path_for_values(shard.values_path)
        if not meta_path.exists():
            continue
        values = np.load(shard.values_path, mmap_mode="r")
        with np.load(meta_path) as meta:
            products = np.asarray(meta["source_product_index"])
            rows = np.asarray(meta["source_row"])
            cols = np.asarray(meta["source_col"])
        by_product: dict[int, dict[tuple[int, int], int]] = {}
        for sample_index, (product, row, col) in enumerate(zip(products, rows, cols)):
            by_product.setdefault(int(product), {})[(int(row), int(col))] = int(sample_index)

        for product, pos_to_index in by_product.items():
            unique_rows = sorted({pos[0] for pos in pos_to_index})
            unique_cols = sorted({pos[1] for pos in pos_to_index})
            windows: list[tuple[int, int]] = []
            for ri in range(0, max(0, len(unique_rows) - tile_patches + 1)):
                for ci in range(0, max(0, len(unique_cols) - tile_patches + 1)):
                    block_rows = unique_rows[ri : ri + tile_patches]
                    block_cols = unique_cols[ci : ci + tile_patches]
                    if all((r, c) in pos_to_index for r in block_rows for c in block_cols):
                        windows.append((ri, ci))
            if len(windows) > max_windows_per_shard:
                keep = rng.choice(len(windows), size=max_windows_per_shard, replace=False)
                windows = [windows[int(i)] for i in keep]
            for ri, ci in windows:
                block_rows = unique_rows[ri : ri + tile_patches]
                block_cols = unique_cols[ci : ci + tile_patches]
                indices = [pos_to_index[(r, c)] for r in block_rows for c in block_cols]
                tile_cube = _assemble_tile_from_indices(values, indices, tile_patches)
                rgb = pseudo_rgb(tile_cube, shard.wavelengths.numpy())
                score = edge_score_from_rgb(rgb)
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "shard_index": int(shard_index),
                        "sample_indices": [int(i) for i in indices],
                        "global_indices": [int(dataset.sample_offsets[shard_index] + i) for i in indices],
                        "values_path": str(shard.values_path),
                        "metadata_path": str(meta_path),
                        "source_product_index": int(product),
                        "source_rows": [int(r) for r in block_rows],
                        "source_cols": [int(c) for c in block_cols],
                        "sensor": shard.sensor,
                        "level": shard.level,
                        "gsd_m": float(shard.gsd_m),
                        "sample_shape": [int(shard.sample_shape[0]), int(tile_cube.shape[1]), int(tile_cube.shape[2])],
                        "rgb": rgb,
                        "cube": tile_cube,
                        "wavelengths": shard.wavelengths.numpy().astype(float).tolist(),
                        "fwhm": None if shard.fwhm is None else shard.fwhm.numpy().astype(float).tolist(),
                        "tile_patches": int(tile_patches),
                    }
    if best is None:
        raise RuntimeError(f"No contiguous {tile_patches}x{tile_patches} tile found for sensor={sensor!r}.")
    return best


def select_dz1_tile_from_json(selection_json: Path, selection_rank: int) -> dict[str, Any]:
    data = json.loads(selection_json.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = data.get("candidates", data.get("items", []))
    else:
        records = data
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"No candidate records found in {selection_json}")
    selected_record: dict[str, Any] | None = None
    for record in records:
        if int(record.get("rank", -1)) == int(selection_rank):
            selected_record = dict(record)
            break
    if selected_record is None:
        index = max(0, min(len(records) - 1, int(selection_rank) - 1))
        selected_record = dict(records[index])

    values_path = Path(selected_record["values_path"])
    values = np.load(values_path, mmap_mode="r")
    tile_patches = int(selected_record.get("tile_patches", 4))
    indices = [int(i) for i in selected_record["sample_indices"]]
    tile_cube = _assemble_tile_from_indices(values, indices, tile_patches)
    wavelengths = np.asarray(selected_record["wavelengths"], dtype=np.float32)
    rgb = pseudo_rgb(tile_cube, wavelengths)
    selected_record["rgb"] = rgb
    selected_record["cube"] = tile_cube
    selected_record["wavelengths"] = wavelengths.astype(float).tolist()
    selected_record["fwhm"] = (
        None
        if selected_record.get("fwhm") is None
        else np.asarray(selected_record.get("fwhm"), dtype=np.float32).astype(float).tolist()
    )
    selected_record["sample_shape"] = [
        int(tile_cube.shape[0]),
        int(tile_cube.shape[1]),
        int(tile_cube.shape[2]),
    ]
    selected_record["selection_json"] = str(selection_json)
    selected_record["selection_rank"] = int(selection_rank)
    return selected_record


def select_sample(dataset, sensor: str, scan_shards: int, scan_samples_per_shard: int) -> dict[str, Any]:
    candidates = [idx for idx, shard in enumerate(dataset.shards) if shard.sensor == sensor]
    if not candidates:
        raise ValueError(f"No shards found for sensor={sensor!r}.")
    if scan_shards > 0:
        candidates = candidates[:scan_shards]

    best: dict[str, Any] | None = None
    for shard_index in candidates:
        shard = dataset.shards[shard_index]
        values = np.load(shard.values_path, mmap_mode="r")
        n = int(shard.count)
        sample_indices = np.linspace(0, n - 1, min(scan_samples_per_shard, n), dtype=int)
        for sample_index in sample_indices:
            cube = np.asarray(values[sample_index], dtype=np.float32)
            rgb = pseudo_rgb(cube, shard.wavelengths.numpy())
            score = edge_score_from_rgb(rgb)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "shard_index": int(shard_index),
                    "sample_index": int(sample_index),
                    "global_index": int(dataset.sample_offsets[shard_index] + sample_index),
                    "values_path": str(shard.values_path),
                    "sensor": shard.sensor,
                    "level": shard.level,
                    "gsd_m": float(shard.gsd_m),
                    "sample_shape": list(map(int, shard.sample_shape)),
                    "rgb": rgb,
                    "cube": cube,
                    "wavelengths": shard.wavelengths.numpy().astype(float).tolist(),
                    "fwhm": None if shard.fwhm is None else shard.fwhm.numpy().astype(float).tolist(),
                    "tile_patches": 1,
                }
    if best is None:
        raise RuntimeError("Sample selection failed.")
    return best


def load_model(args: argparse.Namespace, device: torch.device):
    from aware_open.models.foundation import MetadataConditionedFoundationModel
    from aware_open.spectral_prototypes import load_prototype_bank

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    eo1_bank = load_prototype_bank(args.prototype_bank_eo1)
    gf5_bank = load_prototype_bank(args.prototype_bank_gf5)
    dz1_vnir_bank = load_prototype_bank(args.prototype_bank_dz1_vnir)
    dz1_swir_bank = load_prototype_bank(args.prototype_bank_dz1_swir)
    config = build_config_from_checkpoint_args(checkpoint_args, int(eo1_bank.centers.shape[0]))

    model = MetadataConditionedFoundationModel(
        config,
        eo1_bank_centers=eo1_bank.centers,
        gf5_bank_centers=gf5_bank.centers,
        eo1_bank_wavelengths=eo1_bank.wavelengths_nm,
        gf5_bank_wavelengths=gf5_bank.wavelengths_nm,
        extra_prototype_banks={
            "dz1_vnir": (dz1_vnir_bank.centers, dz1_vnir_bank.wavelengths_nm),
            "dz1_swir": (dz1_swir_bank.centers, dz1_swir_bank.wavelengths_nm),
        },
    )
    result = model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint load result: {result}")
    model.backbone.out_indices = tuple(range(config.encoder.token_encoder_depth))
    model.eval().to(device)
    return model, checkpoint_args


def save_figure(
    output_dir: Path,
    rgb: np.ndarray,
    responses: list[np.ndarray],
    depth: int,
    input_label: str,
    *,
    response_method: str,
    output_stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "font.weight": "bold",
            "axes.titleweight": "bold",
        }
    )

    cols = 4
    rows = int(math.ceil(depth / cols))
    fig = plt.figure(figsize=(9.8, 6.25), constrained_layout=False)
    gs = GridSpec(rows, cols + 1, figure=fig, width_ratios=[1.05, 1, 1, 1, 1], wspace=0.12, hspace=0.18)

    ax0 = fig.add_subplot(gs[:, 0])
    ax0.imshow(rgb)
    ax0.set_title(input_label, pad=4)
    ax0.set_xticks([])
    ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_linewidth(1.1)
        spine.set_edgecolor("#1f2d50")

    for idx, response in enumerate(responses):
        r, c = divmod(idx, cols)
        ax = fig.add_subplot(gs[r, c + 1])
        if response_method == "feature_norm":
            height, width = rgb.shape[:2]
            ax.imshow(rgb, alpha=0.42)
            ax.imshow(
                response,
                cmap="jet",
                alpha=0.62,
                interpolation="bilinear",
                extent=(0, width, height, 0),
                vmin=0.0,
                vmax=1.0,
            )
        else:
            ax.imshow(response, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title(f"Block {idx + 1}", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#4d4d4d")

    fig.text(
        0.5,
        0.018,
        (
            "Layer response = patch-token L2 feature norm after each Transformer block; heatmaps are overlaid on pseudo-RGB."
            if response_method == "feature_norm"
            else "Layer response = patch-token update magnitude between adjacent Transformer blocks; each panel robust-normalized."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="normal",
    )
    fig.savefig(output_dir / f"{output_stem}.png", dpi=420, bbox_inches="tight")
    fig.savefig(output_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    _add_project_to_path(args.project_root)
    os.environ.setdefault("MPLBACKEND", "Agg")

    from aware_open.data.pretrain import (
        ModalitySample,
        SampleMetadata,
        Sigma450KShardDataset,
        _normalize_values,
        collate_modality_samples,
        move_batch_to_device,
    )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    dataset = Sigma450KShardDataset(
        args.shard_root,
        crop_size=args.crop_size,
        random_crop=False,
        normalization=args.normalization,
        manifest_path=args.manifest,
    )
    if args.selection_json is not None:
        selected = select_dz1_tile_from_json(args.selection_json, args.selection_rank)
        cube = _normalize_values(selected["cube"], args.normalization)
        height, width = cube.shape[-2:]
        sample = ModalitySample(
            values=torch.from_numpy(cube.astype(np.float32, copy=False)),
            wavelengths=torch.tensor(selected["wavelengths"], dtype=torch.float32),
            fwhm=None if selected["fwhm"] is None else torch.tensor(selected["fwhm"], dtype=torch.float32),
            valid_mask=None,
            metadata=SampleMetadata(
                modality="space",
                sensor=selected["sensor"],
                level=selected["level"],
                gsd=float(selected["gsd_m"]),
                footprint_area=float(height * width) * (float(selected["gsd_m"]) ** 2),
            ),
        )
    elif args.tile_patches > 1 and args.sensor.startswith("dz1_"):
        selected = select_dz1_tile(dataset, args.sensor, args.scan_shards, args.tile_patches, args.max_windows_per_shard)
        cube = _normalize_values(selected["cube"], args.normalization)
        height, width = cube.shape[-2:]
        sample = ModalitySample(
            values=torch.from_numpy(cube.astype(np.float32, copy=False)),
            wavelengths=torch.tensor(selected["wavelengths"], dtype=torch.float32),
            fwhm=None if selected["fwhm"] is None else torch.tensor(selected["fwhm"], dtype=torch.float32),
            valid_mask=None,
            metadata=SampleMetadata(
                modality="space",
                sensor=selected["sensor"],
                level=selected["level"],
                gsd=float(selected["gsd_m"]),
                footprint_area=float(height * width) * (float(selected["gsd_m"]) ** 2),
            ),
        )
    else:
        selected = select_sample(dataset, args.sensor, args.scan_shards, args.scan_samples_per_shard)
        sample = dataset[selected["global_index"]]
    batch = move_batch_to_device(collate_modality_samples([sample]), device)

    model, checkpoint_args = load_model(args, device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if args.response_method == "feature_norm":
                grid_size, responses = extract_feature_norm_responses(model, batch)
            else:
                grid_size, responses = extract_block_update_responses(model, batch)

    depth = int(model.config.encoder.token_encoder_depth)
    if len(responses) != depth:
        raise RuntimeError(f"Expected {depth} block response maps, got {len(responses)}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_label = "Pseudo-RGB\nDZ1-VNIR tile" if int(selected.get("tile_patches", 1)) > 1 else "Pseudo-RGB\nDZ1-VNIR input"
    output_stem = "Figure_feature_norm_dz1_vnir" if args.response_method == "feature_norm" else "Figure_feature_response_dz1_vnir"
    arrays_name = "feature_norm_arrays.npz" if args.response_method == "feature_norm" else "feature_response_arrays.npz"
    manifest_name = "feature_norm_manifest.json" if args.response_method == "feature_norm" else "feature_response_manifest.json"
    save_figure(
        args.output_dir,
        selected["rgb"],
        responses,
        depth,
        input_label,
        response_method=args.response_method,
        output_stem=output_stem,
    )
    np.savez_compressed(
        args.output_dir / arrays_name,
        pseudo_rgb=selected["rgb"].astype(np.float32),
        responses=np.stack(responses).astype(np.float32),
    )
    response_description = (
        "DINOv3-style patch-token L2 feature norm after each Transformer block, discarding cls+4 storage tokens, robust-normalized per panel"
        if args.response_method == "feature_norm"
        else "per-block patch-token update magnitude after each Transformer block, robust-normalized per panel"
    )
    manifest = {
        "status": "ok",
        "checkpoint": str(args.checkpoint),
        "token_encoder_depth": depth,
        "grid_size": list(grid_size),
        "response_method": response_description,
        "selected_sample": {k: v for k, v in selected.items() if k not in {"rgb", "cube"}},
        "checkpoint_args_subset": {
            "token_encoder_depth": checkpoint_args.get("token_encoder_depth"),
            "model_dim": checkpoint_args.get("model_dim"),
            "local_patch_size": checkpoint_args.get("local_patch_size"),
            "metadata_ablation": checkpoint_args.get("metadata_ablation"),
        },
        "outputs": {
            "png": str(args.output_dir / f"{output_stem}.png"),
            "pdf": str(args.output_dir / f"{output_stem}.pdf"),
            "arrays": str(args.output_dir / arrays_name),
        },
    }
    (args.output_dir / manifest_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
