from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


GOOD_QUALITY_CODE = 0


@dataclass(frozen=True)
class PrototypeBank:
    sensor: str
    wavelengths_nm: torch.Tensor
    fwhm_nm: torch.Tensor
    centers: torch.Tensor
    cluster_counts: torch.Tensor
    support_count: torch.Tensor
    avg_cosine_distance: float
    min_coverage: float
    source_sample_count: int


def good_band_mask_from_quality(quality_codes: np.ndarray | torch.Tensor | None) -> np.ndarray | torch.Tensor | None:
    if quality_codes is None:
        return None
    if isinstance(quality_codes, torch.Tensor):
        return quality_codes.to(dtype=torch.long) == GOOD_QUALITY_CODE
    return np.asarray(quality_codes, dtype=np.int64) == GOOD_QUALITY_CODE


def gaussian_srf_matrix(
    source_wavelengths_nm: torch.Tensor,
    target_wavelengths_nm: torch.Tensor,
    target_fwhm_nm: torch.Tensor,
) -> torch.Tensor:
    source = source_wavelengths_nm.float().view(1, -1)
    target = target_wavelengths_nm.float().view(-1, 1)
    fwhm = target_fwhm_nm.float().view(-1, 1).clamp_min(1.0e-3)
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    distance = (source - target) / sigma
    weights = torch.exp(-0.5 * distance.square())
    return weights


def estimate_median_wavelength_step(
    wavelengths_nm: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    wavelengths = wavelengths_nm.float().view(-1)
    finite_mask = torch.isfinite(wavelengths)
    if valid_mask is not None:
        finite_mask = finite_mask & valid_mask.bool().view(-1)
    usable = wavelengths[finite_mask]
    if usable.numel() < 2:
        return float("inf")
    usable = torch.sort(usable).values
    diffs = torch.diff(usable)
    diffs = diffs[diffs > 0]
    if diffs.numel() == 0:
        return float("inf")
    return float(torch.median(diffs).item())


def build_auxiliary_wavelength_grid(
    target_wavelengths_nm: torch.Tensor,
    target_fwhm_nm: torch.Tensor,
    *,
    step_nm: float,
    margin_scale: float = 3.0,
) -> torch.Tensor:
    if step_nm <= 0:
        raise ValueError(f"step_nm must be positive, got {step_nm}.")
    target = target_wavelengths_nm.float().view(-1)
    fwhm = target_fwhm_nm.float().view(-1)
    start = math.floor(float((target - margin_scale * fwhm).min().item()))
    end = math.ceil(float((target + margin_scale * fwhm).max().item()))
    count = int(math.floor((end - start) / step_nm)) + 1
    grid = start + torch.arange(count, dtype=torch.float32) * float(step_nm)
    if grid[-1] < end:
        grid = torch.cat([grid, torch.tensor([float(end)], dtype=torch.float32)], dim=0)
    return grid


def _sorted_unique_spectrum(
    reflectance: torch.Tensor,
    wavelengths_nm: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = reflectance.float().view(-1)
    wavelengths = wavelengths_nm.float().view(-1)
    if values.shape[0] != wavelengths.shape[0]:
        raise ValueError("reflectance and wavelengths must have the same length.")

    finite_mask = torch.isfinite(values) & torch.isfinite(wavelengths)
    if valid_mask is not None:
        finite_mask = finite_mask & valid_mask.bool().view(-1)

    source_values = values[finite_mask].cpu().numpy().astype(np.float32, copy=False)
    source_wavelengths = wavelengths[finite_mask].cpu().numpy().astype(np.float32, copy=False)
    if source_wavelengths.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    order = np.argsort(source_wavelengths)
    source_wavelengths = source_wavelengths[order]
    source_values = source_values[order]

    unique_wavelengths, inverse = np.unique(source_wavelengths, return_inverse=True)
    if unique_wavelengths.size == source_wavelengths.size:
        return source_values, source_wavelengths

    summed = np.zeros(unique_wavelengths.size, dtype=np.float32)
    counts = np.zeros(unique_wavelengths.size, dtype=np.int64)
    np.add.at(summed, inverse, source_values)
    np.add.at(counts, inverse, 1)
    averaged = summed / np.maximum(counts, 1)
    return averaged.astype(np.float32, copy=False), unique_wavelengths.astype(np.float32, copy=False)


def resample_spectrum_via_auxiliary_grid(
    reflectance: torch.Tensor,
    wavelengths_nm: torch.Tensor,
    *,
    auxiliary_wavelengths_nm: torch.Tensor,
    target_srf_weights: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    max_interpolation_gap_factor: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if max_interpolation_gap_factor <= 0:
        raise ValueError(
            f"max_interpolation_gap_factor must be positive, got {max_interpolation_gap_factor}."
        )

    source_values, source_wavelengths = _sorted_unique_spectrum(
        reflectance,
        wavelengths_nm,
        valid_mask=valid_mask,
    )
    if source_wavelengths.size < 2:
        zeros = target_srf_weights.new_zeros(target_srf_weights.shape[0])
        return zeros, zeros.clone(), float("inf")

    source_step_nm = float(np.median(np.diff(source_wavelengths).astype(np.float64)))
    aux_grid = auxiliary_wavelengths_nm.float().cpu().numpy().astype(np.float32, copy=False)
    interpolated = np.interp(aux_grid, source_wavelengths, source_values, left=np.nan, right=np.nan).astype(
        np.float32,
        copy=False,
    )

    left_index = np.searchsorted(source_wavelengths, aux_grid, side="right") - 1
    left_index = np.clip(left_index, 0, source_wavelengths.size - 1)
    right_index = np.clip(left_index + 1, 0, source_wavelengths.size - 1)
    local_gap = source_wavelengths[right_index] - source_wavelengths[left_index]
    aux_step_nm = float(aux_grid[1] - aux_grid[0]) if aux_grid.size > 1 else source_step_nm
    max_gap_nm = max(float(max_interpolation_gap_factor) * source_step_nm, aux_step_nm * 2.0)
    aux_valid = np.isfinite(interpolated)
    aux_valid &= aux_grid >= float(source_wavelengths[0])
    aux_valid &= aux_grid <= float(source_wavelengths[-1])
    aux_valid &= local_gap <= max_gap_nm + 1.0e-6

    interpolated_tensor = torch.from_numpy(np.nan_to_num(interpolated, nan=0.0, posinf=0.0, neginf=0.0)).float()
    aux_valid_tensor = torch.from_numpy(aux_valid.astype(np.float32, copy=False)).float()
    total_support = target_srf_weights.sum(dim=1).clamp_min(1.0e-12)
    effective_weights = target_srf_weights * aux_valid_tensor.view(1, -1)
    valid_support = effective_weights.sum(dim=1)
    coverage = valid_support / total_support
    normalized = effective_weights / valid_support.clamp_min(1.0e-12).view(-1, 1)
    resampled = normalized @ interpolated_tensor
    return resampled, coverage, source_step_nm


def resample_spectrum_to_sensor(
    reflectance: torch.Tensor,
    wavelengths_nm: torch.Tensor,
    *,
    target_wavelengths_nm: torch.Tensor,
    target_fwhm_nm: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = reflectance.float().view(-1)
    source_wavelengths = wavelengths_nm.float().view(-1)
    if values.shape[0] != source_wavelengths.shape[0]:
        raise ValueError("reflectance and wavelengths must have the same length.")

    finite_mask = torch.isfinite(values) & torch.isfinite(source_wavelengths)
    if valid_mask is not None:
        finite_mask = finite_mask & valid_mask.bool().view(-1)

    raw_weights = gaussian_srf_matrix(source_wavelengths, target_wavelengths_nm, target_fwhm_nm)
    total_support = raw_weights.sum(dim=1).clamp_min(1.0e-12)
    effective_weights = raw_weights * finite_mask.float().view(1, -1)
    valid_support = effective_weights.sum(dim=1)
    coverage = valid_support / total_support

    normalized = effective_weights / valid_support.clamp_min(1.0e-12).view(-1, 1)
    resampled = normalized @ torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return resampled, coverage


def shape_normalize_spectra(values: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    if values.dim() == 1:
        values = values.unsqueeze(0)
        squeeze = True
    elif values.dim() == 2:
        squeeze = False
    else:
        raise ValueError(f"Expected (C,) or (N, C), got {tuple(values.shape)}.")

    data = values.float()
    if valid_mask is None:
        mask = torch.ones_like(data, dtype=torch.bool)
    elif valid_mask.dim() == 1:
        mask = valid_mask.bool().view(1, -1).expand(data.shape[0], -1)
    elif valid_mask.dim() == 2:
        mask = valid_mask.bool()
    else:
        raise ValueError(f"Expected valid_mask rank 1 or 2, got {valid_mask.dim()}.")
    if mask.shape != data.shape:
        raise ValueError(f"valid_mask shape {tuple(mask.shape)} does not match values {tuple(data.shape)}.")

    safe = torch.where(mask, data, torch.zeros_like(data))
    counts = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = safe.sum(dim=-1, keepdim=True) / counts
    centered = torch.where(mask, data - mean, torch.zeros_like(data))
    variance = centered.square().sum(dim=-1, keepdim=True) / counts
    scale = variance.sqrt().clamp_min(1.0e-6)
    standardized = torch.where(mask, centered / scale, torch.zeros_like(data))
    norm = standardized.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(1.0e-6)
    normalized = torch.where(mask, standardized / norm, torch.zeros_like(data))
    return normalized.squeeze(0) if squeeze else normalized


def compute_patch_mean_spectra(values: torch.Tensor, patch_size: int) -> torch.Tensor:
    if values.dim() != 4:
        raise ValueError(f"Expected cube tensor (B, C, H, W), got {tuple(values.shape)}.")
    pooled = F.avg_pool2d(values, kernel_size=patch_size, stride=patch_size)
    return pooled.flatten(2).transpose(1, 2)


def prototype_teacher_distribution(
    patch_mean_spectra: torch.Tensor,
    *,
    bank_centers: torch.Tensor,
    temperature: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized_patch = shape_normalize_spectra(patch_mean_spectra, valid_mask=valid_mask)
    normalized_bank = F.normalize(bank_centers.float(), dim=-1)
    similarity = normalized_patch @ normalized_bank.transpose(0, 1)
    return torch.softmax(similarity / max(float(temperature), 1.0e-6), dim=-1)


def entropy_balance_loss(probabilities: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    mean_prob = probabilities.mean(dim=(0, 1)).clamp_min(eps)
    entropy = -(mean_prob * mean_prob.log()).sum()
    max_entropy = math.log(mean_prob.shape[0])
    return 1.0 - entropy / max(max_entropy, eps)


def _chunked_indices(length: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, length, chunk_size):
        yield start, min(length, start + chunk_size)


def mini_batch_spherical_kmeans(
    samples: torch.Tensor,
    *,
    num_clusters: int,
    num_iters: int = 25,
    batch_size: int = 4096,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if samples.dim() != 2:
        raise ValueError(f"Expected samples (N, D), got {tuple(samples.shape)}.")
    if samples.shape[0] < num_clusters:
        raise ValueError(
            f"num_clusters={num_clusters} requires at least as many samples, got {samples.shape[0]}."
        )

    normalized = F.normalize(samples.float(), dim=-1)
    generator = torch.Generator(device=normalized.device)
    generator.manual_seed(seed)
    init_indices = torch.randperm(normalized.shape[0], generator=generator, device=normalized.device)[:num_clusters]
    centers = normalized[init_indices].clone()

    for _ in range(num_iters):
        numerator = torch.zeros_like(centers)
        counts = torch.zeros(num_clusters, dtype=torch.long, device=normalized.device)
        permutation = torch.randperm(normalized.shape[0], generator=generator, device=normalized.device)
        for start, end in _chunked_indices(permutation.numel(), batch_size):
            batch = normalized[permutation[start:end]]
            assignments = (batch @ centers.transpose(0, 1)).argmax(dim=-1)
            numerator.index_add_(0, assignments, batch)
            counts.index_add_(0, assignments, torch.ones_like(assignments, dtype=torch.long))

        empty = counts == 0
        if empty.any():
            refill = torch.randperm(normalized.shape[0], generator=generator, device=normalized.device)[: int(empty.sum())]
            numerator[empty] = normalized[refill]
            counts[empty] = 1
        centers = F.normalize(numerator / counts.clamp_min(1).unsqueeze(-1), dim=-1)

    final_assignments = torch.empty(normalized.shape[0], dtype=torch.long, device=normalized.device)
    cosine_distance_sum = torch.zeros(1, dtype=torch.float32, device=normalized.device)
    for start, end in _chunked_indices(normalized.shape[0], batch_size):
        batch = normalized[start:end]
        similarity = batch @ centers.transpose(0, 1)
        assignments = similarity.argmax(dim=-1)
        final_assignments[start:end] = assignments
        cosine_distance_sum += (1.0 - similarity.gather(1, assignments.unsqueeze(1)).squeeze(1)).sum()

    cluster_counts = torch.bincount(final_assignments, minlength=num_clusters)
    avg_cosine_distance = float(cosine_distance_sum.item() / normalized.shape[0])
    return centers.cpu(), cluster_counts.cpu(), final_assignments.cpu()


def build_prototype_bank(
    *,
    sensor: str,
    resampled_spectra: torch.Tensor,
    target_wavelengths_nm: torch.Tensor,
    target_fwhm_nm: torch.Tensor,
    num_clusters: int,
    num_iters: int,
    batch_size: int,
    min_coverage: float,
    seed: int,
) -> PrototypeBank:
    centers, cluster_counts, assignments = mini_batch_spherical_kmeans(
        resampled_spectra,
        num_clusters=num_clusters,
        num_iters=num_iters,
        batch_size=batch_size,
        seed=seed,
    )
    support = torch.bincount(assignments, minlength=num_clusters)
    similarity = resampled_spectra.float() @ centers.float().transpose(0, 1)
    chosen = similarity.argmax(dim=-1)
    avg_cosine_distance = float((1.0 - similarity.gather(1, chosen.unsqueeze(1)).squeeze(1)).mean().item())
    return PrototypeBank(
        sensor=sensor,
        wavelengths_nm=target_wavelengths_nm.float().cpu(),
        fwhm_nm=target_fwhm_nm.float().cpu(),
        centers=centers.float().cpu(),
        cluster_counts=cluster_counts.long().cpu(),
        support_count=support.long().cpu(),
        avg_cosine_distance=avg_cosine_distance,
        min_coverage=float(min_coverage),
        source_sample_count=int(resampled_spectra.shape[0]),
    )


def save_prototype_bank(path: str | Path, bank: PrototypeBank, *, extra_metadata: dict[str, object] | None = None) -> None:
    payload = {
        "sensor": bank.sensor,
        "wavelengths_nm": bank.wavelengths_nm,
        "fwhm_nm": bank.fwhm_nm,
        "centers": bank.centers,
        "cluster_counts": bank.cluster_counts,
        "support_count": bank.support_count,
        "avg_cosine_distance": float(bank.avg_cosine_distance),
        "min_coverage": float(bank.min_coverage),
        "source_sample_count": int(bank.source_sample_count),
        "metadata": extra_metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_prototype_bank(path: str | Path) -> PrototypeBank:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return PrototypeBank(
        sensor=str(payload["sensor"]),
        wavelengths_nm=torch.as_tensor(payload["wavelengths_nm"]).float(),
        fwhm_nm=torch.as_tensor(payload["fwhm_nm"]).float(),
        centers=torch.as_tensor(payload["centers"]).float(),
        cluster_counts=torch.as_tensor(payload["cluster_counts"]).long(),
        support_count=torch.as_tensor(payload["support_count"]).long(),
        avg_cosine_distance=float(payload["avg_cosine_distance"]),
        min_coverage=float(payload["min_coverage"]),
        source_sample_count=int(payload["source_sample_count"]),
    )


def summarize_bank(bank: PrototypeBank) -> dict[str, object]:
    return {
        "sensor": bank.sensor,
        "prototype_count": int(bank.centers.shape[0]),
        "feature_dim": int(bank.centers.shape[1]),
        "source_sample_count": int(bank.source_sample_count),
        "avg_cosine_distance": float(bank.avg_cosine_distance),
        "min_coverage": float(bank.min_coverage),
    }
