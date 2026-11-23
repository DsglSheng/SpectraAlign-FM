from __future__ import annotations

import torch


def radial_frequency_masks(
    height: int,
    width: int,
    *,
    low_cutoff_ratio: float,
    mid_cutoff_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 < low_cutoff_ratio < mid_cutoff_ratio < 1.0:
        raise ValueError(
            f"Expected 0 < low_cutoff_ratio < mid_cutoff_ratio < 1, got "
            f"{low_cutoff_ratio}, {mid_cutoff_ratio}."
        )

    fy = torch.fft.fftfreq(height, device=device).view(height, 1)
    fx = torch.fft.fftfreq(width, device=device).view(1, width)
    radius = torch.sqrt(fy.square() + fx.square())
    max_radius = float(radius.max().item()) or 1.0
    normalized = radius / max_radius

    low = normalized <= low_cutoff_ratio
    mid = (normalized > low_cutoff_ratio) & (normalized <= mid_cutoff_ratio)
    high = normalized > mid_cutoff_ratio
    return low, mid, high


def decompose_frequency_targets(
    cube: torch.Tensor,
    *,
    low_cutoff_ratio: float,
    mid_cutoff_ratio: float,
) -> dict[str, torch.Tensor]:
    if cube.dim() != 4:
        raise ValueError(f"Expected cube tensor (B, C, H, W), got {tuple(cube.shape)}.")

    _, _, height, width = cube.shape
    low_mask, mid_mask, high_mask = radial_frequency_masks(
        height,
        width,
        low_cutoff_ratio=low_cutoff_ratio,
        mid_cutoff_ratio=mid_cutoff_ratio,
        device=cube.device,
    )
    fft = torch.fft.fft2(cube.float(), dim=(-2, -1))

    def _ifft(mask: torch.Tensor) -> torch.Tensor:
        masked = fft * mask.view(1, 1, height, width)
        return torch.fft.ifft2(masked, dim=(-2, -1)).real

    low = _ifft(low_mask)
    mid = _ifft(mid_mask)
    high = _ifft(high_mask)
    return {
        "low": low,
        "low_mid": low + mid,
        "full": low + mid + high,
    }
