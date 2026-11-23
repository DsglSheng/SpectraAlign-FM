from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .common import ContextFiLM2d, SelfAttentionBlock
from .hypernet import BandRouting


@dataclass
class TokenizedModality:
    tokens: torch.Tensor
    metadata_token: torch.Tensor
    prototype_tokens: torch.Tensor
    local_tokens: torch.Tensor


class AirSpaceTokenizer2D(nn.Module):
    def __init__(
        self,
        band_dim: int,
        spectral_dim: int,
        spatial_dim: int,
        model_dim: int,
        context_dim: int,
        patch_size: int,
        num_heads: int,
        prototype_depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.band_summary = nn.Linear(band_dim, spectral_dim)
        self.prototype_residual = nn.Linear(band_dim, spectral_dim)
        self.prototype_blocks = nn.ModuleList(
            [SelfAttentionBlock(spectral_dim, num_heads, dropout=dropout) for _ in range(prototype_depth)]
        )

        self.prototype_to_spectral = nn.Linear(spectral_dim, spectral_dim * 2)
        self.dynamic_reduce_bias = nn.Parameter(torch.zeros(spatial_dim))

        self.spatial_multiscale = nn.Sequential(
            nn.Conv2d(spatial_dim, spatial_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(spatial_dim, spatial_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(spatial_dim, spatial_dim, kernel_size=5, padding=2),
        )

        self.spec_to_spatial = nn.Conv2d(spectral_dim, spatial_dim, kernel_size=1)
        self.spatial_to_spec = nn.Conv2d(spatial_dim, spectral_dim, kernel_size=1)
        self.fusion_film = ContextFiLM2d(context_dim, spectral_dim + spatial_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(spectral_dim + spatial_dim, model_dim, kernel_size=1),
            nn.GroupNorm(32, model_dim),
            nn.GELU(),
            nn.Conv2d(model_dim, model_dim, kernel_size=1),
        )

        self.patch_proj = nn.Conv2d(model_dim, model_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, values: torch.Tensor, routing: BandRouting) -> TokenizedModality:
        if values.dim() != 4:
            raise ValueError(f"Expected 2D modality input of shape (B, C, H, W), got {tuple(values.shape)}.")

        spectral_map = torch.einsum("bchw,bcd->bdhw", values, routing.spectral_weights)

        signature = values.mean(dim=(2, 3))
        band_summary = signature.unsqueeze(-1) * self.band_summary(routing.band_tokens)
        prototype_tokens = torch.einsum("bcd,bck->bkd", band_summary, routing.assignments)
        prototype_tokens = prototype_tokens + self.prototype_residual(routing.prototype_tokens)
        for block in self.prototype_blocks:
            prototype_tokens = block(prototype_tokens)

        proto_scale, proto_shift = self.prototype_to_spectral(prototype_tokens.mean(dim=1)).chunk(2, dim=-1)
        spectral_map = spectral_map * (1.0 + proto_scale[:, :, None, None]) + proto_shift[:, :, None, None]

        spatial_map = torch.einsum("bchw,bcd->bdhw", values, routing.spatial_weights)
        spatial_map = spatial_map + self.dynamic_reduce_bias[None, :, None, None]
        spatial_map = self.spatial_multiscale(spatial_map)

        enhanced_spatial = spatial_map * torch.sigmoid(self.spec_to_spatial(spectral_map))
        enhanced_spectral = spectral_map * torch.sigmoid(self.spatial_to_spec(spatial_map))

        fused = torch.cat([enhanced_spectral, enhanced_spatial], dim=1)
        fused = self.fusion_film(fused, routing.sample_context)
        fused = self.fusion(fused)

        local = self.patch_proj(fused).flatten(2).transpose(1, 2)
        empty = local.new_empty(local.shape[0], 0, local.shape[-1])
        return TokenizedModality(tokens=local, metadata_token=empty, prototype_tokens=empty, local_tokens=local)
