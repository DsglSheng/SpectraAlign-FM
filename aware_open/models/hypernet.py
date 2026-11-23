from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ContextFiLM, CrossAttentionBlock, SelfAttentionBlock


@dataclass
class BandRouting:
    band_tokens: torch.Tensor
    spectral_weights: torch.Tensor
    spatial_weights: torch.Tensor
    assignments: torch.Tensor
    prototype_tokens: torch.Tensor
    sample_context: torch.Tensor
    valid_mask: torch.Tensor | None = None


class SharedBandHyperNet(nn.Module):
    def __init__(
        self,
        band_dim: int,
        context_dim: int,
        spectral_dim: int,
        spatial_dim: int,
        num_prototypes: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.band_conditioner = ContextFiLM(context_dim, band_dim)
        self.band_refine = nn.Sequential(
            nn.LayerNorm(band_dim),
            nn.Linear(band_dim, band_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(band_dim * 2, band_dim),
        )

        self.prototype_queries = nn.Parameter(torch.randn(num_prototypes, band_dim) * 0.02)
        self.context_to_query = nn.Linear(context_dim, band_dim)
        self.prototype_cross = CrossAttentionBlock(band_dim, num_heads, dropout=dropout)
        self.prototype_self = nn.ModuleList(
            [SelfAttentionBlock(band_dim, num_heads, dropout=dropout) for _ in range(2)]
        )

        self.spectral_head = nn.Sequential(
            nn.LayerNorm(band_dim),
            nn.Linear(band_dim, spectral_dim),
        )
        self.spatial_head = nn.Sequential(
            nn.LayerNorm(band_dim),
            nn.Linear(band_dim, spatial_dim),
        )
        self.assignment_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        band_tokens: torch.Tensor,
        sample_context: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> BandRouting:
        conditioned = self.band_conditioner(band_tokens, sample_context)
        conditioned = conditioned + self.band_refine(conditioned)
        mask = None if valid_mask is None else valid_mask.bool()
        if mask is not None:
            conditioned = conditioned.masked_fill(~mask.unsqueeze(-1), 0.0)

        queries = self.prototype_queries.unsqueeze(0).expand(conditioned.shape[0], -1, -1)
        queries = queries + self.context_to_query(sample_context).unsqueeze(1)
        prototype_tokens = self.prototype_cross(
            queries,
            conditioned,
            context_padding_mask=None if mask is None else ~mask,
        )
        for block in self.prototype_self:
            prototype_tokens = block(prototype_tokens)

        normalized_bands = F.normalize(conditioned, dim=-1)
        normalized_proto = F.normalize(prototype_tokens, dim=-1)
        logits = torch.einsum("bcd,bkd->bck", normalized_bands, normalized_proto)
        scale = F.softplus(self.assignment_scale) + 1.0e-3
        assignments = torch.softmax(logits / scale, dim=-1)
        spectral_weights = self.spectral_head(conditioned)
        spatial_weights = self.spatial_head(conditioned)
        if mask is not None:
            spectral_weights = spectral_weights.masked_fill(~mask.unsqueeze(-1), 0.0)
            spatial_weights = spatial_weights.masked_fill(~mask.unsqueeze(-1), 0.0)
            assignments = assignments.masked_fill(~mask.unsqueeze(-1), 0.0)

        return BandRouting(
            band_tokens=conditioned,
            spectral_weights=spectral_weights,
            spatial_weights=spatial_weights,
            assignments=assignments,
            prototype_tokens=prototype_tokens,
            sample_context=sample_context,
            valid_mask=mask,
        )
