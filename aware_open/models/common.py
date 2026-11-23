from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        context_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        query = query + attn_out
        query = query + self.ffn(self.ffn_norm(query))
        return query


class ContextFiLM(nn.Module):
    def __init__(self, context_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.scale = nn.Linear(context_dim, feature_dim)
        self.shift = nn.Linear(context_dim, feature_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.scale(context).unsqueeze(1)
        shift = self.shift(context).unsqueeze(1)
        return x * scale + shift


class ContextFiLM2d(nn.Module):
    def __init__(self, context_dim: int, channels: int) -> None:
        super().__init__()
        self.scale = nn.Linear(context_dim, channels)
        self.shift = nn.Linear(context_dim, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.scale(context)[:, :, None, None]
        shift = self.shift(context)[:, :, None, None]
        return x * scale + shift
