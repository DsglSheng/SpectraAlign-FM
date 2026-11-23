from __future__ import annotations

from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

import torch
import torch.nn as nn

from dinov3.hub.backbones import Weights, _make_dinov3_vit


class TokenTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_indices: Sequence[int] = (2, 5, 8, 11),
        patch_size: int = 16,
        pretrained: bool = False,
        weights: str | None = None,
        drop_path_rate: float = 0.0,
        rope_dtype: str = "fp32",
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_indices = tuple(out_indices)
        self.is_frozen = False
        self.transformer = _make_dinov3_vit(
            img_size=224,
            patch_size=patch_size,
            in_chans=3,
            compact_arch_name="vitb",
            pos_embed_rope_base=100.0,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2.0,
            pos_embed_rope_dtype=rope_dtype,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            ffn_ratio=mlp_ratio,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained or (weights is not None),
            weights=self._resolve_weights(weights),
        )
        self.n_storage_tokens = self.transformer.n_storage_tokens
        self.transformer.patch_embed.proj.weight.requires_grad = False
        if self.transformer.patch_embed.proj.bias is not None:
            self.transformer.patch_embed.proj.bias.requires_grad = False

        if self.out_indices:
            max_index = max(self.out_indices)
            if max_index >= len(self.transformer.blocks):
                raise ValueError(
                    f"out_indices={self.out_indices} exceed transformer depth {len(self.transformer.blocks)}."
                )
        if freeze:
            self.freeze()

    @staticmethod
    def _resolve_weights(weights: str | None) -> Weights | str:
        if weights is None:
            return Weights.LVD1689M

        candidate = weights.strip()
        if candidate.upper() in Weights.__members__:
            return Weights[candidate.upper()]
        if urlparse(candidate).scheme in {"http", "https", "file"}:
            return candidate
        return str(Path(candidate).expanduser())

    def freeze(self) -> None:
        self.is_frozen = True
        self.transformer.eval()
        for parameter in self.transformer.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> TokenTransformerEncoder:
        super().train(mode)
        if self.is_frozen:
            self.transformer.eval()
        return self

    def _normalize_sequence(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transformer.untie_cls_and_patch_norms:
            cls_and_storage = self.transformer.cls_norm(sequence[:, : self.n_storage_tokens + 1])
            patch_tokens = self.transformer.norm(sequence[:, self.n_storage_tokens + 1 :])
            return cls_and_storage, patch_tokens

        normalized = self.transformer.norm(sequence)
        return normalized[:, : self.n_storage_tokens + 1], normalized[:, self.n_storage_tokens + 1 :]

    def _prepare_token_sequence(
        self,
        patch_tokens: torch.Tensor,
        grid_size: tuple[int, int],
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if patch_tokens.dim() != 3:
            raise ValueError(f"Expected patch tokens of shape (B, N, D), got {tuple(patch_tokens.shape)}.")

        grid_h, grid_w = grid_size
        expected_patches = grid_h * grid_w
        batch_size, num_patches, dim = patch_tokens.shape
        if num_patches != expected_patches:
            raise ValueError(
                f"grid_size={grid_size} implies {expected_patches} patches, but received {num_patches} tokens."
            )
        if dim != self.embed_dim:
            raise ValueError(f"Expected token dimension {self.embed_dim}, got {dim}.")
        if masks is not None and masks.shape != (batch_size, num_patches):
            raise ValueError(f"Expected masks of shape {(batch_size, num_patches)}, got {tuple(masks.shape)}.")

        if masks is not None:
            masks = masks.to(device=patch_tokens.device, dtype=torch.bool)
            mask_token = self.transformer.mask_token.to(dtype=patch_tokens.dtype, device=patch_tokens.device)
            patch_tokens = torch.where(masks.unsqueeze(-1), mask_token.view(1, 1, -1), patch_tokens)
            cls_token = self.transformer.cls_token
        else:
            cls_token = self.transformer.cls_token + 0 * self.transformer.mask_token

        if self.n_storage_tokens > 0:
            storage_tokens = self.transformer.storage_tokens
        else:
            storage_tokens = patch_tokens.new_empty(1, 0, dim)

        sequence = torch.cat(
            [
                cls_token.expand(batch_size, -1, -1),
                storage_tokens.expand(batch_size, -1, -1),
                patch_tokens,
            ],
            dim=1,
        )
        return sequence, (grid_h, grid_w)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        grid_size: tuple[int, int],
        masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        sequence, (grid_h, grid_w) = self._prepare_token_sequence(patch_tokens, grid_size, masks=masks)
        rope = self.transformer.rope_embed(H=grid_h, W=grid_w) if self.transformer.rope_embed is not None else None

        features = []
        x = sequence
        for idx, block in enumerate(self.transformer.blocks):
            x = block(x, rope)
            if idx in self.out_indices:
                _, patch_feature = self._normalize_sequence(x)
                features.append(patch_feature)

        cls_and_storage, patch = self._normalize_sequence(x)
        return {
            "cls_token": cls_and_storage[:, 0],
            "storage_tokens": cls_and_storage[:, 1:],
            "patch_tokens": patch,
            "intermediate_tokens": features,
            "x_norm_clstoken": cls_and_storage[:, 0],
            "x_storage_tokens": cls_and_storage[:, 1:],
            "x_norm_patchtokens": patch,
            "x_prenorm": x,
            "masks": masks,
        }
