from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from scripts.finetune_classification import AwareSSFusionSeg


class DummyExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_dim = 48
        self.band_dim = 16
        self.context_dim = 12
        self.local_patch_size = 8
        self.encoder = SimpleNamespace(
            backbone=SimpleNamespace(
                transformer=SimpleNamespace(blocks=[])
            )
        )

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
        batch_size = x.shape[0]
        grid_size = (x.shape[-2] // self.local_patch_size, x.shape[-1] // self.local_patch_size)
        token_count = grid_size[0] * grid_size[1]
        generator = torch.Generator(device=x.device).manual_seed(17)
        return {
            "grid_size": grid_size,
            "patch_tokens": torch.randn(batch_size, token_count, self.model_dim, generator=generator, device=x.device),
            "intermediate_tokens": [
                torch.randn(batch_size, token_count, self.model_dim, generator=generator, device=x.device)
                for _ in range(4)
            ],
            "band_tokens": torch.randn(batch_size, x.shape[1], self.band_dim, generator=generator, device=x.device),
            "sample_context": torch.randn(batch_size, self.context_dim, generator=generator, device=x.device),
        }


class ClassificationModelTests(unittest.TestCase):
    def test_hypersigma_style_segmentation_head_outputs_dense_logits(self) -> None:
        model = AwareSSFusionSeg(DummyExtractor(), num_classes=16, feature_layers=4, fusion_channels=16)
        x = torch.randn(2, 200, 40, 40)
        logits, features = model(x, return_features=True)

        self.assertEqual(tuple(logits.shape), (2, 40 * 40, 16))
        self.assertEqual(features["grid_size"], (5, 5))
        self.assertIn("sample_context", features)


if __name__ == "__main__":
    unittest.main()
