from __future__ import annotations

"""Run HyperSIGMA on WHU-Hi HanChuan under the existing Aware dense protocol.

This thin adapter deliberately reuses the frozen classification training loop so
the split, crop generation, augmentation, validation selection, TTA, and metric
calculation are identical to the already completed Aware/SpectralEarth/SpecAware
runs.  Only the backbone and task head are replaced by official HyperSIGMA.
"""

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / "upstream_snapshot"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

import finetune_classification as base  # noqa: E402


@contextmanager
def allow_argparse_namespace_unpickle():
    with torch.serialization.safe_globals([argparse.Namespace]):
        yield


def _arg_value(name: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


class HyperSIGMAExtractor(nn.Module):
    """Compatibility holder for the frozen Aware training entry point."""

    def __init__(
        self,
        checkpoint_path: Path,
        prototype_bank_eo1: Path,
        prototype_bank_gf5: Path,
        metadata: Any,
        freeze_encoder: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        del prototype_bank_eo1, prototype_bank_gf5, freeze_encoder, device
        self.spatial_checkpoint = Path(checkpoint_path)
        self.spectral_checkpoint = Path(
            os.environ.get(
                "HYPERSIGMA_SPECTRAL_CHECKPOINT",
                "/mnt/Data/SpectralData/HyperSIGMA-main/Pretrain/spec-vit-base-ultra-checkpoint-1599.pth",
            )
        )
        self.hypersigma_root = Path(
            os.environ.get("HYPERSIGMA_ROOT", "/mnt/Data/SpectralData/HyperSIGMA-main")
        )
        self.in_channels = int(len(metadata.wavelengths_nm))
        self.local_patch_size = int(_arg_value("--hypersigma-patch-size", "8"))
        self.input_size = int(_arg_value("--img-size", "128"))
        self.model_size = os.environ.get("HYPERSIGMA_MODEL_SIZE", "base")

    def prototype_auxiliary_loss(self, *_: Any, **__: Any) -> dict[str, torch.Tensor]:
        zero = next(self.parameters(), torch.zeros(())).new_zeros(())
        return {"proto_kl": zero, "proto_balance": zero, "proto_entropy": zero}


class HyperSIGMADenseAdapter(nn.Module):
    def __init__(
        self,
        extractor: HyperSIGMAExtractor,
        num_classes: int,
        **_: Any,
    ) -> None:
        super().__init__()
        root = extractor.hypersigma_root
        image_cls_root = root / "ImageClassification"
        if str(image_cls_root) not in sys.path:
            sys.path.insert(0, str(image_cls_root))
        from model import ss_fusion_seg  # type: ignore

        self.extractor = extractor
        self.model = ss_fusion_seg.SSFusionFramework(
            img_size=extractor.input_size,
            in_channels=extractor.in_channels,
            patch_size=extractor.local_patch_size,
            classes=int(num_classes),
            model_size=extractor.model_size,
        )
        if not extractor.spatial_checkpoint.exists():
            raise FileNotFoundError(extractor.spatial_checkpoint)
        if not extractor.spectral_checkpoint.exists():
            raise FileNotFoundError(extractor.spectral_checkpoint)
        with allow_argparse_namespace_unpickle():
            self.model.spat_encoder.init_weights(str(extractor.spatial_checkpoint))
            self.model.spec_encoder.init_weights(str(extractor.spectral_checkpoint))

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if not return_features:
            return self.model(x)
        captured: dict[str, Any] = {}

        def save_spatial(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            captured["spatial"] = output

        handle = self.model.spat_encoder.register_forward_hook(save_spatial)
        try:
            logits = self.model(x)
        finally:
            handle.remove()
        maps = list(captured.get("spatial", ()))
        return logits, {"intermediate_maps": maps, "last_map": maps[-1] if maps else None}


def build_hypersigma_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    layer_decay_rate: float,
) -> torch.optim.Optimizer:
    """Layer-wise AdamW using both HyperSIGMA transformer block namespaces."""
    groups: dict[tuple[float, float], dict[str, Any]] = {}
    depth = 12
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id: int | None = None
        for marker in ("spat_encoder.blocks.", "spec_encoder.blocks."):
            if marker in name:
                try:
                    layer_id = int(name.split(marker, 1)[1].split(".", 1)[0])
                except (IndexError, ValueError):
                    layer_id = None
                break
        scale = 1.0 if layer_id is None else layer_decay_rate ** max(depth - layer_id - 1, 0)
        decay = 0.0 if param.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name else weight_decay
        key = (scale, decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * scale, "weight_decay": decay}
        groups[key]["params"].append(param)
    return torch.optim.AdamW(list(groups.values()), lr=lr, betas=(0.9, 0.999))


def _strip_adapter_only_args() -> None:
    while "--hypersigma-patch-size" in sys.argv:
        idx = sys.argv.index("--hypersigma-patch-size")
        del sys.argv[idx : idx + 2]


if __name__ == "__main__":
    # Read before removal because the compatibility extractor is built later.
    patch_size = _arg_value("--hypersigma-patch-size", "8")
    os.environ["HYPERSIGMA_PATCH_SIZE"] = patch_size
    original_arg_value = _arg_value

    def patched_arg_value(name: str, default: str) -> str:
        if name == "--hypersigma-patch-size":
            return os.environ.get("HYPERSIGMA_PATCH_SIZE", default)
        return original_arg_value(name, default)

    globals()["_arg_value"] = patched_arg_value
    _strip_adapter_only_args()
    base.AwareFeatureExtractor = HyperSIGMAExtractor
    base.AwareSSFusionSeg = HyperSIGMADenseAdapter
    base.build_layer_decay_optimizer = build_hypersigma_optimizer
    base.main()
