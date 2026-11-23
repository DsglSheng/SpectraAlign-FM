from __future__ import annotations

import math
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(spectra)).squeeze(-1)


def _replicate_patch(spectra: torch.Tensor, size: int) -> torch.Tensor:
    return spectra[:, :, None, None].expand(-1, -1, size, size).contiguous()


class AwareBackbone(nn.Module):
    def __init__(
        self,
        *,
        aware_repo: Path,
        checkpoint: Path,
        prototype_bank_eo1: Path,
        prototype_bank_gf5: Path,
        wavelengths_nm: torch.Tensor,
        sensor: str,
        level: str,
        gsd: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        repo = aware_repo.expanduser().resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from aware_open.downstream.adapter import AwareFeatureExtractor, build_downstream_metadata

        metadata = build_downstream_metadata(
            num_bands=int(wavelengths_nm.numel()),
            sensor=sensor,
            level=level,
            gsd=float(gsd),
            wavelengths_nm=wavelengths_nm.detach().cpu().numpy(),
        )
        self.extractor = AwareFeatureExtractor(
            checkpoint_path=checkpoint,
            prototype_bank_eo1=prototype_bank_eo1,
            prototype_bank_gf5=prototype_bank_gf5,
            metadata=metadata,
            freeze_encoder=False,
            device=device,
        )
        self.feature_dim = int(self.extractor.model_dim)
        self.patch_size = int(self.extractor.local_patch_size)
        self.load_info = {"checkpoint": str(checkpoint), "adapter": "native wavelength-aware feature extractor"}

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        patch = _replicate_patch(spectra, self.patch_size)
        return self.extractor.forward_features(patch)["pooled_embedding"]


class HyperSigmaBackbone(nn.Module):
    def __init__(self, *, repo: Path, checkpoint: Path, patch_size: int = 64) -> None:
        super().__init__()
        image_classification = repo.expanduser().resolve() / "ImageClassification"
        if str(image_classification) not in sys.path:
            sys.path.insert(0, str(image_classification))
        from model import SpecViT_fusion

        self.patch_size = int(patch_size)
        self.encoder = SpecViT_fusion.SpectralVisionTransformer(
            NUM_TOKENS=256,
            img_size=self.patch_size,
            in_chans=256,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_path_rate=0.1,
            out_indices=[11],
            interval=3,
            use_checkpoint=False,
            use_abs_pos_emb=False,
        )
        self.load_info = self._load_checkpoint(checkpoint.expanduser().resolve())
        self.feature_dim = 128
        self.load_info["adapter"] = "official HyperSIGMA spectral encoder on a constant 64x64 pixel patch"

    def _load_checkpoint(self, checkpoint: Path) -> dict[str, Any]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
        target = self.encoder.state_dict()
        loaded: dict[str, torch.Tensor] = {}
        for raw_key, value in source.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = str(raw_key)
            if key.startswith("module."):
                key = key[7:]
            if key.startswith("encoder."):
                key = key[len("encoder.") :]
            if key in target and value.shape == target[key].shape:
                loaded[key] = value
        missing, unexpected = self.encoder.load_state_dict(loaded, strict=False)
        return {
            "checkpoint": str(checkpoint),
            "loaded_keys": len(loaded),
            "missing_count": len(missing),
            "missing_preview": list(missing)[:20],
            "unexpected_count": len(unexpected),
        }

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        features = self.encoder(_replicate_patch(spectra, self.patch_size))[-1]
        return features.mean(dim=1)


def _resize_spatial_pos_embed(pos: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if pos.ndim != 3 or pos.shape[1] == target_tokens:
        return pos
    source_hw = int(round(math.sqrt(int(pos.shape[1]))))
    target_hw = int(round(math.sqrt(target_tokens)))
    if source_hw * source_hw != pos.shape[1] or target_hw * target_hw != target_tokens:
        return pos
    resized = pos.reshape(1, source_hw, source_hw, pos.shape[-1]).permute(0, 3, 1, 2)
    resized = F.interpolate(resized, size=(target_hw, target_hw), mode="bicubic", align_corners=False)
    return resized.permute(0, 2, 3, 1).reshape(1, target_tokens, pos.shape[-1])


def _load_spectralgpt_checkpoint(model: nn.Module, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = payload.get("model", payload)
    target = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    dropped: dict[str, str] = {}
    for key, value in source.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key in {"head.weight", "head.bias"}:
            dropped[key] = "task head"
            continue
        if key == "pos_embed_spatial" and key in target and value.shape != target[key].shape:
            value = _resize_spatial_pos_embed(value, int(target[key].shape[1]))
        if key not in target or value.shape != target[key].shape:
            dropped[key] = "missing or shape mismatch"
            continue
        loaded[key] = value
    missing, unexpected = model.load_state_dict(loaded, strict=False)
    return {
        "checkpoint": str(checkpoint),
        "loaded_keys": len(loaded),
        "dropped_count": len(dropped),
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
    }


class SpectralGPTBackbone(nn.Module):
    def __init__(self, *, repo: Path, checkpoint: Path, patch_size: int = 16) -> None:
        super().__init__()
        repo = repo.expanduser().resolve()
        for name in list(sys.modules):
            if name == "util" or name.startswith("util."):
                del sys.modules[name]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import models_vit_tensor

        self.patch_size = int(patch_size)
        self.encoder = models_vit_tensor.VisionTransformer(
            img_size=self.patch_size,
            in_chans=1,
            patch_size=8,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_frames=12,
            t_patch_size=3,
            norm_layer=partial(nn.LayerNorm, eps=1.0e-6),
            num_classes=1,
            drop_path_rate=0.1,
            dropout=0.0,
            sep_pos_embed=True,
            cls_embed=False,
        )
        self.load_info = _load_spectralgpt_checkpoint(self.encoder, checkpoint.expanduser().resolve())
        self.encoder.head = nn.Identity()
        self.encoder.dropout = nn.Identity()
        self.feature_dim = 768
        self.load_info["adapter"] = "linear spectral resampling to 12 frames on a constant 16x16 patch"

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        twelve = F.interpolate(spectra[:, None, :], size=12, mode="linear", align_corners=False).squeeze(1)
        patch = _replicate_patch(twelve, self.patch_size)
        return self.encoder(patch)


class SpectralEarthBackbone(nn.Module):
    def __init__(self, *, repo: Path, checkpoint: Path, patch_size: int = 16) -> None:
        super().__init__()
        repo = repo.expanduser().resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from src.backbones.spec_vit import SpecViTSmall

        self.patch_size = int(patch_size)
        self.encoder = SpecViTSmall(
            patch_size=self.patch_size,
            token_patch_size=4,
            num_classes=0,
            drop_path_rate=0.1,
            drop_rate=0.0,
        )
        self.load_info = self._load_checkpoint(checkpoint.expanduser().resolve())
        self.feature_dim = int(self.encoder.num_features)
        self.load_info["adapter"] = "official SpecViT-S MAE with native 256-band spectra"

    def _load_checkpoint(self, checkpoint: Path) -> dict[str, Any]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            payload = payload["state_dict"]
        if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
            payload = payload["model"]
        target = self.encoder.state_dict()
        loaded: dict[str, torch.Tensor] = {}
        for raw_key, value in payload.items():
            if not isinstance(value, torch.Tensor):
                continue
            candidates = [raw_key]
            if raw_key.startswith("module."):
                candidates.append(raw_key[7:])
            key = next((candidate for candidate in candidates if candidate in target), None)
            if key is None:
                continue
            if key.endswith("vit_core.pos_embed") and value.shape != target[key].shape:
                value = self._resize_pos_embed(value, target[key])
            if value.shape == target[key].shape:
                loaded[key] = value
        missing, unexpected = self.encoder.load_state_dict(loaded, strict=False)
        return {
            "checkpoint": str(checkpoint),
            "loaded_keys": len(loaded),
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
        }

    @staticmethod
    def _resize_pos_embed(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.ndim != 3 or target.ndim != 3 or source.shape[-1] != target.shape[-1]:
            return source
        source_tokens = int(source.shape[1]) - 1
        target_tokens = int(target.shape[1]) - 1
        source_side = int(round(math.sqrt(source_tokens)))
        target_side = int(round(math.sqrt(target_tokens)))
        if source_side * source_side != source_tokens or target_side * target_side != target_tokens:
            return source
        prefix = source[:, :1]
        spatial = source[:, 1:].reshape(1, source_side, source_side, source.shape[-1]).permute(0, 3, 1, 2)
        spatial = F.interpolate(spatial, size=(target_side, target_side), mode="bicubic", align_corners=False)
        spatial = spatial.permute(0, 2, 3, 1).reshape(1, target_tokens, source.shape[-1])
        return torch.cat([prefix, spatial], dim=1)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        patch = _replicate_patch(spectra, self.patch_size)
        tokens = self.encoder.get_intermediate_layers(patch, n=1, norm=True)[0]
        return tokens.mean(dim=1)


def _specaware_level(level: str) -> str:
    return "L1" if level.strip().lower() in {"l1", "toa", "radiance", "unknown"} else "L2"


class SpecAwareBackbone(nn.Module):
    def __init__(
        self,
        *,
        repo: Path,
        checkpoint: Path,
        sentence_model: Path,
        wavelengths_nm: torch.Tensor,
        fwhm_nm: torch.Tensor,
        sensor: str,
        level: str,
        gsd: float,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        repo = repo.expanduser().resolve()
        example = repo / "example"
        os.environ["SPECAWARE_SENTENCE_MODEL"] = str(sentence_model.expanduser().resolve())
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        for name in list(sys.modules):
            if name == "util" or name.startswith("util.") or name == "models_SpecAware_encoder":
                del sys.modules[name]
        for path in (example, repo / "SpecAware", repo):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import models_SpecAware_encoder

        self.patch_size = int(patch_size)
        self.encoder = models_SpecAware_encoder.MaskedHSIAutoencoderViT(
            img_size=self.patch_size,
            patch_size=8,
            in_chans=int(wavelengths_nm.numel()),
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            norm_layer=partial(nn.LayerNorm, eps=1.0e-6),
            drop_path_rate=0.1,
        )
        self.load_info = self._load_checkpoint(checkpoint.expanduser().resolve())
        self.register_buffer("wavelengths_nm", wavelengths_nm.float().flatten(), persistent=False)
        self.register_buffer("fwhm_nm", fwhm_nm.float().flatten(), persistent=False)
        self.register_buffer("gsd", torch.tensor(float(gsd), dtype=torch.float32), persistent=False)
        normalized_sensor = sensor.strip().lower()
        self.sensor = normalized_sensor if normalized_sensor in {"avc", "avng", "av3"} else "others"
        self.level = _specaware_level(level)
        self.feature_dim = 768
        self.load_info["adapter"] = "official wavelength-conditioned SpecAware encoder"

    def _load_checkpoint(self, checkpoint: Path) -> dict[str, Any]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = payload.get("model", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
        target = self.encoder.state_dict()
        loaded: dict[str, torch.Tensor] = {}
        for raw_key, value in source.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = str(raw_key)
            for prefix in ("module.", "encoder.", "backbone.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            if key in target and value.shape == target[key].shape:
                loaded[key] = value
        missing, unexpected = self.encoder.load_state_dict(loaded, strict=False)
        return {
            "checkpoint": str(checkpoint),
            "loaded_keys": len(loaded),
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
        }

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        patch = _replicate_patch(spectra, self.patch_size)
        batch = int(patch.shape[0])
        _, tokens = self.encoder(
            patch,
            GSD=self.gsd.expand(batch),
            wavelength=self.wavelengths_nm,
            fwhm=self.fwhm_nm,
            sensor_name=self.sensor,
            data_level=self.level,
        )
        return tokens[:, 0] if tokens.shape[1] > 1 else tokens.mean(dim=1)


def build_model(
    model_name: str,
    *,
    wavelengths_nm: torch.Tensor,
    fwhm_nm: torch.Tensor,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
    paths: dict[str, Path],
    sensor: str,
    level: str,
    gsd: float,
) -> tuple[RegressionModel, dict[str, Any]]:
    if model_name == "aware":
        backbone = AwareBackbone(
            aware_repo=paths["aware_repo"],
            checkpoint=paths["aware_checkpoint"],
            prototype_bank_eo1=paths["prototype_bank_eo1"],
            prototype_bank_gf5=paths["prototype_bank_gf5"],
            wavelengths_nm=wavelengths_nm,
            sensor=sensor,
            level=level,
            gsd=gsd,
            device=device,
        )
    elif model_name == "hypersigma":
        backbone = HyperSigmaBackbone(repo=paths["hypersigma_repo"], checkpoint=paths["hypersigma_checkpoint"])
    elif model_name == "spectralgpt":
        backbone = SpectralGPTBackbone(repo=paths["spectralgpt_repo"], checkpoint=paths["spectralgpt_checkpoint"])
    elif model_name == "spectralearth":
        backbone = SpectralEarthBackbone(repo=paths["spectralearth_repo"], checkpoint=paths["spectralearth_checkpoint"])
    elif model_name == "specaware":
        backbone = SpecAwareBackbone(
            repo=paths["specaware_repo"],
            checkpoint=paths["specaware_checkpoint"],
            sentence_model=paths["specaware_sentence_model"],
            wavelengths_nm=wavelengths_nm,
            fwhm_nm=fwhm_nm,
            sensor=sensor,
            level=level,
            gsd=gsd,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model = RegressionModel(backbone, backbone.feature_dim, hidden_dim, dropout)
    info = {
        "model": model_name,
        "feature_dim": int(backbone.feature_dim),
        "load_info": backbone.load_info,
    }
    return model, info
