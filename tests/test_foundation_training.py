from __future__ import annotations

import argparse
import shutil
import unittest
import uuid
from pathlib import Path

import torch

from aware_open.config import AwareEncoderConfig, FoundationConfig, FoundationDecoderConfig
from aware_open.models.foundation import MetadataConditionedFoundationModel
from scripts.train_foundation import (
    build_checkpoint_payload,
    build_foundation_config,
    build_optimizer,
    maybe_resume,
    resolve_loss_schedule,
    validate_args,
)


def make_model() -> MetadataConditionedFoundationModel:
    encoder = AwareEncoderConfig(
        model_dim=192,
        band_dim=64,
        context_dim=96,
        spectral_dim=48,
        spatial_dim=48,
        num_prototypes=6,
        local_patch_size=8,
        prototype_heads=4,
        prototype_depth=1,
        token_encoder_depth=2,
        token_encoder_heads=6,
        backbone_out_indices=(0, 1),
        backbone_pretrained=False,
        backbone_drop_path_rate=0.0,
        drop_rate=0.0,
    )
    config = FoundationConfig(
        encoder=encoder,
        decoder=FoundationDecoderConfig(
            decoder_dim=96,
            decoder_depth=1,
            decoder_heads=4,
            decoder_dropout=0.0,
            tdfa_depth=1,
            frequency_groups=4,
        ),
        prototype_bank_size=8,
    )
    generator = torch.Generator().manual_seed(11)
    return MetadataConditionedFoundationModel(
        config,
        eo1_bank_centers=torch.randn(8, 175, generator=generator),
        gf5_bank_centers=torch.randn(8, 150, generator=generator),
        eo1_bank_wavelengths=torch.linspace(400.0, 2400.0, 175),
        gf5_bank_wavelengths=torch.linspace(390.0, 1030.0, 150),
    )


class FoundationTrainingTests(unittest.TestCase):
    def make_workspace_tempdir(self) -> Path:
        base = Path(__file__).resolve().parents[2] / ".tmp-tests"
        base.mkdir(parents=True, exist_ok=True)
        path = base / uuid.uuid4().hex
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_validate_args_rejects_invalid_frequency_cutoffs(self) -> None:
        args = argparse.Namespace(
            epochs=10,
            save_every_epochs=5,
            probe_every_epochs=1,
            progress_refresh_seconds=10.0,
            grad_accum_steps=1,
            mask_ratio=0.75,
            frequency_cutoff_low=0.45,
            frequency_cutoff_mid=0.40,
        )
        with self.assertRaisesRegex(ValueError, "must satisfy 0 < low < mid < 1"):
            validate_args(args)

    def test_resolve_loss_schedule_warms_up_prototype_losses(self) -> None:
        args = argparse.Namespace(
            warmup_steps_loss=40_000,
            proto_kl_weight=0.10,
            proto_balance_weight=0.02,
        )

        warm = resolve_loss_schedule(args, 9_999)
        full = resolve_loss_schedule(args, 40_000)

        self.assertEqual(warm["reconstruction_low"], 0.5)
        self.assertLess(warm["proto_kl"], 0.10)
        self.assertEqual(full["reconstruction_full"], 0.7)
        self.assertAlmostEqual(full["proto_kl"], 0.10)
        self.assertAlmostEqual(full["proto_balance"], 0.02)

    def test_build_foundation_config_uses_requested_bank_size(self) -> None:
        args = argparse.Namespace(
            model_dim=192,
            band_dim=64,
            context_dim=96,
            spectral_dim=48,
            spatial_dim=48,
            num_routing_prototypes=6,
            local_patch_size=8,
            prototype_heads=4,
            prototype_depth=1,
            token_encoder_depth=2,
            token_encoder_heads=6,
            drop_rate=0.0,
            decoder_dim=96,
            decoder_depth=1,
            decoder_heads=4,
            decoder_dropout=0.0,
            tdfa_depth=1,
            frequency_groups=4,
            frequency_cutoff_low=0.15,
            frequency_cutoff_mid=0.40,
            mask_ratio=0.75,
            prototype_temperature_teacher=0.07,
            prototype_temperature_student=0.07,
            proto_kl_weight=0.10,
            proto_balance_weight=0.02,
        )

        config = build_foundation_config(args, prototype_bank_size=256)

        self.assertEqual(config.prototype_bank_size, 256)
        self.assertEqual(config.decoder.tdfa_depth, 1)
        self.assertEqual(config.encoder.model_dim, 192)

    def test_maybe_resume_restores_foundation_checkpoint(self) -> None:
        model = make_model()
        optimizer, _ = build_optimizer(
            argparse.Namespace(
                token_encoder_depth=2,
                weight_decay=0.05,
                lr=1.0e-4,
                beta1=0.9,
                beta2=0.95,
            ),
            model,
        )
        output_dir = self.make_workspace_tempdir()
        checkpoint_path = output_dir / "resume.pt"
        args = argparse.Namespace(resume=checkpoint_path, output_dir=output_dir)
        payload = build_checkpoint_payload(
            epoch=3,
            step=12,
            model=model,
            optimizer=optimizer,
            args=args,
            best_ssl=1.25,
            best_probe=0.33,
        )
        torch.save(payload, checkpoint_path)

        resumed = maybe_resume(args, model, optimizer)

        self.assertEqual(resumed.epoch, 3)
        self.assertEqual(resumed.step, 12)
        self.assertTrue(resumed.loaded_optimizer)
        self.assertAlmostEqual(resumed.best_ssl, 1.25)
        self.assertAlmostEqual(resumed.best_probe, 0.33)


if __name__ == "__main__":
    unittest.main()
