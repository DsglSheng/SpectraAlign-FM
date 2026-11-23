from __future__ import annotations

import unittest

import torch

from aware_open.config import AwareEncoderConfig, FoundationConfig, FoundationDecoderConfig
from aware_open.data_types import ModalityBatch, SampleMetadataBatch
from aware_open.models.foundation import MetadataConditionedFoundationModel
from aware_open.sigma450k_profiles import build_eo1_hyperion_good175_profile, build_gf5_ahsi_vnir150_profile


def make_foundation_model() -> MetadataConditionedFoundationModel:
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
            low_cutoff_ratio=0.15,
            mid_cutoff_ratio=0.40,
        ),
        prototype_bank_size=8,
    )
    eo1_profile = build_eo1_hyperion_good175_profile()
    gf5_profile = build_gf5_ahsi_vnir150_profile()
    generator = torch.Generator().manual_seed(7)
    return MetadataConditionedFoundationModel(
        config,
        eo1_bank_centers=torch.randn(8, 175, generator=generator),
        gf5_bank_centers=torch.randn(8, 150, generator=generator),
        eo1_bank_wavelengths=torch.from_numpy(eo1_profile.wavelengths_nm),
        gf5_bank_wavelengths=torch.from_numpy(gf5_profile.wavelengths_nm),
    )


def make_mixed_space_batch() -> ModalityBatch:
    eo1_profile = build_eo1_hyperion_good175_profile()
    gf5_profile = build_gf5_ahsi_vnir150_profile()
    height = width = 16
    max_bands = 175
    values = torch.zeros(2, max_bands, height, width)
    values[0, :175] = torch.linspace(-1.0, 1.0, 175 * height * width).view(175, height, width)
    values[1, :150] = torch.linspace(0.5, 2.0, 150 * height * width).view(150, height, width)

    wavelengths = torch.zeros(2, max_bands, dtype=torch.float32)
    wavelengths[0, :175] = torch.from_numpy(eo1_profile.wavelengths_nm)
    wavelengths[1, :150] = torch.from_numpy(gf5_profile.wavelengths_nm)

    fwhm = torch.zeros(2, max_bands, dtype=torch.float32)
    fwhm[0, :175] = torch.from_numpy(eo1_profile.fwhm_nm)
    fwhm[1, :150] = torch.from_numpy(gf5_profile.fwhm_nm)

    valid_mask = torch.zeros(2, max_bands, dtype=torch.bool)
    valid_mask[0, :175] = True
    valid_mask[1, :150] = True

    metadata = SampleMetadataBatch(
        modality=["space", "space"],
        sensor=["eo1", "gf5"],
        level=["L1", "L1"],
        gsd=torch.tensor([30.0, 30.0], dtype=torch.float32),
        footprint_area=torch.tensor([height * width * 900.0, height * width * 900.0], dtype=torch.float32),
    )
    return ModalityBatch(
        values=values,
        wavelengths=wavelengths,
        fwhm=fwhm,
        metadata=metadata,
        valid_mask=valid_mask,
    )


class FoundationModelTests(unittest.TestCase):
    def test_forward_emits_progressive_reconstruction_and_prototype_losses(self) -> None:
        model = make_foundation_model()
        batch = make_mixed_space_batch()

        torch.manual_seed(123)
        output = model(space=batch)

        self.assertEqual(output["prediction"]["low"].shape, (2, 4, 64 * 175))
        self.assertEqual(output["prediction"]["mid"].shape, (2, 4, 64 * 175))
        self.assertEqual(output["prediction"]["full"].shape, (2, 4, 64 * 175))
        self.assertEqual(output["normalized_embedding"].shape, (2, 192))
        self.assertIn("reconstruction_low", output["losses"])
        self.assertIn("reconstruction_mid", output["losses"])
        self.assertIn("reconstruction_full", output["losses"])
        self.assertIn("proto_kl", output["losses"])
        self.assertIn("proto_balance", output["losses"])
        self.assertIn("student_prototype_entropy", output["metrics"])
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_invalid_padded_channels_are_ignored(self) -> None:
        model = make_foundation_model()
        model.eval()
        batch = make_mixed_space_batch()
        changed = ModalityBatch(
            values=batch.values.clone(),
            wavelengths=batch.wavelengths.clone(),
            fwhm=batch.fwhm.clone(),
            metadata=batch.metadata,
            valid_mask=batch.valid_mask.clone(),
        )
        changed.values[1, 150:] = 5000.0

        with torch.no_grad():
            torch.manual_seed(123)
            base_output = model(space=batch)
            torch.manual_seed(123)
            changed_output = model(space=changed)

        torch.testing.assert_close(base_output["loss"], changed_output["loss"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            base_output["losses"]["reconstruction_full"],
            changed_output["losses"]["reconstruction_full"],
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            base_output["losses"]["proto_kl"],
            changed_output["losses"]["proto_kl"],
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
