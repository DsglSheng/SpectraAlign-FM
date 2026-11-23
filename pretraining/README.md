# Joint pretraining on HyperGlobal-DZ1

The full recipe uses 524,979 patches from EO-1 Hyperion, GF-5 AHSI, DZ1 VNIR, and DZ1 SWIR. DZ1 SWIR patches retain their native 64 × 64 geometry. DZ1 VNIR patches are calibrated and area-averaged from 128 × 128 to 64 × 64. The VNIR and SWIR patches enter the joint manifest as separate samples.

## Prepare training assets

Run from the repository root after [installation](../README.md#install). Point `INPUT` to your raw DZ1 product directory and, if available, `CLASSIFICATION` to the acquisition metadata CSV. The preparation job writes derived files under `dz1_build/` in the data directory.

```bash
export SPECTRAALIGN_DATA_ROOT="$PWD/data"
export INPUT="$SPECTRAALIGN_DATA_ROOT/raw_dz1"

bash pretraining/scripts/launch_prepare_full.sh
```

Place your EO-1/GF-5 shards under `sigma450k_shards/` in the data directory, or set `HYPERGLOBAL_SHARDS` to their path. If the DZ1 shards are already prepared elsewhere, set `DZ1_SHARDS` to that directory. Use the same ground spectral library for the four sensor-specific banks to reproduce the paper recipe. [Ground spectra download and sources](../docs/ground_spectra.md)

```bash
python scripts/build_ground_prototypes.py \
  --ground-root "$SPECTRAALIGN_DATA_ROOT/ground" \
  --output-dir "$SPECTRAALIGN_DATA_ROOT/prototype_banks" \
  --prototype-count 256 --prototype-min-coverage 0.95 \
  --aux-grid-step-nm 1.0 \
  --eo1-max-source-step-nm 5.0 --gf5-max-source-step-nm 2.0 \
  --sensor both

bash pretraining/scripts/launch_build_joint_assets.sh
```

The second command builds the two DZ1 banks, writes a zero-copy manifest that references the shard files, and runs a mixed-sensor forward pass. Set `GROUND_ROOT` to point the launcher at a ground library outside the default location and pass the same path to the first command.

## Validate and launch

The [final recipe](configs/final_pretrain_from_scratch_v1.json) stores the expected sensor counts, batch size, loss weights, and update budgets. The preflight checks the joint manifest, the four prototype banks, and the mixed-sensor smoke report. The reference EO-1/GF-5 checkpoint is optional and is used for comparison.

```bash
python pretraining/scripts/preflight_final_pretrain.py \
  --config pretraining/configs/final_pretrain_from_scratch_v1.json

CONFIRM_FINAL_PRETRAIN=1 bash pretraining/scripts/launch_final_pretrain_from_scratch_v1.sh
```

The four-GPU launcher starts the neutral reconstruction stage from random initialization, then initializes the full metadata, alignment, and reconstruction stage from its checkpoint. It saves checkpoints under `pretraining/runs/` and resumes each stage from its latest numbered checkpoint. For custom settings, see the arguments in `pretraining/scripts/train_foundation_dz1.py --help`.
