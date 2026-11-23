# EO-1 and GF-5 pretraining

The two-sensor recipe uses EO-1 Hyperion and GF-5 AHSI shards with sensor-specific ground prototype banks. It applies metadata-conditioned encoding, patch-to-prototype KL alignment, and frequency-aware masked reconstruction.

Prepare the two banks by following the `build_ground_prototypes.py` command in the [joint pretraining guide](../pretraining/README.md#prepare-training-assets). Point `SPECTRAALIGN_DATA_ROOT` to the directory with `sigma450k_shards/` and `prototype_banks/`.

```bash
torchrun --nproc_per_node=4 scripts/train_foundation.py \
  --data-root "$SPECTRAALIGN_DATA_ROOT" \
  --sigma-shard-root "$SPECTRAALIGN_DATA_ROOT/sigma450k_shards" \
  --prototype-bank-eo1 "$SPECTRAALIGN_DATA_ROOT/prototype_banks/prototype_bank_eo1.pt" \
  --prototype-bank-gf5 "$SPECTRAALIGN_DATA_ROOT/prototype_banks/prototype_bank_gf5.pt" \
  --output-dir runs/eo1_gf5 \
  --epochs 100 --save-every-epochs 5 --probe-every-epochs 1 \
  --batch-size 16 --precision bf16
```

Add `--resume runs/eo1_gf5/checkpoints/latest.pt` to resume. The run directory contains the active configuration, training log, and checkpoints.

Prototype construction resamples quality-masked ground spectra to a uniform auxiliary wavelength grid, then projects them into each satellite's observation bands with Gaussian spectral response functions. The EO-1 and GF-5 source sampling step thresholds are 5 nm and 2 nm respectively.
