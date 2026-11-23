# Satellite-to-UAV transfer experiment (2026-08-12)

This isolated branch evaluates the final satellite-pretrained Aware-HSI model on
the UAV WHU-Hi HanChuan scene and produces last-backbone-layer feature-norm maps.
It does not modify any previous pretraining, fine-tuning, or manuscript result.

## Controlled protocol

- dataset: WHU-Hi HanChuan (Headwall Nano-Hyperspec, 0.109 m GSD)
- split: 50 train and 20 validation pixels per class, all remaining labelled pixels for test
- seed: 3704
- input: 128 x 128, mean/std normalized native spectral bands
- optimization: 80 epochs, full-parameter fine-tuning, best validation checkpoint
- metrics: OA, AA, Cohen's kappa, and mIoU computed from the full test confusion matrix

The validation-selected Aware-HSI, SpectralEarth, and SpecAware runs are reused
because they already share the same seed, split, input size, epoch budget, and
validation-only checkpoint selection. HyperSIGMA is recomputed because no such
run existed. Each method retains its native task head, spectral input adapter,
and training augmentation; these unavoidable method-level differences are
recorded rather than described as a bitwise-identical training protocol.
