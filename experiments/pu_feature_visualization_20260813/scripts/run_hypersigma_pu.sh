#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/Data/SpectralData/pu_feature_visualization_20260813
PY=/home/user/miniconda3/envs/aware/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/mnt/Data/SpectralData/Aware_open:/mnt/Data/SpectralData/Aware_open/scripts:/mnt/Data/SpectralData/cross_sensor_uav_20260812/code:/mnt/Data/SpectralData/cross_sensor_uav_20260812/upstream_snapshot"
export HYPERSIGMA_ROOT=/mnt/Data/SpectralData/HyperSIGMA-main
export HYPERSIGMA_SPECTRAL_CHECKPOINT=/mnt/Data/SpectralData/HyperSIGMA-main/Pretrain/spec-vit-base-ultra-checkpoint-1599.pth
"${PY}" /mnt/Data/SpectralData/cross_sensor_uav_20260812/code/finetune_hypersigma_hc_transfer.py \
  --checkpoint /mnt/Data/SpectralData/HyperSIGMA-main/Pretrain/spat-vit-base-ultra-checkpoint-1599.pth \
  --prototype-bank-eo1 /mnt/Data/SpectralData/Aware_open/artifacts/prototypes_v3_core_waterzero_artificialx4_20260503/prototype_bank_eo1.pt \
  --prototype-bank-gf5 /mnt/Data/SpectralData/Aware_open/artifacts/prototypes_v3_core_waterzero_artificialx4_20260503/prototype_bank_gf5.pt \
  --dataset pu \
  --data-root /mnt/Data/SpectralData/HyperSIGMA-main/ImageClassification/data \
  --output-dir "${ROOT}/runs/hypersigma_pu_seed3704" \
  --img-size 128 --overlap-size 32 --train-num 20 --val-num 20 --split-mode number \
  --epochs 28 --scheduler-epochs 80 --batch-size 4 --num-workers 0 \
  --lr 6e-5 --weight-decay 0.05 --layer-decay-rate 0.9 \
  --label-smoothing 0.05 --seed 3704 --normalization mean_std \
  --pca-components 30 --pca-whiten --train-repeat-factor 4 --train-spatial-jitter 8 \
  --centered-training --eval-tta --precision bf16 --hypersigma-patch-size 8
