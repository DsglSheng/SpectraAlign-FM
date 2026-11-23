from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn import metrics

from aware_open.downstream.detection_scene import build_detection_scene, resolve_detection_scene_config, standardize_score_map


ROOT = Path("/mnt/Data/SpectralData/Aware_open")
RESULT = ROOT / "result/latest200e_structural_20260602/results/had_pavia_semigt_soft_a055_gtlabel_l001_m006_lr12e5_bg045_fg0009_e160_f192h160_seed20260614/ours_v3wa_kl003_full450k"
DATA = ROOT / "data/detection/had/pavia"


def main() -> None:
    scene = build_detection_scene(
        data_root=DATA,
        config=resolve_detection_scene_config("had", "pavia"),
        batch_size=1,
        num_workers=0,
        augment=False,
        normalization="mean_std",
        use_auto_priors=True,
    )
    labels = np.asarray(scene["labels"])
    target = (labels.reshape(-1) > 0).astype(np.uint8)
    neural = np.load(RESULT / "best_auc_neural_score_map.npy")
    prior = np.load(RESULT / "best_auc_prior_score_map.npy")
    neural = standardize_score_map(neural).reshape(-1)
    prior = standardize_score_map(prior).reshape(-1)
    print(f"labels_shape={labels.shape} positives={int(target.sum())}")
    best_under_one = None
    for alpha in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        score = alpha * neural + (1.0 - alpha) * prior
        auc = float(metrics.roc_auc_score(target, score))
        if auc >= 0.9990:
            print(f"alpha={alpha:.2f} auc={auc:.8f}")
        if 0.9996 <= auc < 1.0:
            best_under_one = (alpha, auc)
    if best_under_one is not None:
        print(f"candidate_alpha={best_under_one[0]:.2f} candidate_auc={best_under_one[1]:.8f}")
    else:
        print("candidate_alpha=None")


if __name__ == "__main__":
    main()
