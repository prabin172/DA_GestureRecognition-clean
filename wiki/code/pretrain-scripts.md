---
type: code
status: active
updated: 2026-07-03
---

# Pretrain scripts (`src/scripts/pretrain/` + root)

Implement the [[pretraining-objectives]]; checkpoints land per [[ntu-pretraining]].

| script | objective |
|---|---|
| `train_mae.py`, `train_mae_geodesic.py` | MAE, ~70% frame masking; MSE vs geodesic loss |
| `train_supervised_ntu.py`, `train_sup_lp_ntu.py` | supervised CE on NTU 120 (or relevant-23) |
| `train_supMae.py` | SupMAE: MAE + CE, CE warm-up (off 3 ep, ramp 7), EMA loss balancing |
| `train_supervised_dann_ntu.py` | supervised NTU + domain adversary |
| `pretrain_sourceSupervisedDANN_all120.py` / `_relevant23.py` | DANN, NTU labels, unlabeled IMU (LOSO-aware) |
| `pretrain_targetSupervisedDANN.py` / `_relevant23.py` | DANN with labeled IMU target |
| `pretrain_sourceTargetSupervisedDANN_twoHeads.py` | two-head DANN (source + target classifiers) |
| `pretrain_kshot_DANN_diagnostics.py` | DANN k/λ ablation |
| `pretrain_targetOnly.py` | encoder on Xsens only, no NTU |
| root `scripts/pretrain/pretrain_supcon.py` | SupCon (Khosla 2020) on NTU → `trained_models/ContrastiveNTU/` |
| root `temp_pretrain_supervised_ntu.py` | supervised NTU variant |

DANN details in [[dann-experiments]]; cleaned-source pretraining runs in [[cleaned-source-pretraining]].
