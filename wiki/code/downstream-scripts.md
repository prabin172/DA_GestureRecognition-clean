---
type: code
status: active
updated: 2026-07-03
---

# Downstream & analysis scripts (`src/scripts/downstream/`, `analysis/`)

Evaluation paradigm: [[loso-protocol|k-shot LOSO]]. The *current main* evaluation scripts are root-level temp scripts ([[temp-scripts]], esp. `scripts/main_experiment/loso_fulltrain_calibration.py`); these `src/` scripts are the earlier generation ([[early-experiments]]).

| script | purpose |
|---|---|
| `LOSO-singleSubject.py` | one subject (`SUBJECT="sub11"`), k∈{1,3,5}, 4 inits × frozen/finetuned + scratch |
| `LOSO-crossSubject.py` | same over all 5 subjects |
| `train_imu_loso.py` | IMU LOSO baseline, no pretraining → `TARGET_ONLY_LOSO/` |
| `train_imu_baseline.py`, `train_imu_baseline-subSplit.py` | IMU-only baselines |
| `train_imu_baseline_subSplit_linearProbe.py`, `k_shot_subSplit_linearProbe.py`, `k_sub8_linearProbe_v2.py`, `k_subSplit_lP-2.py` | linear-probe variants |
| `sub8_adaptation.py`, `sub8_calibrate_test_others.py` | sub8-focused adaptation |
| `scratch_seed_Sweep.py` | seed sweep for scratch baseline |
| `n_actions_k_shot.py` | vary #classes in k-shot |
| `train_dann.py` | downstream DANN adaptation |
| `val_ntu_classification.py` | validate a checkpoint's NTU accuracy |
| `plot_imu_confusion.py` | confusion matrices |
| `analysis/plot_LOSO_crossSubject.py`, `analysis/plot_LOSO_singleSubject.py` | result plots |
