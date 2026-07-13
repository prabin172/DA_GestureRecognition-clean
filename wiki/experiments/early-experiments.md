---
type: experiment
status: superseded
updated: 2026-07-03
---

# Early / superseded experiments

First-generation experiments, superseded by the [[loso-fulltrain-calibrate]] pipeline. Kept for provenance; don't build new results on them.

| experiment | dirs | notes |
|---|---|---|
| LOSO single/cross-subject | `LOSO-singleSubject/`, `LOSO-crossSubject/` | 4 inits × frozen/finetuned, k∈{1,3,5}; scripts in [[downstream-scripts]] |
| IMU-only baselines & linear probes | `IMU/`, `IMU_LOSO/`, `IMU_LINEARPROBE/`, `_V2/`, `LOSO-supLP120-seed0/`, `LOSO-supLP120-multiseed/` | early probes |
| Target-only LOSO | `TARGET_ONLY_LOSO/` | early no-NTU baseline; the real answer is now [[xsens-to-xsens-loso]] |
| DSTformer target-only | `DSTFORMER_QUAT_TARGET_ONLY_LOSO/` | [[models|DSTformerQuatEncoder]] trial; encoder not adopted in the main pipeline |
| Source relatedness ablation | `SourceRelatednessAblation/` | which/how many NTU classes matter — relatedness doesn't rescue transfer (Pillar 2 support) |
| SupMAE λ sweep | `SupMAELambdaSweep/` | per A3: report as robustness-across-λ, not a tuned magic value |
| MAE+SupMAE joint FT | `LOSO-maeSupmaeJointFT-Calibrate/` | diagnostic |
| sub8 studies | `sub8_adaptation.py`, `sub8_calibrate_test_others.py` results | relevant again: sub8 anomaly under [[swing-mode]] |
| Latent visualizations | `UMAP_TargetLatents/` | UMAP/t-SNE of target latents ([[temp-scripts]] viz section) |
| MAE k=3 = 20.70 anomaly | — | resolved: stale/buggy partial run; current MAE k=3 = 87.35, monotone (RESEARCH_LOG §B3) |
