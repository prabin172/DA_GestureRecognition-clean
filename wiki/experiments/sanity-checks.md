---
type: experiment
status: done
updated: 2026-07-03
---

# Sanity checks (within-domain controls)

## NTU→NTU objective sanity (Pillar 1 evidence)
Linear probe (30 ep) on NTU for each pretrained encoder. Script: `scripts/main_experiment/ntu_to_ntu_objective_sanity.py`. Dir: `trained_models/NTU-to-NTU-objective-sanity/` (+ smoke).

| method | final_acc |
|---|---|
| supervised | 59.6% |
| supMAE | 59.2% |
| MAE | 22.6% |
| scratch | 14.8% |

Objectives matter hugely within-domain (~45 pp spread); MAE alone is not discriminative — consistent with frozen-MAE collapse in [[xsens-to-xsens-loso]].

## Xsens-only LOSO sanity
`temp_xsens_only_loso_objective_sanity.py` → `trained_models/XsensOnly-LOSO-objective-sanity-smoke/` — smoke only; superseded by the full [[xsens-to-xsens-loso]] run.

## Quaternion pipeline QA
`temp_quat_pipeline_audit.py`, `temp_quat_quant_sanity.py`, `temp_label_annotation_sanity.py`, NTU quaternion-smoothing sanity (`temp_outputs/quat_sanity/`, slerp smoke dirs). No unit tests exist; these scripts are the correctness record ([[temp-scripts]]).
