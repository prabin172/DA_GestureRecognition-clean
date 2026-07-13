---
type: experiment
status: dead-end
updated: 2026-07-03
---

# Cleaned-source (slerp035) pretraining

**Question:** do NTU quaternion sign-flips/discontinuities hurt pretraining? Clean them (slerp smoothing, threshold 0.35) and compare.

- Cleaning scripts: `temp_clean_ntu_quats.py`, `temp_fix_ntu_quat_discontinuities.py`; cleaned data in `temp_outputs/ntu_quats_cleaned_slerp035/` (+ smoke/variant dirs).
- Pretraining: `trained_models/pretrain_cleaned_slerp035*/`; comparison: `temp_compare_cleaned_source_pretraining.py`, `temp_compare_cleaned_with_sup.py`.
- LOSO results: `trained_models/LOSO-fullTrainCalibrate-cleanedSlerp035/`, `-supervised/`.

## Verdict: dead end
No consistent improvement. supMAE_clean beats supMAE at k=10 (96.3 vs 95.6) but loses at k=1 (77.0 vs 79.9). **Cleaning not needed.** (RESEARCH_LOG §B2.)
