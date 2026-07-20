---
type: code
status: active
updated: 2026-07-03
---

# Root temp scripts (the working set)

~30 root-level `temp_*.py` one-offs. Despite the name, the **current main experiment pipeline lives here**, not in `src/`. Matching `.log` files hold run output; long runs use `nohup` with logs in repo root.

## Load-bearing (current pipeline)
| script | role |
|---|---|
| `scripts/main_experiment/loso_fulltrain_calibration.py` | **THE main experiment script** — full-LOSO base training + k-shot head-only calibration → [[loso-fulltrain-calibrate]] |
| `temp_xsens_to_xsens_loso_calibration.py` | Xsens→Xsens objective LOSO (Job 2); imports the above as `L` so calibration cannot drift. **`sup=True` at lines 121–124 adds CE — its "supmae" is NOT self-supervised** ([[publishability-review]]) → [[xsens-to-xsens-loso]] |
| `scripts/main_experiment/mmd_domain_gap.py` | MMD NTU↔Xsens per encoder → [[mmd-domain-gap]] |
| `scripts/main_experiment/loso_leave_class_out_fewshot.py` | OOV protocol → [[oov-leave-class-out]] |
| `scripts/pretrain/pretrain_supcon.py` | SupCon NTU pretraining |
| `scripts/main_experiment/ntu_to_ntu_objective_sanity.py`, `temp_xsens_only_loso_objective_sanity.py` | within-domain sanity → [[sanity-checks]] |

## Diagnostics / one-offs
- Quaternion QA: `temp_quat_pipeline_audit.py`, `temp_quat_quant_sanity.py`, `temp_label_annotation_sanity.py`, `temp_clean_ntu_quats.py`, `temp_fix_ntu_quat_discontinuities.py` ([[cleaned-source-pretraining]])
- Cleaned-source comparison: `temp_compare_cleaned_source_pretraining.py`, `temp_compare_cleaned_with_sup.py`
- Ablations: `temp_source_relatedness_ablation.py`, `temp_supmae_lambda_sweep.py`, `scripts/main_experiment/loso_leave_class_out_fewshot.py`, `temp_loso_mae_supmae_jointft_calibration.py`
- DSTformer: `temp_overfit_dstformer_quat.py`, `temp_test_dstformer_quat_encoder.py`, `temp_loso_dstformer_quat_target_only.py` ([[early-experiments]])
- Visualization: `temp_umap_target_latents.py`, `temp_tsne_finetuned_sub8_highlight.py`, `temp_visualize_ntu_xsens_quats.py`, `temp_visualize_quat_axes.py`, `temp_visualize_raw_positions.py`, `temp_plot_*.py`
- Misc: `temp_run_overnight_diagnostics.sh`, `verify_models.py`, outputs in `temp_outputs/`
