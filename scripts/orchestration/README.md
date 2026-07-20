# Rerun orchestration

`RUN_FULL_RERUN.sh` chains all 10 stages below, sequentially, single GPU, one nohup process.
Each stage is also runnable standalone (e.g. to resume after a crash — comment out completed
stages in `RUN_FULL_RERUN.sh`, or just run the single stage script directly).

| stage | script | what it does | depends on |
|---|---|---|---|
| 0 | `00_data_pipeline.sh` | builds every `Data_Processed/` subdir (Xsens v2, CZU skeleton/IMU-quat/IMU-raw/mag20, UTD skeleton) from raw `DataCollection/`+`external_data/` | raw data (already copied, see `migration/MANIFEST.md`) |
| 1 | `01_pretrain.sh` | NTU pretraining, 4 objectives (supLP120, mae, supMAE, supcon); scratch = random init, no pretrain needed | `Data_Processed/ntu_quats/` (already copied) |
| 2 | `02_main_loso.sh` | **the main experiment.** 3 seeds x 5 methods x k{0,1,3}, Xsens LOSO (4 subjects train, 5th held out, all folds) | stage 0 (imu_quats_v2) + stage 1 (checkpoints) |
| 3 | `03_main_analysis.sh` | posteriors, McNemar, ECE, CKA (per-encoder + per-target), encoder-space MMD, raw encoder-free MMD²/Frechet | stage 2 |
| 4 | `04_a2_subject_scaling.sh` | subject-count scaling: N∈{0..4} x 3 seeds x 5 methods, then pooled paired-t stats | stage 0 + 1 |
| 5 | `05_oov.sh` | leave-class-out few-shot onboarding | stage 0 + 1 |
| 6 | `06_czu_external.sh` | CZU-MHAD: skeleton (R6), IMU-quat (R6b), dual-branch raw/quat/dual (R6c), DIAL dose-response, CRC baselines, 3-seed pooling | stage 0 + 1 |
| 7 | `07_utd_external.sh` | UTD-MHAD skeleton (R6d), CRC baseline, 3 seeds | stage 0 + 1 |
| 8 | `08_controller.sh` | downstream controller — prototype (cited headline number) + robustness protocol (3 locks) | stage 2 + 3 (posteriors) |
| 9 | `09_czu_cold_start.sh` | T5: does the A2 cold-start lever survive on a strong target? Single seed, scoping check | stage 0 + 1 |

**Verified before launch (2026-07-13):** every script's root-path resolution numerically checked;
`--dry-run` smoke test on the main LOSO script confirmed correct checkpoint paths for all 5
methods; `IMU_batch_processor_v2.py --prototype` ran against the real copied `DataCollection/`
data and passed (`PROTOTYPE PASS`). Not yet verified: full end-to-end completion of any stage —
that's what this run is for.

`legacy/` holds the original repo's orchestration scripts for reference (exact flags used for
the numbers currently in `paper_results.md`) — **do not run them here**, see `legacy/README.md`
for why they'd fail as-is.
