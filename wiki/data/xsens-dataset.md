---
type: data
status: active
updated: 2026-07-10
---

# Xsens IMU (target domain)

- **Raw:** `DataCollection/sub{7–11}/` — `.mvnx` + `.xlsx` exports. Annotations: `DataCollection/Annotations/Omari_Annotations/` (`.anvil` per subject/session); `DataCollection/GroundTruth-Annotations-Offset.csv` maps annotation time to data offsets. **PII note:** raw `.mvnx` headers embed subjects' real first names in the `originalFilename` XML attribute — never export/release `DataCollection/` without de-identifying first ([[open-questions]] T6).
- **Processed, current/locked representation:** `Data_Processed/imu_quats_v2/` — 2 736 `.npy` clips, `(T, 17, 4)` [[lrq]], `mode=reconstructed`. Rebuilt from mvnx **positions** through NTU's own shortest-arc bone construction ([[position-reconstruction-v2]]) — this is what every current experiment (LOSO-v2, Phase 1–3, A2, external-validity runs) trains against.
- **Processed, superseded (kept for provenance):** `Data_Processed/imu_quats/` — same clip/`index.csv` schema, generated from *measured* mvnx/xlsx orientation. Regenerated 2026-06-30 in `mode=swing` (axial-twist-stripped, [[swing-mode]]); an earlier `mode=local` (full measured orientation, twist included) generation preceded it. Both modes are dead ends — swing widened the NTU↔Xsens gap ~3× and v2 is what's used everywhere now. Do not regenerate either unless resurrecting swing as a deliberate ablation (not currently planned).
- **Subjects:** sub7–sub11, sessions `a, b1, b2, c` each. sub7 = known outlier under local mode (mounting twist, ~15–20 pp below others) — resolved under v2 (accounts for most of v2's per-subject gain). sub8's swing-mode degradation (−19.9 pp) is **resolved**: it was a twist-stripping artifact, gone under v2 (−1.0 pp, a tie) — see [[swing-mode]].
- **22 gestures:** wave, bow, crossarms, airpunch, drink, brushteeth, highfive, airkick, pickup, throw, hop, stand, sit, pushchair, squat, crosstoe, jump, sidekick, runonspot, buttkicks, turnaround, walk. OOV reliability varies — crossarms/squat/wave reliable; throw/jump/hop not ([[oov-leave-class-out]]); this drives safety-command assignment in the controller ([[paper-framing]]).
- **Processors:** `src/scripts/IMU_batch_processor.py` (v1, local/swing, orientation-based — superseded) and `src/scripts/IMU_batch_processor_v2.py` (v2, current, position-based). Both parse mvnx/xlsx, apply Anvil timing offsets, downsample 240→30 Hz (stride 8). See [[data-pipeline]].
- **Loader:** `src/data/imu_loader.py` (some downstream scripts duplicate it inline).
