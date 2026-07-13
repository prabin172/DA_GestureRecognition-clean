---
type: experiment
status: done
updated: 2026-07-04
---

# LOSO full-train + calibrate (the main experiment)

**Protocol** ([[loso-protocol]]): init encoder (NTU-pretrained per [[ntu-pretraining]], or scratch) → full fine-tune ~80 ep on the 4 non-held-out subjects → head-only calibration with k shots of the held-out subject → `final_acc` on that subject. Script: `temp_loso_fulltrain_calibration.py`. Single seed, deterministic per subject/tag/k — violates the A3 stats policy ([[publishability-review]] item 7).

## Local mode — `trained_models/LOSO-fullTrainCalibrate/` (2026-06-29, 175 rows incl. dann/supcon)
Mean final_acc over 5 subjects:

| method | k=0 | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| supMAE | 50.3 | **79.9** | **89.9** | 90.5 | 95.6 |
| targetSupDANN | 49.1 | 79.8 | 88.8 | **91.1** | 95.7 |
| MAE | 51.4 | 76.9 | 87.3 | 89.7 | **95.8** |
| scratch | 50.7 | 76.9 | 85.9 | 88.3 | 95.3 |
| sourceSupDANN | 48.8 | 74.0 | 86.3 | 88.1 | 94.0 |
| supLP120 | 45.0 | 72.6 | 84.6 | 86.0 | 93.7 |
| supcon | 50.8 | 71.1 | 83.5 | 88.5 | 92.6 |

- k=1 spread ~7 pp vs ~45 pp within-domain → Pillar 2 compression ([[paper-framing]]).
- **supLP120 = negative transfer** at low k. **SupCon < scratch at every k>0** despite near-lowest MMD → contrastive scoped out (interpretation: NTU-tuned cluster geometry leaves no room for the k-shot head).
- sub7 consistently ~15–20 pp below others (local mode) — means alone insufficient.

## Swing mode (Job 1) — `trained_models/LOSO-fullTrainCalibrate-swing/` (2026-07-01, 100 rows)
DANN excluded (stale encoders). Mean final_acc:

| method | k=0 | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| scratch | 74.4 | **88.2** | 92.2 | 94.6 | 96.4 |
| supLP120 | 69.5 | 83.2 | 91.3 | 94.0 | 96.4 |
| supMAE | **74.5** | 86.9 | **94.8** | **95.0** | **97.5** |
| mae | 73.8 | 86.5 | 93.0 | 94.8 | 96.8 |

- **Scratch wins k=1 outright** (4/5 subjects; supMAE−scratch per-subj: −4.4, −1.6, +4.4, −4.6, −0.2). Only sign-consistent supMAE win: k=3 (+2.6, 5/5).
- Full 80-ep fine-tune on 4 subjects washes out the init — pretraining benefit small/compressed.
- Swing vs local absolute jump is subject-driven — see [[swing-mode]] (sub7 +54.6, sub8 −19.9).

## v2 (position-reconstructed Xsens) — DONE (2026-07-04, single seed)
`trained_models/LOSO-fullTrainCalibrate-v2/`, log `loso_v2.log`. Data = `Data_Processed/imu_quats_v2/` ([[position-reconstruction-v2]]) via `LOSO_IMU_DIR` env override. Mean final_acc:

| method | k=0 | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| supMAE | **57.5** | **84.0** | 90.8 | 93.2 | 96.4 |
| supLP120 | 57.4 | 80.3 | **90.8** | **94.4** | 96.3 |
| scratch | 56.8 | 79.9 | 89.8 | 92.5 | 96.3 |
| mae | 50.7 | 77.0 | 86.6 | 91.2 | 94.8 |

- **Prior benefit restored + amplified.** supMAE−scratch @ k=1 = **+4.1 pp** (local +3.0, swing −1.3) → the gap knob (MMD supmae: local 0.0109, swing 0.0322, v2 0.0092) moves monotonically with the prior's value. 4/5 subjects positive (sub7 +9.7, sub10 +5.2, sub11 +4.3, sub9 +2.3, sub8 −1.0).
- **sub8 swing anomaly gone:** v2 sub8 supMAE−scratch @ k=1 = −1.0 pp (swing was −19.9). The swing collapse was a twist-stripping artifact.
- **supLP120 negative transfer fixed:** +0.4 pp @ k=1 (local was −4.3). mae now weakest.
- **v2 = the chosen preprocessing** ([[position-reconstruction-v2]]). The +4.1/+9.7/etc. above are **seed-42 only**.
- **Multi-seed DONE (3 seeds) → [[multiseed-loso-v2]].** The k=1 +4.1 pp is a seed-42 artifact (seed43 −1.77 → pooled +0.97, n.s.). Seed-robust prior benefit lives at k=0 (+2.52) / k=3 (+1.85, p=.054) and — more clearly — in **AUC-30** (supMAE k=3 +2.04 p=.028, supLP120 k=3 +3.74 p=.010) and **convergence speed** (supLP120 ~3–4 epochs faster, p≤.012). **mae = significant negative transfer** on every metric.

Cleaned-source variants: [[cleaned-source-pretraining]]. Swing synthesis + caveats: [[swing-mode-findings]].
