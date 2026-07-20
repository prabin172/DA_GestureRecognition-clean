---
type: result
status: active
updated: 2026-07-20
---

**2026-07-20: numbers below refreshed from `DA_GestureRecognition-clean`'s independently-retrained checkpoints** — `trained_models/CZU-skeleton-LOSO{,-seed43,-seed44}/summary.csv`, 5 methods trained natively together (no separate supcon-seed dirs in this rerun). Directions all hold; supLP120's zero-shot margin is somewhat smaller (+5.0pp vs +7.2pp) but still clearly significant. See `paper/paper_results.md` R6.

# CZU-MHAD skeleton LOSO — external validity (skeleton→skeleton)

Tier C external-target check: do NTU-pretrained priors help on an **independent public** dataset?
This page was stale (single-seed only) through 2026-07-09 even though the pooled numbers had
landed in `paper/paper_results.md` R6 on 2026-07-07 — brought current here 2026-07-10 alongside
the supcon extension.

- Run: `trained_models/CZU-skeleton-LOSO{,-seed43,-seed44,-supcon-seed42,-supcon-seed43,-supcon-seed44}/`
  — **4 objectives × 3 seeds pooled** (scratch/mae/supMAE/supLP120) **+ supcon × 3 seeds pooled
  separately**. Data `Data_Processed/czu_skeleton_lrq/` (1165 clips, 5 subj:
  cx/cyy/myj/qyh/zyh, 22 actions). Parser `scripts/data_pipeline/czu_parser.py` (CZU Kinect-v2 25-joint order =
  NTU's → reuses `ntu_parser.process_to_local_quats`).
- **This is skeleton→skeleton** (NTU skeleton pretrain → CZU skeleton target) — same modality,
  much smaller domain gap than the NTU→Xsens IMU path. Not the IMU gap.
- Metrics as in [[multiseed-loso-v2]]. n=15 folds (5 subj × 3 seeds) per cell.

## CZU inertial cross-modal — DONE (see [[czu-imu-crossmodal]])

CZU-MHAD ships three modalities: `skeleton_mat/`, `sensor_mat/`, `depth_mat/`. **`sensor_mat/`
= 10 body-worn inertial sensors** (accel+gyro, 6-axis, **no magnetometer** → yaw drift), 1165 clips
at `external_data/czu_mhad_data/CZU-MHAD/sensor_mat/`. This page uses **only the skeleton modality**.
The true skeleton→IMU external replication is now done → [[czu-imu-crossmodal]]: the prior ordering
**inverts** (supMAE best, supLP120 worst) — reconstruction-augmented priors transfer across modality,
pure-supervised priors don't. `trained_models/CZU-IMU-LOSO/` (`scripts/data_pipeline/czu_imu_parser.py`).

## Final accuracy — per-(method,k) mean, seed 42 only (n=5, matches the CRC same-splits table below)

| k | scratch | mae | supMAE | supLP120 |
|---|---------|-----|--------|----------|
| 0 (zero-shot) | 84.9 | 85.0 | 89.0 | **89.9** |
| 1 | 90.2 | 88.5 | 92.8 | **92.5** |
| 3 | 91.9 | 91.5 | 94.9 | **95.0** |

## Final accuracy — pooled 3 seeds (n=15/method/k), all 5 methods trained natively together

| k | scratch | mae | supMAE | supLP120 | supcon |
|---|---------|-----|--------|----------|--------|
| 0 (zero-shot) | 85.04 | 85.35 | 88.18 | **90.03** | **90.27** |
| 1 | 89.58 | 88.49 | 91.69 | 92.20 | **94.25** |
| 3 | 91.82 | 91.75 | 93.67 | 94.80 | **95.23** |

Paired Δ vs scratch, pooled n=15:

| | k=0 | k=1 | k=3 |
|---|-----|-----|-----|
| supLP120−scratch | **+5.00 pp, p=.0012** | +2.61 pp, p=.053 (n.s.) | **+2.98 pp, p=.0085** |
| supMAE−scratch | **+3.15 pp, p=.010** | **+2.11 pp, p=.032** | +1.85 pp, p=.086 (n.s.) |
| mae−scratch | +0.31 pp, p=.80 (n.s.) | −1.09 pp, p=.41 (n.s.) | −0.07 pp, p=.96 (n.s.) |
| **supcon−scratch** | **+5.24 pp, p=.0015** | **+4.67 pp, p=.0003** | **+3.41 pp, p=.0049** |

**supcon is the only objective significant at all three k's**, same as originally found — supLP120's
own significance is now concentrated at k=0/k=3 (k=1 is a trend, p=.053), so supcon's margin is
not just numerically larger but statistically more robust across shot counts on this dataset.

**supcon − supLP120 = +0.91 pp, sign 28/42 folds, p=.044** — supcon still beats supLP120
on this dataset (though the margin and its significance are both smaller than the original run's
p=.0002) — both clearly dominate scratch. supcon is significant at **every** k, unlike supLP120.

## AUC-30 (mean eval_acc % over first 30 calib epochs, single-seed)

| k | scratch | mae | supMAE | supLP120 |
|---|---------|-----|--------|----------|
| 1 | 89.7 | 89.8 | 91.0 | 93.2 |
| 3 | 90.9 | 92.9 | 92.5 | 93.9 |

## Convergence

Uninformative — CZU is near-ceiling (85–94%) and every method reaches ≥90% of final in ~1 epoch (floor).

## Numbers worth noting

- **supLP120 gives a large, pooled-significant zero-shot benefit** (+5.0 pp @ k=0, p=.0012 — smaller
  than the original run's +7.2 pp but still clearly significant, independently reproduced).
- supMAE modestly positive and now significant at k=0/k=1 under pooling (was k=0/k=3).
- **mae is not negative here** (+0.3/−1.1/−0.1 pp, all n.s.), unlike NTU→Xsens where mae is significant
  negative transfer ([[multiseed-loso-v2]]) — the mae-hurts signature does not reproduce on the
  same-modality target.
- **supcon wins even more decisively than supLP120** — significant at every k, and beats supLP120
  itself (p=.044). See [[czu-imu-dual]]'s R6e five-setting synthesis for the
  reframe this drives: both label-aware objectives (softmax supLP120, contrastive supcon) win big
  at small gap; what distinguishes the winners from the large-gap losers is the presence/absence of
  a reconstruction component, not the specific supervised recipe.

## Published-baseline comparison (CZU paper + CRC reproduction on identical splits)

**CZU-MHAD dataset paper** (Chao et al., *IEEE Sensors J.* 2022; arXiv 2202.03283): NOT LOSO,
NOT deep — Collaborative Representation Classifier (CRC, λ=1e-4) on statistical-moment features.
Two split families: **T1–T4 closed** (subjects mixed) vs **T5–T7 open** (cross-subject). Their
cross-subject (open) numbers, per modality: **skeleton ~75.5%**, inertial(10-sensor) ~65–85%,
depth(DMM-HOG) 88–90%, fusion ~84–96%. **Only T5–T7 is protocol-comparable to our LOSO** — never
compare to their ~97% closed test.

**Our reproduction** (`scripts/external/czu/crc_baseline.py` → `trained_models/CZU-skeleton-LOSO/crc_baseline/`):
their feature-family (mean/std/var/skew/kurtosis) + CRC, applied to our LRQ representation on the
**byte-identical LOSO k-shot splits** (read from `splits/*_calibration_split.json`). Head-to-head,
mean acc over 5 folds:

| k | CRC baseline | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|---|
| 0 | 84.81 | 84.87 | 84.95 | 89.03 | **89.94** |
| 1 | 89.31 | 90.15 | 88.48 | 92.76 | **92.54** |
| 3 | 90.82 | 91.90 | 91.46 | 94.92 | **95.04** |

- **scratch ≈ CRC** (84.87 vs 84.81 @ k=0, exact reproduction — scratch never touches a pretrained
  checkpoint): the *learned-from-scratch* encoder barely beats the hand-crafted baseline — the
  architecture alone is not the win.
- **Only the NTU prior opens a clear, consistent gap**: supLP120 − CRC = **+5.1 / +3.2 / +4.2 pp**
  @ k=0/1/3 (smaller at k=0 than the original run's +8.3 pp, similar at k=1/k=3). External-validity
  evidence that the *transferred prior* (not the encoder) carries the value.
- Our CRC (84.8%) > their published cross-subject skeleton (~75.5%) — expected (LRQ + parent-relative
  + 4-subject LOSO dictionary vs their raw positions + T5–T7 splits). Report as a same-splits
  reproduction, not a claim about their exact pipeline.
- Strongest paper use: the CRC row is a same-split baseline in the CZU table; cite their published
  numbers as an independent reference point (addresses the single-dataset-pair reviewer risk).

See also [[loso-fulltrain-calibrate]] · [[multiseed-loso-v2]].
