---
type: result
status: active
updated: 2026-07-10
---

# CZU-MHAD skeleton LOSO — external validity (skeleton→skeleton)

Tier C external-target check: do NTU-pretrained priors help on an **independent public** dataset?
This page was stale (single-seed only) through 2026-07-09 even though the pooled numbers had
landed in `paper/paper_results.md` R6 on 2026-07-07 — brought current here 2026-07-10 alongside
the supcon extension.

- Run: `trained_models/CZU-skeleton-LOSO{,-seed43,-seed44,-supcon-seed42,-supcon-seed43,-supcon-seed44}/`
  — **4 objectives × 3 seeds pooled** (scratch/mae/supMAE/supLP120) **+ supcon × 3 seeds pooled
  separately**. Data `Data_Processed/czu_skeleton_lrq/` (1165 clips, 5 subj:
  cx/cyy/myj/qyh/zyh, 22 actions). Parser `temp_czu_parser.py` (CZU Kinect-v2 25-joint order =
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
pure-supervised priors don't. `trained_models/CZU-IMU-LOSO/` (`temp_czu_imu_parser.py`).

## Final accuracy — per-(method,k) mean, single-seed (n=5, historical)

| k | scratch | mae | supMAE | supLP120 |
|---|---------|-----|--------|----------|
| 0 (zero-shot) | 84.9 | 87.9 | 85.5 | **93.1** |
| 1 | 90.1 | 90.0 | 91.2 | 93.1 |
| 3 | 91.9 | 93.8 | 93.6 | 93.8 |

## Final accuracy — pooled 3 seeds (n=15), scratch vs supcon (own pooled run)

| k | scratch | supcon |
|---|---------|--------|
| 0 (zero-shot) | 85.04 | **92.04** |
| 1 | 89.58 | **94.37** |
| 3 | 91.82 | **96.23** |

(scratch/mae/supMAE/supLP120 pooled means live in `paper/paper_results.md` R6 — this page's
single-seed table above predates that pooling and is kept for the historical p=.07 reference; the
paired-Δ table below has the pooled, current numbers for all five objectives.)

Paired Δ vs scratch, pooled n=15:

| | k=0 | k=1 | k=3 |
|---|-----|-----|-----|
| supLP120−scratch | **+7.21 pp, p=.0004** | +2.57 pp, p=.067 (n.s.) | +2.10 pp, p=.073 (n.s.) |
| supMAE−scratch | +2.23 pp, p=.048 | +1.60 pp, p=.14 (n.s.) | +2.15 pp, p=.014 |
| mae−scratch | +2.34 pp, p=.11 (n.s.) | −0.17 pp, p=.91 (n.s.) | +0.88 pp, p=.52 (n.s.) |
| **supcon−scratch** | **+7.00 pp, p<.0001** | **+4.79 pp, p=.0002** | **+4.41 pp, p=.0001** |

**supcon is the only objective significant at all three k's** — supLP120's own significance is
concentrated at k=0 (its k=1/k=3 pooled p-values are only marginal, .07/.07), so supcon's margin is
not just numerically larger but statistically more robust across shot counts on this dataset.

**supcon − supLP120 = +1.44 pp, sign 34/43 folds, p=.0002** — supcon significantly beats supLP120
on this dataset (though the two are statistically even on UTD, see [[utd-skeleton-loso]] R6d) — both
clearly dominate scratch. supcon is significant at **every** k (unlike supLP120, whose accuracy-level
significance is strongest at k=0/k=3 per [[multiseed-loso-v2]]'s k=1 caveat).

## AUC-30 (mean eval_acc % over first 30 calib epochs, single-seed)

| k | scratch | mae | supMAE | supLP120 |
|---|---------|-----|--------|----------|
| 1 | 89.7 | 89.8 | 91.0 | 93.2 |
| 3 | 90.9 | 92.9 | 92.5 | 93.9 |

## Convergence

Uninformative — CZU is near-ceiling (85–94%) and every method reaches ≥90% of final in ~1 epoch (floor).

## Numbers worth noting

- **supLP120 gives a large, pooled-significant zero-shot benefit** (+7.2 pp @ k=0, p=.0004 — sharper
  than the single-seed +8.3 pp @ p=.07 that originally motivated this page).
- supMAE modestly positive and now significant at k=0/k=3 under pooling.
- **mae is *positive* here** (+2.2 pp mean, n.s.), unlike NTU→Xsens where mae is significant negative
  transfer ([[multiseed-loso-v2]]) — the mae-hurts signature does not reproduce on the same-modality
  target.
- **supcon wins even more decisively than supLP120** — significant at every k, and significantly
  ahead of supLP120 itself (p=.0002). See [[czu-imu-dual]]'s R6e five-setting synthesis for the
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

**Our reproduction** (`temp_czu_crc_baseline.py` → `trained_models/CZU-skeleton-LOSO/crc_baseline/`):
their feature-family (mean/std/var/skew/kurtosis) + CRC, applied to our LRQ representation on the
**byte-identical LOSO k-shot splits** (read from `splits/*_calibration_split.json`). Head-to-head,
mean acc over 5 folds:

| k | CRC baseline | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|---|
| 0 | 84.81 | 84.87 | 87.94 | 85.50 | **93.14** |
| 1 | 89.31 | 90.15 | 89.98 | 91.19 | **93.08** |
| 3 | 90.82 | 91.90 | 93.77 | 93.62 | **93.85** |

- **scratch ≈ CRC** (84.87 vs 84.81 @ k=0): the *learned-from-scratch* encoder barely beats the
  hand-crafted baseline — the architecture alone is not the win.
- **Only the NTU prior opens a clear, consistent gap**: supLP120 − CRC = **+8.3 / +3.8 / +3.0 pp**
  @ k=0/1/3. External-validity evidence that the *transferred prior* (not the encoder) carries the value.
- Our CRC (84.8%) > their published cross-subject skeleton (~75.5%) — expected (LRQ + parent-relative
  + 4-subject LOSO dictionary vs their raw positions + T5–T7 splits). Report as a same-splits
  reproduction, not a claim about their exact pipeline.
- Strongest paper use: the CRC row is a same-split baseline in the CZU table; cite their published
  numbers as an independent reference point (addresses the single-dataset-pair reviewer risk).

See also [[loso-fulltrain-calibrate]] · [[multiseed-loso-v2]].
