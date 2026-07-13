---
type: result
status: active
updated: 2026-07-10
---

# UTD-MHAD skeleton LOSO — second independent same-modality dataset (skeleton→skeleton)

A **second, independently collected public dataset** to test whether the small-gap "supervised
prior wins" result (R6, [[czu-skeleton-loso]]) replicates beyond CZU. Same objectives, same LOSO
k-shot protocol; NTU skeleton prior → UTD-MHAD Kinect skeleton. This is the mirror image of the
cross-modal inversion ([[czu-imu-crossmodal]] R6b): where the gap is small, the pure-supervised
prior is best-in-class.

- Run: `trained_models/UTD-skeleton-LOSO-seed{42,43,44}/summary.csv` — **3 seeds**, 8-subj LOSO
  (s1–s8), k=0/1/3 → 24 folds pooled per cell. Data `Data_Processed/utd_skeleton_lrq/`
  (861 clips, 27 actions, 0 skipped).
- Parser `temp_utd_parser.py`: Kinect-v1 20-joint → NTU-25 remap (Neck + SpineShoulder both ←
  UTD `shoulder_center`); smoke-validated (unit quats, bone-CoV 0.02–0.04, head-above-base +0.59 m).
- Launched via `temp_utd_run.sh` (nohup → `czu_utd_run.log`, `UTD-SKELETON ALL DONE`). Pooled
  stats: same recipe as `temp_czu_multiseed_analyze.py`.

## Final accuracy — per-(method, k) mean, pooled 3 seeds × 8 subj (n=24)

| k | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|
| 0 (zero-shot) | 77.37 | 80.30 | 80.97 | **84.71** |
| 1 | 90.92 | 92.26 | 93.51 | **94.84** |
| 3 | 93.95 | 95.19 | 94.11 | **96.40** |

## Finding — the small-gap "supervised wins" result replicates cleanly (and with more power)

- **supLP120 is the best prior at every k.** supLP120 − scratch = **+7.35 / +3.92 / +2.45 pp**
  (paired-t p<.0001 / .0005 / .016, n=24). Sign test supLP120 > scratch **53/62 folds (p<.0001)**;
  supLP120 > supMAE **40/53 (p=.0003)**. This mirrors CZU R6 (supLP120 +7.2 pp @k0) on an
  independent dataset — the pure-supervised prior wins when the gap is small.
- **supMAE also significant-positive** (+3.60 / +2.59 / +0.15 pp, p=.0002 / .007 / n.s.; sign
  43/59, p=.0006) — but below supLP120.
- **mae is *positive*, not negative** (+1.84 pp mean; sign 42/63, p=.011). As on CZU skeleton, the
  "reconstruction hurts" signature ([[multiseed-loso-v2]] C2) does **not** appear on a same-modality
  small-gap target — it is specific to the cross-modal gap.

## CRC published-baseline anchor (T4, 2026-07-09)

Same recognizer family as the CZU R6 anchor — statistical-moment features (mean/std/var/skew/kurtosis)
+ Collaborative Representation Classifier (CRC-RLS, λ=1e-4) — on the **byte-identical seed-42 LOSO
k-shot splits** (`temp_utd_crc_baseline.py` → `trained_models/UTD-skeleton-LOSO-seed42/crc_baseline/`).
Deterministic given splits, no seeds needed.

| k | CRC baseline | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|---|
| 0 | 69.48 | 75.98 | 79.46 | 82.02 | **84.33** |
| 1 | 92.25 | 90.25 | 92.56 | 93.50 | **94.27** |
| 3 | 95.30 | 93.52 | 94.41 | 94.41 | **96.24** |

supLP120 − CRC = **+14.85 / +2.02 / +0.94 pp** at k=0/1/3 — the transferred prior opens the same
kind of consistent margin over the published-baseline recognizer family that R6 found on CZU.
One difference from CZU worth noting honestly: on CZU, the **scratch** learned encoder was
statistically level with CRC (84.87 vs 84.81 @k0) — "architecture alone is not the advantage."
On UTD, **scratch already beats CRC by +6.5 pp at k=0** (75.98 vs 69.48) before any prior is
added — the learned architecture has some edge here even without pretraining, so the prior's
marginal contribution on top of scratch (+8.35 pp @k0, supLP120−scratch) is smaller than its
margin over CRC. Both readings agree the prior helps most at k=0 and the gap narrows by k=3.

*(TODO — not yet done: cite UTD-MHAD's own published cross-subject accuracy figure as a second
external reference point, per tasks.md T4. Needs a literature figure, not compute — flag for
Planning/human to supply the citation before this goes in the paper.)*

## supcon extension (2026-07-10, pooled 3 seeds n=24)

`trained_models/UTD-skeleton-LOSO-supcon-seed{42,43,44}/summary.csv`. supcon − scratch = **+8.36 /
+4.02 / +2.94 pp @ k=0/1/3, all significant** (p<.0001 / p<.0001 / p=.006); sign test supcon>scratch
**53/59, p<.0001** — matching supLP120's own margin closely. supcon vs supLP120 is statistically even
here (25/52, p=.89, meanΔ +0.5 n.s.) — unlike CZU (where supcon significantly beats supLP120, p=.0002),
on UTD the two label-aware objectives are indistinguishable, both clearly dominating scratch. Combined,
the two-dataset picture is **both supLP120 and supcon win big at small gap, with which one wins by
more varying by dataset** — the operative class is "label-supervised, no reconstruction," not one
specific recipe. Feeds `paper_results.md` R6d + the R6e five-setting map's supcon paragraph.

## Why it matters

Two independent public datasets now agree on the small-gap end of the arc:

| dataset | zero-shot supLP120 − scratch | mae vs scratch |
|---|---|---|
| CZU skeleton (R6) | +7.2 pp (pooled p=.0004) | positive (+2.3, n.s.) |
| **UTD skeleton (R6d)** | **+7.4 pp (p<.0001)** | **positive (+1.8, p=.011)** |

Combined with the cross-modal inversion ([[czu-imu-crossmodal]] R6b: supLP120 *worst*, below
scratch) and the strong-target null ([[czu-imu-dual]] R6c), the full arc is: **supervised prior
best-in-class at small gap (two datasets) → worst-in-class across the cross-modal gap → still worst
once the target is strong.** The prior's value is gap-contingent — the negative-transfer thesis as a
design rule.

Feeds `paper/paper_results.md` R6d. See also [[czu-skeleton-loso]] · [[czu-imu-crossmodal]] ·
[[czu-imu-dual]] · [[multiseed-loso-v2]].
