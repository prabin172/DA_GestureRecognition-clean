---
type: result
status: active
updated: 2026-07-20
---

**2026-07-20: numbers refreshed from `DA_GestureRecognition-clean`'s independently-retrained checkpoints — supLP120 vs supMAE loses significance (was p=.0003, now p=.20), but supcon is the standout, now clearly beating supLP120 (was a wash).** See `paper/paper_results.md` R6d.

# UTD-MHAD skeleton LOSO — second independent same-modality dataset (skeleton→skeleton)

A **second, independently collected public dataset** to test whether the small-gap "supervised
prior wins" result (R6, [[czu-skeleton-loso]]) replicates beyond CZU. Same objectives, same LOSO
k-shot protocol; NTU skeleton prior → UTD-MHAD Kinect skeleton. This is the mirror image of the
cross-modal inversion ([[czu-imu-crossmodal]] R6b): where the gap is small, the pure-supervised
prior is best-in-class.

- Run: `trained_models/UTD-skeleton-LOSO-seed{42,43,44}/summary.csv` — **3 seeds**, 8-subj LOSO
  (s1–s8), k=0/1/3 → 24 folds pooled per cell. Data `Data_Processed/utd_skeleton_lrq/`
  (861 clips, 27 actions, 0 skipped).
- Parser `scripts/data_pipeline/utd_parser.py`: Kinect-v1 20-joint → NTU-25 remap (Neck + SpineShoulder both ←
  UTD `shoulder_center`); smoke-validated (unit quats, bone-CoV 0.02–0.04, head-above-base +0.59 m).
- Launched via `scripts/orchestration/utd_run.sh` (nohup → `czu_utd_run.log`, `UTD-SKELETON ALL DONE`). Pooled
  stats: same recipe as `scripts/external/czu/multiseed_analyze.py`.

## Final accuracy — per-(method, k) mean, pooled 3 seeds × 8 subj (n=24)

| k | scratch | mae | supMAE | supLP120 | supcon |
|---|---|---|---|---|---|
| 0 (zero-shot) | 77.37 | 79.60 | 79.99 | 82.40 | **85.34** |
| 1 | 90.92 | 91.64 | 93.14 | 93.03 | **94.90** |
| 3 | 93.95 | 94.54 | 95.05 | 95.97 | **96.41** |

(2026-07-20 rerun, pooled 3 seeds × 8 subj, 5 methods trained natively.)

## Finding — the small-gap "supervised wins" pattern replicates, though supLP120 vs supMAE is now a wash

- **supLP120 leads at k=0/k=1, trends at k=3.** supLP120 − scratch = **+5.03 / +2.12 pp**
  (paired-t p=.0014 / .0153, n=24); k=3's +2.02 pp is now a trend (p=.067 — was p=.016). supLP120 vs
  supMAE is now statistically even (36/61, p=.20 — was p=.0003, the sharpest single significance
  loss found in this audit): the two priors are close on UTD in this rerun rather than supLP120
  clearly ahead.
- **supMAE also significant-positive at k=0/k=1** (+2.63 / +2.22 pp, p=.0098 / .0255), n.s. at k=3
  (+1.10 pp, p=.30).
- **mae trends positive, not negative** (+2.23/+0.72/+0.59 pp, all n.s. now — was significant at
  k=0, p=.011). As on CZU skeleton, the "reconstruction hurts" signature ([[multiseed-loso-v2]] C2)
  still does not appear on a same-modality small-gap target, though this rerun is noisier here than
  the original.
- **supcon is now the standout on this dataset** (see below) — clearly ahead of supLP120, not a wash.

## CRC published-baseline anchor (T4, 2026-07-09)

Same recognizer family as the CZU R6 anchor — statistical-moment features (mean/std/var/skew/kurtosis)
+ Collaborative Representation Classifier (CRC-RLS, λ=1e-4) — on the **byte-identical seed-42 LOSO
k-shot splits** (`scripts/external/utd/crc_baseline.py` → `trained_models/UTD-skeleton-LOSO-seed42/crc_baseline/`).
Deterministic given splits, no seeds needed.

| k | CRC baseline | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|---|
| 0 | 69.48 | 75.98 | 78.28 | 80.50 | **83.05** |
| 1 | 92.25 | 90.25 | 93.18 | 92.88 | **93.51** |
| 3 | 95.30 | 93.52 | 94.85 | 95.35 | **96.28** |

supLP120 − CRC = **+13.57 / +1.26 / +0.98 pp** at k=0/1/3 (smaller at k=0 than the original run's
+14.85 pp, similar at k=1/k=3) — the transferred prior opens the same kind of consistent margin over
the published-baseline recognizer family that R6 found on CZU.
One difference from CZU worth noting honestly: on CZU, the **scratch** learned encoder was
statistically level with CRC (84.87 vs 84.81 @k0) — "architecture alone is not the advantage."
On UTD, **scratch already beats CRC by +6.5 pp at k=0** (75.98 vs 69.48) before any prior is
added — the learned architecture has some edge here even without pretraining, so the prior's
marginal contribution on top of scratch (+8.35 pp @k0, supLP120−scratch) is smaller than its
margin over CRC. Both readings agree the prior helps most at k=0 and the gap narrows by k=3.

*(TODO — not yet done: cite UTD-MHAD's own published cross-subject accuracy figure as a second
external reference point, per tasks.md T4. Needs a literature figure, not compute — flag for
Planning/human to supply the citation before this goes in the paper.)*

## supcon (native to this rerun, 2026-07-20) — the strongest small-gap result in the paper

supcon − scratch = **+7.97 / +3.98 / +2.46 pp @ k=0/1/3, all significant** (p<.0001 / p=.0002 /
p=.028) — a clearly larger margin than supLP120's own. supcon now clearly beats supLP120 too (sign
test 42/60, p=.0027, meanΔ +1.75 pp — was a wash at p=.89 in the original run). Combined with the CZU
result (where supcon also edges out supLP120), the two-dataset picture shifts from "both supLP120 and
supcon win big, tied for best" to **"supcon is the more reliable small-gap winner of the two,
ahead of supLP120 on both external datasets"** — supLP120 itself is only conditionally best now (wins
UTD k=0/k=1, ties supMAE, trends but doesn't clear significance at k=3). The class-level claim survives
even as the specific ranking within the class got noisier: "label-supervised, no reconstruction
component" still wins at small gap; which specific recipe wins by how much is dataset- and
checkpoint-dependent. Feeds `paper_results.md` R6d + the R6e five-setting map's supcon paragraph.

## Why it matters

Two independent public datasets now agree on the small-gap end of the arc:

| dataset | zero-shot supLP120 − scratch | mae vs scratch |
|---|---|---|
| CZU skeleton (R6) | +5.0 pp (pooled p=.001) | ≈flat (+0.3, n.s.) |
| **UTD skeleton (R6d)** | **+5.0 pp (p=.0014)** | **trending positive (+2.2, n.s.)** |

Combined with the cross-modal inversion ([[czu-imu-crossmodal]] R6b: supLP120 *worst*, below
scratch) and the strong-target null ([[czu-imu-dual]] R6c), the full arc is: **supervised prior
best-in-class at small gap (two datasets) → worst-in-class across the cross-modal gap → still worst
once the target is strong.** The prior's value is gap-contingent — the negative-transfer thesis as a
design rule.

Feeds `paper/paper_results.md` R6d. See also [[czu-skeleton-loso]] · [[czu-imu-crossmodal]] ·
[[czu-imu-dual]] · [[multiseed-loso-v2]].
