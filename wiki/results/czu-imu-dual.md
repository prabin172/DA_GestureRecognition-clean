---
type: result
status: active
updated: 2026-07-10
---

# CZU-MHAD inertial dual-branch — does the prior survive on a *strong* target? (skeleton→IMU)

Follow-up to [[czu-imu-crossmodal]] (R6b). R6b matched the skeleton encoder by forcing an
**orientation-only quaternion** target representation, which capped deep accuracy at 56–61%
(≪ CRC 87–94%). Two confounds remained: was the low accuracy the *encoding* or the *architecture*,
and does the R6b reconstruction-prior benefit survive once the target uses its **full raw signal**?
This run answers both.

- Run: `trained_models/CZU-IMU-DUAL{,-seed43,-seed44}/{raw,quat,dual}_<prior>/summary.csv`.
  **dual mode and raw mode both pooled over 3 seeds** (42/43/44 → 45 folds each, raw seeds 43/44
  added via `temp_t1_multiseed_run.sh` T1.1); quat mode remains single-seed (diagnostic, matches R6b
  splits exactly). 5-subj LOSO, k=0/1/3. **Reuses the byte-identical CZU-IMU-LOSO splits.** Prior-load
  bug fixed → strict 39/39. Pooled stats: `temp_czu_multiseed_analyze.py` (dual) + ad hoc pooling
  script (raw, T1 — same pattern, not yet folded into the shared script).
- Scripts: `temp_czu_imu_raw_export.py` (→ `Data_Processed/czu_imu_raw/`, 10×6 raw accel+gyro,
  z-scored, frame-aligned to the R6b quats), `temp_czu_dualbranch.py`, run `temp_czu_dual_run.sh`
  → `czu_dual.log` (finished clean: `CZU-DUAL ALL DONE`, 2026-07-06).
- Three modes: **raw** (raw-signal scratch branch only — diagnostic), **quat** (R6b orientation-only,
  sanity replication), **dual** (raw-scratch branch + NTU-pretrained quat branch, concat → head).

## Final accuracy — per-(mode, prior, k) mean

raw + dual = pooled 3 seeds (n=45); quat = single-seed diagnostic (n=5).

| mode | prior | k=0 | k=1 | k=3 |
|---|---|---|---|---|
| **raw** (pooled) | scratch | 81.94 | 90.01 | **95.56** |
| quat (1 seed) | scratch | 57.08 | 54.10 | 60.02 |
| quat (1 seed) | supMAE | 55.95 | 55.65 | 59.89 |
| quat (1 seed) | supLP120 | 55.10 | 54.16 | 57.32 |
| **dual** | scratch | 85.99 | 88.79 | 91.94 |
| **dual** | mae | 84.76 | 88.64 | 91.63 |
| **dual** | supMAE | 85.64 | 88.22 | **92.02** |
| **dual** | supLP120 | 84.46 | 86.98 | 89.99 |

CRC (raw accel+gyro moments, [[czu-imu-crossmodal]]): 86.84 / 90.84 / 94.42.

## Finding 1 — representation was the bottleneck, not architecture

The raw-signal scratch branch alone hits **81.9 / 90.0 / 95.6** @k0/1/3 pooled (up slightly from
single-seed 80.7/89.8/95.2, std 9.5/5.2/2.5 pp across n=15), matching/beating CRC (86.8/90.8/94.4)
and ~30 pp above the R6b orientation-only deep encoders (56–61%). The R6b collapse was the
**orientation-only encoding** (no accel magnitude, yaw drift), *not* the deep architecture. Confirmed,
now on 3 seeds.

## Finding 2 — on a strong target, the NTU prior adds no value

Sign tests, dual-branch prior vs `dual/scratch`, pooled over (subject, k, seed) cells (ties dropped):

- **supMAE ≈ scratch** — 17/39 wins, p=0.52 (a wash; report as *no detectable benefit*).
- **mae ≤ scratch** — 13/36, p=0.13.
- **supLP120 < scratch** — 6/41, **p<0.0001** (significantly *worse*; paired-t significant at every k:
  −1.5 / −1.8 / −1.9 pp, p=.0016 / .040 / .0038). Pooling *strengthens* this vs the single-seed p=.003.

And the second branch itself doesn't earn its place: `dual/scratch > raw/scratch` is **19/42 pooled,
p=.644** (up in power from single-seed 6/12, p=1.00 — same conclusion): a small k=0 gain (+4.1 pp
pooled) is paid back by a k=3 loss (−3.6 pp pooled). Adding the NTU-pretrained quaternion branch to
an already-strong raw model is a net wash at best; the quat branch is itself the weak (~55%)
component (see quat rows), so it dilutes rather than complements.

**Reading:** the R6b reconstruction-prior rescue is **contingent on target-representation
poverty**. It shows up only when the target encoder is crippled to orientation-only; give the
target its full raw signal and the prior's contribution vanishes — supMAE ties scratch while supLP120
is again the loser (now at p<.0001). This is *not* a contradiction of R6b; it is its right-hand boundary.

## Dose-response along target richness (pooled 3-seed dial)

A middle rung between R6b (orientation-only) and R6c (full raw) — a 20-channel per-sensor
accel-magnitude + gyro-magnitude target (`temp_czu_imu_mag_export.py` → `Data_Processed/czu_imu_mag20/`,
run `temp_czu_dial_run.sh` + `temp_t1_multiseed_run.sh` (seeds 43/44) →
`trained_models/CZU-IMU-DIAL/mag20{,-seed43,-seed44}/`) — makes the "contingent on poverty"
claim a curve, now pooled. Δ vs scratch (k=0/1/3):

| target representation | scratch acc | supMAE Δ | supLP120 Δ |
|---|---|---|---|
| 0-ch quat-only (R6b) | 56 / 54 / 58 | +3.0 / +2.4 / +3.0 | −3.5 / −2.8 / −3.1 |
| 20-ch magnitudes (dial, pooled) | 84 / 88 / 90 | −1.4 / −1.6 / −1.1 | −4.1 / −5.2 / −3.6 |
| 60-ch full raw (R6c, pooled) | 87 / 89 / 91 | −0.4 / −0.4 / +0.3 | −2.1 / −3.5 / −2.5 |

supMAE's edge exists **only** at the fully-crippled orientation-only target — gone the moment the
target sees any real motion (even coarse magnitudes, where scratch already hits 84%+). supLP120 is
negative at every rung. Pooling tightens the numbers but doesn't change the shape — still
steep-then-flat (20-ch supMAE dips slightly below 60-ch); R6b itself remains single-seed (the dial's
own 0-ch anchor row).

## The three-column arc (the actual result)

One dataset (CZU), one prior family, three settings — the prior's value **and its ranking**
degrade monotonically with the modality gap and target capability:

| Setting | Gap | Target repr. | Best prior | supLP120 (pure-supervised) |
|---|---|---|---|---|
| [[czu-skeleton-loso]] — skeleton→skeleton (R6) | small (same modality) | strong (native skeleton) | **supLP120 +7.2 pp @k0** (pooled p=.0004), beats CRC | **winner** |
| [[czu-imu-crossmodal]] — IMU, orientation-only (R6b) | large (cross-modal) | weak (orientation-only) | supMAE ties scratch; > supLP120 36/44 (p<.0001) | **worst**, below scratch (p≈.04–.06) |
| this page — IMU, raw/dual (R6c) | large (cross-modal) | strong (raw ≈ CRC) | none (supMAE ties scratch, 17/39) | **worst**, p<.0001 |

Two separable axes: **gap** (small in R6 → large in R6b/R6c) and **target representation strength**
(weak in R6b → strong in R6c). Holding the gap large, strengthening the target (R6b→R6c) kills the
reconstruction prior's edge; holding the target strong, widening the gap (R6→R6c) flips the supervised
prior from best to worst. supLP120 goes **hero → villain** as the gap opens: best-in-class same-modality, worst-in-class
cross-modal, and useless (still worst) once the target is strong. A controlled contrast on a
single public dataset — the value of a transferred prior is gap-contingent, exactly the
negative-transfer thesis ([[multiseed-loso-v2]] C2/C3).

## supcon extension (2026-07-10, dual mode pooled 3 seeds n=45)

`trained_models/CZU-IMU-DUAL-supcon-seed{42,43,44}/dual_supcon/summary.csv`. Dual-branch accuracy:
84.74 / 87.65 / 90.87 @ k=0/1/3 (vs dual/scratch 85.99/88.79/91.94). supcon − scratch = **−1.24 / −1.14
/ −1.07 pp** (k=0 significant p=.048, k=1/k=3 marginal p=.12/.08); sign test supcon>scratch **13/43,
p=.014** — significantly worse, same direction as supLP120 (6/41, p<.0001) though less extreme.
**Both label-aware-only priors lose on a strong target**; supMAE (the only reconstruction-bearing
objective tested) is the sole one that ties scratch. Converges the R6b and R6c readings: the axis
that matters is reconstruction-present vs reconstruction-absent, not softmax vs contrastive. Feeds
`paper_results.md` R6c + the R6e five-setting map.

## Caveats

1. **dual and raw modes pooled 3 seeds (n=45 each); quat mode single-seed.** The supLP120-worse
   effect (p<.0001, paired-t every k) is robust; "supMAE ties scratch" is a *null* on n=39 — write it
   as "no detectable benefit," not "proven equal." The raw-vs-dual wash is now pooled too (19/42,
   p=.644 — same conclusion as single-seed, more power).
2. **dual/scratch ≠ strict SOTA.** Raw-only scratch is actually best at k=3 (95.6 vs dual 91.9 pooled);
   the dual architecture is a vehicle for the prior-transfer question, not an accuracy claim.
3. R6b's own table (from `CZU-IMU-LOSO/`) is unchanged; the `quat` rows here are a same-splits
   sanity replication and land in the same 55–60% band.

## Cold-start extension (T5) — see [[czu-dual-cold-start]]

Separate question from Finding 2 above (which uses full N=4-equivalent training): does the prior's
*cold-start* advantage ([[a2-subject-scaling]] R4c, biggest at N=0/1 on Xsens) survive on this strong
target? No — repeating the subject-count sweep on CZU-dual finds no N∈{0,1,2,3} where supLP120 or
supMAE beats scratch; both are significantly worse at N=0, k=1 (the Xsens cold-start sweet spot).
Full writeup: [[czu-dual-cold-start]].

Feeds `paper/paper_results.md` R6c. See also [[czu-imu-crossmodal]] · [[czu-skeleton-loso]] · [[multiseed-loso-v2]] · [[czu-dual-cold-start]].
