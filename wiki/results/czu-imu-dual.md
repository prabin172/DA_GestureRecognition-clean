---
type: result
status: active
updated: 2026-07-20
---

**2026-07-20: numbers refreshed from `DA_GestureRecognition-clean`'s independently-retrained checkpoints.** This page's findings got *sharper* under the retrain, not weaker (unusual — most sections in this audit softened). supcon's negative effect nearly doubled; supLP120 remains significantly worse. Cold-start extension (T5, bottom of this page) now has a 3-seed run in progress, launched 2026-07-20 — was single-seed before. See `paper/paper_results.md` R6c.

# CZU-MHAD inertial dual-branch — does the prior survive on a *strong* target? (skeleton→IMU)

Follow-up to [[czu-imu-crossmodal]] (R6b). R6b matched the skeleton encoder by forcing an
**orientation-only quaternion** target representation, which capped deep accuracy at 56–61%
(≪ CRC 87–94%). Two confounds remained: was the low accuracy the *encoding* or the *architecture*,
and does the R6b reconstruction-prior benefit survive once the target uses its **full raw signal**?
This run answers both.

- Run: `trained_models/CZU-IMU-DUAL{,-seed43,-seed44}/{raw,quat,dual}_<prior>/summary.csv`.
  **dual mode and raw mode both pooled over 3 seeds** (42/43/44 → 45 folds each, raw seeds 43/44
  added via `scripts/orchestration/t1_multiseed_run.sh` T1.1); quat mode remains single-seed (diagnostic, matches R6b
  splits exactly). 5-subj LOSO, k=0/1/3. **Reuses the byte-identical CZU-IMU-LOSO splits.** Prior-load
  bug fixed → strict 39/39. Pooled stats: `scripts/external/czu/multiseed_analyze.py` (dual) + ad hoc pooling
  script (raw, T1 — same pattern, not yet folded into the shared script).
- Scripts: `scripts/data_pipeline/czu_imu_raw_export.py` (→ `Data_Processed/czu_imu_raw/`, 10×6 raw accel+gyro,
  z-scored, frame-aligned to the R6b quats), `scripts/external/czu/dualbranch.py`, run `scripts/orchestration/czu_dual_run.sh`
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
| **dual** | scratch | **85.99** | **88.79** | **91.94** |
| **dual** | mae | 85.28 | 87.96 | 91.35 |
| **dual** | supMAE | 85.86 | 88.54 | 91.58 |
| **dual** | supLP120 | 84.49 | 87.15 | 90.88 |
| **dual** | supcon | 83.82 | 86.88 | 90.04 |

(dual/raw pooled 3 seeds × 5 subj = 15/prior/k, 2026-07-20 rerun; quat mode not re-run, historical
single-seed diagnostic retained. supcon is a 5th prior in this rerun, trained natively — no separate
supcon dir.) CRC (raw accel+gyro moments, [[czu-imu-crossmodal]]): 86.84 / 90.84 / 94.42 (unchanged).
**`dual/scratch` is now the best performer at every k** — no prior beats it, including supMAE.

## Finding 1 — representation was the bottleneck, not architecture

The raw-signal scratch branch alone hits **81.9 / 90.0 / 95.6** @k0/1/3 pooled (up slightly from
single-seed 80.7/89.8/95.2, std 9.5/5.2/2.5 pp across n=15), matching/beating CRC (86.8/90.8/94.4)
and ~30 pp above the R6b orientation-only deep encoders (56–61%). The R6b collapse was the
**orientation-only encoding** (no accel magnitude, yaw drift), *not* the deep architecture. Confirmed,
now on 3 seeds.

## Finding 2 — on a strong target, the NTU prior adds no value (sharper under the retrain)

Sign tests, dual-branch prior vs `dual/scratch`, pooled over (subject, k, seed) cells (ties dropped),
2026-07-20 rerun:

- **supMAE ≈ scratch** — 18/37 wins, p=1.0 (a wash; report as *no detectable benefit*, same reading
  as before).
- **mae trends ≤ scratch** — 15/42, p=.088 (a trend, not significant, same reading as before).
- **supLP120 < scratch** — 10/41, **p=.0015** (significantly worse; paired-t significant at k=0:
  −1.50 pp, p=.017; k=1 −1.64 pp p=.086 trend; k=3 −1.06 pp p=.055 trend). Still clearly the loser,
  though the p-value is less extreme than the original run's p<.0001.
- **supcon < scratch — the single worst pooled result in this dataset.** 3/42, **p<0.0001**, paired-t
  significant at *every* k (−2.17 / −1.90 / −1.89 pp, p=.0002 / .0002 / .0066). This is nearly double
  the effect size supLP120 shows, and roughly double what the original run's supcon extension found
  (−1.2/−1.1/−1.1 pp) — the clearest strengthening in this whole audit.

**Reading:** the R6b direction is corroborated here with sharper significance than R6b itself now
carries. supLP120 and supcon — the two label-aware-only priors — are the clearest losers in the
entire paper on this target; supMAE (the one objective with a reconstruction component) is the only
prior that doesn't clearly hurt.

## Dose-response along target richness (pooled 3-seed dial)

A middle rung between R6b (orientation-only) and R6c (full raw) — a 20-channel per-sensor
accel-magnitude + gyro-magnitude target (`scripts/data_pipeline/czu_imu_mag_export.py` → `Data_Processed/czu_imu_mag20/`,
run `scripts/orchestration/czu_dial_run.sh` + `scripts/orchestration/t1_multiseed_run.sh` (seeds 43/44) →
`trained_models/CZU-IMU-DIAL/mag20{,-seed43,-seed44}/`) — makes the "contingent on poverty"
claim a curve, now pooled. Δ vs scratch (k=0/1/3):

| target representation | scratch acc | supMAE Δ | supLP120 Δ |
|---|---|---|---|
| 0-ch quat-only (R6b) | 56 / 54 / 58 | +0.7 / +0.2 / +1.3 (n.s.) | −0.9 / −1.4 / −0.7 (n.s.) |
| 20-ch magnitudes (dial, pooled) | 84 / 88 / 90 | −1.3 / −1.2 / −0.0 | −3.8 / −5.7 / −3.2 |
| 60-ch full raw (R6c, pooled) | 86 / 89 / 92 | −0.1 / −0.3 / −0.4 | −1.5 / −1.6 / −1.1 |

(2026-07-20 rerun; the 0-ch/R6b row is now n.s. — see [[czu-imu-crossmodal]]'s update — so read that
row as directional only.) supMAE's edge, where it exists at all, is concentrated at the crippled
orientation-only target and fades as the target sees more real motion; supLP120 is negative at every
rung and significantly so from the 20-ch dial onward. The **dial rung (20-ch) is the most
checkpoint-stable point on this curve** — it reproduces closely against the original run (supMAE was
−1.4/−1.6/−1.1, now −1.3/−1.2/−0.0; supLP120 was −4.1/−5.2/−3.6, now −3.8/−5.7/−3.2) — while the R6b
endpoint is the least stable of the three.

## The three-column arc (the actual result)

One dataset (CZU), one prior family, three settings — the prior's value **and its ranking**
degrade monotonically with the modality gap and target capability:

| Setting | Gap | Target repr. | Best prior | supLP120 (pure-supervised) |
|---|---|---|---|---|
| [[czu-skeleton-loso]] — skeleton→skeleton (R6) | small (same modality) | strong (native skeleton) | **supLP120 +5.0 pp @k0** (pooled p=.001), beats CRC | **winner** |
| [[czu-imu-crossmodal]] — IMU, orientation-only (R6b) | large (cross-modal) | weak (orientation-only) | none clearly — supMAE trends > supLP120 (29/43, p=.032, weaker than originally found) | trends worst, n.s. now (p=.41–.59) |
| this page — IMU, raw/dual (R6c) | large (cross-modal) | strong (raw ≈ CRC) | none (`dual/scratch` is best at every k) | **worst**, p=.0015 |

Two separable axes: **gap** (small in R6 → large in R6b/R6c) and **target representation strength**
(weak in R6b → strong in R6c). Holding the gap large, strengthening the target (R6b→R6c) kills the
reconstruction prior's edge; holding the target strong, widening the gap (R6→R6c) flips the supervised
prior from best to worst. supLP120 goes **hero → villain** as the gap opens: best-in-class same-modality, worst-in-class
cross-modal, and useless (still worst) once the target is strong. A controlled contrast on a
single public dataset — the value of a transferred prior is gap-contingent, exactly the
negative-transfer thesis ([[multiseed-loso-v2]] C2/C3).

## supcon (native to this rerun, 2026-07-20 — see Finding 2 above for the numbers)

This rerun trains supcon alongside the other 4 methods from the start (`dual_supcon/summary.csv`,
pooled 3 seeds), rather than as a separate bolted-on extension. Its result is now folded into
Finding 2 above: **supcon is significantly worse than scratch at every k (p<.0001 pooled sign test,
paired-t p≤.0066 every k)** — the single worst pooled result in this dataset, roughly double the
effect size the original repo's separate supcon extension found (−1.2/−1.1/−1.1 pp there vs
−2.17/−1.90/−1.89 pp here). Both label-aware-only priors (supLP120, supcon) lose on a strong target;
supMAE (the only reconstruction-bearing objective tested) is the sole one that ties scratch. Feeds
`paper_results.md` R6c + the R6e five-setting map.

## Caveats

1. **dual and raw modes pooled 3 seeds (n=15/prior/k, 2026-07-20 rerun); quat mode single-seed
   (not re-run).** The supLP120-worse and supcon-worse effects are robust (p=.0015 and p<.0001); "supMAE
   ties scratch" is a *null* on n=37 — write it as "no detectable benefit," not "proven equal."
2. **dual/scratch ≠ strict SOTA.** Raw-only scratch is actually best at k=3 (95.6 vs dual 91.9 pooled);
   the dual architecture is a vehicle for the prior-transfer question, not an accuracy claim.
3. R6b's own table (from `CZU-IMU-LOSO/`) is unchanged in structure but its significance weakened
   under the retrain (see [[czu-imu-crossmodal]]'s 2026-07-20 note); the `quat` rows here are a
   same-splits sanity replication, untouched, and land in the same 55–60% band.

## Cold-start extension (T5) — see [[czu-dual-cold-start]]

Separate question from Finding 2 above (which uses full N=4-equivalent training): does the prior's
*cold-start* advantage ([[a2-subject-scaling]] R4c, biggest at N=0/1 on Xsens) survive on this strong
target? No — repeating the subject-count sweep on CZU-dual finds no N∈{0,1,2,3} where supLP120 or
supMAE beats scratch; both are significantly worse at N=0, k=1 (the Xsens cold-start sweet spot).
**3-seed extension (seeds 43/44) launched 2026-07-20**, in progress as of this writing — was
single-seed (n=5 folds) before. Full writeup: [[czu-dual-cold-start]].

Feeds `paper/paper_results.md` R6c. See also [[czu-imu-crossmodal]] · [[czu-skeleton-loso]] · [[multiseed-loso-v2]] · [[czu-dual-cold-start]].
