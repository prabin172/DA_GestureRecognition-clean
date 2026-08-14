---
type: result
status: active
updated: 2026-07-21
---

# Phase 3 cross-setting extension — does the controller's ranking hold beyond NTU→Xsens?

**Folded into `paper/paper_results.md` as R5b** (2026-07-21), method note in `paper_method.md`
§8.1, limitations note in `paper_discussion.md`. While integrating this, also caught and fixed a
real staleness bug in `paper_method.md` §8's "Action-primitive assignment" paragraph — it still
described the retired reliability-ranked assignment ("two most reliably-recognized gestures
assigned to Safety-Critical States"), directly contradicting §8.1 right below it; missed during
the 2026-07-20 randomization fix because that pass only touched §8.1 and `paper_results.md`, not
§8's own general description. Now corrected to describe the actual uniform random draw.

## What this is

[[phase3-controller]]'s locked R5 numbers only ever simulated the controller on NTU→Xsens. This
extension reproduces **Lock 1 (randomized base outcome) + Lock 2 (critical-cost severity sweep)
only** — **no Lock 3 / iso-safety** — across every other transfer setting in the paper that has (or
could get) compatible per-clip posteriors: CZU-MHAD skeleton (R6), CZU-MHAD IMU orientation-only
(R6b), UTD-MHAD skeleton (R6d), CZU-MHAD dual-branch raw+quat (R6c). Same protocol as
[[controller]] §9: 120 independently-random 7-gesture System Input assignments × 1000 Monte Carlo
task trials per (assignment, method, k, τ/C_crit) cell, drawn fresh per setting from that setting's
own label space (22 classes for the three CZU settings, 27 for UTD).

Script: `scripts/controller/controller_crosssetting.py` (`controller_robust.py`'s Lock1/Lock2 logic,
`simulate()`/`make_prim_of_id()`/`pack()`, refactored to be generic over a `SETTINGS` dict).
`controller_robust.py` itself is untouched; NTU→Xsens is re-simulated here too so every row in the
table below comes from the identical code path (its own numbers land within noise of the locked
`robust/` Lock1/Lock2 numbers — sanity-checked, not re-cited). Output →
`trained_models/Phase3-controller/crosssetting/<setting>/{vocab_sweep,vocab_ordering,costmodel_sweep,costmodel_summary}.csv`
+ `cross_setting_summary.csv`.

## Posterior provenance — what needed retraining and what didn't

Four settings already had trained checkpoints (`loso_fulltrain_calibration.py`, same
architecture/naming everywhere) — per-clip posteriors were dumped via
`scripts/main_experiment/dump_posteriors.py` pointed at each dataset's processed dir, **pure forward
passes, no retraining**:
- `trained_models/CZU-skeleton-LOSO{,-seed43,-seed44}/posteriors/`
- `trained_models/CZU-IMU-LOSO{,-seed43,-seed44}/posteriors/` (orientation-only crossmodal)
- `trained_models/UTD-skeleton-LOSO-seed{42,43,44}/posteriors/`

**CZU dual-raw (R6c) required an actual retrain.** `scripts/external/czu/dualbranch.py --mode dual`
never persisted checkpoints or per-clip posteriors — only aggregate accuracy to `summary.csv`. Added
an opt-in `--dump-posteriors-dir` flag (`evaluate(..., return_preds=True)` + a `dump_posteriors()`
helper, backward-compatible, verified via smoke test) and reran all 3 seeds × 5 priors × mode=dual
via `scripts/orchestration/10_czu_dual_controller_retrain.sh`, writing to **new** out-roots
(`trained_models/CZU-IMU-DUAL-controller[-seed43/44]/`) so the existing locked
`CZU-IMU-DUAL*/summary.csv` (cited in `paper_results.md` R6c) was never touched. Sanity check: this
retrain's own accuracy (from the dumped posteriors) matches the locked R6c summary.csv within
≤0.3pp at every (method,k) cell — same ranking, well inside the project's established
non-determinism noise floor. 225/225 folds completed with no errors.

## The cross-setting table

Lock 1 = mean hard task-success at k=1, τ=0 (ungated, full compounding) across 120 random
assignments. Lock 2 = median mean-cost at k=1, C_crit=50 across the same 120 assignments (higher
cost = worse).

| setting | Lock1 worst | Lock1 best | Lock2 worst | Lock2 best | locks agree (worst / best) |
|---|---|---|---|---|---|
| NTU→Xsens (main) | mae (0.455) | supMAE (0.525) | mae (54.8) | supMAE (46.5) | **yes / yes** |
| CZU skeleton (R6) | mae (0.647) | supcon (0.802) | scratch (32.9) | supcon (22.5) | no / **yes** |
| CZU IMU orientation-only (R6b) | supLP120 (0.107) | scratch (0.124) | scratch (113.3) | mae (105.6) | no / no |
| UTD skeleton (R6d) | scratch (0.717) | supcon (0.823) | mae (26.0) | supcon (19.9) | no / **yes** |
| CZU dual-raw (R6c) | supcon (0.567) | scratch (0.619) | supcon (42.4) | scratch (39.0) | **yes / yes** |

Best-method win counts across the 5 settings: **supcon best in 4/5 Lock1-or-Lock2 slots, scratch
best in 3/5** (scratch wins outright on both locks for CZU dual-raw). Worst-method: mae worst in
4/5 Lock1-or-Lock2 slots, but never the sole story — supLP120, scratch, and supcon each take a
worst-slot somewhere too.

## How the controller ranking relates to each setting's recognition-level result

The **two settings where both locks agree with each other are exactly the two settings where the
underlying recognition-level effect is large and statistically robust**:

- **NTU→Xsens**: mae is the paper's established, seed/checkpoint-stable negative-transfer culprit
  ([[phase1-mcnemar-ece-cka]], p<.001 every k). Controller: mae worst under both locks. Clean
  amplification of a real, significant accuracy effect.
- **CZU dual-raw ([[czu-imu-dual]] R6c)**: supcon is literally *the single worst pooled result in
  the entire paper* on this setting (p<.0001 every k, nearly double supLP120's own effect size);
  `dual/scratch` is the best performer at every k. Controller: supcon worst, scratch best, under
  **both** locks. Same story, same direction, no surprises — the controller reproduces the
  recognition table's clearest result almost exactly.

**The three settings where the locks disagree with each other are exactly the three settings where
the underlying recognition-level effect is small, non-significant, or reversed in sign**:

- **CZU skeleton (R6)**: mae is *not* negative in raw accuracy here (+0.3/−1.1/−0.1pp vs scratch,
  all n.s. — [[czu-skeleton-loso]]); supcon is the strongest recognizer (beats supLP120 itself,
  p=.044). Controller Lock1 still picks out mae as worst — the 12-step/3-safety-critical task
  structure amplifies a flat/non-significant accuracy difference into a visible gap — but Lock2
  disagrees (scratch worst instead). Both locks agree supcon is best, matching recognition. Read
  Lock1's "mae worst" here as a controller-specific amplification of noise, not a confirmed finding
  the way NTU→Xsens's is.
- **UTD skeleton (R6d)**: mae trends *positive* in raw accuracy (not negative — [[utd-skeleton-loso]]),
  yet Lock2 still ranks it worst by cost. Recognition's clear best (supcon, beats supLP120
  significantly) matches the controller's best on both locks. Worst is lock-dependent (scratch vs
  mae) — again, no significant recognition-level signal underneath to anchor it.
- **CZU IMU orientation-only (R6b)**: recognition itself is only a weak trend post-retrain
  (supLP120 trends worst, p=.41–.59, no longer significant — [[czu-imu-crossmodal]]). Absolute
  accuracy is so low (55–60%) that the compounding task structure crushes **every** method's task
  success to near-floor (10.7–12.4%) — locks disagree on both worst *and* best here. This setting
  is better read as "compounding to the point of uninformativeness" than as a clean ranking test:
  when recognition accuracy itself is this close to chance-adjacent, the controller's 12-step,
  3-safety-critical structure doesn't discriminate methods so much as flatten them all toward
  near-total task failure.

## The synthesis

**The controller's cross-method ranking is trustworthy exactly when it's redundant with the
recognition table, and becomes noisy/lock-dependent exactly where the recognition table itself has
no significant signal to amplify.** This is not a weakness specific to the controller — it is the
controller behaving as advertised: a compounding **stress-test harness**, not an independent source
of ground truth. Where the input signal (recognition accuracy/calibration) is strong, compounding
sharpens it into a large, lock-consistent task-level gap (NTU→Xsens's mae, CZU dual-raw's supcon).
Where the input signal is weak or absent, compounding amplifies noise into disagreement between
locks rather than a real finding.

This is a mechanistic echo of the paper's own R6e five-setting arc (supLP120/supcon go
hero→villain as the domain gap widens and the target representation strengthens,
[[czu-imu-dual]]'s three-column table) — the controller doesn't discover a new ranking, it
re-exposes the existing recognition-level arc at the task-success/cost level, faithfully in the two
settings with a real underlying effect, and unreliably in the three where the underlying effect is
itself marginal.

## Numbers ledger

`trained_models/Phase3-controller/crosssetting/{ntu_xsens,czu_skeleton,czu_imu_quat,utd_skeleton,czu_dual_raw}/{vocab_sweep,vocab_ordering,costmodel_sweep,costmodel_summary}.csv`
+ `cross_setting_summary.csv` ← `scripts/controller/controller_crosssetting.py --vocabs 120
--missions 1000`. Posteriors: `dump_posteriors.py` (3 settings, no retrain) +
`scripts/orchestration/10_czu_dual_controller_retrain.sh` (CZU dual-raw, real retrain, verified
against locked R6c accuracy within ≤0.3pp). `controller_robust.py`/`trained_models/Phase3-controller/robust/`
(the locked R5 NTU→Xsens Lock1/2/3 numbers) untouched throughout. Paper: `paper_results.md` R5b,
`paper_method.md` §8.1, `paper_discussion.md` limitations.

Related: [[controller]] · [[phase3-controller]] · [[czu-skeleton-loso]] · [[czu-imu-crossmodal]] ·
[[utd-skeleton-loso]] · [[czu-imu-dual]]
