---
type: result
status: active
updated: 2026-07-20
---

**2026-07-20: 3-seed extension (seeds 43/44) launched** (`scripts/orchestration/09b_czu_cold_start_multiseed.sh`), in progress as of this writing — was single-seed only before. The table below is still the seed-42-only numbers; will be updated with the pooled 3-seed version once the run completes. This was the last remaining single-seed load-bearing result in the paper (see `SESSION_HANDOFF.md` and `paper/paper_results.md` R6c).

# CZU-dual cold-start subject-scaling (T5) — does A2's cold-start lever survive on a strong target?

[[a2-subject-scaling]] (R4c, Xsens) found the prior's benefit peaks at N=0/1 enrolled subjects
(supLP120 +27.5 pp @ k=3, N=1, pooled 3 seeds) and washes out by N=4 — the paper's central
deployment claim. That result was established only on Xsens, a weak/moderate target. [[czu-imu-dual]]
(R6c) separately showed the prior adds nothing (and supLP120/supcon actively hurt) on a **strong**
target once fully trained (N=4-equivalent). This page asks the missing question: **does the
cold-start advantage itself survive on a strong target, or is R4c's lever also representation-poverty-contingent?**

- Run: `--n-train-subjects` added to `scripts/external/czu/dualbranch.py` (same N=0-short-circuits-to-pretrained-init
  pattern as the main A2 harness). Sweep N=0..3, priors {scratch, supLP120, supMAE}, seed 42 only,
  5-subj LOSO → `trained_models/CZU-DUAL-subjectScaling/N{0..3}/dual_<prior>/summary.csv`.
- Single-seed (n=5 folds/cell) — a 3-seed extension is queued in `tasks.md` T1/T5 follow-on but not
  run; several cells are already significant at n=5.

## Accuracy and prior benefit vs scratch

| N | k | scratch | supLP120 Δ | supMAE Δ |
|---|---|---|---|---|
| 0 | 0 | 2.87 | +0.11 (n.s., p=.85) | +0.37 (n.s., p=.72) |
| 0 | 1 | 38.94 | **−5.62 pp, p=.045** | **−5.55 pp, p=.005** |
| 0 | 3 | 54.21 | +0.23 (n.s., p=.92) | **−2.87 pp, p=.035** |
| 1 | 0 | 64.66 | −0.93 (n.s.) | −0.37 (n.s.) |
| 1 | 1 | 71.01 | −3.11 (p=.085, n.s.) | −3.51 (p=.052, n.s.) |
| 1 | 3 | 79.58 | **−2.54 pp, p=.004** | −1.80 (p=.076, n.s.) |
| 2 | 1 | 81.86 | **−3.54 pp, p=.015** | **−4.07 pp, p=.006** |
| 3 | 3 | 90.32 | **−2.83 pp, p=.033** | −3.57 (p=.063, n.s.) |

(Full table across all N×k: `trained_models/CZU-DUAL-subjectScaling/`. N=0,k=0 accuracies are tiny
(2.87–3.24%) — near-chance, pretrained-init-only with no calibration; read k=1/3.)

## Finding — no N at which either prior beats scratch; both actively hurt at cold start

**Neither supLP120 nor supMAE beats `dual/scratch` at any (N,k) tested.** At N=0, k=1 — the exact
cold-start cell where the Xsens A2 result shows its *largest* prior benefit (+18.4 pp pooled) — both
priors are here **significantly worse than scratch** (supLP120 −5.6 pp p=.045; supMAE −5.6 pp p=.005).
The pattern continues through N=1/2/3: several cells reach significance, all negative, none positive.

## Reading

The R4c cold-start deployment lever ("the prior is worth most when you have 0–1 enrolled users") is
**itself contingent on target-representation poverty**, exactly mirroring R6c's finding at full
training (N=4-equivalent). A strong target's from-scratch branch reaches a useful operating point
even with zero enrolled subjects and no prior at all — there is no subject-count regime tested here
where the NTU-pretrained prior helps this target. Put together with R6c: **on a representationally
strong target, the NTU prior is not just useless once trained — it never becomes useful, at any point
along the subject-enrollment curve.** This closes the R4c deployment claim's scope precisely: read it
as "the prior matters most at cold start *when the target representation is weak*," not as a
universal cold-start rule.

## Caveats

- Single-seed, n=5 folds per cell — several cells are significant despite the small n, but this
  should be pooled to 3 seeds before being cited as load-bearing as confidently as R4c/R6c themselves.
- Only 3 priors tested (scratch, supLP120, supMAE) — mae and supcon were not run here; given both
  supLP120 and supMAE (one pure-supervised, one hybrid) show the same negative-to-neutral pattern,
  there's no reason to expect mae/supcon would differ, but it is not directly confirmed.
- N range is 0–3 (CZU has only 4 non-held-out subjects per fold, vs Xsens's 0–4), so this doesn't
  reach the same "washes out" endpoint A2 shows at N=4 — though R6c's separate full-training result
  (all subjects) already establishes the N→max endpoint on this target (prior still hurts, doesn't
  wash to neutral).

Feeds `paper/paper_results.md` R6c (cold-start extension) + `a2-subject-scaling.md` R4c cross-reference.
See also [[a2-subject-scaling]] · [[czu-imu-dual]] · [[czu-imu-crossmodal]].
