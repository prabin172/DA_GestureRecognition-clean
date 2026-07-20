---
type: result
status: active
updated: 2026-07-20
---

# Phase 3 — Control-reliability simulation (C6 / THMS pillar)

**Design knobs probed via a fully-randomized robustness protocol** — three locks vary the vocab
assignment, cost model, and threshold, all evaluated over the same 120 uniformly-random System
Input assignments (no gesture selection based on recall or any other property anywhere). Study:
`scripts/controller/controller_robust.py --vocabs 120 --missions 1000` →
`trained_models/Phase3-controller/robust/{vocab_sweep,vocab_ordering,costmodel_sweep,costmodel_summary,frontier,iso_safety,iso_safety_summary}.csv`
+ PNGs. Full design rationale: [[controller]].

**Terminology (2026-07-16):** the 7 task primitives are called **System Inputs** (2
**Safety-Critical States**, 5 **Routine States**), the task is the **Sequential Control Task**,
not a "mission" — see [[controller]]'s terminology-revision note.

## Setup
Abstract event-driven FSM over the **real** held-out posteriors ([[phase1-mcnemar-ece-cka]] dumps),
5 methods trained natively together (scratch, mae, supMAE, supLP120, supcon). 12-step Sequential
Control Task (2 Safety-Critical steps into one state, 1 into the other → compounding),
confidence-reject safety layer, asymmetric cost. Distractor stream (non-System-Input gestures) →
false-activation. No physics sim.

## The three locks — all sharing the same 120 randomly-drawn assignments
1. **Base outcome model**: 120 uniformly-random 7-gesture→System-Input assignments, 1000 Monte
   Carlo trials per (assignment, method, k, τ) cell; report the distribution of task success.
2. **Critical-cost sweep + two outcome models**, same 120 assignments: hard-safety (critical=fail)
   AND soft-cost (critical=recoverable ×C_crit), C_crit∈{2,5,10,20,50,∞}; report median/IQR of
   mean cost across the 120 assignments.
3. **Iso-safety operating point**, same 120 assignments: fix a false-activation budget, find
   smallest τ meeting it per method per assignment, report the distribution of τ*/success/cost.

## Locked findings — all three locks agree

**mae compounds worst under every lock.** Mean hard task-success across 120 randomized System
Input assignments:

| condition | scratch | mae | supMAE | supLP120 | supcon |
|---|---|---|---|---|---|
| k=1, τ=0 | 0.516 | **0.455** | 0.522 | 0.472 | 0.523 |
| k=1, τ=0.9 | 0.714 | **0.650** | 0.717 | 0.703 | 0.740 |
| k=3, τ=0 | 0.663 | **0.643** | 0.722 | 0.723 | 0.713 |
| k=3, τ=0.9 | 0.830 | **0.792** | 0.861 | 0.883 | 0.895 |

**Lock 2 (median mean-cost across 120 assignments, k=1):**

| C_crit | mae | scratch | supLP120 | supMAE | supcon |
|---|---|---|---|---|---|
| 20 | **32.3** | 29.0 | 30.4 | 28.5 | 28.6 |
| 50 | **55.5** | 48.3 | 51.2 | 47.1 | 47.8 |
| ∞ | **756,017** | 644,016 | 704,016 | 611,518 | 638,517 |

mae has the highest median mean-cost at every C_crit swept, at both k=1 and k=3 — a completely
different outcome model from Lock 1, same ordering.

**Lock 3 (iso-safety, 1% budget, mean across 120 assignments):**

| k | mae | scratch | supLP120 | supMAE | supcon |
|---|---|---|---|---|---|
| 1 task-success | **0.360** | 0.514 | 0.561 | 0.524 | 0.646 |
| 3 task-success | **0.741** | 0.797 | 0.862 | 0.812 | 0.848 |

mae has the lowest task-success at the safety operating point at both k=1 and k=3 (also true at
the stricter 0.5% budget, not tabulated). supLP120/supcon — the two best-calibrated objectives
(R4a) — lead at k=3, consistent with the calibration→throughput story.

**Net: mae compounds worst, consistently, under every stress test** — Lock 1 (randomized base
outcome), Lock 2 (cost-severity sweep), and Lock 3 (iso-safety threshold) all agree, evaluated
over the identical shared set of 120 random task designs. Calibration still governs the
safety/throughput trade at k=3 (Lock 3).

## Secondary honest finding — does not change the ranking

supLP120 has a *confident false-critical-activation* mode (occasionally maps an unrelated gesture
onto a well-separated Safety-Critical-State anchor with high confidence) → dips just below scratch
on the ungated Lock 1 metric at k=1 (0.472 vs 0.516), and trends as the second-costliest method
(behind mae) as Lock 2's penalty grows severe. This is a narrow, secondary effect — it never
overtakes mae as the worst-compounding objective under any lock, and does not contradict supLP120
remaining the best-calibrated method on average (R4a) or leading Lock 3's k=3 task-success. Full
mechanism: [[controller]] §10.

## Methodology note: this replaced an earlier fixed-vocab design

An earlier version of Locks 2/3 evaluated on one fixed gesture assignment (ranked by pooled
recall) rather than the randomized design above. That approach is fully retired — no reported
number here or in the paper uses it. It was replaced specifically because a reliability-ranked
assignment is a choice that could itself be seen as tuned, undermining the point of having
robustness locks at all. The superseded fixed-vocab run is kept at
`trained_models/Phase3-controller/robust-fixedvocab-superseded/` for the record, not cited
anywhere. Full rationale: [[controller]] §2, §9, §13.

Related: [[controller]] (full design doc — mechanics, rationale, the 3 locks explained, planned live-study extension) · [[phase1-mcnemar-ece-cka]] · [[a2-subject-scaling]] · [[multiseed-loso-v2]] · [[paper-framing]]
