---
type: result
status: active
updated: 2026-07-20
---

# Phase 3 — Control-reliability simulation (C6 / THMS pillar)

**Design knobs probed (2026-07-05b) via a robustness protocol** — three locks vary the vocab assignment, cost model, and threshold independently to check whether the method ordering is invariant to them. Prototype `scripts/controller/controller_sim.py`; the study is `scripts/controller/controller_robust.py` → `trained_models/Phase3-controller/robust/{vocab_sweep,vocab_ordering,costmodel_sweep,frontier,iso_safety}.csv` + PNGs.

**Terminology (2026-07-16):** the 7 task primitives are called **System Inputs** (2 **Safety-Critical States**, 5 **Routine States**), the task is the **Sequential Control Task**, not a "mission" — see [[controller]]'s terminology-revision note for why the old pick-and-place/grasp/release naming was dropped.

**2026-07-20: re-run on independently-retrained checkpoints (`DA_GestureRecognition-clean`'s full rerun) — the locks no longer agree, and that's now the finding.** Lock 1 (the primary protocol) still shows mae worst. Locks 2/3 now show **supLP120** as worst at high penalty severity — a real, mechanistically-explained effect (supLP120's known "confident false-critical-activation" mode, previously a minor caveat, is more pronounced on this checkpoint and dominates under harsh cost models), not a bug. The "ordering is invariant to the knobs" framing no longer holds as stated. Full numbers and the corrected narrative: `paper/paper_results.md` R5 (2026-07-20 update) — this page's tables below are kept as the historical locked run; do not cite them as current without checking the paper's R5 section first.

## Setup
Abstract event-driven FSM over the **real** held-out posteriors ([[phase1-mcnemar-ece-cka]] dumps). 12-step Sequential Control Task (2 Safety-Critical steps into one state, 1 into the other → compounding), confidence-reject safety layer, asymmetric cost. Distractor stream (non-System-Input gestures) → false-activation. No physics sim.

## The three locks
1. **Randomized System Input assignment** (kills "cherry-picked gestures"): resample the 7-gesture→primitive map 120× at random, report the distribution.
2. **Critical-cost sweep + two outcome models** (kills "harsh instant-fail rule"): hard-safety (critical=fail) AND soft-cost (critical=recoverable ×C_crit), C_crit∈{2,5,10,20,50,∞}.
3. **Iso-safety operating point** (kills "tuned τ to win"): fix a false-activation budget, find smallest τ meeting it per method, compare throughput. Tuning-free.

## Locked findings (original run — superseded, see 2026-07-20 note above)
**mae compounds worst under Lock 1 — reproduces. Locks 2/3 do not, in the 2026-07-20 rerun.** Mean hard task-success across 120 System Input assignments (worst in bold; original run's numbers, kept for history):

| condition | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|
| k1 τ0 | 0.516 | **0.452** | 0.530 | 0.479 |
| k1 τ0.9 | 0.714 | **0.655** | 0.724 | 0.700 |
| k3 τ0 | 0.663 | **0.634** | 0.710 | 0.716 |
| k3 τ0.9 | 0.830 | **0.774** | 0.847 | 0.879 |

mae ≥ supMAE in only 12% (τ0) / 15% (τ0.9) of assignments at k1. Highest task cost at **every** C_crit (k1, C_crit=20: mae 20.8 vs supMAE 15.6). C2 at the task level, knob-independent.

**Calibration → throughput at fixed safety (iso-safety, 1% budget, k1):** supLP120 meets budget at τ*=0.90 (vs 0.95–0.99), cost **16.7** (13–20% faster), success 0.953; only supLP120 meets the strict 0.5% budget at k1 with strong success (τ*=0.97, 0.877) while scratch/supMAE can't reach it.

**Honest finding = design principle:** supLP120 has a *confident false-critical-activation* mode (confidently maps some gestures onto the well-separated Safety-Critical-State anchors) → dips below scratch on the *ungated* metric at k1. Exactly why the correct framing is iso-safety + assigning the Safety-Critical States to the most-separable gestures. Ungated metric understates calibration's value.

Net locked claims: **(1) mae compounds worst, (2) calibration governs the safety/throughput trade** — across 120 System Input assignments × 2 outcome models × 6 C_crit × tuning-free operating point.

## SupCon extension (2026-07-10) + a discovered vocab-coupling quirk

Re-ran with `supcon` added to `METHODS` → `trained_models/Phase3-controller/robust-supcon/` (new dir; the
`robust/` numbers above are untouched, never overwritten). **Lock 1 reproduces exactly** (same values
to 3 decimals for scratch/mae/supMAE/supLP120) — Lock 1's per-vocab RNG doesn't depend on which
other methods are loaded. Adding supcon: k=1 τ=0 ungated success 0.506 (between supLP120 0.479 and
supMAE 0.530); k=3 τ=0.9 gated success **0.880, edging out supLP120's 0.879 as the single best of five**.
mae remains worst throughout — beats supcon in only 22% of vocabs at k=1 τ=0 (12% vs supMAE, 34% vs
supLP120).

**Locks 2/3 did NOT reproduce exactly for the original 4 methods, and here's why.** `reliability_ordered_vocab()`
ranks the 12-step task's gesture assignment by pooled k=3 recall over **every method present in the
loaded posterior pool** (`df`, unfiltered by method) — not per-method. Adding supcon's posteriors to
the pool therefore silently changed which 7 gestures get assigned to which System Input for Locks 2/3
(Lock 1 uses independent randomized assignments each run, hence unaffected). This is a real script
coupling, not a data or seeding bug — confirmed by checking `simulate()`/`false_activation()` use only
the passed `rng` argument (no raw `np.random.*` calls), and Lock 1's exact reproduction rules out a
data-pooling issue. **Practical upshot:** the original locked Lock 2/3 tables (this page, above) are
safe and unaffected (separate output dir, was never re-run in place). supcon's own Lock 2/3 numbers
were computed under this re-derived assignment and should not be pooled with the locked table's rows
for the other 4 methods. Reported on their own terms, the direction replicates: mae has the highest
mean cost at every C_crit swept (2 → ∞) under the new assignment too, and supLP120 again reaches the 1%
false-activation budget at the lowest τ* (0.85) and lowest cost (16.6), with supcon second-cheapest
(19.0). **mae-worst and calibration-governs-safety hold under an independently-derived assignment** —
an adventitious extra robustness check, not one of the three planned locks. **TODO (not urgent):**
`reliability_ordered_vocab()` should filter by a canonical method (e.g. scratch or the union of only
the originally-locked methods) before computing recall, so future method additions don't perturb it;
not fixed here to avoid touching the script's behavior for the already-locked run.

Related: [[controller]] (full design doc — mechanics, rationale, the 3 locks explained, planned live-study extension) · [[phase1-mcnemar-ece-cka]] · [[a2-subject-scaling]] · [[multiseed-loso-v2]] · [[paper-framing]]
