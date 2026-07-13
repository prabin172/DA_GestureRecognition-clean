---
type: result
status: active
updated: 2026-07-05b
---

# Phase 3 — Control-reliability simulation (C6 / THMS pillar)

**Design knobs LOCKED (2026-07-05b) via a robustness protocol** — the method ordering is shown invariant to the knobs rather than frozen at arbitrary values. Prototype `temp_controller_sim.py`; the locked study is `temp_controller_robust.py` → `trained_models/Phase3-controller/robust/{vocab_sweep,vocab_ordering,costmodel_sweep,frontier,iso_safety}.csv` + PNGs.

## Setup
Abstract event-driven FSM over the **real** held-out posteriors ([[phase1-mcnemar-ece-cka]] dumps). 12-step pick-place mission (2 grasp/1 release → compounding), confidence-reject safety layer, asymmetric cost. Distractor stream (non-command gestures) → false-activation. No physics sim.

## The three locks
1. **Randomized vocabulary** (kills "cherry-picked gestures"): resample the 7-gesture→primitive map 120× at random, report the distribution.
2. **Critical-cost sweep + two outcome models** (kills "harsh instant-fail rule"): hard-safety (critical=fail) AND soft-cost (critical=recoverable ×C_crit), C_crit∈{2,5,10,20,50,∞}.
3. **Iso-safety operating point** (kills "tuned τ to win"): fix a false-activation budget, find smallest τ meeting it per method, compare throughput. Tuning-free.

## Locked findings
**mae compounds worst — invariant.** Mean hard task-success across 120 vocabs (worst in bold):

| condition | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|
| k1 τ0 | 0.516 | **0.452** | 0.530 | 0.479 |
| k1 τ0.9 | 0.714 | **0.655** | 0.724 | 0.700 |
| k3 τ0 | 0.663 | **0.634** | 0.710 | 0.716 |
| k3 τ0.9 | 0.830 | **0.774** | 0.847 | 0.879 |

mae ≥ supMAE in only 12% (τ0) / 15% (τ0.9) of vocabs at k1. Highest mission cost at **every** C_crit (k1, C_crit=20: mae 20.8 vs supMAE 15.6). C2 at the task level, knob-independent.

**Calibration → throughput at fixed safety (iso-safety, 1% budget, k1):** supLP120 meets budget at τ*=0.90 (vs 0.95–0.99), cost **16.7** (13–20% faster), success 0.953; only supLP120 meets the strict 0.5% budget at k1 with strong success (τ*=0.97, 0.877) while scratch/supMAE can't reach it.

**Honest finding = design principle:** supLP120 has a *confident false-critical-activation* mode (confidently maps some gestures onto the well-separated grasp/release anchors) → dips below scratch on the *ungated* metric at k1. Exactly why the correct framing is iso-safety + assigning safety-critical commands to the most-separable gestures. Ungated metric understates calibration's value.

Net locked claims: **(1) mae compounds worst, (2) calibration governs the safety/throughput trade** — across 120 vocabs × 2 outcome models × 6 C_crit × tuning-free operating point.

## SupCon extension (2026-07-10) + a discovered vocab-coupling quirk

Re-ran with `supcon` added to `METHODS` → `trained_models/Phase3-controller/robust-supcon/` (new dir; the
`robust/` numbers above are untouched, never overwritten). **Lock 1 reproduces exactly** (same values
to 3 decimals for scratch/mae/supMAE/supLP120) — Lock 1's per-vocab RNG doesn't depend on which
other methods are loaded. Adding supcon: k=1 τ=0 ungated success 0.506 (between supLP120 0.479 and
supMAE 0.530); k=3 τ=0.9 gated success **0.880, edging out supLP120's 0.879 as the single best of five**.
mae remains worst throughout — beats supcon in only 22% of vocabs at k=1 τ=0 (12% vs supMAE, 34% vs
supLP120).

**Locks 2/3 did NOT reproduce exactly for the original 4 methods, and here's why.** `reliability_ordered_vocab()`
ranks the 12-step mission's gesture assignment by pooled k=3 recall over **every method present in the
loaded posterior pool** (`df`, unfiltered by method) — not per-method. Adding supcon's posteriors to
the pool therefore silently changed which 7 gestures get assigned to which primitive for Locks 2/3
(Lock 1 uses independent randomized vocabularies each run, hence unaffected). This is a real script
coupling, not a data or seeding bug — confirmed by checking `simulate()`/`false_activation()` use only
the passed `rng` argument (no raw `np.random.*` calls), and Lock 1's exact reproduction rules out a
data-pooling issue. **Practical upshot:** the original locked Lock 2/3 tables (this page, above) are
safe and unaffected (separate output dir, was never re-run in place). supcon's own Lock 2/3 numbers
were computed under this re-derived vocabulary and should not be pooled with the locked table's rows
for the other 4 methods. Reported on their own terms, the direction replicates: mae has the highest
mean cost at every C_crit swept (2 → ∞) under the new vocab too, and supLP120 again reaches the 1%
false-activation budget at the lowest τ* (0.85) and lowest cost (16.6), with supcon second-cheapest
(19.0). **mae-worst and calibration-governs-safety hold under an independently-derived vocabulary** —
an adventitious extra robustness check, not one of the three planned locks. **TODO (not urgent):**
`reliability_ordered_vocab()` should filter by a canonical method (e.g. scratch or the union of only
the originally-locked methods) before computing recall, so future method additions don't perturb it;
not fixed here to avoid touching the script's behavior for the already-locked run.

Related: [[controller]] (full design doc — mechanics, rationale, the 3 locks explained, planned live-study extension) · [[phase1-mcnemar-ece-cka]] · [[a2-subject-scaling]] · [[multiseed-loso-v2]] · [[paper-framing]]
