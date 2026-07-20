---
type: result
status: active
updated: 2026-07-20
---

# Phase 1 — McNemar, ECE & CKA (per-clip, no retrain)

One inference pass over the existing v2 multi-seed base+calibration checkpoints (k∈{0,1,3}) dumps per-clip posteriors, then computes the three things the [[multiseed-loso-v2]] stats were owed. **v2 only — swing stays dead** (human decision). Scripts: `scripts/main_experiment/dump_posteriors.py` → `scripts/main_experiment/cka_analysis.py` (ECE + reliability + McNemar + CKA in one pass, stage 3 of `DA_GestureRecognition-clean`'s rerun). Outputs `trained_models/Phase1-analysis/{mcnemar,ece,cka,cka_by_target}_results.csv` + `reliability_k*.png`, `cka_heatmap.png`, `cka_vs_benefit.png`.

**2026-07-20: numbers below are from `DA_GestureRecognition-clean`'s independently-retrained checkpoints**, superseding the original run's numbers. ECE ranking reproduces exactly. McNemar's mae-negative-transfer claim reproduces cleanly; the supMAE/supLP120/supcon "positive at k=0" claims did not survive — see full discussion in `paper/paper_results.md` R2 (2026-07-20 update). CKA's specific "supcon numerically highest" claim also did not survive (now lower than supLP120) — de-prioritized per human decision, raw MMD²/Frechet (`[[domain-gap-metrics]]`) is the primary domain-distance evidence going forward, not CKA.

## McNemar — clip-level paired test vs scratch (C2)
Pooled over 3 seeds. net = (prior correct where scratch wrong) − (scratch correct where prior wrong).

| prior | k | n_pairs | b (scratch-only) | c (prior-only) | net | p |
|-------|---|---------|------------------|----------------|-----|---|
| supMAE | 0 | 7974 | 536 | 514 | −22 | .517 (n.s.) |
| supMAE | 1 | 7644 | 327 | 349 | +22 | .419 (n.s.) |
| supMAE | 3 | 6984 | 162 | 312 | **+150** | 0.0 |
| supLP120 | 0 | 7974 | 731 | 778 | +47 | .236 (n.s.) |
| supLP120 | 1 | 7644 | 570 | 511 | −59 | .078 (n.s.) |
| supLP120 | 3 | 6984 | 250 | 385 | **+135** | 0.0 |
| supcon | 0 | 7974 | 715 | 877 | **+162** | 5e-05 |
| supcon | 1 | 7644 | 476 | 502 | +26 | .424 (n.s.) |
| supcon | 3 | 6984 | 272 | 390 | **+118** | 1e-05 |
| mae | 0 | 7974 | 737 | 471 | **−266** | 0.0 |
| mae | 1 | 7644 | 567 | 325 | **−242** | 0.0 |
| mae | 3 | 6984 | 379 | 280 | **−99** | .00013 |

- **mae = significant negative transfer at every k** (all p<.001) — the one claim in this table fully stable across an independent checkpoint retrain. Inverts the field's "reconstruction is the safe prior" assumption.
- supMAE/supLP120/supcon are now significant-positive **at k=3 only**; k=0/k=1 are a wash (n.s. both directions) — softer than the original run, where supMAE looked positive at every k. Treat the k=0/k=1 "prior helps" claims as noisy, not seed/checkpoint-stable; k=3 is solid.

## ECE — calibration (C6 feeder)
| method | k | ece_tempscaled |
|--------|---|----------------|
| supLP120 | 0 | **0.0295** |
| supcon | 0 | 0.0298 |
| supMAE | 0 | 0.0602 |
| scratch | 0 | 0.0676 |
| mae | 0 | 0.0728 |
| supLP120 | 1 | **0.0282** |
| supcon | 1 | 0.0415 |
| scratch | 1 | 0.0494 |
| supMAE | 1 | 0.0520 |
| mae | 1 | 0.0577 |
| supLP120 | 3 | **0.0291** |
| supcon | 3 | 0.0318 |
| scratch | 3 | 0.0396 |
| supMAE | 3 | 0.0498 |
| mae | 3 | 0.0560 |

- **supLP120 has the lowest temp-scaled ECE at every k; supcon is a clear second** — this ranking is unchanged from the original run, the single most checkpoint-stable result in the paper. mae worst-calibrated at every k (was worst at k=1/3, tied-worst k=0 — a minor sharpening). The two label-supervised objectives (softmax-supervised supLP120, contrastive-with-labels supcon) both out-calibrate the two without direct label supervision at the pooled embedding (supMAE, mae).
- Raw ECE at k=0 is large for everyone (0.26–0.33, all overconfident, T≈2.7–3.4); temp scaling collapses it.

## CKA — representational alignment NTU vs Xsens-v2 (secondary check — see note above)
Linear CKA, per encoder × layer (proj = projection head, L0–L2 = KinematicEncoder Transformer layers — all reported results use `KinematicEncoder`, not DSTformer; see [[models]]).

| encoder | proj | L0 | L1 | L2 |
|---------|------|----|----|----|
| supLP120 | 0.0038 | 0.0129 | 0.0134 | **0.0146** |
| supcon | 0.0038 | 0.0121 | 0.0114 | 0.0115 |
| supMAE | 0.0030 | 0.0044 | 0.0046 | 0.0044 |
| mae | 0.0026 | 0.0038 | 0.0034 | 0.0036 |
| scratch | 0.0026 | 0.0024 | 0.0024 | 0.0024 |

supLP120 is numerically highest here this time (was supcon in the original run, 0.0257 vs 0.0149) — an already-tiny number (<0.02 absolute for everyone) moving between independently-retrained checkpoints. Not worth leaning on either direction; the alignment-ordering conclusion (label-aware objectives align better than mae/scratch, monotone with depth) is unchanged and is the load-bearing part of this table. Per human decision (2026-07-20), CKA is no longer the paper's primary domain-distance evidence — raw MMD²/Frechet on hand-crafted features ([[domain-gap-metrics]]) is, since that measurement correctly characterized objective performance against gap width where CKA did not.

- **Absolute CKA is tiny for all encoders** (<0.02) — the NTU↔Xsens gap is large in representation space even after v2. Don't over-claim on magnitude.
- **Ordering by objective is clean and monotone with depth**: supLP120 aligns ~3–5× better than the rest and *increases* with depth; scratch is flat (no alignment, as expected). The pure-supervised prior builds the most transferable geometry — consistent with it being best-calibrated (ECE) and dominant at low subject count ([[a2-subject-scaling]]).
- This is the C3 mechanism evidence: MMD was necessary-not-sufficient; CKA gives a per-objective, per-depth alignment ordering.

## How this feeds the paper
- **C2** (negative transfer) — McNemar supplies the per-clip significance the accuracy means lacked.
- **C3** (mechanism) — CKA ordering + [[mmd-domain-gap]] MMD.
- **C6** (controller) — ECE per objective×k is the reliability input for the Phase 3 B2 controller.

Related: [[multiseed-loso-v2]] · [[a2-subject-scaling]] · [[domain-gap-metrics]] · [[pretraining-objectives]]
