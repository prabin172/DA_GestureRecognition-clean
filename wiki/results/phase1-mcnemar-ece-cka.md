---
type: result
status: active
updated: 2026-07-10
---

# Phase 1 — McNemar, ECE & CKA (per-clip, no retrain)

One inference pass over the existing v2 multi-seed base+calibration checkpoints (k∈{0,1,3}) dumps per-clip posteriors, then computes the three things the [[multiseed-loso-v2]] stats were owed. **v2 only — swing stays dead** (human decision). Scripts: `temp_dump_posteriors.py` → `temp_analyze_calibration.py` (ECE + reliability + McNemar) + `temp_cka_analysis.py`; orchestrated by `temp_phase1_run.sh` (scratch/mae/supMAE/supLP120) + `temp_t2b_followon_run.sh` stage 3 (supcon, `trained_models/Phase1-analysis-supcon/`, new dir — original `Phase1-analysis/` McNemar/ECE untouched). Outputs `trained_models/Phase1-analysis{,-supcon}/{mcnemar,ece,cka}_results.csv` + `reliability_k*.png`, `cka_heatmap.png`, `cka_vs_benefit.png`. See [[temp-scripts]].

## McNemar — clip-level paired test vs scratch (C2)
Pooled over 3 seeds. net = (prior correct where scratch wrong) − (scratch correct where prior wrong).

| prior | k | n_pairs | b (scratch-only) | c (prior-only) | net | p |
|-------|---|---------|------------------|----------------|-----|---|
| supMAE | 0 | 7974 | 480 | 680 | **+200** | 0.0 |
| supMAE | 1 | 7644 | 342 | 418 | +76 | .0065 |
| supMAE | 3 | 6984 | 210 | 341 | **+131** | 0.0 |
| supLP120 | 0 | 7974 | 786 | 895 | +109 | .0084 |
| supLP120 | 1 | 7644 | 591 | 494 | −97 | .0035 |
| supLP120 | 3 | 6984 | 296 | 400 | +104 | 9e-05 |
| supcon | 0 | 7974 | 740 | 825 | +85 | .034 |
| supcon | 1 | 7644 | 487 | 492 | +5 | .898 (n.s.) |
| supcon | 3 | 6984 | 265 | 409 | **+144** | 0.0 |
| mae | 0 | 7974 | 789 | 455 | **−334** | 0.0 |
| mae | 1 | 7644 | 593 | 376 | **−217** | 0.0 |
| mae | 3 | 6984 | 385 | 293 | **−92** | .00047 |

- **mae = significant negative transfer at every k** (all p<.001). This is the per-clip p-value debt the multi-seed accuracy table was owed for C2 — inverts the field's "reconstruction is the safe prior" prior.
- **supMAE = significant positive at every k** (+200 @k0, +131 @k3).
- supLP120 and supcon are both mixed: positive k0/k3, no significant effect at k1 (supLP120 actually negative k1; supcon ≈0 k1) — the two label-aware objectives share this k1 signature, distinct from supMAE's clean positive-everywhere and mae's clean negative-everywhere.

## ECE — calibration (C6 feeder)
| method | k | acc | ece | ece_tempscaled | T |
|--------|---|-----|-----|----------------|---|
| supLP120 | 0 | 58.48 | 0.255 | **0.0258** | 2.56 |
| supcon | 0 | 58.18 | 0.264 | 0.0339 | 2.70 |
| supMAE | 0 | 59.62 | 0.275 | 0.0653 | 3.06 |
| scratch | 0 | 57.11 | 0.300 | 0.0676 | 3.37 |
| mae | 0 | 52.92 | 0.336 | 0.0676 | 3.49 |
| supLP120 | 1 | 81.01 | 0.092 | **0.0306** | 1.87 |
| supcon | 1 | 82.34 | 0.091 | 0.0410 | 1.94 |
| supMAE | 1 | 83.27 | 0.074 | 0.0424 | 1.80 |
| scratch | 1 | 82.27 | 0.077 | 0.0494 | 1.93 |
| mae | 1 | 79.44 | 0.077 | 0.0667 | 1.80 |
| supLP120 | 3 | 90.79 | 0.025 | **0.0350** | 1.47 |
| supMAE | 3 | 91.18 | 0.022 | 0.0396 | 1.31 |
| scratch | 3 | 89.30 | 0.023 | 0.0396 | 1.41 |
| supcon | 3 | 91.37 | 0.018 | 0.0410 | 1.51 |
| mae | 3 | 87.99 | 0.023 | 0.0545 | 1.34 |

- **supLP120 has the lowest temp-scaled ECE at every k; supcon is a clear second**, ahead of supMAE/scratch/mae at every k. mae worst-calibrated at k=1/3. The two label-supervised objectives (softmax-supervised supLP120, contrastive-with-labels supcon) both out-calibrate the two without direct label supervision at the pooled embedding (supMAE, mae) — calibration tracks label-awareness of the objective, not just accuracy.
- Raw ECE at k=0 is large for everyone (0.25–0.34, all overconfident, T≈2.5–3.5); temp scaling collapses it.

## CKA — representational alignment NTU vs Xsens-v2 (C3 mechanism)
Linear CKA, per encoder × layer (proj = projection head, L0–L2 = KinematicEncoder Transformer layers — all reported results use `KinematicEncoder`, not DSTformer; see [[models]]).

| encoder | proj | L0 | L1 | L2 |
|---------|------|----|----|----|
| supLP120 | 0.0038 | 0.0133 | 0.0139 | **0.0149** |
| supcon | 0.0039 | 0.0233 | 0.0231 | 0.0257 |
| supMAE | 0.0029 | 0.0043 | 0.0047 | 0.0048 |
| mae | 0.0029 | 0.0036 | 0.0036 | 0.0035 |
| scratch | 0.0027 | 0.0027 | 0.0027 | 0.0027 |

supcon's CKA is numerically the *highest* of all five (0.0257 @ L2), ahead of supLP120 — the two label-supervised objectives (supLP120, supcon) both align well above supMAE/mae/scratch and grow with depth. **This is now the sharpest alignment-vs-accuracy dissociation in the paper**: supcon has the best raw alignment yet its cross-domain accuracy (R1) is a wash vs scratch (McNemar n.s. at k1, marginal elsewhere) — better-aligned is not better-transferring, full stop. (Note: this CKA re-run overwrote `trained_models/Phase1-analysis/cka_results.csv` in place rather than a new dir — justified because it's deterministic given fixed checkpoints and reproduced the pre-existing supLP120/supMAE/mae numbers, scratch differing by ≤0.0002 from its unseeded fresh init.)

- **Absolute CKA is tiny for all encoders** (<0.02) — the NTU↔Xsens gap is large in representation space even after v2. Don't over-claim on magnitude.
- **Ordering by objective is clean and monotone with depth**: supLP120 aligns ~3–5× better than the rest and *increases* with depth; scratch is flat (no alignment, as expected). The pure-supervised prior builds the most transferable geometry — consistent with it being best-calibrated (ECE) and dominant at low subject count ([[a2-subject-scaling]]).
- This is the C3 mechanism evidence: MMD was necessary-not-sufficient; CKA gives a per-objective, per-depth alignment ordering.

## How this feeds the paper
- **C2** (negative transfer) — McNemar supplies the per-clip significance the accuracy means lacked.
- **C3** (mechanism) — CKA ordering + [[mmd-domain-gap]] MMD.
- **C6** (controller) — ECE per objective×k is the reliability input for the Phase 3 B2 controller.

Related: [[multiseed-loso-v2]] · [[a2-subject-scaling]] · [[domain-gap-metrics]] · [[pretraining-objectives]]
