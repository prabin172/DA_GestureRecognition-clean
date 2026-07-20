---
type: concept
status: active
updated: 2026-07-20
---

# Domain-gap metrics (MMD, CKA)

Pillar 3 ([[paper-framing]]) requires *showing* that representation mismatch — not objective or label relatedness — dominates cross-domain transfer.

**2026-07-20 (human decision):** CKA is de-prioritized as evidence for the gap-ordering claim — the
raw MMD²/Frechet measurement (bottom of this page) is the one that actually characterizes objective
performance against gap width correctly, and is now the paper's primary domain-distance evidence.
CKA remains useful for the *separate* "alignment necessary-not-sufficient" mechanism finding (R3),
but its specific numbers moved under `DA_GestureRecognition-clean`'s independent checkpoint retrain
(see updated tables below) and are not worth defending precisely.

## MMD (current, flawed)
Squared Maximum Mean Discrepancy between NTU and Xsens feature distributions in each encoder's 512-d feature space; n=500 samples per side, seed 0. Script: `scripts/main_experiment/mmd_domain_gap.py`. Results: [[mmd-domain-gap]].

Three flaws found in [[publishability-review]] (item 5):
1. **Scale confound** — fixed sigmas (0.5–5) on unnormalized features; cross-encoder rows not comparable. Fix: median-heuristic sigma or normalized features.
2. **Asymmetric under swing** — twist removed from Xsens only.
3. **No uncertainty** — single draw, no CI. Also "scratch" is a random-init encoder; its MMD isn't a domain gap — don't interpret it.

## Symmetric-swing test (done, 2026-07-04)
`mmd_domain_gap_symmetric.py (dead, not in this repo)` swing-projects NTU too. Result ≈ asymmetric swing → the swing gap increase was twist-stripping damaging Xsens, not the asymmetry. Led to the v2 position-reconstruction fix ([[position-reconstruction-v2]]), whose MMD is below local for every encoder.

## CKA — NTU vs Xsens-v2 (done, Phase 1)
Layer-wise linear CKA, `scripts/main_experiment/cka_analysis.py`, per encoder × depth (proj, L0–L2 = `KinematicEncoder` Transformer layers). Scale-invariant → kills MMD flaw 1. Results: [[phase1-mcnemar-ece-cka]]. supLP120 aligns 3–5× better than the rest and *increases* with depth; scratch is flat; absolute CKA is tiny for everyone (<0.02) — the cross-modal gap is large even after v2.

## CKA per target — T3 gap-axis measurement (done 2026-07-09)
Extended `scripts/main_experiment/cka_analysis.py --multi-target` to compute the same layer-wise linear/RBF CKA between NTU and **each** of the four targets used across the paper (all share the (T,17,4) LRQ schema): Xsens-v2 (claimed **middle** gap), CZU-MHAD skeleton (claimed **small**), CZU-MHAD IMU orientation quats (claimed **large**), UTD-MHAD skeleton (claimed **small**). n≈1000 clips/side, seed 0, NTU features reused across targets per encoder (only 4 NTU forward passes total). Outputs: `trained_models/Phase1-analysis/cka_by_target{,_summary}.csv`, `cka_by_target_heatmap.png`.

**Mean linear CKA (L0–L2) by target × encoder:**

| target | scratch | supLP120 | supMAE | mae | supcon |
|---|---|---|---|---|---|
| xsens_v2 (middle, claimed) | 0.0034 | 0.0137 | 0.0053 | 0.0044 | 0.0114 |
| czu_skeleton (small, claimed) | 0.0059 | 0.0253 | 0.0119 | 0.0075 | 0.0200 |
| czu_imu_quat (large, claimed) | 0.0090 | 0.0183 | 0.0128 | 0.0111 | 0.0169 |
| utd_skeleton (small, claimed) | 0.0092 | 0.0343 | 0.0177 | 0.0126 | 0.0289 |

(2026-07-20: refreshed from `DA_GestureRecognition-clean`'s independently-retrained checkpoints — numbers moved but the qualitative pattern below is unchanged.)

**Honest result, still standing: CKA does NOT confirm the small→middle→large gap ordering, and is internally inconsistent within the "small gap" bucket itself.** On the highest-signal encoder (supLP120), utd_skeleton ranks highest (consistent with "small gap") and czu_skeleton ranks second — but **xsens_v2 (claimed middle gap) now ranks lowest of all four targets**, below even czu_imu_quat (claimed large). This is a different specific inconsistency than the original run found (there it was czu_skeleton ranking lowest), but the conclusion is the same: CKA doesn't reliably track *gap width* across independently-collected datasets, in either run. This is not treated as a crisis for the thesis: it extends the existing "necessary-not-sufficient" mechanism finding (R3) one level further, and is exactly why (per 2026-07-20 human decision) CKA is no longer the paper's primary domain-distance evidence — the raw MMD²/Frechet measurement below is, since it's unaffected by which checkpoint you retrain (no pretrained encoder involved) and correctly orders all four targets in both runs. Folded into `paper_results.md` R3 + R6e honest caveat.

## Key empirical nuance
MMD ordering does NOT align perfectly with k-shot accuracy: SupCon has near-lowest MMD but the worst transfer; MAE has higher MMD than supervised but similar k=1 accuracy. **Low gap is necessary-not-sufficient; discriminative structure matters too.** CKA per-target (above) extends this: even gap *width itself* isn't cleanly captured by CKA. State both explicitly in the paper.

## Raw-feature domain gap — method-independent (done 2026-07-12) — CONFIRMS the gap ordering CKA missed
Everything above (MMD, CKA) is computed **inside a trained encoder's feature space** — confounded by the pretraining objective, so it answers "how different do these two datasets look to THIS encoder," not "how different are the datasets themselves." New script `scripts/main_experiment/raw_domain_gap.py` removes the encoder entirely: reuses the CRC-baseline hand-crafted feature (`scripts/external/czu/crc_baseline.py::clip_features` — mean/std/var/skew/kurtosis over the 68 flattened LRQ channels, 340-d, no model/no pretraining) as the feature space for all 5 datasets (they share the (T,17,4) LRQ schema). n=1000/side (utd=861, full pool), seed 0, pooled-standardized per pair (scale-fair like CKA). All 10 pairwise combos of {ntu, xsens_v2, czu_skeleton, czu_imu_quat, utd_skeleton} → `trained_models/RawDomainGap/raw_domain_gap.csv` + 3 heatmaps. Two metrics per pair:
- **MMD²** — RBF kernel, median-heuristic bandwidth (fixes the fixed-sigma scale confound the old encoder-space MMD had).
- **Frechet distance** — Gaussian closed form on a PCA(50) projection (numerically stable covariance at n≈1000).

(A third metric, proxy A-distance via a domain classifier, was also computed but saturated near-maximal on every pair — trivially separable, no ranking signal — and is dropped from reporting per human decision 2026-07-12; raw values still in the CSV.)

**NTU-anchored result (the one that matters for the paper's small/middle/large gap claim):**
| target | mmd² | frechet |
|---|---|---|
| czu_skeleton (claimed small) | **0.0560** | **176.8** |
| utd_skeleton (claimed small) | **0.0960** | **202.1** |
| xsens_v2 (claimed middle) | **0.1218** | **281.9** |
| czu_imu_quat (claimed large) | **0.2049** | **393.8** |

**MMD² and Frechet distance BOTH confirm the claimed small<small<middle<large gap ordering exactly and monotonically — the opposite outcome from T3's encoder-space CKA above, which did NOT confirm it (czu_skeleton ranked lowest/most-dissimilar there).** Removing the encoder from the measurement recovers the ordering that CKA's per-encoder alignment couldn't find — consistent with the necessary-not-sufficient story: the encoder's *learned* alignment doesn't track raw distributional gap cleanly, but the raw gap itself does track the paper's narrative ordering. **Folded into `paper/paper_results.md` (R3 new subsection + R6e caveat rewritten from "remains narrative" to "measured, encoder-free confirmation") and `paper_discussion.md` §1 (new paragraph after the CKA necessary-not-sufficient point) and `paper_method.md` §7.4.** The gap ordering now has a measured, encoder-independent basis it previously lacked.
