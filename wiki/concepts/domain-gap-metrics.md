---
type: concept
status: active
updated: 2026-07-09
---

# Domain-gap metrics (MMD, CKA)

Pillar 3 ([[paper-framing]]) requires *showing* that representation mismatch — not objective or label relatedness — dominates cross-domain transfer.

## MMD (current, flawed)
Squared Maximum Mean Discrepancy between NTU and Xsens feature distributions in each encoder's 512-d feature space; n=500 samples per side, seed 0. Script: `temp_mmd_domain_gap.py`. Results: [[mmd-domain-gap]].

Three flaws found in [[publishability-review]] (item 5):
1. **Scale confound** — fixed sigmas (0.5–5) on unnormalized features; cross-encoder rows not comparable. Fix: median-heuristic sigma or normalized features.
2. **Asymmetric under swing** — twist removed from Xsens only.
3. **No uncertainty** — single draw, no CI. Also "scratch" is a random-init encoder; its MMD isn't a domain gap — don't interpret it.

## Symmetric-swing test (done, 2026-07-04)
`temp_mmd_domain_gap_symmetric.py` swing-projects NTU too. Result ≈ asymmetric swing → the swing gap increase was twist-stripping damaging Xsens, not the asymmetry. Led to the v2 position-reconstruction fix ([[position-reconstruction-v2]]), whose MMD is below local for every encoder.

## CKA — NTU vs Xsens-v2 (done, Phase 1)
Layer-wise linear CKA, `temp_cka_analysis.py`, per encoder × depth (proj, L0–L2 = `KinematicEncoder` Transformer layers). Scale-invariant → kills MMD flaw 1. Results: [[phase1-mcnemar-ece-cka]]. supLP120 aligns 3–5× better than the rest and *increases* with depth; scratch is flat; absolute CKA is tiny for everyone (<0.02) — the cross-modal gap is large even after v2.

## CKA per target — T3 gap-axis measurement (done 2026-07-09)
Extended `temp_cka_analysis.py --multi-target` to compute the same layer-wise linear/RBF CKA between NTU and **each** of the four targets used across the paper (all share the (T,17,4) LRQ schema): Xsens-v2 (claimed **middle** gap), CZU-MHAD skeleton (claimed **small**), CZU-MHAD IMU orientation quats (claimed **large**), UTD-MHAD skeleton (claimed **small**). n≈1000 clips/side, seed 0, NTU features reused across targets per encoder (only 4 NTU forward passes total). Outputs: `trained_models/Phase1-analysis/cka_by_target{,_summary}.csv`, `cka_by_target_heatmap.png`.

**Mean linear CKA (L0–L2) by target × encoder:**

| target | scratch | supLP120 | supMAE | mae | supcon |
|---|---|---|---|---|---|
| xsens_v2 (middle, claimed) | 0.0048 | 0.0290 | 0.0105 | 0.0077 | 0.0240 |
| czu_skeleton (small, claimed) | 0.0045 | 0.0267 | 0.0101 | 0.0074 | 0.0221 |
| czu_imu_quat (large, claimed) | 0.0108 | 0.0260 | 0.0152 | 0.0118 | 0.0221 |
| utd_skeleton (small, claimed) | 0.0071 | 0.0327 | 0.0137 | 0.0121 | 0.0257 |

(`supcon` column added 2026-07-09, T2.2 stage 4; `scratch` values shift slightly run-to-run since it's an unseeded fresh random init, not a fixed checkpoint. supcon shows the same czu_skeleton-below-xsens_v2 inconsistency as supLP120 — 0.0221 < 0.0240 — reinforcing rather than changing the honest-null finding below.)

**Honest result: CKA does NOT confirm the small→middle→large gap ordering, and is internally inconsistent within the "small gap" bucket itself.** On the highest-signal encoder (supLP120), utd_skeleton does rank highest (consistent with "small gap"), but **czu_skeleton ranks lowest of all four targets — below both xsens_v2 (claimed middle) and czu_imu_quat (claimed large)**. Two datasets claimed to share the same small-gap bucket land at opposite ends of the CKA range. This is not treated as a crisis for the thesis (per human decision 2026-07-09): it extends the existing "necessary-not-sufficient" mechanism finding (R3) one level further — CKA doesn't reliably track *gap width* across datasets, let alone predict transfer benefit, so the paper's gap ordering stays justified narratively (modality/device/vocabulary, per [[czu-imu-dual]] R6e) and the load-bearing evidence for the ordering is the downstream sign-test/accuracy pattern (R6e), not a single alignment metric. Folded into `paper_results.md` R3 + R6e honest caveat.

## Key empirical nuance
MMD ordering does NOT align perfectly with k-shot accuracy: SupCon has near-lowest MMD but the worst transfer; MAE has higher MMD than supervised but similar k=1 accuracy. **Low gap is necessary-not-sufficient; discriminative structure matters too.** CKA per-target (above) extends this: even gap *width itself* isn't cleanly captured by CKA. State both explicitly in the paper.

## Raw-feature domain gap — method-independent (done 2026-07-12) — CONFIRMS the gap ordering CKA missed
Everything above (MMD, CKA) is computed **inside a trained encoder's feature space** — confounded by the pretraining objective, so it answers "how different do these two datasets look to THIS encoder," not "how different are the datasets themselves." New script `temp_raw_domain_gap.py` removes the encoder entirely: reuses the CRC-baseline hand-crafted feature (`temp_czu_crc_baseline.py::clip_features` — mean/std/var/skew/kurtosis over the 68 flattened LRQ channels, 340-d, no model/no pretraining) as the feature space for all 5 datasets (they share the (T,17,4) LRQ schema). n=1000/side (utd=861, full pool), seed 0, pooled-standardized per pair (scale-fair like CKA). All 10 pairwise combos of {ntu, xsens_v2, czu_skeleton, czu_imu_quat, utd_skeleton} → `trained_models/RawDomainGap/raw_domain_gap.csv` + 3 heatmaps. Two metrics per pair:
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
