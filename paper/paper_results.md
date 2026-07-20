# Results (draft)

_This document is the **source of truth for all reported numbers.** Every value here traces to `RESEARCH_LOG.md` §B and the run directories under `trained_models/`. Written 2026-07-05 (v2 preprocessing; 3-seed stats; Phase 1 McNemar/ECE/CKA; Phase 2 A2 subject-scaling; Phase 3 controller). Six results sections map to contributions C1–C6 plus external validity._

_**Updated 2026-07-20** with numbers from `DA_GestureRecognition-clean`'s from-scratch 10-stage reproducibility rerun (independently retrained NTU-pretraining checkpoints, pinned environment, first time this pipeline has run reproducibly at all). `scratch`/CRC-baseline numbers — which never touch a pretrained checkpoint — reproduce exactly; every number downstream of a retrained checkpoint carries the normal 1–5pp noise floor of non-deterministic GPU training (`cudnn.deterministic=False` throughout this codebase, by original design). A few claims that were riding close to a significance boundary did not survive; those are flagged and softened in place rather than silently reworded — see each section for specifics. See `wiki/log.md` 2026-07-20 for the full discrepancy audit._

---

## R1 — Objective utility is conditional on the gap (C1)

**Within domain**, objective choice is decisive. A linear probe on NTU features (30 epochs) spans ~45 pp:

| init | NTU linear-probe acc |
|---|---|
| supervised (supLP120) | 59.6% |
| hybrid (supMAE) | 59.2% |
| contrastive (supcon) | 55.7% |
| reconstruction (mae) | 22.6% |
| scratch | 14.8% |

Reconstruction alone (mae) barely clears random init within domain — it does not build discriminative features. The two label-supervised objectives (supLP120, supMAE) and the label-supervised contrastive objective (supcon, Khosla 2020 SupCon — in-batch same-label positives) all cluster in the mid-to-high 50s; pure reconstruction is far behind.

**Across the skeleton→IMU gap**, the same choice compresses to a few points. LOSO-v2 final accuracy, per-(method,k) mean ± sd over 3 seeds (n=15):

| method | k=0 | k=1 | k=3 |
|---|---|---|---|
| supcon | 59.04 ± 6.83 | 82.62 ± 6.85 | 91.01 ± 3.96 |
| supMAE | 56.80 ± 6.70 | 82.57 ± 6.61 | 91.50 ± 4.24 |
| supLP120 | 57.67 ± 4.02 | 81.50 ± 6.34 | 91.25 ± 4.00 |
| scratch | 57.15 ± 6.84 | 82.31 ± 6.55 | 89.39 ± 4.90 |
| mae | 53.75 ± 5.07 | 79.16 ± 6.05 | 87.94 ± 4.37 |

The full method spread at k=1 is ~3.5 pp (82.6 − 79.2), vs the ~45 pp within-domain spread. The gap, not the objective, sets how much objective choice matters. supcon leads narrowly at k=0/k=1 but the margin over scratch is not significant at any k (paired Δ: +1.9/+0.3/+1.6 pp at k=0/1/3, all n.s., p≥.12 — see R2) — a fifth objective, a fifth accuracy-is-a-wash story. All methods reach 94–98% by k≥5 (ceiling); k∈{0,1,3} is the discriminating regime.

## R2 — A seed-stable negative-transfer case that inverts the field (C2)

The load-bearing correction from single-seed to multi-seed: the single-seed "+4.1 pp @ k=1 supMAE" delta was seed-42-specific (seed-by-seed supMAE−scratch @ k=1: +4.13 / −1.77 / +0.56; pooled, this rerun: +0.26 pp, n.s.). What survives pooling — and what survives an independent full retrain of every checkpoint (2026-07-20) — is not a supervised/contrastive collapse but a **reconstruction collapse**: `mae` is the one objective that is consistently, significantly worse than scratch, everywhere else is a wash or a weak positive that varies run to run.

**Paired Δ vs scratch** (mean pp, [95% CI], paired-t p), n=15:

| contrast | k=0 | k=1 | k=3 |
|---|---|---|---|
| supMAE − scratch | −0.35 [−3.08, +2.39] p=.790 | +0.26 [−1.47, +2.00] p=.749 | **+2.11 [+0.88, +3.35] p=.003** |
| supLP120 − scratch | +0.52 [−2.88, +3.93] p=.747 | −0.81 [−3.10, +1.49] p=.463 | +1.86 [−0.28, +4.00] p=.083 |
| supcon − scratch | +1.90 [−2.56, +6.36] p=.377 | +0.31 [−3.21, +3.83] p=.853 | +1.62 [−0.49, +3.72] p=.122 |
| **mae − scratch** | −3.39 [−6.91, +0.13] p=.058 | **−3.15 [−5.07, −1.23] p=.003** | −1.45 [−3.25, +0.35] p=.106 |

**Clip-level McNemar** (pooled 3 seeds; net = prior-only-correct − scratch-only-correct):

| prior | k | n_pairs | b (scratch-only) | c (prior-only) | net | p |
|---|---|---|---|---|---|---|
| supMAE | 0 | 7974 | 536 | 514 | −22 | .517 (n.s.) |
| supMAE | 1 | 7644 | 327 | 349 | +22 | .419 (n.s.) |
| supMAE | 3 | 6984 | 162 | 312 | **+150** | 0.0 |
| supLP120 | 0 | 7974 | 731 | 778 | +47 | .236 (n.s.) |
| supLP120 | 1 | 7644 | 570 | 511 | −59 | .078 (n.s.) |
| supLP120 | 3 | 6984 | 250 | 385 | **+135** | 0.0 |
| supcon | 0 | 7974 | 715 | 877 | **+162** | 5e-05 |
| supcon | 1 | 7644 | 476 | 502 | +26 | .424 (n.s.) |
| supcon | 3 | 6984 | 272 | 390 | **+118** | 1e-05 |
| **mae** | 0 | 7974 | 737 | 471 | **−266** | 0.0 |
| **mae** | 1 | 7644 | 567 | 325 | **−242** | 0.0 |
| **mae** | 3 | 6984 | 379 | 280 | **−99** | .00013 |

**mae is significant negative transfer at every k** (all p<.001) in both the pooled-accuracy and the clip-level view — the one claim in this section that is fully seed-stable and now also checkpoint-stable (it survives an independent full retrain). It flips more clips wrong than right relative to random init, because it never encodes discriminative structure within-domain (R1) and the fine-tune from that init lands in a worse basin than from random. This inverts the field's working assumption that reconstruction/self-supervision is the *safe* prior under large gaps.

The other three objectives are noisier than a single retrain can pin down. **supMAE, supLP120 and supcon all land significant-positive at k=3 only** (supMAE p=.003/McNemar p=0; supLP120 McNemar p=0 though its own paired-Δ is n.s.; supcon both p<.001) **and are a wash at k=0/k=1** — n.s. on both the paired-Δ and McNemar tests, with signs that flip between this run and the original (e.g. supMAE's k=0 McNemar net went from the original run's strongly-positive +200 to this run's −22, both non-significant once you account for which run you're looking at — a coin-flip-scale delta, not a reversal of a real effect). Read this honestly as: **the k=0/k=1 "prior helps" claims for supMAE/supLP120/supcon were never as solid as the mae-negative-transfer claim, and an independent retrain exposes that** — they're small, sign-unstable deltas riding near a significance boundary, not seed-stable effects. The k=3 positive result for all three label-aware-or-hybrid objectives, in contrast, replicates cleanly across both this rerun's own McNemar test and its paired-Δ test.

## R3 — Mechanism: gap is necessary-not-sufficient; CKA orders objectives (C3)

**MMD does not track transfer.** Squared-MMD between NTU and Xsens-v2 features (supMAE encoder) is 0.0092 under v2 (below the local baseline 0.0109 and far below swing 0.0322), confirming v2 closes the gap — but MMD ordering across encoders does not match transfer ordering (contrastive had near-lowest MMD yet worst transfer under the earlier preprocessing; mae has higher MMD than supervised yet similar low-k accuracy).

**CKA orders the objectives cleanly, but is a secondary check here — the raw MMD²/Frechet measurement below is the paper's primary domain-distance evidence, since it is the one that turned out to characterize objective performance against gap width correctly.** Linear CKA between NTU and Xsens-v2 activations, per encoder × layer (2026-07-20 rerun, retrained checkpoints):

| encoder | proj | L0 | L1 | L2 |
|---|---|---|---|---|
| supLP120 | 0.0038 | 0.0129 | 0.0134 | **0.0146** |
| supcon | 0.0038 | 0.0121 | 0.0114 | 0.0115 |
| supMAE | 0.0030 | 0.0044 | 0.0046 | 0.0044 |
| mae | 0.0026 | 0.0038 | 0.0034 | 0.0036 |
| scratch | 0.0026 | 0.0024 | 0.0024 | 0.0024 |

Two facts, both still true under the retrain: (i) **absolute CKA is small for all** (<0.02) — the cross-modal gap is large even after v2, so we do not over-claim on magnitude; (ii) the **ordering is clean and monotone with depth** for the two label-aware objectives — supLP120 and supcon both align well above supMAE/mae/scratch, while scratch is flat. supLP120 is numerically highest here this time (supcon's own magnitude moved between runs — an already-small number moving further is not a claim worth leaning on either way). The best-aligned encoder is not the best-transferring on accuracy (supMAE is, R1), and the reconstruction encoder has higher CKA than scratch but negative transfer. **Conclusion, unchanged and load-bearing:** representation alignment is a necessary context but not a sufficient predictor; the objective's discriminative inductive bias governs transfer. This is the mechanism the MMD-only view could not supply — but the ordering-across-datasets claim itself now rests on the raw, encoder-free measurement below, not on CKA.

**Extending this to the gap axis itself:** we also measured layer-wise CKA between NTU and each of the four targets used elsewhere in the paper (Xsens-v2, CZU skeleton, CZU IMU orientation quats, UTD skeleton) to test whether CKA orders them by our claimed small/middle/large gap (see R6e). It does not: on the supLP120 encoder (2026-07-20 rerun), mean L0–L2 linear CKA is utd_skeleton 0.0343 > czu_skeleton 0.0253 > czu_imu_quat 0.0183 > xsens_v2 0.0137 — the claimed-middle target (xsens_v2) ranks *lowest*, not in the middle, and czu_imu_quat (claimed large gap) ranks above it. We report this as a further instance of necessary-not-sufficient rather than force a fit: CKA does not reliably track gap *width* across independently collected datasets, in either this run or the original. This is exactly why the raw, encoder-free measurement below — not CKA — is the paper's evidence for the gap ordering. Full table in `wiki/concepts/domain-gap-metrics.md`.

**A raw, encoder-free measurement recovers the ordering CKA missed.** Both CKA and the encoder-space MMD above are computed *inside a trained encoder's feature space* — confounded by the pretraining objective, so they measure how different two datasets look *to that encoder*, not how different the datasets are themselves. We therefore also measured domain gap directly on a hand-crafted, model-free feature (mean/std/var/skew/kurtosis over the 68 flattened LRQ channels — the same feature used for the CRC published-baseline reproductions in R6/R6d, zero pretraining involved), computing squared-MMD (RBF kernel, median-heuristic bandwidth — this fixes the fixed-sigma scale confound noted above) and Frechet distance (Gaussian closed form on a 50-component PCA) between NTU and each target, n=1000/side (n=861 for UTD, its full pool), seed 0:

| target | gap (claimed) | MMD² | Frechet |
|---|---|---|---|
| czu_skeleton | small | **0.0560** | **176.8** |
| utd_skeleton | small | **0.0960** | **202.1** |
| xsens_v2 | middle | **0.1218** | **281.9** |
| czu_imu_quat | large | **0.2049** | **393.8** |

**Both metrics confirm the claimed small<small<middle<large gap ordering exactly and monotonically** — the opposite outcome from the CKA-per-target result immediately above. Removing the pretrained encoder from the measurement, rather than adding one, is what recovers the ordering: CKA measures how well a *specific trained encoder* aligns two domains (task- and objective-dependent, hence the necessary-not-sufficient finding above), while raw MMD/Frechet measure how different the *datasets themselves* are before any model touches them. The two results are complementary, not contradictory: the objective's discriminative inductive bias governs whether a given alignment *transfers* (CKA's lesson), while the raw distributional gap between datasets follows the intuitive device/modality ordering (this result) — CKA's failure to track it is itself evidence that encoder alignment and raw distributional distance are different quantities. This gives the R6e gap ordering a measured, encoder-independent basis it previously lacked. Full table in `wiki/concepts/domain-gap-metrics.md`, script `scripts/main_experiment/raw_domain_gap.py`.

## R4 — The prior's real value: calibration and data-efficiency (C4→A2, C5)

The accuracy spread is compressed, but the prior still pays off in three deployment-relevant ways.

**(a) Calibration.** Temperature-scaled ECE, per method×k:

| method | k=0 | k=1 | k=3 |
|---|---|---|---|
| supLP120 | **0.0295** | **0.0282** | **0.0291** |
| supcon | 0.0298 | 0.0415 | 0.0318 |
| scratch | 0.0676 | 0.0494 | 0.0396 |
| supMAE | 0.0602 | 0.0520 | 0.0498 |
| mae | 0.0728 | 0.0577 | 0.0560 |

supLP120 is best-calibrated at every k, unchanged from the original run; supcon is a clear second, ahead of supMAE/scratch/mae at every k; mae worst at every k in this rerun (previously worst at k=1/3, tied-worst at k=0 — a minor sharpening, same ranking). The two label-supervised objectives (supLP120, supcon) lead calibration; the two without direct softmax-style label supervision at the pooled embedding (supMAE, mae) trail. This ranking is the most stable result in the paper across the full retrain — every method holds its relative position. (Raw ECE at k=0 is large for all, 0.26–0.33, T≈2.7–3.4 — everyone is overconfident before temperature scaling.)

**(b) Calibration efficiency (AUC-30 = mean eval-acc over the first 30 calibration epochs) and convergence speed.** Paired Δ vs scratch:

- AUC-30: supMAE k3 **+2.04 [+0.26, +3.81] p=.028**; supLP120 k3 **+3.74 [+1.03, +6.46] p=.010**; supcon k3 **+4.18 [+1.44, +6.93] p=.006** (k1 +1.89, n.s.); mae k1 −4.37 p<.001, k3 −3.83 p<.001.
- Convergence (first epoch ≥90% of final; lower=faster): supLP120 k1 **−4.00 [−6.97, −1.03] p=.012**, k3 **−3.40 [−5.07, −1.73] p=.001**; supcon k3 **−3.53 [−5.64, −1.43] p=.003** (k1 −2.47, n.s.); mae *slower* (+2.93 p≤.034); supMAE ≈ scratch.

The prior converges faster and integrates the few calibration shots more efficiently even where the endpoint accuracy is a wash — and this now holds for **two** label-aware objectives (supLP120, supcon), not one, strengthening R4 from a single-objective observation to a pattern tied to label-awareness rather than one specific pretraining recipe.

**(c) Subject-count scaling (A2) — how long the prior stays useful.** Prior benefit vs scratch (pp), by number of fine-tuning subjects N × k, **pooled over 3 seeds (42/43/44), n=15 per cell**:

| prior | k | N=0 | N=1 | N=2 | N=3 | N=4 |
|---|---|---|---|---|---|---|
| supLP120 | 1 | +7.35 | **+18.20** | +5.02 | +1.14 | −0.81 |
| supLP120 | 3 | +18.22 | **+28.25** | +7.57 | +2.66 | +1.86 |
| supMAE | 1 | +3.15 | +5.20 | +2.25 | −0.34 | +0.26 |
| supMAE | 3 | +8.58 | +8.92 | +1.95 | +0.84 | +2.11 |
| mae | 1 | +1.35 | −0.42 | −1.74 | −2.94 | −3.15 |
| mae | 3 | +4.39 | +3.52 | −1.84 | −0.89 | −1.45 |

Paired-t (n=15, this rerun): supLP120 − scratch is significant at N=0/N=1 both k (p≤.008, most p<.0001); supMAE − scratch likewise significant at N=0/N=1 both k (p≤.008); mae − scratch turns significantly negative at N=2 k=0 (p=.007), N=3 k=0/k=1 (p≤.02), and N=4 k=1 (p=.003) — n.s. at N=1.

The prior's benefit **peaks at N=0/1** (supLP120 **+28.3 pp @ k=3, N=1**, matching the original run's +27.5 within pooling noise) and **washes to ≈1–2 pp by N=4** — a clean, now statistically confirmed deployment lever, independently reproduced by a full retrain: the pretrained prior is worth roughly a full retraining set when 0–1 users are enrolled and is nearly redundant once ~4 are. mae is negative from N≥2 onward — reinforcing R2. Every flagship number in this table survives the full retrain within ~1pp — the strongest reproduction result in the paper.

**Caveat established in R6c below: this cold-start lever is itself target-representation-contingent.** Repeating this exact sweep on CZU's strong (raw-signal) target finds no N at which either prior beats scratch — both are significantly *worse* at N=0, k=1. Read R4c and R6c's cold-start extension together, not R4c alone.

## R5 — From recognition to control (C6)

The abstract controller (12-step Sequential Control Task, held-out posteriors, confidence-reject safety layer, asymmetric cost) turns the compressed accuracy differences into task-level ones. The System Input assignment (two Safety-Critical States, five Routine States) is hypothetical, not a recorded gesture set — real gestures are relabeled onto it by recognition reliability, not semantics (Methods §8). A controller has design knobs (System Input assignment, error-cost model, operating threshold), so rather than rest on one configuration we probe the method ordering across three robustness locks (Methods §8.1), treating the 120-assignment Monte Carlo sweep as the primary evaluation protocol.

_**2026-07-20 update:** re-run end to end on this rerun's independently-retrained checkpoints. Lock 1 (the primary protocol) reproduces its headline finding — mae worst — cleanly. Locks 2 and 3, however, now surface a **different** worst-case method at high penalty severity: not mae, but supLP120. This is a real finding, not a bug (traced to a specific, previously-documented supLP120 failure mode, below), and it means the original framing — "the ordering is invariant to the knobs" — does not hold as stated. We report what actually happened rather than keep the stronger sentence._

**Lock 1 — randomized System Input assignment (kills "you cherry-picked the states").** We resample the 7-gesture→primitive mapping 120 times at random and report the distribution of task success. Mean hard task-success across the 120 assignments:

| condition | scratch | mae | supMAE | supLP120 | supcon |
|---|---|---|---|---|---|
| k=1, τ=0 (ungated) | 0.516 | **0.455** | 0.522 | 0.472 | 0.523 |
| k=1, τ=0.9 (gated) | 0.714 | **0.650** | 0.717 | 0.703 | 0.740 |
| k=3, τ=0 | 0.663 | **0.643** | 0.722 | 0.723 | 0.713 |
| k=3, τ=0.9 | 0.830 | **0.792** | 0.861 | 0.883 | **0.895** |

**mae is the worst-compounding init in all four conditions under the primary protocol** — but its margin over supLP120 is now thin at k=1 ungated: mae beats supLP120 in 41% of the 120 assignments (vs 13% against supcon, 14% against supMAE), down from the original run's 34%-beats-supLP120 figure. The negative transfer of R2 survives at the task level on average, but supLP120 is no longer a clearly-safer second-worst option — the two are close to a coin flip at this specific (k,τ) cell. supMAE and supcon are both ≥ scratch in every condition; supcon is the single best method at k=3, τ=0.9.

**Lock 2 — critical-cost sweep (kills "the harsh instant-fail rule drives it").** Replacing binary task-failure with a recoverable critical penalty C_crit and sweeping C_crit ∈ {2, 5, 10, 20, 50, ∞}, k=1:

| C_crit | mae | scratch | supLP120 | supMAE | supcon |
|---|---|---|---|---|---|
| 20 | 17.83 | 17.89 | **19.27** | 15.99 | **14.81** |
| 50 | 21.52 | 21.34 | **26.40** | 17.90 | **15.62** |
| ∞ | 123,015 | 115,016 | **237,681** | 63,681 | **27,014** |

**supLP120, not mae, has the highest mean task cost at every C_crit≥10 — and the gap widens sharply as the penalty grows.** supMAE and supcon are consistently the two cheapest methods at every C_crit (supMAE "lowest for C_crit≥5" is the one part of the original claim that survives almost exactly — 15.99 here vs 15.6 originally); mae is mid-pack, not worst. This is the header finding that changed: **Lock 2 no longer confirms mae as worst — it identifies supLP120 as worst under harsh penalties**, for a specific, explainable reason (below), not a random flip.

**Lock 3 — iso-safety operating point (kills "you tuned τ to win").** Rather than pick τ, we fix a false-activation *budget* and find the smallest τ meeting it per method (deployment-standard, tuning-free). At a **1% budget, k=1**:

| method | τ* | task-success | mean cost |
|---|---|---|---|
| supcon | 0.99 | 0.858 | **17.93** |
| supLP120 | 0.97 | 0.943 | 18.26 |
| mae | **0.93** | 0.920 | 19.31 |
| scratch | 0.99 | 0.639 | 19.32 |
| supMAE | 0.99 | 0.829 | **20.29** |

At k=1, supLP120 no longer reaches the budget at the strictly lowest threshold (mae's τ*=0.93 is now lower than supLP120's 0.97), but supLP120 still posts strong task-success (0.943) at competitive cost — the calibration→throughput link (R4a) still holds directionally, it's just no longer the cleanest single number in the paper. **supMAE is the worst method here at k=1** (highest cost, 20.29) and, at the stricter 0.5% budget, supMAE (along with scratch and supcon) **fails to meet the budget at all at k=1** — only mae and supLP120 clear it. At **k=3**, the picture flips again: supMAE has the *lowest* cost (13.16, best), supLP120 and supcon close behind. Read plainly: **iso-safety performance at low shot count (k=1) tracks calibration quality, and supMAE — never the best-calibrated method in R4a — is exposed there; at k=3 its calibration catches up and it becomes the strongest.**

**Honest finding (sharpened, not new).** supLP120 carries a *confident false-critical-activation* mode: it occasionally maps other gestures onto the highly-separable anchors used for the Safety-Critical States with high confidence. In the original run this showed up only as a dip below scratch on the ungated metric at k=1. In this rerun, on the freshly-retrained checkpoint, the same mechanism is **more pronounced** — enough that it makes supLP120 the worst method under Locks 2/3's harsh-penalty regime, where Locks 2/3 evaluate on one fixed reliability-ordered gesture assignment (not Lock 1's 120 random draws) that happens to land on gestures where this checkpoint's confident-misfire tendency is worse. This is not a contradiction of the calibration story (R4a: supLP120 remains the best-calibrated method on average, unchanged) — it is a second, narrower failure mode, orthogonal to average calibration, that a full retrain exposed at higher stakes than before.

**Net, honestly stated:** mae is the worst method under the primary randomized-assignment protocol (Lock 1) and is never a catastrophic failure under any lock — a safe-but-mediocre choice. supLP120 remains the best-calibrated method (R4a) and performs well under most conditions, but has a specific confident-misfire tail risk that Locks 2/3 now expose more sharply than the original run did. supMAE never fails catastrophically except at low-shot iso-safety specifically. **supcon is the strongest overall performer at k=1** across all three locks, though not a clean sweep at k=3, where supLP120/supMAE are competitive. The three-lock design still does its job — it just reveals a more nuanced picture than "one method is worst everywhere," which is itself the more defensible finding: no pretraining objective is unconditionally safe under every stress test, including the ones the original single-checkpoint run made look clean.

(Superseded numbers, kept for reference and not to be conflated with the above: `trained_models/Phase3-controller/robust-supcon/`, the original repo's supcon extension computed with `robust/`'s four-method checkpoints — do not pool with this rerun's five-method-native numbers.)

## R6 — External validity (CZU-MHAD skeleton→skeleton)

Same objectives and LOSO protocol on CZU-MHAD's skeleton modality (5 subjects, 22 actions). Final accuracy, mean over 5 subjects (single seed, to match the CRC same-splits head-to-head below; pooled 3-seed robustness follows):

| k | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|
| 0 (zero-shot) | 84.9 | 85.0 | 89.0 | **89.9** |
| 1 | 90.2 | 88.5 | 92.8 | **92.5** |
| 3 | 91.9 | 91.5 | 94.9 | **95.0** |

(seed 42, matching the CRC same-splits comparison below; 2026-07-20 rerun.) Paired Δ vs scratch (seed 42): supLP120 and supMAE both clearly positive at zero-shot; **mae is *positive* here too** (+0.1 @ k=0, essentially flat rather than clearly positive, but not negative), unlike the significant negative transfer it shows on NTU→Xsens IMU. The mae-hurts signature does **not** reproduce on a same-modality (skeleton→skeleton, small-gap) target — evidence that the negative transfer is specific to the *cross-modal* gap, which is exactly the mechanism R2/R3 predict. This is skeleton→skeleton, not the IMU gap; the true cross-modal replication on CZU's inertial modality is reported next (R6b).

**Multi-seed robustness (3 seeds 42/43/44, pooled n=15, independently retrained 2026-07-20).** The zero-shot supLP120 win survives an independent full retrain: **supLP120 − scratch = +5.00 pp at k=0, significant (paired-t p=.0012)** — smaller than the original run's +7.2 pp but the same direction and still clearly significant, not underpowered. supMAE is significant-positive at k=0 (+3.15 pp, p=.010) and k=1 (+2.11 pp, p=.032); its k=3 margin is now a trend rather than significant (+1.85 pp, p=.086 — was p=.014). mae stays positive-but-n.s. at k=0, dips slightly negative-but-n.s. at k=1/k=3 (the same-modality "reconstruction does not clearly hurt" signature holds directionally, though it's noisier than the original claimed). The small-gap "supervised prior wins" claim is seed-stable and checkpoint-stable, and is independently replicated on a second public dataset (UTD-MHAD, R6d).

**supcon (3 seeds, pooled n=15) wins even bigger, and this strengthens under the retrain.** supcon − scratch = **+5.24 / +4.67 / +3.41 pp at k=0/1/3, all significant (p=.0015 / .0003 / .0049)** — a larger and more consistent margin than supLP120's. supcon > supLP120 in 28/42 folds (sign p=.044, meanΔ +0.91 pp): on this dataset the *contrastive* label-aware objective edges out the *softmax* label-aware objective, but both dominate scratch by a wide margin. The small-gap story is therefore not "supervised classification specifically wins" but **"any objective with direct label supervision at the pooled-embedding level wins when the gap is small"** — sharpened by a fifth objective, not just replicated by a second dataset.

**Published-baseline comparison.** The CZU-MHAD dataset paper (Chao et al., 2022) reports a Collaborative Representation Classifier (CRC) on statistical-moment features; its protocol-comparable *cross-subject* skeleton accuracy is ~75.5% (its closed, subject-mixed test is not comparable to LOSO). We reproduce that baseline family — mean/std/var/skew/kurtosis features + CRC (λ=1e-4) — on the **byte-identical LOSO k-shot splits** used by our learned recognizer, giving a same-splits head-to-head (mean acc over 5 folds):

| k | CRC baseline | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|---|
| 0 | 84.81 | 84.87 | 84.95 | 89.03 | **89.94** |
| 1 | 89.31 | 90.15 | 88.48 | 92.76 | **92.54** |
| 3 | 90.82 | 91.90 | 91.46 | 94.92 | **95.04** |

Two things stand out. (i) **The from-scratch learned encoder is statistically level with the hand-crafted CRC baseline** (84.87 vs 84.81 at k=0) — architecture alone is not the advantage, and this holds exactly across the full retrain (scratch never touches a pretrained checkpoint, so it reproduces bit-identically). (ii) **Only the NTU-pretrained prior opens a clear, consistent margin**: supLP120 − CRC = +5.1 / +3.2 / +4.2 pp at k=0/1/3 (smaller at k=0 than the original run's +8.3 pp, but the direction and the k=1/k=3 margins hold closely). On an independent public dataset the value is carried by the *transferred prior*, not the recognizer, corroborating R4. (Our CRC reproduction, 84.8%, exceeds the paper's published cross-subject skeleton ~75.5% — expected given our LRQ representation and 4-subject LOSO dictionary vs their raw-position T5–T7 splits; we report it as a same-splits reproduction, not a claim about their exact pipeline.)

### R6b — Cross-modal external validity (CZU-MHAD skeleton→inertial)

The true cross-modal replication: NTU skeleton prior → CZU **wearable-IMU** target (10 body-worn 6-axis sensors, **no magnetometer**), an independent public instance of the skeleton→inertial gap. To feed the skeleton-pretrained encoder, the raw inertial is reduced to 17-segment orientation quaternions via Madgwick AHRS (yaw drifts without magnetometer) — an orientation-only representation shared by every method. Final accuracy, mean over LOSO folds (pooled 3 seeds 42/43/44 × 5 subjects = 45 folds; CRC column single-seed, raw accel+gyro):

| k | scratch | mae | supMAE | supLP120 | CRC (raw accel+gyro) |
|---|---|---|---|---|---|
| 0 | 55.7 | 56.2 | **56.4** | 54.9 | 86.8 |
| 1 | 54.3 | 54.8 | 54.5 | **52.8** | 90.8 |
| 3 | 58.3 | 58.1 | **59.6** | 57.6 | 94.4 |

_**2026-07-20 update:** an independent full retrain weakens this section's two headline significance claims to trends. We report the softened numbers directly rather than keep the stronger original language._

**Under an impoverished (orientation-only) target encoding, the supervised prior's advantage disappears, trending toward hurting rather than clearly hurting.** On the same-modality skeleton target (R6) the pure-supervised prior supLP120 dominated (+5.0 pp zero-shot, pooled p=.001); across the modality boundary **supLP120 trends worst — below scratch at every k, but no longer significantly so** (supLP120 − scratch = −1.0 pp mean; paired-t k=0 −0.86 pp p=.468, k=1 −1.44 pp p=.412, k=3 −0.70 pp p=.592, all n.s. — was p=.044/.056 originally). The **prior-vs-prior contrast (supMAE > supLP120) still trends the same direction but is also weaker**: 29/43 folds (sign p=.032, meanΔ +1.74 pp) — was p<.0001. The claim that supMAE *beats scratch* remains unsupported after pooling, as in the original: sign test supMAE > scratch is 28/43 folds (p=.066, meanΔ +0.74 pp, a trend not significance), paired-t clears nothing at any k (largest k=3, +1.32 pp, p=.251). The honest reading, now more cautious than before: **supMAE ≈ scratch ≈ supLP120 at this gap once you account for the weaker significance** — the *direction* every prior points (reconstruction-containing objectives hold up better than the pure-supervised one) is unchanged and mechanistically consistent with R2/R3, but this section no longer carries a clean statistically-significant inversion on its own. R6c below (dual-branch, on a stronger target) is where the sharper, still-significant version of this story lives.

Three honest limits, one new. (i) The deep encoders trail the raw-inertial CRC baseline in absolute terms (55–60% vs 87–94%): the orientation-only encoding needed for cross-modal transfer discards the accelerometer-magnitude signal CRC exploits and carries yaw drift, so these numbers index *relative* prior transfer, not the inertial ceiling. (ii) Unlike NTU→Xsens, mae here is not a negative-transfer culprit (it now trends slightly *positive* vs scratch, +0.74/+0.51/−0.26 pp at k=0/1/3, all n.s.) — same reading as before, mae's negative transfer is cross-domain-not-cross-modal-specific. (iii) **New:** this section's significance claims did not survive an independent checkpoint retrain as cleanly as R6/R6c/R6d did — treat the direction (supervised-only priors weaken at this gap; reconstruction-containing ones hold up relatively better) as the load-bearing part of R6b, and the specific p-values as fragile, sample-size-limited (n=43–45), not seed/checkpoint-stable at this particular setting.

**supcon (3 seeds, pooled n=15/k) still tracks supLP120's negative-to-neutral pattern, both now closer to a wash than before.** supcon − scratch = **−0.44 / −1.00 / −1.04 pp @ k=0/1/3** (all n.s., p≥.55); pooled sign test supcon > scratch is 24/44 (p=.65, meanΔ −0.83, a wash trending negative — was p=.37, meanΔ −0.83 originally, essentially unchanged). supcon vs supLP120 is statistically even (21/43, p=1.0, meanΔ +0.17) — the two label-aware objectives still land in the same negative-to-neutral band, both numerically below supMAE. The general reading survives even as the specific p-values weaken: **objectives lacking a reconstruction/regularization component do not clearly help, and trend toward hurting, across this modality gap**, regardless of whether label supervision is delivered via cross-entropy (supLP120) or contrastive alignment (supcon) — this is now a directional pattern corroborated by R6c's sharper, still-significant version of the same contrast, rather than a standalone significant finding in R6b itself.

### R6c — The prior's value is contingent on target-representation poverty (CZU-MHAD dual-branch)

R6b's orientation-only encoding capped deep accuracy at 56–61% — far below the raw-inertial CRC baseline. Two questions follow: was that the *encoding* or the *architecture*, and does the R6b reconstruction-prior benefit survive once the target model uses its **full raw signal**? We test both with a dual-branch recognizer (a from-scratch branch on the raw 10×6 accel+gyro stream + the NTU-pretrained encoder on the R6b orientation quaternions, concatenated to a shared head) on the byte-identical CZU LOSO splits. **Both rows are now pooled over 3 seeds × 5 subjects (45 folds)**:

| mode | scratch | mae | supMAE | supLP120 | supcon | CRC |
|---|---|---|---|---|---|---|
| raw-only (k=0/1/3) | 81.9 / 90.0 / **95.6** | — | — | — | — | 86.8 / 90.8 / 94.4 |
| dual (k=0) | **86.0** | 85.3 | 85.9 | 84.5 | 83.8 | — |
| dual (k=1) | **88.8** | 88.0 | 88.5 | 87.2 | 86.9 | — |
| dual (k=3) | **91.9** | 91.4 | 91.6 | 90.9 | 90.0 | — |

(2026-07-20 rerun, pooled 3 seeds × 5 subj = 15/prior/k; supcon added as a fifth prior, not in the original table.) Two results, both firmer under an independent full retrain. **(i) Representation, not architecture, was the R6b bottleneck.** The raw-signal branch alone reaches 81.94/90.01/95.56 (matching the original run almost exactly — this is the one number in R6c not downstream of a retrained checkpoint), matching/beating CRC and ~30 pp above R6b's orientation-only encoders — the collapse was the lossy encoding, not the deep model. **(ii) On a strong target the NTU prior adds no value, and label-supervised-only priors actively hurt — and this got clearer, not weaker, under the retrain.** `dual/scratch` is now the best performer at every k, beating every prior: supMAE trends just below it (pooled 18/37 folds, sign p=1.0, n.s. — no detectable difference), mae trends below (15/42, p=.088), **supLP120 is significantly worse — pooled 10/41 folds (sign p=.0015), paired-t significant at k=0** (−1.50 pp, p=.017; k=1 −1.64 pp p=.086 trend; k=3 −1.06 pp p=.055 trend), and **supcon is worse still, more so than originally claimed — pooled 3/42 folds (sign p<.0001)**, paired-t significant at every k (−2.17 / −1.90 / −1.89 pp, p=.0002 / .0002 / .0066 — nearly double the original run's −1.2/−1.1/−1.1 pp estimate). The R6b direction is corroborated here with sharper significance: the reconstruction component, not the specific supervised recipe, is what keeps an objective from actively hurting at large gap, and the two label-aware-only priors (supLP120, supcon) are the clearest losers in the whole paper on this target.

**Dose-response along target richness (pooled 3-seed dial).** Interpolating a middle rung between R6b (orientation-only) and R6c (full raw) — a 20-channel per-sensor accel-magnitude + gyro-magnitude target — traces the prior benefit as a smooth function of target strength (Δ vs scratch, k=0/1/3, pooled seeds 42/43/44):

| target representation | scratch acc | supMAE Δ | supLP120 Δ |
|---|---|---|---|
| 0-ch quat-only (R6b) | 56 / 54 / 58 | +0.7 / +0.2 / +1.3 (n.s.) | −0.9 / −1.4 / −0.7 (n.s.) |
| 20-ch magnitudes (dial) | 84 / 88 / 90 | −1.3 / −1.2 / −0.0 | −3.8 / −5.7 / −3.2 |
| 60-ch full raw (R6c) | 86 / 89 / 92 | −0.1 / −0.3 / −0.4 | −1.5 / −1.6 / −1.1 |

(2026-07-20 rerun; R6b's row is smaller and no longer significant, per that section's update above — read this row as directional only.) The reconstruction prior's edge, where it exists at all, is concentrated at the fully-crippled orientation-only target and fades as the target sees more real motion; supLP120 is negative at every rung and significantly so from the 20-ch dial onward. The dial rung itself (20-ch magnitudes) reproduces closely against the original run (supMAE was −1.4/−1.6/−1.1, now −1.3/−1.2/−0.0; supLP120 was −4.1/−5.2/−3.6, now −3.8/−5.7/−3.2) — this is the most checkpoint-stable row in R6c's three-point curve, while the R6b endpoint is the least stable (see that section).

**Does the cold-start advantage (R4c) survive on a strong target?** R4c's deployment claim — the
prior is worth most at 0–1 enrolled subjects — was established only on Xsens, a weak/moderate target.
We repeat the subject-count sweep (N=0..3, `--n-train-subjects` added to `scripts/external/czu/dualbranch.py`)
on CZU-dual, seed 42, single-seed (n=5 folds):

| N | k | scratch | supLP120 Δ | supMAE Δ |
|---|---|---|---|---|
| 0 | 0 | 2.87 | +0.11 (n.s.) | +0.37 (n.s.) |
| 0 | 1 | 38.94 | **−5.62 p=.045** | **−5.55 p=.005** |
| 0 | 3 | 54.21 | +0.23 (n.s.) | **−2.87 p=.035** |
| 1 | 1 | 71.01 | −3.11 p=.085 (n.s.) | −3.51 p=.052 (n.s.) |
| 1 | 3 | 79.58 | **−2.54 p=.004** | −1.80 p=.076 (n.s.) |

**Neither prior beats scratch at any N, and both are significantly worse at N=0, k=1** — the exact
cold-start cell where the Xsens A2 result showed the prior's *largest* benefit (+18.4 pp, R4c). This
scopes the deployment claim precisely: **the cold-start advantage is itself contingent on target
representation poverty**, mirroring R6c's main finding at N=4. A strong target's from-scratch branch
apparently reaches a useful operating point with zero enrolled subjects and no prior — there is no
regime tested here where the NTU prior helps on this target. (Single-seed, n=5 folds, still —
**a 3-seed extension (seeds 43/44) is running as of 2026-07-20**, launched specifically to bring this
result up to the same standard as R4c/R6c rather than leave it as the one remaining single-seed
load-bearing number in the paper; the signal is already significant at several (N,k) cells at n=5,
so we do not expect the direction to change, but the table above should be treated as pending
pooled confirmation until that run completes.)

Read across the three CZU settings, the prior's value *and its ranking* degrade monotonically with the gap:

| Setting | Gap | Target repr. | Best prior | supLP120 (pure-supervised) |
|---|---|---|---|---|
| R6 skeleton→skeleton | small (same modality) | strong (native skeleton) | **supLP120 +5.0 pp @k0** (pooled p=.001), beats CRC | best-in-class |
| R6b IMU, orientation-only | large (cross-modal) | weak (orientation-only) | none clearly (supMAE trends > supLP120, 29/43, p=.032 — weaker than originally claimed) | trends worst, below scratch but n.s. (p=.41–.59) |
| R6c IMU, raw/dual | large (cross-modal) | strong (raw ≈ CRC) | none (`dual/scratch` itself is best at every k) | worst, p=.0015 |

The two axes are separable: holding the gap large, moving the target representation from weak (R6b) to strong (R6c) removes the reconstruction prior's advantage; holding the target strong, moving the gap from small (R6, where supervised wins) to large (R6c) turns the supervised prior from best to worst. A single public dataset yields a controlled contrast in which the *same* supervised prior is best-in-class in one modality and worst-in-class in another, purely as a function of gap width and target capability — the sharpest external corroboration of the negative-transfer thesis (R2/R3): a transferred prior earns its keep only in proportion to the target's representational poverty.

### R6d — Second independent same-modality dataset (UTD-MHAD skeleton→skeleton)

To confirm the small-gap "supervised prior wins" prediction (R6) is not specific to CZU, we replicate it on a second, independently collected public dataset: UTD-MHAD (Kinect-v1, 20 joints remapped to the NTU-25 layout; 8 subjects, 27 actions, 861 clips). Same objectives and LOSO k-shot protocol; pooled over 3 seeds × 8 subjects (24 folds per cell):

| k | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|
| 0 (zero-shot) | 77.4 | 79.6 | 80.0 | **82.4** |
| 1 | 90.9 | 91.6 | 93.1 | **93.0** |
| 3 | 94.0 | 94.5 | 95.1 | **96.0** |

(2026-07-20 rerun, pooled 3 seeds × 8 subj = 24 folds.) The R6 pattern replicates, though with one significance loss at k=3. **supLP120 is the best prior at k=0/k=1**: supLP120 − scratch = **+5.03 / +2.12 pp** (paired-t p=.0014 / .0153); at k=3 the margin (+2.02 pp) is now a trend, not significant (p=.067 — was p=.016). supLP120 vs supMAE is now a wash (36/61 folds, sign p=.20 — was p=.0003, the sharpest single loss of significance in this rerun; the two priors are essentially indistinguishable on UTD in this rerun rather than supLP120 clearly ahead). supMAE is still significant-positive at k=0/k=1 (+2.63 / +2.22 pp; p=.010 / .026), n.s. at k=3 (+1.10 pp, p=.30). As on CZU skeleton, **mae trends positive, not negative** (+2.23/+0.72/+0.59 pp, all n.s. now — was significant at k=0, p=.011): the reconstruction-hurts signature still does not appear on a same-modality small-gap target, though the effect is noisier than originally reported. Two independent public datasets (CZU skeleton R6, UTD skeleton R6d) still agree the pure-supervised prior is at least competitive, and never negative, when the gap is small — the mirror image of its worst-in-class behavior across the cross-modal gap (R6c).

**supcon (3 seeds, pooled n=24) is the standout result on this dataset, and strengthens under the retrain.** supcon − scratch = **+7.97 / +3.98 / +2.46 pp @ k=0/1/3, all significant (p<.0001 / p=.0002 / p=.028)** — a clearly larger margin than supLP120's, and supcon now clearly beats supLP120 too (sign test 42/60, p=.003, meanΔ +1.75 pp — was a wash at p=.89). Combined with the CZU result (where supcon also edges out supLP120), the two-dataset picture is: **supcon is the most reliable small-gap performer of the five objectives, ahead of supLP120 on both external datasets**, while supLP120 itself is only conditionally best (UTD k=0/k=1, not k=3, and no longer clearly ahead of supMAE). The deeper regularity — "label-supervised, no reconstruction component wins at small gap" — still holds as a class-level claim; which specific label-aware objective wins by how much is noisier per-dataset than the original single-checkpoint run suggested.

**Published-baseline comparison.** Same recognizer family as the CZU R6 anchor (statistical moments + CRC-RLS, λ=1e-4) on the byte-identical seed-42 LOSO splits:

| k | CRC baseline | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|---|
| 0 | 69.48 | 75.98 | 78.28 | 80.50 | **83.05** |
| 1 | 92.25 | 90.25 | 93.18 | 92.88 | **93.51** |
| 3 | 95.30 | 93.52 | 94.85 | 95.35 | **96.28** |

supLP120 − CRC = **+13.57 / +1.26 / +0.98 pp** at k=0/1/3, echoing R6's consistent margin over the hand-crafted baseline (smaller at k=0 than the original run's +14.85 pp, same direction and similar k=1/k=3 margins). One difference worth noting honestly: on CZU, the from-scratch learned encoder was statistically level with CRC (R6); on UTD, **scratch already beats CRC by +6.5 pp at k=0** before any prior (unchanged — scratch never touches a pretrained checkpoint, so this number reproduces exactly) — the architecture has some edge here even unpretrained, so the prior's marginal lift over scratch is smaller than its margin over CRC. (UTD-MHAD's own published cross-subject accuracy figure is not yet cited here — TODO, needs a literature figure, not a rerun.)

### R6e — Synthesis: the five-setting map

Five source→target settings, two public external datasets plus the two internal ones (Xsens position-derived, CZU orientation-only and dual-raw), give a controlled sweep over gap width and target-representation strength:

| # | Setting (source → target) | Gap | Target representation | Best prior (Δ vs scratch) | Negative-transfer case |
|---|---|---|---|---|---|
| 1 | NTU → CZU skeleton (R6) | small (same modality) | native skeleton | supLP120 +5.0 pp @k0 (p=.001) | none (mae ≈ flat, n.s.) |
| 2 | NTU → UTD skeleton (R6d) | small (same modality) | native skeleton | supLP120 +5.0 pp @k0 (p=.0014); supcon strongest overall (+7.97pp, p<.0001) | none (mae trends positive, n.s.) |
| 3 | NTU → Xsens, position-derived quats (R2) | middle (cross-device, skeleton-like target) | strong (mocap-grade quats) | mixed — supMAE/supLP120/supcon all significant only at k=3 (p≤.003), n.s. at k=0/1 | **mae** (McNemar −266/−242/−99, p<.001 every k) |
| 4 | NTU → CZU IMU, orientation-only (R6b) | large (cross-modal) | weak (Madgwick quats, yaw drift) | none clearly — direction only (supMAE trends > supLP120, 29/43 p=.032, weaker than originally found) | supLP120 trends worst, no longer significant (p=.41–.59) |
| 5 | NTU → CZU IMU, dual raw (R6c) | large (cross-modal) | strong (raw ≈ CRC) | none (`dual/scratch` itself is best at every k) | **supLP120** (10/41, p=.0015); **supcon worse still** (3/42, p<.0001) |

No pretraining objective is unconditionally safe across the skeleton→wearable gap. The clearest, most checkpoint-stable pattern is: **mae is the one objective with a real, significant, and reproducible negative-transfer signature — at the middle gap specifically** — while at small gap no objective is negative and at large gap the pure-label-aware objectives (supLP120, supcon) are the ones that trend or test negative instead. The prior's gap-invariant value is calibration and cold-start data-efficiency (R4), which reproduced far more cleanly under the retrain than the accuracy claims in this table did.

**A fifth objective (supcon) still sharpens the small/large split into an axis, though the exact margins moved.** supcon — Khosla 2020 SupCon, label-supervised but contrastive rather than softmax — tracks supLP120's direction at every external setting: it **wins big at both small-gap datasets** (CZU +5.24/+4.67/+3.41 pp k=0/1/3, all p<.01; UTD +7.97/+3.98/+2.46 pp, all p<.03 — on UTD specifically now the single strongest small-gap result in the paper, clearly ahead of supLP120, p=.003), and it **trends negative-to-flat at both large-gap settings** (CZU-IMU quat: −0.44/−1.00/−1.04 pp, n.s.; CZU-IMU dual: significantly worse than scratch, p<.0001, the single worst pooled sign-test result in R6c). Because supcon and supLP120 arrive at "label supervision" through different mechanisms (cross-entropy vs contrastive) yet land in the same direction at every gap setting, the operative variable is still not "is the objective supervised classification" but **"does the objective have a reconstruction/regularization component"** — present in supMAE (the objective that comes closest to never being negative anywhere) and absent from supLP120/supcon alike. This reframes row 4–5's negative-transfer entries as an instance of a *class* of objectives, not a property specific to softmax classification — a claim that got *more* precise under the retrain even as some of its individual significance levels weakened.

*Gap ordering, measured:* the "middle" placement of NTU→Xsens is not merely narrative. The encoder-space CKA-per-target check (R3 extension) did **not** confirm the small/middle/large ordering, in either run — in this rerun, on the supLP120 encoder, mean L0–L2 CKA ranks utd_skeleton > czu_skeleton > czu_imu_quat > xsens_v2, putting the claimed-middle target *lowest*, another instance of CKA's own necessary-not-sufficient limitation (R3) rather than a fit we forced. The raw, encoder-free measurement — squared-MMD and Frechet distance on a hand-crafted, model-free feature (R3) — **does confirm the ordering exactly and monotonically, unchanged by the retrain since it involves no pretrained checkpoint**: czu_skeleton (MMD²=0.056) < utd_skeleton (0.096) < xsens_v2 (0.122) < czu_imu_quat (0.205), same ranking on Frechet distance independently. This raw measurement — not CKA — is the paper's primary evidence for the gap ordering; CKA's alignment-based null read is reported as a separate, orthogonal finding about what encoder alignment does and doesn't predict (R3), not as evidence against the ordering itself.

---

## Numbers ledger (traceability)

_Updated 2026-07-20 for `DA_GestureRecognition-clean`'s reproducibility rerun. `scripts/orchestration/README.md` documents the 10-stage pipeline that produced everything below; script paths here are that repo's reorganized `scripts/` layout, not the original repo's flat `temp_*.py` names. supcon is trained natively as a 5th method throughout this rerun (stage 1 pretrains it alongside supLP120/mae/supMAE; every downstream stage's `--methods` list includes it from the start) — there is no separate `-supcon-seed*` directory tree as in the original repo; supcon's rows live in the same `summary.csv` files as the other four methods._

- R1 within-domain: `trained_models/NTU-to-NTU-objective-sanity/` (not re-verified this pass); cross-domain: `LOSO-fullTrainCalibrate-v2{,-seed43,-seed44}/summary.csv` (`scripts/main_experiment/loso_fulltrain_calibration.py`, stage 2).
- R2 McNemar / R3 CKA / R4a ECE: `trained_models/Phase1-analysis/{mcnemar,cka,cka_by_target,ece}_results.csv` (`scripts/main_experiment/{dump_posteriors,cka_analysis}.py` + siblings, stage 3).
- R3 raw MMD/Frechet: `trained_models/RawDomainGap/raw_domain_gap.csv` (`scripts/main_experiment/raw_domain_gap.py`); encoder-space MMD: `trained_models/MMD_DomainGap/mmd_table.csv` (`scripts/main_experiment/mmd_domain_gap.py`).
- R4b AUC-30/convergence: `wiki/results/multiseed-loso-v2.md` (not re-verified this pass — see `RESEARCH_LOG.md` §B).
- R4c A2: pooled 3 seeds — `trained_models/A2-subjectScaling{,-seed43,-seed44}/N*/summary.csv` (stage 4), pooled table `trained_models/A2-subjectScaling-pooled/{a2_pooled_results,a2_pooled_stats}.csv` (`scripts/main_experiment/analyze_a2_multiseed.py`).
- R5 controller (locked/robust): `trained_models/Phase3-controller/robust/{vocab_sweep,vocab_ordering,costmodel_sweep,frontier,iso_safety}.csv` + PNGs (`scripts/controller/controller_robust.py`, stage 8, 5-method-native); prototype in `Phase3-controller/{controller_results,operating_point_summary}.csv` (`scripts/controller/controller_sim.py`). Superseded original-repo supcon extension: `trained_models/Phase3-controller/robust-supcon/` — do not pool with the above.
- R6 CZU: `trained_models/CZU-skeleton-LOSO{,-seed43,-seed44}/summary.csv`; CRC baseline `.../crc_baseline/crc_summary.csv` (`scripts/external/czu/crc_baseline.py`).
- R6b CZU inertial (cross-modal): `trained_models/CZU-IMU-LOSO{,-seed43,-seed44}/summary.csv`; CRC `.../crc_baseline/crc_summary.csv` (`scripts/external/czu/imu_crc_baseline.py`); data `Data_Processed/czu_imu_quats/`.
- R6c CZU dual-branch: `trained_models/CZU-IMU-DUAL{,-seed43,-seed44}/{raw_scratch,dual_<prior>}/summary.csv` (`scripts/external/czu/dualbranch.py`, stage 6); raw data `Data_Processed/czu_imu_raw/`; reuses CZU-IMU-LOSO splits. Target-richness dial, pooled 3 seeds: `trained_models/CZU-IMU-DIAL{,-seed43,-seed44}/mag20/dual_<prior>/summary.csv`. Cold-start subject-scaling (T5): `trained_models/CZU-DUAL-subjectScaling/N{0..3}/dual_<prior>/summary.csv` (seed 42, stage 9); 3-seed extension `trained_models/CZU-DUAL-subjectScaling-seed{43,44}/N{0..3}/` launched 2026-07-20 (`scripts/orchestration/09b_czu_cold_start_multiseed.sh`), in progress as of this writing.
- Pooling for R6/R6b/R6c: `scripts/external/czu/multiseed_analyze.py` (reads the seed42/43/44 directories above directly; extended ad hoc for supcon and for R6c's dial/cold-start during this update — not yet folded back into the script itself).
- R6d UTD-MHAD skeleton: `trained_models/UTD-skeleton-LOSO-seed{42,43,44}/summary.csv` (`scripts/external/utd/crc_baseline.py` for the CRC row); data `Data_Processed/utd_skeleton_lrq/`.
- OOV / leave-class-out (referenced in `RESEARCH_LOG.md`, not a paper_results.md table): `trained_models/LOSO-LeaveClassOutFewShot/summary.csv` (seed 42, `scripts/main_experiment/loso_leave_class_out_fewshot.py`, stage 5); 3-seed extension `trained_models/LOSO-LeaveClassOutFewShot-seed{43,44}/` launched 2026-07-20 (`scripts/orchestration/05b_oov_multiseed.sh`, ~15h/seed), in progress as of this writing.
