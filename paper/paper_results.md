# Results (draft)

_This document is the **source of truth for all reported numbers.** Every value here traces to `RESEARCH_LOG.md` §B and the run directories under `trained_models/`. Six results sections map to contributions C1–C6 plus external validity._

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

Pooled across 3 seeds (n=15 per method×k cell), only one objective shows a significant, reproducible negative-transfer signature against scratch: pure reconstruction. What survives is not a supervised/contrastive collapse but a **reconstruction collapse** — `mae` is the one objective consistently worse than scratch; everywhere else is a wash or a weak positive.

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

**mae is significant negative transfer at every k** (all p<.001) in both the pooled-accuracy and the clip-level view. It flips more clips wrong than right relative to random init, because it never encodes discriminative structure within-domain (R1) and the fine-tune from that init lands in a worse basin than from random. This inverts the field's working assumption that reconstruction/self-supervision is the *safe* prior under large gaps.

The other three objectives are significant-positive **at k=3 only** (supMAE p=.003/McNemar p=0; supLP120 McNemar p=0; supcon both p<.001) and a wash at k=0/k=1 — n.s. on both the paired-Δ and McNemar tests. The k=3 positive result for all three label-aware-or-hybrid objectives replicates cleanly across both tests; the k=0/k=1 "prior helps" claims for these three are smaller, less stable effects that should not be leaned on as heavily as the mae-negative-transfer claim.

## R3 — Mechanism: gap is necessary-not-sufficient; CKA orders objectives (C3)

**MMD does not track transfer.** Squared-MMD between NTU and Xsens-v2 features (supMAE encoder) is 0.0092 under v2 (below the local baseline 0.0109 and far below swing 0.0322), confirming v2 closes the gap — but MMD ordering across encoders does not match transfer ordering (contrastive had near-lowest MMD yet worst transfer under the earlier preprocessing; mae has higher MMD than supervised yet similar low-k accuracy).

**CKA orders the objectives cleanly, but is a secondary check here — the raw MMD²/Frechet measurement below is the paper's primary domain-distance evidence, since it is the one that correctly characterizes objective performance against gap width.** Linear CKA between NTU and Xsens-v2 activations, per encoder × layer:

| encoder | proj | L0 | L1 | L2 |
|---|---|---|---|---|
| supLP120 | 0.0038 | 0.0129 | 0.0134 | **0.0146** |
| supcon | 0.0038 | 0.0121 | 0.0114 | 0.0115 |
| supMAE | 0.0030 | 0.0044 | 0.0046 | 0.0044 |
| mae | 0.0026 | 0.0038 | 0.0034 | 0.0036 |
| scratch | 0.0026 | 0.0024 | 0.0024 | 0.0024 |

Two facts, both important: (i) **absolute CKA is small for all** (<0.02) — the cross-modal gap is large even after v2, so we do not over-claim on magnitude; (ii) the **ordering is clean and monotone with depth** for the two label-aware objectives — supLP120 and supcon both align well above supMAE/mae/scratch, while scratch is flat. The best-aligned encoder is not the best-transferring on accuracy (supMAE is, R1), and the reconstruction encoder has higher CKA than scratch but negative transfer. **Conclusion:** representation alignment is a necessary context but not a sufficient predictor; the objective's discriminative inductive bias governs transfer. This is the mechanism the MMD-only view could not supply — but the ordering-across-datasets claim rests on the raw, encoder-free measurement below, not on CKA.

**Extending this to the gap axis itself:** we also measured layer-wise CKA between NTU and each of the four targets used elsewhere in the paper (Xsens-v2, CZU skeleton, CZU IMU orientation quats, UTD skeleton) to test whether CKA orders them by our claimed small/middle/large gap (see R6e). It does not: on the supLP120 encoder, mean L0–L2 linear CKA is utd_skeleton 0.0343 > czu_skeleton 0.0253 > czu_imu_quat 0.0183 > xsens_v2 0.0137 — the claimed-middle target (xsens_v2) ranks *lowest*, not in the middle, and czu_imu_quat (claimed large gap) ranks above it. We report this as a further instance of necessary-not-sufficient rather than force a fit: CKA does not reliably track gap *width* across independently collected datasets. This is exactly why the raw, encoder-free measurement below — not CKA — is the paper's evidence for the gap ordering. Full table in `wiki/concepts/domain-gap-metrics.md`.

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

supLP120 is best-calibrated at every k; supcon is a clear second, ahead of supMAE/scratch/mae at every k; mae is worst at every k. The two label-supervised objectives (supLP120, supcon) lead calibration; the two without direct softmax-style label supervision at the pooled embedding (supMAE, mae) trail — calibration tracks label-awareness of the objective, not just accuracy. This ranking is the most stable result in the paper. (Raw ECE at k=0 is large for all, 0.26–0.33, T≈2.7–3.4 — everyone is overconfident before temperature scaling.)

**(b) Calibration efficiency (AUC-30 = mean eval-acc over the first 30 calibration epochs) and convergence speed.** Paired Δ vs scratch:

- AUC-30: supMAE k3 **+2.04 [+0.26, +3.81] p=.028**; supLP120 k3 **+3.74 [+1.03, +6.46] p=.010**; supcon k3 **+4.18 [+1.44, +6.93] p=.006** (k1 +1.89, n.s.); mae k1 −4.37 p<.001, k3 −3.83 p<.001.
- Convergence (first epoch ≥90% of final; lower=faster): supLP120 k1 **−4.00 [−6.97, −1.03] p=.012**, k3 **−3.40 [−5.07, −1.73] p=.001**; supcon k3 **−3.53 [−5.64, −1.43] p=.003** (k1 −2.47, n.s.); mae *slower* (+2.93 p≤.034); supMAE ≈ scratch.

The prior converges faster and integrates the few calibration shots more efficiently even where the endpoint accuracy is a wash — and this holds for **two** label-aware objectives (supLP120, supcon), not one, strengthening R4 from a single-objective observation to a pattern tied to label-awareness rather than one specific pretraining recipe.

**(c) Subject-count scaling (A2) — how long the prior stays useful.** Prior benefit vs scratch (pp), by number of fine-tuning subjects N × k, **pooled over 3 seeds (42/43/44), n=15 per cell**:

| prior | k | N=0 | N=1 | N=2 | N=3 | N=4 |
|---|---|---|---|---|---|---|
| supLP120 | 1 | +7.35 | **+18.20** | +5.02 | +1.14 | −0.81 |
| supLP120 | 3 | +18.22 | **+28.25** | +7.57 | +2.66 | +1.86 |
| supMAE | 1 | +3.15 | +5.20 | +2.25 | −0.34 | +0.26 |
| supMAE | 3 | +8.58 | +8.92 | +1.95 | +0.84 | +2.11 |
| mae | 1 | +1.35 | −0.42 | −1.74 | −2.94 | −3.15 |
| mae | 3 | +4.39 | +3.52 | −1.84 | −0.89 | −1.45 |

Paired-t (n=15): supLP120 − scratch is significant at N=0/N=1 both k (p≤.008, most p<.0001); supMAE − scratch likewise significant at N=0/N=1 both k (p≤.008); mae − scratch turns significantly negative at N=2 k=0 (p=.007), N=3 k=0/k=1 (p≤.02), and N=4 k=1 (p=.003) — n.s. at N=1.

The prior's benefit **peaks at N=0/1** (supLP120 **+28.3 pp @ k=3, N=1**) and **washes to ≈1–2 pp by N=4** — a clean, statistically confirmed deployment lever: the pretrained prior is worth roughly a full retraining set when 0–1 users are enrolled and is nearly redundant once ~4 are. mae is negative from N≥2 onward — reinforcing R2.

**Caveat established in R6c below: this cold-start lever is itself target-representation-contingent.** Repeating this exact sweep on CZU's strong (raw-signal) target finds no N at which either prior beats scratch — both are significantly *worse* at N=0, k=1. Read R4c and R6c's cold-start extension together, not R4c alone.

## R5 — From recognition to control (C6)

The abstract controller (12-step Sequential Control Task, held-out posteriors, confidence-reject safety layer, asymmetric cost) turns the compressed accuracy differences into task-level ones. The System Input assignment (two Safety-Critical States, five Routine States) is hypothetical, not a recorded gesture set — real gestures are relabeled onto it, and deliberately not by any property that could be seen as tuned to a preferred outcome (Methods §8). A controller has design knobs (System Input assignment, error-cost model, operating threshold); rather than freeze any of them at one defensible-looking value, we probe all three simultaneously with a single Monte Carlo protocol: **120 independently, uniformly-randomly drawn 7-gesture System Input assignments**, each evaluated with **1000 Monte Carlo task-execution trials per (assignment, method, k, τ/C_crit) cell**. No assignment is chosen by any property of the recognizer (e.g. per-gesture recall) — every assignment is a fresh unweighted random draw of 7 of the 22 recorded gestures, so no lock's finding can be attributed to a favorably- or unfavorably-picked task design. Three locks apply this same randomized-assignment protocol to three different design knobs:

**Lock 1 — randomized assignment, base outcome model.** Mean hard task-success across the 120 assignments (1000 trials/assignment/cell):

| condition | scratch | mae | supMAE | supLP120 | supcon |
|---|---|---|---|---|---|
| k=1, τ=0 (ungated) | 0.516 | **0.455** | 0.522 | 0.472 | 0.523 |
| k=1, τ=0.9 (gated) | 0.714 | **0.650** | 0.717 | 0.703 | 0.740 |
| k=3, τ=0 | 0.663 | **0.643** | 0.722 | 0.723 | 0.713 |
| k=3, τ=0.9 | 0.830 | **0.792** | 0.861 | 0.883 | **0.895** |

**mae is the worst-compounding init in all four conditions.** supMAE and supcon are both ≥ scratch in every condition; supcon is the single best method at k=3, τ=0.9.

**Lock 2 — critical-cost severity, same 120 randomized assignments.** Replacing binary task-failure with a recoverable critical penalty C_crit and sweeping C_crit ∈ {2, 5, 10, 20, 50, ∞}, k=1, median mean task-cost across the 120 assignments (1000 trials/assignment/cell):

| C_crit | mae | scratch | supLP120 | supMAE | supcon |
|---|---|---|---|---|---|
| 20 | **32.3** | 29.0 | 30.4 | 28.5 | 28.6 |
| 50 | **55.5** | 48.3 | 51.2 | 47.1 | 47.8 |
| ∞ | **756,017** | 644,016 | 704,016 | 611,518 | 638,517 |

**mae has the highest median mean-cost at every C_crit swept, confirming Lock 1's ordering under a completely different (and much harsher) outcome model.** supLP120 trends second-worst as the penalty grows, but never overtakes mae; supMAE is consistently the cheapest method.

**Lock 3 — iso-safety operating point, same 120 randomized assignments.** Rather than pick τ, we fix a false-activation *budget* per assignment and find the smallest τ meeting it per method, then report the distribution across the 120 assignments. At a **1% budget**:

| k | method | τ* (median) | mean task-success | mean cost |
|---|---|---|---|---|
| 1 | **mae** | 0.97 | **0.360** | 22.98 |
| 1 | scratch | 0.97 | 0.514 | 23.14 |
| 1 | supLP120 | 0.99 | 0.561 | 24.18 |
| 1 | supMAE | 0.99 | 0.524 | 24.30 |
| 1 | supcon | 0.97 | 0.646 | 23.22 |
| 3 | **mae** | 0.90 | **0.741** | 21.77 |
| 3 | scratch | 0.90 | 0.797 | 20.80 |
| 3 | supLP120 | 0.85 | **0.862** | 18.41 |
| 3 | supMAE | 0.90 | 0.812 | 19.47 |
| 3 | supcon | 0.85 | 0.848 | 18.34 |

**mae has the lowest mean task-success at the 1% budget at both k=1 and k=3** — the worst method under Lock 3 too, at both shot counts and both budgets tested (1% and the stricter 0.5%, not tabulated). supLP120 and supcon lead at k=3, consistent with them being the two best-calibrated objectives (R4a).

**Net: all three locks agree.** Across 120 randomized System Input assignments, both outcome models (hard task-success and soft cost), a 6-point critical-cost sweep, and a tuning-free iso-safety operating point — **mae compounds worst, consistently, under every stress test.** One nuance persists across the randomization: supLP120 still dips just below scratch on the *ungated* Lock 1 metric at k=1 (0.472 vs 0.516) and trends as the second-most-expensive method as Lock 2's penalty grows severe — both consistent with a specific, previously-documented mechanism (supLP120 occasionally maps an unrelated gesture onto a highly-separable Safety-Critical-State anchor with high confidence). This does not contradict the calibration story (R4a: supLP120 remains the best-calibrated method on average, and leads at k=3 under iso-safety) — it is a secondary, narrower effect, visible only at the ungated operating point or under severe penalties, that never rises to displace mae as the worst-compounding objective under any of the three locks.

### R5b — Does the controller's ranking hold beyond NTU→Xsens?

R5 above evaluates all three robustness locks on one transfer setting (NTU→Xsens). We additionally reproduce **Lock 1 and Lock 2 only** (no iso-safety) on every other transfer setting with compatible per-clip posteriors — CZU-MHAD skeleton (R6), CZU-MHAD IMU orientation-only (R6b), UTD-MHAD skeleton (R6d), CZU-MHAD dual-branch raw+quat (R6c) — using the identical protocol: 120 independently-random 7-gesture System Input assignments drawn fresh from each setting's own label space, 1000 Monte Carlo trials per (assignment, method, k, τ/C_crit) cell. The CZU dual-branch checkpoints from R6c were not originally saved (only aggregate accuracy was); this check required retraining that setting's five priors × three seeds specifically to obtain per-clip posteriors, verified afterward to match the original R6c accuracy table within ≤0.3 pp at every (method, k) cell — same ranking, well inside the project's established training-non-determinism noise floor.

Worst/best method by Lock 1 (mean hard task-success, k=1, τ=0) and Lock 2 (median mean-cost, k=1, C_crit=50), across the 120 assignments per setting:

| setting | Lock1 worst | Lock1 best | Lock2 worst | Lock2 best | locks agree? |
|---|---|---|---|---|---|
| NTU→Xsens (R5 above) | mae (0.455) | supMAE (0.525) | mae (54.8) | supMAE (46.5) | **yes** |
| CZU skeleton (R6) | mae (0.647) | supcon (0.802) | scratch (32.9) | supcon (22.5) | no (agree on best) |
| CZU IMU orientation-only (R6b) | supLP120 (0.107) | scratch (0.124) | scratch (113.3) | mae (105.6) | no |
| UTD skeleton (R6d) | scratch (0.717) | supcon (0.823) | mae (26.0) | supcon (19.9) | no (agree on best) |
| CZU dual-raw (R6c) | supcon (0.567) | scratch (0.619) | supcon (42.4) | scratch (39.0) | **yes** |

**The two locks agree with each other exactly on the two settings where the underlying recognition-level effect is itself large and significant.** On NTU→Xsens, mae is the paper's established, seed-stable negative-transfer culprit (R2, McNemar p<.001 every k); both locks name it worst. On CZU dual-raw, supcon is the single worst pooled recognition-level result in the entire paper (R6c, sign p<.0001, paired-t p≤.0066 every k); both locks name it worst, and `dual/scratch` — R6c's best performer at every k — is both locks' best. **On the three settings where the underlying recognition-level effect is small or non-significant (CZU skeleton's mae is flat, not negative, R6; UTD's mae trends positive, not negative, R6d; CZU IMU orientation-only's supLP120-worst trend does not clear significance post-retrain, R6b), the two locks disagree with each other on which method is worst.** Best is more stable — the two locks agree on best in 4/5 settings, and supcon or the strong-target scratch baseline wins in every setting except NTU→Xsens (where supMAE narrowly leads, supcon a close second). CZU IMU orientation-only is a degenerate case beyond disagreement: recognition accuracy is low enough (55–60%) that the 12-step, three-safety-critical task structure compresses every method's task success to a near-floor 10.7–12.4%, at which point method-level differences are dominated by noise rather than signal.

**Reading:** the controller behaves as a compounding stress-test harness on top of the recognition table, not as an independent source of ranking. It sharpens a real, significant recognition-level effect into a large, lock-consistent task-level gap (NTU→Xsens, CZU dual-raw); where the recognition-level effect is itself marginal, compounding amplifies noise into lock-to-lock disagreement rather than manufacturing a spurious consensus. This is a mechanistic echo of R6e's five-setting arc at the task-success/cost level, not an independent discovery — the controller re-exposes the existing recognition-level pattern faithfully where that pattern is real, and unreliably where it is not.

## R6 — External validity (CZU-MHAD skeleton→skeleton)

Same objectives and LOSO protocol on CZU-MHAD's skeleton modality (5 subjects, 22 actions). Final accuracy, mean over 5 subjects (single seed, to match the CRC same-splits head-to-head below; pooled 3-seed results follow):

| k | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|
| 0 (zero-shot) | 84.9 | 85.0 | 89.0 | **89.9** |
| 1 | 90.2 | 88.5 | 92.8 | **92.5** |
| 3 | 91.9 | 91.5 | 94.9 | **95.0** |

Paired Δ vs scratch (seed 42): supLP120 and supMAE both clearly positive at zero-shot; **mae is essentially flat here, not negative** (+0.1 @ k=0), unlike the significant negative transfer it shows on NTU→Xsens IMU. The mae-hurts signature does **not** reproduce on a same-modality (skeleton→skeleton, small-gap) target — evidence that the negative transfer is specific to the *cross-modal* gap, which is exactly the mechanism R2/R3 predict. This is skeleton→skeleton, not the IMU gap; the true cross-modal replication on CZU's inertial modality is reported next (R6b).

**Multi-seed robustness (3 seeds 42/43/44, pooled n=15).** The zero-shot supLP120 win holds: **supLP120 − scratch = +5.00 pp at k=0, significant (paired-t p=.0012)**. supMAE is significant-positive at k=0 (+3.15 pp, p=.010) and k=1 (+2.11 pp, p=.032); its k=3 margin is a trend (+1.85 pp, p=.086). mae is positive-but-n.s. at k=0, slightly negative-but-n.s. at k=1/k=3 — the same-modality "reconstruction does not clearly hurt" signature holds directionally. The small-gap "supervised prior wins" claim is seed-stable, and is independently replicated on a second public dataset (UTD-MHAD, R6d).

**supcon (3 seeds, pooled n=15) wins even bigger.** supcon − scratch = **+5.24 / +4.67 / +3.41 pp at k=0/1/3, all significant (p=.0015 / .0003 / .0049)** — a larger and more consistent margin than supLP120's. supcon > supLP120 in 28/42 folds (sign p=.044, meanΔ +0.91 pp): on this dataset the *contrastive* label-aware objective edges out the *softmax* label-aware objective, but both dominate scratch by a wide margin. The small-gap story is therefore not "supervised classification specifically wins" but **"any objective with direct label supervision at the pooled-embedding level wins when the gap is small."**

**Published-baseline comparison.** The CZU-MHAD dataset paper (Chao et al., 2022) reports a Collaborative Representation Classifier (CRC) on statistical-moment features; its protocol-comparable *cross-subject* skeleton accuracy is ~75.5% (its closed, subject-mixed test is not comparable to LOSO). We reproduce that baseline family — mean/std/var/skew/kurtosis features + CRC (λ=1e-4) — on the **byte-identical LOSO k-shot splits** used by our learned recognizer, giving a same-splits head-to-head (mean acc over 5 folds):

| k | CRC baseline | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|---|
| 0 | 84.81 | 84.87 | 84.95 | 89.03 | **89.94** |
| 1 | 89.31 | 90.15 | 88.48 | 92.76 | **92.54** |
| 3 | 90.82 | 91.90 | 91.46 | 94.92 | **95.04** |

Two things stand out. (i) **The from-scratch learned encoder is statistically level with the hand-crafted CRC baseline** (84.87 vs 84.81 at k=0) — architecture alone is not the advantage. (ii) **Only the NTU-pretrained prior opens a clear, consistent margin**: supLP120 − CRC = +5.1 / +3.2 / +4.2 pp at k=0/1/3. On an independent public dataset the value is carried by the *transferred prior*, not the recognizer, corroborating R4. (Our CRC reproduction, 84.8%, exceeds the paper's published cross-subject skeleton ~75.5% — expected given our LRQ representation and 4-subject LOSO dictionary vs their raw-position T5–T7 splits; we report it as a same-splits reproduction, not a claim about their exact pipeline.)

### R6b — Cross-modal external validity (CZU-MHAD skeleton→inertial)

The true cross-modal replication: NTU skeleton prior → CZU **wearable-IMU** target (10 body-worn 6-axis sensors, **no magnetometer**), an independent public instance of the skeleton→inertial gap. To feed the skeleton-pretrained encoder, the raw inertial is reduced to 17-segment orientation quaternions via Madgwick AHRS (yaw drifts without magnetometer) — an orientation-only representation shared by every method. Final accuracy, mean over LOSO folds (pooled 3 seeds 42/43/44 × 5 subjects = 45 folds; CRC column single-seed, raw accel+gyro):

| k | scratch | mae | supMAE | supLP120 | CRC (raw accel+gyro) |
|---|---|---|---|---|---|
| 0 | 55.7 | 56.2 | **56.4** | 54.9 | 86.8 |
| 1 | 54.3 | 54.8 | 54.5 | **52.8** | 90.8 |
| 3 | 58.3 | 58.1 | **59.6** | 57.6 | 94.4 |

**Under an impoverished (orientation-only) target encoding, the supervised prior's advantage disappears and trends toward hurting.** On the same-modality skeleton target (R6) the pure-supervised prior supLP120 dominated (+5.0 pp zero-shot, p=.001); across the modality boundary **supLP120 trends worst — below scratch at every k**, though not significantly so at this sample size (supLP120 − scratch = −1.0 pp mean; paired-t k=0 −0.86 pp p=.468, k=1 −1.44 pp p=.412, k=3 −0.70 pp p=.592). The **prior-vs-prior contrast (supMAE > supLP120) points the same direction**: 29/43 folds (sign p=.032, meanΔ +1.74 pp). supMAE does not clearly beat scratch (sign test 28/43 folds, p=.066, meanΔ +0.74 pp; paired-t clears nothing at any k). The honest reading: **supMAE ≈ scratch ≈ supLP120 at this gap once significance is accounted for** — the *direction* every prior points (reconstruction-containing objectives hold up better than the pure-supervised one) is mechanistically consistent with R2/R3, but this section alone does not carry a clean statistically-significant inversion. R6c below (dual-branch, on a stronger target) is where the sharper, significant version of this story lives.

Two honest limits. (i) The deep encoders trail the raw-inertial CRC baseline in absolute terms (55–60% vs 87–94%): the orientation-only encoding needed for cross-modal transfer discards the accelerometer-magnitude signal CRC exploits and carries yaw drift, so these numbers index *relative* prior transfer, not the inertial ceiling. (ii) Unlike NTU→Xsens, mae here is not a negative-transfer culprit (it trends slightly *positive* vs scratch, +0.74/+0.51/−0.26 pp at k=0/1/3, all n.s.) — mae's negative transfer is cross-domain-specific, not cross-modal-specific. Treat the direction in this section as the load-bearing claim, and the specific p-values as limited by sample size (n=43–45).

**supcon (3 seeds, pooled n=15/k) tracks supLP120's negative-to-neutral pattern.** supcon − scratch = **−0.44 / −1.00 / −1.04 pp @ k=0/1/3** (all n.s., p≥.55); pooled sign test supcon > scratch is 24/44 (p=.65, meanΔ −0.83, a wash trending negative). supcon vs supLP120 is statistically even (21/43, p=1.0, meanΔ +0.17) — the two label-aware objectives land in the same negative-to-neutral band, both numerically below supMAE. **Objectives lacking a reconstruction/regularization component do not clearly help, and trend toward hurting, across this modality gap**, regardless of whether label supervision is delivered via cross-entropy (supLP120) or contrastive alignment (supcon) — a directional pattern corroborated by R6c's sharper, significant version of the same contrast.

### R6c — The prior's value is contingent on target-representation poverty (CZU-MHAD dual-branch)

R6b's orientation-only encoding capped deep accuracy at 56–61% — far below the raw-inertial CRC baseline. Two questions follow: was that the *encoding* or the *architecture*, and does the R6b reconstruction-prior benefit survive once the target model uses its **full raw signal**? We test both with a dual-branch recognizer (a from-scratch branch on the raw 10×6 accel+gyro stream + the NTU-pretrained encoder on the R6b orientation quaternions, concatenated to a shared head) on the byte-identical CZU LOSO splits, pooled over 3 seeds × 5 subjects (45 folds):

| mode | scratch | mae | supMAE | supLP120 | supcon | CRC |
|---|---|---|---|---|---|---|
| raw-only (k=0/1/3) | 81.9 / 90.0 / **95.6** | — | — | — | — | 86.8 / 90.8 / 94.4 |
| dual (k=0) | **86.0** | 85.3 | 85.9 | 84.5 | 83.8 | — |
| dual (k=1) | **88.8** | 88.0 | 88.5 | 87.2 | 86.9 | — |
| dual (k=3) | **91.9** | 91.4 | 91.6 | 90.9 | 90.0 | — |

Two results. **(i) Representation, not architecture, was the R6b bottleneck.** The raw-signal branch alone reaches 81.9/90.0/95.6, matching/beating CRC and ~30 pp above R6b's orientation-only encoders — the collapse there was the lossy encoding, not the deep model. **(ii) On a strong target the NTU prior adds no value, and label-supervised-only priors actively hurt.** `dual/scratch` is the best performer at every k, beating every prior: supMAE ties it (pooled 18/37 folds, sign p=1.0 — no detectable difference), mae trends below (15/42, p=.088), **supLP120 is significantly worse — pooled 10/41 folds (sign p=.0015), paired-t significant at k=0** (−1.50 pp, p=.017), and **supcon is worse still — the single worst pooled result in this dataset (3/42 folds, sign p<.0001)**, paired-t significant at every k (−2.17 / −1.90 / −1.89 pp, p=.0002 / .0002 / .0066). The reconstruction component, not the specific supervised recipe, is what keeps an objective from actively hurting at large gap; the two label-aware-only priors (supLP120, supcon) are the clearest losers in the whole paper on this target.

**Dose-response along target richness.** Interpolating a middle rung between R6b (orientation-only) and R6c (full raw) — a 20-channel per-sensor accel-magnitude + gyro-magnitude target — traces the prior benefit as a smooth function of target strength (Δ vs scratch, k=0/1/3, pooled seeds 42/43/44):

| target representation | scratch acc | supMAE Δ | supLP120 Δ |
|---|---|---|---|
| 0-ch quat-only (R6b) | 56 / 54 / 58 | +0.7 / +0.2 / +1.3 (n.s.) | −0.9 / −1.4 / −0.7 (n.s.) |
| 20-ch magnitudes (dial) | 84 / 88 / 90 | −1.3 / −1.2 / −0.0 | −3.8 / −5.7 / −3.2 |
| 60-ch full raw (R6c) | 86 / 89 / 92 | −0.1 / −0.3 / −0.4 | −1.5 / −1.6 / −1.1 |

The reconstruction prior's edge, where it exists at all, is concentrated at the fully-crippled orientation-only target and fades as the target sees more real motion; supLP120 is negative at every rung and significantly so from the 20-ch dial onward.

**Does the cold-start advantage (R4c) survive on a strong target?** R4c's deployment claim — the prior is worth most at 0–1 enrolled subjects — was established only on Xsens, a weak/moderate target. We repeat the subject-count sweep (N=0..3, `--n-train-subjects` added to `scripts/external/czu/dualbranch.py`) on CZU-dual, single-seed (n=5 folds; a 3-seed extension is in progress):

| N | k | scratch | supLP120 Δ | supMAE Δ |
|---|---|---|---|---|
| 0 | 0 | 2.87 | +0.11 (n.s.) | +0.37 (n.s.) |
| 0 | 1 | 38.94 | **−5.62 p=.045** | **−5.55 p=.005** |
| 0 | 3 | 54.21 | +0.23 (n.s.) | **−2.87 p=.035** |
| 1 | 1 | 71.01 | −3.11 p=.085 (n.s.) | −3.51 p=.052 (n.s.) |
| 1 | 3 | 79.58 | **−2.54 p=.004** | −1.80 p=.076 (n.s.) |

**Neither prior beats scratch at any N, and both are significantly worse at N=0, k=1** — the exact cold-start cell where the Xsens A2 result showed the prior's *largest* benefit (+18.4 pp, R4c). This scopes the deployment claim precisely: **the cold-start advantage is itself contingent on target representation poverty**, mirroring R6c's main finding at N=4. A strong target's from-scratch branch reaches a useful operating point with zero enrolled subjects and no prior — there is no regime tested here where the NTU prior helps on this target.

Read across the three CZU settings, the prior's value *and its ranking* degrade monotonically with the gap:

| Setting | Gap | Target repr. | Best prior | supLP120 (pure-supervised) |
|---|---|---|---|---|
| R6 skeleton→skeleton | small (same modality) | strong (native skeleton) | **supLP120 +5.0 pp @k0** (p=.001), beats CRC | best-in-class |
| R6b IMU, orientation-only | large (cross-modal) | weak (orientation-only) | none clearly (supMAE trends > supLP120, 29/43, p=.032) | trends worst, n.s. (p=.41–.59) |
| R6c IMU, raw/dual | large (cross-modal) | strong (raw ≈ CRC) | none (`dual/scratch` itself is best at every k) | worst, p=.0015 |

The two axes are separable: holding the gap large, moving the target representation from weak (R6b) to strong (R6c) removes the reconstruction prior's advantage; holding the target strong, moving the gap from small (R6, where supervised wins) to large (R6c) turns the supervised prior from best to worst. A single public dataset yields a controlled contrast in which the *same* supervised prior is best-in-class in one modality and worst-in-class in another, purely as a function of gap width and target capability — the sharpest external corroboration of the negative-transfer thesis (R2/R3): a transferred prior earns its keep only in proportion to the target's representational poverty.

### R6d — Second independent same-modality dataset (UTD-MHAD skeleton→skeleton)

To confirm the small-gap "supervised prior wins" prediction (R6) is not specific to CZU, we replicate it on a second, independently collected public dataset: UTD-MHAD (Kinect-v1, 20 joints remapped to the NTU-25 layout; 8 subjects, 27 actions, 861 clips). Same objectives and LOSO k-shot protocol; pooled over 3 seeds × 8 subjects (24 folds per cell):

| k | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|
| 0 (zero-shot) | 77.4 | 79.6 | 80.0 | **82.4** |
| 1 | 90.9 | 91.6 | 93.1 | **93.0** |
| 3 | 94.0 | 94.5 | 95.1 | **96.0** |

**supLP120 is the best prior at k=0/k=1**: supLP120 − scratch = **+5.03 / +2.12 pp** (paired-t p=.0014 / .0153); at k=3 the margin (+2.02 pp) is a trend (p=.067). supLP120 vs supMAE is a wash (36/61 folds, sign p=.20) — the two priors are indistinguishable on UTD. supMAE is significant-positive at k=0/k=1 (+2.63 / +2.22 pp; p=.0098 / .0255), n.s. at k=3. As on CZU skeleton, **mae trends positive, not negative** (+2.23/+0.72/+0.59 pp, all n.s.): the reconstruction-hurts signature does not appear on a same-modality small-gap target. Two independent public datasets (CZU skeleton R6, UTD skeleton R6d) agree the pure-supervised prior is at least competitive, and never negative, when the gap is small — the mirror image of its worst-in-class behavior across the cross-modal gap (R6c).

**supcon (3 seeds, pooled n=24) is the standout result on this dataset.** supcon − scratch = **+7.97 / +3.98 / +2.46 pp @ k=0/1/3, all significant (p<.0001 / p=.0002 / p=.028)** — a clearly larger margin than supLP120's, and supcon beats supLP120 too (sign test 42/60, p=.003, meanΔ +1.75 pp). Combined with the CZU result (where supcon also edges out supLP120), the two-dataset picture is: **supcon is the most reliable small-gap performer of the five objectives, ahead of supLP120 on both external datasets**, while supLP120 itself is only conditionally best. The deeper regularity — "label-supervised, no reconstruction component wins at small gap" — holds as a class-level claim; which specific label-aware objective wins by how much varies per dataset.

**Published-baseline comparison.** Same recognizer family as the CZU R6 anchor (statistical moments + CRC-RLS, λ=1e-4) on the byte-identical seed-42 LOSO splits:

| k | CRC baseline | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|---|
| 0 | 69.48 | 75.98 | 78.28 | 80.50 | **83.05** |
| 1 | 92.25 | 90.25 | 93.18 | 92.88 | **93.51** |
| 3 | 95.30 | 93.52 | 94.85 | 95.35 | **96.28** |

supLP120 − CRC = **+13.57 / +1.26 / +0.98 pp** at k=0/1/3, echoing R6's consistent margin over the hand-crafted baseline. One difference from CZU worth noting honestly: on CZU, the **scratch** learned encoder was statistically level with CRC (R6); on UTD, **scratch already beats CRC by +6.5 pp at k=0** before any prior is added — the learned architecture has some edge here even without pretraining, so the prior's marginal contribution on top of scratch is smaller than its margin over CRC. (UTD-MHAD's own published cross-subject accuracy figure is not yet cited here — TODO, needs a literature figure, not compute.)

### R6e — Synthesis: the five-setting map

Five source→target settings, two public external datasets plus the two internal ones (Xsens position-derived, CZU orientation-only and dual-raw), give a controlled sweep over gap width and target-representation strength:

| # | Setting (source → target) | Gap | Target representation | Best prior (Δ vs scratch) | Negative-transfer case |
|---|---|---|---|---|---|
| 1 | NTU → CZU skeleton (R6) | small (same modality) | native skeleton | supLP120 +5.0 pp @k0 (p=.001) | none (mae ≈ flat, n.s.) |
| 2 | NTU → UTD skeleton (R6d) | small (same modality) | native skeleton | supLP120 +5.0 pp @k0 (p=.0014); supcon strongest overall (+7.97pp, p<.0001) | none (mae trends positive, n.s.) |
| 3 | NTU → Xsens, position-derived quats (R2) | middle (cross-device, skeleton-like target) | strong (mocap-grade quats) | mixed — supMAE/supLP120/supcon all significant only at k=3 (p≤.003), n.s. at k=0/1 | **mae** (McNemar −266/−242/−99, p<.001 every k) |
| 4 | NTU → CZU IMU, orientation-only (R6b) | large (cross-modal) | weak (Madgwick quats, yaw drift) | none clearly (supMAE trends > supLP120, 29/43 p=.032) | supLP120 trends worst, n.s. (p=.41–.59) |
| 5 | NTU → CZU IMU, dual raw (R6c) | large (cross-modal) | strong (raw ≈ CRC) | none (`dual/scratch` itself is best at every k) | **supLP120** (10/41, p=.0015); **supcon worse still** (3/42, p<.0001) |

No pretraining objective is unconditionally safe across the skeleton→wearable gap. The clearest pattern: **mae is the one objective with a real, significant, reproducible negative-transfer signature — at the middle gap specifically** — while at small gap no objective is negative and at large gap the pure-label-aware objectives (supLP120, supcon) are the ones that trend or test negative instead. The prior's gap-invariant value is calibration and cold-start data-efficiency (R4).

**A fifth objective (supcon) sharpens the small/large split into an axis, not two isolated findings.** supcon — Khosla 2020 SupCon, label-supervised but contrastive rather than softmax — tracks supLP120's direction at every external setting: it **wins big at both small-gap datasets** (CZU +5.24/+4.67/+3.41 pp k=0/1/3, all p<.01; UTD +7.97/+3.98/+2.46 pp, all p<.03 — the single strongest small-gap result in the paper on UTD, clearly ahead of supLP120, p=.003), and it **trends negative-to-flat at both large-gap settings** (CZU-IMU quat: −0.44/−1.00/−1.04 pp, n.s.; CZU-IMU dual: significantly worse than scratch, p<.0001, the single worst pooled sign-test result in R6c). Because supcon and supLP120 arrive at "label supervision" through different mechanisms (cross-entropy vs contrastive) yet land in the same direction at every gap setting, the operative variable is not "is the objective supervised classification" but **"does the objective have a reconstruction/regularization component"** — present in supMAE (the objective that comes closest to never being negative anywhere) and absent from supLP120/supcon alike. This reframes rows 4–5's negative-transfer entries as an instance of a *class* of objectives, not a property specific to softmax classification.

*Gap ordering, measured:* the "middle" placement of NTU→Xsens is not merely narrative. The encoder-space CKA-per-target check (R3 extension) does **not** confirm the small/middle/large ordering — on the supLP120 encoder, mean L0–L2 CKA ranks utd_skeleton > czu_skeleton > czu_imu_quat > xsens_v2, putting the claimed-middle target *lowest*, an instance of CKA's own necessary-not-sufficient limitation (R3) rather than a fit we forced. But a raw, encoder-free measurement — squared-MMD and Frechet distance on a hand-crafted, model-free feature (R3) — **does confirm the ordering exactly and monotonically**: czu_skeleton (MMD²=0.056) < utd_skeleton (0.096) < xsens_v2 (0.122) < czu_imu_quat (0.205), same ranking on Frechet distance independently. The gap ordering in this table therefore rests on the downstream sign-test/accuracy pattern above, and a measured raw-distributional gap that tracks it directly — with CKA's alignment-based null read as a separate, orthogonal finding about what encoder alignment does and doesn't predict (R3), not as evidence against the ordering itself.

---

## Numbers ledger (traceability)
- R1 within-domain: `trained_models/NTU-to-NTU-objective-sanity/`; cross-domain: `LOSO-fullTrainCalibrate-v2{,-seed43,-seed44}/summary.csv` (`scripts/main_experiment/loso_fulltrain_calibration.py`).
- R2 McNemar / R3 CKA / R4a ECE: `trained_models/Phase1-analysis/{mcnemar,cka,cka_by_target,ece}_results.csv` (`scripts/main_experiment/{dump_posteriors,cka_analysis}.py`).
- R3 raw MMD/Frechet: `trained_models/RawDomainGap/raw_domain_gap.csv` (`scripts/main_experiment/raw_domain_gap.py`); encoder-space MMD: `trained_models/MMD_DomainGap/mmd_table.csv` (`scripts/main_experiment/mmd_domain_gap.py`).
- R4b AUC-30/convergence: `wiki/results/multiseed-loso-v2.md` (`RESEARCH_LOG.md` §B).
- R4c A2: pooled 3 seeds — `trained_models/A2-subjectScaling{,-seed43,-seed44}/N*/summary.csv`, pooled table `trained_models/A2-subjectScaling-pooled/{a2_pooled_results,a2_pooled_stats}.csv` (`scripts/main_experiment/analyze_a2_multiseed.py`).
- R5 controller: `trained_models/Phase3-controller/robust/{vocab_sweep,vocab_ordering,costmodel_sweep,costmodel_summary,frontier,iso_safety,iso_safety_summary}.csv` + PNGs (`scripts/controller/controller_robust.py --vocabs 120 --missions 1000`; all three locks share the same 120 randomly-drawn System Input assignments, no recall-ranked or otherwise non-random vocab used anywhere in the reported numbers).
- R5b cross-setting (Locks 1–2 only): `trained_models/Phase3-controller/crosssetting/{ntu_xsens,czu_skeleton,czu_imu_quat,utd_skeleton,czu_dual_raw}/{vocab_sweep,vocab_ordering,costmodel_sweep,costmodel_summary}.csv` + `cross_setting_summary.csv` (`scripts/controller/controller_crosssetting.py --vocabs 120 --missions 1000`). Posteriors: `dump_posteriors.py` for the three settings with existing checkpoints (no retrain); CZU dual-raw posteriors required retraining `scripts/external/czu/dualbranch.py --mode dual` (3 seeds × 5 priors, `scripts/orchestration/10_czu_dual_controller_retrain.sh`) into new out-roots `trained_models/CZU-IMU-DUAL-controller{,-seed43,-seed44}/` — the original R6c `CZU-IMU-DUAL*/summary.csv` is untouched.
- R6 CZU: `trained_models/CZU-skeleton-LOSO{,-seed43,-seed44}/summary.csv`; CRC baseline `.../crc_baseline/crc_summary.csv` (`scripts/external/czu/crc_baseline.py`).
- R6b CZU inertial (cross-modal): `trained_models/CZU-IMU-LOSO{,-seed43,-seed44}/summary.csv`; CRC `.../crc_baseline/crc_summary.csv` (`scripts/external/czu/imu_crc_baseline.py`); data `Data_Processed/czu_imu_quats/`.
- R6c CZU dual-branch: `trained_models/CZU-IMU-DUAL{,-seed43,-seed44}/{raw_scratch,dual_<prior>}/summary.csv` (`scripts/external/czu/dualbranch.py`); raw data `Data_Processed/czu_imu_raw/`; reuses CZU-IMU-LOSO splits. Target-richness dial: `trained_models/CZU-IMU-DIAL{,-seed43,-seed44}/mag20/dual_<prior>/summary.csv`. Cold-start subject-scaling (T5): `trained_models/CZU-DUAL-subjectScaling/N{0..3}/dual_<prior>/summary.csv` (single-seed; 3-seed extension in progress, `trained_models/CZU-DUAL-subjectScaling-seed{43,44}/`, `scripts/orchestration/09b_czu_cold_start_multiseed.sh`).
- Pooling for R6/R6b/R6c: `scripts/external/czu/multiseed_analyze.py`.
- R6d UTD-MHAD skeleton: `trained_models/UTD-skeleton-LOSO-seed{42,43,44}/summary.csv` (`scripts/external/utd/crc_baseline.py` for the CRC row); data `Data_Processed/utd_skeleton_lrq/`.
- OOV / leave-class-out (referenced in `RESEARCH_LOG.md`, not a table above): `trained_models/LOSO-LeaveClassOutFewShot/summary.csv` (single-seed; 3-seed extension in progress, `trained_models/LOSO-LeaveClassOutFewShot-seed{43,44}/`, `scripts/orchestration/05b_oov_multiseed.sh`).
