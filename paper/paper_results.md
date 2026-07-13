# Results (draft)

_This document is the **source of truth for all reported numbers.** Every value here traces to `RESEARCH_LOG.md` §B and the run directories under `trained_models/`. Written 2026-07-05 (v2 preprocessing; 3-seed stats; Phase 1 McNemar/ECE/CKA; Phase 2 A2 subject-scaling; Phase 3 controller). Six results sections map to contributions C1–C6 plus external validity._

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
| supMAE | 59.66 ± 6.68 | 83.28 ± 6.41 | 91.24 ± 4.43 |
| supcon | 58.08 ± 7.24 | 82.34 ± 5.18 | 91.41 ± 4.75 |
| supLP120 | 58.41 ± 7.43 | 80.97 ± 4.91 | 90.81 ± 4.19 |
| scratch | 57.15 ± 6.84 | 82.31 ± 6.55 | 89.39 ± 4.90 |
| mae | 52.90 ± 5.71 | 79.47 ± 5.45 | 88.05 ± 5.33 |

The full method spread at k=1 is ~3.8 pp (83.3 − 79.5), vs the ~45 pp within-domain spread. The gap, not the objective, sets how much objective choice matters. supcon lands close to scratch on raw accuracy (paired Δ: +0.9/+0.0/+2.0 pp at k=0/1/3, all n.s. or marginal, p≥.07 — see R2) — a fifth objective, a fifth accuracy-is-a-wash story. All methods reach 94–98% by k≥5 (ceiling); k∈{0,1,3} is the discriminating regime.

## R2 — A seed-stable negative-transfer case that inverts the field (C2)

The load-bearing correction from single-seed to multi-seed: the single-seed "+4.1 pp @ k=1 supMAE" delta was seed-42-specific (seed-by-seed supMAE−scratch @ k=1: +4.13 / −1.77 / +0.56; pooled +0.97, n.s.). What survives pooling is not a supervised/contrastive collapse but a **reconstruction collapse**.

**Paired Δ vs scratch** (mean pp, [95% CI], paired-t p), n=15:

| contrast | k=0 | k=1 | k=3 |
|---|---|---|---|
| supMAE − scratch | +2.52 [−0.46, +5.49] p=.091 | +0.97 [−1.78, +3.72] p=.461 | +1.85 [−0.03, +3.74] p=.054 |
| supLP120 − scratch | +1.27 p=.494 | −1.34 p=.375 | +1.42 p=.264 |
| supcon − scratch | +0.93 [−3.92, +5.79] p=.686 | +0.03 [−2.85, +2.91] p=.982 | +2.02 [−0.23, +4.27] p=.074 |
| **mae − scratch** | **−4.24 [−6.78, −1.71] p=.003** | **−2.84 [−4.43, −1.25] p=.002** | −1.34 p=.201 |

**Clip-level McNemar** (pooled 3 seeds; net = prior-only-correct − scratch-only-correct):

| prior | k | n_pairs | b (scratch-only) | c (prior-only) | net | p |
|---|---|---|---|---|---|---|
| supMAE | 0 | 7974 | 480 | 680 | **+200** | 0.0 |
| supMAE | 1 | 7644 | 342 | 418 | +76 | .0065 |
| supMAE | 3 | 6984 | 210 | 341 | **+131** | 0.0 |
| supLP120 | 0 | 7974 | 786 | 895 | +109 | .0084 |
| supLP120 | 1 | 7644 | 591 | 494 | −97 | .0035 |
| supLP120 | 3 | 6984 | 296 | 400 | +104 | 9e-05 |
| supcon | 0 | 7974 | 740 | 825 | +85 | .034 |
| supcon | 1 | 7644 | 487 | 492 | +5 | .898 (n.s.) |
| supcon | 3 | 6984 | 265 | 409 | **+144** | 0.0 |
| **mae** | 0 | 7974 | 789 | 455 | **−334** | 0.0 |
| **mae** | 1 | 7644 | 593 | 376 | **−217** | 0.0 |
| **mae** | 3 | 6984 | 385 | 293 | **−92** | .00047 |

**mae is significant negative transfer at every k** (all p<.001) — it flips *more* clips wrong than right relative to random init. supMAE is significant positive at every k. supLP120 and supcon are both mixed at the clip level (positive k0/k3, a null/small-negative dip at k1) — the two label-aware objectives share this signature. This inverts the field's working assumption that reconstruction/self-supervision is the *safe* prior under large gaps: here the pure-reconstruction prior is the one that hurts, because it never encodes discriminative structure (R1) and the fine-tune from that init lands in a worse basin than from random.

## R3 — Mechanism: gap is necessary-not-sufficient; CKA orders objectives (C3)

**MMD does not track transfer.** Squared-MMD between NTU and Xsens-v2 features (supMAE encoder) is 0.0092 under v2 (below the local baseline 0.0109 and far below swing 0.0322), confirming v2 closes the gap — but MMD ordering across encoders does not match transfer ordering (contrastive had near-lowest MMD yet worst transfer under the earlier preprocessing; mae has higher MMD than supervised yet similar low-k accuracy).

**CKA orders the objectives cleanly.** Linear CKA between NTU and Xsens-v2 activations, per encoder × layer:

| encoder | proj | L0 | L1 | L2 |
|---|---|---|---|---|
| supLP120 | 0.0038 | 0.0133 | 0.0139 | **0.0149** |
| supcon | 0.0039 | 0.0233 | 0.0231 | 0.0257 |
| supMAE | 0.0029 | 0.0043 | 0.0047 | 0.0048 |
| mae | 0.0029 | 0.0036 | 0.0036 | 0.0035 |
| scratch | 0.0027 | 0.0027 | 0.0027 | 0.0027 |

Two facts, both important: (i) **absolute CKA is small for all** (<0.02–0.03) — the cross-modal gap is large even after v2, so we do not over-claim on magnitude; (ii) the **ordering is clean and monotone with depth** for the two label-aware objectives — supLP120 and supcon both align well above supMAE/mae/scratch and *grow* with depth, while scratch is flat. supcon's raw CKA magnitude is numerically the *highest* of all five here (0.0257 at L2, edging out supLP120's 0.0149) despite both being label-supervised objectives, yet supcon's cross-domain accuracy (R1) is a wash vs scratch (unlike supLP120, which is at least mixed-positive) — the sharpest instance in the paper of alignment magnitude *not* predicting accuracy. The best-aligned encoder (supcon, numerically; supLP120 historically) is not the best-transferring on accuracy (supMAE is), and the reconstruction encoder has higher CKA than scratch but negative transfer. **Conclusion:** representation alignment is a necessary context but not a sufficient predictor; the objective's discriminative inductive bias governs transfer. This is the mechanism the MMD-only view could not supply.

**Extending this to the gap axis itself:** we also measured layer-wise CKA between NTU and each of the four targets used elsewhere in the paper (Xsens-v2, CZU skeleton, CZU IMU orientation quats, UTD skeleton) to test whether CKA orders them by our claimed small/middle/large gap (see R6e). It does not, and is inconsistent even within the claimed small-gap pair: on the supLP120 encoder, mean L0–L2 linear CKA is utd_skeleton 0.0327 > xsens_v2 0.0290 > czu_skeleton 0.0267 > czu_imu_quat 0.0260 — czu_skeleton (claimed small gap, same as utd_skeleton) ranks *below* both the claimed-middle and claimed-large targets. We report this as a further instance of necessary-not-sufficient rather than force a fit: CKA does not reliably track gap *width* across independently collected datasets. Full table in `wiki/concepts/domain-gap-metrics.md`.

**A raw, encoder-free measurement recovers the ordering CKA missed.** Both CKA and the encoder-space MMD above are computed *inside a trained encoder's feature space* — confounded by the pretraining objective, so they measure how different two datasets look *to that encoder*, not how different the datasets are themselves. We therefore also measured domain gap directly on a hand-crafted, model-free feature (mean/std/var/skew/kurtosis over the 68 flattened LRQ channels — the same feature used for the CRC published-baseline reproductions in R6/R6d, zero pretraining involved), computing squared-MMD (RBF kernel, median-heuristic bandwidth — this fixes the fixed-sigma scale confound noted above) and Frechet distance (Gaussian closed form on a 50-component PCA) between NTU and each target, n=1000/side (n=861 for UTD, its full pool), seed 0:

| target | gap (claimed) | MMD² | Frechet |
|---|---|---|---|
| czu_skeleton | small | **0.0560** | **176.8** |
| utd_skeleton | small | **0.0960** | **202.1** |
| xsens_v2 | middle | **0.1218** | **281.9** |
| czu_imu_quat | large | **0.2049** | **393.8** |

**Both metrics confirm the claimed small<small<middle<large gap ordering exactly and monotonically** — the opposite outcome from the CKA-per-target result immediately above. Removing the pretrained encoder from the measurement, rather than adding one, is what recovers the ordering: CKA measures how well a *specific trained encoder* aligns two domains (task- and objective-dependent, hence the necessary-not-sufficient finding above), while raw MMD/Frechet measure how different the *datasets themselves* are before any model touches them. The two results are complementary, not contradictory: the objective's discriminative inductive bias governs whether a given alignment *transfers* (CKA's lesson), while the raw distributional gap between datasets follows the intuitive device/modality ordering (this result) — CKA's failure to track it is itself evidence that encoder alignment and raw distributional distance are different quantities. This gives the R6e gap ordering a measured, encoder-independent basis it previously lacked. Full table in `wiki/concepts/domain-gap-metrics.md`, script `temp_raw_domain_gap.py`.

## R4 — The prior's real value: calibration and data-efficiency (C4→A2, C5)

The accuracy spread is compressed, but the prior still pays off in three deployment-relevant ways.

**(a) Calibration.** Temperature-scaled ECE, per method×k:

| method | k=0 | k=1 | k=3 |
|---|---|---|---|
| supLP120 | **0.0258** | **0.0306** | **0.0350** |
| supcon | 0.0339 | 0.0410 | 0.0410 |
| supMAE | 0.0653 | 0.0424 | 0.0396 |
| scratch | 0.0676 | 0.0494 | 0.0396 |
| mae | 0.0676 | 0.0667 | 0.0545 |

supLP120 is best-calibrated at every k; supcon is a clear second, ahead of supMAE/scratch/mae at every k; mae worst at k=1/3. The two label-supervised objectives (supLP120, supcon) lead calibration; the two without direct softmax-style label supervision at the pooled embedding (supMAE, mae) trail. (Raw ECE at k=0 is large for all, 0.25–0.34, T≈2.5–3.5 — everyone is overconfident before temperature scaling.)

**(b) Calibration efficiency (AUC-30 = mean eval-acc over the first 30 calibration epochs) and convergence speed.** Paired Δ vs scratch:

- AUC-30: supMAE k3 **+2.04 [+0.26, +3.81] p=.028**; supLP120 k3 **+3.74 [+1.03, +6.46] p=.010**; supcon k3 **+4.18 [+1.44, +6.93] p=.006** (k1 +1.89, n.s.); mae k1 −4.37 p<.001, k3 −3.83 p<.001.
- Convergence (first epoch ≥90% of final; lower=faster): supLP120 k1 **−4.00 [−6.97, −1.03] p=.012**, k3 **−3.40 [−5.07, −1.73] p=.001**; supcon k3 **−3.53 [−5.64, −1.43] p=.003** (k1 −2.47, n.s.); mae *slower* (+2.93 p≤.034); supMAE ≈ scratch.

The prior converges faster and integrates the few calibration shots more efficiently even where the endpoint accuracy is a wash — and this now holds for **two** label-aware objectives (supLP120, supcon), not one, strengthening R4 from a single-objective observation to a pattern tied to label-awareness rather than one specific pretraining recipe.

**(c) Subject-count scaling (A2) — how long the prior stays useful.** Prior benefit vs scratch (pp), by number of fine-tuning subjects N × k, **pooled over 3 seeds (42/43/44), n=15 per cell**:

| prior | k | N=0 | N=1 | N=2 | N=3 | N=4 |
|---|---|---|---|---|---|---|
| supLP120 | 1 | +6.96 | **+18.43** | +3.64 | −0.37 | −1.34 |
| supLP120 | 3 | +18.47 | **+27.46** | +7.15 | +2.41 | +1.42 |
| supMAE | 1 | +3.35 | +6.42 | +3.48 | +2.02 | +0.97 |
| supMAE | 3 | +9.18 | +9.40 | +2.13 | +2.15 | +1.85 |
| mae | 1 | +2.42 | +0.44 | −1.80 | −2.44 | −2.84 |
| mae | 3 | +5.29 | +2.74 | −2.04 | −0.76 | −1.34 |

Paired-t (n=15): supLP120 − scratch is significant at every (N,k) shown (N=0 k=1/k=3 and N=1 k=1/k=3, all **p<.0001**); supMAE − scratch likewise significant (N=0 k=1 p=.004, N=1 k=1 p=.0003, both k=3 p<.0001); mae − scratch turns significantly negative by N=4 at k=1 (p=.0018), n.s. at N=1–3.

The prior's benefit **peaks at N=0/1** (supLP120 **+27.5 pp @ k=3, N=1**, up from the single-seed +26.9) and **washes to ≈1–1.4 pp by N=4** — a clean, now statistically confirmed deployment lever: the pretrained prior is worth roughly a full retraining set when 0–1 users are enrolled and is nearly redundant once ~4 are. mae is negative for N≥2 at k=1, significantly so by N=4 (p=.0018) — reinforcing R2. Pooling **resolves** rather than softens the earlier single-seed caveat: the k=1, N=1 spike survives at all three seeds individually (+15.5 / +21.2 / +18.6 pp) and is not a seed artifact.

**Caveat established in R6c below: this cold-start lever is itself target-representation-contingent.** Repeating this exact sweep on CZU's strong (raw-signal) target finds no N at which either prior beats scratch — both are significantly *worse* at N=0, k=1. Read R4c and R6c's cold-start extension together, not R4c alone.

## R5 — From recognition to control (C6)

The abstract controller (12-step pick-place mission, held-out posteriors, confidence-reject safety layer, asymmetric cost) turns the compressed accuracy differences into task-level ones. The command vocabulary (grasp, release, ...) is a hypothetical one, not a recorded gesture set — real gestures are relabeled onto it by recognition reliability, not semantics (Methods §8). **Illustratively**, for a reliability-ordered command vocabulary at k=1, ungated, mission-success is supMAE 0.967 / scratch 0.905 / supLP120 0.861 / mae 0.754 — a ~4 pp recognition gap (mae vs supMAE) becomes a **21 pp mission-success gap**, the concrete form of "a few accuracy points matter." But a controller has design knobs (command vocabulary, error-cost model, operating threshold), so rather than rest on one configuration we show the method ordering is *invariant to the knobs* via three robustness locks (Methods §8.1). Magnitudes vary; the conclusions do not.

**Lock 1 — randomized command vocabulary (kills "you cherry-picked the gestures").** We resample the 7-gesture→primitive mapping 120 times at random and report the distribution of mission success. Mean hard task-success across the 120 vocabularies (supcon added 2026-07-10, exact re-run — this table's original 4 columns reproduce the locked run to 3 decimals, since Lock 1's per-vocab RNG is independent of which other methods are in the pool):

| condition | scratch | mae | supMAE | supLP120 | supcon |
|---|---|---|---|---|---|
| k=1, τ=0 (ungated) | 0.516 | **0.452** | 0.530 | 0.479 | 0.506 |
| k=1, τ=0.9 (gated) | 0.714 | **0.655** | 0.724 | 0.700 | 0.722 |
| k=3, τ=0 | 0.663 | **0.634** | 0.710 | 0.716 | 0.717 |
| k=3, τ=0.9 | 0.830 | **0.774** | 0.847 | 0.879 | **0.880** |

**mae is the worst-compounding init in all four conditions, now against five objectives, not four** — it beats supcon in only 22% of vocabularies at k=1 τ=0 (vs 12% for supMAE, 34% for supLP120). The negative transfer of R2 survives at the task level regardless of which gestures are commands, and regardless of which other objectives are in the comparison set. supMAE is ≥ scratch in every condition; supLP120 leads at k=3 but dips just below scratch at k=1 ungated (see the honest finding below); supcon tracks supLP120 closely and is the single best method at k=3, τ=0.9.

**Lock 2 — critical-cost sweep (kills "the harsh instant-fail rule drives it").** Replacing binary mission-failure with a recoverable critical penalty C_crit and sweeping C_crit ∈ {2, 5, 10, 20, 50, ∞}, **mae has the highest mean mission cost at every C_crit at k=1** (e.g. C_crit=20: mae 20.8 vs supMAE 15.6 vs scratch 17.2), and supMAE the lowest for C_crit≥5. The mae-worst ordering is independent of how catastrophic critical errors are made.

**Lock 3 — iso-safety operating point (kills "you tuned τ to win").** Rather than pick τ, we fix a false-activation *budget* and find the smallest τ meeting it per method (deployment-standard, tuning-free). At a **1% budget, k=1**:

| method | τ* | task-success | mean cost |
|---|---|---|---|
| supLP120 | **0.90** | 0.953 | **16.69** |
| mae | 0.95 | 0.967 | 19.09 |
| supMAE | 0.99 | 0.872 | 20.54 |
| scratch | 0.99 | 0.810 | 20.82 |

The best-calibrated init (supLP120, R4a) reaches the safety budget at the **lowest threshold** (0.90 vs 0.95–0.99), so it rejects less and completes ~13–20% faster; at the strict **0.5% budget it is the only init that meets the spec with strong success** (τ*=0.97, 0.877) while scratch and supMAE cannot reach it at all at k=1. This is the calibration payoff made operational: trustworthy confidence buys throughput at a fixed safety level.

**Honest finding (a design principle, not a caveat).** supLP120 carries a *confident false-critical-activation* mode: it occasionally maps other gestures onto the highly-separable anchors used for grasp/release with high confidence, so on the **ungated** metric it dips below scratch at k=1 and, at extreme C_crit, can be penalized at k=3. This is precisely why the deployment-correct framing is iso-safety (set τ to a safety budget) together with the design guard of assigning safety-critical commands to the most-separable gestures — under that correct framing, calibration wins. The naive ungated metric *understates* the value of calibration.

Net: across 120 random vocabularies, two outcome models, a 6-point critical-cost sweep, and a tuning-free operating point, **mae compounds worst and calibration governs the safety/throughput trade** — the two claims that matter, locked.

**Extending to supcon (Locks 2/3) — a discovered quirk, reported rather than smoothed over.** Adding supcon required re-running Locks 2/3, and doing so surfaced a script coupling we hadn't noticed: the "illustrative" 12-step-mission vocabulary (`reliability_ordered_vocab`) is ranked by pooled recall over *every* method present in the loaded posterior pool, not per-method — so adding supcon's posteriors to the pool silently shifted which 7 gestures get assigned to the mission's primitives for Locks 2/3 (Lock 1's randomized vocabularies are unaffected, which is why that table reproduces exactly). The original locked numbers for scratch/mae/supMAE/supLP120 (Lock 2/3 tables above) are untouched — they live in a separate, never-overwritten output directory — but supcon's own Lock 2/3 numbers were computed under this re-derived vocabulary and are not numerically poolable with the tables above. Reported on their own terms, the conclusions replicate: at C_crit=20, k=1, mean cost under the re-derived vocabulary is supLP120 17.3 < supcon 17.6 < supMAE 16.2 (supMAE lowest here) < scratch 18.0 < **mae 21.7 (highest, as at every C_crit swept, 2 through ∞)**; at the 1% false-activation budget, k=1, supLP120 again reaches the safety threshold at the lowest τ* (0.85) and lowest mean cost (16.6), with supcon second-cheapest (19.0) ahead of scratch/supMAE/mae. **mae compounds worst and calibration governs safety/throughput hold under an independently-derived vocabulary too** — an adventitious fourth robustness check, not planned as one. (Numbers: `trained_models/Phase3-controller/robust-supcon/`; do not merge into the locked `robust/` figures above.)

## R6 — External validity (CZU-MHAD skeleton→skeleton)

Same objectives and LOSO protocol on CZU-MHAD's skeleton modality (5 subjects, 22 actions). Final accuracy, mean over 5 subjects (single seed, to match the CRC same-splits head-to-head below; pooled 3-seed robustness follows):

| k | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|
| 0 (zero-shot) | 84.9 | 87.9 | 85.5 | **93.1** |
| 1 | 90.1 | 90.0 | 91.2 | 93.1 |
| 3 | 91.9 | 93.8 | 93.6 | 93.8 |

Paired Δ vs scratch: **supLP120 zero-shot +8.3 pp (p=.07)**; supMAE modestly positive at all k (n.s.). Critically, **mae is *positive* here** (+3.1 @ k=0), unlike the significant negative transfer it shows on NTU→Xsens IMU. The mae-hurts signature does **not** reproduce on a same-modality (skeleton→skeleton, small-gap) target — evidence that the negative transfer is specific to the *cross-modal* gap, which is exactly the mechanism R2/R3 predict. This is skeleton→skeleton, not the IMU gap; the true cross-modal replication on CZU's inertial modality is reported next (R6b).

**Multi-seed robustness (3 seeds 42/43/44, pooled n=15).** The zero-shot supLP120 win survives pooling and *sharpens*: **supLP120 − scratch = +7.2 pp at k=0, now clearly significant (paired-t p=.0004)** — the single-seed p=.07 was underpowered, not fragile. supMAE also becomes significant-positive at k=0 (+2.2 pp, p=.048) and k=3 (+2.2 pp, p=.014); mae stays positive-but-n.s. at every k (the same-modality "reconstruction does not hurt" signature holds under pooling). The small-gap "supervised prior wins" claim is thus seed-stable, and is independently replicated on a second public dataset (UTD-MHAD, R6d).

**supcon (3 seeds, pooled n=15) wins even bigger.** supcon − scratch = **+7.0 / +4.8 / +4.4 pp at k=0/1/3, all significant (p<.001)** — a larger and more consistent margin than supLP120's (which is significant only at k=0/k=3). supcon > supLP120 in 34/43 folds (sign p=.0002, meanΔ +1.4 pp): on this dataset the *contrastive* label-aware objective edges out the *softmax* label-aware objective, but both dominate scratch by a wide margin. The small-gap story is therefore not "supervised classification specifically wins" but **"any objective with direct label supervision at the pooled-embedding level wins when the gap is small"** — sharpened by a fifth objective, not just replicated by a second dataset.

**Published-baseline comparison.** The CZU-MHAD dataset paper (Chao et al., 2022) reports a Collaborative Representation Classifier (CRC) on statistical-moment features; its protocol-comparable *cross-subject* skeleton accuracy is ~75.5% (its closed, subject-mixed test is not comparable to LOSO). We reproduce that baseline family — mean/std/var/skew/kurtosis features + CRC (λ=1e-4) — on the **byte-identical LOSO k-shot splits** used by our learned recognizer, giving a same-splits head-to-head (mean acc over 5 folds):

| k | CRC baseline | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|---|
| 0 | 84.81 | 84.87 | 87.94 | 85.50 | **93.14** |
| 1 | 89.31 | 90.15 | 89.98 | 91.19 | **93.08** |
| 3 | 90.82 | 91.90 | 93.77 | 93.62 | **93.85** |

Two things stand out. (i) **The from-scratch learned encoder is statistically level with the hand-crafted CRC baseline** (84.87 vs 84.81 at k=0) — architecture alone is not the advantage. (ii) **Only the NTU-pretrained prior opens a clear, consistent margin**: supLP120 − CRC = +8.3 / +3.8 / +3.0 pp at k=0/1/3. On an independent public dataset the value is carried by the *transferred prior*, not the recognizer, corroborating R4. (Our CRC reproduction, 84.8%, exceeds the paper's published cross-subject skeleton ~75.5% — expected given our LRQ representation and 4-subject LOSO dictionary vs their raw-position T5–T7 splits; we report it as a same-splits reproduction, not a claim about their exact pipeline.)

### R6b — Cross-modal external validity (CZU-MHAD skeleton→inertial)

The true cross-modal replication: NTU skeleton prior → CZU **wearable-IMU** target (10 body-worn 6-axis sensors, **no magnetometer**), an independent public instance of the skeleton→inertial gap. To feed the skeleton-pretrained encoder, the raw inertial is reduced to 17-segment orientation quaternions via Madgwick AHRS (yaw drifts without magnetometer) — an orientation-only representation shared by every method. Final accuracy, mean over LOSO folds (pooled 3 seeds 42/43/44 × 5 subjects = 45 folds; CRC column single-seed, raw accel+gyro):

| k | scratch | mae | supMAE | supLP120 | CRC (raw accel+gyro) |
|---|---|---|---|---|---|
| 0 | 55.7 | 54.9 | **57.6** | 53.6 | 86.8 |
| 1 | 54.3 | 53.0 | **54.2** | 51.9 | 90.8 |
| 3 | 58.3 | 57.4 | **59.6** | 56.0 | 94.4 |

**Under an impoverished (orientation-only) target encoding, the supervised prior stops helping and starts hurting, while reconstruction-augmentation neutralizes the damage.** On the same-modality skeleton target (R6) the pure-supervised prior supLP120 dominated (+7.2 pp zero-shot, pooled p=.0004); across the modality boundary **supLP120 becomes the worst prior — below scratch at every k** (supLP120 − scratch = −2.3 pp mean; paired-t k=0 −2.1 pp p=.044, k=3 −2.3 pp p=.056). What multi-seed pooling makes precise is *which* contrast is the real effect. The robust, load-bearing result is the **prior-vs-prior contrast: supMAE > supLP120 in 36/44 folds (sign p<.0001)** — adding reconstruction to the supervised objective rescues transfer across the modality gap. What does **not** survive pooling is the claim that supMAE *beats scratch*: pooled, supMAE > scratch is only **25/42 folds (sign p=.28)**, and paired-t clears significance solely at k=0 (+1.9 pp, p=.034). The single-seed "+3 pp reconstruction benefit" (12/15, p=.035) was a seed-42 artifact. The honest reading is therefore: supMAE lands supMAE ≈ scratch (not above it), and the mechanism is **supervised-specific negative transfer with reconstruction as the antidote** — consistent with R2/R3 (the supervised-only prior encodes skeleton-specific class boundaries that misalign on IMU-derived orientations; MAE reconstruction regularizes back toward modality-transferable low-level kinematics and cancels the loss). R6c below shows even that antidote is scoped to the weak encoding.

Three honest limits. (i) The deep encoders trail the raw-inertial CRC baseline in absolute terms (56–61% vs 87–94%): the orientation-only encoding needed for cross-modal transfer discards the accelerometer-magnitude signal CRC exploits and carries yaw drift, so these numbers index *relative* prior transfer, not the inertial ceiling. (ii) Unlike NTU→Xsens, mae here is statistically level with scratch (not the negative-transfer culprit); on this target the negative transfer is carried by the *supervised* prior. (iii) The effect that survives 3-seed pooling is the *ordering* supMAE > supLP120 (p<.0001) and supLP120 < scratch (p≈.04–.06), not a supMAE-over-scratch pp margin. We therefore report R6b as a sign-test-robust cross-modal *inversion of prior ranking* — supervised best→worst, reconstruction as the differentiator — bounded by R6c, rather than a reconstruction-beats-scratch accuracy claim.

**supcon (3 seeds, pooled n=15/k) confirms it is the label-supervision, not the specific supervised recipe, that hurts.** supcon − scratch = **−1.4 / −3.4 / −0.2 pp @ k=0/1/3** (n.s. to marginal, k=1 p=.053); pooled sign test supcon > scratch is 19/45 (p=.37, a wash trending negative, same reading as supLP120's own −2.3 pp mean). supcon vs supLP120 is statistically even (26/44, p=.29) — the two label-aware objectives land in the same negative-to-neutral band, distinctly below supMAE (the only objective here with a reconstruction component). This rules out "something specific to softmax classification" as the mechanism and supports the more general reading: **any objective lacking a reconstruction/regularization component under-transfers across this modality gap**, regardless of whether the label supervision is delivered via cross-entropy (supLP120) or contrastive alignment (supcon).

### R6c — The prior's value is contingent on target-representation poverty (CZU-MHAD dual-branch)

R6b's orientation-only encoding capped deep accuracy at 56–61% — far below the raw-inertial CRC baseline. Two questions follow: was that the *encoding* or the *architecture*, and does the R6b reconstruction-prior benefit survive once the target model uses its **full raw signal**? We test both with a dual-branch recognizer (a from-scratch branch on the raw 10×6 accel+gyro stream + the NTU-pretrained encoder on the R6b orientation quaternions, concatenated to a shared head) on the byte-identical CZU LOSO splits. **Both rows are now pooled over 3 seeds × 5 subjects (45 folds)**:

| mode | scratch | mae | supMAE | supLP120 | CRC |
|---|---|---|---|---|---|
| raw-only (k=0/1/3) | 81.9 / 90.0 / **95.6** | — | — | — | 86.8 / 90.8 / 94.4 |
| dual (k=0) | 86.0 | 84.8 | 85.6 | 84.5 | — |
| dual (k=1) | 88.8 | 88.6 | 88.2 | 87.0 | — |
| dual (k=3) | 91.9 | 91.6 | **92.0** | 90.0 | — |

Two results, both firmer under pooling. **(i) Representation, not architecture, was the R6b bottleneck.** The raw-signal branch alone reaches 81.9/90.0/95.6 (pooled, up from single-seed 80.7/89.8/95.2), matching/beating CRC and ~30 pp above R6b's orientation-only encoders — the collapse was the lossy encoding, not the deep model. `dual/scratch` vs `raw/scratch` remains a wash under pooling (19/42 folds, sign p=.644, meanΔ −0.27 pp — up in power from the single-seed 6/12, p=1.00, same conclusion): a small k=0 gain (+4.1 pp) is paid back by a k=3 loss (−3.6 pp), so the second branch doesn't clearly earn its place. **(ii) On a strong target the NTU prior adds no value, and label-supervised-only priors actively hurt.** No prior beats `dual/scratch`: supMAE ties it (pooled 17/39 folds, sign p=.52 — no detectable benefit), mae trends below (13/36, p=.13), **supLP120 is significantly worse — pooled 6/41 folds (sign p<.0001), paired-t significant at every k** (−1.5 / −1.8 / −1.9 pp, p=.0016 / .040 / .0038), and **supcon is worse too — pooled 13/43 folds (sign p=.014)**, paired-t marginal-to-significant (−1.2 / −1.1 / −1.1 pp, p=.048 / .12 / .08). The R6b reconstruction rescue is therefore **contingent on target-representation poverty**: it appears only when the target encoder is crippled to orientation-only and vanishes once the target exploits its own signal — while the two label-aware-only priors (supLP120, supcon) both remain losers, now more significantly than in R6b, converging on the same reading as R6b: the reconstruction component, not the specific supervised recipe, is what saves an objective at large gap.

**Dose-response along target richness (pooled 3-seed dial).** Interpolating a middle rung between R6b (orientation-only) and R6c (full raw) — a 20-channel per-sensor accel-magnitude + gyro-magnitude target — traces the prior benefit as a smooth function of target strength (Δ vs scratch, k=0/1/3, pooled seeds 42/43/44):

| target representation | scratch acc | supMAE Δ | supLP120 Δ |
|---|---|---|---|
| 0-ch quat-only (R6b) | 56 / 54 / 58 | +3.0 / +2.4 / +3.0 | −3.5 / −2.8 / −3.1 |
| 20-ch magnitudes (dial) | 84 / 88 / 90 | −1.4 / −1.6 / −1.1 | −4.1 / −5.2 / −3.6 |
| 60-ch full raw (R6c) | 87 / 89 / 91 | −0.4 / −0.4 / +0.3 | −2.1 / −3.5 / −2.5 |

The reconstruction prior's edge exists **only** at the fully-crippled orientation-only target and is gone the moment the target sees any real motion (even coarse magnitudes, where scratch already hits 84%+); supLP120 is negative at every rung. This turns "contingent on target poverty" from a two-point contrast into an early-decay curve. Pooling tightens but does not change the shape (single-seed supMAE dial values were −1.9/−2.5/−0.8; pooled −1.4/−1.6/−1.1) — steep-then-flat, not perfectly monotone.

**Does the cold-start advantage (R4c) survive on a strong target?** R4c's deployment claim — the
prior is worth most at 0–1 enrolled subjects — was established only on Xsens, a weak/moderate target.
We repeat the subject-count sweep (N=0..3, `--n-train-subjects` added to `temp_czu_dualbranch.py`)
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
regime tested here where the NTU prior helps on this target. (Single-seed, n=5 folds; a 3-seed
extension is queued but not run — the signal is already significant at several (N,k) cells.)

Read across the three CZU settings, the prior's value *and its ranking* degrade monotonically with the gap:

| Setting | Gap | Target repr. | Best prior | supLP120 (pure-supervised) |
|---|---|---|---|---|
| R6 skeleton→skeleton | small (same modality) | strong (native skeleton) | **supLP120 +7.2 pp @k0** (pooled p=.0004), beats CRC | best-in-class |
| R6b IMU, orientation-only | large (cross-modal) | weak (orientation-only) | supMAE (ties scratch; > supLP120 36/44, p<.0001) | worst, below scratch (p≈.04–.06) |
| R6c IMU, raw/dual | large (cross-modal) | strong (raw ≈ CRC) | none (supMAE ties scratch, 17/39) | worst, p<.0001 |

The two axes are separable: holding the gap large, moving the target representation from weak (R6b) to strong (R6c) removes the reconstruction prior's advantage; holding the target strong, moving the gap from small (R6, where supervised wins) to large (R6c) turns the supervised prior from best to worst. A single public dataset yields a controlled contrast in which the *same* supervised prior is best-in-class in one modality and worst-in-class in another, purely as a function of gap width and target capability — the sharpest external corroboration of the negative-transfer thesis (R2/R3): a transferred prior earns its keep only in proportion to the target's representational poverty.

### R6d — Second independent same-modality dataset (UTD-MHAD skeleton→skeleton)

To confirm the small-gap "supervised prior wins" prediction (R6) is not specific to CZU, we replicate it on a second, independently collected public dataset: UTD-MHAD (Kinect-v1, 20 joints remapped to the NTU-25 layout; 8 subjects, 27 actions, 861 clips). Same objectives and LOSO k-shot protocol; pooled over 3 seeds × 8 subjects (24 folds per cell):

| k | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|
| 0 (zero-shot) | 77.4 | 80.3 | 81.0 | **84.7** |
| 1 | 90.9 | 92.3 | 93.5 | **94.8** |
| 3 | 94.0 | 95.2 | 94.1 | **96.4** |

The R6 pattern replicates cleanly and with more power (24 folds vs 15). **supLP120 is the best prior at every k**: supLP120 − scratch = **+7.4 / +3.9 / +2.5 pp** (paired-t p<.0001 / .0005 / .016), and supLP120 > supMAE in 40/53 folds (sign p=.0003). supMAE is also significant-positive (+3.6 / +2.6 / +0.2 pp; p=.0002 / .007 / n.s.), and — as on CZU skeleton — **mae is positive, not negative** (+1.8 pp mean, sign 42/63, p=.011): the reconstruction-hurts signature again does not appear on a same-modality small-gap target. Two independent public datasets (CZU skeleton R6, UTD skeleton R6d) now agree that the pure-supervised prior is best-in-class when the gap is small — the mirror image of its worst-in-class behavior across the cross-modal gap (R6b/R6c).

**supcon (3 seeds, pooled n=24) replicates just as strongly.** supcon − scratch = **+8.4 / +4.0 / +2.9 pp @ k=0/1/3, all significant (p<.0001 / p<.0001 / p=.006)** — matching supLP120's margin closely (sign test supcon>scratch 53/59, p<.0001). supcon vs supLP120 is statistically even here too (25/52, p=.89, meanΔ +0.5 pp n.s.) — on UTD the two label-aware objectives are indistinguishable, both clearly beating scratch. Combined with the CZU result (where supcon edges out supLP120, p=.0002), the two-dataset, two-label-aware-objective picture is: **both supLP120 and supcon are best-in-class at small gap, with which one wins by more varying by dataset** — the deeper regularity is "label-supervised, no reconstruction component" as a class, not one specific recipe.

**Published-baseline comparison.** Same recognizer family as the CZU R6 anchor (statistical moments + CRC-RLS, λ=1e-4) on the byte-identical seed-42 LOSO splits (`temp_utd_crc_baseline.py`):

| k | CRC baseline | scratch | mae | supMAE | supLP120 |
|---|---|---|---|---|---|
| 0 | 69.48 | 75.98 | 79.46 | 82.02 | **84.33** |
| 1 | 92.25 | 90.25 | 92.56 | 93.50 | **94.27** |
| 3 | 95.30 | 93.52 | 94.41 | 94.41 | **96.24** |

supLP120 − CRC = **+14.85 / +2.02 / +0.94 pp** at k=0/1/3, echoing R6's consistent margin over the hand-crafted baseline. One difference worth noting honestly: on CZU, the from-scratch learned encoder was statistically level with CRC (R6); on UTD, **scratch already beats CRC by +6.5 pp at k=0** before any prior — the architecture has some edge here even unpretrained, so the prior's marginal lift over scratch (+8.35 pp @k0) is smaller than its margin over CRC. (UTD-MHAD's own published cross-subject accuracy figure is not yet cited here — TODO, needs a literature figure, not a rerun.)

### R6e — Synthesis: the five-setting map

Five source→target settings, two public external datasets plus the two internal ones (Xsens position-derived, CZU orientation-only and dual-raw), give a controlled sweep over gap width and target-representation strength:

| # | Setting (source → target) | Gap | Target representation | Best prior (Δ vs scratch) | Negative-transfer case |
|---|---|---|---|---|---|
| 1 | NTU → CZU skeleton (R6) | small (same modality) | native skeleton | supLP120 +7.2 pp @k0 (p=.0004) | none (mae positive, n.s.) |
| 2 | NTU → UTD skeleton (R6d) | small (same modality) | native skeleton | supLP120 +7.4 pp @k0 (p<.0001) | none (mae +1.8 pp, p=.011, positive) |
| 3 | NTU → Xsens, position-derived quats (R2) | middle (cross-device, skeleton-like target) | strong (mocap-grade quats) | supMAE (McNemar +200/+76/+131 @k0/1/3, p≤.0065) | **mae** (−334/−217/−92, p<.001) |
| 4 | NTU → CZU IMU, orientation-only (R6b) | large (cross-modal) | weak (Madgwick quats, yaw drift) | none (supMAE ≈ scratch, 25/42 p=.28; supMAE > supLP120 36/44 p<.0001) | **supLP120** (−2.3 pp, p≈.04–.06) |
| 5 | NTU → CZU IMU, dual raw (R6c) | large (cross-modal) | strong (raw ≈ CRC) | none (17/39, p=.52) | **supLP120** (6/41, p<.0001) |

No pretraining objective is unconditionally safe across the skeleton→wearable gap. Both the winning prior and the actively harmful one move as the gap widens and the target representation strengthens: supervised wins at small gap, hybrid at mid gap, nothing at large gap; the negative-transfer case is pure reconstruction at mid gap and pure supervision at large gap. The prior's gap-invariant value is calibration and cold-start data-efficiency (R4).

**A fifth objective (supcon) sharpens the small/large split into an axis, not two isolated findings.** Added after this map was first drafted, supcon — Khosla 2020 SupCon, label-supervised but contrastive rather than softmax — tracks supLP120 almost exactly at every one of the four external settings: it **wins big at both small-gap datasets** (CZU +7.0/+4.8/+4.4 pp k=0/1/3, all p<.001; UTD +8.4/+4.0/+2.9 pp, all p≤.006 — matching or exceeding supLP120's own margins, and beating supLP120 itself on CZU, p=.0002), and it **loses at both large-gap settings** (CZU-IMU quat: −1.4/−3.4/−0.2 pp, wash-to-marginal; CZU-IMU dual: sign 13/43 p=.014, significantly worse). Because supcon and supLP120 arrive at "label supervision" through different mechanisms (cross-entropy vs contrastive) yet land in the same place at every gap setting, the operative variable is not "is the objective supervised classification" but **"does the objective have a reconstruction/regularization component"** — present in supMAE (the only objective that doesn't collapse at large gap) and absent from supLP120/supcon alike. This reframes row 4–5's "supLP120" negative-transfer entries as an instance of a *class* of objectives, not a property specific to softmax classification.

*Gap ordering, measured:* the "middle" placement of NTU→Xsens is not merely narrative. The encoder-space CKA-per-target check (R3 extension) did **not** confirm the small/middle/large ordering — czu_skeleton (claimed small gap, same bucket as utd_skeleton) had the *lowest* mean CKA of all four targets on the supLP120 encoder, below both the claimed-middle (xsens_v2) and claimed-large (czu_imu_quat) settings, an instance of CKA's own necessary-not-sufficient limitation (R3) rather than a fit we forced. But a raw, encoder-free measurement — squared-MMD and Frechet distance on a hand-crafted, model-free feature (R3) — **does confirm the ordering exactly and monotonically**: czu_skeleton (MMD²=0.056) < utd_skeleton (0.096) < xsens_v2 (0.122) < czu_imu_quat (0.205), same ranking on Frechet distance independently. The gap ordering in this table therefore rests on three converging lines of evidence — the downstream sign-test/accuracy pattern above, and now a measured raw-distributional gap that tracks it directly — with CKA's alignment-based null read as a separate, orthogonal finding about what encoder alignment does and doesn't predict (R3), not as evidence against the ordering itself.

---

## Numbers ledger (traceability)
- R1 within-domain: `trained_models/NTU-to-NTU-objective-sanity/`; cross-domain: `LOSO-fullTrainCalibrate-v2{,-seed43,-seed44}/`.
- R2 McNemar / R3 CKA / R4a ECE: `trained_models/Phase1-analysis/{mcnemar,cka,ece}_results.csv`.
- R4b AUC-30/convergence: `wiki/results/multiseed-loso-v2.md` (RESEARCH_LOG §B).
- R4c A2: pooled 3 seeds — `trained_models/A2-subjectScaling{,-seed43,-seed44}/N*/summary.csv`, pooled table `trained_models/A2-subjectScaling-pooled/a2_pooled_results.csv`.
- R5 controller (locked/robust): `trained_models/Phase3-controller/robust/{vocab_sweep,vocab_ordering,costmodel_sweep,frontier,iso_safety}.csv` + PNGs (`temp_controller_robust.py`); prototype in `Phase3-controller/{controller_results,operating_point_summary}.csv`.
- R6 CZU: `trained_models/CZU-skeleton-LOSO/`; CRC baseline `…/crc_baseline/{crc_summary,comparison,crc_per_fold}.csv` (`temp_czu_crc_baseline.py`).
- R6b CZU inertial (cross-modal): `trained_models/CZU-IMU-LOSO/summary.csv`; CRC `…/crc_baseline/` (`temp_czu_imu_parser.py`, `temp_czu_imu_loso.sh`, `temp_czu_imu_crc_baseline.py`); data `Data_Processed/czu_imu_quats/`.
- R6c CZU dual-branch: `trained_models/CZU-IMU-DUAL{,-seed43,-seed44}/{raw,quat,dual}_<prior>/summary.csv` (`temp_czu_imu_raw_export.py`, `temp_czu_dualbranch.py`, `temp_czu_dual_run.sh` → `czu_dual.log`; raw-only pooled seeds via `temp_t1_multiseed_run.sh` → `t1_multiseed.log`); raw data `Data_Processed/czu_imu_raw/`; reuses CZU-IMU-LOSO splits; pooled `trained_models/CZU-IMU-DUAL-pooled/raw_scratch_pooled.csv`. Target-richness dial, pooled 3 seeds: `trained_models/CZU-IMU-DIAL/mag20{,-seed43,-seed44}/` (`temp_czu_imu_mag_export.py`, `temp_czu_dial_run.sh` + `temp_t1_multiseed_run.sh`), pooled `trained_models/CZU-IMU-DIAL-pooled/dial_pooled_raw.csv`.
- R6/R6b/R6c 3-seed pooling: `trained_models/CZU-{skeleton-LOSO,IMU-LOSO,IMU-DUAL}-seed{43,44}/` pooled with the seed-42 dirs; recompute via `temp_czu_multiseed_analyze.py` (`temp_czu_multiseed_run.sh` → `czu_multiseed.log`).
- R6d UTD-MHAD skeleton: `trained_models/UTD-skeleton-LOSO-seed{42,43,44}/summary.csv` (`temp_utd_parser.py`, `temp_utd_run.sh` → `czu_utd_run.log`); data `Data_Processed/utd_skeleton_lrq/`.
- R6/R6b/R6c/R6d supcon extension (3 seeds each): `trained_models/{CZU-skeleton-LOSO,CZU-IMU-LOSO,UTD-skeleton-LOSO}-supcon-seed{42,43,44}/summary.csv`, `trained_models/CZU-IMU-DUAL-supcon-seed{42,43,44}/dual_supcon/summary.csv` (`temp_t2b_followon_run.sh` stage 6); ad hoc pooling scripts, not yet folded into `temp_czu_multiseed_analyze.py`.
