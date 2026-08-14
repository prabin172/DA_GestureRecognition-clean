# Methods (draft)

_Source of truth for numbers is `paper_results.md` / `RESEARCH_LOG.md` §B. This document specifies the shared representation, encoder, pretraining objectives, transfer protocol, preprocessing, and the analyses (statistics, calibration, representation similarity, subject-scaling, controller). Written 2026-07-05, reflecting the v2 preprocessing and the multi-seed + Phase 1–3 analyses._

---

## 1. Overview of the study design

We hold a single controlled variable — the **pretraining objective** used to initialize a motion encoder — fixed everything else, and measure what transfers from an abundant vision-captured skeleton source (NTU RGB+D) to a data-scarce body-worn inertial target (Xsens). Both modalities are mapped to one representation (Local Relative Quaternions, §2) so that a single encoder architecture (§3) serves both. We compare five initializations — `scratch`, `mae`, `supMAE`, `supLP120`, and `SupCon` (§4) — under a deployment-realistic subject-held-out few-shot calibration protocol (§5), across a preprocessing that closes the modality gap (§6). We then analyze the results along five axes: paired multi-seed statistics (§7.1), clip-level significance via McNemar (§7.2), calibration (§7.3), representation similarity via CKA (§7.4), subject-count scaling (§7.5), and a downstream control-reliability simulation (§8).

## 2. Shared representation — Local Relative Quaternions (LRQ)

Every clip, from either modality, is encoded as a tensor of shape `(T, 17, 4)`: `T` temporal frames at 30 Hz, 17 body segments, and a unit quaternion per segment. Each segment's orientation is expressed **relative to its parent** in the kinematic hierarchy rather than in world space, making the representation invariant to global body rotation and translation.

- **17-segment body model:** Pelvis, T8, Head, {R,L}-Shoulder, {R,L}-UpperArm, {R,L}-Forearm, {R,L}-Hand, {R,L}-UpperLeg, {R,L}-LowerLeg, {R,L}-Foot. Hierarchy: Pelvis→T8→Head; T8→Shoulder→UpperArm→Forearm→Hand; Pelvis→UpperLeg→LowerLeg→Foot. The NTU joint topology is remapped to match the Xsens segment set exactly.
- **Flattening:** the `(17, 4)` per-frame orientation is flattened to a 68-d vector, so a clip is `(T, 68)`.
- **Temporal padding:** clips are zero-padded to `max_frames = 120`; a boolean mask (1 = real frame, 0 = pad) travels with each clip and is converted to a Transformer padding mask so pooling (`masked_mean`) averages only over real frames.

NTU quaternions are computed as bone rotations relative to an N-pose reference, projected through the hierarchy (`src/data/ntu_parser.py`). Xsens quaternions are computed by the v2 pipeline (§6).

## 3. Encoder architecture

The primary encoder (`KinematicEncoder`, `src/models/kinematic_encoder.py`) is a Transformer over the temporal axis: Linear 68→512, a learnable positional embedding `(1, 120, 512)`, and a 3-layer `TransformerEncoder` (pre-norm, 8 heads, 4× feed-forward). With `pool=False` it returns per-frame features `(B, T, 512)` for the reconstruction decoder; with `pool=True` it returns an L2-normalized clip embedding `(B, 512)` for classification. A padding mask `(B, T)` is threaded throughout.

The MAE decoder (`KinematicDecoder`) is a 1-layer Transformer + linear projection back to 68-d, used only during reconstruction pretraining. (A heavier MotionBERT-style `DSTformerQuatEncoder` exists in the codebase and was used in early experiments; all reported results use `KinematicEncoder`.)

## 4. Pretraining objectives (the independent variable)

All source pretraining is on NTU. Each objective produces one encoder checkpoint that seeds the transfer protocol (§5).

| id | objective | training signal |
|---|---|---|
| `scratch` | none (control) | random init; no source pretraining |
| `mae` | masked autoencoding (pure reconstruction) | mask ~70% of frames, reconstruct the full quaternion sequence (geodesic quaternion loss `1 − |q·q̂|`) |
| `supMAE` | hybrid reconstruction + supervision | shared encoder, multi-task MAE + cross-entropy on NTU labels, CE warm-up and EMA loss balancing |
| `supLP120` | pure supervised | cross-entropy over the 120 NTU action classes |
| `SupCon` | supervised contrastive (Khosla et al. 2020) | contrastive loss with label-defined positives on NTU |

`supMAE` follows the vision-domain SupMAE formulation and is cited/renamed accordingly ("Sup+MAE (hybrid)") in prose. `SupCon` is included as a **scoped negative exhibit**, not a first-class arm: it was evaluated only under the earlier single-seed local preprocessing, where it underperformed `scratch` at every k>0; it is not part of the v2 multi-seed sweep and is reported as such (see Discussion). `DANN` gradient-reversal variants exist in the codebase (mid-pack under the earlier preprocessing) and are cited as a domain-adaptation reference point, not a headline arm.

## 5. Transfer protocol — subject-held-out k-shot calibration (LOSO)

The Xsens target has 5 subjects (sub7–sub11) and 22 gestures. Each subject in turn is the fully held-out test subject (LOSO). For each held-out subject and each objective:

1. **Base phase.** Initialize the encoder from the objective's checkpoint (or random for `scratch`), then fine-tune the full model on the 4 non-held-out subjects (80 epochs).
2. **Calibration phase.** Freeze the encoder; take `k` labeled examples per class *from the held-out subject*, train the classification head only, and evaluate on the remainder of that subject's clips. `k ∈ {0, 1, 3, 5, 10}`; `k=0` = no calibration (zero-shot to the new subject after base training).

This isolates the prior: `scratch`, `supMAE`, and `supLP120` are fine-tuned on the *same* 4 subjects and calibrated identically, differing **only** in the initialization. Thus `supMAE − scratch` and `supLP120 − scratch` are apples-to-apples answers to the deployment question — "does adding a cross-domain prior on top of the 4 available subjects adapt better to the 5th?"

**Discriminating regime.** All methods reach 94–98% at k≥5 (ceiling); k∈{0,1,3} is where objectives separate, so those are the primary reporting ks.

## 6. Preprocessing — v2 position-reconstructed Xsens

The Xsens→LRQ conversion is the second controlled lever. The paper's primary preprocessing (**v2**) rebuilds each Xsens segment orientation from the mvnx **position** streams through NTU's own shortest-arc `get_bone_quaternion` construction, plus a per-session T-pose world alignment (`src/scripts/IMU_batch_processor_v2.py`, 2736 clips → `Data_Processed/imu_quats_v2/`). Because it reconstructs orientation the way NTU constructs it (twist excluded identically), it closes the source→target feature gap: MMD² is *below* the earlier measured-orientation baseline for every encoder.

Two earlier variants are retained only as gap reference points: **local** (v1: measured parent-relative orientation, including axial twist) and **swing** (axial twist stripped from Xsens post hoc). The swing variant was originally intended to *close* the gap but *widened* it ~3× — a symmetric test (twist-stripping NTU too) showed the increase came from twist removal damaging the Xsens signal, not from asymmetry. Swing is therefore reported as a **methodological finding** ("surface preprocessing does not do what intuition predicts; feature-distance metrics are necessary-not-sufficient"), not as an intentional experimental arm. All headline results use v2.

## 7. Analyses

### 7.1 Multi-seed paired statistics
The base+calibration protocol is run at three seeds (42, 43, 44) for k∈{0,1,3}, giving n = 15 (5 subjects × 3 seeds) per (method, k). We report per-(method,k) mean ± sd, paired Δ vs `scratch` with 95% CIs and paired-t p-values (subject-and-seed-blocked). At n=5 a two-sided Wilcoxon bottoms out at p=0.0625, so subject-level tests cannot reach p<0.05 alone; multi-seed pooling and clip-level McNemar (§7.2) carry significance instead. Selection rule: never best-test-epoch as primary; hyperparameters frozen on same-domain/validation data.

### 7.2 Clip-level McNemar
Because each held-out subject contributes ~400–500 clips, we dump per-clip posteriors (predicted class, softmax confidence, correctness, logits) from each base+calibration checkpoint (`temp_dump_posteriors.py`) and run a paired **McNemar test** on prior-vs-`scratch` predictions, pooled over seeds. We report the discordant counts b (scratch-only correct) and c (prior-only correct), net = c − b, and the McNemar p-value per (prior, k). No retraining — this reuses the multi-seed checkpoints.

### 7.3 Calibration (ECE)
From the same posteriors we compute Expected Calibration Error (15-bin), before and after single-parameter temperature scaling (temperature fit per method×k), and reliability diagrams. This quantifies how trustworthy each objective's confidence stream is — the input the controller (§8) consumes.

### 7.4 Representation similarity (CKA)
Layer-wise linear and RBF **Centered Kernel Alignment** between NTU and Xsens-v2 activations, per encoder, at the projection head and each of the three Transformer blocks (`temp_cka_analysis.py`). CKA is scale-invariant, removing the MMD scale confound. We also retain squared-MMD (median-heuristic and fixed-σ) as the necessary-not-sufficient companion. CKA is computed on **v2 only** (the swing arm is dead).

We also used the same layer-wise CKA to test whether the small/middle/large gap ordering across our five source→target settings (R6e) is measurable rather than narrative: `temp_cka_analysis.py --multi-target` computes NTU-vs-target CKA for each of Xsens-v2, CZU-MHAD skeleton, CZU-MHAD IMU orientation quats, and UTD-MHAD skeleton (n≈1000 clips/side, seed 0, inference-only). It does not confirm the ordering (Results R3) — we report the attempt and the null result rather than omit it or force a fit.

**Raw, encoder-free domain gap.** CKA and the encoder-space MMD above are both computed inside a *trained encoder's* feature space, confounding the measurement with the pretraining objective. To measure the datasets' distributional gap independent of any model, `temp_raw_domain_gap.py` computes a hand-crafted, model-free feature per clip — mean/std/var/skew/kurtosis over the 68 flattened LRQ channels (340-d; the same feature used for the CRC published-baseline reproductions, §6/R6/R6d) — for NTU and each of the four targets, then reports two metrics on pooled-standardized features (n=1000/side, n=861 for UTD, its full pool; seed 0): squared-MMD with an RBF kernel and median-heuristic bandwidth (fixing the fixed-σ scale confound of the encoder-space MMD), and Frechet distance (Gaussian closed form on a 50-component PCA projection, for numerical stability of the covariance term at this sample size).

### 7.5 Subject-count scaling (A2)
To ask *how long the prior stays useful as target data accumulates*, we sweep the number of fine-tuning subjects `N ∈ {0,1,2,3,4}` used in the base phase (`--n-train-subjects`, taking the first N of the sorted non-held-out subjects; N=0 short-circuits to the pretrained init followed by k-shot calibration only; N=4 recovers the full protocol of §5). 3 seeds (42/43/44), 5 LOSO folds per N per seed (n=15/cell), methods {scratch, mae, supMAE, supLP120}, k∈{0,1,3}. We report prior benefit vs scratch as a function of N (`temp_a2_run.sh` + `temp_t1_multiseed_run.sh`, `temp_analyze_a2.py`). We additionally repeat this sweep (N∈{0..3}, seed 42, {scratch, supLP120, supMAE}) on the CZU dual-branch target (§7's strong-target recognizer) to test whether the cold-start advantage generalizes beyond a representationally weak target.

### 7.6 A fifth objective: supervised contrastive (SupCon)
To complete the objective family named in the abstract, we add a fifth pretraining objective — Khosla et al. (2020) SupCon, label-supervised contrastive learning (in-batch same-label positives) — trained on NTU-120 identically to the other four, and re-run every analysis above (R1 linear probe and cross-domain LOSO-v2, R2 McNemar, R3 CKA, R4 ECE/AUC/convergence, R5 controller robustness, R6/R6b/R6c/R6d external datasets) with supcon included.

## 8. Control-reliability simulation (C6 / THMS pillar)

To translate recognition accuracy and calibration into human-machine-systems reliability metrics, the models' posterior streams are evaluated through an abstract, event-driven Finite State Machine (FSM) that consumes the **actual recognizer's posterior stream** — no physics engine (the scientific content is uncertainty→reliability propagation; a simulator would add risk without signal). `scripts/controller/controller_sim.py` (single-configuration prototype, not cited for any reported number) / `scripts/controller/controller_robust.py` (the robustness protocol, §8.1, source of every reported controller number). States are defined purely by consequence, not by any physical semantics: a **Safety-Critical State** is one where a misclassification or false activation triggers immediate task failure; a **Routine State** allows a **Recoverable Error** at a time/effort penalty.

- **Action-primitive assignment.** Seven abstract **Action Primitives** (System Inputs) make up the vocabulary for a **12-step Sequential Control Task**; none is itself a recorded gesture class — none of the 22 recorded classes is defined as a System Input a priori. Seven of the 22 real gestures are instead *relabeled* onto these primitives by a **uniform random draw** (`rng.choice`, no replacement) — no gesture's recall, confidence, or any other recognizer-derived property enters the draw, and the two Safety-Critical States are themselves two of the seven randomly-drawn primitives, not a reliability-selected pair. The assignment is deliberately non-semantic and non-tuned — the task and its System Input assignment are a vehicle for studying how recognition and calibration propagate to task-level outcomes, not a claim that this dataset contains any specific real-world command gestures. Specific physical movements are therefore not named against states in what follows.
- **Sequential Control Task.** An ordered 12-step task containing three Safety-Critical steps (drawn from the two Safety-Critical States) and nine Routine steps, so recognition errors on safety-critical steps compound.
- **FSM + safety layer.** For each task step with an intended System Input c, a held-out clip of the mapped gesture is sampled and its recognizer posterior consumed. A **confidence-threshold reject** (τ) re-prompts when max-softmax < τ (dwell/temporal-smoothing option requires agreement across repeated reads). A correct input advances the FSM; a **misclassification or false activation into a Safety-Critical State is an immediate task failure** (unsafe action); a Recoverable Error in a Routine State incurs a cancel-and-reissue cost. Excessive consecutive rejections abort the task.
- **Distractor / null stream.** Clips whose true class is *not* a System Input drive the **false-activation rate** (fraction that pass τ and map to a System Input).
- **Metrics**, reported per (init × k) and swept over τ: task success rate, mean time-to-completion (in cost units), rejection rate, corrective-inputs-per-task, and false-activation rate. Monte-Carlo (1000–3000 task runs/condition), fixed RNG seed.

### 8.1 Robustness protocol — locking the design knobs
A controller has design parameters (System Input assignment, error-cost model, operating threshold) that could each be tuned to manufacture a result. Instead of freezing arbitrary values, we probe all three simultaneously with a single Monte Carlo protocol: **120 independently, uniformly-randomly drawn 7-gesture System Input assignments**, each evaluated with **1000 Monte Carlo task-execution trials per (assignment, method, k, τ/C_crit) cell**. No assignment is chosen by any property of the recognizer (e.g. per-gesture recall) — every assignment is a fresh unweighted random draw of 7 of the 22 recorded gestures. Three locks apply this same randomized-assignment protocol to three different design knobs (`controller_robust.py`; outputs `trained_models/Phase3-controller/robust/`):

- **Lock 1 — randomized assignment, base outcome model.** Across the 120 assignments we report the *distribution* of task success and the fraction of assignments in which each pairwise method ordering holds.
- **Lock 2 — critical-cost sweep + two outcome models, same 120 assignments.** In a single pass we compute both a **hard-safety** outcome (any Safety-Critical error → task failure, binary success) and a **soft-cost** outcome (Safety-Critical errors are recoverable but incur a penalty C_crit), sweeping C_crit ∈ {2, 5, 10, 20, 50, ∞} (∞ recovers hard-safety), reporting the median and IQR of mean cost across the 120 assignments per (method, k, C_crit). This shows the ordering is independent of how catastrophic Safety-Critical errors are made.
- **Lock 3 — iso-safety operating point, same 120 assignments.** Because real-world systems operate on strict false-activation budgets rather than a static accuracy threshold, we fix a false-activation *budget* (1%, 0.5%) and find, per assignment, the smallest τ meeting it per method, reporting the distribution of τ*/success/cost across the 120 assignments — a deployment-standard, tuning-free way to set thresholds. The full τ-frontier (task-success vs false-activation), averaged over assignments, is also reported.

All reported controller claims come from this robustness protocol; no single fixed-assignment configuration is reported in isolation, and no lock's finding depends on a non-random gesture selection. The base cost weights (T_exec = 1, T_reject = 1, T_correct = 3, max consecutive rejects = 5) are held fixed while the load-bearing knobs are swept as above.

Locks 1 and 2 (not Lock 3) are additionally reproduced on every other transfer setting with compatible per-clip posteriors, using the identical shared-assignment protocol drawn fresh per setting from that setting's own label space (`controller_crosssetting.py`; Results R5b).

## 9. External-validity check — CZU-MHAD skeleton→skeleton

As an independent public target we run the same objectives and LOSO protocol on CZU-MHAD's skeleton modality (1165 clips, 5 subjects, 22 actions; Kinect-v2 25-joint order mapped to NTU's via `temp_czu_parser.py` → `Data_Processed/czu_skeleton_lrq/`). This is **skeleton→skeleton** — same modality, a much smaller gap than NTU→Xsens IMU — so it tests whether the objective orderings replicate on other data, not the IMU gap itself. CZU also ships 10 body-worn 6-axis inertial sensors (`sensor_mat/`, no magnetometer); the true cross-modal replication via that IMU data is left as future work.

## 10. Reproducibility notes
Encoders, checkpoints (`encoder_state_dict`/`decoder_state_dict`/`head_state_dict` + metadata), and per-clip posteriors are retained under `trained_models/`. Runs: v2 multi-seed `LOSO-fullTrainCalibrate-v2{,-seed43,-seed44}/`; Phase-1 analysis `Phase1-analysis/`; subject-scaling `A2-subjectScaling/`; controller `Phase3-controller/`; external validity `CZU-skeleton-LOSO/`. Data regeneration is deterministic given the parsers above.
