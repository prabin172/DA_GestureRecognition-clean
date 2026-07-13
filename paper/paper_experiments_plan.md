# Experiment plan — strengthening the THMS submission

_Created 2026-07-04. Companion to `paper_idea.md`. Priority set by human: **Tier A now, Tier B required, Tier C only if a public (skeleton, full-body-IMU) pair exists (no new in-house data), Tier D opportunistic.**_

Every experiment below is tagged with the reviewer objection it kills. The three that decide acceptance:
1. **"One dataset pair, one gap point"** — external validity (→ Tier C, and A1 turns 1 pair into a swept axis).
2. **"MMD correlation ≠ mechanism"** — Pillar 3 asserted until CKA (→ A3, D1).
3. **"Controller is a toy / no human-systems science"** — THMS venue fit (→ Tier B).

Dependency spine: **multi-seed sweep (running) → A1/A2/A3 → B1 → B2**. C and D run in parallel, off the critical path.

---

## Tier A — do immediately (cheap, no new data, highest leverage/effort)

### A1. Continuous gap sweep → dose-response law — **RUNNING (2026-07-04)**
**Goal.** Replace discrete gap points with a continuous *prior-benefit vs domain-gap* curve. Monotone dose-response = causal claim + the paper's killer figure.

**Method (corrected — swing is DROPPED, it was falsified).** The two principled endpoints are **v2** (position-derived, twist-free = what a skeleton source sees, low gap) and **local** (measured IMU orientation, full axial twist, higher gap). The gap between them *is* the axial-twist information IMUs have and skeletons don't. Interpolate per segment/frame:
`q(α) = R^α ⊗ q_v2`, `R = q_local ⊗ q_v2⁻¹` (residual = measured twist). α=0→v2, α=1→local, α>1→exaggerated twist (gap wider than local, cleanly). Pure post-process over the two frame-aligned processed sets — no re-parsing.
- Script: `temp_gap_interpolate.py` (smoke-verified: q(0)==v2, q(1)==local exactly). Datasets `Data_Processed/imu_quats_alpha{0.25,0.5,0.75,1.5,2.0}` (0.00→v2 dir, 1.00→local dir).
- Driver: `temp_alpha_sweep.sh` (nohup, `alpha_sweep.log`) — generates α-sets, **waits for the multi-seed run to finish** (GPU), then single-seed LOSO (k=0,1,3; scratch,supLP120,supMAE,mae) at each α → `trained_models/AlphaSweep/alpha{A}/`.
- Gap per α (MMD/CKA): deferred to analysis session (A3).

**Output.** Figure: x = α (or MMD/CKA), y = supMAE−scratch @ k=1, per-subject scatter. Expect monotone decline as twist (gap) grows. Validate `R`'s axis ≈ bone axis → "the gap is twist."
**Cost.** Low. **Dep.** queued behind multi-seed (auto-starts).

### A2. Subject-count scaling curve
**Goal.** Answer the real deployment question: *how many target subjects before the prior stops helping?* Expected: prior largest at N=1, washes out by N=4. New deployment finding, not a robustness check.

**Method.** Vary N target fine-tune subjects ∈ {1,2,3,4} (subsets of the 4 non-held-out), held-out subject fixed, on v2 data. Requires a small `--train-subjects` subset option in `temp_loso_fulltrain_calibration.py` (currently uses all 4 non-held-out). Report prior-vs-no-prior (supMAE−scratch, supLP120−scratch) at each N, k∈{0,1,3}.

**Output.** Curve: x = # target subjects, y = prior benefit @ k=1. Directly rebuts "with 4 subjects the prior is redundant" by showing *when* it isn't.
**Cost.** Low-moderate — one arg + a subject-subset loop; more base-trainings.

### A3. CKA + transfer-gap correlation (mechanism — mandatory)
**Goal.** Upgrade Pillar 3 from asserted to shown. MMD has a scale confound; CKA is scale-invariant.
**Method.** Layer-wise linear+RBF CKA on NTU-vs-Xsens features for v2/local/swing **and the A1 α-points**. Per-objective (same-domain acc − cross-domain acc) vs CKA similarity. Script unwritten — highest-value missing analysis (`paper_idea.md` §7.3).
**Output.** (i) CKA bars confirming gap ordering 0.0092/0.0109/0.0322 holds scale-invariantly; (ii) transfer-gap-vs-CKA scatter operationalizing "mismatch predicts transfer."
**Cost.** Moderate (write CKA script). **Dep.** needs the α-datasets from A1 for the richest version.

---

## Tier B — required for THMS (uncertainty → control)

**Prior art (copy the tooling, own the linkage).** Calibration measurement — Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 (temperature scaling, ECE, reliability diagrams). Calibration degrades under shift — Ovadia et al., NeurIPS 2019. Confidence-based **rejection** improves gesture-control usability, threshold→throughput best ~0.6–0.75 — myoelectric control lit (Robertson/Englehart; *CNN Confidence Estimation for Rejection-Based Hand Gesture Classification*, IEEE 2021). Reject theory — Chow 1970. **Nobody has connected pretraining objective → calibration-under-domain-gap → task reliability** — that chain is ours. Recommendation: **do NOT build a new calibration/uncertainty method**; use standard tools and make the novelty the linkage + the skeleton→IMU/control application.

### B1. Calibration quality per objective — plan
**Goal.** Test whether supMAE isn't just more accurate but better *calibrated* under the domain gap (calibration is what a controller needs).
**Steps.** (1) Add a `--dump-posteriors` eval pass to `temp_loso_fulltrain_calibration.py` (or a standalone loader over saved calibration checkpoints) that writes per-clip softmax + true label per (method, subject, k). Checkpoints exist under each run's `models/`. (2) Compute ECE + reliability diagrams per method × k on held-out clips; report pre/post temperature-scaling. (3) Optional: MC-dropout / deep-ensemble uncertainty only if softmax is too crude (start softmax — cheapest, standard).
**Output.** ECE table + reliability diagrams per objective; the claim that feeds B2.
**Cost.** Low (post-hoc, no retraining). **Dep.** final v2 checkpoints (can prototype now on the single-seed v2 run; finalize on multi-seed).

### B2. Controller that differentiates by objective — plan
**Goal.** THMS earner. Better-calibrated priors → higher task success under *compounding* asymmetric errors even at equal accuracy.
**Steps.** Build `paper_idea.md` §8 — abstract event-driven FSM over the B1 real posteriors; asymmetric sequential pick-place mission (a false `grasp` costs more than a false `next`); safety layer (confidence threshold + reject, dwell, mode manager); distractor/null stream from held-out non-target gestures for false-activation rate. Report task success / time / false-activation / rejection across 4 inits × k + the confidence-threshold tradeoff sweep (lifted from the myoelectric playbook, now indexed by pretraining objective). Near-zero new ML.
**Output.** Task-reliability differentiated by objective; threshold→safety/throughput curve.
**Cost.** Moderate (FSM + metrics harness). **Dep.** B1 + final recognizer numbers. Gated — build after recognizer results freeze; each step human-OK'd (RESEARCH_LOG rule 3).

---

## Tier C — external validity via a PUBLIC pair (no new in-house data)

Human constraint: **won't collect new subjects, but will use any public dataset at any cost.** Representational requirement: the target must be **full-body multi-IMU** (to reconstruct the 17-segment LRQ) with gesture-like classes overlapping the 22. Single-IMU sets (UTD-MHAD) don't fit without changing the representation.

### Chosen dataset: CZU-MHAD ([arXiv 2202.03283](https://arxiv.org/pdf/2202.03283), [repo](https://github.com/yujmo/czu_mhad))
**22 actions, 5 subjects, 8 reps = 880 clips** — same tiny-N LOSO regime as our Xsens. Actions overlap NTU (waves, hammer, grasp, draw-x, draw-circle, kicks, clap, bend, turns). Kinect v2 skeleton (25 joints, 30 fps) **+ 10 MPU9250 IMUs**.

**Data format (verified from repo demo/):** per clip three `.mat` — `_skeleton` (188×100), `_sensor` (1×10 object; each sensor `(N_i,7)` = accel xyz + gyro xyz + timestamp; **asynchronous per-sensor sampling, NO magnetometer**), `_depth`. Sensor placement (10): L/R elbow, L/R wrist, chest, abdomen, L/R knee, L/R ankle. **Full data (4.33 GB) is NOT in the git clone — only a 1-clip demo.** Downloading from Google Drive now → `external_data/czu_mhad_full.zip` (`czu_download.log`, background).

### C-plan: how we use it, what we do, what we compare
**The elegant bonus — CZU has BOTH modalities on the same clips**, so we can build the gap lever *within* CZU exactly like our v2-vs-local:
1. **Low-gap anchor (skeleton):** CZU Kinect positions → LRQ via *our own* v2 construction (`get_bone_quaternion`, `IMU_batch_processor_v2.py`). Twist-free, = what a skeleton source sees.
2. **Higher-gap target (IMU):** 10 sensors → per-sensor timestamp resample to a common clock → 6-axis accel+gyro fusion (Madgwick/Mahony, IMU-only) → segment orientations → LRQ. **Caveat: no mag → yaw drifts** (report as a limitation; may need per-clip yaw de-drift or heading reset).
3. **Sensor→segment mapping:** 10 CZU joints onto the 17-segment model; segments without a sensor (head, hands, feet, spine sub-segments) → identity or interpolate (declare).

**What we compare / the claim:** run NTU-pretrained objectives (scratch, supLP120, supMAE, mae) → LOSO k-shot on CZU-IMU, and measure the NTU↔CZU gap (MMD/CKA) for CZU-skeleton-LRQ vs CZU-IMU-LRQ. **If prior-benefit still tracks the gap on this fully independent dataset, external validity is nailed with the same *mechanism*, not just a second number.** Secondary: does the v2-vs-local twist-gap effect reproduce within CZU?

**Status — phase 1 (skeleton) DONE + RUNNING (2026-07-04).** Downloaded/unzipped (1165 clips). **Key finding: CZU stores the identical NTU/Kinect-v2 25-joint order** (proven from demo `skeleton_display.m` bone list J, Y-up) → CZU positions feed our own `src/data/ntu_parser.process_to_local_quats` with **no reindexing**. `temp_czu_parser.py` validated (unit quats, bone CoV 0.025–0.044 = rigid → order correct, head 0.77m above spine-base); generated `Data_Processed/czu_skeleton_lrq/` (drop-in index.csv, dry-run OK). `temp_czu_loso.sh` (nohup) **queued behind the α-sweep** → NTU-pretrained encoders LOSO k∈{0,1,3} → `trained_models/CZU-skeleton-LOSO/`. This is the first external-validity number (independent public target). **Phase 2 (IMU fusion) still gated:** 10 sensors → 6-axis accel+gyro fusion → LRQ, **no mag → yaw drift**; build + validate before any run.
**Backup if IMU fusion too noisy:** Berkeley MHAD (12 subj, 11 actions, mocap skeleton + 6 accel). **Fallback for pure gap-generalization:** DIP-IMU (17 IMUs, perfect sensor match, general-motion labels). UTD-MHAD out (single wrist IMU).

---

## Tier D — mechanism depth / preempt reviewers (opportunistic)

- **D1. Negative-transfer mechanism probes.** Why supLP120/SupCon hurt: frozen-feature target-class linear separability, feature rank / effective dimensionality, silhouette/cluster geometry. Makes C2-of-paper (negative transfer) mechanistic. Cheap; pairs with A3.
- **D2. λ-sweep for supMAE.** Robustness across sup/recon weighting → "robustness across λ" curve, preempts "you tuned λ." Moderate (retrain supMAE at several λ).
- **D3. Foundation-model baseline (UniMTS / IMU2CLIP).** Contextualize vs a big pretrained IMU model; preempt "why not foundation?" Losing on small data is still informative. Moderate (external model plumbing).

---

## Sequencing

1. **Now (parallel to running multi-seed):** A1 parser knob + single-seed α curve; A2 `--train-subjects` + scaling curve; C1 dataset scout.
2. **After multi-seed lands:** multi-seed the A1 anchors + A2 endpoints; A3 CKA (needs α-datasets); B1 calibration on final checkpoints.
3. **After recognizer freeze:** B2 controller. In parallel: D1 (with A3), D2, D3 as budget allows.
4. **If C1 finds a viable public pair:** C2 replication — highest payoff, off critical path.

Feeds back into `paper_idea.md` (§7.1/§7.3/§7.5/§8, C1–C6) and `wiki/results/`.
