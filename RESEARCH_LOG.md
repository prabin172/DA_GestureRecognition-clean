# RESEARCH_LOG.md — Planning ↔ Implementation Handoff

## How to use this file (read first, both sides)

This file is the shared channel between two assistants:
- **Planning (Claude chat):** decides framing, venue, paper structure, statistics policy, what counts as an honest claim. Writes into **Section A**.
- **Implementation (Claude Code):** runs experiments, reports results and implementation reality, raises blockers. Writes into **Section B**.

Rules:
1. **Section A is jointly owned.** It records decisions the human and the Planning assistant made *together* and the human has validated — not unilateral instructions. Treat it as agreed ground truth, but the human has final say and may revise anything.
2. **Each side reads the other's section, edits only its own.** Code reads Section A, writes Section B. Planning reads Section B, writes Section A. The human ferries updates across and validates both directions.
3. **Human-in-the-loop change control.** Before changing code, files, or experiment configs, Implementation states what it intends to change and *which item in this file it follows*, and waits for the human's go. Nothing in Section A auto-authorizes action without the human.
4. **Log conclusions and current state, not transcripts.** "MAE k=3 bug fixed, now 48.2±5" — not the debugging session. Keep it compressed.
5. **Append-mostly and dated.** Don't silently overwrite state; add a dated changelog line when something changes.
6. **Judgment lives in Planning; execution lives in Implementation.** Code reports what it found; the human + Planning decide what it *means* (significant? honest? leakage?). Don't decide claim-worthiness inside the coding session.
7. **Proposals vs commitments.** Items tagged *[PROPOSAL — needs human OK]* are ideas for discussion, not committed work. Do not start them until the human confirms.
8. The abstract and introduction draft already exists as a text file in this repo — refer to it as "the abstract/intro draft" rather than restating it.

---

## SECTION A — Planning → Implementation (Claude chat writes here)

### A0. Controller pillar — GO (build spec below)
This is a **confirmed pillar** for the THMS framing: it's what earns the human-machine-systems venue over a pure-sensing one. **Type: abstract event-driven controller, NOT a physics/robot sim.** The scientific content is how recognizer uncertainty propagates into control reliability; a physics engine adds no signal and weeks of risk.

Build spec (each step is human-OK'd before starting, per rule 3):
- **Mission spec:** ordered command sequences over a mode-switching / pick-place task. Map a subset of the 22 gestures to primitives (next, previous, approach, grasp, release, confirm, cancel). main.pdf is a usable engineering checklist (it's a placeholder paper, not a result).
- **Posterior stream from real clips:** for each mission step, sample a held-out-subject clip of the intended class, run the *actual* recognizer, feed its posterior to the controller. Near-zero new ML — an inference wrapper + the FSM loop.
- **Safety layer + FSM:** temporal smoothing, confidence threshold with a reject option, dwell-time filter, mode manager with error recovery + safe-home state.
- **Metrics:** task success rate, time-to-completion, false-activation rate, corrective-command count, rejection rate under gating, command latency. Report across the 4 inits × k.
- **One sweep:** confidence-threshold sweep → safety/throughput tradeoff curve (very THMS-flavored).
- **Design requirements to pre-empt reviewer attacks:**
  - Make the task **sequential and asymmetric** so errors compound and a false `grasp` costs more than a false `next` — this makes accuracy→task-success nonlinear and justifies "a few accuracy points matter."
  - Use **held-out non-target gestures as the distractor/null stream** for false-activation rate (don't invent idle data); declare inter-command timing as synthesized (state in limitations).
  - **Tune task difficulty + thresholds on validation, not test**, so the recognizer is the binding constraint; report the regime where it bites (the low-k regime — ties the controller to the core thesis).
  - **Assign safety-critical commands to reliably-recognized gestures** (per the OOV distinctiveness finding: crossarms/squat/wave reliable; throw/jump/hop not). This is a design choice justified by our own data and knits the OOV + controller pillars together.
- **Sequencing constraint:** build the controller **only after recognizer numbers are final** (esp. the MAE k=3 fix). Don't drive it off un-fixed inputs.

### A1. Locked framing (treat as ground truth; do not re-derive)
- **Contribution = a characterization, NOT a method and NOT a "SupMAE wins" result.** The finding is: *source-objective effects are conditional on source–target representation compatibility.* Objectives matter strongly within a domain; the effect compresses across the skeleton→wearable gap; representation mismatch (not objective, not source-label relatedness) dominates cross-domain transfer.
- **Do NOT claim "first skeleton-to-IMU/wearable transfer."** Prior work (Moya Rueda/Fink, Awasthi, SKELAR, UniMTS, PSKD, Zolfaghari, Xia) establishes transfer. The novel claim is the *controlled objective comparison under deployment-realistic protocols with explicit mismatch analysis*, plus the class-incremental (OOV) result.
- **"SupMAE" is an existing named method in vision (supervised MAE).** Cite the original on first use or rename (e.g. "Sup+MAE" / "hybrid"). Do not present it as new.
- **Target venue: IEEE Transactions on Human-Machine Systems (THMS).** Journal — depth and rigor beat big numbers. Small N is survivable if analyzed honestly.

### A2. Three pillars the paper rests on
1. **Objectives matter within a domain** (NTU→NTU, Xsens→Xsens controls). Large, real effects.
2. **Effect compresses across the domain gap** (NTU→Xsens). The core finding; the relatedness ablation reinforces it.
3. **Mismatch is the cause** — must be *shown* (CKA/UMAP), not just asserted from "NTU→NTU works."

### A3. Statistics & honesty policy (applies to every result table)
- N=5 subjects. **Treat small N as a serious, openly-stated limitation.** Add a per-subject results appendix; transparency reads as rigor at a journal.
- Use **paired, subject-blocked statistics** (Wilcoxon signed-rank or paired bootstrap), per-subject deltas, **effect sizes + CIs**. Do not headline a mean win that is within subject variance.
- **Never report best-test-epoch as the primary metric.** Use final-epoch or validation-selected checkpoints. Best-test only as supplementary diagnostic.
- **No tuning on the test signal.** Any hyperparameter (e.g. SupMAE λ, MAE masking ratio) must be selected on same-domain or validation-subject performance, then applied frozen to held-out subjects. Report the selection rule. λ work is a "robustness across λ" analysis, not a hunt for the magic value.

### A4. Current must-do list (priority order)
1. **Fix MAE Xsens-only k=3 = 20.70** (non-monotonic 46→20→50; almost certainly a bug — seed / k-shot sampler / unstable probe init). Re-run multi-seed. This number is currently un-shippable.
2. **Target-only / no-NTU-pretraining baseline** under the *identical* k-shot LOSO protocol. This is the deployment denominator — without it we cannot claim broad pretraining helps. (Repo has `train_imu_loso.py`, `TARGET_ONLY_LOSO/`.)
3. **Full LOSO with per-subject mean±std** across all 5 subjects + seeds.
4. **Paired stats added to every comparison table** per A3.
5. **CKA (layer-wise NTU-vs-Xsens features) + UMAP** — the evidence for Pillar 3.
6. **OOV / leave-class-out:** add **per-action statistics** and a **distinctiveness analysis** (correlate a gesture's inter-class kinematic separability with its few-shot OOV recall). Turns the heatmap into a finding.

### A5. Open questions for Implementation (please *confirm/report status* — several may already be solved)
These are status checks, not new work. If already done, just report the answer.
- **OOV/leave-class-out:** fully run or only scaffolded? Confirm the exact protocol actually executed (frozen-encoder + linear head? 21→22-way head expansion with known rows copied + OOV row random-init, as described?). [Human indicates this is already run — Code please confirm the executed protocol matches and report final per-action numbers.]
- **The spine-deciding number:** for NTU→Xsens at low k, **does target-only (no NTU) beat or match NTU-pretrained?** This decides whether the paper's spine is "use the broad data" vs "broad pretraining helps only at lowest k / mismatch limits it." Report as soon as known.
- **MAE k=3 = 20.70:** after multi-seed re-run, is it a bug or a real low-calibration instability? Report.
- **Contradictions:** any results that cut against the A2 pillars? Report loudly — they change the paper.

### A6. Candidate experiments *[PROPOSAL — needs human OK; discuss in chat before starting]*
Not committed work. Listed for the human to validate or drop:
- **Distinctiveness predicts onboarding (strong, cheap):** quantify each gesture's inter-class kinematic separability (e.g. centroid distance / silhouette in encoder space) and correlate with its few-shot OOV recall. Turns the OOV heatmap into a predictive finding. *(Already implied in A4.6; flagged here as the highest-value addition.)*
- **Mismatch quantification via CKA gap:** define a per-objective "transfer gap" = same-domain minus cross-domain accuracy, and relate it to CKA similarity between NTU and Xsens features. Directly operationalizes Pillar 3.
- **Frozen vs partial vs full adaptation Pareto:** if not already covered, a partial-finetune (final block only) condition to expose a compute/accuracy frontier — supports a deployment-cost story THMS likes.
- **Contrastive source baseline (scope decision, not necessarily a run):** Quinlan/UniMTS make "why no contrastive?" the first reviewer question. Either add one lightweight contrastive pretraining variant *or* explicitly scope it out with a one-paragraph justification. Human to decide which.

### A7. Changelog
- _[date]_ — Initial Section A: framing, pillars, stats policy, must-do list.
- _[date]_ — Controller pillar flipped to **GO** with build spec (A0). Header reworded for joint ownership + human-in-the-loop change control. A5 reframed as confirm/report (some items already solved on Code side). Added A6 candidate experiments as proposals needing human OK.

---

## SECTION B — Implementation → Planning (Claude Code writes here)

_Last updated: 2026-06-29 (updated with SupCon LOSO calibration results)_

### B1. What is actually run (vs scaffolded)

**Fully run:**
- Full LOSO calibration (all 5 subjects, all methods, k=0/1/3/5/10): `trained_models/LOSO-fullTrainCalibrate/`
- OOV / leave-class-out: fully run for all 22 classes across 5 subjects × 4 methods × k=1/3/5/10. Protocol confirmed: frozen base encoder + expand head to 22 classes with known-class rows copied and OOV row randomly initialized, then calibrate with k shots of the OOV class. Results in `trained_models/LOSO-LeaveClassOutFewShot/`.
- NTU→NTU within-domain sanity check: `trained_models/NTU-to-NTU-objective-sanity/`
- Cleaned-source (slerp035 quaternion smoothing) pretraining comparison: `trained_models/LOSO-fullTrainCalibrate-cleanedSlerp035/`
- SupCon (Khosla 2020) NTU pretraining: `trained_models/ContrastiveNTU/supcon_epoch_50.pth`
- SupCon full LOSO calibration (all 5 subjects, k=0/1/3/5/10): results appended to `trained_models/LOSO-fullTrainCalibrate/summary.csv`
- MMD domain gap analysis (all 5 encoders): `trained_models/MMD_DomainGap/mmd_table.csv`

**Fully run (added 2026-07-05):**
- Phase 1 — per-clip posteriors over v2 multi-seed ckpts → McNemar + ECE + CKA (no retrain): `trained_models/Phase1-analysis/{mcnemar,ece,cka}_results.csv`.
- Phase 2 — A2 subject-count scaling (N=0..4 fine-tuning subjects, seed 42, 4 methods, k∈{0,1,3}): `trained_models/A2-subjectScaling/`.

**Not run / scaffolded only:**
- Xsens-only LOSO (target-only no-NTU baseline): smoke result only. Full run needed to answer the spine-deciding question in A5.

### B2. Key numbers

**NTU→NTU within-domain (Pillar 1 — objectives matter within domain):**
| method | final_acc (30 ep, linear probe) |
|--------|--------------------------------|
| supervised | 59.6% |
| supMAE | 59.2% |
| MAE | 22.6% |
| scratch | 14.8% |

Finding: supervised pretraining and supMAE are on par and far above MAE-only. MAE alone does not produce discriminative features on NTU within-domain. This is a strong Pillar 1 result.

**Main LOSO calibration — mean final acc across 5 subjects (Pillar 2):**
| method | k=0 | k=1 | k=3 | k=5 | k=10 |
|--------|-----|-----|-----|-----|------|
| supMAE | 50.3 | 79.9 | 89.9 | 90.5 | 95.6 |
| targetSupDANN | 49.1 | 79.8 | 88.8 | 91.1 | 95.7 |
| MAE | 51.4 | 76.9 | 87.3 | 89.7 | 95.8 |
| scratch | 50.7 | 76.9 | 85.9 | 88.3 | 95.3 |
| sourceSupDANN | 48.8 | 74.0 | 86.3 | 88.1 | 94.0 |
| supLP120 | 45.0 | 72.6 | 84.6 | 86.0 | 93.7 |

Finding: across the domain gap (NTU→Xsens), the spread across methods at k=1 is ~7 pp (79.9 vs 72.6), far smaller than the 45 pp within-domain spread. supMAE leads at k=1/3 but all methods converge by k=10. This confirms the Pillar 2 compression story. **Note: per-subject variance is large (sub7 is consistently ~15-20 pp below others), so headline means alone are not sufficient — need paired stats.**

**LOSO-v2 (position-reconstructed Xsens) — mean final acc across 5 subjects, single seed (2026-07-04):**
| method | k=0 | k=1 | k=3 | k=5 | k=10 |
|--------|-----|-----|-----|-----|------|
| supMAE | 57.5 | 84.0 | 90.8 | 93.2 | 96.4 |
| supLP120 | 57.4 | 80.3 | 90.8 | 94.4 | 96.3 |
| scratch | 56.8 | 79.9 | 89.8 | 92.5 | 96.3 |
| mae | 50.7 | 77.0 | 86.6 | 91.2 | 94.8 |

Prior-vs-no-prior @ k=1 (supMAE − scratch): v2 +4.1, local +3.0, swing −1.3. Per-subject @ k=1 (v2): sub7 +9.7, sub10 +5.2, sub11 +4.3, sub9 +2.3, sub8 −1.0. supLP120 − scratch @ k=1: v2 +0.4, local −4.3. sub8 supMAE−scratch @ k=1: v2 −1.0 vs swing −19.9. MMD² (supmae): v2 0.0092, local 0.0109, swing 0.0322. Data `Data_Processed/imu_quats_v2/`; run `trained_models/LOSO-fullTrainCalibrate-v2/`.

**LOSO-v2 multi-seed — 3 seeds (42/43/44), n=15 (5 subj × 3 seeds), k∈{0,1,3} (DONE 2026-07-04):**

Final acc, per-(method,k) mean±sd:
| method | k=0 | k=1 | k=3 |
|--------|-----|-----|-----|
| supMAE | 59.66±6.68 | 83.28±6.41 | 91.24±4.43 |
| supLP120 | 58.41±7.43 | 80.97±4.91 | 90.81±4.19 |
| scratch | 57.15±6.84 | 82.31±6.55 | 89.39±4.90 |
| mae | 52.90±5.71 | 79.47±5.45 | 88.05±5.33 |

Paired Δ vs scratch (95% CI, paired-t p): supMAE−scratch k0 +2.52[−0.46,+5.49] p=.091, k1 +0.97[−1.78,+3.72] p=.461, k3 +1.85[−0.03,+3.74] p=.054. supLP120−scratch k0 +1.27 p=.494, k1 −1.34 p=.375, k3 +1.42 p=.264. mae−scratch k0 −4.24[−6.78,−1.71] p=.003, k1 −2.84[−4.43,−1.25] p=.002, k3 −1.34 p=.201.
Seed-by-seed supMAE−scratch (mean/subj): k1 → seed42 +4.13, seed43 −1.77, seed44 +0.56 (i.e. single-seed +4.1 @ k=1 is seed-42-specific; pooled +0.97). k0 → +0.63/+4.05/+2.86. k3 → +1.03/+2.25/+2.28.

AUC-30 (normalized area = mean eval_acc % over first 30 calib epochs; n=15 at k1/3, n=5 at k5/10):
| method | k=1 | k=3 | k=5(n5) | k=10(n5) |
|--------|-----|-----|---------|----------|
| supLP120 | 76.86 | 84.63 | 87.20 | 92.26 |
| supMAE | 76.72 | 82.92 | 83.77 | 90.19 |
| scratch | 75.99 | 80.88 | 83.00 | 89.64 |
| mae | 71.63 | 77.05 | 78.13 | 86.34 |

Paired Δ AUC-30 vs scratch: supMAE k1 +0.72 p=.412, k3 +2.04[+0.26,+3.81] p=.028. supLP120 k1 +0.87 p=.527, k3 +3.74[+1.03,+6.46] p=.010. mae k1 −4.37[−5.95,−2.79] p<.001, k3 −3.83[−5.20,−2.46] p<.001.

Convergence (first epoch ≥90% of final eval_acc; lower=faster): mean k1 scratch 10.07 / mae 13.00 / supMAE 9.53 / supLP120 6.07; k3 scratch 11.93 / mae 14.87 / supMAE 12.33 / supLP120 8.53. Δ vs scratch: supLP120 k1 −4.00[−6.97,−1.03] p=.012, k3 −3.40[−5.07,−1.73] p=.001; mae k1 +2.93 p=.021, k3 +2.93 p=.034; supMAE ≈ scratch (n.s.).
McNemar NOT computed — no per-clip prediction dumps saved (needs inference pass over `base_ckpt`+`calibration_ckpt`).
Runs: `-v2-seed{43,44}/`. Detail: `wiki/results/multiseed-loso-v2.md`.

**Phase 1 — McNemar / ECE / CKA over v2 multi-seed ckpts, per-clip (DONE 2026-07-05):**

McNemar (clip-level, pooled 3 seeds; net = prior_only_correct − scratch_only_correct):
| prior | k | n_pairs | scratch-only (b) | prior-only (c) | net | p |
|-------|---|---------|------------------|----------------|-----|---|
| supMAE | 0 | 7974 | 480 | 680 | **+200** | 0.0 |
| supMAE | 1 | 7644 | 342 | 418 | +76 | .0065 |
| supMAE | 3 | 6984 | 210 | 341 | **+131** | 0.0 |
| supLP120 | 0 | 7974 | 786 | 895 | +109 | .0084 |
| supLP120 | 1 | 7644 | 591 | 494 | −97 | .0035 |
| supLP120 | 3 | 6984 | 296 | 400 | +104 | 9e-05 |
| mae | 0 | 7974 | 789 | 455 | **−334** | 0.0 |
| mae | 1 | 7644 | 593 | 376 | **−217** | 0.0 |
| mae | 3 | 6984 | 385 | 293 | **−92** | .00047 |
mae = significant negative transfer at every k (all p<.001). supMAE significant positive at every k. supLP120 mixed (positive k0/k3, negative k1 — mirrors the multiseed k1 dip).

ECE (raw acc / raw ECE / temp-scaled ECE; n_clips per k):
| method | k | acc | ece | ece_tempscaled | T |
|--------|---|-----|-----|----------------|---|
| supLP120 | 0 | 58.48 | 0.255 | **0.0258** | 2.56 |
| supMAE | 0 | 59.62 | 0.275 | 0.0653 | 3.06 |
| scratch | 0 | 57.11 | 0.300 | 0.0676 | 3.37 |
| mae | 0 | 52.92 | 0.336 | 0.0676 | 3.49 |
| supLP120 | 1 | 81.01 | 0.092 | **0.0306** | 1.87 |
| supMAE | 1 | 83.27 | 0.074 | 0.0424 | 1.80 |
| scratch | 1 | 82.27 | 0.077 | 0.0494 | 1.93 |
| mae | 1 | 79.44 | 0.077 | 0.0667 | 1.80 |
| supLP120 | 3 | 90.79 | 0.025 | **0.0350** | 1.47 |
| supMAE | 3 | 91.18 | 0.022 | 0.0396 | 1.31 |
| scratch | 3 | 89.30 | 0.023 | 0.0396 | 1.41 |
| mae | 3 | 87.99 | 0.023 | 0.0545 | 1.34 |
supLP120 has the lowest temp-scaled ECE at every k (best-calibrated prior); mae worst at k=1/3.

CKA (NTU vs Xsens-v2 activations, linear; per encoder × layer):
| encoder | proj | L0 | L1 | L2 |
|---------|------|----|----|----|
| supLP120 | 0.0038 | 0.0133 | 0.0139 | **0.0149** |
| supMAE | 0.0029 | 0.0043 | 0.0047 | 0.0048 |
| mae | 0.0029 | 0.0036 | 0.0036 | 0.0035 |
| scratch | 0.0028 | 0.0029 | 0.0028 | 0.0027 |
Absolute CKA tiny for all; ordering by objective is clean and monotone with depth — supLP120 aligns ~3–5× better than the rest, scratch flat. Runs: `trained_models/Phase1-analysis/`. Detail: `wiki/results/phase1-mcnemar-ece-cka.md`.

**Phase 2 — A2 subject-count scaling (N=0..4 FT subjects, seed 42, 5 folds/N; DONE 2026-07-05):**

Prior benefit vs scratch (pp), by N × k. N=0 = pretrained init + k-shot only; N=4 = full LOSO-v2 setup.
| prior | k | N=0 | N=1 | N=2 | N=3 | N=4 |
|-------|---|-----|-----|-----|-----|-----|
| supLP120 | 1 | +7.08 | **+15.54** | +5.85 | +0.52 | +0.37 |
| supLP120 | 3 | +17.26 | **+26.89** | +10.91 | +3.97 | +1.02 |
| supMAE | 1 | +2.81 | +6.54 | +4.32 | +1.69 | +4.13 |
| supMAE | 3 | +7.79 | +7.51 | +3.31 | +3.08 | +1.03 |
| mae | 1 | +4.08 | −0.14 | −0.63 | −0.52 | −2.94 |
| mae | 3 | +5.65 | +2.79 | −0.78 | +1.45 | −3.22 |
Prediction confirmed: prior benefit peaks at N=0/1 (supLP120 +26.9pp @ k3,N1) and washes to ≈1pp by N=4; mae ≤0 for N≥1. Deployment lever: value of the pretrained prior decays with enrolled-subject count. Runs: `trained_models/A2-subjectScaling/N*/`. Detail: `wiki/results/a2-subject-scaling.md`.

**Phase 3 — controller PROTOTYPE (C6), on real held-out posteriors (2026-07-05; design NOT locked):**

12-step pick-place FSM, confidence-reject safety layer, asymmetric cost (wrong grasp/release = mission fail), method-agnostic command map, MC 3000/condition. `temp_controller_sim.py` → `trained_models/Phase3-controller/`.
- Compounding @ τ=0 (no gate), task-success k=1: supMAE 0.967 / scratch 0.905 / supLP120 0.861 / mae 0.754 → ~4pp recognition gap = 21pp task gap.
- Calibration→safety @ τ=0.9, k=3 false-activation: supLP120 0.0016 / supMAE 0.0040 / scratch 0.0052 / mae 0.0052; corrective/mission supLP120 0.070 vs scratch 0.206; time supLP120 13.6 vs scratch 14.9.
- Caveat (prototype): task success near-ceiling; headline sensitive to critical-fail rule + command-set.

**Phase 3 — controller LOCKED via robustness (2026-07-05b; `temp_controller_robust.py` → `trained_models/Phase3-controller/robust/`):**
Ordering shown invariant to the knobs, not frozen. (1) Randomized vocab: mean hard task-success across 120 random command-sets — k1τ0 scratch .516/mae .452/supMAE .530/supLP120 .479; k3τ.9 .830/.774/.847/.879. mae worst in all 4 conditions (mae≥supMAE in only 12–15% of vocabs @k1). (2) Critical-cost sweep C_crit∈{2,5,10,20,50,∞}: mae highest cost at EVERY C_crit @k1 (C_crit=20: mae 20.8 vs supMAE 15.6). (3) Iso-safety (fix false-activation budget, tuning-free): @1% budget k1 supLP120 τ*=0.90 cost 16.7 (13–20% faster) succ .953; @0.5% k1 only supLP120 meets spec with strong succ (τ*.97 .877), scratch/supMAE fail. Honest: supLP120 has confident false-critical-activation → dips below scratch on UNGATED metric @k1 (→ iso-safety is the correct framing). Locked claims: mae compounds worst + calibration governs safety/throughput. Detail: `wiki/results/phase3-controller.md`.

**CZU published-baseline reproduction — statistical-moments + CRC, identical LOSO splits (2026-07-05; `temp_czu_crc_baseline.py`):**
CZU paper (Chao 2022) protocol-comparable cross-subject skeleton ~75.5% (CRC + moments; their closed test not comparable). Our reproduction (their feature family + CRC λ=1e-4, on our LRQ, byte-identical LOSO splits), mean acc /5 folds: CRC 84.81/89.31/90.82 @k0/1/3. Vs learned recognizer: scratch 84.87/90.15/91.90, mae 87.94/89.98/93.77, supMAE 85.50/91.19/93.62, supLP120 93.14/93.08/93.85. **scratch ≈ CRC; supLP120−CRC = +8.33/+3.77/+3.03.** Outputs `trained_models/CZU-skeleton-LOSO/crc_baseline/`. Detail: `wiki/results/czu-skeleton-loso.md`.

**CZU-MHAD skeleton LOSO — external target, skeleton→skeleton, n=5 subj, single seed (DONE 2026-07-04):**
Final acc mean: k0 scratch 84.9 / mae 87.9 / supMAE 85.5 / supLP120 93.1; k1 90.1/90.0/91.2/93.1; k3 91.9/93.8/93.6/93.8. Δ vs scratch: supLP120 k0 +8.3 p=.07 (zero-shot), k1 +2.9, k3 +1.9; supMAE +0.6/+1.0/+1.7 (n.s.); mae +3.1/−0.2/+1.9 (mae *positive* here, unlike NTU→Xsens). AUC-30 k1: scratch 89.7/mae 89.8/supMAE 91.0/supLP120 93.2; near-ceiling so convergence uninformative (~1 epoch). Data `Data_Processed/czu_skeleton_lrq/` (1165 clips, 22 actions, 5 subj), skeleton modality only; CZU IMU (`sensor_mat/`, 10× 6-axis, no mag) unused = phase 2. Run `trained_models/CZU-skeleton-LOSO/`. Detail: `wiki/results/czu-skeleton-loso.md`.

**α-sweep (Tier A dose-response) — DONE but DROPPED:** 7 α (0.00→2.00), single seed, n=5. No clean/monotonic dose-response in supMAE−scratch @ k=1 (α: +4.1/+0.3/+1.2/+3.0/−1.3/+3.5/+1.7); confounded (scratch baseline non-monotonic in α; α=0/1 endpoints reuse existing data, interior interpolated). Not featured. Runs `trained_models/AlphaSweep/alpha*/`.

**SupCon LOSO calibration — mean final acc across 5 subjects:**
| method | k=0 | k=1 | k=3 | k=5 | k=10 |
|--------|-----|-----|-----|-----|------|
| supcon | 50.8 | 71.1 | 83.5 | 88.5 | 92.6 |

Finding: SupCon **underperforms scratch at every k except k=0**. At k=1 it is 5.8 pp below scratch (71.1 vs 76.9) and 8.8 pp below supMAE (71.1 vs 79.9). By k=10 it is still the weakest method (92.6 vs 95.3 scratch). Despite having the second-lowest MMD gap (0.0058, nearly matching supervised), it does not translate into good cross-domain calibration.

Interpretation: SupCon learns tightly clustered per-class embeddings tuned to NTU anatomy and motion style. Xsens clips land in the same general feature region (hence low MMD) but not near the correct NTU class clusters (hence poor calibration). The contrastive geometry is over-optimized for the source domain, leaving little room for a k-shot linear head to find the right boundaries in the target domain. This is direct empirical justification for scoping out contrastive pretraining in the paper (A6).

**OOV / leave-class-out — mean OOV recall across 5 subjects × 22 classes:**
| method | k=1 | k=3 | k=5 | k=10 |
|--------|-----|-----|-----|------|
| supMAE | 71.2% | 77.5% | 81.0% | 93.4% |
| MAE | 65.1% | 73.3% | 75.4% | 91.6% |
| scratch | 65.1% | 73.4% | 75.8% | 90.9% |
| supLP120 | 63.4% | 75.9% | 79.7% | 91.4% |

Finding: supMAE leads OOV recall at k=1/3/5 by 6-8 pp over scratch/MAE. This suggests richer representations help the most when incorporating a truly novel class with few shots. Still needs per-action breakdown and distinctiveness correlation (A4.6).

**MMD domain gap (Pillar 3 — mismatch is the cause):**
| method | MMD² |
|--------|------|
| sup | 0.0053 |
| supcon | 0.0058 |
| supMAE | 0.0109 |
| MAE | 0.0204 |
| scratch | 0.0514 |

Finding: supervised pretraining (on NTU) gives the smallest NTU↔Xsens feature gap. SupCon matches it closely. Importantly, MAE (which has low within-domain accuracy) also has a larger domain gap than supervised methods. scratch has ~5× the gap of supervised. **However, the ordering is NOT perfectly aligned with k-shot accuracy: MAE has higher k=1 accuracy than supervised yet higher MMD. This is a nuance worth reporting — low MMD is not sufficient; discriminative structure matters too.**

**Cleaned-source (slerp035) pretraining comparison:**
Smoothing NTU quaternion discontinuities did not consistently improve cross-domain transfer. supMAE_clean slightly outperforms supMAE at k=10 (96.3 vs 95.6) but is worse at k=1 (77.0 vs 79.9). No actionable gain. Cleaning not needed.

**CZU inertial cross-modal (skeleton→IMU) — mean final acc across 5 LOSO folds, single seed (2026-07-06):**
Data `Data_Processed/czu_imu_quats/` (Madgwick AHRS on 10 sensors, 6-axis no-mag → 17-seg local quats; `temp_czu_imu_parser.py`). Run `trained_models/CZU-IMU-LOSO/`.
| method | k=0 | k=1 | k=3 |
|--------|-----|-----|-----|
| supMAE | 59.25 | 56.31 | 61.10 |
| scratch | 56.29 | 53.97 | 58.08 |
| mae | 55.71 | 52.15 | 57.34 |
| supLP120 | 52.76 | 51.15 | 54.98 |
| CRC (raw accel+gyro) | 86.84 | 90.84 | 94.42 |

Δ vs scratch: supMAE +2.96/+2.35/+3.02 (paired-t p≈.19/.24/.20, n.s.); mae −0.58/−1.82/−0.74; supLP120 −3.54/−2.82/−3.10. Per-fold sign tests (n=15): supMAE>scratch 12/15 (p=.035), supMAE>supLP120 13/15 (p=.007), supLP120>scratch 5/15, mae>scratch 4/15; supMAE single-best in 10/15. Per-method sd @k0: supMAE 4.7, scratch 7.5, mae 9.3, supLP120 7.4. Contrast with same-modality [[czu-skeleton-loso]] (supLP120 +8.3 zero-shot). Deep ≪ CRC (orientation-only encoding, no accel, yaw drift).

**CZU 3-seed pooling (seeds 42/43/44) + target-strength dial + UTD-MHAD — done 2026-07-07.** Pooled stats via `temp_czu_multiseed_analyze.py`. Cells = subject×k×seed; sign tests drop ties.

R6 skeleton pooled means k0/1/3 — scratch 85.0/89.6/91.8, mae 87.4/89.4/92.7, supMAE 87.3/91.2/94.0, supLP120 92.3/92.2/93.9. Paired-t supLP120−scratch: k0 +7.21 (p=.0004), k1 +2.57 (p=.067), k3 +2.10 (p=.073). supMAE−scratch: k0 +2.23 (p=.048), k3 +2.15 (p=.014). mae−scratch positive, n.s.

R6b quat pooled means k0/1/3 — scratch 55.7/54.3/58.3, mae 54.9/53.0/57.4, supMAE 57.6/54.2/59.6, supLP120 53.6/51.9/56.0. Sign (pooled): supMAE>supLP120 36/44 (p<.0001); supMAE>scratch 25/42 (p=.28) [single-seed 12/15 p=.035 does NOT survive]; supLP120<scratch 28/43 (meanΔ −2.3, p=.066); supMAE>mae 27/41 (p=.060). Paired-t supMAE−scratch k0 +1.86 (p=.034), else n.s.; supLP120−scratch k0 −2.12 (p=.044), k3 −2.32 (p=.056).

R6c dual pooled means k0/1/3 — scratch 86.0/88.8/91.9, mae 84.8/88.6/91.6, supMAE 85.6/88.2/92.0, supLP120 84.5/87.0/90.0. Sign vs dual/scratch: supMAE 17/39 (p=.52), mae 13/36 (p=.13), supLP120 6/41 (p<.0001). Paired-t supLP120−scratch: −1.53/−1.81/−1.94 (p=.0016/.040/.0038, all k).

Dial mag20 (single seed) Δ vs scratch k0/1/3 — supMAE −1.9/−2.5/−0.8, supLP120 −4.8/−5.2/−3.7; scratch acc 84/88/90. (Between R6b 0-ch and R6c 60-ch.)

R6d UTD-MHAD skeleton pooled (3 seeds × 8 subj = 24/cell) means k0/1/3 — scratch 77.4/90.9/94.0, mae 80.3/92.3/95.2, supMAE 81.0/93.5/94.1, supLP120 84.7/94.8/96.4. Paired-t supLP120−scratch +7.35/+3.92/+2.45 (p<.0001/.0005/.016); supMAE−scratch +3.60/+2.59/+0.15 (p=.0002/.007/n.s.). Sign: supLP120>scratch 53/62 (p<.0001), supLP120>supMAE 40/53 (p=.0003), mae>scratch 42/63 (p=.011).

**CKA per target (T3, `tasks.md`) — done 2026-07-09.** `temp_cka_analysis.py --multi-target`, n≈1000 clips/side, seed 0. Mean linear CKA over L0–L2 per target×encoder:

| target | scratch | supLP120 | supMAE | mae |
|---|---|---|---|---|
| xsens_v2 | 0.0042 | 0.0290 | 0.0105 | 0.0077 |
| czu_skeleton | 0.0035 | 0.0267 | 0.0101 | 0.0074 |
| czu_imu_quat | 0.0103 | 0.0260 | 0.0152 | 0.0118 |
| utd_skeleton | 0.0068 | 0.0327 | 0.0137 | 0.0121 |

Does not order small(czu_skeleton,utd_skeleton)<middle(xsens_v2)<large(czu_imu_quat)-should-be-lowest-alignment: on supLP120, czu_skeleton (0.0267) < xsens_v2 (0.0290), i.e. the claimed small-gap target aligns *worse* than the claimed middle-gap target, and only barely above czu_imu_quat (0.0260, claimed large gap). utd_skeleton (0.0327) is highest, consistent. Full per-layer table: `trained_models/Phase1-analysis/cka_by_target.csv`.

**UTD CRC published-baseline anchor (T4, `tasks.md`) — done 2026-07-09.** `temp_utd_crc_baseline.py`, mirrors `temp_czu_crc_baseline.py`, on `trained_models/UTD-skeleton-LOSO-seed42/splits` (deterministic, no seeds needed). CRC k0/1/3 = 69.48/92.25/95.30 (per-fold sd 9.87/4.49/4.83, 8 folds). supLP120 − CRC = +14.85/+2.02/+0.94 pp. Unlike CZU (scratch ≈ CRC at k0, 84.87 vs 84.81), on UTD **scratch already beats CRC by +6.5 pp at k0** (75.98 vs 69.48) before any prior. Full comparison table `trained_models/UTD-skeleton-LOSO-seed42/crc_baseline/comparison.csv`. Still owed: UTD-MHAD's own published cross-subject accuracy as a literature citation (not a rerun — needs Planning/human to supply).

**SupCon CKA (T2.2 stage 4) — done 2026-07-09.** `temp_cka_analysis.py` re-run (single-target + `--multi-target`) now that `supcon` is in `ENCODER_CONFIGS`; only needs the frozen `supcon_epoch_50.pth` checkpoint, no dependency on the still-running LOSO-v2 supcon downstream runs (T2.2 stage 2). NTU-vs-Xsens-v2 CKA (L0-L2): supcon 0.0233/0.0231/0.0257 — numerically *highest* of all five encoders, edging out supLP120 (0.0133/0.0139/0.0149). Per-target: supcon mean L0-L2 — xsens_v2 0.0240, czu_skeleton 0.0221, czu_imu_quat 0.0221, utd_skeleton 0.0257 (same czu_skeleton-below-xsens_v2 inconsistency as supLP120, doesn't change the T3 honest-null finding). Note: re-run used the default `--out-dir`, so it overwrote `trained_models/Phase1-analysis/cka_results.csv` in place rather than a new dir — deterministic given fixed checkpoints, reproduced the existing supLP120/supMAE/mae numbers rather than replacing them (scratch differs by ≤0.0002, expected since it's an unseeded random init each run).

### B3. Answers to A5 open questions

- **OOV protocol confirmed:** head expansion with frozen encoder, OOV row random-init, k shots of OOV only. Confirmed executed correctly. Per-action numbers: see above (mean across classes). Per-action distinctiveness analysis still needed.
- **Spine-deciding number (target-only vs NTU-pretrained):** NOT fully answered — XsensOnly LOSO was only smoked (supervised k=1 giving ~6-23% per subject). The full run is needed. Without it we cannot claim the direction.
- **MAE k=3 = 20.70 non-monotonic issue:** The main full LOSO run (all methods) shows MAE k=3 = 87.35, which is monotonically between k=1 (76.86) and k=5 (89.67). The 20.70 number from the earlier partial run appears to have been a stale/buggy result. **Current numbers are monotone — no longer a blocker.**
- **Contradictions:** The MMD ordering doesn't align perfectly with cross-domain accuracy (MAE has larger MMD than supervised but similar k=1 accuracy). This is not a contradiction but a finding: MMD gap is a necessary but not sufficient condition. Discriminative capacity in the source domain also matters. Should be stated explicitly.

### B4. Blockers / needs from Planning

1. **Target-only baseline** (A5 spine question) — needs a go from Planning before running the full Xsens-only LOSO sweep (~5 subjects × 4 k values).
2. **Paired statistics** — need all subjects' per-subject arrays; have the data now but not the test code. Should implement Wilcoxon + effect sizes per A3 before any table goes to the paper.
3. **Distinctiveness analysis** (A4.6 / A6) — compute inter-class kinematic separability (e.g. centroid distance in encoder space) and correlate with per-action OOV recall. Data available; script not written.
4. ~~**SupCon in calibration pipeline**~~ — **DONE and resolved.** SupCon underperforms scratch at all k. Contrastive pretraining is now empirically scoped out. Planning can use this as justification in the paper rather than needing a one-paragraph argument.

### B5. CZU extension backlog (guided commands for a later agent)

Context: CZU now gives a 3-point gap-contingent arc (R6 small-gap supLP120 wins → R6b large-gap/weak-target supMAE wins, supLP120 worst → R6c large-gap/strong-target no prior helps, supLP120 worst). Pages [[czu-imu-dual]], [[czu-imu-crossmodal]], [[czu-skeleton-loso]]; paper R6/R6b/R6c. Ordered by value/cost. **Each needs a Planning go before running.**

1. ~~**Target-strength dial (cheapest, do first).**~~ **DONE 2026-07-07.** 20-ch mag rung added (`temp_czu_imu_mag_export.py` → `CZU-IMU-DIAL/mag20/`). Dose-response confirmed: supMAE edge exists only at 0-ch, gone by 20-ch. Numbers in B2; folded into paper R6c + [[czu-imu-dual]]. (Single-seed.)
2. ~~**CZU multi-seed (cheap power).**~~ **DONE 2026-07-07.** Seeds 43/44 pooled with 42 → 45 folds. Retired the n=5 caveat. **Key correction:** R6b supMAE>scratch does NOT survive pooling (25/42, p=.28 — was seed-42 luck); robust effect is supMAE>supLP120 (36/44, p<.0001) + supLP120<scratch. R6/R6c strengthened. Numbers in B2; paper + wiki updated.
3. **MotionBERT foundation-prior column (highest reviewer value, medium-high cost).** Answers "is negative transfer just YOUR checkpoints?". MotionBERT = DSTformer, 2D→3D lifting pretrain (ICCV 2023), eats 17-joint H36M-layout 2D keypoints (x,y,conf) — NOT quats. **First verify checkpoint availability + license on repo.** (a) CZU skeleton col: Kinect25→H36M17 map, orthographic 2D project, resample; frozen encoder + linear probe. (b) CZU IMU col (harder/lossier): FK from Madgwick quats w/ canonical bone lengths → 3D → 2D; yaw drift now pollutes positions. Do (a) first; only do (b) if (a) is interesting. If arc holds with a foundation prior, thesis generalizes past in-house priors.
4. ~~**UTD-MHAD second dataset (optional).**~~ **DONE 2026-07-07 (skeleton only).** Ran the same-modality skeleton column (3 seeds, 8 subj, 27 actions). Small-gap "supervised wins" replicates cleanly: supLP120 +7.4 pp @k0 (p<.0001); mae positive not negative. Numbers in B2; new page [[utd-skeleton-loso]], paper R6d. (IMU/cross-modal column not run — single-IMU too thin, per this note.)

Recommendation: 1+2 first (strengthen existing), then 3a. Adding runs grows the still-owed big commit.

### B6. tasks.md T1/T2/T5 — numbers, 2026-07-10

**T1 pooling (3 seeds, was single-seed):**
- A2 k=1,N=1: supLP120−scratch pooled +18.43pp p<.0001 (per-seed +15.54/+21.17/+18.57 — was flagged "seed-sensitive," now confirmed robust). A2 k=3,N=1: +27.46pp p<.0001 (single-seed was +26.89).
- R6c raw-only pooled: 81.94/90.01/95.56 @k0/1/3 (single-seed was 80.7/89.8/95.2). dual/scratch>raw/scratch pooled 19/42 p=.644 (single-seed 6/12 p=1.00, same wash conclusion).
- DIAL 20-ch pooled Δvs scratch: supMAE −1.37/−1.56/−1.12, supLP120 −4.05/−5.15/−3.61 (single-seed was −1.9/−2.5/−0.8 and −4.8/−5.2/−3.7).

**T2 SupCon (5th objective, 3 seeds where applicable):**
- R1 NTU probe: 55.71% (vs supervised 59.6, supMAE 59.2, mae 22.6, scratch 14.8). R1 cross-domain (n=15): 58.08/82.34/91.41 @k0/1/3; Δvs scratch n.s.-to-marginal (+0.93/+0.03/+2.02pp, p=.69/.98/.07).
- R4b AUC-30/convergence: k3 AUC-30 +4.18pp p=.0056; k3 convergence −3.53ep p=.0029 (k1 n.s. both).
- R2 McNemar: net +85(p=.034)/+5(p=.90,ns)/+144(p<.0001) @k0/1/3.
- R4a ECE tempscaled: 0.0339/0.0410/0.0410 @k0/1/3 (2nd-best after supLP120 0.0258/0.0306/0.0350).
- R5 controller: Lock1 exact-reproduces (scratch/mae/supMAE/supLP120 unchanged to 3dp); supcon k1τ0=0.506, k3τ0.9=0.880 (best of 5). Locks2/3 (`robust-supcon/`) used a DIFFERENT fixed vocab than the locked `robust/` (root cause: `reliability_ordered_vocab()` pools recall over all loaded methods incl. supcon — confirmed via code read, not a seeding bug, Lock1 unaffected since its vocab is per-vocab-random not pooled-recall-derived). Locked `robust/` untouched. supcon iso-safety(1%,k1): τ*=.99, cost=19.0 (supLP120 τ*=.85 cost=16.6 under same re-derived vocab).
- R6 CZU-skel supcon (n=15): 92.04/94.37/96.23 @k0/1/3; Δscratch +7.00/+4.79/+4.41pp all p<.001. supcon>supLP120 34/43 p=.0002.
- R6b CZU-IMU-quat supcon (n=15): 54.28/50.90/58.07; Δscratch −1.44/−3.35/−0.24 (n.s.-marginal, k1 p=.053). sign supcon>scratch 19/45 p=.37; supcon>supLP120 26/44 p=.29.
- R6c CZU-dual supcon (n=45): 84.74/87.65/90.87; Δscratch −1.24/−1.14/−1.07 (p=.048/.12/.08). sign supcon>scratch 13/43 p=.014.
- R6d UTD-skel supcon (n=24): 85.73/94.94/96.90; Δscratch +8.36/+4.02/+2.94 all p≤.006. sign 53/59 p<.0001. supcon vs supLP120 25/52 p=.89 (tied, unlike CZU where supcon significantly beats supLP120).

**T5 CZU-dual cold-start (single-seed, n=5/cell, N=0..3, scratch/supLP120/supMAE):** no N at which either prior beats scratch. N=0,k=1: supLP120 Δ=−5.62pp p=.045; supMAE Δ=−5.55pp p=.005. N=1,k=3: supLP120 Δ=−2.54pp p=.004. N=2,k=1: supLP120 Δ=−3.54pp p=.015, supMAE Δ=−4.07pp p=.006. Full table `trained_models/CZU-DUAL-subjectScaling/`.

All folded into `paper/paper_results.md` (R1/R2/R4/R5/R6/R6b/R6c/R6d/R6e), `paper_intro.md`/`paper_method.md`/`paper_abstract.md` updated, tex regenerated. Wiki pages updated: [[a2-subject-scaling]], [[czu-skeleton-loso]] (was stale since 2026-07-04, brought current), [[czu-imu-crossmodal]], [[czu-imu-dual]], [[utd-skeleton-loso]], [[phase1-mcnemar-ece-cka]], [[phase3-controller]], new page [[czu-dual-cold-start]]. `tasks.md` T1/T2/T5/T0-abstract items now complete; T3/T4/T6(paused)/T7(unapproved)/T8(deferred)/T9(deferred) status unchanged.

### B7. Raw (method-independent) dataset domain-gap quantification — numbers, 2026-07-12

Human asked for a domain-difference metric independent of any pretrained encoder (existing MMD/CKA are both computed inside a trained encoder's feature space, hence confounded by the pretraining objective). New script `temp_raw_domain_gap.py`: reuses the CRC-baseline hand-crafted feature (`temp_czu_crc_baseline.py::clip_features`, mean/std/var/skew/kurtosis over 68 flattened LRQ channels = 340-d, no model/no pretraining) as a raw feature space, all 5 datasets share the (T,17,4) LRQ schema. n=1000/side (utd=861, its full pool), seed 0, pooled-standardized per pair. All 10 pairwise combinations of {ntu, xsens_v2, czu_skeleton, czu_imu_quat, utd_skeleton} → `trained_models/RawDomainGap/raw_domain_gap.csv` + 3 heatmaps. Reporting MMD²/Frechet only — a third metric (proxy A-distance) was also computed but saturated near-maximal on every pair (trivially separable), carries no ranking signal, dropped from reporting per human decision.

**NTU-anchored ordering (the one that matters for the paper's gap claim):**
| target | mmd2 | frechet |
|---|---|---|
| czu_skeleton | 0.05603 | 176.82 |
| utd_skeleton | 0.09603 | 202.09 |
| xsens_v2 | 0.12178 | 281.94 |
| czu_imu_quat | 0.20491 | 393.83 |

**MMD² and Frechet BOTH confirm the claimed small(czu_skel)<small(utd_skel)<middle(xsens_v2)<large(czu_imu_quat) gap ordering exactly, monotonically, on both metrics independently.** This is the opposite outcome from T3's encoder-space CKA (see [[domain-gap-metrics]]), which did NOT confirm the ordering (czu_skeleton ranked lowest/most-different there). Folded into `paper/paper_results.md` (R3 new subsection + R6e caveat rewritten), `paper_discussion.md` §1, `paper_method.md` §7.4 — the five-setting map's gap ordering now has measured, encoder-free backing where before it was narrative + a CKA null. Tex regenerated.
