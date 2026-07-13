# Paper idea — IEEE THMS submission blueprint

_Status: draft 2026-07-05 (v2 preprocessing; **multi-seed stats landed; Phase 1 McNemar/ECE/CKA + Phase 2 A2 subject-scaling + Phase 3 controller done**). Synthesizes the locked framing (`RESEARCH_LOG.md` §A / [[paper-framing]]), the v2 multi-seed results ([[multiseed-loso-v2]], [[phase1-mcnemar-ece-cka]], [[a2-subject-scaling]], [[phase3-controller]]), the external-validity check ([[czu-skeleton-loso]]), the publishability audit ([[publishability-review]]), and the external deep-research literature report (`Undermind ... vision/skeleton to IMU.pdf`)._

> **RESHAPE (2026-07-05, post multi-seed + Phase 1–3).** The single-seed "+4.1 pp @ k=1 supMAE" delta was a **seed-42 artifact** (pooled +0.97, n.s.). The load-bearing findings moved:
> - **C1** compression ~45pp → ~7pp: **holds.**
> - **C2 negative transfer INVERTED.** The seed-stable, McNemar-significant negative-transfer case is **`mae` (pure reconstruction)**, below scratch at every k (p<.001) — this *inverts the field's "reconstruction is the safe prior" assumption*. Supervised (`supLP120`) is neutral-to-positive on accuracy and *best-calibrated*; contrastive (SupCon) was a local-mode single-seed artifact, demoted.
> - **C3 mechanism:** MMD necessary-not-sufficient **+ CKA** (supLP120 aligns 3–5× the rest, monotone with depth).
> - **C4 gap-knob is DEAD** (swing was a twist-stripping artifact, not an intentional lever). Replaced by **A2 subject-count scaling** (prior benefit peaks at N=0/1, washes to ≈1pp by N=4) + the v2-vs-swing finding demoted to a *methodological* note.
> - **C5:** the prior's real value is **calibration efficiency** — AUC-30 k3 supMAE +2.04 p=.028 / supLP120 +3.74 p=.010; convergence 3–4 epochs faster (supLP120 k1 −4.00 p=.012); best ECE (supLP120).
> - **C6 controller (Phase 3, DONE):** objective→calibration→task-reliability chain. mae's negative transfer *compounds* (task-success 0.75 vs 0.97 @ k=1, τ=0); supLP120's calibration yields the safest operating points.
> - **External validity:** CZU skeleton→skeleton, supLP120 zero-shot **+8.3 pp p=.07**; mae is *positive* there (+3.1) — the mae-hurts signature is *specific to the cross-modal IMU gap*, strengthening C2.
>
> Sections below retain the original single-seed prose where it is still the design rationale; **`paper_results.md` is the source of truth for numbers.**

---

## 1. Context — why this document exists

We have a working skeleton→IMU transfer pipeline and a pile of results, but the results as framed in the old handoff overstate a "SupMAE wins / skip-NTU self-supervision" story that (a) rests on a mislabeled condition and one cherry-picked `k`, and (b) is not what the literature makes valuable. The Undermind report is unambiguous about where the open gap actually is:

> **No existing study systematically varies *both* pretraining objective *and* domain gap in skeleton/vision→IMU transfer, nor explicitly demonstrates and explains negative transfer of supervised skeleton pretraining under large gaps.** The field has "voted with its feet" — abandoning pure supervised objectives for reconstruction/hybrid ones as gaps grow — but no one has measured *when* and *why*.

That is exactly the study our data already half-supports. This document reframes the paper around that gap, lists the concrete experiments needed to make it impeccable, and specifies the controller pillar that earns the THMS (human–machine-systems) venue rather than a pure-sensing one. It requires substantial additional work (a completed 2×2, multi-seed stats, CKA, a symmetric gap test, and the controller) — that is the point: this is the version that survives review.

---

## 2. The one-sentence claim (locked)

**The utility of a pretraining objective is conditional on source–target representation compatibility: discriminative-supervised objectives help under small gaps but degrade to redundancy or negative transfer as the skeleton→wearable gap grows, while reconstruction/hybrid objectives are more robust — and we show this with a controlled objective × domain-gap sweep, a representation-level mechanism (CKA), and a downstream control task that turns recognizer uncertainty into task reliability.**

This is a **characterization, not a method win.** We do not claim "first skeleton-to-IMU transfer" (prior work owns that — §5) and we do not headline "SupMAE is best." SupMAE/hybrid being *robust across regimes* is a supporting observation, not the thesis.

## 3. Title candidates

1. *When Does Skeleton Pretraining Help Wearable Gesture Recognition? A Controlled Study of Pretraining Objectives Across the Domain Gap*
2. *Conditional Utility of Pretraining Objectives in Skeleton→IMU Transfer: From Representation Mismatch to Control Reliability*
3. *Objectives Matter Less Across the Gap: Characterizing Negative Transfer in Vision-to-Wearable Gesture Recognition* (THMS-flavored; leads with the counterintuitive finding)

Recommend #2 — names the mechanism (representation mismatch) and the THMS hook (control reliability) in one line.

---

## 4. Novelty, positioned against prior work

The Undermind report clusters ~30 core papers. None does our combination. The table below is the competitor grid to reproduce in the Related Work section; the right column is the wedge.

| Prior work | What it does | What it does **not** do (our wedge) |
|---|---|---|
| Moya Rueda & Fink 2021; Awasthi et al. 2022 (pose→IMU supervised transfer) | Supervised pose/skeleton pretraining → IMU finetune; gains under *moderate* gaps | No objective comparison; no OOD/novel-class; no negative-transfer measurement |
| PSKD (Ni et al. 2022); TAS (Qiao et al. 2026); D-CAT (2025) | Skeleton→IMU knowledge distillation / cross-attention alignment | Architectural bridging, not objective ablation; supervised or hybrid *by design*, never contrasted |
| SKELAR (Li & Gupta 2025); Pose-Sensor (Zolfaghari et al. 2024) | Reconstruction-style skeleton pretraining then adapt to IMU | Adopt reconstruction *a priori*; no head-to-head supervised-vs-recon ablation |
| PRIMUS (Das & Malekzadeh 2025); BenchHAR (Cai & Hong 2026) | Systematic objective comparison under OOD — **IMU-only** | Never uses skeleton/vision as the pretraining modality; within-IMU, so gap is small |
| IMU2CLIP (2022); UniMTS (2024); AURA-MFM (2025) | Foundation-style multimodal (text/video/mocap/IMU) contrastive spaces | Zero/few-shot capable but no controlled objective×gap analysis; huge-scale, not deployment-realistic small-N |
| Simulation line: IMUTube (2020), Video2IMU (2022), CROMOSim (2022–24) | Generate synthetic IMU from pose/video for augmentation | Reconstruction/regression only; no reusable-encoder objective comparison; no negative transfer study |

**Our unique cell:** skeleton/vision as the *pretraining modality* × {supervised, reconstruction, hybrid, contrastive} objectives × {within-domain, cross-domain, gap-manipulated} settings × {LOSO few-shot calibration, novel-class onboarding} protocols, with an explicit representation-mismatch mechanism and a demonstrated negative-transfer case — then propagated into a control-reliability task.

---

## 5. The gap we fill (from the Undermind synthesis)

The report identifies four evidence strands and one hole:

1. **Small gaps → supervised helps** (Moya Rueda/Fink, Awasthi, PSKD): pose→IMU with similar taxonomies.
2. **Large gaps → field migrates to reconstruction/contrastive/hybrid** (SKELAR, TAS, MAE-video+IMU, IMU2CLIP, UniMTS): objectives that *preserve generative structure* survive modality/subject/class shift.
3. **Systematic objective-vs-gap comparisons with robust conclusions exist only within IMU** (BenchHAR: hybrid recon+contrastive wins OOD; PRIMUS: +15% few-shot on OOD).
4. **Foundation/zero-shot models** avoid class-taxonomy collapse via contrastive alignment (AURA-MFM, IMU2CLIP) — consistent with "supervised shapes latents too narrowly."

**The hole:** nobody has (i) used skeleton/vision as the pretraining source, (ii) put purely-supervised vs reconstruction vs hybrid vs contrastive objectives head-to-head, (iii) under *systematically varied* skeleton→IMU gaps (within vs cross, plus a manipulated gap), (iv) while explicitly identifying and *mechanistically explaining* negative transfer. That is our paper.

---

## 6. Contributions (as they will appear in the intro)

- **C1 — A controlled objective × domain-gap characterization.** Within-domain, objective choice causes a ~45 pp accuracy spread (NTU linear probe: supervised 59.6%, supMAE 59.2%, MAE 22.6%, scratch 14.8%). Across the skeleton→IMU gap the same choice compresses to ~7 pp at k=1 (NTU→Xsens LOSO). The effect is *conditional on the gap*, not intrinsic to the objective.
- **C2 — A demonstrated, explained negative-transfer case.** Purely-supervised NTU pretraining (`supLP120`) transfers *below scratch* at low k (72.6 vs 76.9 at k=1); contrastive (`SupCon`) is the *worst* transferring objective at every k>0 (71.1 at k=1) *despite* the second-lowest feature-space gap (MMD 0.0058 ≈ supervised 0.0053). This is the exact "supervised/contrastive over-specializes the source geometry" negative-transfer signature the literature only hints at.
- **C3 — A representation-level mechanism, not an assertion.** Low domain gap is **necessary but not sufficient**: MMD ordering does not match transfer ordering (SupCon low-gap/worst-transfer; MAE higher-gap/similar-transfer). We show *why* via layer-wise CKA + UMAP and a per-objective "transfer gap" (same-domain minus cross-domain accuracy) correlated with representation similarity.
- **C4 — Prior value scales with a *manipulated* domain gap.** Three preprocessings of the same Xsens data sweep the NTU↔Xsens MMD² over 0.0092 (v2) → 0.0109 (local) → 0.0322 (swing) for supMAE, and the hybrid prior's few-shot benefit tracks it monotonically (+4.1 / +3.0 / −1.3 pp at k=1). The characterization rests on a *controllable causal lever*, not a single gap value; the hybrid (supMAE) is the only prior positive across all three regimes (robustness), while the pure-supervised prior flips sign (negative under local, positive under v2). See §7.5.
- **C5 — Deployment-realistic protocols.** Subject-held-out k-shot LOSO calibration (the wearable reality: a few labeled reps from a new user) and novel-class onboarding (OOV: 21→22-way head expansion, k-shot). Under OOV, richer/hybrid representations lead by 6–8 pp at k≤5 — objectives matter most exactly when data is scarcest.
- **C6 — Uncertainty → control reliability (the THMS pillar).** An abstract event-driven controller consumes the *actual* recognizer's posterior stream from held-out clips; we quantify how objective/init/`k` choices propagate into task-success, false-activation, and rejection curves, and show a confidence-threshold safety/throughput tradeoff. This converts "a few accuracy points" into human–machine-system consequences.

---

## 7. Experimental design to make it impeccable

This is the "massive change" set. Grouped by the pillar each item defends. Items marked **(new)** are not yet run.

### 7.1 The deployment question is the spine
The operational framing: **we always hold the 4 non-held-out Xsens subjects; does adding a weak cross-domain prior (an NTU-pretrained encoder) on top of that data adapt better to the 5th subject than the 4 subjects alone?** This is *the* claim, and its clean answer already lives inside Job 1 — no new run needed to state it:

- **`scratch`** = random init → fine-tune on the 4 subjects → head-calibrate on k shots of sub5 = "just the 4 subjects, no prior."
- **`supMAE`** = NTU-pretrained (the prior) → fine-tune on the *same* 4 subjects → same head-calibration = "prior + the 4 subjects."
- **`supLP120`** = pure-supervised NTU prior, otherwise identical.

All three are fine-tuned on the same 4 subjects and adapt to sub5 identically; they differ *only* in the prior. So `supMAE` vs `scratch` and `supLP120` vs `scratch` are apples-to-apples answers to the deployment question — **not confounded.** What they show is the thesis in miniature, swept across three gap regimes (supMAE−scratch @ k=1): the hybrid prior helps under the tightest (**v2**) gap **+4.1 pp**, helps less under local **+3.0 pp**, and washes out under the widest (swing) gap **−1.3 pp** — value scales monotonically with how close the domains are (MMD² supmae 0.0092 v2 / 0.0109 local / 0.0322 swing). The pure-supervised prior (`supLP120`) flips from **negative transfer** under local (−4.3 pp) to slightly positive under v2 (+0.4 pp) — objective- *and* gap-dependent prior value, with the gap now a controllable lever (§7.5).

- **Objective families (already have encoders):** supervised (`supLP120`), reconstruction (`MAE`), hybrid (`supMAE` / Sup+MAE — *rename & cite the vision original*), contrastive (`SupCon`), and `scratch` control. Spans the four families the Undermind report says a credible study must compare.
- **The NTU-frozen cell is mechanism, not confound-fixing (new).** The one genuinely missing cell (NTU-pretrained + **frozen**, no fine-tune) answers a *secondary* question — "can you skip the 4-subject fine-tune and just freeze the prior?" — and, by contrast with NTU-finetuned, shows *why* the prior washes out (the 80-epoch fine-tune on 4 subjects overwrites it). Report it as explanation, not as the deciding comparison. Same script (`temp_xsens_to_xsens_loso_calibration.py` with NTU init).
- **Gap levels as a controlled variable.** Within-domain (NTU→NTU, Xsens→Xsens) vs cross-domain (NTU→Xsens) vs gap-manipulated (three preprocessings on one dataset — **v2** positions / local / swing, spanning MMD² 0.0092→0.0109→0.0322 for supmae, a ~3.5× swept range — see §7.5). Report every objective at every gap level.

### 7.2 Statistics that survive n=5 (defends every table) — **(new)**
- Multi-seed (3–5 seeds) at the discriminating ks (k∈{0,1,3}); ceiling at k≥5 (94–98%) means only low-k separates methods — say so and de-emphasize k≥5 as primary.
- **Clip-level McNemar per subject** (~400–500 paired clips/subject) → report "significant in x/5 subjects" rather than fighting the n=5 Wilcoxon floor (p=0.0625).
- Per-subject deltas + effect sizes + CIs as descriptive companions. Per-subject appendix (transparency reads as rigor at a journal).
- Selection rule stated: never best-test-epoch as primary; hyperparameters (λ, mask ratio) chosen on same-domain/validation and frozen. λ becomes a "robustness across λ" curve, not a magic-value hunt.

### 7.3 Mechanism (Pillar 3 / C3) — **(new)**
- **Layer-wise CKA** NTU-vs-Xsens features (scale-invariant → fixes the MMD scale confound). Script unwritten; highest-value missing analysis.
- **UMAP** of source vs target embeddings per objective (qualitative companion).
- **Symmetric gap test:** rerun MMD/CKA on *swing-projected NTU* so the twist-removal is applied to both sides (current swing MMD is asymmetric — Xsens-only). Plus median-heuristic sigma and bootstrap CIs.
- **Transfer-gap correlation:** per-objective (same-domain acc − cross-domain acc) vs CKA similarity — operationalizes "mismatch predicts transfer."

### 7.4 Novel-class onboarding (C5) — partly **(new)**
- OOV run is complete (22 classes × 5 subjects × 4 methods × k). Add **per-action statistics** and the **distinctiveness analysis**: correlate each gesture's inter-class kinematic separability (centroid distance / silhouette in encoder space) with its few-shot OOV recall. This turns the heatmap into a predictive finding and feeds the controller's safe-command assignment (crossarms/squat/wave reliable; throw/jump/hop not).

### 7.5 The gap-manipulation lever: v2 / local / swing = three gap regimes on one dataset — **(updated 2026-07-04)**
The headline mechanism. Three preprocessings of the *same* Xsens data span a swept NTU↔Xsens gap, and the prior-vs-no-prior result tracks it monotonically (supMAE−scratch @ k=1):

| regime | how | MMD² (supmae) | prior benefit k=1 |
|---|---|---|---|
| **v2** (chosen) | rebuild Xsens from mvnx positions through NTU's own shortest-arc construction ([[position-reconstruction-v2]]) | **0.0092** | **+4.1 pp** |
| local (v1) | measured orientation, parent-relative | 0.0109 | +3.0 pp |
| swing | fixed-axis twist strip on Xsens only | 0.0322 | −1.3 pp (washes out) |

Tighten the gap → the prior helps more; widen it → it washes out. This is a *causal* demonstration via a controllable knob, not a correlation across datasets — directly answering the Undermind report's "systematically varied domain gap" demand. **v2 is the paper's primary preprocessing** (real, directional gap reduction *below* baseline local); swing is the gap-widening arm of the same lever; local is the midpoint.

Honesty debts — **status**:
- **The original swing hypothesis was falsified** (swing was meant to *close* the gap; MMD *grew* ~3×). Keep as a finding: surface preprocessing does not do what intuition predicts — evidence for the necessary-not-sufficient nature of feature-distance metrics. v2 is what actually closes the gap, and it does so by matching NTU's *construction*, not by post-hoc twist removal.
- **sub8 — RESOLVED.** The −19.9 pp swing collapse was a twist-stripping artifact: under v2, sub8 supMAE−scratch @ k=1 = −1.0 pp (a tie). No longer a blocker; report as evidence that the swing anomaly was preprocessing-induced, not a real subject/mounting effect.
- **Make the gap measurement symmetric** — done for swing (symmetric-swing MMD ≈ asymmetric, confirming twist-strip damaged Xsens). Still add CKA (§7.3) on v2 vs local vs swing to show the gap ordering holds under a scale-invariant metric.
- **mae is weakest under v2** (77.0 @ k=1, below scratch's 79.9) — new, minor: reconstruction-only lost ground when the gap tightened. One discussion line; consistent with hybrid > pure-reconstruction, not a threat.

### 7.6 Scope decisions to pre-empt reviewers
- **Contrastive:** already run and empirically scoped — SupCon underperforms scratch at every k>0 → keep as the negative-transfer exhibit (C2), not as a missing baseline. One paragraph justifying "why not more contrastive."
- **DANN:** local-mode results exist (targetSupDANN ≈ supMAE); swing DANN only if swing becomes the chosen preprocessing. Otherwise report local-mode DANN as a domain-adaptation reference point.
- **Simulation baselines (IMUTube/Video2IMU):** cite as the alternative paradigm; explicitly out of scope (we study reusable encoders, not synthetic-data augmentation).

---

## 8. Controller pillar (earns THMS) — design spec

Abstract event-driven controller, **not** a physics sim (uncertainty→reliability is the scientific content; a physics engine adds risk, no signal). Build **only after recognizer numbers are final.**

- **Task:** ordered command sequence over a mode-switching pick-place mission. Map a subset of the 22 gestures → primitives {next, previous, approach, grasp, release, confirm, cancel}. Sequential + asymmetric so errors compound (a false `grasp` costs more than a false `next`) — this makes accuracy→task-success nonlinear and justifies "a few points matter."
- **Posterior stream:** for each mission step, sample a held-out-subject clip of the intended class, run the *actual* recognizer, feed its posterior to the FSM. Near-zero new ML.
- **Safety layer:** temporal smoothing, confidence threshold with reject option, dwell-time filter, mode manager with error recovery + safe-home. Held-out non-target gestures = distractor/null stream for false-activation rate.
- **Metrics:** task success, time-to-completion, false-activation rate, corrective-command count, rejection rate, command latency — reported across the 4 inits × k.
- **One sweep:** confidence-threshold → safety/throughput tradeoff curve (very THMS).
- **Design guards:** tune difficulty + thresholds on validation not test; assign safety-critical commands to reliably-recognized gestures (per the OOV distinctiveness finding — knits OOV + controller pillars). Declare synthesized inter-command timing in limitations.

---

## 9. Paper structure (target: IEEE THMS regular paper)

1. **Introduction** — see `paper_intro.md` (intro→background→previous work→gap→contributions).
2. **Related Work** — the competitor grid (§4) organized on three axes: cross-modal configuration, pretraining objective, domain-gap setting.
3. **Shared representation & method** — Local Relative Quaternions (17 segments × 4 quat = 68-d @ 30 Hz), the Transformer (`KinematicEncoder`; a DSTformer variant exists but was used only in early, unreported experiments) encoder, the objective zoo, the LOSO k-shot calibration protocol.
4. **Objective × gap characterization** — Pillars 1–2, C1–C2 (within-domain spread → cross-domain compression → negative transfer).
5. **Mechanism** — CKA/UMAP/transfer-gap (C3), gap-manipulation robustness (C4).
6. **Novel-class onboarding** — OOV + distinctiveness (C5).
7. **From recognition to control** — controller pillar (C6).
8. **Discussion & limitations** — n=5 stated openly; single dataset pair; synthesized control timing.
9. **Conclusion.**

---

## 10. Rejection risks & mitigations

| Risk | Reviewer line of attack | Mitigation |
|---|---|---|
| **Small N (5 subjects)** | "Underpowered, not generalizable" | Own it in the abstract; per-subject appendix; clip-level McNemar per subject; multi-seed; frame as depth-over-scale (THMS-appropriate) |
| **Single dataset pair (NTU↔Xsens)** | "One gap point, no external validity" | Gap-manipulation (v2/local/swing, ~3.5× MMD swept on one dataset) turns it into a *controlled* variable with the prior's benefit tracking it monotonically; position as controlled study, not benchmark chase |
| **Mislabeled / cherry-picked results (the old spine)** | "supmae uses labels; k=5 cherry-picked" | Already fixed: relabel `supmae`→"target supervised+MAE"; report the true no-label `mae` frozen collapse as evidence; headline "matches within noise," not "wins" |
| **"Why not more/bigger contrastive or a foundation model?"** | UniMTS/IMU2CLIP exist | Scope paragraph: we study controlled objectives on deployment-realistic small data, not foundation-scale; SupCon negative-transfer result *is* the contrastive evidence |
| **Mechanism asserted not shown** | "You infer mismatch from NTU→NTU" | CKA + UMAP + transfer-gap correlation; explicitly state low-gap-necessary-not-sufficient with the SupCon counterexample |
| **Controller is a toy** | "No real robot" | Frame as abstract reliability analysis; the science is uncertainty propagation; real recognizer posteriors from held-out clips; asymmetric task + safety curve |
| **Ceiling at k≥5** | "Methods indistinguishable" | Report low-k as primary discriminator; state ceiling explicitly; controller shows even ceiling-level accuracy differences matter under compounding errors |

**Biggest single risk = the honesty of the reframe.** The same data supports a clean claim (conditional utility + redundancy + negative transfer) once we stop overclaiming a method win. Every table must survive the paired-stats and 2×2-completion bar before it becomes a paper claim.

---

## 11. Work plan / sequencing (deployment-framed)

1. **Lock the prior-vs-no-prior comparison — the load-bearing claim.** `supMAE` vs `scratch` and `supLP120` vs `scratch`, on the chosen **v2** preprocessing. Multi-seed **running** (seeds 42/43/44 at k∈{0,1,3}, `loso_v2_multiseed.log`) → then clip-level McNemar per subject + per-subject deltas/CIs. The v2 +4.1 pp k=1 delta (4/5 subj positive) carries the deployment thesis and is single-seed until the sweep lands — make it bulletproof first. (§7.1, §7.2)
2. **Preprocessing DECIDED = v2.** ✓ v2 chosen (gap below local, prior benefit restored + amplified, sub8 anomaly resolved, supLP120 negative transfer fixed). Swing = the gap-widening arm of the lever; local = midpoint. Remaining: CKA on v2/local/swing to confirm the gap ordering under a scale-invariant metric (§7.3). (§7.5)
3. **Write CKA; run CKA + UMAP + symmetric MMD with CIs** — the representation mechanism and the clean measured gap axis. (§7.3)
4. **NTU-frozen cell** — mechanism follow-up explaining *why* the prior washes out (fine-tune overwrites it). Not a blocker for the headline. (§7.1)
5. **OOV per-action stats + distinctiveness correlation.** (§7.4)
6. **Freeze recognizer numbers → build controller** (each step human-OK'd per RESEARCH_LOG rule 3). (§8)
7. Draft in the §9 order; `paper_intro.md` is the intro seed.

## 12. Open decisions for the human

- ~~Local vs swing framing~~ **DECIDED (2026-07-04): v2 primary.** Three preprocessings = three gap regimes; the prior's benefit tracks the gap monotonically (+4.1/+3.0/−1.3 pp k=1 for v2/local/swing). v2 is primary (tightest gap, prior helps most); local + swing are the other two points on the swept lever. sub8 diagnosis resolved (swing artifact, dead under v2). See §7.5.
- Whether the controller ships in the same paper or as a companion (THMS favors the integrated story; it is also the highest-effort pillar).
- Rename of `supMAE` in prose ("Sup+MAE" vs "hybrid recon+sup") for consistency with the cited vision original.
