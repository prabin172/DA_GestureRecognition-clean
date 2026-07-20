---
type: concept
status: active
updated: 2026-07-16
---

# The downstream controller (C6 / THMS pillar)

## What this is actually trying to prove (read this first)

**The hypothesis, stated plainly:** does the choice of pretraining objective matter *beyond* the raw accuracy table — specifically, do (a) small accuracy differences compound into much larger differences once recognition is used to drive a sequence of actions, and (b) does calibration quality (not just accuracy) govern how safely/quickly such a system can be operated? R1–R4 already show accuracy differences between objectives shrink to a few points once the domain gap compresses them, and that calibration differs even when accuracy doesn't. The controller exists to answer: **do those few points, and that calibration difference, actually matter for anything a human would care about?** It answers this by taking the recognizer exactly as trained (no new ML) and dropping it into a hypothetical sequential control task, then measuring whether the small upstream differences turn into large downstream ones (they do — R5). It is a stress-test harness for the recognizer's numbers, not a robot-building exercise.

**Terminology revision (2026-07-16):** earlier drafts of this page (and of `paper_method.md`/`paper_results.md`) named the seven task primitives after a hypothetical pick-and-place task — `grasp`, `release`, `next`, `previous`, `approach`, `confirm`, `cancel` — and disclosed, correctly, that none of these are gestures anyone performed. That disclosure was necessary but the names themselves were still a liability: a reader skimming for the physical-action words could walk away with a claim never made (that this dataset contains real grasp/release gestures). `paper/method.tex` and `paper/results.tex` were revised to drop physical-action naming entirely — the seven task primitives are now called **System Inputs**, two of them (the ones a wrong prediction into/out of causes immediate task failure) are the **Safety-Critical States**, and the remaining five (where a wrong prediction is recoverable at a cost) are the **Routine States**. The task itself is the **Sequential Control Task**, not a "mission" or "pick-and-place" scenario. This page, `paper_method.md`, `paper_results.md`, and `live_study_protocol.md` are rewritten below to match — bringing the markdown sources and this wiki page in line with the tex, which had drifted ahead of them undocumented. **Nothing about the underlying simulation, code, or locked numbers changed** — `scripts/controller/controller_sim.py`/`scripts/controller/controller_robust.py` still use the literal strings `grasp`/`release`/`next`/etc. as internal Python identifiers (renaming them would require re-running and re-locking `trained_models/Phase3-controller/robust/`, which was deliberately not done); only how the abstraction is *named and described* changed. Where this page points to code, it still cites those literal identifiers alongside the new conceptual names.

The paper's sixth contribution and the reason the venue is THMS (human–machine systems) rather than a pure-sensing venue: a chain from **pretraining objective → calibration quality → task-level reliability**. Everything upstream ([[pretraining-objectives]], [[loso-protocol]], [[phase1-mcnemar-ece-cka]]) produces a recognizer and its per-clip posteriors; the controller is what turns "a few accuracy/calibration points" into a human-machine-systems consequence (task success, safety, throughput) instead of leaving it as an abstract accuracy table.

Two implementations exist, in order of authority:
- **Prototype**, `scripts/controller/controller_sim.py` — one fixed configuration. Its exact numbers ARE cited — the "4pp recognition gap → 21pp task-success gap" headline in R5's opening "Illustratively" paragraph (supMAE 0.967/scratch 0.905/supLP120 0.861/mae 0.754) is this script's direct output, not a robustness-protocol number.
- **Robustness protocol**, `scripts/controller/controller_robust.py` — the load-bearing one for every *claim/conclusion* in the paper (mae compounds worst; calibration governs safety/throughput). The prototype supplies the one illustrative headline number; every generalizable claim beyond that single number comes from here, across 120 randomized System Input assignments / 2 outcome models / a tuning-free operating point (§9).

Findings/locked numbers live in [[phase3-controller]] — this page is the design: what the controller is, why every piece is shaped the way it is, and what it is meant to stand in for.

## 1. Why a controller at all, and why *abstract*

The recognizer tables elsewhere in the paper show accuracy differences of a few percentage points between pretraining objectives once the domain gap compresses them (R1). The reviewer-facing risk is "so what — a few points don't matter." The controller's job is to show that they *do*, by putting the recognizer in a loop where errors have consequences that compound (§3) and where a wrong action in a Safety-Critical State is categorically worse than a wrong action in a Routine State (§4).

It is deliberately **not** a physics simulator or a real robot. `paper_method.md` §8 states the rationale directly: *"the scientific content is uncertainty→reliability propagation; a simulator would add risk without signal."* A physics engine would introduce its own approximation error (contact dynamics, actuator noise, timing) that has nothing to do with the recognizer being studied — it would dilute the causal chain the paper is trying to isolate, not strengthen it. The controller is an event-driven finite-state machine: it consumes real recognizer output and asks "does this System Input sequence complete," nothing about how a gripper would actually move.

The other deliberate choice is that it runs on **real held-out posteriors**, not synthetic error models. Every "prediction" the FSM sees during a Monte Carlo trial is a resampled softmax output from an actual held-out-subject LOSO clip (`scripts/controller/controller_sim.py:43-51,90-95`; `scripts/controller/controller_robust.py:56-71` for the robust version's packed-array form). This means the controller's task-success numbers are exactly as trustworthy as the recognizer's own accuracy and calibration — there is no synthetic noise model to second-guess, and the confidence values the safety layer thresholds on (§4) are the recognizer's actual (optionally temperature-scaled) softmax confidences, the same numbers reported in R4a's ECE analysis.

## 2. System Input assignment and mapping — the relabeling, made explicit

**None of the 22 recorded Xsens gestures is, or names, a System Input.** The full recorded set is: `airkick, airpunch, bow, brushteeth, buttkicks, crossarms, crosstoe, drink, highfive, hop, jump, pickup, pushchair, runonspot, sidekick, sit, squat, stand, throw, turnaround, walk, wave` (`trained_models/LOSO-fullTrainCalibrate-v2/label_map.json`). Seven of these are picked out and *relabeled* as System Inputs for a hypothetical Sequential Control Task — no subject was ever asked to perform any of the seven task actions, and (per the terminology revision above) those seven actions are no longer even given physical-sounding names: two become the **Safety-Critical States**, the other five the **Routine States**.

**The actual illustrative assignment** (`trained_models/Phase3-controller/mapping.csv`, from the prototype run — the CSV itself still uses the original implementation labels, shown in parentheses below):

| real recorded gesture | relabeled as | pooled k=3 recall |
|---|---|---|
| squat | **Safety-Critical State 1** (`grasp` in code) | 0.9967 |
| bow | **Safety-Critical State 2** (`release` in code) | 0.9845 |
| wave | Routine State 1 (`confirm`) | 0.9663 |
| crossarms | Routine State 2 (`approach`) | 0.9634 |
| sit | Routine State 3 (`cancel`) | 0.9391 |
| crosstoe | Routine State 4 (`next`) | 0.9370 |
| pickup | Routine State 5 (`previous`) | 0.9288 |

The assignment rule is **recall rank, full stop** — the most reliably-recognized gesture (`squat`, 99.67% recall) is relabeled Safety-Critical State 1 (the state we chose to make most safety-critical) simply because it is the top-ranked gesture, not because a squat looks like a grasping action. The clearest evidence the rule is non-semantic: `pickup` — the one gesture in the whole 22-class set whose *name* actually suggests a grasping action — is **not** the one relabeled to a Safety-Critical State. It ranks 7th by recall and lands on Routine State 5 instead. If the mapping were semantic, `pickup` would be the obvious choice for the "grasp-like" slot; it isn't used that way, which is the honest evidence that this is a reliability-driven relabeling exercise, not a claim that any recorded gesture *is* a real-world action of any kind.

**Why do this at all, rather than pick 7 gestures that already look like plausible commands?** Because the paper's question is about the recognizer's *statistical* properties (accuracy, calibration) propagating to task outcomes — not about whether any particular gesture is intuitive for teleoperation. Tying System Input identity to recall rank, rather than to human intuition about gesture meaning, is what makes the mapping **method-agnostic**: it is constructed from properties pooled over all pretraining objectives (below), so no single objective's own strengths get to pick which gestures are safety-critical. The hypothetical Sequential Control Task is a *vehicle* for studying error compounding and calibration (§3–§4) — it is explicitly not a claim that this dataset contains any specific real-world command gesture set, which is also why (per the terminology revision) the task no longer even borrows physical-action names.

The mapping mechanism itself is deliberately **method-agnostic** so no single pretraining objective's own strengths bias which gestures get the safety-critical roles:

**Prototype's illustrative mapping** (`build_mapping`, `scripts/controller/controller_sim.py:54-74`):
1. Pool recall over **all four methods** at k=3 (not per-method — this is the "method-agnostic" property) per gesture, from the loaded posterior set.
2. Exclude locomotion/whole-body-transition classes (`walk, runonspot, turnaround, buttkicks, hop, jump, stand`) — these make poor discrete System Inputs regardless of recognition quality.
3. Rank the remaining gestures by pooled recall, most-reliable first.
4. Assign the seven System Inputs in a fixed priority order (implementation order: `grasp, release, confirm, approach, cancel, next, previous`) — i.e. the two Safety-Critical States get the two most reliably-recognized gestures.

**Robustness protocol's equivalent** (`reliability_ordered_vocab`, `scripts/controller/controller_robust.py:139-152`) does the same thing but is also the artifact reused by Locks 2 and 3 (§9) — see the coupling quirk in §13 for a real bug this caused.

This "assign the Safety-Critical States to the most reliably-recognized gestures" rule is a stated **design guard**, not incidental — `paper_idea.md` §8 calls it out explicitly (there, still in its original pick-and-place framing, since that document is a historical design note and not rewritten), and it is the mechanism behind the "honest finding" in §10 (supLP120's confident-false-activation mode on exactly these high-confidence anchor gestures).

## 3. The Sequential Control Task — why errors compound

A fixed ordered 12-step task, expressed as primitive names in the prototype and as primitive indices in the robust version (identical sequence; code still uses the original implementation labels):

```
next, next, previous, approach, grasp, confirm, next, approach, grasp, release, confirm, cancel
```
(`scripts/controller/controller_sim.py:39-40`; `scripts/controller/controller_robust.py:44-45` as `[0,0,1,2,3,5,0,2,3,4,5,6]` against `PRIMS = [next,previous,approach,grasp,release,confirm,cancel]`)

It contains **three Safety-Critical steps** — two into Safety-Critical State 1 (`grasp`), one into Safety-Critical State 2 (`release`) — deliberately: this is what makes recognition differences *compound* rather than average out. A task with only one safety-critical step would let a single lucky/unlucky recognition event dominate the outcome; three safety-critical steps in one 12-step sequence means the probability of a clean task run is a product of per-step reliabilities, which is exactly the nonlinearity the paper wants to demonstrate ("a few accuracy points" → a much larger task-success gap). This structural choice is what produces R5's headline number: a ~4pp recognition gap (mae vs supMAE) becomes a 21pp task-success gap under the prototype's fixed configuration.

## 4. FSM mechanics — one task step

For each step in the task with intended System Input `c`:

1. **Sample.** Draw a random held-out clip whose *true* label is the gesture mapped to `c`, read its recognizer confidence and predicted class (`sample_pred`, `scripts/controller/controller_sim.py:90-95`; robust version inlines the equivalent array lookup in `simulate`, `scripts/controller/controller_robust.py:88-93`).
2. **Confidence-threshold reject.** If confidence < τ, the step is rejected (re-prompt, cost `T_REJECT`), and the loop retries. `MAX_REJECT` (5) consecutive rejects on one step aborts the whole task — modeling a real interface giving up on an unresponsive user/sensor rather than looping forever (`scripts/controller/controller_sim.py:103-111`).
3. **Dwell/temporal-smoothing (prototype only, opt-in via `--dwell`).** If `dwell > 1`, the step additionally requires `dwell-1` more consecutive above-threshold reads that *agree* on the same predicted System Input before accepting — a discrete stand-in for the kind of temporal smoothing a real streaming recognizer would use (`scripts/controller/controller_sim.py:112-122`). Not used in any locked/reported result (default `dwell=1`); documented here because it exists in the code and is part of the design space `paper_idea.md` §8 originally scoped ("temporal smoothing, dwell-time filter").
4. **Correct input.** If the accepted prediction's System Input equals the intended one, the step succeeds at cost `T_EXEC`, FSM advances.
5. **Wrong input — the safety branch.** If the accepted prediction's System Input differs from intended:
   - If **either** the intended System Input **or** the predicted System Input is one of the two Safety-Critical States (`CRITICAL = {"grasp", "release"}` in code) — i.e. a wrong input *into* or *out of* a Safety-Critical State — the task **fails immediately** (this is the unsafe-action case: either the user wanted a critical action and got something else, or the system did something critical the user didn't ask for).
   - Otherwise the error is **recoverable**: cancel-and-reissue at cost `T_CORRECT` (3×), then the step is considered resolved and the FSM advances.
   (`scripts/controller/controller_sim.py:123-130`; robust version's `simulate`, `scripts/controller/controller_robust.py:100-104`, computes both a hard binary outcome and a soft-cost outcome from this same branch in one pass — see §9 Lock 2.)

## 5. Cost model

| constant | value | meaning |
|---|---|---|
| `T_EXEC` | 1.0 | one gesture attempt (a normal step) |
| `T_REJECT` | 1.0 | low-confidence rejection → re-prompt |
| `T_CORRECT` | 3.0 | recoverable wrong input → cancel + reissue |
| `MAX_REJECT` | 5 | consecutive rejects on one step before task abort |

(`scripts/controller/controller_sim.py:31-34`; identical values passed as `costs = (1.0, 1.0, 3.0, 5)` in `scripts/controller/controller_robust.py:178`.)

The cost model is **asymmetric by construction** — a recoverable error costs 3× a clean step, and a critical error costs either "the whole task" (hard outcome) or a swept penalty `C_crit` (soft outcome, robust version only, §9 Lock 2) — specifically so that errors compound nonlinearly into task-level consequences rather than linearly averaging into a slightly-lower success rate. This is the mechanism, not just the task length (§3), that converts "mae has slightly lower accuracy" into "mae has a much lower task-success rate."

## 6. False-activation / distractor stream

Not every gesture is a System Input. Held-out clips whose *true* label is **not** one of the 7 task gestures form a distractor stream: for each, if the recognizer's confidence clears τ **and** its predicted class happens to be a System Input gesture, that counts as a **false activation** — the system acting on an input the user never intended to issue (`scripts/controller/controller_sim.py:136-144`; `false_activation`, `scripts/controller/controller_robust.py:116-128`). This is the safety-relevant complement to task success: a controller that never rejects (τ=0) will have high throughput but a high false-activation rate; raising τ trades throughput for safety. This tradeoff is the entire content of Lock 3 (§9).

## 7. Metrics reported

Per (method × k), swept over τ:
- **task success rate** — fraction of Monte Carlo task runs completed without failure/abort.
- **mean time-to-completion** — in cost units (§5), only over successful task runs.
- **rejection rate** — fraction of all presented reads that were below-threshold.
- **corrective-inputs-per-task** — mean count of recoverable cancel-and-reissue events.
- **false-activation rate** — §6.

The robust version additionally reports **mean_cost** (the soft-outcome dual of task success, §9 Lock 2) and, for Lock 1, the **distribution** of task success across randomized System Input assignments rather than a single number (§9 Lock 1).

## 8. Prototype run (`scripts/controller/controller_sim.py`) — illustration only

Single fixed configuration: the method-agnostic mapping (§2), the one named Sequential Control Task (§3), 4 methods (`scratch, mae, supMAE, supLP120` — supcon not in the prototype), k∈{1,3}, and a τ sweep `[0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]` (`scripts/controller/controller_sim.py:174`), 3000 Monte Carlo trials per (method,k,τ) cell, fixed RNG seed (default 7). Outputs → `trained_models/Phase3-controller/`: `controller_results.csv` (full sweep), `operating_point_summary.csv` (fixed τ=0.9 slice), `mapping.csv`, and three PNGs (success-vs-τ, safety/throughput frontier, success-by-method bar chart at the fixed operating point). Illustrative headline: at k=1, τ=0 (ungated), task-success is supMAE 0.967 / scratch 0.905 / supLP120 0.861 / mae 0.754 — the "4pp recognition gap → 21pp task-success gap" number quoted in the abstract and R5's opening.

This configuration is retained in the paper for readability, but **no claim rests on these exact numbers** — a single System Input assignment, a single cost model, and a single τ are each a choice a reviewer could call cherry-picked. That is what §9 exists to close off.

## 9. Robustness protocol (`scripts/controller/controller_robust.py`) — what's actually claimed

Rather than freeze the three design knobs (System Input assignment, error-cost/critical-failure model, operating threshold) at defensible-but-arbitrary values, the robustness protocol demonstrates the **method ordering is invariant** to each of them independently. Three locks, each aimed at a specific reviewer objection:

### Lock 1 — randomized System Input assignment
*Kills: "you cherry-picked easy/hard gestures for the safety-critical slots."*
Resample 7 distinct gesture IDs uniformly at random (not reliability-ranked) and assign them to the 7 System Inputs 120 times (`--vocabs`, default 120; `scripts/controller/controller_robust.py:184-201`, `make_prim_of_id:131-136`). For each random assignment, every method is simulated at both τ=0 (full compounding) and τ=0.9 (gated), k∈{1,3}, `--missions` (default 1000) Monte Carlo trials each. Output is not a single number but a **distribution**: `vocab_sweep.csv` (raw per-assignment results) and `vocab_ordering.csv` (pairwise method-vs-method deltas across the 120 assignments — median, IQR, and the fraction of assignments where one method's hard_success ≥ another's, `scripts/controller/controller_robust.py:205-223`). The claim "mae compounds worst" is stated not as a single-configuration fact but as "mae ≥ supMAE in only ~12–15% of the 120 random assignments at k=1" — a distributional claim, harder to attack as cherry-picked.

### Lock 2 — critical-cost sweep + two outcome models
*Kills: "the harsh instant-task-failure rule for critical errors drives your result."*
`simulate()` (`scripts/controller/controller_robust.py:74-113`) computes **both outcome models in a single pass** over the same Monte Carlo trials:
- **hard_success** — binary: any critical error (§4 step 5, safety branch) or task-abort (§4 step 2) → failure. This is what Lock 1 reports.
- **mean_cost** — soft: a critical error is *recoverable* but incurs a swept penalty `C_crit` added to the running task cost, rather than ending the task outright.

`C_crit` is swept over `{2, 5, 10, 20, 50, 1e6}` (the last effectively recovers hard-safety, since no plausible task cost approaches 1e6) at a **fixed named System Input assignment** (`reliability_ordered_vocab`, §2), τ=0, `--frontier-missions` (default 3000) trials per cell (`scripts/controller/controller_robust.py:232-244`). The claim is that mae has the highest mean_cost at *every* C_crit in the sweep — the ordering doesn't depend on how catastrophic a critical error is modeled as being.

### Lock 3 — iso-safety operating point
*Kills: "you tuned τ to make your preferred method win."*
Rather than pick one τ and compare success rates there (which invites exactly that objection), fix a **false-activation budget** (1% or 0.5%) and, per method, find the *smallest* τ that meets it (`scripts/controller/controller_robust.py:246-282`) — this is a deployment-standard way to set an operating threshold and requires no tuning against the outcome metric at all. Compare task-success and mean-cost **at each method's own budget-meeting τ\***, not at a shared τ. The full τ-frontier (task-success vs false-activation, swept over the same `TAUS` list as the prototype, `scripts/controller/controller_robust.py:47`) is also reported (`frontier.csv`) for the plot showing the whole tradeoff curve, not just the one operating point.

### What the three locks jointly establish
Across 120 random System Input assignments × 2 outcome models × a 6-point critical-cost sweep × a tuning-free operating point, two claims survive every knob: **(1) mae compounds worst** (highest cost/lowest success at essentially every configuration tried) and **(2) calibration governs the safety/throughput trade** (the best-calibrated init, supLP120, reaches a given false-activation budget at the lowest τ, hence completes tasks faster at the same safety level). See [[phase3-controller]] for the exact locked numbers.

## 10. The honest finding baked into the design

supLP120 (the best-calibrated objective, R4a) has a **confident false-critical-activation mode**: because its confidence stream is well-calibrated and sharply separates classes, it sometimes maps an unrelated gesture onto a highly-separable Safety-Critical-State anchor confidently — enough to clear even a moderate τ. This means on the **ungated** metric (τ=0) it can dip *below* scratch at k=1, which looks at first like calibration not helping. The controller's design is what surfaces this as informative rather than as noise: because Lock 3's iso-safety framing evaluates each method at *its own* safety-meeting τ rather than a shared one, the correct deployment framing (fix a safety budget, then compare throughput) is exactly the one under which calibration's value is visible — the naive ungated comparison *understates* it. This is treated in the paper as a design principle the controller demonstrates, not a caveat to explain away (`paper_results.md` R5, "Honest finding").

## 11. What the abstraction is meant to stand in for

Mapping the simulation's vocabulary back to what it represents, for a reader checking whether the abstraction is honest:

| controller element | stands in for | is it real or invented? |
|---|---|---|
| the 7 System Inputs (2 Safety-Critical States, 5 Routine States) | abstract action-classes of a hypothetical Sequential Control Task, deliberately given no physical-action names (2026-07-16 revision) | **invented** — no such gestures were recorded (§2) |
| the 7 gestures relabeled to those System Inputs (`squat`→Safety-Critical State 1, etc.) | the recognizer's actual, real predictions | **real** — these are genuine recorded gesture classes and genuine recognizer output; only the *label* attached to them (§2) is invented |
| FSM + System Inputs | a discrete gesture-driven command interface (e.g. assistive-robot teleoperation, AR/VR command layer) | the interface pattern is real-world standard; this specific task is hypothetical |
| confidence-threshold reject (τ) | any real recognizer's confidence-gated action-trigger — this is standard practice, not a paper-specific invention | real practice, applied here |
| Safety-Critical vs Routine State classification | the real distinction between an action that must be undone-on-error-detection (a Routine-State misfire) and one that cannot (a Safety-Critical-State action already executed) | the distinction is real; which two System Inputs are "critical" here is a modeling choice (§2), not derived from the data |
| distractor/false-activation stream | non-command motion in a real deployment session (fidgeting, conversation gestures, unrelated activity) that the system must not act on | real recorded clips (the other 15 gesture classes), real recognizer output |
| Monte Carlo over held-out posteriors | many independent deployment attempts, each drawing from the *actual* distribution of recognizer confidence/correctness the paper measured | real — no synthetic error model anywhere (§1) |
| iso-safety operating point | the standard way a real safety-critical system is *actually* commissioned — fix an acceptable false-trigger rate, then see what throughput you get | real-world standard practice, applied here |

**The one-sentence version:** every *number* the controller reports (confidence, correct/incorrect, task success, false-activation rate) comes from real recognizer behavior on real held-out data; only the *names* attached to which gesture means what System Input are invented, and that invention is disclosed and reliability-driven rather than hidden or semantic (§2) — and, as of the 2026-07-16 revision, no longer even dressed up in physical-action language.

What it explicitly does **not** claim to represent: that any recorded gesture is or resembles any real-world physical action, real inter-command timing (synthesized, not measured — declared as a limitation), real physical dynamics (deliberately excluded, §1), or any specific robot/task beyond the illustrative Sequential Control Task framing.

## 12. Planned extension: live human-in-the-loop study (T7 — NOT approved)

The controller as built is a *simulation* over already-collected held-out posteriors. `paper/live_study_protocol.md` (written 2026-07-09, **explicitly not approved** — discussion draft only, no building/recruiting/IRB contact without human sign-off) specifies how this pillar would convert into a small live validation, because two of its claims are stronger with human-subject evidence than with simulation alone:
1. Recognition differences compound into task-level differences (R5's headline).
2. Calibration governs the safety/throughput trade at iso-safety operating points (R5 Lock 3).

**Design as specified (subject to the open questions below):**
- **N = 5–6 new subjects**, never seen by any model — true cold-start LOSO, and doubles as a dataset-size expansion for the T6 release (5 → 10–11 subjects total).
- **Per-subject session (~60–90 min):** suit up in the Xsens mocap suit (same hardware as `DataCollection/sub{7-11}/`) → record k=3 calibration shots live per gesture in the **same fixed 7-input assignment** as the offline study (§2's mapping, fixed once from existing offline posteriors so it isn't re-derived per subject — keeps this a direct extension of Lock 1's assignment-robustness result, not a new variable) → on-device head-only calibration fine-tune (same protocol/hyperparameters as `loso_fulltrain_calibration.py`'s k=3 stage: 30 epochs, AdamW, lr 1e-3 — cheap enough to run in near-real-time between suit-up and the task) → perform the same 12-step Sequential Control Task (§3) against a **screen-based simulated executor** (still no physical robot — same rationale as §1).
- **Conditions, within-subject, counterbalanced:** 2 inits (supLP120 vs scratch) × 2 operating points (ungated τ=0 vs the iso-safety τ\* from R5 Lock 3's 1% budget) = 4 conditions × ~3 task repetitions ≈ 12 task runs/subject.
- **Metrics:** task success, time-to-completion, false-activation count, rejection count, corrective-input count, plus NASA-TLX per condition (subjective workload — a standard THMS-reviewer expectation, not currently measurable from simulation).
- **Analysis:** paired within-subject comparisons; step-level McNemar across repetitions, reusing the same statistical machinery as R2's clip-level McNemar rather than inventing new tooling.

**What is not built yet (blocking prerequisites):**
- **Real-time inference path** (~1 week estimated build+pilot): Xsens MVN live stream → position-derived v2 quaternions (same transform as the offline `IMU_batch_processor.py` pipeline, [[position-reconstruction-v2]]) → `KinematicEncoder` forward pass → posterior → FSM step. Every piece except the live streaming glue already exists.
- **IRB/ethics confirmation** — not started; gesture studies of this kind are typically expedited/exempt but this must be confirmed with the institution first.
- **Inter-command timing** — the live study would be the first source of *measured* timing data (simulation synthesizes it).

**Open questions logged for the human before any approval** (`live_study_protocol.md` §5): whether to add `supMAE` as a 3rd init condition (best in the k=1 ungated condition per R5), whether N=5–6 is enough given the step-level McNemar power argument or whether the T6 release headcount goal should drive a larger N, who owns the IRB submission and on what timeline, and physical space/hardware logistics.

## 13. Gotchas

- **`robust/` is locked — never overwrite it.** `trained_models/Phase3-controller/robust/` holds the numbers cited by exact value throughout `paper_results.md` R5. `scripts/controller/controller_robust.py` supports `--out-dir` (`scripts/controller/controller_robust.py:162-169`) specifically so any future method addition (e.g. the supcon extension) writes to a new directory (`robust-supcon/`) instead of overwriting the locked one. If running this script, always check the target `--out-dir` doesn't already hold locked numbers.
- **`reliability_ordered_vocab()` pools recall across every method present in the loaded posterior data, not per-method** (`scripts/controller/controller_robust.py:139-152`, `k3.groupby("true_id")` over the whole unfiltered `df`). Lock 1's randomized assignments are independent of this and unaffected, but Locks 2 and 3 both use this single named assignment — so adding a new method's posteriors to the loaded `RUNDIRS` silently changes which 7 gestures are assigned to which System Input for Locks 2/3 in any subsequent run. This is exactly what happened when supcon was added (2026-07-10): Lock 1 reproduced exactly, Locks 2/3 did not, because the named assignment shifted. Root-caused via code read, not a seeding/data bug — see [[phase3-controller]] for the full account. Not fixed in the script (would require filtering `reliability_ordered_vocab` to a canonical method subset) to avoid perturbing the already-locked run's provenance.
- **Prototype and robust script duplicate logic with slightly different data shapes** — the prototype filters a pandas DataFrame per draw (`by_class[gesture]`, simple but slower); the robust version pre-packs everything into numpy arrays keyed by (method,k,gesture) once (`pack()`, `scripts/controller/controller_robust.py:56-71`) because it needs to run ~120× more simulate() calls (assignments × k × τ × methods) and the pandas-per-draw approach would not scale. They are not meant to be diffed line-for-line — the robust version is the one whose numbers are ever cited.
- **The prototype's `dwell` option is unused in every reported result.** It exists in the code and is part of the original design-spec (`paper_idea.md` §8's "temporal smoothing, dwell-time filter"), but no locked run uses `dwell>1`.
- **Code identifiers still say `grasp`/`release`/etc.** — only the paper/wiki-facing names changed (2026-07-16). `scripts/controller/controller_sim.py`/`scripts/controller/controller_robust.py`'s `PRIMS`/`CRITICAL`/`MISSION` constants are untouched; do not expect the code to match this page's vocabulary literally.

## Related
[[phase3-controller]] (locked numbers + findings) · [[pretraining-objectives]] · [[phase1-mcnemar-ece-cka]] (ECE/calibration the controller consumes) · [[paper-framing]] (C6 pillar definition) · `paper/live_study_protocol.md` (T7, unapproved) · `paper_method.md` §8/§8.1 · `paper_results.md` R5
