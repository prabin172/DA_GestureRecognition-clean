---
type: concept
status: active
updated: 2026-07-20
---

# The downstream controller (C6 / THMS pillar)

**2026-07-20 — methodology finalized: fully randomized System Input assignment throughout.** An
earlier version of this page described a fixed, recall-ranked gesture assignment (used by the
prototype and by two of the three robustness locks) and a resulting disagreement between locks
about which pretraining objective compounds worst. That fixed-assignment approach has been
**fully replaced**: `scripts/controller/controller_robust.py` now draws **120 independently,
uniformly-random 7-gesture System Input assignments** and evaluates all three locks over that
same shared set — no assignment anywhere in the reported numbers is chosen by recall, gesture
semantics, or any other property. Under this design, **all three locks agree**: `mae` compounds
worst under every one. See §9 and [[phase3-controller]] for the numbers. The old fixed-vocab run
is kept only as `trained_models/Phase3-controller/robust-fixedvocab-superseded/` for the record —
not cited anywhere, not part of the paper's narrative.

## What this is actually trying to prove (read this first)

**The hypothesis, stated plainly:** does the choice of pretraining objective matter *beyond* the raw accuracy table — specifically, do (a) small accuracy differences compound into much larger differences once recognition is used to drive a sequence of actions, and (b) does calibration quality (not just accuracy) govern how safely/quickly such a system can be operated? R1–R4 already show accuracy differences between objectives shrink to a few points once the domain gap compresses them, and that calibration differs even when accuracy doesn't. The controller exists to answer: **do those few points, and that calibration difference, actually matter for anything a human would care about?** It answers this by taking the recognizer exactly as trained (no new ML) and dropping it into a hypothetical sequential control task, then measuring whether the small upstream differences turn into large downstream ones (they do — R5). It is a stress-test harness for the recognizer's numbers, not a robot-building exercise.

**Terminology revision (2026-07-16):** earlier drafts of this page (and of `paper_method.md`/`paper_results.md`) named the seven task primitives after a hypothetical pick-and-place task — `grasp`, `release`, `next`, `previous`, `approach`, `confirm`, `cancel` — and disclosed, correctly, that none of these are gestures anyone performed. That disclosure was necessary but the names themselves were still a liability: a reader skimming for the physical-action words could walk away with a claim never made (that this dataset contains real grasp/release gestures). `paper/method.tex` and `paper/results.tex` were revised to drop physical-action naming entirely — the seven task primitives are now called **System Inputs**, two of them (the ones a wrong prediction into/out of causes immediate task failure) are the **Safety-Critical States**, and the remaining five (where a wrong prediction is recoverable at a cost) are the **Routine States**. The task itself is the **Sequential Control Task**, not a "mission" or "pick-and-place" scenario. This page, `paper_method.md`, `paper_results.md`, and `live_study_protocol.md` are rewritten below to match — bringing the markdown sources and this wiki page in line with the tex, which had drifted ahead of them undocumented. **Nothing about the underlying simulation, code, or locked numbers changed** — `scripts/controller/controller_sim.py`/`scripts/controller/controller_robust.py` still use the literal strings `grasp`/`release`/`next`/etc. as internal Python identifiers (renaming them would require re-running and re-locking `trained_models/Phase3-controller/robust/`, which was deliberately not done); only how the abstraction is *named and described* changed. Where this page points to code, it still cites those literal identifiers alongside the new conceptual names.

The paper's sixth contribution and the reason the venue is THMS (human–machine systems) rather than a pure-sensing venue: a chain from **pretraining objective → calibration quality → task-level reliability**. Everything upstream ([[pretraining-objectives]], [[loso-protocol]], [[phase1-mcnemar-ece-cka]]) produces a recognizer and its per-clip posteriors; the controller is what turns "a few accuracy/calibration points" into a human-machine-systems consequence (task success, safety, throughput) instead of leaving it as an abstract accuracy table.

**One methodology is reported: the robustness protocol**, `scripts/controller/controller_robust.py`. It is the source of every claim/conclusion in the paper (mae compounds worst; calibration governs safety/throughput), evaluated across 120 randomized System Input assignments, three robustness locks, and a tuning-free operating point (§9). A single-configuration prototype script (`scripts/controller/controller_sim.py`) also exists in the codebase and is useful for quick manual checks, but **no reported number in the paper comes from it** — it is not part of the narrative below.

Findings/locked numbers live in [[phase3-controller]] — this page is the design: what the controller is, why every piece is shaped the way it is, and what it is meant to stand in for.

## 1. Why a controller at all, and why *abstract*

The recognizer tables elsewhere in the paper show accuracy differences of a few percentage points between pretraining objectives once the domain gap compresses them (R1). The reviewer-facing risk is "so what — a few points don't matter." The controller's job is to show that they *do*, by putting the recognizer in a loop where errors have consequences that compound (§3) and where a wrong action in a Safety-Critical State is categorically worse than a wrong action in a Routine State (§4).

It is deliberately **not** a physics simulator or a real robot. `paper_method.md` §8 states the rationale directly: *"the scientific content is uncertainty→reliability propagation; a simulator would add risk without signal."* A physics engine would introduce its own approximation error (contact dynamics, actuator noise, timing) that has nothing to do with the recognizer being studied — it would dilute the causal chain the paper is trying to isolate, not strengthen it. The controller is an event-driven finite-state machine: it consumes real recognizer output and asks "does this System Input sequence complete," nothing about how a gripper would actually move.

The other deliberate choice is that it runs on **real held-out posteriors**, not synthetic error models. Every "prediction" the FSM sees during a Monte Carlo trial is a resampled softmax output from an actual held-out-subject LOSO clip (`scripts/controller/controller_sim.py:43-51,90-95`; `scripts/controller/controller_robust.py:56-71` for the robust version's packed-array form). This means the controller's task-success numbers are exactly as trustworthy as the recognizer's own accuracy and calibration — there is no synthetic noise model to second-guess, and the confidence values the safety layer thresholds on (§4) are the recognizer's actual (optionally temperature-scaled) softmax confidences, the same numbers reported in R4a's ECE analysis.

## 2. System Input assignment — fully random, no property of the data involved

**None of the 22 recorded Xsens gestures is, or names, a System Input.** The full recorded set is: `airkick, airpunch, bow, brushteeth, buttkicks, crossarms, crosstoe, drink, highfive, hop, jump, pickup, pushchair, runonspot, sidekick, sit, squat, stand, throw, turnaround, walk, wave` (`trained_models/LOSO-fullTrainCalibrate-v2/label_map.json`). Seven of these are picked out and *relabeled* as System Inputs for a hypothetical Sequential Control Task on every one of 120 independent draws — no subject was ever asked to perform any of the seven task actions, and (per the terminology revision above) those seven actions are no longer even given physical-sounding names: two become the **Safety-Critical States**, the other five the **Routine States**.

**The assignment rule is a uniform random draw, full stop** (`scripts/controller/controller_robust.py:182-186`): `rng.choice(all_ids, size=7, replace=False)`, repeated 120 times with `--vocabs`. No gesture's recall, confidence, or any other recognizer-derived property enters the draw. This is deliberately stronger than an earlier design (superseded 2026-07-20) that ranked gestures by pooled recall and assigned the two most-reliable ones to the Safety-Critical slots — that recall-ranked approach is retired entirely; every reported number in the paper now comes from the fully random draw described here.

**Why do this at all, rather than pick 7 gestures that already look like plausible commands, or rank them by reliability?** Because the paper's question is about the recognizer's *statistical* properties (accuracy, calibration) propagating to task outcomes — not about whether any particular gesture is intuitive for teleoperation, and not about a specific, defensible-looking task design that a reviewer could still call a choice. A uniform random draw, repeated 120 times and reported as a distribution (§9), removes that choice entirely: no design decision, by us or by the recognizer's own behavior, determines which gestures end up safety-critical. The hypothetical Sequential Control Task is a *vehicle* for studying error compounding and calibration (§3–§4) — it is explicitly not a claim that this dataset contains any specific real-world command gesture set, which is also why (per the terminology revision) the task no longer even borrows physical-action names.

## 3. The Sequential Control Task — why errors compound

A fixed ordered 12-step task, expressed as primitive names in the prototype and as primitive indices in the robust version (identical sequence; code still uses the original implementation labels):

```
next, next, previous, approach, grasp, confirm, next, approach, grasp, release, confirm, cancel
```
(`scripts/controller/controller_sim.py:39-40`; `scripts/controller/controller_robust.py:44-45` as `[0,0,1,2,3,5,0,2,3,4,5,6]` against `PRIMS = [next,previous,approach,grasp,release,confirm,cancel]`)

It contains **three Safety-Critical steps** — two into Safety-Critical State 1, one into Safety-Critical State 2 — deliberately: this is what makes recognition differences *compound* rather than average out. A task with only one safety-critical step would let a single lucky/unlucky recognition event dominate the outcome; three safety-critical steps in one 12-step sequence means the probability of a clean task run is a product of per-step reliabilities, which is exactly the nonlinearity the paper wants to demonstrate ("a few accuracy points" → a much larger task-success gap). This structural choice is what produces R5's headline finding: mae's few-point recognition gap becomes a much larger task-success gap, confirmed across 120 randomized task designs (§9).

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

## 8. Prototype script (`scripts/controller/controller_sim.py`) — not used for any reported number

A single-configuration script exists in the codebase for quick manual sanity checks: one System Input assignment, one cost model, one τ sweep, 4 methods (supcon not included), 3000 Monte Carlo trials per (method,k,τ) cell. It is **not cited anywhere in the paper** and no claim rests on its output — a single fixed assignment and a single cost model are each a choice a reviewer could call cherry-picked, which is exactly why §9's protocol exists instead. Documented here only so a reader of the code isn't confused by its presence; skip to §9 for the actual methodology.

## 9. Robustness protocol (`scripts/controller/controller_robust.py`) — the only methodology reported

Rather than freeze the three design knobs (System Input assignment, error-cost/critical-failure model, operating threshold) at defensible-but-arbitrary values, the robustness protocol probes all three with a single Monte Carlo design: **120 independently, uniformly-random 7-gesture System Input assignments** (`--vocabs`, default 120; `scripts/controller/controller_robust.py:182-186`), each evaluated with **1000 Monte Carlo task-execution trials per (assignment, method, k, τ/C_crit) cell** (`--missions`, default 1000). No assignment is chosen by recall, gesture semantics, or any other property (§2) — every assignment is a fresh unweighted random draw, and (as of 2026-07-20) **all three locks below share this exact same set of 120 assignments**, generated once and reused (`vocab_ids_list`, `scripts/controller/controller_robust.py:187-191`). Three locks apply this shared randomized design to three different design knobs, each aimed at a specific reviewer objection:

### Lock 1 — base outcome model across the 120 random assignments
*Kills: "you cherry-picked easy/hard gestures for the safety-critical slots."*
Every method is simulated at both τ=0 (full compounding) and τ=0.9 (gated), k∈{1,3}, across all 120 random assignments. Output is not a single number but a **distribution**: `vocab_sweep.csv` (raw per-assignment results) and `vocab_ordering.csv` (pairwise method-vs-method deltas across the 120 assignments — median, IQR, and the fraction of assignments where one method's hard_success ≥ another's, `scripts/controller/controller_robust.py:200-218`).

### Lock 2 — critical-cost sweep, same 120 assignments
*Kills: "the harsh instant-task-failure rule for critical errors drives your result."*
`simulate()` (`scripts/controller/controller_robust.py:69-108`) computes **both outcome models in a single pass**:
- **hard_success** — binary: any critical error (§4 step 5, safety branch) or task-abort (§4 step 2) → failure. This is what Lock 1 reports.
- **mean_cost** — soft: a critical error is *recoverable* but incurs a swept penalty `C_crit` added to the running task cost, rather than ending the task outright.

`C_crit` is swept over `{2, 5, 10, 20, 50, 1e6}` (the last effectively recovers hard-safety) at each of the same 120 random assignments (`scripts/controller/controller_robust.py:227-244`), τ=0. Output is `costmodel_sweep.csv` (per-assignment) and `costmodel_summary.csv` (median/IQR of mean_cost across the 120 assignments per method×k×C_crit).

### Lock 3 — iso-safety operating point, same 120 assignments
*Kills: "you tuned τ to make your preferred method win."*
Rather than pick one τ and compare success rates there, fix a **false-activation budget** (1% or 0.5%) and, per method per assignment, find the *smallest* τ that meets it (`scripts/controller/controller_robust.py:257-282`) — a deployment-standard way to set an operating threshold that requires no tuning against the outcome metric at all. Output is `frontier.csv` (per-assignment τ-sweep) and `iso_safety.csv`/`iso_safety_summary.csv` (per-assignment and aggregated τ*/success/cost across the 120 assignments).

### What the three locks jointly establish
Across 120 randomized System Input assignments × 2 outcome models (hard-success, soft-cost) × a 6-point critical-cost sweep × a tuning-free operating point — **all evaluated over the identical shared set of random assignments** — two claims survive every knob: **(1) mae compounds worst** (Lock 1: lowest task-success in all 4 conditions; Lock 2: highest median cost at every C_crit; Lock 3: lowest task-success at both k and both false-activation budgets tested) and **(2) calibration governs the safety/throughput trade** (the two best-calibrated inits, supLP120 and supcon, lead Lock 3's task-success at k=3). See [[phase3-controller]] for the exact numbers.

## 10. A secondary, honest finding (does not change the ranking)

supLP120 (the best-calibrated objective, R4a) has a **confident false-critical-activation mode**: because its confidence stream is well-calibrated and sharply separates classes, it sometimes maps an unrelated gesture onto a highly-separable Safety-Critical-State anchor confidently — enough to clear even a moderate τ. Even under full 120-assignment randomization, this shows up as a narrow dip: supLP120 falls just below scratch on the **ungated** Lock 1 metric at k=1 (0.472 vs 0.516), and trends as the second-costliest method (behind mae) as Lock 2's critical-cost penalty grows severe. It never rises to displace mae as the worst-compounding objective under any lock. This does not contradict the calibration story — supLP120 remains the best-calibrated method on average (R4a) and leads Lock 3's task-success at k=3 — it is a secondary effect, visible only at specific operating points, treated as a design-relevant nuance rather than smoothed over (`paper_results.md` R5).

## 11. What the abstraction is meant to stand in for

Mapping the simulation's vocabulary back to what it represents, for a reader checking whether the abstraction is honest:

| controller element | stands in for | is it real or invented? |
|---|---|---|
| the 7 System Inputs (2 Safety-Critical States, 5 Routine States) | abstract action-classes of a hypothetical Sequential Control Task, deliberately given no physical-action names (2026-07-16 revision) | **invented** — no such gestures were recorded (§2) |
| the 7 gestures relabeled to those System Inputs, on each of 120 independent random draws | the recognizer's actual, real predictions | **real** — these are genuine recorded gesture classes and genuine recognizer output; only the *label* attached to them (§2) is invented, and which gestures get which label is itself randomized rather than fixed |
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
- **Per-subject session (~60–90 min):** suit up in the Xsens mocap suit (same hardware as `DataCollection/sub{7-11}/`) → record k=3 calibration shots live per gesture in **one fixed 7-input assignment**, drawn once (not re-derived per subject) so all subjects run the identical task design — **note (2026-07-20): since §2's assignment is now randomized rather than fixed offline, this live-study design needs a fresh decision on which single assignment to fix for a live session, or whether to randomize across subjects instead; not resolved here, flagged for whoever picks this protocol back up** → on-device head-only calibration fine-tune (same protocol/hyperparameters as `loso_fulltrain_calibration.py`'s k=3 stage: 30 epochs, AdamW, lr 1e-3 — cheap enough to run in near-real-time between suit-up and the task) → perform the same 12-step Sequential Control Task (§3) against a **screen-based simulated executor** (still no physical robot — same rationale as §1).
- **Conditions, within-subject, counterbalanced:** 2 inits (supLP120 vs scratch) × 2 operating points (ungated τ=0 vs the iso-safety τ\* from R5 Lock 3's 1% budget) = 4 conditions × ~3 task repetitions ≈ 12 task runs/subject.
- **Metrics:** task success, time-to-completion, false-activation count, rejection count, corrective-input count, plus NASA-TLX per condition (subjective workload — a standard THMS-reviewer expectation, not currently measurable from simulation).
- **Analysis:** paired within-subject comparisons; step-level McNemar across repetitions, reusing the same statistical machinery as R2's clip-level McNemar rather than inventing new tooling.

**What is not built yet (blocking prerequisites):**
- **Real-time inference path** (~1 week estimated build+pilot): Xsens MVN live stream → position-derived v2 quaternions (same transform as the offline `IMU_batch_processor.py` pipeline, [[position-reconstruction-v2]]) → `KinematicEncoder` forward pass → posterior → FSM step. Every piece except the live streaming glue already exists.
- **IRB/ethics confirmation** — not started; gesture studies of this kind are typically expedited/exempt but this must be confirmed with the institution first.
- **Inter-command timing** — the live study would be the first source of *measured* timing data (simulation synthesizes it).

**Open questions logged for the human before any approval** (`live_study_protocol.md` §5): whether to add `supMAE` as a 3rd init condition (best in the k=1 ungated condition per R5), whether N=5–6 is enough given the step-level McNemar power argument or whether the T6 release headcount goal should drive a larger N, who owns the IRB submission and on what timeline, and physical space/hardware logistics.

## 13. Gotchas

- **`robust/` is locked — never overwrite it.** `trained_models/Phase3-controller/robust/` holds the numbers cited by exact value throughout `paper_results.md` R5, produced by the fully-randomized protocol (§9). `scripts/controller/controller_robust.py` supports `--out-dir` specifically so any future re-run writes to a new directory instead of overwriting the locked one. If running this script, always check the target `--out-dir` doesn't already hold locked numbers.
- **`reliability_ordered_vocab()` still exists in the code but is no longer used by any lock (2026-07-20).** It's called once at startup purely to print a reference vocabulary for transparency; Locks 1/2/3 all draw their own independent random assignments instead. Historical note for anyone reading old commits: this function used to pool recall across every method present in the loaded posterior data (`k3.groupby("true_id")` over the whole unfiltered `df`), which meant adding a new method's posteriors to the loaded `RUNDIRS` could silently change which 7 gestures got assigned to which System Input for what were then Locks 2/3's fixed-vocab sweeps — this is exactly what happened when supcon was first added (2026-07-10), and is the reason the fixed-vocab design was retired in favor of full randomization. The superseded fixed-vocab output is kept at `trained_models/Phase3-controller/robust-fixedvocab-superseded/` for the record.
- **`--vocabs`/`--missions` control the shared Monte Carlo design.** Default 120 assignments × 1000 trials/cell, applied identically to all three locks (§9). Increasing either tightens the distributions at proportionally higher runtime cost (the full 120×1000 run across all three locks takes ~4.5 minutes on the current hardware).
- **A single-configuration prototype script exists** (`scripts/controller/controller_sim.py`, §8) for quick manual checks — not used for any reported number, and not kept in sync with the robust script's logic; don't expect them to match line-for-line.
- **Code identifiers still say `grasp`/`release`/etc.** — only the paper/wiki-facing names changed (2026-07-16). `scripts/controller/controller_sim.py`/`scripts/controller/controller_robust.py`'s `PRIMS`/`CRITICAL`/`MISSION` constants are untouched; do not expect the code to match this page's vocabulary literally.

## Related
[[phase3-controller]] (locked numbers + findings) · [[pretraining-objectives]] · [[phase1-mcnemar-ece-cka]] (ECE/calibration the controller consumes) · [[paper-framing]] (C6 pillar definition) · `paper/live_study_protocol.md` (T7, unapproved) · `paper_method.md` §8/§8.1 · `paper_results.md` R5
