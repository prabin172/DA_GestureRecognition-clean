# The downstream controller (C6 / THMS pillar) — advisor briefing
_Prepared 2026-07-20. Status: open methodological question, not a locked result. Prabin has not yet
put hands-on effort into this component — it was executed under supervision._

---

## Slide 1 — Why a controller exists at all

**The problem it answers:** elsewhere in the paper, accuracy differences between pretraining
objectives shrink to a few percentage points once the domain gap (skeleton → wearable IMU) is large.
A reviewer's obvious objection: *"so what — a few points don't matter."*

**What the controller does:** takes the recognizer exactly as trained — no new ML, no retraining —
and wires its real held-out predictions into a synthetic multi-step command task, where some steps
are safety-critical (wrong prediction → task fails outright) and others are routine (wrong prediction
→ costly but recoverable). Then it asks: do those "few accuracy points" turn into a much bigger
difference once errors can compound?

**What it deliberately is not:**
- Not a physics simulator, not a robot. It's an event-driven finite-state machine.
- Not a synthetic error model — every "prediction" it sees is a real recognizer output on a real
  held-out clip, resampled via Monte Carlo.

_Suggested figure: none needed — this is a concept slide. Could use a simple 3-box diagram:
[Recognizer] → [FSM / Controller] → [Task success / safety metrics]._

---

## Slide 2 — How the task is constructed

- 22 gestures were actually recorded. **7 of them are relabeled** as inputs to a hypothetical
  12-step command sequence; **2 of those 7** are marked "safety-critical," the other 5 "routine."
- **No subject ever performed a "grasp" or "release" gesture.** The 7 gestures are picked from the
  real 22 and relabeled purely by *how reliably the recognizer classifies them* (recall rank) —
  not by which gesture looks like a plausible command.
- **Proof this is not secretly semantic:** the one recorded gesture named `pickup` — the single
  gesture whose *name* suggests a grasping action — is specifically **not** one of the two
  safety-critical picks. It ranks 7th by recall and lands on a routine slot instead.
- The 12-step sequence deliberately visits the 2 safety-critical states **3 times** — this is what
  makes small recognition differences compound into large task-level differences, rather than
  averaging out.

_Suggested figure: table/diagram of the 7-gesture → System-Input mapping with recall values
(`trained_models/Phase3-controller/mapping.csv`) — a small table works well here. Optionally the
12-step sequence as a flow diagram with the 3 safety-critical visits highlighted._

---

## Slide 3 — Robustness design (why this isn't "pick a config that wins")

A controller like this has three knobs a reviewer could accuse of being tuned to produce a
favorable result: **which gestures are safety-critical, how harshly an error is penalized, and what
confidence threshold gates an action.** Three checks, one per knob:

| Lock | What it does | Kills the objection |
|---|---|---|
| **1 — Randomized assignment** | Re-draw which 7 gestures are the task inputs, 120 times at random; report the distribution, not one number | "You cherry-picked easy/hard gestures" |
| **2 — Cost-severity sweep** | Sweep how catastrophic a safety-critical error is (from mild recoverable penalty to near-infinite), 6 settings | "You picked a harsh failure rule to make your point" |
| **3 — Iso-safety threshold** | Instead of hand-picking a confidence threshold, derive it automatically per method from a fixed false-activation budget | "You tuned the threshold to win" |

_Suggested figure: the frontier plot (`trained_models/Phase3-controller/robust/frontier.csv` →
existing PNG) showing task-success vs. false-activation tradeoff curves, one line per method._

---

## Slide 4 — What we found (original, single-checkpoint run)

- Across all three locks, one objective (`mae`, pure reconstruction) was consistently the
  worst-compounding method — highest cost / lowest success at essentially every setting tried.
- The best-calibrated objective (`supLP120`) reached a given safety budget at the lowest confidence
  threshold — meaning it completes tasks faster at the same safety level. Calibration, not just
  accuracy, governs this tradeoff.
- Headline number quoted in the abstract: a ~4-point recognition gap became a ~21-point
  task-success gap under compounding.

_Suggested figure: bar chart, task-success by method at one operating point
(`trained_models/Phase3-controller/success_by_method.png` already exists)._

---

## Slide 5 — What changed after an independent full retrain (2026-07-20)

This repo's from-scratch reproducibility rerun retrained every pretraining checkpoint independently
(different hardware, same code, first time this pipeline ever ran in a pinned/reproducible
environment). Re-running the controller on the fresh checkpoints:

- **Lock 1 (randomized assignment, averaged over 120 draws): still shows `mae` worst.** This part
  reproduced.
- **Locks 2 and 3, and the single-configuration headline number — all of which use one *fixed*,
  reliability-ranked gesture assignment rather than Lock 1's 120 random draws — now show a
  *different* method, `supLP120`, as worst**, and by a wide margin under harsh penalties:

| Cost severity (C_crit) | mae | scratch | supLP120 | supMAE | supcon |
|---|---|---|---|---|---|
| mild (2) | 15.6 | 15.8 | 15.0 | 14.8 | 14.3 |
| moderate (20) | 17.8 | 17.9 | **19.3** | 16.0 | 14.8 |
| severe (50) | 21.5 | 21.3 | **26.4** | 17.9 | 15.6 |
| effectively-infinite | 123,015 | 115,016 | **237,681** | 63,681 | 27,014 |

(lower = better; `supLP120` degrades fastest as the penalty for a safety-critical error grows)

- **The abstract's specific "4-point → 21-point" number does not reproduce.** At the same
  configuration, task success is now: supcon 0.975 (best) → supMAE 0.927 → scratch 0.898 →
  mae 0.888 → **supLP120 0.790 (worst)**. `mae` is no longer the worst method in this view.

**This is not a bug.** It traces to a mechanism the original design already anticipated and
documented: `supLP120` (the best-calibrated objective on average) has a *confident
false-critical-activation* mode — it occasionally, confidently misclassifies an unrelated gesture
onto one of the highly-separable safety-critical anchors. On the freshly-retrained checkpoint this
specific failure mode is more pronounced, and it dominates once the safety-critical penalty is
severe. It does not contradict the calibration story (`supLP120` is still the best-calibrated
method on average) — it's a second, narrower failure mode that a harsher stress test exposed.

_Suggested figure: the same bar chart as Slide 4, side-by-side (original vs. rerun), to make the
flip visually obvious. Or the cost-severity table above as a simple line chart, one line per
method, x-axis = C_crit (log scale), y-axis = mean cost._

---

## Slide 6 — The open question (this is the actual discussion point)

**The three robustness checks — built specifically to make the "worst objective" claim
bulletproof — currently disagree with each other on which objective that is.**

- Randomized view → `mae` worst (on average, across many hypothetical task designs).
- Fixed, reliability-ranked view → `supLP120` worst (under the one specific, best-justified task
  design, and increasingly so as errors get more costly).

Both are legitimate, both are explainable, and reporting only one would be selective. Decisions
needed:
1. Report both explicitly as two different (and both true) claims, rather than picking one
   "worst" objective?
2. Treat the randomized view (Lock 1) as primary since it's what the paper already calls "the
   primary evaluation protocol" — and demote the fixed-view headline number out of the abstract?
3. Something else — e.g., is there a principled reason to trust one view over the other for a
   *deployed* system (which would use one fixed assignment, not 120 random ones)?

**Not yet done, in case it comes up:** the live human-in-the-loop extension exists only as an
unapproved discussion draft (`paper/live_study_protocol.md`) — no IRB contact, no recruiting, no
building. The real-time inference path also doesn't exist yet.

_Suggested figure: none — discussion slide._

---

## Appendix — file pointers, if asked
- Design doc: `wiki/concepts/controller.md`
- Locked numbers (both runs): `wiki/results/phase3-controller.md`, `trained_models/Phase3-controller/`
- Paper section: `paper/paper_results.md` R5 (has the 2026-07-20 update note inline)
- Scripts: `scripts/controller/controller_sim.py` (single-config prototype),
  `scripts/controller/controller_robust.py` (three-lock robustness protocol)
