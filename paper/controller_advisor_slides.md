# The downstream controller (C6 / THMS pillar) — advisor briefing
_Prepared 2026-07-20. Status: results below are from the current, final methodology
(fully randomized System Input assignment throughout — see Slide 4). Prabin has not yet
put hands-on effort into this component — it was executed under supervision._

---

## Slide 1 — Why a controller exists at all

**The problem it answers:** elsewhere in the paper, accuracy differences between pretraining
objectives shrink to a few percentage points once the domain gap (skeleton → wearable IMU) is
large. A reviewer's obvious objection: *"so what — a few points don't matter."*

**What the controller does:** takes the recognizer exactly as trained — no new training, no
synthetic error model — and wires its real held-out predictions into a synthetic multi-step
command task, where some steps are safety-critical (wrong prediction → task fails outright) and
others are routine (wrong prediction → costly but recoverable). Then it asks: do those "few
accuracy points" turn into a much bigger difference once errors can compound?

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
- **No subject ever performed a "grasp" or "release" gesture.** The 7 gestures relabeled for the
  task are a completely **random draw** from the 22 recorded gestures — not chosen by which
  gesture the recognizer classifies best, not chosen by which gesture looks like a plausible
  command, not chosen by any property of the data at all.
- The 12-step sequence deliberately visits the 2 safety-critical states **3 times** — this is
  what makes small recognition differences compound into large task-level differences, rather
  than averaging out.

_Suggested figure: the 12-step sequence as a flow diagram, with the 3 safety-critical visits
highlighted._

---

## Slide 3 — The full methodology: everything is randomized, nothing is picked

This is the part worth stating precisely, because it's the answer to "is this controlled or
cherry-picked":

- **120 independently, uniformly-random draws** of which 7 gestures become the task's inputs.
  No draw is chosen by recognizer accuracy, gesture semantics, or any other property — every
  draw is a fresh unweighted random sample.
- **1000 Monte Carlo task-execution trials** per (assignment, method, shot-count, threshold)
  cell — each trial replays the 12-step task against freshly resampled real recognizer output.
- **All three robustness checks below share this exact same set of 120 random assignments** —
  there is no separate "illustrative" or "representative" configuration anywhere in the results.
  An earlier version of this analysis picked one fixed gesture assignment (ranked by recognizer
  reliability) for two of the three checks; that approach has been fully replaced with the random
  design above, specifically to remove any appearance of a hand-picked task design.

_Suggested figure: none — this is the methodology slide, keep it as text/bullets for clarity._

---

## Slide 4 — Robustness design (three checks, one shared random design)

A controller like this has three knobs a reviewer could accuse of being tuned to produce a
favorable result. Three checks, one per knob, **all run over the same 120 random assignments**:

| Lock | What it does | Kills the objection |
|---|---|---|
| **1 — Base outcome** | Task success/failure across the 120 random assignments | "You cherry-picked easy/hard gestures" |
| **2 — Cost-severity sweep** | Sweep how catastrophic a safety-critical error is (6 settings, mild to near-infinite), same 120 assignments | "You picked a harsh failure rule to win" |
| **3 — Iso-safety threshold** | Derive the confidence threshold automatically per method from a fixed safety budget, same 120 assignments | "You tuned the threshold to win" |

_Suggested figure: the frontier plot (`trained_models/Phase3-controller/robust/frontier.png`)
showing task-success vs. false-activation tradeoff curves, one line per method, averaged over
the 120 assignments._

---

## Slide 5 — Result: all three checks agree

**One pretraining objective — `mae`, pure reconstruction — compounds worst under every single
stress test.**

**Lock 1 (base outcome, mean across 120 assignments), k=1:**

| τ | scratch | mae | supMAE | supLP120 | supcon |
|---|---|---|---|---|---|
| 0 (ungated) | 0.516 | **0.455 (worst)** | 0.522 | 0.472 | 0.523 |
| 0.9 (gated) | 0.714 | **0.650 (worst)** | 0.717 | 0.703 | 0.740 |

**Lock 2 (median task cost across 120 assignments, k=1) — lower is better:**

| Cost severity | mae | scratch | supLP120 | supMAE | supcon |
|---|---|---|---|---|---|
| mild | 18.3 | 17.5 | 17.7 | 17.1 | 17.4 |
| moderate | 32.3 | 29.0 | 30.4 | **28.5 (best)** | 28.6 |
| severe | **55.5 (worst)** | 48.3 | 51.2 | 47.1 | 47.8 |

**Lock 3 (iso-safety, 1% false-activation budget) — mean task success across 120 assignments:**

| k | mae | scratch | supLP120 | supMAE | supcon |
|---|---|---|---|---|---|
| 1 | **0.360 (worst)** | 0.514 | 0.561 | 0.524 | **0.646 (best)** |
| 3 | **0.741 (worst)** | 0.797 | **0.862 (best)** | 0.812 | 0.848 |

**Reading:** mae is worst on task success (Locks 1 & 3) and worst on cost (Lock 2), at every shot
count and every severity level tested. This ordering does not depend on which of the three
robustness checks you look at — the whole point of building three independent checks was to make
sure of exactly that.

_Suggested figure: bar chart of task success by method at one operating point
(`trained_models/Phase3-controller/robust/vocab_distribution.png`), plus the cost-severity line
chart (`costmodel_sweep.png`)._

---

## Slide 6 — One honest secondary finding (not hidden, doesn't change the ranking)

- `supLP120` (the best-calibrated objective — see the separate calibration results) still dips
  *just* below scratch on the ungated Lock 1 metric at k=1 (0.472 vs 0.516), and trends as the
  second-most-expensive method as Lock 2's penalty gets severe.
- Mechanism: `supLP120`'s confidence is well-calibrated and sharply separates classes, so it
  occasionally, *confidently* misclassifies an unrelated gesture onto a safety-critical anchor —
  confidently enough to clear even a moderate threshold.
- This does **not** contradict the calibration story — `supLP120` is still the best-calibrated
  method on average, and it leads at k=3 under the iso-safety view (Lock 3 above). It's a narrow,
  secondary effect, visible only at specific operating points, and it never rises to displace
  `mae` as the worst-compounding objective under any of the three checks.

_Suggested figure: none — discussion point, present as text._

---

## Slide 7 — What's not yet done

- **Live human-in-the-loop study**: exists only as an unapproved discussion draft
  (`paper/live_study_protocol.md`) — no IRB contact, no recruiting, no building.
- **Real-time inference path**: doesn't exist yet; the controller currently replays already-collected
  held-out posteriors, not a live stream.
- **My own hands-on involvement**: I have supervised this component's design and direction but have
  not yet implemented or debugged it myself — flagging that directly.

_Suggested figure: none._

---

## Appendix — file pointers, if asked
- Design doc: `wiki/concepts/controller.md`
- Locked numbers: `wiki/results/phase3-controller.md`, `trained_models/Phase3-controller/robust/`
  (`vocab_sweep.csv` = Lock 1, `costmodel_sweep.csv`/`costmodel_summary.csv` = Lock 2,
  `frontier.csv`/`iso_safety.csv`/`iso_safety_summary.csv` = Lock 3)
- Paper section: `paper/paper_results.md` R5, `paper/paper_method.md` §8/§8.1
- Script: `scripts/controller/controller_robust.py --vocabs 120 --missions 1000`
