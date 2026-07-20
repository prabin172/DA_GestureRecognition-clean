# Live human-in-the-loop study — protocol (draft for approval)

**STATUS: NOT APPROVED. Discussion draft only, per `tasks.md` T7. Do not build the real-time
inference path, recruit subjects, or contact the IRB/ethics office until the human explicitly
signs off on this document.** Written 2026-07-09, fleshing out `tasks.md` T7's spec.

## 1. Purpose

Convert the simulated controller (`paper_results.md` R5, built from `temp_controller_sim.py` /
`temp_controller_robust.py` on real held-out posteriors) into a small live validation. Two claims
from the simulation need human-subject evidence to earn the THMS (human-machine-systems) venue,
not just a sensing result:

1. **Recognition differences compound into task-level differences** (R5: a 4 pp recognition
   gap becomes a ~21 pp task-success gap under the simulated FSM).
2. **Calibration governs the safety/throughput trade at iso-safety operating points** (R5 Lock 3:
   supLP120 reaches a 1% false-activation budget at the lowest threshold, ~13–20% faster).

## 2. Minimal design (est. 3–4 weeks elapsed, per tasks.md)

- **N = 5–6 NEW subjects.** Never seen by any pretrained/fine-tuned model → true LOSO cold-start
  (matches the paper's k-shot-calibration framing exactly, not a new protocol). Doubles as a
  dataset expansion for the T6 release (5 → 10–11 subjects total).
- **Per-subject session (~60–90 min):**
  1. Suit up in the Xsens mocap suit (same hardware as `DataCollection/sub{7-11}/`).
  2. Record k=3 calibration shots live, per gesture in the fixed 7-input assignment
     (§3) — mirrors the k=3 calibration condition already reported throughout the paper (R1–R6).
  3. On-device calibration: head-only fine-tune from the frozen pretrained encoder, same
     protocol as `temp_loso_fulltrain_calibration.py`'s k=3 `head_only` calibration stage
     (30 epochs, AdamW, lr 1e-3) — this is the one piece that must run in near-real-time
     between suit-up and the task (seconds, not minutes; head-only fine-tune on 512-d
     pooled features is cheap).
  4. Perform the 12-step Sequential Control Task from `temp_controller_sim.py` (`MISSION = ["next",
     "next", "previous", "approach", "grasp", "confirm", "next", "approach", "grasp",
     "release", "confirm", "cancel"]` in code — implementation constant names are unchanged, see
     [[controller]]'s terminology-revision note) against a **screen-based simulated executor** — no
     physical robot. Keeps cost near zero and matches the abstract-controller framing already
     defended in the paper (Methods §8: "abstract event-driven controller, not a physics/robot
     sim").
- **Conditions, within-subject, counterbalanced:** 2 inits (`supLP120` = best-calibrated per R4a
  vs `scratch` = no-prior) × 2 operating points (ungated τ=0 vs iso-safety τ* at the 1%
  false-activation budget from R5 Lock 3) = **4 conditions, ~3 task repetitions each**
  (~12 task runs/subject, consistent with the ~60–90 min budget).
- **Metrics:** task success (binary), time-to-completion, false-activation count, rejection
  count (under gating), corrective-input count; NASA-TLX questionnaire per condition
  (subjective workload — cheap, standard, and squarely a THMS-reviewer expectation).
- **Analysis:** paired within-subject comparisons (each subject sees all 4 conditions); step-level
  McNemar across the ~36 task repetitions per subject for power at small N (mirrors the
  clip-level McNemar already used for R2, so the statistical machinery is reused, not invented).

## 3. System Input assignment

Reuse the **data-driven mapping already implemented** in `temp_controller_sim.py::build_mapping`:
rank the 22 Xsens gestures by pooled k=3 recall (method-agnostic, so the map doesn't favor any
one init), exclude locomotion/ambiguous classes (`walk, runonspot, turnaround, buttkicks, hop,
jump, stand`), and assign the most-reliable remaining gestures to the two Safety-Critical States
first, then the five Routine States, in the implementation's fixed priority order (`grasp, release,
confirm, approach, cancel, next, previous` in code). This is the same design guard already stated
in `RESEARCH_LOG.md` A0 (assign the Safety-Critical States to reliably-recognized gestures —
crossarms/squat/wave reliable, throw/jump/hop not, per the OOV distinctiveness finding). **Fix the
assignment once** (from the existing offline posteriors, before any live session) so all subjects
use the identical 7-gesture assignment — this keeps the live study a direct extension of R5's
Lock-1 assignment-robustness result rather than a new variable.

## 4. Technical prerequisites (none built yet)

- **Real-time inference path** (~1 week build + pilot): Xsens MVN live stream → position-derived
  v2 quaternions (same transform as `src/scripts/IMU_batch_processor.py`'s offline v2 pipeline,
  see [[position-reconstruction-v2]]) → `KinematicEncoder` forward pass → posterior → FSM step in
  `temp_controller_sim.py`'s command loop. The offline pieces (parser, encoder, calibration loop,
  FSM) all exist; the streaming glue does not.
- **IRB/ethics check** — gesture-recognition studies of this kind are typically expedited/exempt,
  but this must be **confirmed with the institution before any building or recruiting starts**.
  Not started.
- **Inter-command timing**: synthesized in the simulation (R5 limitations); the live study
  provides the first real timing data — worth capturing explicitly as a secondary outcome even
  though R5 didn't need it.

## 5. Open questions for the human (before approval)

1. Is 2 inits × 2 operating points the right scope, or should `supMAE` (best in the k=1 ungated
   condition per R5) also be included as a 3rd init — raising session length or repetitions?
2. Confirm N=5–6 is sufficient given the step-level McNemar power argument, or whether the T6
   data-release headcount goal (10–11 total) should drive a larger N.
3. Who runs/owns the IRB submission, and what is the realistic timeline given the paper's
   submission deadline?
4. Physical space / hardware logistics for the Xsens suit-up + screen-based executor sessions —
   not addressed here, needs a location and equipment owner.

## 6. What this document is NOT

Not a build plan, not a subject-recruitment plan, not an IRB submission. It is the one-page spec
`tasks.md` T7 asked for, to get a human go/no-go before any of §4's prerequisites are started.
