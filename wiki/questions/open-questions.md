---
type: question
status: active
updated: 2026-07-10
---

# Open questions & TODO

## T6 data release — PII found in raw mvnx, plan written, packaging PAUSED (2026-07-09)
**Human decision:** release both raw mvnx and processed v2 quats (nothing blocks releasing both) — but **do NOT package/export yet**; write the plan and revisit later.

**Finding:** all 18 raw `.mvnx` files (`DataCollection/sub{7-11}/*.mvnx`) embed real subject PII in the `<subject>` XML tag's `originalFilename` attribute — not visible from the filename or any processed output, only inside the raw file header. Confirmed across every subject:

| dir | `originalFilename` embeds |
|---|---|
| sub7 | "Amit" |
| sub8 | "Charles" |
| sub9 | "Ahmed" |
| sub10 | "Pankaj" |
| sub11 | "Grishma" |

Full attribute example: `originalFilename="C:\Users\timil\OneDrive - Florida State University\Documents\Amit-Jan24-001.mvn"` — also exposes the operator's OS username (`timil`) and institutional affiliation. The `label="Prabin"` attribute is the operator/recorder, not the subject, and is separately present in every file. `recDate`/`recDateMSecsSinceEpoch` give exact session timestamps (lower sensitivity, but generalizable to "leaks exact session dates" if that matters for consent language).

**De-identification plan (not yet executed):**
1. Write an export script (new, e.g. `temp_release_mvnx_deidentify.py`) that copies each mvnx, and in the `<subject>` tag: replaces `label` with the generic subject code (S1–S5, remapped from sub7–sub11), strips `originalFilename` entirely, strips or coarsens `recDate`/`recDateMSecsSinceEpoch` (e.g. keep only the date, drop time-of-day + the epoch-ms field), leaves all motion/segment/point data untouched (that's the actual scientific content).
2. Also export the processed v2 quats (`Data_Processed/imu_quats_v2/`) — these `.npy` + `index.csv` files were already checked and do **not** contain this metadata (never touched the mvnx header), so they're lower-risk, but should still get the same sub7–11 → S1–S5 subject-code remap in filenames/`index.csv` for consistency with the mvnx release.
3. Write the datasheet README (capture protocol, 22 gestures, 5 subjects, sensor placement, suggested license CC-BY-4.0) per `tasks.md` T6.1.
4. Do NOT touch the original `DataCollection/` files in place — de-identify into a new export directory only.
5. Before actually publishing anywhere (Zenodo/OSF/GitHub): confirm the consent-form/IRB language actually covers what's being stripped vs. kept — the human flagged wanting to "check something" implicitly by pausing here, worth an explicit re-confirmation before upload even after the export script exists.

**Next step when resumed:** write and dry-run the de-identification script on one subject first, diff the before/after XML by hand, then get a go-ahead before running on all 5 and before any actual upload.

## T7 — live human-in-the-loop study (unapproved, 2026-07-09)
Protocol written (`paper/live_study_protocol.md`) but explicitly **not approved** — no recruiting, IRB contact, or real-time-inference build work until the human signs off. See `tasks.md` T7 and `SESSION_HANDOFF.md`.

## Genuinely still open, not in the current `tasks.md` plan (deprioritized in the 2026-07-09 replan — flag for Planning)
These two were live items in the original [[publishability-review]] but dropped out when `tasks.md` was rewritten around the five-setting synthesis (T0–T9). Neither is run, and neither is addressed as a stated limitation in `paper/paper_discussion.md` either — worth an explicit decide-or-drop before submission, not a silent gap.
1. **Complete the 2×2** (init source × frozen/fine-tuned): NTU-pretrained **frozen** + target-supmae **fine-tuned** are the two missing cells (Job 1 = NTU-init + full fine-tune; Job 2 = target-pretrain + frozen already run, [[xsens-to-xsens-loso]]). Same scripts, two more runs.
2. **OOV per-action stats + distinctiveness analysis** (correlate a gesture's inter-class kinematic separability in encoder space with its few-shot OOV recall) — [[oov-leave-class-out]]. Cheap; would turn the OOV heatmap into a predictive finding and justify the controller's safety-command assignment with data rather than just observation.

## Optional polish (not urgent, per `SESSION_HANDOFF.md`)
- `paper/paper_conclusion.md` and `paper/paper_discussion.md` §2 (Relation to prior work) still describe the objective family / external-validity story in a couple of places as it stood before SupCon's T2 full-parity pass — not wrong, just less sharp than `paper_intro.md`/`paper_results.md`/`paper_abstract.md` now are.

## Deferred (explicit human decision — do not start without asking)
- **T8** — full reproducible rerun including multi-seed NTU *pretraining* (every objective currently has one pretrained checkpoint; pretraining-seed variance is unquantified).
- **T9** — second-backbone replication on `DSTformerQuatEncoder` ([[models]]).
- **DANN under swing** — moot; swing is dead, v2 is locked ([[dann-experiments]]).

## Standing puzzles
- ~~Why does sub8 degrade under swing when sub7 improves +54.6?~~ **RESOLVED** — did not survive v2 (sub8 supMAE−scratch @ k=1: −19.9 swing → −1.0 v2). Was a swing twist-stripping artifact, not a real subject effect. See [[swing-mode]].
- **MMD/CKA↔accuracy misalignment (necessary-not-sufficient)** — a standing finding, not a bug; state explicitly. Extended 2026-07-09 by the CKA-per-target honest null: CKA doesn't even reliably track *gap width* across datasets, let alone predict transfer benefit — see [[domain-gap-metrics]].
