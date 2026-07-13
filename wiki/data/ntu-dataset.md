---
type: data
status: active
updated: 2026-07-03
---

# NTU RGB+D (source domain)

- **Processed location:** `Data_Processed/ntu_quats/` — 43 490 `.npy` files, `S{set}C{cam}P{person}R{rep}A{action}.npy`, each `(T, 17, 4)` float32 [[lrq]].
- **Classes:** 120 (`A001`–`A120`). A 23-class "relevant" subset (semantically matching the 22 Xsens gestures) lives in `src/data/ntu_relevant_action_ids.json`.
- **Parser:** `src/data/ntu_parser.py` — raw `.skeleton` → 3D joints → shortest-arc bone quaternions vs N-pose → local relative via the 17-segment hierarchy (matches Xsens exactly).
- **Batch processor:** `src/scripts/NTU_batch_processor.py`.
- **Loader:** `src/data/ntu_loader.py` — `UnifiedNTUDataset`, modes `'mae'` (quats, mask), `'supervised'` (+action id), `'mil'` (+word2vec 300-d label vector via `src/data/word2vec_utils.py`).

## Variants
- **Cleaned (slerp035):** quaternion discontinuity smoothing, `temp_outputs/ntu_quats_cleaned_slerp035/` etc. Verdict: no actionable transfer gain — [[cleaned-source-pretraining]].
- **Swing-projected NTU:** does NOT exist; needed for a symmetric [[swing-mode]] MMD test.

Used by all [[pretraining-objectives]] and the NTU→NTU [[sanity-checks]].
