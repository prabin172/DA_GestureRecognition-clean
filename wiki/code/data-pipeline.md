---
type: code
status: active
updated: 2026-07-10
---

# Data pipeline code

Raw → [[lrq]] `.npy` clips → datasets. Downstream of this everything is `(T, 17, 4)` float32 @30 Hz, padded to 120 frames.

## Batch processors
- **`src/scripts/IMU_batch_processor.py`** (v1, superseded) — Xsens side. Parses `.mvnx`/`.xlsx` segment *orientations*, reads Anvil annotations + offsets, downsamples 240→30 Hz (stride 8), writes clips + `index.csv` to `Data_Processed/imu_quats/`. Config at top: `MODE = "swing"` (`local`|`swing`, line 11), `remove_twist_xyzw()` implements [[swing-mode]]. Neither mode's output is used by any current experiment — v2 (below) is the locked representation; this script/output is kept for provenance only.
- **`src/scripts/IMU_batch_processor_v2.py`** (v2, current best) — Xsens from mvnx **positions** through NTU's `get_bone_quaternion` shortest-arc construction (imported → can't drift) + per-session T-pose world alignment; reuses v1's offset/anvil/slicing (identical alignment). Hands→identity (no distal joint). `--prototype` runs self-checks. Output → `Data_Processed/imu_quats_v2/` (`mode=reconstructed`). See [[position-reconstruction-v2]].
- **`src/scripts/NTU_batch_processor.py`** — runs `ntu_parser.py` over the raw skeleton tree → `Data_Processed/ntu_quats/`.

## Parser
- **`src/data/ntu_parser.py`** — `.skeleton` text → 3D joints → shortest-arc bone quats vs N-pose reference → local-relative via the 17-segment hierarchy (matches Xsens topology exactly).

## Loaders
- **`src/data/ntu_loader.py`** — `UnifiedNTUDataset`, modes `'mae'` / `'supervised'` / `'mil'` (see [[ntu-dataset]]).
- **`src/data/imu_loader.py`** — Xsens clip dataset. NOTE: several downstream scripts define their own inline `IMUClipDataset` instead — duplication hazard.
- **`src/data/word2vec_utils.py`** — action-name embeddings for MIL. Mapping jsons: `action_names.json`, `imu_to_ntu_action_map.json`, `ntu_relevant_action_ids.json`.

## Utilities
- `src/scripts/dataset_auditor.py`, `src/scripts/inspect_imu_index.py` — dataset sanity/inspection.
