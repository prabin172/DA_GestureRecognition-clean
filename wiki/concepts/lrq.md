---
type: concept
status: active
updated: 2026-07-03
---

# Local Relative Quaternions (LRQ)

The core data representation and the project's central technical bet: both [[ntu-dataset|NTU skeleton]] and [[xsens-dataset|Xsens IMU]] data are converted to the **same format** — `(T, 17, 4)` float32, 17 body segments × 4 quaternion components, 30 Hz — so one encoder can serve both domains.

Each segment's orientation is expressed **relative to its parent** in the kinematic hierarchy, not in world space. This makes the representation invariant to global body rotation/translation.

## 17-segment body model
`Pelvis, T8, Head, R-Shoulder, R-UpperArm, R-Forearm, R-Hand, L-Shoulder, L-UpperArm, L-Forearm, L-Hand, R-UpperLeg, R-LowerLeg, R-Foot, L-UpperLeg, L-LowerLeg, L-Foot`

Hierarchy: Pelvis → T8 → Head; T8 → each Shoulder → UpperArm → Forearm → Hand; Pelvis → each UpperLeg → LowerLeg → Foot. NTU topology is mapped to match Xsens exactly (`src/data/ntu_parser.py`).

## Temporal padding
All clips zero-padded to `max_frames=120`. A boolean padding mask (1=real, 0=pad) travels with the data; encoders convert it to a Transformer padding mask. `masked_mean` pools only over real frames.

## Variants of the quaternion extraction
- **local** (v1 default): full local relative quaternion from *measured* Xsens orientation, including axial twist.
- **swing**: axial twist removed post-hoc — see [[swing-mode]]. Superseded (it widened the gap).
- **reconstructed (v2, current best)**: Xsens orientation rebuilt from mvnx *positions* via NTU's own shortest-arc `get_bone_quaternion` → shares NTU's construction, twist excluded identically. Closes the domain gap. See [[position-reconstruction-v2]].
- **cleaned (slerp035)**: NTU sign-flip/discontinuity smoothing — see [[cleaned-source-pretraining]]. Verdict: no actionable gain.

## Where it's computed
- NTU: `src/data/ntu_parser.py` — bone quaternions vs N-pose reference, projected through the hierarchy. See [[data-pipeline]].
- Xsens v1: `src/scripts/IMU_batch_processor.py` — from `.mvnx`/`.xlsx` segment *orientations*, 240→30 Hz.
- Xsens v2: `src/scripts/IMU_batch_processor_v2.py` — from mvnx *positions* through NTU's construction. See [[data-pipeline]], [[position-reconstruction-v2]].
