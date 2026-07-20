---
type: concept
status: active
updated: 2026-07-03
---

# Swing mode (axial-twist removal)

Remove the **axial twist** component from Xsens segment quaternions, keeping swing-only orientation. Implemented as `remove_twist_xyzw()` in `src/scripts/IMU_batch_processor.py`; selected via `MODE = "swing"` (line 11, currently uncommitted). `Data_Processed/imu_quats/` was regenerated in swing mode 2026-06-30 (2 736 clips, `index.csv` `mode` col = `swing`). Do not regenerate unless twist logic changes.

## Hypothesis (branch `swing-mode-xsens`)
Skeleton-derived quaternions can't capture segment roll, so removing twist from Xsens should move it *closer* to NTU → smaller gap → better cross-domain calibration.

## Outcome: hypothesis WRONG in feature space — and superseded by v2
[[mmd-domain-gap|MMD]] NTU↔Xsens **increased ~3×** under swing for every encoder except supervised. The symmetric test is now run (`mmd_domain_gap_symmetric.py (dead, not in this repo)`): swing-projecting NTU too barely moved it (still ~3× wide) → the increase was **not** the asymmetry but twist-stripping *damaging* Xsens (a fixed-axis, pose-dependent, lossy distortion — not NTU's clean swing-only orientation).

**Superseded:** the correct fix is to rebuild Xsens from positions through NTU's own shortest-arc construction — [[position-reconstruction-v2]] (v2). v2 MMD is *below* local for every encoder. Swing is now, at most, a mounting-variance-normalization ablation.

## But swing helped within-Xsens — unevenly
Absolute accuracy jumped (scratch k=0: 50.7 local → 74.4 swing), by cutting inter-subject mounting/calibration twist noise. Per-subject it's not uniform (scratch k=0, swing − local):

| subject | Δ pp |
|---|---|
| sub7 | **+54.6** |
| sub11 | +36.2 |
| sub10 | +36.0 |
| sub9 | +11.6 |
| sub8 | **−19.9** |

Mean +23.7 is carried by sub7 (the known outlier — swing evidently fixed its mounting twist). **sub8 got much worse and is unexplained** — must be resolved before publishing ([[open-questions]]). Honest story: twist removal normalizes mounting variance — usually helps, can hurt.

Full swing results: [[swing-mode-findings]].
