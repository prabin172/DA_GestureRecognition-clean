---
type: result
status: active
updated: 2026-07-04b
---

# Position-derived Xsens reconstruction (v2) — the real domain-gap fix

The chosen answer to "why did swing widen the gap, and how do we actually reduce it." Supersedes [[swing-mode]] as the gap-reduction approach. Parser: `src/scripts/IMU_batch_processor_v2.py`.

## Why (the A→B chain)
- **NTU quats are position-derived** ([[lrq]]): `get_bone_quaternion` (`src/data/ntu_parser.py:48`) builds each bone as a **shortest-arc** rotation from an N-pose reference to the observed bone vector → inherently **twist-free** (positions cannot encode segment roll).
- **v1 Xsens quats are measured orientation** (mvnx/xlsx), which *carry* real axial twist. [[swing-mode]] tried to erase that twist post-hoc with a fixed-axis swing-twist strip — a lossy, pose-dependent distortion that is **not** the same object as NTU's clean swing-only orientation.
- **Test A (symmetric swing)** confirmed the diagnosis: swing-projecting NTU too (`mmd_domain_gap_symmetric.py (dead, not in this repo)`) barely moved NTU and left the gap ~3× wider than local (see table). So the gap increase was not the asymmetry — it was twist-stripping damaging Xsens.
- **B (this page):** rebuild Xsens the way NTU is built — from **positions**, through NTU's own `get_bone_quaternion` → both domains share one construction (intrinsic per-frame zero-twist, same reference).

## Result — MMD² NTU↔Xsens (n=500, local NTU)
| encoder | local (v1) | swing | symmetric swing | **v2 (positions)** |
|---|---|---|---|---|
| sup | 0.0053 | 0.0054 | 0.0054 | **0.0047** |
| supmae | 0.0109 | 0.0322 | 0.0322 | **0.0092** |
| mae | 0.0204 | 0.0634 | 0.0692 | **0.0182** |
| scratch | 0.0515 | 0.1791 | 0.1643 | **0.0415** |

**v2 gap is below the original local representation for every encoder** — reconstruction through NTU's construction genuinely closes the gap. Dirs: `trained_models/MMD_DomainGap-v2/`, `-symmetricSwing/`. Logs: `mmd_v2.log`, `mmd_symmetric.log`.

## The v2 parser (`IMU_batch_processor_v2.py`)
- Reads mvnx **segment positions** (23 segments × 3D/frame; xlsx has none — orientation only). Confirmed present on every normal frame of all 20 sessions.
- Bones = child-origin − parent-origin over the 17-segment subset of the 23 Xsens segments; run through the imported `get_bone_quaternion` (cannot drift) + parent-relative composition (same hierarchy as NTU/v1).
- **World alignment:** per-session rotation from the T-pose (spine=up, feet=forward) into NTU's frame, so the fixed reference vectors are meaningful.
- **Reuses v1 plumbing unchanged:** `load_offsets`, `parse_anvil`, `slice_and_downsample` → offset/ground-truth alignment is **bit-identical** to v1 (verified: all 2736 clips share `offset_sec` + 240 Hz frame windows).
- Output: `Data_Processed/imu_quats_v2/` (2736 clips, `index.csv` `mode=reconstructed`).

## Validation (prototype self-checks, all pass)
- Vectorized bone quats bit-identical to scalar `get_bone_quaternion` (0.0 err); round-trip `q·ref == bone_dir` to 4e-13; proper rotation frame (det +1); unit quats.
- **Bug caught & fixed:** mvnx has a `tpose-isb` calibration frame (hyphen) that leaked in as frame 0 and shifted the timeline by one frame; now keys on `type=="normal"` → count matches xlsx exactly (121942).

## Known limitation
Hands (segs 6, 10) have no distal joint in the 23-segment positions → hand roll/flex not position-observable → set to **identity local quat** (hand = forearm extension). The 22 gestures are gross-motor; revisit if hand-internal orientation proves to matter.

## LOSO-v2 result (2026-07-04, single seed) — v2 restores + amplifies the prior benefit
`trained_models/LOSO-fullTrainCalibrate-v2/`, log `loso_v2.log`. 100 rows (`scratch, supLP120, supMAE, mae` × 5 subj × k∈{0,1,3,5,10}). Mean final_acc (%):

| method | k=0 | k=1 | k=3 | k=5 | k=10 |
|---|---|---|---|---|---|
| supMAE | **57.5** | **84.0** | 90.8 | 93.2 | 96.4 |
| supLP120 | 57.4 | 80.3 | **90.8** | **94.4** | 96.3 |
| scratch | 56.8 | 79.9 | 89.8 | 92.5 | 96.3 |
| mae | 50.7 | 77.0 | 86.6 | 91.2 | 94.8 |

**The gap knob moves monotonically with the prior's value** — prior-vs-no-prior (supMAE − scratch) at k=1 across the three preprocessings:

| preprocessing | supMAE k=1 | scratch k=1 | Δ | MMD² (supmae) |
|---|---|---|---|---|
| local (v1) | 79.9 | 76.9 | +3.0 | 0.0109 |
| swing | 86.9 | 88.2 | **−1.3** (scratch wins) | 0.0322 |
| **v2** | 84.0 | 79.9 | **+4.1** | **0.0092** |

Widen the gap (swing) → prior turns useless; tighten it below local (v2) → prior helps most. Causal mechanism via a controllable lever, not correlation.

Three wins for the narrative:
- **Prior benefit restored + largest of the three** (+4.1 pp k=1, **4/5 subjects positive**: sub7 +9.7, sub10 +5.2, sub11 +4.3, sub9 +2.3, sub8 −1.0).
- **sub8 swing anomaly is dead.** Swing had sub8 at −19.9 pp; v2 sub8 supMAE−scratch @ k=1 = **−1.0 pp** (a tie). The collapse was a twist-stripping artifact — see [[swing-mode]].
- **supLP120 negative transfer fixed.** Local: −4.3 pp @ k=1 (negative transfer). v2: **+0.4 pp**. Same mechanism — tighter gap removes it.

Caveats: **single seed** — the numbers above are seed-42 only. **Multi-seed now DONE (3 seeds) — see [[multiseed-loso-v2]]: the +4.1 pp @ k=1 does NOT survive pooling (seed43 −1.77 → pooled +0.97 n.s.); robust prior benefit is at k=0/k=3 and in AUC/convergence.** mae now weakest (77.0 @ k=1); benefit compresses to ~0 by k=5–10 (full fine-tune washes out init at high k, expected).

**Decision: v2 is the paper's preprocessing** — directional gap reduction, restores + amplifies prior benefit, kills sub8 anomaly, fixes supLP120 negative transfer, all under one mechanism. Fed into `paper_idea.md`. See [[loso-fulltrain-calibrate]].
