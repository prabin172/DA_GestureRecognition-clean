---
type: result
status: active
updated: 2026-07-20
---

**2026-07-20: numbers refreshed from `DA_GestureRecognition-clean`'s independently-retrained checkpoints — this section's two headline significance claims weakened to trends.** supLP120<scratch and supMAE>supLP120 both still point the same direction but no longer clear p<.05/p<.0001 respectively. Direction is unchanged; treat the specific p-values here as fragile at this sample size (n=43–45), not seed/checkpoint-stable. See `paper/paper_results.md` R6b for the full discussion.

# CZU-MHAD inertial LOSO — TRUE cross-modal external validity (skeleton→IMU)

The independent-public **cross-modal** replication: NTU skeleton prior → CZU wearable-**IMU**
target. Complements the same-modality [[czu-skeleton-loso]] (skeleton→skeleton). This is the
external analogue of the main NTU→Xsens path — a second, independent instance of the
skeleton→inertial gap on a public dataset.

- Run: `trained_models/CZU-IMU-LOSO{,-seed43,-seed44}/` (**3 seeds** 42/43/44, 5 subj LOSO,
  k=0/1/3 → 45 folds pooled). Data `Data_Processed/czu_imu_quats/` (1165 clips, 22 actions).
  Parser `scripts/data_pipeline/czu_imu_parser.py`.
- Launched via `scripts/orchestration/czu_imu_loso.sh` (seed 42) + `scripts/orchestration/czu_multiseed_run.sh` (seeds 43/44);
  CRC via `scripts/external/czu/imu_crc_baseline.py`. Pooled stats: `scripts/external/czu/multiseed_analyze.py`.

## How CZU inertial is made encoder-compatible (the representation bottleneck)

CZU `sensor_mat/` = 10 MPU9250 sensors, **6-axis accel+gyro only (no magnetometer in the
released data)**, async ~340–555 Hz, one spurious ~40 s timestamp gap per clip. The
NTU-pretrained encoder eats `(T,17,4)` **local-relative orientation quaternions**, so the parser:
1. Madgwick AHRS per sensor (accel+gyro → orientation; **yaw unobservable → drifts**).
2. SLERP-resample to 30 Hz; per-sensor initial-pose normalization (removes mounting + world-yaw offset).
3. Map 10 sensors → 17 segments (7 uninstrumented → identity local); global→parent-relative
   local quats (same hierarchy as `src/scripts/IMU_batch_processor.py`).

**Sensor index→body order was recovered empirically** from per-action gyro energy (not in the
.mat): `0=Abdomen,1=Chest,2=L Elbow,3=L Wrist,4=R Elbow,5=R Wrist,6=L Knee,7=L Ankle,8=R Knee,
9=R Ankle`. Validated per action (bend→abdomen; R-arm→R-arm segs; L-kick→L-leg segs).

**Key bottleneck:** matching the skeleton encoder forces an **orientation-only** representation —
the accelerometer magnitude signal (which the CRC baseline exploits) is discarded, and yaw drifts.
Absolute deep accuracy is therefore far below CRC. But scratch and all priors share this
representation, so the **prior-vs-scratch comparison stays valid**.

## Final accuracy — per-(method,k) mean, pooled 3 seeds × 5 subj (n=45)

| k | scratch | mae | **supMAE** | supLP120 | supcon | CRC (raw accel+gyro, 1 seed) |
|---|---|---|---|---|---|---|
| 0 | 55.72 | 56.23 | **56.40** | 54.87 | 55.28 | 86.84 |
| 1 | 54.25 | 54.76 | 54.48 | **52.81** | 53.25 | 90.84 |
| 3 | 58.31 | 58.05 | **59.63** | 57.61 | 57.27 | 94.42 |

(2026-07-20 rerun, pooled 3 seeds × 5 subj, all 5 methods trained natively together.)

## The headline, softened: direction holds, significance mostly does not

- Same-modality (CZU skeleton [[czu-skeleton-loso]], UTD skeleton [[utd-skeleton-loso]]):
  **supLP120 still clearly wins** (+5.0 pp zero-shot, pooled p=.001) — pure-supervised prior wins there.
- Cross-modal (CZU inertial, here): **supLP120 still trends worst — below scratch at every k, but no
  longer significantly so** (was p=.044–.056, now p=.41–.59). supMAE is the numerically best method
  at every k but does not clear significance vs scratch either.

Per-fold sign tests, pooled over (subject, k, seed) cells (ties dropped):

| contrast | original run | **this rerun (2026-07-20)** | reading |
|---|---|---|---|
| supMAE > supLP120 | 36/44 (p<.0001) | **29/43 (p=.032)** | direction holds, much weaker |
| supLP120 < scratch | 28/43 (p=.066), meanΔ −2.3 pp | **19/44 (p=.45)**, meanΔ −1.0 pp | trend only now |
| supMAE > scratch | 25/42 (p=.28) | **28/43 (p=.066)** | still a trend, not significance, as before |

Paired-t by k (n=15): supLP120 − scratch = −0.86 (k0, p=.47) / −1.44 (k1, p=.41) / −0.70 (k3, p=.59)
— all n.s. now, was p=.044/.13/.056; supMAE − scratch = +0.68 (k0, p=.43) / +0.23 (k1, p=.84) / +1.32
(k3, p=.25) — all n.s. in both runs.

**Interpretation, more cautious than before:** the *direction* every claim pointed toward — supervised-only
priors weaken at this gap, reconstruction-containing priors hold up relatively better — is unchanged
and mechanistically consistent with the mechanism story ([[multiseed-loso-v2]] C2/C3). But this section
no longer carries a standalone statistically-significant finding on its own after an independent
checkpoint retrain; the sharper, still-significant version of the same contrast lives in [[czu-imu-dual]]
(R6c), on a stronger target.

**Boundary (see [[czu-imu-dual]], R6c):** the sharper, still-significant version of this contrast is on a
stronger target. When the target is given its full raw signal (raw scratch already ≈ CRC), `dual/scratch`
is the best performer at every k, and **supLP120 is significantly worse (10/41, p=.0015, paired-t
significant at k=0)**. So R6b is the *middle* of a three-column arc (small-gap supLP120 wins → weak
cross-modal supLP120 trends worse (this page) → strong cross-modal supLP120 significantly worse, R6c),
not a standalone "reconstruction transfers to IMU" claim.

## supcon (2026-07-20, pooled 3 seeds n=15/k, trained natively — no separate supcon dir in this rerun)

supcon − scratch = **−0.44 / −1.00 / −1.04 pp @ k=0/1/3** (n.s., p≥.55 — was n.s.-to-marginal
originally too, essentially unchanged); sign test supcon>scratch 24/44 (p=.65, a wash trending
negative, meanΔ −0.83 — nearly identical to the original run's p=.37, meanΔ −0.83). supcon vs supLP120
is statistically even (21/43, p=1.0). **The two label-aware objectives (softmax supLP120, contrastive
supcon) still land in the same negative-to-neutral band here**, distinctly below supMAE — the only
objective at this setting with a reconstruction component. This directional pattern is the most stable
part of R6b under the retrain: what predicts weaker transfer at this gap is the *absence* of
reconstruction, not the presence of any particular supervised recipe. Feeds `paper_results.md` R6b +
the R6e five-setting map's supcon paragraph.

## Caveats (do not overclaim)

1. **The pp margin over scratch is not real; the ranking direction is, but weakly.** Pooled, supMAE does
   **not** beat scratch (28/43 folds, p=.066, a trend). What survives — more weakly than the original
   run found — is the *prior-vs-prior* direction (supMAE trends > supLP120, 29/43, p=.032) and supLP120
   trending below scratch (p=.41–.59, no longer significant). Report the direction, not a clean
   statistically-significant inversion, at this specific setting — [[czu-imu-dual]] carries the
   significant version.
2. **Deep ≪ CRC (55–60% vs 87–94%)** — the orientation-only encoding (no accel, yaw drift) is a
   lossy bottleneck for inertial; raw-signal CRC is far stronger in absolute terms. Prior-vs-scratch
   comparison is valid (shared representation); absolute numbers are not a claim about IMU ceiling.
3. **Does NOT reproduce "mae = negative transfer."** On NTU→Xsens, mae is *the* negative-transfer
   culprit; here mae trends slightly positive and supLP120 is the one that trends worst. Different
   pattern — the cross-modal weakening here is carried by the *label-aware-only* priors, not mae.

## CRC published-baseline (raw inertial)

CRC on raw 6-axis statistical moments (mean/std/var/skew/kurtosis × 10 sensors × 6 ch = 300-dim),
same LOSO splits: **86.8 / 90.8 / 94.4%** @k0/1/3. Comparable to the CZU paper's cross-subject
inertial range (~65–85%, T5–T7); higher here (all-10-sensor moments + 4-subject LOSO dictionary).
`trained_models/CZU-IMU-LOSO/crc_baseline/`.

See also [[czu-imu-dual]] (R6c boundary) · [[czu-skeleton-loso]] · [[multiseed-loso-v2]] · [[loso-fulltrain-calibrate]].
