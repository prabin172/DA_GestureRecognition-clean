#!/usr/bin/env python3
"""CZU-MHAD inertial (sensor_mat) -> 17-segment local-relative quats, drop-in for the LOSO harness.

The TRUE cross-modal external replication: NTU skeleton prior -> CZU wearable-IMU target.
CZU sensor_mat ships 10 MPU9250 sensors, 6-axis (3 accel g + 3 gyro deg/s) + timestamp (us),
NO magnetometer in the released data -> yaw is unobservable (drifts). Sensors are async
(~340-555 Hz, per-sensor rate) with occasional large timestamp gaps.

Placement (paper Fig. 5): {Left,Right} x {Elbow,Wrist,Knee,Ankle} + Chest + Abdomen.

Pipeline (matches src/scripts/IMU_batch_processor.py conventions so the harness runs unmodified):
  1. Madgwick AHRS per sensor at native cadence (dt from timestamps, outlier-clamped) ->
     absolute orientation quaternion series (world frame: gravity + arbitrary yaw).
  2. SLERP-resample each sensor to a common 30 Hz grid (N_out = round(duration_s * 30)).
  3. Per-sensor initial-pose normalization: q <- q0^-1 * q  (removes constant mounting +
     world-yaw offset; makes each sensor's start pose ~identity). Documented frame-handling.
  4. Map 10 sensors -> 17 segment slots (uninstrumented -> identity quat).
  5. global -> parent-relative local quats via the same HIERARCHY as the Xsens/NTU pipeline.
  6. Save (T,17,4) xyzw float32 + index.csv matching Data_Processed/czu_skeleton_lrq/ format
     (file, session, label, n_frames_30hz, mode) with IDENTICAL filenames to the skeleton run
     so the byte-identical LOSO split JSONs can be reused for a paired comparison.

Output -> Data_Processed/czu_imu_quats/
"""
import argparse, re, csv
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation as R, Slerp

ROOT = Path(__file__).resolve().parent.parent.parent
SENSOR_DIR = ROOT / "external_data/czu_mhad_data/CZU-MHAD/sensor_mat"
OUT_DIR = ROOT / "Data_Processed/czu_imu_quats"
TARGET_HZ = 30.0
MAX_FRAMES = 120          # loader truncates beyond this anyway
INIT_POSE_SEC = 0.30      # window used as per-sensor reference "rest" pose
DT_CLAMP_S = 0.02         # dts above this (e.g. the 40 s gap) are treated as one step

ACTIONS = {
    1: "right_high_wave", 2: "left_high_wave", 3: "right_horiz_wave", 4: "left_horiz_wave",
    5: "hammer_right", 6: "grasp_right", 7: "draw_fork_right", 8: "draw_fork_left",
    9: "draw_circle_right", 10: "draw_circle_left", 11: "right_kick_fwd", 12: "left_kick_fwd",
    13: "right_kick_side", 14: "left_kick_side", 15: "clap", 16: "bend_down",
    17: "wave_up_down", 18: "sur_place", 19: "left_body_turn", 20: "right_body_turn",
    21: "left_lateral", 22: "right_lateral",
}
NAME_RE = re.compile(r"([a-z]+)_a(\d+)_t(\d+)")

# 17-segment order (index -> name), matching SEGMENTS_17 in IMU_batch_processor.py
HIERARCHY = {0: None, 1: 0, 2: 1, 3: 1, 4: 3, 5: 4, 6: 5, 7: 1, 8: 7, 9: 8,
             10: 9, 11: 0, 12: 11, 13: 12, 14: 0, 15: 14, 16: 15}

# CZU sensor index (0..9) -> body location. This order was RECOVERED EMPIRICALLY from
# per-action gyro energy (not documented in the .mat): bend->sensor0 (trunk); R-arm actions
# light up 5>4, L-arm 3>2, R-leg 9>8, L-leg 7>6. The more-active sensor of each limb pair is
# the distal segment (higher angular velocity), giving wrist/ankle > elbow/knee. Sensor 0 vs 1
# (trunk): bend flexes the lower trunk most -> 0=Abdomen, 1=Chest.
CZU_SENSOR_LOCATIONS = [
    "Abdomen", "Chest", "Left Elbow", "Left Wrist", "Right Elbow",
    "Right Wrist", "Left Knee", "Left Ankle", "Right Knee", "Right Ankle",
]
LOCATION_TO_SEG = {
    "Abdomen": 0,        # Pelvis
    "Chest": 1,          # T8
    "Right Elbow": 4,    # Right Upper Arm
    "Right Wrist": 5,    # Right Forearm
    "Left Elbow": 8,     # Left Upper Arm
    "Left Wrist": 9,     # Left Forearm
    "Right Knee": 11,    # Right Upper Leg
    "Right Ankle": 12,   # Right Lower Leg
    "Left Knee": 14,     # Left Upper Leg
    "Left Ankle": 15,    # Left Lower Leg
}
IDENTITY_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
INSTRUMENTED_SEGS = sorted(LOCATION_TO_SEG.values())            # 10 mapped segments
UNINSTRUMENTED_SEGS = [s for s in range(17) if s not in INSTRUMENTED_SEGS]  # 7 -> identity


# ---------------- quaternion helpers (xyzw, matching IMU_batch_processor) ----------------
def quat_normalize(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, 1e-12, None)

def quat_conj(q):
    return np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)

def quat_mul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], axis=-1)

def fix_sign_continuity(q):
    q = q.copy()
    for t in range(1, len(q)):
        if np.dot(q[t], q[t-1]) < 0:
            q[t] = -q[t]
    return q


# ---------------- Madgwick AHRS (IMU-only, no magnetometer), wxyz internally ----------------
def madgwick_imu(acc, gyr, dt, beta=0.1):
    """acc: (N,3) in g (any scale, normalized internally). gyr: (N,3) in rad/s. dt: (N,) seconds.
    Returns orientation quats (N,4) in xyzw. Standard Madgwick 2011 IMU update."""
    N = len(acc)
    q = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
    out = np.empty((N, 4))              # store wxyz then convert
    for i in range(N):
        gx, gy, gz = gyr[i]
        ax, ay, az = acc[i]
        n = np.sqrt(ax*ax + ay*ay + az*az)
        if n > 1e-9:
            ax, ay, az = ax/n, ay/n, az/n
            qw, qx, qy, qz = q
            # gradient (objective f = R^T * g_hat - a)
            f1 = 2*(qx*qz - qw*qy) - ax
            f2 = 2*(qw*qx + qy*qz) - ay
            f3 = 2*(0.5 - qx*qx - qy*qy) - az
            j11, j12, j13, j14 = -2*qy, 2*qz, -2*qw, 2*qx
            j21, j22, j23, j24 = 2*qx, 2*qw, 2*qz, 2*qy
            j32, j33 = -4*qx, -4*qy
            # gradient wrt (qw,qx,qy,qz)  (j31=j34=0)
            grad = np.array([
                j14*f1 + j24*f2,                 # d/dqw
                j11*f1 + j21*f2 + j32*f3,        # d/dqx  (j31=0)
                j12*f1 + j22*f2 + j33*f3,        # d/dqy
                j13*f1 + j23*f2,                 # d/dqz  (j34=0)
            ])
            gn = np.linalg.norm(grad)
            if gn > 1e-12:
                grad = grad / gn
        else:
            grad = np.zeros(4)
        qw, qx, qy, qz = q
        qDot = 0.5 * np.array([
            -qx*gx - qy*gy - qz*gz,
             qw*gx + qy*gz - qz*gy,
             qw*gy - qx*gz + qz*gx,
             qw*gz + qx*gy - qy*gx,
        ])
        qDot = qDot - beta * grad
        q = q + qDot * dt[i]
        q = q / np.linalg.norm(q)
        out[i] = q
    # wxyz -> xyzw
    return np.stack([out[:, 1], out[:, 2], out[:, 3], out[:, 0]], axis=1)


def parse_name(stem):
    m = NAME_RE.match(stem)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def sensor_orientation_30hz(raw, n_out):
    """raw: (F,7) one sensor. Returns (n_out,4) xyzw absolute orientation, resampled to 30 Hz,
    initial-pose normalized so the start ~identity."""
    ts = raw[:, 6].astype(np.float64)
    order = np.argsort(ts, kind="stable")
    raw = raw[order]; ts = ts[order]
    acc = raw[:, 0:3].astype(np.float64)
    gyr = np.deg2rad(raw[:, 3:6].astype(np.float64))
    dt = np.diff(ts) / 1e6                       # us -> s
    dt = np.concatenate([[np.median(dt[dt > 0]) if np.any(dt > 0) else 1/340.0], dt])
    dt = np.clip(dt, 1e-4, DT_CLAMP_S)           # kill the ~40 s spurious gaps
    q_native = madgwick_imu(acc, gyr, dt)        # (F,4) xyzw
    q_native = quat_normalize(q_native)
    q_native = fix_sign_continuity(q_native)
    # initial-pose reference over first INIT_POSE_SEC
    tsec = np.cumsum(dt); tsec -= tsec[0]
    ref_mask = tsec <= INIT_POSE_SEC
    if ref_mask.sum() < 2:
        ref_mask[:min(5, len(q_native))] = True
    q0 = R.from_quat(q_native[ref_mask]).mean().as_quat()   # avg quat
    q_native = quat_mul(quat_conj(np.broadcast_to(q0, q_native.shape)), q_native)
    q_native = quat_normalize(q_native)
    # SLERP to n_out uniform frames over [0, tsec[-1]]
    key_t = tsec
    slerp = Slerp(key_t, R.from_quat(q_native))
    out_t = np.linspace(key_t[0], key_t[-1], n_out)
    return slerp(out_t).as_quat()                # (n_out,4) xyzw


def global_to_local(Qg):
    """Qg: (T,17,4) xyzw global -> parent-relative local (same as IMU_batch_processor)."""
    Ql = np.empty_like(Qg)
    Ql[:, 0, :] = Qg[:, 0, :]
    for i in range(1, 17):
        p = HIERARCHY[i]
        Ql[:, i, :] = quat_mul(quat_conj(Qg[:, p, :]), Qg[:, i, :])
    Ql = quat_normalize(Ql)
    for i in range(17):
        Ql[:, i, :] = fix_sign_continuity(Ql[:, i, :])
    return Ql.astype(np.float32)


def process_clip(mat_path):
    sensor = sio.loadmat(mat_path)["sensor"]      # (10,1) object
    n_sensors = sensor.shape[0]
    # clip duration from the densest sensor (clamp gaps), n_out at 30 Hz
    durs = []
    for i in range(n_sensors):
        ts = np.sort(sensor[i, 0][:, 6].astype(np.float64))
        d = np.clip(np.diff(ts) / 1e6, 1e-4, DT_CLAMP_S)
        durs.append(d.sum())
    dur_s = float(np.median(durs))
    n_out = int(np.clip(round(dur_s * TARGET_HZ), 8, MAX_FRAMES))
    # global orientation per segment (identity default)
    Qg = np.tile(IDENTITY_XYZW, (n_out, 17, 1)).astype(np.float64)
    for i in range(n_sensors):
        loc = CZU_SENSOR_LOCATIONS[i]
        seg = LOCATION_TO_SEG.get(loc)
        if seg is None:
            continue
        Qg[:, seg, :] = sensor_orientation_30hz(sensor[i, 0], n_out)
    Ql = global_to_local(Qg)
    # uninstrumented segments carry no measurement -> zero articulation vs parent (identity local),
    # rather than the spurious inv(parent) the hierarchy would otherwise inject.
    Ql[:, UNINSTRUMENTED_SEGS, :] = IDENTITY_XYZW.astype(np.float32)
    return Ql, n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only first N clips (debug)")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(SENSOR_DIR.glob("*.mat"))
    if args.limit:
        files = files[:args.limit]
    rows = []
    skipped = 0
    for n, f in enumerate(files):
        parsed = parse_name(f.stem)
        if parsed is None:
            skipped += 1; continue
        subj, aid, rep = parsed
        if aid not in ACTIONS:
            skipped += 1; continue
        try:
            Ql, n_out = process_clip(f)
        except Exception as e:
            print(f"  SKIP {f.name}: {e}"); skipped += 1; continue
        out_name = f"{subj}__a{aid:02d}__t{rep:02d}.npy"
        np.save(out_dir / out_name, Ql)
        rows.append(dict(file=out_name, session=subj, label=ACTIONS[aid],
                         n_frames_30hz=n_out, mode="czu_imu"))
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(files)} processed")
    with (out_dir / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "session", "label", "n_frames_30hz", "mode"])
        w.writeheader(); w.writerows(rows)
    print(f"\nDone. {len(rows)} clips -> {out_dir} (skipped {skipped})")
    subs = sorted(set(r["session"] for r in rows))
    print(f"subjects: {subs}  actions: {len(set(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()
