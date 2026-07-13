"""
IMU batch processor v2 — POSITION-derived local quaternions (option B).

Motivation (see paper_idea.md sec 7.5 + the symmetric-swing MMD test):
The v1 pipeline reads Xsens *measured segment orientations* (xlsx) and, in swing
mode, strips axial twist with a fixed-axis swing-twist decomposition. That does NOT
match how NTU quaternions are built. NTU orientations are derived from 3D *positions*
via a shortest-arc rotation (src/data/ntu_parser.get_bone_quaternion), which is
inherently twist-free (positions cannot encode segment roll). Stripping twist from a
measured orientation is a lossy, pose-dependent distortion — a different object than
NTU's clean swing-only orientation — which is why swing *widened* the domain gap.

v2 rebuilds Xsens the same way NTU is built: take per-frame 3D segment positions from
the .mvnx, form bone vectors, and run them through the *identical* get_bone_quaternion
shortest-arc construction + parent-relative composition. Both domains then share one
convention (intrinsic per-frame zero-twist, same reference), so they are genuinely
commensurable.

Frame alignment: NTU uses fixed reference directions (UP=+Y, RIGHT=+X, FORWARD=+Z).
Xsens world axes are recovered per session from the T-pose (spine=up, feet=forward)
and positions are rotated into the NTU convention before construction, so the same
reference vectors are meaningful.

Hands: the .mvnx has no distal hand joint (only 23 segment origins), so hand roll/flex
is not position-observable. Hand local quats are set to identity (hand = extension of
forearm). Documented limitation; the 22 gestures are gross-motor (arms/torso/legs).

Usage:
    .venv/bin/python src/scripts/IMU_batch_processor_v2.py --prototype   # one session, self-checks
    .venv/bin/python src/scripts/IMU_batch_processor_v2.py               # full regen -> Data_Processed/imu_quats_v2/
"""

from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse NTU's exact bone construction (cannot drift) ...
from src.data.ntu_parser import get_bone_quaternion
# ... and v1's session plumbing (offsets, anvil, slicing, index) unchanged.
from scripts.data_pipeline.IMU_batch_processor import (
    load_offsets, parse_anvil, sanitize_label, slice_and_downsample,
    clear_output_dir, quat_normalize, fix_sign_continuity,
    IMU_HZ, DS_STRIDE, OFFSET_CSV_REL, DATA_ROOT_REL, ANVIL_ROOT_REL,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR_REL = Path("Data_Processed/imu_quats_v2")
MODE_TAG = "reconstructed"

UP      = np.array([0.0, 1.0, 0.0])
DOWN    = np.array([0.0, -1.0, 0.0])
FORWARD = np.array([0.0, 0.0, 1.0])
RIGHT   = np.array([1.0, 0.0, 0.0])
LEFT    = np.array([-1.0, 0.0, 0.0])

# 17-segment bones as (proximal_label, distal_label) over Xsens segment ORIGINS.
# None distal => end-effector not position-observable (hands) => identity local quat.
# Mirrors src/data/ntu_parser.segments_config anatomy + ref vectors + hierarchy.
BONES = [
    ("Pelvis",        "T8",            UP),       # 0  Pelvis (lower/mid spine)
    ("T8",            "Neck",          UP),       # 1  T8 (upper spine)
    ("Neck",          "Head",          UP),       # 2  Head
    ("RightShoulder", "RightUpperArm", RIGHT),    # 3  R Shoulder (clavicle)
    ("RightUpperArm", "RightForeArm",  DOWN),     # 4  R Upper Arm
    ("RightForeArm",  "RightHand",     DOWN),     # 5  R Forearm
    ("RightHand",     None,            DOWN),     # 6  R Hand -> identity
    ("LeftShoulder",  "LeftUpperArm",  LEFT),     # 7  L Shoulder (clavicle)
    ("LeftUpperArm",  "LeftForeArm",   DOWN),     # 8  L Upper Arm
    ("LeftForeArm",   "LeftHand",      DOWN),     # 9  L Forearm
    ("LeftHand",      None,            DOWN),     # 10 L Hand -> identity
    ("RightUpperLeg", "RightLowerLeg", DOWN),     # 11 R Upper Leg
    ("RightLowerLeg", "RightFoot",     DOWN),     # 12 R Lower Leg
    ("RightFoot",     "RightToe",      FORWARD),  # 13 R Foot
    ("LeftUpperLeg",  "LeftLowerLeg",  DOWN),     # 14 L Upper Leg
    ("LeftLowerLeg",  "LeftFoot",      DOWN),     # 15 L Lower Leg
    ("LeftFoot",      "LeftToe",       FORWARD),  # 16 L Foot
]

# parent segment index (same as NTU / v1)
HIERARCHY = {0: None, 1: 0, 2: 1, 3: 1, 4: 3, 5: 4, 6: 5, 7: 1, 8: 7, 9: 8,
             10: 9, 11: 0, 12: 11, 13: 12, 14: 0, 15: 14, 16: 15}
HAND_IDX = {6, 10}


# ---------------------------------------------------------------------------
# .mvnx parsing
# ---------------------------------------------------------------------------
def parse_mvnx_positions(path: Path):
    """Returns (label_to_idx, tpose_pos[23,3], frames_pos[N,23,3])."""
    label_to_idx = {}
    tpose_pos = None
    frames = []
    ctx = ET.iterparse(str(path), events=("end",))
    for _, el in ctx:
        tag = el.tag.split("}")[-1]
        if tag == "segment":
            lbl = el.get("label")
            sid = el.get("id")
            if lbl is not None and sid is not None and lbl not in label_to_idx:
                label_to_idx[lbl] = int(sid) - 1   # mvnx ids are 1-based
        elif tag == "frame":
            ftype = el.get("type")
            p = el.find("{*}position")
            if p is None or not (p.text and p.text.strip()):
                el.clear(); continue
            vals = np.fromstring(p.text, sep=" ")
            if vals.size % 3 != 0:
                el.clear(); continue
            pos = vals.reshape(-1, 3)
            # Data frames are type=="normal" (matches the xlsx timeline the offsets
            # assume). Calibration frames (identity/tpose/tpose-isb) precede them and
            # are dropped; the tpose frame is kept only for world-axis alignment.
            if ftype == "tpose" and tpose_pos is None:
                tpose_pos = pos
            elif ftype == "normal":
                frames.append(pos)
            el.clear()
    if tpose_pos is None:
        raise ValueError(f"No tpose frame in {path}")
    return label_to_idx, tpose_pos, np.asarray(frames, dtype=np.float64)


def compute_align(tpose_pos: np.ndarray, idx: dict) -> np.ndarray:
    """World->NTU rotation from the T-pose anatomy (spine=up, feet=forward).
    Returns R_align (3,3), proper right-handed, mapping v_ntu = R_align @ v_world."""
    P = tpose_pos
    up = P[idx["Neck"]] - P[idx["Pelvis"]]
    up /= np.linalg.norm(up)
    fwd = (P[idx["RightToe"]] - P[idx["RightFoot"]]) + (P[idx["LeftToe"]] - P[idx["LeftFoot"]])
    fwd /= np.linalg.norm(fwd)
    right = np.cross(up, fwd)               # X = Y x Z
    right /= np.linalg.norm(right)
    fwd = np.cross(right, up)               # re-orthonormalize Z = X x Y
    fwd /= np.linalg.norm(fwd)
    return np.stack([right, up, fwd], axis=0)


def _bone_quats_vec(p_start: np.ndarray, p_end: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Vectorized get_bone_quaternion over frames. xyzw. Numerically matches the
    scalar ntu_parser.get_bone_quaternion (asserted in --prototype)."""
    bone = p_end - p_start
    norm = np.linalg.norm(bone, axis=-1, keepdims=True)
    safe = norm[:, 0] > 1e-6
    bone = np.where(norm > 1e-6, bone / np.clip(norm, 1e-12, None), 0.0)
    ref = ref / np.linalg.norm(ref)
    axis = np.cross(np.broadcast_to(ref, bone.shape), bone)
    axis_norm = np.linalg.norm(axis, axis=-1)
    dot = np.clip(bone @ ref, -1.0, 1.0)
    angle = np.arccos(dot)

    out = np.zeros((bone.shape[0], 4)); out[:, 3] = 1.0        # identity default
    good = (axis_norm > 1e-6) & safe
    if good.any():
        a = axis[good] / axis_norm[good, None]
        out[good] = R.from_rotvec(a * angle[good, None]).as_quat()
    # antiparallel (dot<0, axis degenerate): 180deg about a fixed perpendicular
    anti = (axis_norm <= 1e-6) & safe & (dot < 0)
    if anti.any():
        temp = np.array([1.0, 0, 0]) if abs(ref[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(ref, temp); ax /= np.linalg.norm(ax)
        out[anti] = R.from_rotvec(np.broadcast_to(ax, (anti.sum(), 3)) * np.pi).as_quat()
    return out


def reconstruct_local_quats(frames_pos: np.ndarray, R_align: np.ndarray, idx: dict) -> np.ndarray:
    """frames_pos (N,23,3) world -> Q_local (N,17,4) xyzw, NTU construction."""
    P = frames_pos @ R_align.T                                # into NTU frame
    N = P.shape[0]
    global_q = np.zeros((N, 17, 4)); global_q[..., 3] = 1.0
    for i, (prox, dist, ref) in enumerate(BONES):
        if dist is None:
            continue                                          # hand -> identity global (=> identity local vs forearm)
        global_q[:, i, :] = _bone_quats_vec(P[:, idx[prox], :], P[:, idx[dist], :], ref)

    Q_local = np.zeros((N, 17, 4)); Q_local[..., 3] = 1.0
    for i in range(17):
        parent = HIERARCHY[i]
        if parent is None:
            Q_local[:, i, :] = global_q[:, i, :]
        elif i in HAND_IDX:
            Q_local[:, i, :] = np.array([0, 0, 0, 1.0])       # explicit identity local
        else:
            qp_inv = R.from_quat(global_q[:, parent, :]).inv()
            qc = R.from_quat(global_q[:, i, :])
            Q_local[:, i, :] = (qp_inv * qc).as_quat()
    Q_local = quat_normalize(Q_local)
    for i in range(17):
        Q_local[:, i, :] = fix_sign_continuity(Q_local[:, i, :])
    return Q_local.astype(np.float32)


# ---------------------------------------------------------------------------
# Prototype self-checks
# ---------------------------------------------------------------------------
def prototype():
    data_root = (PROJECT_ROOT / DATA_ROOT_REL).resolve()
    mvnx = sorted(data_root.glob("*/*.mvnx"))[0]
    print(f"Prototype on {mvnx.name}\n")
    idx, tpose, frames = parse_mvnx_positions(mvnx)
    need = {b[0] for b in BONES} | {b[1] for b in BONES if b[1]}
    missing = need - set(idx)
    print(f"segments needed present: {'ALL' if not missing else missing}")
    print(f"frames: {frames.shape}")

    Ra = compute_align(tpose, idx)
    print(f"\nR_align det={np.linalg.det(Ra):+.4f} (want +1)  orthonormal_err={np.abs(Ra@Ra.T-np.eye(3)).max():.2e}")

    # scalar-vs-vector parity on a random bone/frame sample
    rng = np.random.default_rng(0)
    P = frames @ Ra.T
    max_err = 0.0
    for i, (prox, dist, ref) in enumerate(BONES):
        if dist is None: continue
        fs = rng.choice(P.shape[0], 200, replace=False)
        vec = _bone_quats_vec(P[fs][:, idx[prox], :], P[fs][:, idx[dist], :], ref)
        for j, fr in enumerate(fs[:25]):
            sc = get_bone_quaternion(P[fr, idx[prox]], P[fr, idx[dist]], ref)
            d = min(np.linalg.norm(vec[j] - sc), np.linalg.norm(vec[j] + sc))  # quat double-cover
            max_err = max(max_err, d)
    print(f"vectorized vs scalar get_bone_quaternion  max_err={max_err:.2e} (want <1e-5)")

    # round-trip: global_quat applied to ref must recover the normalized bone vector
    rt_err = 0.0
    for i, (prox, dist, ref) in enumerate(BONES):
        if dist is None: continue
        q = _bone_quats_vec(P[:, idx[prox], :], P[:, idx[dist], :], ref)
        rot_ref = R.from_quat(q).apply(ref / np.linalg.norm(ref))
        bone = P[:, idx[dist], :] - P[:, idx[prox], :]
        bone = bone / np.linalg.norm(bone, axis=-1, keepdims=True)
        rt_err = max(rt_err, np.abs(rot_ref - bone).max())
    print(f"round-trip (q*ref == bone_dir)  max_err={rt_err:.2e} (want <1e-4)")

    Q = reconstruct_local_quats(frames, Ra, idx)
    norms = np.linalg.norm(Q, axis=-1)
    print(f"\nQ_local {Q.shape}  unit_norm_err={np.abs(norms-1).max():.2e}  finite={np.isfinite(Q).all()}")
    print(f"hand segs identity: seg6={np.allclose(Q[:,6],[0,0,0,1])} seg10={np.allclose(Q[:,10],[0,0,0,1])}")

    ok = (not missing) and abs(np.linalg.det(Ra)-1) < 1e-3 and max_err < 1e-5 and rt_err < 1e-4 \
         and np.abs(norms-1).max() < 1e-4 and np.isfinite(Q).all()
    print(f"\n{'PROTOTYPE PASS' if ok else 'PROTOTYPE FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Full batch
# ---------------------------------------------------------------------------
def main_full():
    data_root = (PROJECT_ROOT / DATA_ROOT_REL).resolve()
    anvil_root = (PROJECT_ROOT / ANVIL_ROOT_REL).resolve()
    offset_csv = (PROJECT_ROOT / OFFSET_CSV_REL).resolve()
    out_dir = (PROJECT_ROOT / OUT_DIR_REL).resolve()
    clear_output_dir(out_dir)

    offsets = load_offsets(offset_csv)
    print(f"Loaded {len(offsets)} session offsets -> {out_dir}")

    index_path = out_dir / "index.csv"
    n_written = 0
    with index_path.open("w", newline="", encoding="utf-8") as f_index:
        writer = csv.DictWriter(f_index, fieldnames=[
            "file", "session", "label", "start_sec", "end_sec", "offset_sec",
            "start_frame_240", "end_frame_240", "n_frames_30hz", "mode"])
        writer.writeheader()

        for session_id, offset_sec in offsets.items():
            sub_folder = session_id.split("-")[0]
            mvnx_path = data_root / sub_folder / f"{session_id}.mvnx"
            anvil_path = anvil_root / f"{session_id}.anvil"
            if not mvnx_path.exists() or not anvil_path.exists():
                print(f"  skip {session_id} (missing mvnx/anvil)"); continue
            print(f"Processing {session_id} ...", flush=True)

            idx, tpose, frames = parse_mvnx_positions(mvnx_path)
            Ra = compute_align(tpose, idx)
            Q_local = reconstruct_local_quats(frames, Ra, idx)

            counter = 0
            for (start_sec, end_sec, label_raw) in parse_anvil(anvil_path):
                label = sanitize_label(label_raw)
                clip, f0, f1 = slice_and_downsample(Q_local, start_sec, end_sec, offset_sec)
                if clip is None:
                    continue
                counter += 1
                out_name = f"{session_id}__{counter:04d}__{label}.npy"
                np.save(out_dir / out_name, clip)
                writer.writerow({
                    "file": out_name, "session": session_id, "label": label,
                    "start_sec": start_sec, "end_sec": end_sec, "offset_sec": offset_sec,
                    "start_frame_240": f0, "end_frame_240": f1,
                    "n_frames_30hz": clip.shape[0], "mode": MODE_TAG})
                n_written += 1
            print(f"  {session_id}: {counter} clips")
    print(f"\nDONE. {n_written} clips -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototype", action="store_true", help="one session + self-checks, no writes")
    args = ap.parse_args()
    if args.prototype:
        prototype()
    else:
        main_full()
