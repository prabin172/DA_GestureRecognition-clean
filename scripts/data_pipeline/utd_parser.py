#!/usr/bin/env python3
"""UTD-MHAD skeleton -> 17-segment LRQ, drop-in for the LOSO harness (second same-modality target).

UTD-MHAD uses Kinect-v1 with 20 joints in a DIFFERENT order/count from NTU/Kinect-v2's 25, so
(unlike CZU) we remap. Authoritative order from Sample_Code/Skeleton_joint_order.txt:
  1 head, 2 shoulder_center, 3 spine, 4 hip_center, 5-8 L arm(sho/elb/wri/hand),
  9-12 R arm, 13-16 L leg(hip/knee/ankle/foot), 17-20 R leg.
NTU 25 order (see czu_parser.py): 1 SpineBase,2 SpineMid,3 Neck,4 Head,5-8 L arm,9-12 R arm,
  13-16 L leg,17-20 R leg,21 SpineShoulder,22-25 hand tips/thumbs.

process_to_local_quats consumes NTU joints 1-21 only (22-25 unused). The one gap: NTU has both
Neck(3) and SpineShoulder(21) where Kinect-v1 has a single shoulder_center; we map BOTH to UTD's
shoulder_center (a defensible collapse — Kinect-v1 lacks a separate neck joint). Coordinate frame:
Kinect-v1 d_skel is Y-up, Z-depth (same camera convention as NTU); the demo's axis swap is
plotting-only. Validate with --smoke (bone CoV small + head above base).

Each .mat: d_skel (20,3,F). Output: (F,17,4) float32 LRQ npy + index.csv matching
Data_Processed/czu_skeleton_lrq/ so loso_fulltrain_calibration.py runs unmodified.
"""
import argparse, sys, re, csv
from pathlib import Path
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.data.ntu_parser import process_to_local_quats  # noqa: E402

SK_DIR = ROOT / "external_data/utd_mhad/Skeleton"
OUT_DIR = ROOT / "Data_Processed/utd_skeleton_lrq"

ACTIONS = {
    1: "swipe_left", 2: "swipe_right", 3: "wave", 4: "clap", 5: "throw",
    6: "arm_cross", 7: "basketball_shoot", 8: "draw_x", 9: "draw_circle_cw",
    10: "draw_circle_ccw", 11: "draw_triangle", 12: "bowling", 13: "boxing",
    14: "baseball_swing", 15: "tennis_swing", 16: "arm_curl", 17: "tennis_serve",
    18: "push", 19: "knock", 20: "catch", 21: "pickup_throw", 22: "jog",
    23: "walk", 24: "sit2stand", 25: "stand2sit", 26: "lunge", 27: "squat",
}
NAME_RE = re.compile(r"a(\d+)_s(\d+)_t(\d+)")

# NTU 1-based joint index -> UTD 1-based joint index (see module docstring).
NTU_FROM_UTD = {
    1: 4, 2: 3, 3: 2, 4: 1,                 # SpineBase<-hip_center, SpineMid<-spine, Neck<-shoulder_center, Head<-head
    5: 5, 6: 6, 7: 7, 8: 8,                 # L arm
    9: 9, 10: 10, 11: 11, 12: 12,           # R arm
    13: 13, 14: 14, 15: 15, 16: 16,         # L leg
    17: 17, 18: 18, 19: 19, 20: 20,         # R leg
    21: 2,                                  # SpineShoulder<-shoulder_center (== Neck source)
}


def load_positions(mat_path):
    """Returns (F, 25, 3) NTU-ordered joint xyz from UTD d_skel (20,3,F). 22-25 left zero (unused)."""
    d = sio.loadmat(mat_path)["d_skel"]                 # (20,3,F)
    utd = np.transpose(d, (2, 0, 1)).astype(np.float64) # (F,20,3)
    F = utd.shape[0]
    ntu = np.zeros((F, 25, 3), np.float64)
    for ntu_i, utd_i in NTU_FROM_UTD.items():
        ntu[:, ntu_i - 1] = utd[:, utd_i - 1]
    return ntu


def parse_name(stem):
    m = NAME_RE.match(stem)
    if not m:
        return None
    aid, subj, rep = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"s{subj}", aid, rep


def bone_len_cov(pos):
    """Rigid-bone sanity: low coeff-of-variation of bone length over frames (joint order right)."""
    J = [(1, 2), (2, 21), (3, 4), (5, 6), (6, 7), (9, 10), (10, 11), (1, 13),
         (13, 14), (14, 15), (1, 17), (17, 18), (18, 19)]
    covs = []
    for a, b in J:
        d = np.linalg.norm(pos[:, a - 1] - pos[:, b - 1], axis=1)
        if d.mean() > 1e-6:
            covs.append(d.std() / d.mean())
    return float(np.mean(covs)), float(np.max(covs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="validate a few clips, no write")
    args = ap.parse_args()

    mats = sorted(SK_DIR.glob("*.mat"))
    assert mats, f"no mat in {SK_DIR}"

    if args.smoke:
        print(f"{len(mats)} skeleton clips")
        for p in mats[:6]:
            info = parse_name(p.stem)
            pos = load_positions(p)
            lrq, _ = process_to_local_quats(pos)
            lrq = np.asarray(lrq)
            norms = np.linalg.norm(lrq.reshape(-1, 4), axis=1)
            mcov, xcov = bone_len_cov(pos)
            spine_up = (pos[:, 3, 1] - pos[:, 0, 1]).mean()   # head.y - spinebase.y
            print(f"  {p.stem}: subj={info[0]} a{info[1]} F={pos.shape[0]} "
                  f"lrq={lrq.shape} unit={np.allclose(norms,1,atol=1e-3)} "
                  f"boneCoV mean={mcov:.3f} max={xcov:.3f} head-above-base={spine_up:.2f}m")
        print("SMOKE: bone CoV should be small (<~0.08) and head-above-base>0 if joint order is right")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for i, p in enumerate(mats):
        info = parse_name(p.stem)
        if info is None:
            skipped += 1; continue
        subj, aid, rep = info
        pos = load_positions(p)
        F = pos.shape[0]
        if F < 8:
            skipped += 1; continue
        lrq, _ = process_to_local_quats(pos)
        lrq = np.asarray(lrq, dtype=np.float32)
        fname = f"{subj}__a{aid:02d}__t{rep:02d}.npy"
        np.save(OUT_DIR / fname, lrq)
        rows.append({"file": fname, "session": subj, "label": ACTIONS[aid],
                     "n_frames_30hz": F, "mode": "utd_skeleton"})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(mats)}")

    with open(OUT_DIR / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "session", "label", "n_frames_30hz", "mode"])
        w.writeheader(); w.writerows(rows)
    subs = sorted(set(r["session"] for r in rows))
    print(f"DONE {len(rows)} clips ({skipped} skipped) -> {OUT_DIR}")
    print(f"subjects={subs}  actions={len(ACTIONS)}")


if __name__ == "__main__":
    main()
