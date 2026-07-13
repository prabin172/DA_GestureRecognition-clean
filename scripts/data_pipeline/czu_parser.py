#!/usr/bin/env python3
"""CZU-MHAD skeleton -> 17-segment LRQ, drop-in for the LOSO harness.

Key fact (verified from demo/skeleton_display.m bone list J): CZU stores the SAME
NTU/Kinect-v2 25-joint order (1=SpineBase, 2=SpineMid, 3=Neck, 4=Head,
5=ShoulderL..8=HandL, 9=ShoulderR..12=HandR, 13=HipL..16=FootL, 17=HipR..20=FootR,
21=SpineShoulder, 22-25=hand tips/thumbs), Y-up. So CZU joint POSITIONS feed our
own NTU builder (src/data/ntu_parser.process_to_local_quats) with NO reindexing.

Each skeleton .mat: (F, 100) = (frames, 25 joints x [x,y,z,timestamp]).
Output per clip: (F, 17, 4) float32 LRQ npy + index.csv (file,session,label,n_frames_30hz)
matching Data_Processed/imu_quats_v2/ so loso_fulltrain_calibration.py runs unmodified.
"""
import argparse, sys, re, csv
from pathlib import Path
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.data.ntu_parser import process_to_local_quats  # noqa: E402

SK_DIR = ROOT / "external_data/czu_mhad_data/CZU-MHAD/skeleton_mat"
OUT_DIR = ROOT / "Data_Processed/czu_skeleton_lrq"

ACTIONS = {
    1: "right_high_wave", 2: "left_high_wave", 3: "right_horiz_wave", 4: "left_horiz_wave",
    5: "hammer_right", 6: "grasp_right", 7: "draw_fork_right", 8: "draw_fork_left",
    9: "draw_circle_right", 10: "draw_circle_left", 11: "right_kick_fwd", 12: "left_kick_fwd",
    13: "right_kick_side", 14: "left_kick_side", 15: "clap", 16: "bend_down",
    17: "wave_up_down", 18: "sur_place", 19: "left_body_turn", 20: "right_body_turn",
    21: "left_lateral", 22: "right_lateral",
}
NAME_RE = re.compile(r"([a-z]+)_a(\d+)_t(\d+)")


def load_positions(mat_path):
    """Returns (F, 25, 3) joint xyz. Drops the per-joint timestamp column."""
    sk = sio.loadmat(mat_path)["skeleton"]          # (F, 100)
    F = sk.shape[0]
    arr = sk.reshape(F, 25, 4)[:, :, :3].astype(np.float64)
    return arr


def parse_name(stem):
    m = NAME_RE.match(stem)
    if not m:
        return None
    subj, aid, rep = m.group(1), int(m.group(2)), int(m.group(3))
    return subj, aid, rep


def bone_len_cov(pos):
    """Ordering sanity: rigid bones -> low coeff-of-variation of length over frames.
    Uses the demo J bone pairs (1-indexed joints)."""
    J = [(1,2),(2,21),(3,4),(3,21),(5,6),(6,7),(9,10),(10,11),(1,13),(13,14),
         (14,15),(1,17),(17,18),(18,19)]
    covs = []
    for a, b in J:
        d = np.linalg.norm(pos[:, a-1] - pos[:, b-1], axis=1)
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
        for p in mats[:4]:
            info = parse_name(p.stem)
            pos = load_positions(p)
            lrq, _ = process_to_local_quats(pos)      # (F,17,4)
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
                     "n_frames_30hz": F, "mode": "czu_skeleton"})
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(mats)}")

    with open(OUT_DIR / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "session", "label", "n_frames_30hz", "mode"])
        w.writeheader(); w.writerows(rows)
    subs = sorted(set(r["session"] for r in rows))
    print(f"DONE {len(rows)} clips ({skipped} skipped) -> {OUT_DIR}")
    print(f"subjects={subs}  actions={len(ACTIONS)}")


if __name__ == "__main__":
    main()
