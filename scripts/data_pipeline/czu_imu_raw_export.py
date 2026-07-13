#!/usr/bin/env python3
"""Export CZU inertial RAW 6-axis signal (accel+gyro), aligned frame-for-frame to the quat clips.

Companion to czu_imu_parser.py. Where the parser reduces the IMU to orientation quaternions
(the shared space with NTU skeleton), THIS keeps the full raw signal CRC uses — 10 sensors x
6 channels (accel g + gyro deg/s) = 60 channels/frame. Resampled to the SAME n_frames as the
matching quat clip so masks align for a dual-branch (quat prior + raw target-only) model.

Per-channel z-score standardization (global train stats) so the transformer sees well-scaled
input (accel ~1g vs gyro ~tens deg/s). Stats saved to channel_stats.npz.

Output -> Data_Processed/czu_imu_raw/  ({T,60} float32 npy + index.csv, matching filenames)
"""
import re, csv
from pathlib import Path
import numpy as np
import scipy.io as sio
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SENSOR_DIR = ROOT / "external_data/czu_mhad_data/CZU-MHAD/sensor_mat"
QUAT_DIR = ROOT / "Data_Processed/czu_imu_quats"          # source of n_frames + filenames + labels
OUT_DIR = ROOT / "Data_Processed/czu_imu_raw"
DT_CLAMP_S = 0.02
NAME_RE = re.compile(r"([a-z]+)__a(\d+)__t(\d+)")
# recovered sensor order == native .mat order (see czu_imu_parser.py):
# 0=Abdomen,1=Chest,2=L Elbow,3=L Wrist,4=R Elbow,5=R Wrist,6=L Knee,7=L Ankle,8=R Knee,9=R Ankle


def npy_to_mat(npy_name):
    m = NAME_RE.match(npy_name.replace(".npy", ""))
    subj, aid, rep = m.group(1), int(m.group(2)), int(m.group(3))
    return SENSOR_DIR / f"{subj}_a{aid}_t{rep}.mat"


def raw_signal(mat_path, n_out):
    """(n_out, 60) resampled raw accel+gyro. Linear interp per channel over cleaned cumulative time."""
    sensor = sio.loadmat(mat_path)["sensor"]              # (10,1) object
    cols = []
    for i in range(sensor.shape[0]):
        a = sensor[i, 0]
        ts = a[:, 6].astype(np.float64)
        order = np.argsort(ts, kind="stable")
        a = a[order]; ts = ts[order]
        sig = a[:, :6].astype(np.float64)                 # accel(3)+gyro(3)
        dt = np.diff(ts) / 1e6
        dt = np.clip(dt, 1e-4, DT_CLAMP_S)
        t = np.concatenate([[0.0], np.cumsum(dt)])
        out_t = np.linspace(t[0], t[-1], n_out)
        cols.append(np.stack([np.interp(out_t, t, sig[:, c]) for c in range(6)], axis=1))
    return np.concatenate(cols, axis=1).astype(np.float32)  # (n_out, 60)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(QUAT_DIR / "index.csv")
    rows, acc_sum, acc_sumsq, n_tot = [], np.zeros(60), np.zeros(60), 0
    # first pass: write clips, accumulate stats
    for _, r in idx.iterrows():
        n_out = int(r["n_frames_30hz"])
        sig = raw_signal(npy_to_mat(r["file"]), n_out)     # (T,60) unstandardized
        np.save(OUT_DIR / r["file"], sig)                  # store RAW; standardize at load or now?
        acc_sum += sig.sum(0); acc_sumsq += (sig ** 2).sum(0); n_tot += sig.shape[0]
        rows.append(dict(file=r["file"], session=r["session"], label=r["label"],
                         n_frames_30hz=n_out, mode="czu_imu_raw"))
    mean = acc_sum / n_tot
    std = np.sqrt(np.maximum(acc_sumsq / n_tot - mean ** 2, 1e-8))
    np.savez(OUT_DIR / "channel_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    # second pass: standardize in place (so the training loader can read directly)
    for r in rows:
        p = OUT_DIR / r["file"]
        sig = np.load(p)
        np.save(p, ((sig - mean) / std).astype(np.float32))
    with (OUT_DIR / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "session", "label", "n_frames_30hz", "mode"])
        w.writeheader(); w.writerows(rows)
    print(f"Done. {len(rows)} raw clips -> {OUT_DIR}")
    print(f"channel mean range [{mean.min():.3f},{mean.max():.3f}]  std range [{std.min():.3f},{std.max():.3f}]")
    print(f"n_frames match quat: {all(int(r['n_frames_30hz'])>0 for r in rows)}")


if __name__ == "__main__":
    main()
