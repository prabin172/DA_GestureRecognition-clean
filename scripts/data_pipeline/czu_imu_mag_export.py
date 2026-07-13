#!/usr/bin/env python3
"""Export CZU inertial MAGNITUDE signal — the MIDDLE point of the target-strength dial (item 1).

Between R6b (quat-only, orientation, 0 target channels) and R6c (full raw 10x6=60ch), this gives a
coarse-dynamics target: per-sensor accel-magnitude and gyro-magnitude (rotation-invariant, no
per-axis direction) = 10 sensors x 2 = 20 channels/frame. Richer than orientation-only (adds motion
energy) but strictly less than the full 6-axis raw signal -> a monotone rung on the "how strong is
the target representation" axis.

Mirrors czu_imu_raw_export.py exactly (same .mat source, same recovered sensor order, same
resample-to-quat-n_frames, same global per-channel z-score) so masks align frame-for-frame with the
quat prior branch and clips are paired with czu_imu_raw / czu_imu_quats.

Output -> Data_Processed/czu_imu_mag20/  ({T,20} float32 npy + index.csv, matching filenames)
"""
import csv
from pathlib import Path
import numpy as np
import scipy.io as sio
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SENSOR_DIR = ROOT / "external_data/czu_mhad_data/CZU-MHAD/sensor_mat"
QUAT_DIR = ROOT / "Data_Processed/czu_imu_quats"          # source of n_frames + filenames + labels
OUT_DIR = ROOT / "Data_Processed/czu_imu_mag20"
DT_CLAMP_S = 0.02
N_CH = 20                                                 # 10 sensors x (accel-mag, gyro-mag)

# reuse the filename->mat mapping identical to the raw exporter
import re
NAME_RE = re.compile(r"([a-z]+)__a(\d+)__t(\d+)")


def npy_to_mat(npy_name):
    m = NAME_RE.match(npy_name.replace(".npy", ""))
    subj, aid, rep = m.group(1), int(m.group(2)), int(m.group(3))
    return SENSOR_DIR / f"{subj}_a{aid}_t{rep}.mat"


def mag_signal(mat_path, n_out):
    """(n_out, 20) resampled per-sensor [|accel|, |gyro|]. Same cleaning/interp as raw exporter."""
    sensor = sio.loadmat(mat_path)["sensor"]              # (10,1) object
    cols = []
    for i in range(sensor.shape[0]):
        a = sensor[i, 0]
        ts = a[:, 6].astype(np.float64)
        order = np.argsort(ts, kind="stable")
        a = a[order]; ts = ts[order]
        sig = a[:, :6].astype(np.float64)                 # accel(3)+gyro(3)
        amag = np.linalg.norm(sig[:, :3], axis=1)         # |accel|
        gmag = np.linalg.norm(sig[:, 3:6], axis=1)        # |gyro|
        dt = np.diff(ts) / 1e6
        dt = np.clip(dt, 1e-4, DT_CLAMP_S)
        t = np.concatenate([[0.0], np.cumsum(dt)])
        out_t = np.linspace(t[0], t[-1], n_out)
        cols.append(np.stack([np.interp(out_t, t, amag), np.interp(out_t, t, gmag)], axis=1))
    return np.concatenate(cols, axis=1).astype(np.float32)  # (n_out, 20)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(QUAT_DIR / "index.csv")
    rows, acc_sum, acc_sumsq, n_tot = [], np.zeros(N_CH), np.zeros(N_CH), 0
    for _, r in idx.iterrows():
        n_out = int(r["n_frames_30hz"])
        sig = mag_signal(npy_to_mat(r["file"]), n_out)     # (T,20) unstandardized
        np.save(OUT_DIR / r["file"], sig)
        acc_sum += sig.sum(0); acc_sumsq += (sig ** 2).sum(0); n_tot += sig.shape[0]
        rows.append(dict(file=r["file"], session=r["session"], label=r["label"],
                         n_frames_30hz=n_out, mode="czu_imu_mag20"))
    mean = acc_sum / n_tot
    std = np.sqrt(np.maximum(acc_sumsq / n_tot - mean ** 2, 1e-8))
    np.savez(OUT_DIR / "channel_stats.npz", mean=mean.astype(np.float32), std=std.astype(np.float32))
    for r in rows:
        p = OUT_DIR / r["file"]
        sig = np.load(p)
        np.save(p, ((sig - mean) / std).astype(np.float32))
    with (OUT_DIR / "index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "session", "label", "n_frames_30hz", "mode"])
        w.writeheader(); w.writerows(rows)
    print(f"Done. {len(rows)} mag20 clips -> {OUT_DIR}")
    print(f"channel mean range [{mean.min():.3f},{mean.max():.3f}]  std range [{std.min():.3f},{std.max():.3f}]")


if __name__ == "__main__":
    main()
