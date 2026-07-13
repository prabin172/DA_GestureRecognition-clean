#!/usr/bin/env python3
"""tasks.md T4 — UTD-MHAD published-baseline anchor, mirrors crc_baseline.py (CZU).

Same statistical-moment features + Collaborative Representation Classifier (CRC-RLS,
lambda=1e-4) recognizer family used for the CZU-MHAD R6 anchor, applied to UTD-MHAD's
shared LRQ representation on the byte-identical LOSO k-shot splits the learned recognizer
used (trained_models/UTD-skeleton-LOSO-seed42/splits) -> a same-splits head-to-head, and
a literature anchor row for R6d. Deterministic given splits -> no seeds needed (uses the
seed-42 splits only, matching the R6 CRC baseline's single-comparison convention).

Outputs -> trained_models/UTD-skeleton-LOSO-seed42/crc_baseline/
  crc_per_fold.csv     per (subject,k) accuracy + macro-F1
  crc_summary.csv      per-k mean over 8 folds
  comparison.csv       CRC baseline vs the learned recognizer (seed-42 summary.csv)
"""
import os, json, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, "Data_Processed/utd_skeleton_lrq")
RUN = os.path.join(ROOT, "trained_models/UTD-skeleton-LOSO-seed42")
SPLITS = os.path.join(RUN, "splits")
OUT = os.path.join(RUN, "crc_baseline")
SUBJECTS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
KS = [0, 1, 3]
LAMBDA = 1e-4

try:
    from scipy.stats import skew, kurtosis
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def clip_features(path):
    """(T,17,4) LRQ clip -> 68*5 statistical-moment feature vector.
    Drops zero-padding frames (all-zero rows); moments per flattened dim over time."""
    a = np.load(path).astype(np.float64)          # (T,17,4)
    a = a.reshape(a.shape[0], -1)                  # (T,68)
    keep = np.abs(a).sum(axis=1) > 1e-8            # drop all-zero (pad) frames
    a = a[keep] if keep.any() else a
    mean = a.mean(0); std = a.std(0); var = a.var(0)
    if HAVE_SCIPY:
        sk = skew(a, axis=0, bias=False)
        ku = kurtosis(a, axis=0, bias=False)
    else:
        c = a - mean
        s = np.where(std > 1e-8, std, 1.0)
        sk = (c ** 3).mean(0) / s ** 3
        ku = (c ** 4).mean(0) / s ** 4 - 3.0
    f = np.concatenate([mean, std, var, sk, ku])
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def label_of(fname):
    return fname.split("__")[1]                    # 'a01'..'a27' == class id


def crc_predict(Xtr, ytr, Xte, lam=LAMBDA):
    """CRC-RLS. Xtr:(d,N) columns=dictionary atoms. Returns predicted class ids for Xte:(d,M)."""
    d, N = Xtr.shape
    G = Xtr.T @ Xtr + lam * np.eye(N)              # (N,N)
    Proj = np.linalg.solve(G, Xtr.T)               # (N,d)
    Coef = Proj @ Xte                              # (N,M)
    classes = np.unique(ytr)
    M = Xte.shape[1]
    res = np.full((len(classes), M), np.inf)
    for ci, c in enumerate(classes):
        m = ytr == c
        Xc = Xtr[:, m]; Cc = Coef[m]               # (d,nc),(nc,M)
        recon = Xc @ Cc                            # (d,M)
        num = np.linalg.norm(Xte - recon, axis=0)
        den = np.linalg.norm(Cc, axis=0) + 1e-12
        res[ci] = num / den
    return classes[np.argmin(res, axis=0)]


def macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"scipy moments: {HAVE_SCIPY}")
    all_files = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(DATA, "*.npy")))]
    feat = {fn: clip_features(os.path.join(DATA, fn)) for fn in all_files}
    lab = {fn: label_of(fn) for fn in all_files}
    classes = np.array(sorted(set(lab.values())))
    print(f"{len(all_files)} clips, {len(classes)} classes, feat-dim {len(next(iter(feat.values())))}")

    rows = []
    for k in KS:
        for s in SUBJECTS:
            sp = json.load(open(os.path.join(SPLITS, f"{s}_k{k}_calibration_split.json")))
            calib = sp["calib_files"]; ev = sp["eval_files"]
            train_files = [fn for fn in all_files if fn.split("__")[0] != s] + list(calib)
            Xtr = np.stack([feat[fn] for fn in train_files]).T          # (d,N)
            ytr = np.array([lab[fn] for fn in train_files])
            Xte = np.stack([feat[fn] for fn in ev]).T                   # (d,M)
            yte = np.array([lab[fn] for fn in ev])
            mu = Xtr.mean(1, keepdims=True); sd = Xtr.std(1, keepdims=True) + 1e-8
            Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
            Xtr /= (np.linalg.norm(Xtr, axis=0, keepdims=True) + 1e-12)
            Xte /= (np.linalg.norm(Xte, axis=0, keepdims=True) + 1e-12)
            pred = crc_predict(Xtr, ytr, Xte)
            acc = float((pred == yte).mean()) * 100
            f1 = macro_f1(yte, pred, classes)
            rows.append(dict(test_subject=s, k=k, n_train=len(train_files),
                             eval_samples=len(ev), acc=round(acc, 3),
                             macro_f1=round(f1, 4)))
            print(f"  k={k} subj={s}: acc={acc:.2f} (train {len(train_files)}, eval {len(ev)})")
    pf = pd.DataFrame(rows)
    pf.to_csv(os.path.join(OUT, "crc_per_fold.csv"), index=False)

    summ = pf.groupby("k").agg(crc_acc=("acc", "mean"), crc_acc_sd=("acc", "std"),
                               crc_macro_f1=("macro_f1", "mean")).round(3).reset_index()
    summ.to_csv(os.path.join(OUT, "crc_summary.csv"), index=False)
    print("\nCRC baseline (mean over 8 LOSO folds, UTD-MHAD):")
    print(summ.to_string(index=False))

    rec = pd.read_csv(os.path.join(RUN, "summary.csv"))
    rec_mean = rec.groupby(["method", "k"]).final_acc.mean().reset_index()
    comp = summ[["k", "crc_acc"]].copy()
    for m in ["scratch", "mae", "supMAE", "supLP120"]:
        comp[m] = comp.k.map(
            rec_mean[rec_mean.method == m].set_index("k").final_acc.round(2))
    comp = comp.rename(columns={"crc_acc": "CRC_baseline"})
    comp.to_csv(os.path.join(OUT, "comparison.csv"), index=False)
    print("\nComparison — CRC baseline vs learned recognizer (seed-42 acc, 8 folds):")
    print(comp.to_string(index=False))
    print(f"\nWrote -> {OUT}")


if __name__ == "__main__":
    main()
