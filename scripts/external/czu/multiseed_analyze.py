#!/usr/bin/env python3
"""Pool CZU seeds 42/43/44 and recompute R6/R6b/R6c stats.

R6  skeleton (CZU-skeleton-LOSO)  : small-gap, supLP120 win
R6b quat     (CZU-IMU-LOSO)       : cross-modal, supMAE win / supLP120 worst
R6c dual     (CZU-IMU-DUAL/dual_*) : strong target, prior adds nothing

Sign tests pool per (subject,k,seed) cells, dropping ties (two-sided binomtest).
Paired-t pools the same cells (ttest_rel). Original single-seed used n=5 (means)
and (subject,k) cells for sign tests.
"""
import pandas as pd, numpy as np
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent / "trained_models"
SEEDS = {42: "", 43: "-seed43", 44: "-seed44"}
KS = [0, 1, 3]
METHODS = ["scratch", "mae", "supMAE", "supLP120"]


def load_loso(base):
    """CZU-skeleton-LOSO / CZU-IMU-LOSO style summaries -> long df."""
    rows = []
    for seed, suf in SEEDS.items():
        f = ROOT / f"{base}{suf}" / "summary.csv"
        d = pd.read_csv(f)
        d = d[["test_subject", "method", "k", "final_acc"]].copy()
        d["seed"] = seed
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def load_dual(prior):
    """CZU-IMU-DUAL/dual_<prior>/summary.csv -> long df (mode=dual)."""
    rows = []
    for seed, suf in SEEDS.items():
        f = ROOT / f"CZU-IMU-DUAL{suf}" / f"dual_{prior}" / "summary.csv"
        d = pd.read_csv(f)  # cols: test_subject,mode,prior,k,eval_samples,final_acc
        d = d[["test_subject", "prior", "k", "final_acc"]].copy()
        d["method"] = prior
        d["seed"] = seed
        rows.append(d[["test_subject", "method", "k", "final_acc", "seed"]])
    return pd.concat(rows, ignore_index=True)


def mean_table(df, label):
    print(f"\n=== {label}: mean final_acc over folds (pooled 3 seeds x 5 subj = 15/fold-k) ===")
    piv = df.pivot_table(index="k", columns="method", values="final_acc", aggfunc="mean")
    piv = piv[[m for m in METHODS if m in piv.columns]]
    print(piv.round(2).to_string())
    return piv


def paired_cells(df, a, b):
    """Return aligned arrays of method a vs b over (subject,k,seed) cells."""
    keys = ["test_subject", "k", "seed"]
    da = df[df.method == a].set_index(keys)["final_acc"]
    db = df[df.method == b].set_index(keys)["final_acc"]
    idx = da.index.intersection(db.index)
    return da.loc[idx].values, db.loc[idx].values


def signtest(df, a, b):
    xa, xb = paired_cells(df, a, b)
    diff = xa - xb
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    n = wins + losses
    p = stats.binomtest(wins, n, 0.5).pvalue if n > 0 else float("nan")
    print(f"  sign  {a} > {b}: {wins}/{n}  (ties={ties})  binom p={p:.4f}  meanΔ={diff.mean():+.2f}")
    return wins, n, p


def paired_t_by_k(df, a, b):
    print(f"  paired-t {a} - {b} by k (cells = 5 subj x 3 seed = 15):")
    for k in KS:
        sub = df[df.k == k]
        xa, xb = paired_cells(sub, a, b)
        d = xa - xb
        t, p = stats.ttest_rel(xa, xb)
        print(f"    k={k}: Δ={d.mean():+.2f} pp  t={t:+.3f}  p={p:.4f}  (n={len(d)})")


# ---------------- R6 skeleton ----------------
print("#" * 70)
print("# R6 skeleton->skeleton (CZU-skeleton-LOSO), pooled seeds 42/43/44")
print("#" * 70)
sk = load_loso("CZU-skeleton-LOSO")
mean_table(sk, "R6 skeleton")
print("\nKey claim: supLP120 zero-shot beats scratch (single-seed was +8.3pp p=.07)")
paired_t_by_k(sk, "supLP120", "scratch")
paired_t_by_k(sk, "supMAE", "scratch")
paired_t_by_k(sk, "mae", "scratch")

# ---------------- R6b quat ----------------
print("\n" + "#" * 70)
print("# R6b cross-modal quat (CZU-IMU-LOSO), pooled seeds 42/43/44")
print("#" * 70)
q = load_loso("CZU-IMU-LOSO")
mean_table(q, "R6b quat")
print("\nSign tests pooled over (subj,k,seed) cells (single-seed was 12/15 & 13/15):")
signtest(q, "supMAE", "scratch")
signtest(q, "supMAE", "supLP120")
signtest(q, "supMAE", "mae")
signtest(q, "supLP120", "scratch")  # was worst, below scratch
paired_t_by_k(q, "supMAE", "scratch")
paired_t_by_k(q, "supLP120", "scratch")

# ---------------- R6c dual ----------------
print("\n" + "#" * 70)
print("# R6c dual-branch (CZU-IMU-DUAL/dual_*), pooled seeds 42/43/44")
print("#" * 70)
dual = pd.concat([load_dual(p) for p in METHODS], ignore_index=True)
mean_table(dual, "R6c dual")
print("\nSign tests vs dual/scratch (single-seed: supMAE 7/15 p=1.0; supLP120 1/13 p=.003):")
signtest(dual, "supMAE", "scratch")
signtest(dual, "mae", "scratch")
signtest(dual, "supLP120", "scratch")
paired_t_by_k(dual, "supMAE", "scratch")
paired_t_by_k(dual, "supLP120", "scratch")
print("\nNote: raw-only and quat DUAL modes are single-seed (diagnostic) -> not pooled.")
