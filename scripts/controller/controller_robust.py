#!/usr/bin/env python3
"""Phase 3 / C6 — ROBUSTNESS version: lock the controller design knobs by showing
the method ORDERING is invariant to them, rather than freezing arbitrary values.

Three locks (each kills a reviewer objection to controller_sim.py):
  1. Command-gesture set  -> randomize over many vocabularies; report the
     DISTRIBUTION of the method ordering (kills "you cherry-picked easy gestures").
  2. Critical-error rule   -> two outcome models (hard-safety binary + soft-cost)
     with a swept critical-cost C_crit; show ordering invariant (kills "the harsh
     instant-fail rule drives your result").
  3. Operating threshold   -> ISO-SAFETY operating point: fix a false-activation
     budget, find tau meeting it per method, compare throughput (kills "you tuned
     tau to win"). Full tau-frontier also reported.

Consumes the same real held-out posteriors as controller_sim.py.
Outputs -> trained_models/Phase3-controller/robust/
  vocab_sweep.csv        per (vocab, method, k, tau_regime): hard success + soft cost
  vocab_ordering.csv     distribution of pairwise method deltas across vocabularies
  costmodel_sweep.csv    per (method, k, C_crit): soft mean-cost (fixed vocab)
  iso_safety.csv         per (method, k, budget): tau*, task_success, mean_time
  *.png figures
"""
import os, glob, json, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNDIRS = ["trained_models/LOSO-fullTrainCalibrate-v2",
           "trained_models/LOSO-fullTrainCalibrate-v2-seed43",
           "trained_models/LOSO-fullTrainCalibrate-v2-seed44"]
OUT = os.path.join(ROOT, "trained_models/Phase3-controller/robust")
# Clean rerun (DA_GestureRecognition-clean, 2026-07-13): all 5 methods including supcon are
# trained together in each seed dir from the start (scripts/orchestration/02_main_loso.sh) --
# no incremental "-supcon-seed*" patch dirs needed, unlike the original repo's history.
METHODS = ["scratch", "mae", "supMAE", "supLP120", "supcon"]

PRIMS = ["next", "previous", "approach", "grasp", "release", "confirm", "cancel"]
CRIT_PRIMS = {PRIMS.index("grasp"), PRIMS.index("release")}   # {3,4}
# 12-step pick-place mission as primitive indices (2 grasp, 1 release)
MISSION = [0, 0, 1, 2, 3, 5, 0, 2, 3, 4, 5, 6]

TAUS = np.array([0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99])


def load():
    frames = [pd.read_csv(f) for d in RUNDIRS
              for f in glob.glob(os.path.join(ROOT, d, "posteriors", "*.csv"))]
    return pd.concat(frames, ignore_index=True)


def pack(df, labelmap):
    """method -> k -> {gid: (conf[], pred_id[])}, plus a global true_id array per slice."""
    packed = {}
    for method in METHODS:
        packed[method] = {}
        for k in (1, 3):
            sl = df[(df.method == method) & (df.k == k)]
            g = {}
            for gid in range(len(labelmap)):
                rows = sl[sl.true_id == gid]
                g[gid] = (rows.conf.to_numpy(np.float32), rows.pred_id.to_numpy(np.int64))
            packed[method][k] = dict(by_g=g,
                                     true=sl.true_id.to_numpy(np.int64),
                                     pred=sl.pred_id.to_numpy(np.int64),
                                     conf=sl.conf.to_numpy(np.float32))
    return packed


def simulate(slice_data, prim_of_id, tau, C_crit, costs, rng, n_missions):
    """One (method,k,vocab,tau,C_crit). Single pass yields BOTH outcome models:
      hard_success = fraction of missions with zero critical errors & no reject-abort
      mean_cost    = soft model: critical errors are recoverable but cost C_crit
    """
    T_exec, T_reject, T_correct, MAX_REJECT = costs
    by_g = slice_data["by_g"]
    gest_of_prim = {int(prim_of_id[g]): g for g in range(len(prim_of_id)) if prim_of_id[g] >= 0}
    hard_ok_n = 0
    tot_cost = 0.0
    rej = 0; corr = 0; pres = 0
    for _ in range(n_missions):
        cost = 0.0; had_crit = False; aborted = False
        for intended in MISSION:
            g = gest_of_prim[intended]
            conf_a, pred_a = by_g[g]
            nrej = 0
            while True:
                i = rng.integers(len(conf_a)); pres += 1
                conf = conf_a[i]; pp = prim_of_id[pred_a[i]]
                if conf < tau:
                    cost += T_reject; rej += 1; nrej += 1
                    if nrej > MAX_REJECT:
                        aborted = True; break
                    continue
                cost += T_exec
                if pp == intended:
                    break                                   # correct
                if (intended in CRIT_PRIMS) or (pp in CRIT_PRIMS):
                    cost += C_crit; had_crit = True; break  # critical (soft: recover)
                cost += T_correct; corr += 1; break         # recoverable
            if aborted:
                break
        tot_cost += cost
        if not had_crit and not aborted:
            hard_ok_n += 1
    return dict(hard_success=hard_ok_n / n_missions,
                mean_cost=tot_cost / n_missions,
                rejection_rate=rej / max(pres, 1),
                corrective_per_mission=corr / n_missions)


def false_activation(slice_data, prim_of_id, tau, vocab_ids, rng, cap=8000):
    """distractor = clips whose TRUE class is not a command gesture."""
    true = slice_data["true"]; pred = slice_data["pred"]; conf = slice_data["conf"]
    mask = ~np.isin(true, vocab_ids)
    dt, dp, dc = true[mask], pred[mask], conf[mask]
    if len(dc) > cap:
        idx = rng.choice(len(dc), size=cap, replace=False)
        dp, dc = dp[idx], dc[idx]
    passed = dc >= tau
    if passed.sum() == 0:
        return 0.0
    acts = prim_of_id[dp[passed]] >= 0
    return float(acts.sum()) / len(dc)


def make_prim_of_id(vocab_ids, n_classes):
    """vocab_ids: list of 7 gesture ids in primitive order 0..6."""
    arr = -np.ones(n_classes, dtype=np.int64)
    for prim_idx, gid in enumerate(vocab_ids):
        arr[gid] = prim_idx
    return arr


def reliability_ordered_vocab(df, labelmap, exclude_locomotion=True):
    """The illustrative named mapping: rank by pooled k=3 recall, most-reliable -> most-critical."""
    k3 = df[df.k == 3]
    rec = k3.assign(ok=(k3.pred_id == k3.true_id)).groupby("true_id")["ok"].mean()
    EXCL = {"walk", "runonspot", "turnaround", "buttkicks", "hop", "jump", "stand"}
    inv = {v: kk for kk, v in labelmap.items()}
    ranked = [gid for gid in rec.sort_values(ascending=False).index
              if not (exclude_locomotion and inv[gid] in EXCL)]
    # assignment order: most reliable -> grasp, release, confirm, approach, cancel, next, previous
    prim_assign_order = ["grasp", "release", "confirm", "approach", "cancel", "next", "previous"]
    vocab = [None] * 7
    for gid, prim in zip(ranked, prim_assign_order):
        vocab[PRIMS.index(prim)] = gid
    return vocab


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--missions", type=int, default=1000)
    ap.add_argument("--vocabs", type=int, default=120)
    ap.add_argument("--frontier-missions", type=int, default=3000)
    ap.add_argument("--out-dir", default=None,
                     help="Override OUT (default trained_models/Phase3-controller/robust, "
                          "which holds the LOCKED R5 numbers already cited in paper_results.md "
                          "-- use a NEW dir, e.g. .../robust-supcon, for any re-run that changes "
                          "METHODS so the locked numbers are never overwritten).")
    args = ap.parse_args()
    if args.out_dir is not None:
        OUT = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    os.makedirs(OUT, exist_ok=True)
    labelmap = json.load(open(os.path.join(
        ROOT, "trained_models/LOSO-fullTrainCalibrate-v2/label_map.json")))
    n_classes = len(labelmap)
    inv = {v: k for k, v in labelmap.items()}

    df = load()
    packed = pack(df, labelmap)
    costs = (1.0, 1.0, 3.0, 5)          # T_exec, T_reject, T_correct, MAX_REJECT
    named_vocab = reliability_ordered_vocab(df, labelmap)
    print("Illustrative (reliability-ordered) vocabulary:")
    for p, gid in zip(PRIMS, named_vocab):
        print(f"  {p:9s} <- {inv[gid]}")

    # ---------- LOCK 1: randomized command-vocabulary sweep ----------
    # random 7 distinct gestures -> 7 primitives, each vocab tested at tau=0 (full
    # compounding) and tau=0.9 (gated). C_crit=inf recovered via hard_success.
    rng = np.random.default_rng(args.seed)
    all_ids = np.arange(n_classes)
    vocab_rows = []
    for v in range(args.vocabs):
        vids = list(rng.choice(all_ids, size=7, replace=False))
        prim_of_id = make_prim_of_id(vids, n_classes)
        for k in (1, 3):
            for tau in (0.0, 0.9):
                for method in METHODS:
                    r = simulate(packed[method][k], prim_of_id, tau, C_crit=20.0,
                                 costs=costs, rng=np.random.default_rng(1000 + v),
                                 n_missions=args.missions)
                    vocab_rows.append(dict(vocab=v, method=method, k=k, tau=tau, **r))
        if (v + 1) % 20 == 0:
            print(f"  vocab sweep {v+1}/{args.vocabs}")
    vdf = pd.DataFrame(vocab_rows)
    vdf.to_csv(os.path.join(OUT, "vocab_sweep.csv"), index=False)

    # ordering distribution: per (k,tau) pairwise deltas vs each method
    ord_rows = []
    for k in (1, 3):
        for tau in (0.0, 0.9):
            piv = vdf[(vdf.k == k) & (vdf.tau == tau)].pivot_table(
                index="vocab", columns="method", values="hard_success")
            for a in METHODS:
                for b in METHODS:
                    if a == b:
                        continue
                    d = piv[a] - piv[b]
                    ord_rows.append(dict(
                        k=k, tau=tau, a=a, b=b,
                        median_delta=round(float(d.median()), 4),
                        iqr_lo=round(float(d.quantile(.25)), 4),
                        iqr_hi=round(float(d.quantile(.75)), 4),
                        frac_a_ge_b=round(float((d >= 0).mean()), 4)))
    odf = pd.DataFrame(ord_rows)
    odf.to_csv(os.path.join(OUT, "vocab_ordering.csv"), index=False)
    print("\nLOCK1 ordering — fraction of vocabularies where row-method >= col-method (hard_success):")
    for k in (1, 3):
        for tau in (0.0, 0.9):
            m = odf[(odf.k == k) & (odf.tau == tau)].pivot(index="a", columns="b",
                                                           values="frac_a_ge_b")
            print(f"  k={k} tau={tau}:")
            print(m.reindex(index=METHODS, columns=METHODS).to_string())

    # ---------- LOCK 2: critical-cost sweep (fixed named vocab) ----------
    prim_named = make_prim_of_id(named_vocab, n_classes)
    cc_rows = []
    for C_crit in [2.0, 5.0, 10.0, 20.0, 50.0, 1e6]:
        for k in (1, 3):
            for method in METHODS:
                r = simulate(packed[method][k], prim_named, tau=0.0, C_crit=C_crit,
                             costs=costs, rng=np.random.default_rng(args.seed),
                             n_missions=args.frontier_missions)
                cc_rows.append(dict(method=method, k=k, C_crit=C_crit,
                                    mean_cost=r["mean_cost"], hard_success=r["hard_success"]))
    ccdf = pd.DataFrame(cc_rows)
    ccdf.to_csv(os.path.join(OUT, "costmodel_sweep.csv"), index=False)

    # ---------- LOCK 3: iso-safety operating point (fixed named vocab) ----------
    vocab_ids = np.array(named_vocab)
    fr_rows = []
    for k in (1, 3):
        for tau in TAUS:
            for method in METHODS:
                r = simulate(packed[method][k], prim_named, tau=float(tau), C_crit=20.0,
                             costs=costs, rng=np.random.default_rng(args.seed),
                             n_missions=args.frontier_missions)
                fa = false_activation(packed[method][k], prim_named, float(tau),
                                      vocab_ids, np.random.default_rng(args.seed))
                fr_rows.append(dict(method=method, k=k, tau=float(tau),
                                    false_activation=fa, **r))
    frdf = pd.DataFrame(fr_rows)
    frdf.to_csv(os.path.join(OUT, "frontier.csv"), index=False)

    iso_rows = []
    for budget in (0.01, 0.005):
        for k in (1, 3):
            for method in METHODS:
                s = frdf[(frdf.method == method) & (frdf.k == k)].sort_values("tau")
                ok = s[s.false_activation <= budget]
                if len(ok):
                    row = ok.iloc[0]                     # smallest tau meeting budget
                    iso_rows.append(dict(method=method, k=k, budget=budget,
                                         tau_star=round(float(row.tau), 3),
                                         task_success=round(float(row.hard_success), 4),
                                         mean_cost=round(float(row.mean_cost), 3),
                                         false_activation=round(float(row.false_activation), 5)))
                else:
                    iso_rows.append(dict(method=method, k=k, budget=budget,
                                         tau_star=None, task_success=None,
                                         mean_cost=None, false_activation=None))
    isodf = pd.DataFrame(iso_rows)
    isodf.to_csv(os.path.join(OUT, "iso_safety.csv"), index=False)
    print("\nLOCK3 iso-safety (smallest tau with false-activation <= budget):")
    print(isodf.to_string(index=False))

    # ---------- figures ----------
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"scratch": "tab:gray", "mae": "tab:green",
              "supMAE": "tab:orange", "supLP120": "tab:blue", "supcon": "tab:purple"}

    # LOCK1 distribution boxplot of hard_success per method (tau=0, k=1) across vocabs
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, tau in zip(axes, (0.0, 0.9)):
        data = [vdf[(vdf.k == 1) & (vdf.tau == tau) & (vdf.method == m)].hard_success.values
                for m in METHODS]
        bp = ax.boxplot(data, tick_labels=METHODS, showmeans=True)
        ax.set_title(f"k=1, tau={tau}: task-success across {args.vocabs} random vocabularies")
        ax.set_ylabel("hard task-success"); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "vocab_distribution.png"), dpi=110)

    # LOCK2 cost vs C_crit
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, k in zip(axes, (1, 3)):
        for m in METHODS:
            s = ccdf[(ccdf.method == m) & (ccdf.k == k)].sort_values("C_crit")
            ax.plot(s.C_crit, s.mean_cost, "-o", color=colors[m], label=m, ms=4)
        ax.set_xscale("log"); ax.set_title(f"k={k}: mean mission cost vs critical penalty")
        ax.set_xlabel("C_crit (log)"); ax.set_ylabel("mean cost"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "costmodel_sweep.png"), dpi=110)

    # LOCK3 frontier (task-success vs false-activation)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, k in zip(axes, (1, 3)):
        for m in METHODS:
            s = frdf[(frdf.method == m) & (frdf.k == k)].sort_values("false_activation")
            ax.plot(s.false_activation, s.hard_success, "-o", color=colors[m], label=m, ms=4)
        for b in (0.01, 0.005):
            ax.axvline(b, ls="--", c="k", alpha=.3)
        ax.set_title(f"k={k}: safety/throughput frontier (tau sweep)")
        ax.set_xlabel("false-activation rate"); ax.set_ylabel("task-success")
        ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "frontier.png"), dpi=110)
    print(f"\nWrote -> {OUT}")


if __name__ == "__main__":
    main()
