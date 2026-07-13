#!/usr/bin/env python3
"""Phase 3 / C6 — abstract event-driven controller over the REAL recognizer posteriors.

Near-zero new ML: consumes the per-clip softmax stream dumped in
trained_models/LOSO-fullTrainCalibrate-v2{,-seed43,-seed44}/posteriors/*.csv
(these are held-out-subject LOSO clips). Maps a subset of the 22 gestures to
control primitives, runs a Monte-Carlo pick-place mission through an FSM with a
safety layer (confidence-threshold reject + asymmetric error cost), and reports
task-level metrics per (init x k) and across a confidence-threshold sweep.

Outputs -> trained_models/Phase3-controller/
  controller_results.csv         (method,k,tau, task/safety/throughput metrics)
  operating_point_summary.csv     (metrics at a single fixed tau, per method x k)
  mapping.csv                     (gesture -> primitive assignment + its recall)
  success_vs_tau.png, safety_throughput.png, success_by_method.png
"""
import os, ast, glob, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNDIRS = [
    "trained_models/LOSO-fullTrainCalibrate-v2",
    "trained_models/LOSO-fullTrainCalibrate-v2-seed43",
    "trained_models/LOSO-fullTrainCalibrate-v2-seed44",
]
OUT = os.path.join(ROOT, "trained_models/Phase3-controller")
METHODS = ["scratch", "mae", "supMAE", "supLP120"]

# ---- cost model (time units; asymmetric so errors compound) ----
T_EXEC     = 1.0   # one gesture attempt
T_REJECT   = 1.0   # low-confidence -> re-prompt, user repeats
T_CORRECT  = 3.0   # non-critical wrong command: cancel + reissue
MAX_REJECT = 5     # consecutive rejects on a step before mission abort
# safety-critical primitives: a wrong command *of or into* these = mission failure
CRITICAL = {"grasp", "release"}

# ---- mission: ordered pick-place sequence (2 grasps, 1 release => compounding) ----
MISSION = ["next", "next", "previous", "approach", "grasp", "confirm",
           "next", "approach", "grasp", "release", "confirm", "cancel"]


def load_posteriors():
    frames = []
    for rd in RUNDIRS:
        for f in glob.glob(os.path.join(ROOT, rd, "posteriors", "*.csv")):
            frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    # keep what we need; conf is the max softmax prob (already temp-free from dump)
    return df[["seed", "subject", "method", "k", "true_label", "pred_id",
               "true_id", "conf"]].copy()


def build_mapping(df):
    """Assign 7 command primitives to gestures, method-agnostically.
    Rank gestures by mean recall pooled over ALL methods at k=3 (tuning the
    map on aggregate, not per-method, so it favors no init). Exclude locomotion
    /ambiguous classes from command duty; give safety-critical primitives the
    most reliable gestures (design guard, paper_idea sec 8)."""
    k3 = df[df.k == 3]
    rec = k3.groupby("true_label").apply(
        lambda g: (g.pred_id == g.true_id).mean(), include_groups=False)
    rec = rec.sort_values(ascending=False)
    # locomotion / whole-body transitions make poor discrete commands
    EXCLUDE = {"walk", "runonspot", "turnaround", "buttkicks", "hop", "jump", "stand"}
    ranked = [g for g in rec.index if g not in EXCLUDE]
    # most reliable -> most safety-critical primitive
    order = ["grasp", "release", "confirm", "approach", "cancel", "next", "previous"]
    mapping = {}
    for prim, gest in zip(order, ranked):
        mapping[gest] = prim
    rows = [{"gesture": g, "primitive": p, "recall_k3": round(float(rec[g]), 4)}
            for g, p in mapping.items()]
    return mapping, pd.DataFrame(rows)


def run(df, mapping, tau, rng, n_trials=3000, dwell=1):
    """Monte-Carlo the mission for one (method,k) slice already filtered in df.
    df: posteriors for one method,k. mapping: gesture->primitive.
    Returns dict of task/safety/throughput metrics."""
    prim_of_class = mapping                       # gesture-name -> primitive
    gest_of_prim = {p: g for g, p in mapping.items()}
    cmd_gestures = set(mapping)                    # gestures that are commands
    # index rows by true gesture for fast sampling
    by_class = {g: df[df.true_label == g] for g in df.true_label.unique()}
    # a "predicted primitive" for a clip: if pred class is a command gesture, that
    # primitive, else None (pred is a non-command -> no activation)
    id2label = {v: k for k, v in LABELMAP.items()}

    def sample_pred(gesture):
        pool = by_class[gesture]
        r = pool.iloc[rng.integers(len(pool))]
        pred_lbl = id2label[int(r.pred_id)]
        pred_prim = prim_of_class.get(pred_lbl, None)
        return float(r.conf), pred_prim

    succ = 0; times = []; rejects = 0; corrections = 0; steps_total = 0; presentations = 0
    for _ in range(n_trials):
        t = 0.0; ok = True; ncorr = 0
        for intended in MISSION:
            target_gesture = gest_of_prim[intended]
            nrej = 0
            while True:
                # dwell: require `dwell` consecutive above-threshold agreeing reads
                conf, pred_prim = sample_pred(target_gesture); presentations += 1
                if conf < tau:                       # reject -> re-prompt
                    rejects += 1; nrej += 1; t += T_REJECT
                    if nrej > MAX_REJECT:
                        ok = False
                    if not ok: break
                    continue
                if dwell > 1:
                    agree = True
                    for _d in range(dwell - 1):
                        c2, p2 = sample_pred(target_gesture); presentations += 1
                        if c2 < tau or p2 != pred_prim:
                            agree = False; break
                    if not agree:
                        rejects += 1; nrej += 1; t += T_REJECT
                        if nrej > MAX_REJECT: ok = False
                        if not ok: break
                        continue
                t += T_EXEC
                if pred_prim == intended:            # correct command
                    break
                # wrong command issued
                if intended in CRITICAL or pred_prim in CRITICAL:
                    ok = False; break               # dangerous -> mission failure
                t += T_CORRECT; corrections += 1     # recoverable: cancel + reissue
                break
            steps_total += 1
            if not ok: break
        if ok:
            succ += 1; times.append(t)

    # false-activation: distractor stream = clips whose true class is NOT a command
    distract = df[~df.true_label.isin(cmd_gestures)]
    fa = 0; n_fa = min(len(distract), 8000)
    idx = rng.choice(len(distract), size=n_fa, replace=False)
    for i in idx:
        r = distract.iloc[int(i)]
        if float(r.conf) >= tau:
            if prim_of_class.get(id2label[int(r.pred_id)], None) is not None:
                fa += 1
    return dict(
        task_success=succ / n_trials,
        mean_time=float(np.mean(times)) if times else float("nan"),
        rejection_rate=rejects / presentations,
        corrective_per_mission=corrections / n_trials,
        false_activation_rate=fa / n_fa,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--dwell", type=int, default=1)
    ap.add_argument("--op-tau", type=float, default=0.9, help="fixed operating point")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    global LABELMAP
    import json
    LABELMAP = json.load(open(os.path.join(
        ROOT, "trained_models/LOSO-fullTrainCalibrate-v2/label_map.json")))

    df = load_posteriors()
    mapping, mapdf = build_mapping(df)
    mapdf.to_csv(os.path.join(OUT, "mapping.csv"), index=False)
    print("Gesture->primitive mapping (by pooled k=3 recall):")
    print(mapdf.to_string(index=False))

    TAUS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
    rows = []
    for method in METHODS:
        for k in [1, 3]:
            sl = df[(df.method == method) & (df.k == k)]
            for tau in TAUS:
                rng = np.random.default_rng(args.seed)
                m = run(sl, mapping, tau, rng, n_trials=args.trials, dwell=args.dwell)
                m.update(method=method, k=k, tau=tau)
                rows.append(m)
            print(f"  done {method} k={k}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "controller_results.csv"), index=False)

    op = res[np.isclose(res.tau, args.op_tau)].copy()
    op = op[["method", "k", "task_success", "mean_time", "rejection_rate",
             "corrective_per_mission", "false_activation_rate"]]
    op.to_csv(os.path.join(OUT, "operating_point_summary.csv"), index=False)
    print(f"\nOperating point tau={args.op_tau}:")
    print(op.to_string(index=False))

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"scratch": "tab:gray", "mae": "tab:green",
              "supMAE": "tab:orange", "supLP120": "tab:blue"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, k in zip(axes, [1, 3]):
        for method in METHODS:
            s = res[(res.method == method) & (res.k == k)].sort_values("tau")
            ax.plot(s.tau, s.task_success, "-o", color=colors[method], label=method, ms=4)
        ax.set_title(f"k={k}: task success vs confidence threshold")
        ax.set_xlabel("confidence threshold tau"); ax.set_ylabel("task success rate")
        ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "success_vs_tau.png"), dpi=110)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, k in zip(axes, [1, 3]):
        for method in METHODS:
            s = res[(res.method == method) & (res.k == k)].sort_values("false_activation_rate")
            ax.plot(s.false_activation_rate, s.task_success, "-o",
                    color=colors[method], label=method, ms=4)
        ax.set_title(f"k={k}: safety/throughput tradeoff (tau sweep)")
        ax.set_xlabel("false-activation rate (lower=safer)")
        ax.set_ylabel("task success rate"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "safety_throughput.png"), dpi=110)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(METHODS)); w = 0.35
    for i, k in enumerate([1, 3]):
        vals = [op[(op.method == m) & (op.k == k)].task_success.values[0] for m in METHODS]
        ax.bar(x + i * w, vals, w, label=f"k={k}")
    ax.set_xticks(x + w / 2); ax.set_xticklabels(METHODS)
    ax.set_ylabel("task success"); ax.set_title(f"task success @ tau={args.op_tau}")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "success_by_method.png"), dpi=110)
    print(f"\nWrote -> {OUT}")


if __name__ == "__main__":
    main()
