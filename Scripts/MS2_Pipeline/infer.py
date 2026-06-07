#!/usr/bin/env python3
"""
infer.py  --  MS2 INFERENCE pipeline
====================================

Input : one signal file (.h5 or .csv) + a trained model.
Output: results (csv + json) and a plot.

    python infer.py --input <file> --model <model.joblib> [--out DIR]

The model file knows its own task:
  identify      -> predicts the appliance for each active window of the signal,
                   writes predictions.csv, summary.json (time per appliance,
                   most-likely device) and a labelled timeline plot.
  disaggregate  -> predicts each appliance's power for every window, writes
                   disaggregation.csv (power vs time), summary.json (energy per
                   appliance in kWh) and a per-appliance power plot. If the file
                   has ground truth, it also reports MAE.

Outputs go to <out>/<input-name>/ (default: ./output/<input-name>/).
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nilm_pipeline as nl


def _appliance_colors():
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(nl.CANON)}


def run_identify(sig, bundle, args, out):
    feats, model = bundle["features"], bundle["model"]
    wf = nl.window_features(sig, args.window, args.stride, args.on_threshold)
    wf["predicted"] = "off"; wf["pred2"] = ""; wf["confidence"] = np.nan; wf["conf2"] = np.nan
    act = wf[wf.active]
    classes = list(getattr(model, "classes_", []))
    if len(act):
        X = act[feats].to_numpy(float)
        wf.loc[act.index, "predicted"] = model.predict(X)
        if hasattr(model, "predict_proba") and len(classes) >= 2:
            proba = model.predict_proba(X)
            order = np.argsort(proba, axis=1)
            r = np.arange(len(proba))
            wf.loc[act.index, "confidence"] = proba[r, order[:, -1]]
            wf.loc[act.index, "pred2"] = [classes[i] for i in order[:, -2]]
            wf.loc[act.index, "conf2"] = proba[r, order[:, -2]]
    wf[["start_s", "end_s", "active", "predicted", "confidence", "pred2", "conf2"]].to_csv(
        os.path.join(out, "predictions.csv"), index=False)

    secs = (wf[wf.active].groupby("predicted").size() * args.stride).sort_values(ascending=False)
    has_gt = sig.gt_P is not None
    summary = {"input": sig.name, "task": "identify", "is_scenario": bool(has_gt),
               "active_seconds_per_appliance": {k: float(v) for k, v in secs.items()}}
    if not has_gt:                       # single-device input -> one answer is meaningful
        summary["most_likely_appliance"] = (secs.index[0] if len(secs) else None)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)

    # ---- build per-sample predicted class (primary + 2nd guess) ----
    colors = _appliance_colors()
    hrs = sig.t / 3600.0
    sr = sig.sample_rate_hz
    step = max(1, int(round(args.stride * sr)))
    prim = np.full(sig.n, None, dtype=object)
    sec = np.full(sig.n, None, dtype=object)
    for _, row in wf[wf.active].iterrows():    # fresh rows (now carry predictions)
        i0 = int(round(row.start_s * sr)); i1 = min(i0 + step, sig.n)
        prim[i0:i1] = row.predicted
        if row.pred2 and isinstance(row.conf2, float) and row.conf2 > 0.25:
            sec[i0:i1] = row.pred2

    from matplotlib.patches import Patch
    nrows = 2 if has_gt else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 4.6 * nrows), sharex=True, squeeze=False)
    ax = axes[0, 0]
    ax.plot(hrs, sig.P, color="0.35", lw=0.7, zorder=3)
    used = []
    for app in nl.CANON:
        m = (prim == app)
        if m.any():
            ax.fill_between(hrs, 0, sig.P, where=m, color=colors[app], alpha=0.55,
                            linewidth=0, zorder=2); used.append(app)
        sm = (sec == app)
        if sm.any():
            ax.fill_between(hrs, 0, sig.P, where=sm, color=colors[app], alpha=0.22,
                            hatch="///", linewidth=0, zorder=2)
    ax.axhline(0, color="0.7", lw=0.5)
    handles = [Patch(color=colors[a], label=a) for a in used]
    if any((sec == a).any() for a in nl.CANON):
        handles.append(Patch(facecolor="0.8", hatch="///", label="2nd guess"))
    if handles:
        ax.legend(handles=handles, fontsize=7, ncol=2, loc="upper left")
    ttl = "Identification (per-window prediction; area colored by appliance)"
    if not has_gt and summary.get("most_likely_appliance"):
        ttl = f"Identification — most likely: {summary['most_likely_appliance']}"
    ax.set_title(f"{ttl}\n{sig.name}"); ax.set_ylabel("P (W)")

    if has_gt:                            # ground-truth comparison panel
        ax2 = axes[1, 0]
        for j, nm in enumerate(sig.gt_names):
            base = nm.rsplit("_", 1)[0]
            p = sig.gt_P[:, j]
            if np.nanmax(np.abs(p)) > 5:
                ax2.plot(hrs, p, color=colors.get(base, "0.5"), lw=0.9, label=base)
        ax2.axhline(0, color="0.7", lw=0.5)
        ax2.set_title("Ground truth — actual per-appliance power")
        ax2.set_ylabel("P (W)"); ax2.set_xlabel("hour")
        ax2.legend(fontsize=7, ncol=2, loc="upper left")
    else:
        ax.set_xlabel("hour")

    plt.tight_layout(); plt.savefig(os.path.join(out, "identify_timeline.png"), dpi=110); plt.close()
    print(f"  result: {summary.get('most_likely_appliance', '(scenario: see colors + ground-truth panel)')}")
    print("  wrote predictions.csv, summary.json, identify_timeline.png")


def run_disaggregate(sig, bundle, args, out):
    model, names = bundle["model"], bundle["appliances"]
    ws = bundle.get("window_s", 30.0)
    X, Y, _ = nl.aggregate_windows(sig, ws)
    P = model.predict(X)
    hours = np.arange(X.shape[0]) * ws / 3600.0
    df = pd.DataFrame(P, columns=names); df.insert(0, "hour", hours)
    df.to_csv(os.path.join(out, "disaggregation.csv"), index=False)

    energy = {names[i]: float(P[:, i].sum() * ws / 3600.0 / 1000.0) for i in range(len(names))}
    summary = {"input": sig.name, "task": "disaggregate",
               "energy_kWh_per_appliance": energy}
    if Y is not None:
        from sklearn.metrics import mean_absolute_error
        summary["per_appliance_mae_W"] = {
            names[i]: float(mean_absolute_error(Y[:, i], P[:, i])) for i in range(len(names))}
        summary["overall_mae_W"] = float(np.mean(list(summary["per_appliance_mae_W"].values())))
        print(f"  overall MAE (ground truth present): {summary['overall_mae_W']:.1f} W")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)

    top = sorted(range(len(names)), key=lambda i: -abs(P[:, i]).sum())[:4]
    fig, ax = plt.subplots(len(top), 1, figsize=(11, 7), sharex=True)
    for k, j in enumerate(top):
        ax[k].plot(hours, P[:, j], color="#e07b39", lw=1.0, label="predicted")
        if Y is not None:
            ax[k].plot(hours, Y[:, j], color="#1f4e79", lw=1.0, ls="--", label="true")
        ax[k].set_ylabel(f"{names[j]}\nP (W)"); ax[k].legend(fontsize=8, loc="upper right")
    ax[-1].set_xlabel("hour")
    ax[0].set_title(f"Disaggregation — {sig.name}")
    plt.tight_layout(); plt.savefig(os.path.join(out, "disaggregation.png"), dpi=110); plt.close()
    print(f"  wrote disaggregation.csv, summary.json, disaggregation.png")


def run_presence(sig, bundle, args, out):
    model, names = bundle["model"], bundle["appliances"]
    ws = bundle.get("window_s", 30.0); on_W = bundle.get("on_W", 15.0)
    X, Ytrue, _ = nl.aggregate_presence(sig, ws, on_W=on_W)
    Pp = np.asarray(model.predict(X)).astype(int)
    hours = np.arange(X.shape[0]) * ws / 3600.0
    df = pd.DataFrame(Pp, columns=names); df.insert(0, "hour", hours)
    df.to_csv(os.path.join(out, "presence.csv"), index=False)

    frac = {names[i]: float(Pp[:, i].mean()) for i in range(len(names))}
    summary = {"input": sig.name, "task": "presence", "fraction_on": frac}
    if Ytrue is not None:
        from sklearn.metrics import f1_score
        summary["per_appliance_f1"] = {names[i]: float(f1_score(Ytrue[:, i], Pp[:, i], zero_division=0))
                                       for i in range(len(names))}
        summary["macro_f1"] = float(np.mean(list(summary["per_appliance_f1"].values())))
        print(f"  presence macro-F1 vs ground truth: {summary['macro_f1']:.3f}")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)

    colors = _appliance_colors()
    nrows = 2 if Ytrue is not None else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 0.45 * len(names) * nrows + 1.5),
                             squeeze=False, sharex=True)

    def gantt(ax, M, title):
        for i, nm in enumerate(names):
            ax.fill_between(hours, i + 0.05, i + 0.95, where=M[:, i].astype(bool),
                            color=colors.get(nm, "0.5"), step="mid", linewidth=0)
        ax.set_yticks([i + 0.5 for i in range(len(names))]); ax.set_yticklabels(names, fontsize=8)
        ax.set_ylim(0, len(names)); ax.set_title(title)

    gantt(axes[0, 0], Pp, f"Predicted appliance presence — {sig.name}")
    if Ytrue is not None:
        gantt(axes[1, 0], Ytrue, "Ground-truth presence")
    axes[-1, 0].set_xlabel("hour")
    plt.tight_layout(); plt.savefig(os.path.join(out, "presence_timeline.png"), dpi=110); plt.close()
    print("  wrote presence.csv, summary.json, presence_timeline.png")


def main():
    ap = argparse.ArgumentParser(description="MS2 inference pipeline")
    ap.add_argument("--input", required=True, help="one .csv or .h5 signal file")
    ap.add_argument("--model", required=True, help="trained model .joblib from train.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--stride", type=float, default=30.0)
    ap.add_argument("--on-threshold", type=float, default=5.0)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    sig = nl.load_signal(args.input)
    _ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(HERE, "output", f"infer_{nl._stem(args.input)}_{_ts}")
    os.makedirs(out, exist_ok=True)
    print(f"input={sig.name} ({sig.source})  task={bundle['task']}  out={out}")
    if bundle.get("dl"):
        import deep_models
        deep_models.infer(sig, bundle, args, out)
    elif bundle["task"] == "identify":
        run_identify(sig, bundle, args, out)
    elif bundle["task"] == "presence":
        run_presence(sig, bundle, args, out)
    else:
        run_disaggregate(sig, bundle, args, out)


if __name__ == "__main__":
    main()
