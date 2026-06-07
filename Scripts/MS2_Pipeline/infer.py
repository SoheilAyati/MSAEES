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


def run_identify(sig, bundle, args, out):
    feats, model = bundle["features"], bundle["model"]
    wf = nl.window_features(sig, args.window, args.stride, args.on_threshold)
    wf["predicted"] = "off"; wf["confidence"] = np.nan
    act = wf[wf.active]
    if len(act):
        X = act[feats].to_numpy(float)
        pred = model.predict(X)
        wf.loc[act.index, "predicted"] = pred
        if hasattr(model, "predict_proba"):
            wf.loc[act.index, "confidence"] = model.predict_proba(X).max(1)
    wf[["start_s", "end_s", "active", "predicted", "confidence"]].to_csv(
        os.path.join(out, "predictions.csv"), index=False)

    secs = (wf[wf.active].groupby("predicted").size() * args.stride).sort_values(ascending=False)
    summary = {"input": sig.name, "task": "identify",
               "most_likely_appliance": (secs.index[0] if len(secs) else None),
               "active_seconds_per_appliance": {k: float(v) for k, v in secs.items()}}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)

    # timeline plot: P with active window centres coloured by predicted appliance
    hrs = sig.t / 3600.0
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hrs, sig.P, color="0.6", lw=0.5, zorder=1)
    cmap = plt.get_cmap("tab10")
    cats = list(secs.index)
    for k, app in enumerate(cats):
        m = (wf.predicted == app) & wf.active
        mid = (wf.start_s[m] + wf.end_s[m]) / 2 / 3600.0
        yy = np.interp(mid, hrs, sig.P)
        ax.scatter(mid, yy, s=22, color=cmap(k % 10), label=app, zorder=3)
    ax.set_xlabel("hour"); ax.set_ylabel("P (W)")
    ax.set_title(f"Identification — {sig.name}  (most likely: {summary['most_likely_appliance']})")
    if cats:
        ax.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout(); plt.savefig(os.path.join(out, "identify_timeline.png"), dpi=110); plt.close()
    print(f"  most-likely appliance: {summary['most_likely_appliance']}")
    print(f"  wrote predictions.csv, summary.json, identify_timeline.png")


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
    out = args.out or os.path.join(HERE, "output", nl._stem(args.input))
    os.makedirs(out, exist_ok=True)
    print(f"input={sig.name} ({sig.source})  task={bundle['task']}  out={out}")
    if bundle["task"] == "identify":
        run_identify(sig, bundle, args, out)
    else:
        run_disaggregate(sig, bundle, args, out)


if __name__ == "__main__":
    main()
