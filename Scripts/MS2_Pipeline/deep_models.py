"""
deep_models.py  --  neural-network (MLP) path for the MS2 pipeline
==================================================================

The "deep learning" path: instead of hand-crafted summary features, a neural
network (multi-layer perceptron) is trained on the RAW windowed waveform
([P, Q, THD_I] samples inside each window). It learns from the signal shape,
which the feature-based RF/LightGBM models can't see.

It uses scikit-learn's MLP, so it needs NO extra dependency and runs in your
existing environment (a true PyTorch CNN/LSTM is a heavier future upgrade that
needs a torch env). Supported tasks: disaggregate (regression) and presence
(multi-label) - the scenario tasks where waveform shape helps most.

Called from train.py (--model mlp) and infer.py (when the model bundle is a DL
model). Models are saved as model_<task>_mlp.joblib so they don't overwrite the
RF/LGBM models.
"""
from __future__ import annotations
import json, os, sys

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_absolute_error, f1_score

import nilm_pipeline as nl


def _colors(names=None):
    cmap = plt.get_cmap("tab10")
    return {n: cmap(i % 10) for i, n in enumerate(names if names is not None else nl.CANON)}


def _collect(files, window_s, on_W=15.0):
    """Stack raw-sequence windows + targets across scenario files. The appliance
    vocabulary is derived from the data (falls back to CANON if scan finds none)."""
    names = nl.scan_canon(files) or nl.CANON
    Xs, YPOW, YPRES, gs, k = [], [], [], [], 0
    for f in files:
        s = nl.load_signal(f)
        if s.gt_P is None:
            del s; continue
        Xf, Ypow, Ypres, _ = nl.aggregate_sequences(s, window_s, on_W=on_W, canon=names)
        Xs.append(Xf); YPOW.append(Ypow); YPRES.append(Ypres)
        gs.append(np.full(len(Xf), k)); k += 1
        del s
    if not Xs:
        sys.exit("the MLP path needs scenario .h5 files with /ground_truth")
    return (np.vstack(Xs), np.vstack(YPOW), np.vstack(YPRES),
            np.concatenate(gs), k, names)


def _split(n, g, k, test_size=0.3):
    if k >= 2:
        return next(GroupShuffleSplit(1, test_size=test_size, random_state=0)
                    .split(np.zeros(n), np.zeros(n), g))
    return train_test_split(np.arange(n), test_size=test_size, random_state=0)


def train(files, args, out):
    on_w = getattr(args, "on_w", 15.0)
    X, Ypow, Ypres, g, k, names = _collect(files, args.window, on_W=on_w)
    tr, te = _split(len(X), g, k)
    print(f"MLP {args.task}: {X.shape[0]} windows x {X.shape[1]} raw inputs "
          f"from {k} scenarios")

    if args.task == "disaggregate":
        net = Pipeline([("sc", StandardScaler()),
                        ("net", MLPRegressor(hidden_layer_sizes=(256, 128),
                                             max_iter=120, early_stopping=True,
                                             random_state=0))]).fit(X[tr], Ypow[tr])
        P = net.predict(X[te])
        per = {names[i]: float(mean_absolute_error(Ypow[te][:, i], P[:, i]))
               for i in range(len(names))}
        metrics = {"task": "disaggregate", "model": "mlp", "dl": True,
                   "appliances": names, "window_s": args.window,
                   "overall_mae_W": float(np.mean(list(per.values()))),
                   "per_appliance_mae_W": per}
        bundle = {"task": "disaggregate", "dl": True, "model": net,
                  "appliances": names, "window_s": args.window}
        fn, mn = "model_disaggregate_mlp.joblib", "train_disaggregate_mlp_metrics.json"
        print(f"  [MLP] held-out overall MAE = {metrics['overall_mae_W']:.1f} W")
        print("  per-appliance MAE(W): " + ", ".join(f"{a}={v:.0f}" for a, v in per.items()))
    else:  # presence
        net = Pipeline([("sc", StandardScaler()),
                        ("net", MLPClassifier(hidden_layer_sizes=(256, 128),
                                              max_iter=120, early_stopping=True,
                                              random_state=0))]).fit(X[tr], Ypres[tr])
        Pp = np.asarray(net.predict(X[te]))
        per = {names[i]: float(f1_score(Ypres[te][:, i], Pp[:, i], zero_division=0))
               for i in range(len(names))}
        metrics = {"task": "presence", "model": "mlp", "dl": True,
                   "appliances": names, "window_s": args.window, "on_W": on_w,
                   "macro_f1": float(np.mean(list(per.values()))), "per_appliance_f1": per}
        bundle = {"task": "presence", "dl": True, "model": net,
                  "appliances": names, "window_s": args.window, "on_W": on_w}
        fn, mn = "model_presence_mlp.joblib", "train_presence_mlp_metrics.json"
        print(f"  [MLP] held-out presence macro-F1 = {metrics['macro_f1']:.3f}")
        print("  per-appliance F1: " + ", ".join(f"{a}={v:.2f}" for a, v in per.items()))

    bundle["metrics"] = metrics
    joblib.dump(bundle, os.path.join(out, fn))
    json.dump(metrics, open(os.path.join(out, mn), "w"), indent=2)
    print(f"  saved -> {out}/{fn}")


def infer(sig, bundle, args, out):
    model, names = bundle["model"], bundle["appliances"]
    ws = bundle.get("window_s", 30.0); on_W = bundle.get("on_W", 15.0)
    Xf, Ypow, Ypres, _ = nl.aggregate_sequences(sig, ws, on_W=on_W, canon=names)
    hours = np.arange(Xf.shape[0]) * ws / 3600.0
    colors = _colors(names)

    if bundle["task"] == "disaggregate":
        P = model.predict(Xf)
        pd.DataFrame(P, columns=names).assign(hour=hours).to_csv(
            os.path.join(out, "disaggregation.csv"), index=False)
        summary = {"input": sig.name, "task": "disaggregate (mlp)",
                   "energy_kWh_per_appliance":
                   {names[i]: float(P[:, i].sum() * ws / 3600.0 / 1000.0) for i in range(len(names))}}
        if Ypow is not None:
            summary["per_appliance_mae_W"] = {
                names[i]: float(mean_absolute_error(Ypow[:, i], P[:, i])) for i in range(len(names))}
            summary["overall_mae_W"] = float(np.mean(list(summary["per_appliance_mae_W"].values())))
            print(f"  [MLP] overall MAE vs ground truth: {summary['overall_mae_W']:.1f} W")
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
        top = sorted(range(len(names)), key=lambda i: -abs(P[:, i]).sum())[:4]
        fig, ax = plt.subplots(len(top), 1, figsize=(11, 7), sharex=True)
        for k, j in enumerate(top):
            ax[k].plot(hours, P[:, j], color="#e07b39", lw=1.0, label="predicted")
            if Ypow is not None:
                ax[k].plot(hours, Ypow[:, j], color="#1f4e79", lw=1.0, ls="--", label="true")
            ax[k].set_ylabel(f"{names[j]}\nP (W)"); ax[k].legend(fontsize=8, loc="upper right")
        ax[-1].set_xlabel("hour"); ax[0].set_title(f"Disaggregation (MLP) - {sig.name}")
        plt.tight_layout(); plt.savefig(os.path.join(out, "disaggregation.png"), dpi=110); plt.close()
        print("  wrote disaggregation.csv, summary.json, disaggregation.png")
    else:  # presence
        Pp = np.asarray(model.predict(Xf)).astype(int)
        pd.DataFrame(Pp, columns=names).assign(hour=hours).to_csv(
            os.path.join(out, "presence.csv"), index=False)
        summary = {"input": sig.name, "task": "presence (mlp)",
                   "fraction_on": {names[i]: float(Pp[:, i].mean()) for i in range(len(names))}}
        if Ypres is not None:
            summary["per_appliance_f1"] = {
                names[i]: float(f1_score(Ypres[:, i], Pp[:, i], zero_division=0)) for i in range(len(names))}
            summary["macro_f1"] = float(np.mean(list(summary["per_appliance_f1"].values())))
            print(f"  [MLP] presence macro-F1 vs ground truth: {summary['macro_f1']:.3f}")
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
        nrows = 2 if Ypres is not None else 1
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 0.45 * len(names) * nrows + 1.5),
                                 squeeze=False, sharex=True)

        def gantt(ax, M, title):
            for i, nm in enumerate(names):
                ax.fill_between(hours, i + 0.05, i + 0.95, where=M[:, i].astype(bool),
                                color=colors.get(nm, "0.5"), step="mid", linewidth=0)
            ax.set_yticks([i + 0.5 for i in range(len(names))]); ax.set_yticklabels(names, fontsize=8)
            ax.set_ylim(0, len(names)); ax.set_title(title)

        gantt(axes[0, 0], Pp, f"Predicted presence (MLP) - {sig.name}")
        if Ypres is not None:
            gantt(axes[1, 0], Ypres, "Ground-truth presence")
        axes[-1, 0].set_xlabel("hour")
        plt.tight_layout(); plt.savefig(os.path.join(out, "presence_timeline.png"), dpi=110); plt.close()
        print("  wrote presence.csv, summary.json, presence_timeline.png")
