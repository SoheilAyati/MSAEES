#!/usr/bin/env python3
"""
train.py  --  MS2 TRAINING pipeline
===================================

Input : labelled signal files (.h5 / .csv).   Output: a trained model + metrics.

    python train.py --data <files|globs|dirs> [--task identify|disaggregate]
                    [--model rf|lgbm] [--features auto|common|full] [--out DIR]

Two tasks:
  identify     (default)  learn  window-of-signal -> appliance name.
                          Needs single-appliance / labelled files (each file's
                          label = its appliance/device). Saves a classifier.
  disaggregate            learn  aggregate-window -> per-appliance power.
                          Needs scenario .h5 files that contain /ground_truth.
                          Saves a multi-output regressor.

--features (identify only):
  auto    full feature set if every file has harmonics, else the common subset
  common  P,Q,S,PF,THD_I,... only -> the model also runs on real PAC4200 CSVs
  full    adds per-order harmonic features (synthetic h5 only)

Outputs (in --out, default ./output):
  model_<task>.joblib          trained model + feature/label lists
  train_<task>_metrics.json    held-out scores
  train_identify_confusion.png (identify only)

Files are streamed one at a time, so memory stays small even for large corpora.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, mean_absolute_error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nilm_pipeline as nl


def expand_inputs(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, "**", "*.h5"), recursive=True)
            files += glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)
        elif any(c in p for c in "*?["):
            files += glob.glob(p, recursive=True)
        else:
            files.append(p)
    return sorted(set(files))


def _clf(kind):
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        c = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=0, verbose=-1)
    else:
        c = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=0, n_jobs=-1)
    return Pipeline([("imp", SimpleImputer(strategy="mean")),
                     ("sc", StandardScaler()), ("clf", c)])


def train_identify(files, args, out):
    rows, all_harm = [], True
    for f in files:                                   # stream one file at a time
        s = nl.load_signal(f)
        if not s.label:
            continue
        all_harm = all_harm and s.has_harmonics
        wf = nl.window_features(s, args.window, args.stride, args.on_threshold)
        wf = wf[wf.active].copy()
        wf["label"] = s.label
        wf["group"] = os.path.basename(os.path.dirname(f)) or nl._stem(f)  # instance = seed/file
        rows.append(wf)
        del s
    if not rows:
        sys.exit("identify needs labelled single-appliance/device files")
    df = pd.concat(rows, ignore_index=True)
    feats = {"common": nl.FEATURES_COMMON, "full": nl.FEATURES_FULL}.get(args.features) \
        or (nl.FEATURES_FULL if all_harm else nl.FEATURES_COMMON)
    print(f"identification: {len(df)} active windows, {df.label.nunique()} classes, "
          f"{len(df.group.unique())} files | features="
          f"{'common (real-meter compatible)' if feats is nl.FEATURES_COMMON else 'full+harmonics'}")

    X, y, g = df[feats].to_numpy(float), df.label.to_numpy(), df.group.to_numpy()
    if df.groupby('label')['group'].nunique().min() >= 2:
        tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(X, y, g))
        split = "grouped: tested on held-out instances"
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0, stratify=y)
        split = "stratified rows (one instance per class)"
    model = _clf(args.model).fit(X[tr], y[tr])
    yp = model.predict(X[te]); labels = sorted(np.unique(y))
    metrics = {"task": "identify", "model": args.model, "features": feats, "split": split,
               "n_windows": int(len(df)), "classes": labels,
               "holdout_macro_f1": float(f1_score(y[te], yp, average="macro")),
               "holdout_accuracy": float(accuracy_score(y[te], yp))}
    print(f"  {split}\n  HELD-OUT macro-F1={metrics['holdout_macro_f1']:.3f} "
          f"acc={metrics['holdout_accuracy']:.3f}")
    joblib.dump({"task": "identify", "model": model, "features": feats, "labels": labels},
                os.path.join(out, "model_identify.joblib"))
    _confusion(y[te], yp, labels, os.path.join(out, "train_identify_confusion.png"),
               f"Identification held-out (macro-F1={metrics['holdout_macro_f1']:.2f})")
    json.dump(metrics, open(os.path.join(out, "train_identify_metrics.json"), "w"), indent=2)
    print(f"  saved -> {out}/model_identify.joblib")


def train_disaggregate(files, args, out):
    Xs, Ys, gs, k = [], [], [], 0
    for f in files:                                   # stream one scenario at a time
        s = nl.load_signal(f)
        if s.gt_P is None:
            del s; continue
        X, Y, names = nl.aggregate_windows(s, args.window)
        Xs.append(X); Ys.append(Y); gs.append(np.full(len(X), k)); k += 1
        del s
    if not Xs:
        sys.exit("disaggregate needs scenario .h5 files with /ground_truth")
    X, Y, g = np.vstack(Xs), np.vstack(Ys), np.concatenate(gs)
    print(f"disaggregation: {X.shape[0]} windows from {k} scenarios")
    if k >= 2:
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, Y, g))
    else:
        tr, te = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0)
    base = (_lgbm_reg() if args.model == "lgbm"
            else RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))
    model = MultiOutputRegressor(base).fit(X[tr], Y[tr])
    P = model.predict(X[te])
    per = {names[i]: float(mean_absolute_error(Y[te][:, i], P[:, i])) for i in range(len(names))}
    metrics = {"task": "disaggregate", "model": args.model, "features": nl.AGG_FEATURES,
               "appliances": names, "overall_mae_W": float(np.mean(list(per.values()))),
               "per_appliance_mae_W": per}
    print(f"  held-out overall MAE={metrics['overall_mae_W']:.1f} W")
    print("  per-appliance MAE(W): " + ", ".join(f"{a}={v:.0f}" for a, v in per.items()))
    joblib.dump({"task": "disaggregate", "model": model, "features": nl.AGG_FEATURES,
                 "appliances": names, "window_s": args.window},
                os.path.join(out, "model_disaggregate.joblib"))
    json.dump(metrics, open(os.path.join(out, "train_disaggregate_metrics.json"), "w"), indent=2)
    print(f"  saved -> {out}/model_disaggregate.joblib")


def _lgbm_reg():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=0, verbose=-1)


def _base_clf(kind):
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=0, verbose=-1)
    return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1,
                                  class_weight="balanced")


def train_presence(files, args, out):
    """Multi-label: for each aggregate window predict which appliances are ON."""
    Xs, Ys, gs, k = [], [], [], 0
    for f in files:                                   # stream one scenario at a time
        s = nl.load_signal(f)
        if s.gt_P is None:
            del s; continue
        X, Y, names = nl.aggregate_presence(s, args.window, on_W=15.0)
        Xs.append(X); Ys.append(Y); gs.append(np.full(len(X), k)); k += 1
        del s
    if not Xs:
        sys.exit("presence needs scenario .h5 files with /ground_truth")
    X, Y, g = np.vstack(Xs), np.vstack(Ys), np.concatenate(gs)
    print(f"presence: {X.shape[0]} windows from {k} scenarios, {Y.shape[1]} appliances")
    if k >= 2:
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, Y, g))
    else:
        tr, te = train_test_split(np.arange(len(X)), test_size=0.3, random_state=0)
    # presence uses RandomForest: robust to always-on appliances (single-class
    # columns like baseload) that would break a per-output LightGBM.
    model = MultiOutputClassifier(_base_clf("rf")).fit(X[tr], Y[tr])
    Pp = np.asarray(model.predict(X[te]))
    per = {names[i]: float(f1_score(Y[te][:, i], Pp[:, i], zero_division=0)) for i in range(len(names))}
    metrics = {"task": "presence", "model": "rf", "features": nl.AGG_FEATURES,
               "appliances": names, "on_W": 15.0,
               "macro_f1": float(np.mean(list(per.values()))), "per_appliance_f1": per}
    print(f"  held-out presence macro-F1 = {metrics['macro_f1']:.3f}")
    print("  per-appliance F1: " + ", ".join(f"{a}={v:.2f}" for a, v in per.items()))
    joblib.dump({"task": "presence", "model": model, "features": nl.AGG_FEATURES,
                 "appliances": names, "window_s": args.window, "on_W": 15.0},
                os.path.join(out, "model_presence.joblib"))
    json.dump(metrics, open(os.path.join(out, "train_presence_metrics.json"), "w"), indent=2)
    print(f"  saved -> {out}/model_presence.joblib")


def _confusion(yt, yp, labels, path, title):
    C = confusion_matrix(yt, yp, labels=labels).astype(float)
    Cn = C / C.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if Cn[i, j] > .01:
                ax.text(j, i, f"{Cn[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if Cn[i, j] > .5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main():
    ap = argparse.ArgumentParser(description="MS2 training pipeline")
    ap.add_argument("--data", nargs="+", required=True, help="files, globs, or dirs")
    ap.add_argument("--task", choices=["identify", "disaggregate", "presence"], default="identify")
    ap.add_argument("--model", choices=["rf", "lgbm", "mlp"], default="rf",
                    help="mlp = neural network on the raw waveform (disaggregate/presence)")
    ap.add_argument("--features", choices=["auto", "common", "full"], default="auto",
                    help="identify only; 'common' = real-PAC4200 compatible")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--stride", type=float, default=30.0)
    ap.add_argument("--on-threshold", type=float, default=5.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = expand_inputs(args.data)
    if not files:
        sys.exit("no input files found")
    print(f"task={args.task}  inputs={len(files)} file(s)  out={args.out}")
    if args.model == "mlp":
        if args.task not in ("disaggregate", "presence"):
            sys.exit("--model mlp supports --task disaggregate or presence")
        import deep_models
        deep_models.train(files, args, args.out)
    else:
        {"identify": train_identify, "disaggregate": train_disaggregate,
         "presence": train_presence}[args.task](files, args, args.out)


if __name__ == "__main__":
    main()
