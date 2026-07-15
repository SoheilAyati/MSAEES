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
import argparse, datetime, glob, json, os, sys

import h5py
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


def _now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


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
        # one unreadable recording must not take the whole corpus with it: an
        # aborted capture leaves a truncated .h5 stub, and it would otherwise
        # abort EVERY training run from then on
        try:
            s = nl.load_signal(f)
        except (OSError, KeyError) as e:
            print(f"  SKIP (unreadable, ignoring): {os.path.basename(f)}  [{e}]")
            continue
        if not s.label:
            continue
        if nl.is_mixed_label(s.label):                # 'a__b' = several devices at once
            print(f"  skip (mixed recording, not single-device): {os.path.basename(f)}")
            del s; continue
        all_harm = all_harm and s.has_harmonics
        wf = nl.window_features(s, args.window, args.stride, args.on_threshold)
        if wf.empty or "active" not in wf.columns:        # recording shorter than one window
            print(f"  skip (no full window): {os.path.basename(f)}")
            del s; continue
        wf = wf[wf.active].copy()
        # class label = the appliance FAMILY, so 'standing_fan_high_no_rotation'
        # and 'standing_fan_low_rotation' train the same class 'standing_fan'
        wf["label"] = s.label if args.raw_labels else nl.parse_family(s.label)
        wf["group"] = os.path.basename(os.path.dirname(f)) or nl._stem(f)  # instance = seed/file
        wf["stem"] = nl._stem(f)
        rows.append(wf)
        del s
    if not rows:
        sys.exit("identify needs labelled single-appliance/device files")
    df = pd.concat(rows, ignore_index=True)
    if df.group.nunique() == 1:
        # all files in one flat folder (the recordings dir): each FILE is the
        # instance, so the held-out split tests unseen recordings, not
        # memorized rows of the same recording
        df["group"] = df["stem"]
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
               "n_windows": int(len(df)), "classes": labels, "trained_utc": _now_utc(),
               "holdout_macro_f1": float(f1_score(y[te], yp, average="macro")),
               "holdout_accuracy": float(accuracy_score(y[te], yp))}
    print(f"  {split}\n  HELD-OUT macro-F1={metrics['holdout_macro_f1']:.3f} "
          f"acc={metrics['holdout_accuracy']:.3f}")
    joblib.dump({"task": "identify", "model": model, "features": feats, "labels": labels,
                 "window_s": args.window, "stride_s": args.stride, "metrics": metrics},
                os.path.join(out, "model_identify.joblib"))
    _confusion(y[te], yp, labels, os.path.join(out, "train_identify_confusion.png"),
               f"Identification held-out (macro-F1={metrics['holdout_macro_f1']:.2f})")
    json.dump(metrics, open(os.path.join(out, "train_identify_metrics.json"), "w"), indent=2)
    print(f"  saved -> {out}/model_identify.joblib")


def _scenario_harm_ok(sig, path) -> bool:
    """Does this scenario carry a REAL aggregate current spectrum for the
    loads it contains? The mixer stamps metadata.harmonics_complete when every
    member recording had one (authoritative); without the stamp, fall back to
    content: substantial power with a ~zero spectrum means some member was
    recorded without harmonics and zero-filled."""
    try:
        with h5py.File(path, "r") as h:
            if "metadata" in h and "harmonics_complete" in h["metadata"].attrs:
                return bool(h["metadata"].attrs["harmonics_complete"])
    except OSError:
        pass
    if sig.harm_I is None:
        return False
    energy = np.sqrt(np.nansum(np.asarray(sig.harm_I, float) ** 2, axis=1))
    act = np.abs(np.nan_to_num(sig.P)) > 20.0
    if not act.any():
        return True
    return float((energy[act] < 1e-4).mean()) <= 0.05


def _agg_features_for(args, harm_ok: bool) -> list:
    """Aggregate feature list for this training run. 'auto' keeps the five
    harmonic columns only when EVERY scenario has a trustworthy spectrum --
    a model must never learn 'family X = zero harmonics' from a recording
    gap the live meter does not have."""
    mode = getattr(args, "agg_features", "auto") or "auto"
    if mode == "harm" or (mode == "auto" and harm_ok):
        return list(nl.AGG_FEATURES)
    return list(nl.AGG_FEATURES_BASE)


def _collect_agg(files, args, what):
    """Stack aggregate windows + power/presence targets across scenario files,
    with the appliance vocabulary derived from the data itself. Also reports
    whether every file carried real harmonic content (drives the feature
    choice in _agg_features_for)."""
    canon = nl.scan_canon(files)
    if not canon:
        sys.exit(f"{what} needs scenario .h5 files with /ground_truth")
    Xs, Ys, gs, k, harm_ok = [], [], [], 0, True
    for f in files:                                   # stream one scenario at a time
        try:
            s = nl.load_signal(f)
        except (OSError, KeyError) as e:
            print(f"  SKIP (unreadable, ignoring): {os.path.basename(f)}  [{e}]")
            continue
        if s.gt_P is None:
            del s; continue
        harm_ok = harm_ok and _scenario_harm_ok(s, f)
        X, Y, _ = nl.aggregate_windows(s, args.window, canon=canon)
        Xs.append(X); Ys.append(Y); gs.append(np.full(len(X), k)); k += 1
        del s
    if not Xs:
        sys.exit(f"{what} needs scenario .h5 files with /ground_truth")
    return np.vstack(Xs), np.vstack(Ys), np.concatenate(gs), k, canon, harm_ok


def _agg_split(X, Y, g, k):
    if k >= 2:
        return next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(X, Y, g))
    return train_test_split(np.arange(len(X)), test_size=0.3, random_state=0)


def train_disaggregate(files, args, out):
    X, Y, g, k, names, harm_ok = _collect_agg(files, args, "disaggregate")
    feats = _agg_features_for(args, harm_ok)
    X = X[:, :len(feats)]
    print(f"disaggregation: {X.shape[0]} windows from {k} scenarios, "
          f"{len(names)} appliances: {', '.join(names)} | {len(feats)} features"
          + ("" if harm_ok else " (harmonic columns dropped: corpus incomplete)"))
    tr, te = _agg_split(X, Y, g, k)
    base = (_lgbm_reg() if args.model == "lgbm"
            else RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))
    model = MultiOutputRegressor(base).fit(X[tr], Y[tr])
    P = model.predict(X[te])
    per = {names[i]: float(mean_absolute_error(Y[te][:, i], P[:, i])) for i in range(len(names))}
    metrics = {"task": "disaggregate", "model": args.model, "features": feats,
               "appliances": names, "trained_utc": _now_utc(), "n_windows": int(X.shape[0]),
               "overall_mae_W": float(np.mean(list(per.values()))),
               "per_appliance_mae_W": per}
    print(f"  held-out overall MAE={metrics['overall_mae_W']:.1f} W")
    print("  per-appliance MAE(W): " + ", ".join(f"{a}={v:.0f}" for a, v in per.items()))
    joblib.dump({"task": "disaggregate", "model": model, "features": feats,
                 "appliances": names, "window_s": args.window, "metrics": metrics},
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
    X, Yp, g, k, names, harm_ok = _collect_agg(files, args, "presence")
    feats = _agg_features_for(args, harm_ok)
    X = X[:, :len(feats)]
    Y = (np.abs(Yp) > args.on_w).astype(int)
    print(f"presence: {X.shape[0]} windows from {k} scenarios, {Y.shape[1]} appliances "
          f"(on > {args.on_w:g} W) | {len(feats)} features"
          + ("" if harm_ok else " (harmonic columns dropped: corpus incomplete)"))
    tr, te = _agg_split(X, Y, g, k)
    # presence uses RandomForest: robust to always-on appliances (single-class
    # columns like baseload) that would break a per-output LightGBM.
    model = MultiOutputClassifier(_base_clf("rf")).fit(X[tr], Y[tr])
    Pp = np.asarray(model.predict(X[te]))
    per, macro, support = nl.presence_f1(Y[te], Pp, names)
    metrics = {"task": "presence", "model": "rf", "features": feats,
               "appliances": names, "on_W": args.on_w, "trained_utc": _now_utc(),
               "n_windows": int(X.shape[0]),
               "macro_f1": macro, "per_appliance_f1": per,
               "holdout_on_windows": support}
    print(f"  held-out presence macro-F1 = {metrics['macro_f1']:.3f} "
          f"(over appliances present in the held-out set)")
    print("  per-appliance F1: " + ", ".join(
        f"{a}={v:.2f}" if v is not None else f"{a}=n/a" for a, v in per.items()))
    joblib.dump({"task": "presence", "model": model, "features": feats,
                 "appliances": names, "window_s": args.window, "on_W": args.on_w,
                 "metrics": metrics},
                os.path.join(out, "model_presence.joblib"))
    json.dump(metrics, open(os.path.join(out, "train_presence_metrics.json"), "w"), indent=2)
    print(f"  saved -> {out}/model_presence.joblib")


def train_mix(files, args, out):
    """Presence + disaggregation together: ONE bundle that answers both
    'which appliances are ON' and 'how much power does each draw' per window.
    This is the model the live monitor uses."""
    X, Ypow, g, k, names, harm_ok = _collect_agg(files, args, "mix")
    feats = _agg_features_for(args, harm_ok)
    X = X[:, :len(feats)]
    Yon = (np.abs(Ypow) > args.on_w).astype(int)
    print(f"mix (presence+disaggregate): {X.shape[0]} windows from {k} scenarios, "
          f"{len(names)} appliances: {', '.join(names)} | {len(feats)} features"
          + ("" if harm_ok else " (harmonic columns dropped: corpus incomplete)"))
    tr, te = _agg_split(X, Ypow, g, k)

    presence = MultiOutputClassifier(_base_clf("rf")).fit(X[tr], Yon[tr])
    base = (_lgbm_reg() if args.model == "lgbm"
            else RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))
    power = MultiOutputRegressor(base).fit(X[tr], Ypow[tr])

    Pon = np.asarray(presence.predict(X[te]))
    Pw = power.predict(X[te])
    per_f1, macro_f1, support = nl.presence_f1(Yon[te], Pon, names)
    per_mae = {names[i]: float(mean_absolute_error(Ypow[te][:, i], Pw[:, i]))
               for i in range(len(names))}
    # combined output = power gated by presence (off -> 0 W); its MAE is the
    # honest end-to-end error of what the live monitor actually displays
    per_mae_gated = {names[i]: float(mean_absolute_error(Ypow[te][:, i], Pw[:, i] * Pon[:, i]))
                     for i in range(len(names))}
    metrics = {"task": "mix", "model": args.model, "features": feats,
               "appliances": names, "on_W": args.on_w, "trained_utc": _now_utc(),
               "n_windows": int(X.shape[0]),
               "presence_macro_f1": macro_f1,
               "per_appliance_f1": per_f1,
               "holdout_on_windows": support,
               "power_overall_mae_W": float(np.mean(list(per_mae.values()))),
               "per_appliance_mae_W": per_mae,
               "gated_overall_mae_W": float(np.mean(list(per_mae_gated.values()))),
               "per_appliance_gated_mae_W": per_mae_gated}
    print(f"  held-out presence macro-F1 = {metrics['presence_macro_f1']:.3f} "
          f"(over appliances present in the held-out set)   "
          f"power MAE = {metrics['power_overall_mae_W']:.1f} W   "
          f"gated MAE = {metrics['gated_overall_mae_W']:.1f} W")
    print("  per-appliance F1: " + ", ".join(
        f"{a}={v:.2f}" if v is not None else f"{a}=n/a" for a, v in per_f1.items()))
    print("  per-appliance MAE(W): " + ", ".join(f"{a}={v:.0f}" for a, v in per_mae.items()))
    joblib.dump({"task": "mix", "presence": presence, "power": power,
                 "features": feats, "appliances": names,
                 "window_s": args.window, "on_W": args.on_w, "metrics": metrics},
                os.path.join(out, "model_mix.joblib"))
    json.dump(metrics, open(os.path.join(out, "train_mix_metrics.json"), "w"), indent=2)
    print(f"  saved -> {out}/model_mix.joblib")


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
    ap.add_argument("--task", choices=["identify", "disaggregate", "presence", "mix"],
                    default="identify",
                    help="mix = presence + disaggregate in one bundle (for the live monitor)")
    ap.add_argument("--model", choices=["rf", "lgbm", "mlp"], default="rf",
                    help="mlp = neural network on the raw waveform (disaggregate/presence)")
    ap.add_argument("--features", choices=["auto", "common", "full"], default="auto",
                    help="identify only; 'common' = real-PAC4200 compatible")
    ap.add_argument("--agg-features", choices=["auto", "base", "harm"], default="base",
                    help="mix/presence/disaggregate feature set. 'base' (default) "
                         "= the 17 P/Q/PF/step features; measured on the real "
                         "mixed recordings it beats the harmonic set (set-F1 "
                         "0.767 vs 0.674, which also hallucinated big devices). "
                         "'harm' appends the 5 aggregate-spectrum columns; "
                         "'auto' = harm only when every scenario carries a real "
                         "spectrum. Revisit after the per-order features are "
                         "re-validated on a fully re-recorded corpus.")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--stride", type=float, default=30.0)
    ap.add_argument("--on-threshold", type=float, default=5.0)
    ap.add_argument("--on-w", type=float, default=15.0,
                    help="presence/mix: |appliance power| above this counts as ON (W)")
    ap.add_argument("--raw-labels", action="store_true",
                    help="identify: keep full recording labels instead of device families")
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
         "presence": train_presence, "mix": train_mix}[args.task](files, args, args.out)


if __name__ == "__main__":
    main()
