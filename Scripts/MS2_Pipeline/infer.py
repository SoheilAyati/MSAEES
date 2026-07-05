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


def _appliance_colors(names=None):
    names = list(names) if names is not None else nl.CANON
    cmap = plt.get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(names)}


def expected_set_accuracy(sig, names, Pon, min_frac=0.15):
    """Device-set accuracy from the recording's own label, no ground truth needed.

    Real recordings are named after what was plugged in ('water_boiler_on',
    'pv__table_lamp_on'), so the expected device set is known even though there
    is no per-sample ground truth. A device counts as DETECTED when it is
    predicted ON in at least `min_frac` of the windows. Returns a summary dict
    (or None when the label carries no usable device names).
    """
    if sig.gt_P is not None or not sig.label:
        return None          # scenarios have real ground truth; use that instead
    expected = [f for f in nl.parse_families(sig.label) if f in names]
    if not expected:
        return None
    frac = {names[i]: float(np.asarray(Pon)[:, i].mean()) for i in range(len(names))}
    detected = [n for n, v in frac.items() if v >= min_frac]
    tp = len(set(expected) & set(detected))
    prec = tp / len(detected) if detected else 0.0
    rec = tp / len(expected) if expected else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"expected_devices": expected, "detected_devices": detected,
            "missed": sorted(set(expected) - set(detected)),
            "false_alarms": sorted(set(detected) - set(expected)),
            "set_precision": round(prec, 3), "set_recall": round(rec, 3),
            "set_f1": round(f1, 3),
            "fraction_windows_on": {k: round(v, 3) for k, v in frac.items() if v > 0}}


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
    # Colour by the labels that actually occur (the model's own classes plus any
    # ground-truth bases), so the prediction panel is coloured even when the model
    # was trained on labels outside CANON - e.g. real recordings like 'laptop_ravi'
    # or 'stand_cooler_1'. Previously this looped over CANON only, so a model whose
    # classes were not in CANON produced an all-grey (uncoloured) prediction panel.
    gt_bases = [nm.rsplit("_", 1)[0] for nm in sig.gt_names] if sig.gt_P is not None else []
    label_universe = list(dict.fromkeys(list(classes) + gt_bases)) or list(nl.CANON)
    colors = _appliance_colors(label_universe)
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
    for app in label_universe:
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
    if any((sec == a).any() for a in label_universe):
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
    X, Y, _ = nl.aggregate_windows(sig, ws, canon=names)
    X = nl.slice_features(X, bundle.get("features"))
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
    X, Ytrue, _ = nl.aggregate_presence(sig, ws, on_W=on_W, canon=names)
    X = nl.slice_features(X, bundle.get("features"))
    Pp = np.asarray(model.predict(X)).astype(int)
    proba = nl.presence_proba(model, X)
    hours = np.arange(X.shape[0]) * ws / 3600.0
    df = pd.DataFrame(Pp, columns=names); df.insert(0, "hour", hours)
    for i, nm in enumerate(names):
        df[f"prob_{nm}"] = np.round(proba[:, i], 3)
    df.to_csv(os.path.join(out, "presence.csv"), index=False)

    frac = {names[i]: float(Pp[:, i].mean()) for i in range(len(names))}
    # confidence = how sure the model is of the on/off calls it made
    conf = float(np.mean(np.where(Pp.astype(bool), proba, 1.0 - proba)))
    summary = {"input": sig.name, "task": "presence", "fraction_on": frac,
               "mean_confidence": round(conf, 3)}
    if bundle.get("metrics"):
        summary["model_holdout_metrics"] = bundle["metrics"]
    acc_line = f"mean confidence {conf:.2f}"
    if Ytrue is not None:
        per, macro, support = nl.presence_f1(Ytrue, Pp, names)
        summary["per_appliance_f1"] = per
        summary["macro_f1"] = macro
        summary["gt_on_windows"] = support
        acc_line = f"macro-F1 {summary['macro_f1']:.2f} vs ground truth · " + acc_line
        print(f"  presence macro-F1 vs ground truth: {summary['macro_f1']:.3f}")
    setacc = expected_set_accuracy(sig, names, Pp)
    if setacc:
        summary["label_set_accuracy"] = setacc
        acc_line = (f"device set F1 {setacc['set_f1']:.2f} "
                    f"(expected {'+'.join(setacc['expected_devices'])}) · " + acc_line)
        print(f"  expected devices: {setacc['expected_devices']}  ->  "
              f"detected: {setacc['detected_devices']}  (set-F1 {setacc['set_f1']:.2f})")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)

    colors = _appliance_colors(names)
    nrows = 2 if Ytrue is not None else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 0.45 * len(names) * nrows + 1.5),
                             squeeze=False, sharex=True)

    def gantt(ax, M, title):
        for i, nm in enumerate(names):
            ax.fill_between(hours, i + 0.05, i + 0.95, where=M[:, i].astype(bool),
                            color=colors.get(nm, "0.5"), step="mid", linewidth=0)
        ax.set_yticks([i + 0.5 for i in range(len(names))]); ax.set_yticklabels(names, fontsize=8)
        ax.set_ylim(0, len(names)); ax.set_title(title)

    gantt(axes[0, 0], Pp, f"Predicted appliance presence — {sig.name}\n{acc_line}")
    if Ytrue is not None:
        gantt(axes[1, 0], Ytrue, "Ground-truth presence")
    axes[-1, 0].set_xlabel("hour")
    plt.tight_layout(); plt.savefig(os.path.join(out, "presence_timeline.png"), dpi=110); plt.close()
    print("  wrote presence.csv, summary.json, presence_timeline.png")


def run_mix(sig, bundle, args, out):
    """Combined presence + disaggregation: which devices are ON and how much
    power each one draws, in one timeline. Power is gated by presence (a device
    predicted OFF contributes 0 W), and the leftover between the measured total
    and the sum of per-device estimates is reported as unexplained residual."""
    names = bundle["appliances"]
    ws = bundle.get("window_s", 30.0); on_W = bundle.get("on_W", 15.0)
    X, Ypow, _ = nl.aggregate_windows(sig, ws, canon=names)
    X = nl.slice_features(X, bundle.get("features"))
    Ytrue_on = None if Ypow is None else (np.abs(Ypow) > on_W).astype(int)
    Pon = np.asarray(bundle["presence"].predict(X)).astype(int)
    proba = nl.presence_proba(bundle["presence"], X)
    Pw = bundle["power"].predict(X)
    Pgated = Pw * Pon
    hours = np.arange(X.shape[0]) * ws / 3600.0
    Ptot_meas = X[:, 0]                                   # Ptot_mean feature
    residual = Ptot_meas - Pgated.sum(axis=1)

    df = pd.DataFrame({"hour": hours, "P_total_measured_W": np.round(Ptot_meas, 1),
                       "residual_W": np.round(residual, 1)})
    for i, nm in enumerate(names):
        df[f"on_{nm}"] = Pon[:, i]
        df[f"prob_{nm}"] = np.round(proba[:, i], 3)
        df[f"P_{nm}_W"] = np.round(Pgated[:, i], 1)
    df.to_csv(os.path.join(out, "mix_timeline.csv"), index=False)

    energy = {names[i]: float(Pgated[:, i].sum() * ws / 3600.0 / 1000.0)
              for i in range(len(names))}
    conf = float(np.mean(np.where(Pon.astype(bool), proba, 1.0 - proba)))
    expl = 1.0 - (np.abs(residual).sum() / max(np.abs(Ptot_meas).sum(), 1e-9))
    summary = {"input": sig.name, "task": "mix",
               "fraction_on": {names[i]: float(Pon[:, i].mean()) for i in range(len(names))},
               "energy_kWh_per_appliance": energy,
               "mean_confidence": round(conf, 3),
               "explained_power_fraction": round(float(expl), 3),
               "mean_abs_residual_W": round(float(np.abs(residual).mean()), 1)}
    if bundle.get("metrics"):
        summary["model_holdout_metrics"] = bundle["metrics"]
    acc_bits = [f"confidence {conf:.2f}", f"explained power {100*expl:.0f}%"]
    if Ypow is not None:
        from sklearn.metrics import mean_absolute_error
        per, macro, support = nl.presence_f1(Ytrue_on, Pon, names)
        summary["per_appliance_f1"] = per
        summary["macro_f1"] = macro
        summary["gt_on_windows"] = support
        summary["per_appliance_mae_W"] = {
            names[i]: float(mean_absolute_error(Ypow[:, i], Pgated[:, i]))
            for i in range(len(names))}
        summary["overall_mae_W"] = float(np.mean(list(summary["per_appliance_mae_W"].values())))
        acc_bits.insert(0, f"presence F1 {summary['macro_f1']:.2f} · "
                           f"power MAE {summary['overall_mae_W']:.0f} W vs ground truth")
        print(f"  vs ground truth: presence macro-F1 {summary['macro_f1']:.3f}, "
              f"gated power MAE {summary['overall_mae_W']:.1f} W")
    setacc = expected_set_accuracy(sig, names, Pon)
    if setacc:
        summary["label_set_accuracy"] = setacc
        acc_bits.insert(0, f"device set F1 {setacc['set_f1']:.2f} "
                           f"(expected {'+'.join(setacc['expected_devices'])})")
        print(f"  expected devices: {setacc['expected_devices']}  ->  "
              f"detected: {setacc['detected_devices']}  (set-F1 {setacc['set_f1']:.2f})")
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)

    # ---- combined plot: stacked per-device power + presence gantt ----
    colors = _appliance_colors(names)
    if hours[-1] < 0.5:                      # short recording -> minutes axis
        hours = hours * 60.0
        xlab = "minute"
    else:
        xlab = "hour"
    live = [i for i in range(len(names)) if np.abs(Pgated[:, i]).max() > 1]
    nrows = 3 if Ytrue_on is not None else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3.2 + 2.6 + 0.4 * len(names) * (nrows - 1)),
                             sharex=True,
                             gridspec_kw={"height_ratios": [2.2] + [1] * (nrows - 1)})
    ax = axes[0]
    if live:
        ax.stackplot(hours, [np.clip(Pgated[:, i], 0, None) for i in live],
                     labels=[names[i] for i in live],
                     colors=[colors[names[i]] for i in live], alpha=0.8)
    neg = np.clip(Pgated, None, 0)
    if (neg < 0).any():
        ax.stackplot(hours, [neg[:, i] for i in live],
                     colors=[colors[names[i]] for i in live], alpha=0.8)
    ax.plot(hours, Ptot_meas, color="black", lw=1.3, label="measured total")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_ylabel("P (W)")
    ax.set_title(f"Mix (presence + disaggregation) — {sig.name}\n" + " · ".join(acc_bits))

    def gantt(ax, M, title):
        for i, nm in enumerate(names):
            ax.fill_between(hours, i + 0.05, i + 0.95, where=M[:, i].astype(bool),
                            color=colors.get(nm, "0.5"), step="mid", linewidth=0)
        ax.set_yticks([i + 0.5 for i in range(len(names))]); ax.set_yticklabels(names, fontsize=8)
        ax.set_ylim(0, len(names)); ax.set_title(title, fontsize=9)

    gantt(axes[1], Pon, "Predicted presence")
    if Ytrue_on is not None:
        gantt(axes[2], Ytrue_on, "Ground-truth presence")
    axes[-1].set_xlabel(xlab)
    plt.tight_layout(); plt.savefig(os.path.join(out, "mix_timeline.png"), dpi=110); plt.close()
    print("  wrote mix_timeline.csv, summary.json, mix_timeline.png")


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
    elif bundle["task"] == "mix":
        run_mix(sig, bundle, args, out)
    else:
        run_disaggregate(sig, bundle, args, out)


if __name__ == "__main__":
    main()
