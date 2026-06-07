"""
run_ms2_demo.py  --  Milestone 2 end-to-end smoke demo
======================================================

Runs the whole Stage 2->5 chain on the data already in the repo and prints a
summary, so you can see every piece working before generating more scenarios:

  1. EVENT DETECTION on the mixed scenario, scored against ground truth.
  2. FEATURE EXTRACTION from the 9 single-appliance files -> labelled table.
  3. CLUSTERING of that table (unsupervised) with ARI/NMI vs the true labels.
  4. CLASSIFICATION (Random Forest, + LightGBM if installed) with k-fold F1.
  5. REAL DATA: event detection + steady-state fingerprints for the 6 CSVs.

It also saves a few PNGs to Scripts/MS2/demo_output/.

Usage:
    python run_ms2_demo.py                 # uses repo-relative default paths
    python run_ms2_demo.py /path/to/repo   # override repo root

sklearn-only; no deep-learning dependency.
"""

from __future__ import annotations

import os
import sys
import glob

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, "..", ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nilm_io as io
import event_detection as ed
import feature_extraction as fx
import clustering as cl
import classification as cls

OUT = os.path.join(HERE, "demo_output")
os.makedirs(OUT, exist_ok=True)


def banner(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def stage1_events():
    banner("STAGE 2  --  EVENT DETECTION on scenario_normal.h5")
    path = os.path.join(REPO, "Synthetic_Data", "Mixed", "scenario_normal.h5")
    scn = io.load_scenario(path)
    P = scn.meas["P_total"]
    # Score against *significant* aggregate switching (|dP|>=50W). Sub-state
    # micro-changes (PC load flutter, washing-machine motor cycling) are real
    # but mostly invisible at the PCC, so a 50 W floor is the fair benchmark.
    truth = io.ground_truth_events(scn, min_dP=50.0)

    hart = ed.detect_edges(P, scn.sample_rate_hz, threshold_W=40.0, min_gap_s=2.0)
    # CUSUM on a lightly smoothed signal with a generous threshold; raw P
    # with a low threshold over-segments the PV ramp and motor cycling.
    Psm = np.convolve(P, np.ones(5) / 5, mode="same")
    cusum = ed.detect_cusum(Psm, scn.sample_rate_hz, drift_W=25.0,
                            threshold_W=250.0, min_gap_s=4.0)

    m_hart = ed.evaluate_events(hart, truth, scn.sample_rate_hz, tolerance_s=1.5)
    m_cus = ed.evaluate_events(cusum, truth, scn.sample_rate_hz, tolerance_s=1.5)
    print(f"ground-truth events (|dP|>=50W): {len(truth)}")
    print(f"  Hart edges : detected={m_hart['n_detected']:4d}  "
          f"P={m_hart['precision']:.2f} R={m_hart['recall']:.2f} F1={m_hart['f1']:.2f}  "
          f"(timing err {m_hart['mean_timing_error_s']}s)")
    print(f"  CUSUM      : detected={m_cus['n_detected']:4d}  "
          f"P={m_cus['precision']:.2f} R={m_cus['recall']:.2f} F1={m_cus['f1']:.2f}")

    sr = scn.sample_rate_hz
    sl = slice(int(9 * 3600 * sr), int(11 * 3600 * sr))
    t = scn.hours[sl]
    plt.figure(figsize=(11, 4))
    plt.plot(t, P[sl], lw=0.6, color="#1f4e79", label="P_total")
    for e in hart:
        if sl.start <= e.idx < sl.stop:
            plt.axvline(scn.hours[e.idx], color="#e07b39", lw=0.6, alpha=0.7)
    plt.xlabel("hour of day"); plt.ylabel("P_total (W)")
    plt.title("Event detection (Hart edges) -- 09:00-11:00 window")
    plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "events_window.png"), dpi=110); plt.close()
    return scn


def stage2_features_and_models():
    banner("STAGE 3-5  --  FEATURES, CLUSTERING, CLASSIFICATION (single-appliance files)")
    paths = sorted(glob.glob(os.path.join(REPO, "Synthetic_Data", "Single", "*.h5")))
    df = fx.build_window_table_from_singles(paths, window_s=30.0, stride_s=15.0,
                                            on_threshold_W=5.0, max_per_class=250)
    print(f"feature table: {len(df)} windows x {df.shape[1]} cols "
          f"from {len(paths)} appliances (<=250/class, 30s windows)")
    print("windows per appliance:")
    print(df["label"].value_counts().to_string())

    feats_all = ["P_mean", "Q_mean", "S_mean", "PF_mean", "QP_ratio",
                 "inrush_ratio", "rise_time_s", "P_std", "duration_s",
                 "h3", "h5", "h7", "h_centroid", "h_energy"]
    feats_all = [c for c in feats_all if c in df.columns]

    n_app = df["label"].nunique()
    klabels = cl.cluster(df, feats_all, method="kmeans", n_clusters=n_app)
    sc = cl.score(df["label"].to_numpy(), klabels, df, feats_all)
    print(f"\nKMeans(k={n_app}) vs true labels: "
          f"ARI={sc['ARI']:.2f} NMI={sc['NMI']:.2f} "
          f"homog={sc['homogeneity']:.2f} compl={sc['completeness']:.2f}"
          + (f" silhouette={sc['silhouette']:.2f}" if 'silhouette' in sc else ""))

    emb = cl.embed_2d(df, feats_all, method="pca")
    plt.figure(figsize=(8, 6))
    for lab in sorted(df["label"].unique()):
        m = df["label"].to_numpy() == lab
        plt.scatter(emb[m, 0], emb[m, 1], s=18, alpha=0.7, label=lab)
    plt.title("Appliance windows in feature space (PCA-2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(fontsize=7, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "feature_space_pca.png"), dpi=110); plt.close()

    print("\nClassification (label = appliance):")
    for kind in ("rf", "lgbm"):
        try:
            cvs = cls.cv_score(df, feats_all, kind=kind, n_splits=5)
        except ImportError as e:
            print(f"  {kind:4s}: skipped ({e})"); continue
        if cvs is None:
            print(f"  {kind:4s}: too few samples for CV"); continue
        print(f"  {kind:4s}: {cvs['folds']}-fold macro-F1 = "
              f"{cvs['mean_macro_f1']:.3f} +/- {cvs['std']:.3f}")

    model, mtr = cls.train_classifier(df, feats_all, kind="rf", test_size=0.3)
    print(f"  RF holdout: acc={mtr['accuracy']:.3f} macro-F1={mtr['macro_f1']:.3f}")
    imp = cls.feature_importance(model, feats_all)
    if imp is not None:
        print("  top features:", ", ".join(f"{k}({v:.2f})" for k, v in imp.head(5).items()))
    return df


def stage3_real():
    banner("REAL DATA  --  event detection + steady-state fingerprints (6 CSVs)")
    rows = []
    for csv in sorted(glob.glob(os.path.join(REPO, "Pre_Measured", "*.csv"))):
        rr = io.load_real_csv(csv)
        thr = max(3.0, 0.1 * np.nanmax(rr.P))
        ev = ed.detect_edges(rr.P, rr.sample_rate_hz, threshold_W=thr, min_gap_s=2.0)
        fp = fx.features_for_real_run(rr, on_threshold_W=3.0)
        if len(fp):
            on = "  ".join(
                f"P={r.P_mean:.1f}W Q={r.Q_mean:.1f}var PF={r.PF_mean:.2f} THD_I={r.THD_I_mean:.0f}%"
                for _, r in fp.iterrows())
        else:
            on = "(no on-segment)"
        print(f"  {rr.device_name:22s} rate={rr.sample_rate_hz:.1f}Hz events={len(ev):2d} | {on}")
        if len(fp):
            r0 = fp.iloc[0].to_dict()
            r0["device"] = rr.device_name
            rows.append(r0)
    if rows:
        cols = ["device", "P_mean", "Q_mean", "S_mean", "PF_mean", "THD_I_mean"]
        tbl = pd.DataFrame(rows)[cols]
        tbl.to_csv(os.path.join(OUT, "real_fingerprints.csv"), index=False)
        print("\nsaved fingerprint table -> " + os.path.join(OUT, "real_fingerprints.csv"))


if __name__ == "__main__":
    print(f"repo root: {REPO}")
    stage1_events()
    stage2_features_and_models()
    stage3_real()
    banner("DONE")
    print(f"plots + tables in: {OUT}")
