"""
feature_extraction.py  --  Milestone 2, Stage 3
===============================================

Turns raw channels into per-segment feature vectors -- one row per "appliance
operating interval" (or fixed window) -- that clustering and classification
consume.

Three feature families (cf. appliance_generator.md Appendix A, which says which
appliance is most distinguishable in which domain):

  steady-state   mean P, Q, S, PF, cos-phi over the segment; the classical
                 Hart (P, Q) signature plane.
  harmonic       3rd / 5th / 7th current-harmonic magnitudes, spectral centroid
                 and energy of the current-harmonic spectrum, THD_I.
  transient      inrush ratio (peak/steady), rise time, power variability;
                 plus context: duration and time-of-day.

Table builders:

  build_table_from_singles()        one row per active segment (event-style).
  build_window_table_from_singles() one row per fixed window over ON regions;
        more, more class-balanced samples -> the table used for ML.
  features_for_real_run()           steady-state ON fingerprint of a real
        PAC4200 CSV in COMMON feature space (synthetic->real transfer).

COMMON_FEATURE_COLUMNS lists the features computable on BOTH synthetic and real
data -- the only ones a transfer model may use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nilm_io import (SingleAppliance, RealRun, load_single, order_index)

# Features available on real PAC4200 data too (no per-order harmonics there).
COMMON_FEATURE_COLUMNS = ["P_mean", "Q_mean", "S_mean", "PF_mean", "THD_I_mean",
                          "QP_ratio"]


# --------------------------------------------------------------------------
# Segment helpers
# --------------------------------------------------------------------------
def active_segments(P, sample_rate_hz, on_threshold_W=5.0, min_len_s=3.0,
                    merge_gap_s=3.0):
    """Return [(start, end), ...] index ranges where |P| exceeds a threshold.

    Uses |P| so PV generation (negative P) is also captured as 'active'.
    Short gaps are bridged and very short segments dropped.
    """
    P = np.asarray(P, dtype=float)
    on = np.abs(P) > on_threshold_W
    merge_gap = int(round(merge_gap_s * sample_rate_hz))
    idx = np.where(on)[0]
    if len(idx) == 0:
        return []
    segs = []
    s = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap:
            segs.append((s, prev + 1))
            s = i
        prev = i
    segs.append((s, prev + 1))
    min_len = int(round(min_len_s * sample_rate_hz))
    return [(a, b) for a, b in segs if b - a >= min_len]


def _harmonic_features(Imag_seg):
    """Harmonic features from a (segment_len, 39) current-magnitude block."""
    if Imag_seg is None or Imag_seg.size == 0:
        return dict(h3=np.nan, h5=np.nan, h7=np.nan, h_centroid=np.nan,
                    h_energy=np.nan)
    m = np.nanmean(Imag_seg, axis=0)            # mean magnitude per order
    total = np.sum(m) + 1e-9
    orders = np.arange(2, 41)
    centroid = float(np.sum(orders * m) / total)
    return dict(
        h3=float(m[order_index(3)]),
        h5=float(m[order_index(5)]),
        h7=float(m[order_index(7)]),
        h_centroid=centroid,
        h_energy=float(np.sqrt(np.sum(m ** 2))),
    )


def _steady_and_transient(P_seg, Q_seg, sample_rate_hz):
    P_seg = np.asarray(P_seg, dtype=float)
    Q_seg = np.asarray(Q_seg, dtype=float)
    n = len(P_seg)
    lo, hi = int(0.2 * n), max(int(0.8 * n), int(0.2 * n) + 1)
    P_ss = float(np.median(P_seg[lo:hi]))
    Q_ss = float(np.median(Q_seg[lo:hi]))
    S_ss = float(np.hypot(P_ss, Q_ss))
    pf = float(P_ss / S_ss) if S_ss > 1e-6 else np.nan
    head = P_seg[:max(1, int(sample_rate_hz))]
    inrush = float(np.max(np.abs(head)) / (abs(P_ss) + 1e-6))
    target = 0.9 * P_ss
    rise = np.argmax(np.abs(P_seg) >= abs(target)) / sample_rate_hz
    return dict(P_mean=P_ss, Q_mean=Q_ss, S_mean=S_ss, PF_mean=pf,
                QP_ratio=float(Q_ss / (abs(P_ss) + 1e-6)),
                inrush_ratio=inrush, rise_time_s=float(rise),
                P_std=float(np.std(P_seg)),
                duration_s=float(n / sample_rate_hz))


# --------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------
def features_for_single(sa: SingleAppliance, on_threshold_W=5.0):
    """Feature rows for every active segment of a single-appliance file."""
    rows = []
    for (a, b) in active_segments(sa.P, sa.sample_rate_hz, on_threshold_W):
        feats = _steady_and_transient(sa.P[a:b], sa.Q[a:b], sa.sample_rate_hz)
        feats.update(_harmonic_features(sa.harm_I_mag[a:b]))
        feats["THD_I_mean"] = np.nan
        feats["label"] = sa.name
        feats["start_idx"], feats["end_idx"] = int(a), int(b)
        rows.append(feats)
    return rows


def build_table_from_singles(single_paths, on_threshold_W=5.0) -> pd.DataFrame:
    """Build a labelled feature table (one row per active segment)."""
    rows = []
    for p in single_paths:
        sa = load_single(p)
        rows.extend(features_for_single(sa, on_threshold_W))
    return pd.DataFrame(rows)


def windows_for_single(sa: SingleAppliance, window_s=30.0, stride_s=15.0,
                       on_threshold_W=5.0):
    """Fixed-length window features over the ON regions of a single file.

    This is the table used for classification/clustering: windowing the ON
    periods gives many, more class-balanced samples than one-row-per-segment
    (an always-on appliance would otherwise contribute a single huge segment).
    """
    sr = sa.sample_rate_hz
    w = max(1, int(round(window_s * sr)))
    step = max(1, int(round(stride_s * sr)))
    rows = []
    for (a, b) in active_segments(sa.P, sr, on_threshold_W):
        starts = list(range(a, b - w + 1, step))
        if not starts:                       # segment shorter than a window
            starts = [a]
        for s in starts:
            e = min(s + w, b)
            feats = _steady_and_transient(sa.P[s:e], sa.Q[s:e], sr)
            feats.update(_harmonic_features(sa.harm_I_mag[s:e]))
            feats["THD_I_mean"] = np.nan      # not stored per-appliance
            feats["label"] = sa.name
            feats["start_idx"], feats["end_idx"] = int(s), int(e)
            rows.append(feats)
    return rows


def build_window_table_from_singles(single_paths, window_s=30.0, stride_s=15.0,
                                    on_threshold_W=5.0, max_per_class=None,
                                    random_state=0) -> pd.DataFrame:
    """Windowed labelled feature table; optionally cap rows per class to balance."""
    rows = []
    for p in single_paths:
        sa = load_single(p)
        rows.extend(windows_for_single(sa, window_s, stride_s, on_threshold_W))
    df = pd.DataFrame(rows)
    if max_per_class is not None and len(df):
        rng = np.random.RandomState(random_state)
        # Manual per-class subsample. (Do NOT use groupby().apply(): in
        # pandas >= 2.2 it drops the grouping column, losing 'label'.)
        parts = [g.sample(min(len(g), max_per_class), random_state=rng)
                 for _, g in df.groupby("label")]
        df = pd.concat(parts).reset_index(drop=True)
    return df


def features_for_real_run(rr: RealRun, on_threshold_W=3.0) -> pd.DataFrame:
    """Steady-state ON fingerprint(s) of a real run, in COMMON feature space."""
    rows = []
    for (a, b) in active_segments(rr.P, rr.sample_rate_hz, on_threshold_W):
        P_ss = float(np.median(rr.P[a:b]))
        Q_ss = float(np.median(rr.Q[a:b]))
        S_ss = float(np.median(rr.S[a:b]))
        pf = float(np.nanmedian(rr.PF[a:b]))
        thd = float(np.nanmedian(rr.THD_I[a:b]))
        rows.append(dict(
            P_mean=P_ss, Q_mean=Q_ss, S_mean=S_ss, PF_mean=pf,
            THD_I_mean=thd, QP_ratio=float(Q_ss / (abs(P_ss) + 1e-6)),
            label=rr.device_name, start_idx=int(a), end_idx=int(b),
            duration_s=float((b - a) / rr.sample_rate_hz),
        ))
    return pd.DataFrame(rows)
