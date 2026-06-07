"""
nilm_pipeline.py  --  shared library for the MS2 pipelines
==========================================================

One small module that both pipelines (train.py, infer.py) import. It hides the
difference between the input file types so the rest of the code is simple:

    load_signal(path)        ->  Signal   (works for .csv real-meter runs,
                                           .h5 single-appliance, .h5 scenario)
    window_features(sig)     ->  DataFrame, one row per time window  (for
                                           appliance IDENTIFICATION)
    aggregate_windows(sig)   ->  (X, Y, names)  aggregate features + per-
                                           appliance power targets (for
                                           DISAGGREGATION; needs ground truth)

Feature columns are deliberately few. FEATURES_FULL is used when the input has
per-order harmonics (synthetic h5); FEATURES_COMMON is the subset a real
PAC4200 CSV also provides, so a model can transfer to the real meter.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

try:
    import h5py
except ImportError as e:                                    # pragma: no cover
    raise ImportError("h5py required: pip install h5py") from e

CANON = ["baseload", "ev", "fridge", "hair_dryer", "pc", "pv", "resistive",
         "synchronous", "washing_machine"]

FEATURES_COMMON = ["P_mean", "Q_mean", "S_mean", "PF_mean", "QP_ratio",
                   "THD_I_mean", "P_std", "P_min", "P_max"]
FEATURES_HARM = ["h3", "h5", "h7", "h_centroid", "h_energy"]
FEATURES_FULL = FEATURES_COMMON + FEATURES_HARM

# aggregate (disaggregation) feature set
AGG_FEATURES = ["Ptot_mean", "Ptot_std", "Ptot_min", "Ptot_max", "Qtot_mean",
                "PL1_mean", "PL2_mean", "PL3_mean", "PF_mean", "THDI_mean", "hour"]


class Signal:
    """Uniform container for any input file."""
    def __init__(self, **kw):
        self.source = kw["source"]            # 'csv' | 'h5_single' | 'h5_scenario'
        self.name = kw.get("name", "signal")
        self.label = kw.get("label")          # appliance/device label if known
        self.sample_rate_hz = float(kw.get("sample_rate_hz", 5.0))
        self.t = kw["t"]                       # seconds from start
        self.P = kw["P"]; self.Q = kw["Q"]; self.S = kw["S"]
        self.PF = kw["PF"]; self.THD_I = kw["THD_I"]
        self.P_phase = kw.get("P_phase")       # (T,3) or None
        self.harm_I = kw.get("harm_I")         # (T,39) or None
        self.gt_names = kw.get("gt_names")     # list or None
        self.gt_P = kw.get("gt_P")             # (T,N) or None

    @property
    def has_harmonics(self):
        return self.harm_I is not None

    @property
    def n(self):
        return len(self.P)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_signal(path: str) -> Signal:
    if str(path).lower().endswith(".csv"):
        return _load_csv(path)
    return _load_h5(path)


def _clean(a):
    a = np.array(a, dtype=np.float64)   # np.array() copies -> writable (asarray may be read-only)
    a[~np.isfinite(a)] = np.nan
    return a


def _load_csv(path):
    df = pd.read_csv(path, sep=";")
    t = pd.to_datetime(df["timestamp_iso"], utc=True, format="ISO8601")
    t_s = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    sr = 1.0 / np.median(np.diff(t_s)) if len(t_s) > 1 else 5.0
    P = _clean(df["p_total_w"]); Q = _clean(df["q_total_var"])
    return Signal(source="csv", name=str(df["device_name"].iloc[0]),
                  label=str(df["device_name"].iloc[0]), sample_rate_hz=sr, t=t_s,
                  P=P, Q=Q, S=_clean(df["s_total_va"]), PF=_clean(df["pf_total"]),
                  THD_I=_clean(df["thd_i_l1_percent"]))


def _attr(v):
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


def _load_h5(path):
    with h5py.File(path, "r") as f:
        meta = {k: _attr(v) for k, v in f["metadata"].attrs.items()} if "metadata" in f else {}
        sr = float(meta.get("sample_rate_hz", 5.0))
        ts = f["timestamp"][:]
        t_s = (ts - ts[0]) / 1e6
        m = f["measurements"]
        if "P_total" in m:                                  # aggregate scenario
            P = _clean(m["P_total"][:]); Q = _clean(m["Q_total"][:])
            S = _clean(m["S_total"][:]); PF = _clean(m["PF_total"][:])
            THD = _clean(m["THD_I_L1"][:])
            Pph = np.column_stack([_clean(m[f"P_L{i}"][:]) for i in (1, 2, 3)])
            harm = m["harmonics/I_mag_L1"][:] if "harmonics/I_mag_L1" in m else None
            gtn = gtP = None
            if "ground_truth" in f:
                gtn = [_attr(x) for x in f["ground_truth"]["appliance_names"][:]]
                gtP = f["ground_truth"]["P_contribution"][:].astype(np.float64)
            return Signal(source="h5_scenario", name=_stem(path), sample_rate_hz=sr,
                          t=t_s, P=P, Q=Q, S=S, PF=PF, THD_I=THD, P_phase=Pph,
                          harm_I=harm, gt_names=gtn, gt_P=gtP)
        else:                                               # single appliance
            P = _clean(m["P"][:]); Q = _clean(m["Q"][:])
            S = np.hypot(P, Q)
            PF = np.divide(P, S, out=np.ones_like(P), where=S > 1e-6)
            harm = m["harmonics_I_mag"][:] if "harmonics_I_mag" in m else None
            name = "unknown"
            try:
                name = json.loads(meta.get("appliance_metadata", "{}")).get("name", "unknown")
            except Exception:
                pass
            return Signal(source="h5_single", name=name, label=name, sample_rate_hz=sr,
                          t=t_s, P=P, Q=Q, S=S, PF=PF, THD_I=np.full_like(P, np.nan),
                          harm_I=harm)


def _stem(path):
    import os
    return os.path.splitext(os.path.basename(path))[0]


# --------------------------------------------------------------------------
# Identification features (one row per window)
# --------------------------------------------------------------------------
def _harm_feats(block):
    if block is None or block.size == 0:
        return dict(h3=np.nan, h5=np.nan, h7=np.nan, h_centroid=np.nan, h_energy=np.nan)
    mag = np.nanmean(block, axis=0)                # orders 2..40 -> idx n-2
    orders = np.arange(2, 41)
    energy = float(np.sqrt(np.nansum(mag ** 2)))
    centroid = float(np.nansum(orders * mag) / (np.nansum(mag) + 1e-9))
    return dict(h3=float(mag[1]), h5=float(mag[3]), h7=float(mag[5]),
                h_centroid=centroid, h_energy=energy)


def window_features(sig: Signal, window_s=30.0, stride_s=30.0, on_threshold_W=5.0):
    """One feature row per window across the whole signal (active flag included)."""
    sr = sig.sample_rate_hz
    w = max(1, int(round(window_s * sr)))
    step = max(1, int(round(stride_s * sr)))
    rows = []
    for s in range(0, sig.n - w + 1, step):
        e = s + w
        P = sig.P[s:e]; Q = sig.Q[s:e]
        lo, hi = int(0.2 * w), max(int(0.8 * w), int(0.2 * w) + 1)
        P_ss = float(np.nanmedian(P[lo:hi])); Q_ss = float(np.nanmedian(Q[lo:hi]))
        S_ss = float(np.hypot(P_ss, Q_ss))
        feat = dict(
            P_mean=P_ss, Q_mean=Q_ss, S_mean=S_ss,
            PF_mean=float(P_ss / S_ss) if S_ss > 1e-6 else np.nan,
            QP_ratio=float(Q_ss / (abs(P_ss) + 1e-6)),
            P_std=float(np.nanstd(P)), P_min=float(np.nanmin(P)), P_max=float(np.nanmax(P)),
        )
        h = _harm_feats(sig.harm_I[s:e]) if sig.has_harmonics else _harm_feats(None)
        feat.update(h)
        if np.isfinite(sig.THD_I[s:e]).any():                 # measured THD_I
            feat["THD_I_mean"] = float(np.nanmedian(sig.THD_I[s:e]))
        else:                                                  # derive from harmonics
            Ifund = np.hypot(P_ss, Q_ss) / 230.0
            feat["THD_I_mean"] = 100.0 * feat["h_energy"] / (Ifund + 1e-6) if sig.has_harmonics else np.nan
        feat["start_s"] = float(sig.t[s]); feat["end_s"] = float(sig.t[min(e, sig.n) - 1])
        feat["active"] = bool(np.nanmean(np.abs(P)) > on_threshold_W)
        rows.append(feat)
    return pd.DataFrame(rows)


def feature_set_for(signals):
    """Full feature set only if every signal has harmonics; else the common subset."""
    return FEATURES_FULL if all(s.has_harmonics for s in signals) else FEATURES_COMMON


# --------------------------------------------------------------------------
# Disaggregation features (aggregate window -> per-appliance power)
# --------------------------------------------------------------------------
def aggregate_windows(sig: Signal, window_s=30.0):
    """Return (X, Y, appliance_names). Y is None if the file has no ground truth."""
    sr = sig.sample_rate_hz
    w = max(1, int(round(window_s * sr)))
    n = (sig.n // w) * w
    def W(a):
        return np.asarray(a[:n], float).reshape(-1, w)
    hour = ((sig.t - sig.t[0]) / 3600.0) % 24
    Pph = sig.P_phase if sig.P_phase is not None else np.zeros((sig.n, 3))
    X = np.column_stack([
        np.nanmean(W(sig.P), 1), np.nanstd(W(sig.P), 1), np.nanmin(W(sig.P), 1), np.nanmax(W(sig.P), 1),
        np.nanmean(W(sig.Q), 1),
        np.nanmean(W(Pph[:, 0]), 1), np.nanmean(W(Pph[:, 1]), 1), np.nanmean(W(Pph[:, 2]), 1),
        np.nanmean(W(sig.PF), 1), np.nanmean(W(sig.THD_I), 1), np.nanmean(W(hour), 1),
    ])
    X = np.nan_to_num(X)
    Y = None
    if sig.gt_P is not None:
        Y = np.zeros((X.shape[0], len(CANON)))
        for col, nm in enumerate(sig.gt_names):
            base = nm.rsplit("_", 1)[0]
            if base in CANON:
                Y[:, CANON.index(base)] = np.nanmean(W(sig.gt_P[:, col]), 1)
    return X, Y, CANON


def aggregate_presence(sig, window_s=30.0, on_W=15.0):
    """Multi-hot 'appliance active' targets per aggregate window (|power| > on_W).

    Reuses aggregate_windows; returns (X, Y_multihot, names). Y is None when the
    file has no ground truth.
    """
    X, Yp, names = aggregate_windows(sig, window_s)
    Y = None if Yp is None else (np.abs(Yp) > on_W).astype(int)
    return X, Y, names


def aggregate_sequences(sig, window_s=30.0, on_W=15.0):
    """Flatten the raw P/Q/THD waveform of each (non-overlapping) window.

    Returns (X_flat, Y_power, Y_presence, names):
      X_flat     (N, 3*W)   raw [P, Q, THD_I] samples per window (the 'sequence')
      Y_power    (N, n_app)  window-mean per-appliance power   (None if no GT)
      Y_presence (N, n_app)  1 if |power| > on_W else 0        (None if no GT)

    Used by the neural-network (MLP) path, which learns from the waveform shape
    itself rather than the hand-crafted summary features.
    """
    sr = sig.sample_rate_hz
    w = max(1, int(round(window_s * sr)))
    n = (sig.n // w) * w

    def Wn(a):
        return np.asarray(a[:n], float).reshape(-1, w)

    chans = [np.nan_to_num(Wn(sig.P)), np.nan_to_num(Wn(sig.Q)), np.nan_to_num(Wn(sig.THD_I))]
    Xf = np.concatenate(chans, axis=1).astype(np.float32)        # (N, 3*W)
    Ypow = Ypres = None
    if sig.gt_P is not None:
        Ypow = np.zeros((Xf.shape[0], len(CANON)))
        for col, nm in enumerate(sig.gt_names):
            base = nm.rsplit("_", 1)[0]
            if base in CANON:
                Ypow[:, CANON.index(base)] = np.nanmean(Wn(sig.gt_P[:, col]), 1)
        Ypres = (np.abs(Ypow) > on_W).astype(int)
    return Xf, Ypow, Ypres, CANON
