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
import re
import numpy as np
import pandas as pd

try:
    import h5py
except ImportError as e:                                    # pragma: no cover
    raise ImportError("h5py required: pip install h5py") from e

# Default appliance vocabulary: it both sizes the disaggregate/presence model
# output and is the set of names a scenario's ground truth can map to. A target
# column is filled only when the base of a ground-truth appliance name
# ("<name>_<instance>") is in this list (see aggregate_windows / _presence).
# Since the vocabulary became DYNAMIC (train.py derives it from the training
# data and stores it in the model bundle), CANON is only the fallback used when
# no explicit vocabulary is passed.
SYNTHETIC_APPLIANCES = ["baseload", "ev", "fridge", "hair_dryer", "pc", "pv",
                        "resistive", "synchronous", "washing_machine"]
# Real PAC4200 appliance families produced by mix_measured_scenarios.py. Appended
# (never inserted) so the synthetic indices 0-8 stay stable for existing models.
# Without these, measured scenarios map to NO class -> every target is zero -> the
# model learns "always off" and inference returns all zeros.
MEASURED_APPLIANCES = ["laptop", "stand_cooler", "table_fan", "table_pv"]
CANON = SYNTHETIC_APPLIANCES + MEASURED_APPLIANCES

# --------------------------------------------------------------------------
# Family colors: ONE canonical color per appliance family, shared by every
# chart (the mix_measured_scenarios decomposition PNGs and the live.py
# dashboard), so a device keeps the same color everywhere and the plots can
# be compared side by side. Each family has two steps of the SAME hue:
# 'light' for white-background matplotlib figures, 'dark' for the dark
# dashboard surface (each step set validated for CVD separation and contrast
# on its own surface).
FAMILY_COLORS = {
    "laptop":         {"light": "#2a78d6", "dark": "#3987e5"},   # blue
    "table_fan":      {"light": "#1baf7a", "dark": "#199e70"},   # aqua
    "pv":             {"light": "#eda100", "dark": "#c98500"},   # yellow
    "standing_fan":   {"light": "#008300", "dark": "#008300"},   # green
    "standing_lamp":  {"light": "#4a3aa7", "dark": "#9085e9"},   # violet
    "water_boiler":   {"light": "#e34948", "dark": "#e66767"},   # red
    "table_lamp":     {"light": "#e87ba4", "dark": "#d55181"},   # magenta
    "coffee_machine": {"light": "#eb6834", "dark": "#d95926"},   # orange
}


def _djb2(s: str) -> int:
    """32-bit djb2 string hash. live.py implements the identical function in
    JS so an unmapped family still gets the SAME color in both charts."""
    h = 5381
    for ch in s:
        h = (h * 33 + ord(ch)) & 0xFFFFFFFF
    return h


def family_color(name: str, mode: str = "light") -> str:
    """Canonical chart color for an appliance family (accepts instance names
    like 'table_fan_1' too). Families not in FAMILY_COLORS hash
    deterministically into the same palette, so newly taught devices keep a
    stable color across restarts and across both charts."""
    fam = parse_family(name)
    slot = FAMILY_COLORS.get(fam)
    if slot is None:
        keys = list(FAMILY_COLORS)
        slot = FAMILY_COLORS[keys[_djb2(fam) % len(keys)]]
    return slot["dark" if mode == "dark" else "light"]


# --------------------------------------------------------------------------
# Label parsing: recording label / filename  ->  device families
# --------------------------------------------------------------------------
# Recording labels encode the device plus its *setting*, e.g.
# "standing_fan_high_no_rotation" or "water_boiler_on". A double underscore
# separates SIMULTANEOUS devices in one recording: "pv__water_boiler_on" means
# PV and the water boiler were both connected. parse_family() collapses a
# single-device label to its family by stripping trailing state/setting words;
# parse_families() first splits on '__'.
STATE_WORDS = {
    "on", "off", "run", "running", "standby", "idle", "only", "trig",
    "low", "med", "medium", "high", "min", "max",
    "rotation", "rotate", "rotating", "no", "swing", "withswing", "mix", "mixed", "directoff",
    "small", "delay", "slow", "fast", "test", "again", "new",
}
# leading session prefixes: 'test_water_boiler_on' is the same physical device
# as 'water_boiler_on', recorded in a test session
PREFIX_WORDS = {"test", "tmp", "temp", "demo", "trial", "probe", "check"}
_TS_RE = re.compile(r"_\d{8}_\d{6}$")        # trailing _YYYYmmdd_HHMMSS
_LVL_RE = re.compile(r"^(lvl|level|stufe|speed|st)\d*$")


def strip_timestamp(stem: str) -> str:
    """Remove the recorder's trailing _YYYYmmdd_HHMMSS from a file stem."""
    return _TS_RE.sub("", str(stem))


def parse_family(label: str) -> str:
    """Collapse one single-device label to its appliance family.

    'standing_fan_high_no_rotation' -> 'standing_fan'
    'coffee_machine_run'            -> 'coffee_machine'
    'pv_only'                       -> 'pv'
    Trailing state/setting tokens and pure numbers are stripped; if everything
    would be stripped the original label is kept.
    """
    lab = strip_timestamp(str(label or "").strip().lower())
    toks = [t for t in lab.split("_") if t]
    while len(toks) > 1 and toks[0] in PREFIX_WORDS:
        toks.pop(0)
    while len(toks) > 1:
        t = toks[-1]
        if t in STATE_WORDS or t.isdigit() or _LVL_RE.match(t):
            toks.pop()
        else:
            break
    return "_".join(toks) if toks else (lab or "appliance")


def parse_families(label: str) -> list:
    """'pv__water_boiler_on' -> ['pv', 'water_boiler'] (order kept, deduped)."""
    lab = strip_timestamp(str(label or "").strip().lower())
    fams = []
    for part in lab.split("__"):
        f = parse_family(part)
        if f and f not in fams:
            fams.append(f)
    return fams


def is_mixed_label(label: str) -> bool:
    """True when the label names more than one simultaneous device."""
    return "__" in strip_timestamp(str(label or ""))

FEATURES_COMMON = ["P_mean", "Q_mean", "S_mean", "PF_mean", "QP_ratio",
                   "THD_I_mean", "P_std", "P_min", "P_max"]
FEATURES_HARM = ["h3", "h5", "h7", "h_centroid", "h_energy"]
FEATURES_FULL = FEATURES_COMMON + FEATURES_HARM

# aggregate (disaggregation) feature set. ORDER MATTERS and new features are
# only ever APPENDED: a model bundle stores the feature list it was trained
# with, and inference slices the freshly built matrix to that length, so old
# models keep working after the set grows.
# Pstep_max / Qstep_at_Pstep / n_steps are EVENT features: the largest settled
# power step inside the window, the reactive step at the same instant, and how
# many steps occurred. Steady-state sums cannot tell "boiler + lamp" from
# "boiler drawing more", but the switch-on step identifies the joining device.
AGG_FEATURES = ["Ptot_mean", "Ptot_std", "Ptot_min", "Ptot_max", "Qtot_mean",
                "PL1_mean", "PL2_mean", "PL3_mean", "PF_mean", "THDI_mean", "hour",
                "Qtot_std", "QP_ratio", "Stot_mean",
                "Pstep_max", "Qstep_at_Pstep", "n_steps",
                # per-order harmonic content of the aggregate current (window
                # means of the per-sample series from harm_series; zero when
                # the source carries no spectrum). Appended per the rule above.
                "h3_mean", "h5_mean", "h7_mean", "h_centroid_mean", "h_energy_mean"]


def slice_features(X, bundle_features):
    """Trim the feature matrix to what the (possibly older) model was trained on."""
    k = len(bundle_features) if bundle_features else X.shape[1]
    return X[:, :k] if X.shape[1] > k else X


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
        self.harm_ts = kw.get("harm_ts")       # (T,5) precomputed harm_series
                                               # (live buffer: no raw spectrum)
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
def scan_canon(files) -> list:
    """Appliance vocabulary = every family that appears in the ground truth of
    the given scenario files. Cheap pre-pass (reads only the name list), so the
    model output adapts to whatever devices the corpus contains -- this is what
    lets newly taught devices enter the model on the next retrain."""
    bases = set()
    for f in files:
        if not str(f).lower().endswith(".h5"):
            continue
        try:
            with h5py.File(f, "r") as h:
                if "ground_truth" in h and "appliance_names" in h["ground_truth"]:
                    for nm in h["ground_truth"]["appliance_names"][:]:
                        nm = nm.decode() if isinstance(nm, (bytes, bytearray)) else str(nm)
                        bases.add(nm.rsplit("_", 1)[0])
        except OSError:
            continue
    return sorted(bases)


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

        def chan(*names):
            """First present channel among `names`, cleaned; None if none exist."""
            for nm in names:
                if nm in m:
                    return _clean(m[nm][:])
            return None

        def phase_power():                                  # (T,3) or None
            if all(f"P_L{i}" in m for i in (1, 2, 3)):
                return np.column_stack([_clean(m[f"P_L{i}"][:]) for i in (1, 2, 3)])
            return None

        def harm_I():
            for nm in ("harmonics/I_mag_L1", "harmonics_I_mag"):
                if nm in m:
                    return m[nm][:]
            return None

        # Power is 'P'/'Q' in synthetic single files and 'P_total'/'Q_total' in the
        # scenario channel layout (synthetic scenarios AND the live PAC4200 recorder).
        P = chan("P", "P_total")
        Q = chan("Q", "Q_total")
        S = chan("S_total")
        if S is None:
            S = np.hypot(P, Q)
        PF = chan("PF_total")
        if PF is None:
            PF = np.divide(P, S, out=np.ones_like(P), where=S > 1e-6)
        # THD of CURRENT is optional. Synthetic scenarios store it as THD_I_L1; the
        # real PAC4200 does not expose current-THD by default, so fall back to NaN
        # (window_features then derives THD from the harmonic spectrum instead).
        THD = chan("THD_I_L1")
        if THD is None:
            THD = np.full_like(P, np.nan)

        # A file is a DISAGGREGATION SCENARIO only if it actually carries per-
        # appliance ground truth. The PAC4200 recorder writes single appliances in
        # this same channel layout (P_total, P_L1..3, ...) but with NO ground truth,
        # so the presence of 'P_total' alone must NOT route here (that was the bug
        # that raised KeyError: 'THD_I_L1' on real single-appliance recordings).
        if "ground_truth" in f:
            gtn = [_attr(x) for x in f["ground_truth"]["appliance_names"][:]]
            gtP = f["ground_truth"]["P_contribution"][:].astype(np.float64)
            return Signal(source="h5_scenario", name=_stem(path), sample_rate_hz=sr,
                          t=t_s, P=P, Q=Q, S=S, PF=PF, THD_I=THD,
                          P_phase=phase_power(), harm_I=harm_I(),
                          gt_names=gtn, gt_P=gtP)

        # Otherwise it is a LABELLED SINGLE APPLIANCE. The label lives in
        # metadata.appliance_label (PAC4200 recorder) or the nested appliance_metadata
        # JSON (synthetic single files); identify uses this as the class label.
        name = meta.get("appliance_label") or ""
        if not name:
            try:
                name = json.loads(meta.get("appliance_metadata", "{}")).get("name", "")
            except Exception:
                name = ""
        name = name or "unknown"
        return Signal(source="h5_single", name=name, label=name, sample_rate_hz=sr,
                      t=t_s, P=P, Q=Q, S=S, PF=PF, THD_I=THD,
                      P_phase=phase_power(), harm_I=harm_I())


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


def harm_series(harm_I):
    """Per-sample harmonic feature series (T,5): [h3, h5, h7, h_centroid,
    h_energy] from a per-order current spectrum (T,39; orders 2..40).

    Same quantities as _harm_feats, but evaluated per sample instead of per
    window: aggregate_windows averages the series over each window, and the
    live monitor computes the identical five scalars on every meter poll
    (before the raw spectrum is dropped from its ring buffer), so training
    and live inference stay numerically equal."""
    harm = np.atleast_2d(np.asarray(harm_I, float))
    orders = np.arange(2, 2 + harm.shape[1])
    energy = np.sqrt(np.nansum(harm ** 2, axis=1))
    centroid = np.nansum(orders * harm, axis=1) / (np.nansum(harm, axis=1) + 1e-9)
    return np.column_stack([harm[:, 1], harm[:, 3], harm[:, 5], centroid, energy])


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
def gt_bases(sig: Signal) -> list:
    """Appliance families present in a scenario's ground truth (sorted, unique)."""
    if sig.gt_names is None:
        return []
    return sorted({nm.rsplit("_", 1)[0] for nm in sig.gt_names})


def _gt_matrix(sig, Wfn, n_rows, canon):
    """Window-mean per-appliance power mapped onto `canon` columns."""
    Y = np.zeros((n_rows, len(canon)))
    for col, nm in enumerate(sig.gt_names):
        base = nm.rsplit("_", 1)[0]
        if base in canon:
            Y[:, canon.index(base)] += np.nanmean(Wfn(sig.gt_P[:, col]), 1)
    return Y


def _step_features(Pw, Qw, step_min_W=10.0):
    """Per-window event features from the windowed P/Q matrices (N, w):
    the largest settled step (pre/post medians around the sharpest sample-to-
    sample change), the Q step at that same instant, and the step count."""
    N, w = Pw.shape
    dP_max = np.zeros(N)
    dQ_at = np.zeros(N)
    n_steps = np.zeros(N)
    if w < 4:
        return dP_max, dQ_at, n_steps
    P = np.nan_to_num(Pw)
    Q = np.nan_to_num(Qw)
    d = np.diff(P, axis=1)                         # (N, w-1)
    j = np.argmax(np.abs(d), axis=1)
    k = max(1, min(5, w // 4))
    for i in range(N):
        jj = int(j[i])
        pre = slice(max(0, jj - k + 1), jj + 1)
        post = slice(jj + 1, min(w, jj + 1 + k))
        dP_max[i] = np.median(P[i, post]) - np.median(P[i, pre])
        dQ_at[i] = np.median(Q[i, post]) - np.median(Q[i, pre])
    n_steps = (np.abs(d) > step_min_W).sum(axis=1).astype(float)
    return dP_max, dQ_at, n_steps


def aggregate_windows(sig: Signal, window_s=30.0, canon=None):
    """Return (X, Y, appliance_names). Y is None if the file has no ground truth.

    `canon` is the appliance vocabulary that sizes/orders the target columns;
    it defaults to the legacy CANON list. train.py passes the vocabulary it
    derived from the training data, infer/live pass the one stored in the
    model bundle.
    """
    canon = list(canon) if canon is not None else list(CANON)
    sr = sig.sample_rate_hz
    w = max(1, int(round(window_s * sr)))
    n = (sig.n // w) * w
    def W(a):
        return np.asarray(a[:n], float).reshape(-1, w)
    hour = ((sig.t - sig.t[0]) / 3600.0) % 24
    Pph = sig.P_phase if sig.P_phase is not None else np.zeros((sig.n, 3))
    # harmonic feature series: precomputed (live buffer) > raw spectrum (h5)
    # > NaN (no spectrum; the columns become 0 via nan_to_num below)
    if sig.harm_ts is not None and len(sig.harm_ts) == sig.n:
        H = np.asarray(sig.harm_ts, float)
    elif sig.harm_I is not None and len(sig.harm_I) == sig.n:
        H = harm_series(sig.harm_I)
    else:
        H = np.full((sig.n, 5), np.nan)
    import warnings
    with warnings.catch_warnings():
        # all-NaN windows (e.g. PF or THD_I of an idle phase) are legitimate
        # here; they become 0 via nan_to_num below
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Pm = np.nanmean(W(sig.P), 1)
        Qm = np.nanmean(W(sig.Q), 1)
        dP_max, dQ_at, n_steps = _step_features(W(sig.P), W(sig.Q))
        X = np.column_stack([
            Pm, np.nanstd(W(sig.P), 1), np.nanmin(W(sig.P), 1), np.nanmax(W(sig.P), 1),
            Qm,
            np.nanmean(W(Pph[:, 0]), 1), np.nanmean(W(Pph[:, 1]), 1), np.nanmean(W(Pph[:, 2]), 1),
            np.nanmean(W(sig.PF), 1), np.nanmean(W(sig.THD_I), 1), np.nanmean(W(hour), 1),
            np.nanstd(W(sig.Q), 1), Qm / (np.abs(Pm) + 1e-6), np.hypot(Pm, Qm),
            dP_max, dQ_at, n_steps,
            np.nanmean(W(H[:, 0]), 1), np.nanmean(W(H[:, 1]), 1),
            np.nanmean(W(H[:, 2]), 1), np.nanmean(W(H[:, 3]), 1),
            np.nanmean(W(H[:, 4]), 1),
        ])
    X = np.nan_to_num(X)
    Y = None
    if sig.gt_P is not None:
        Y = _gt_matrix(sig, W, X.shape[0], canon)
    return X, Y, canon


def aggregate_presence(sig, window_s=30.0, on_W=15.0, canon=None):
    """Multi-hot 'appliance active' targets per aggregate window (|power| > on_W).

    Reuses aggregate_windows; returns (X, Y_multihot, names). Y is None when the
    file has no ground truth.
    """
    X, Yp, names = aggregate_windows(sig, window_s, canon=canon)
    Y = None if Yp is None else (np.abs(Yp) > on_W).astype(int)
    return X, Y, names


def aggregate_sequences(sig, window_s=30.0, on_W=15.0, canon=None):
    """Flatten the raw P/Q/THD waveform of each (non-overlapping) window.

    Returns (X_flat, Y_power, Y_presence, names):
      X_flat     (N, 3*W)   raw [P, Q, THD_I] samples per window (the 'sequence')
      Y_power    (N, n_app)  window-mean per-appliance power   (None if no GT)
      Y_presence (N, n_app)  1 if |power| > on_W else 0        (None if no GT)

    Used by the neural-network (MLP) path, which learns from the waveform shape
    itself rather than the hand-crafted summary features.
    """
    canon = list(canon) if canon is not None else list(CANON)
    sr = sig.sample_rate_hz
    w = max(1, int(round(window_s * sr)))
    n = (sig.n // w) * w

    def Wn(a):
        return np.asarray(a[:n], float).reshape(-1, w)

    chans = [np.nan_to_num(Wn(sig.P)), np.nan_to_num(Wn(sig.Q)), np.nan_to_num(Wn(sig.THD_I))]
    Xf = np.concatenate(chans, axis=1).astype(np.float32)        # (N, 3*W)
    Ypow = Ypres = None
    if sig.gt_P is not None:
        Ypow = _gt_matrix(sig, Wn, Xf.shape[0], canon)
        Ypres = (np.abs(Ypow) > on_W).astype(int)
    return Xf, Ypow, Ypres, canon


# --------------------------------------------------------------------------
# Multi-label F1 that is honest about absent classes
# --------------------------------------------------------------------------
def presence_f1(Y_true, Y_pred, names):
    """Per-appliance F1 + macro over the appliances that actually occur.

    An appliance with no ON window in the evaluation data AND no ON prediction
    is scored None (nothing to detect, nothing falsely detected) instead of 0,
    so it does not drag the macro average down. Returns (per_dict, macro,
    support_dict); support = number of truly-ON windows per appliance.
    """
    from sklearn.metrics import f1_score
    Y_true = np.asarray(Y_true); Y_pred = np.asarray(Y_pred)
    per, support, scored = {}, {}, []
    for i, nm in enumerate(names):
        support[nm] = int(Y_true[:, i].sum())
        if Y_true[:, i].any() or Y_pred[:, i].any():
            v = float(f1_score(Y_true[:, i], Y_pred[:, i], zero_division=0))
            per[nm] = v; scored.append(v)
        else:
            per[nm] = None
    macro = float(np.mean(scored)) if scored else 0.0
    return per, macro, support


# --------------------------------------------------------------------------
# Presence probabilities (works around MultiOutputClassifier quirks)
# --------------------------------------------------------------------------
def presence_proba(model, X):
    """P(on) per appliance from a MultiOutputClassifier, shape (n, n_appliances).

    Handles single-class outputs (an appliance that is always-on or always-off
    in the training data has only one class; predict_proba then has one column).
    """
    n = X.shape[0]
    try:
        per_out = model.predict_proba(X)          # list of (n, n_classes)
    except AttributeError:
        pred = np.asarray(model.predict(X), float)
        return pred if pred.ndim == 2 else pred.reshape(n, -1)
    cols = []
    for est, p in zip(model.estimators_, per_out):
        classes = list(getattr(est, "classes_", [0, 1]))
        p = np.asarray(p)
        cols.append(p[:, classes.index(1)] if 1 in classes else np.zeros(n))
    return np.column_stack(cols)
