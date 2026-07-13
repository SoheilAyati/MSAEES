#!/usr/bin/env python3
"""
live.py  --  LIVE NILM monitor: connect a device and recognize it as it runs
=============================================================================

Connects to the PAC4200 (same Modbus reader as Scripts/PAC4200_reader), runs
the trained mix model (presence + disaggregation) on a sliding window of the
live signal, and serves a dashboard that answers, continuously:

    *  WHICH devices are ON right now, with estimated WATTS and CONFIDENCE
    *  WHEN each device switched on/off  (event log with exact timestamps)
    *  HOW SURE the system is (model held-out accuracy + live explained-power)

and closes the loop when it does NOT know the answer:

    *  sustained unexplained power  ->  "Unknown device (~180 W) since 14:32:05
       -- what is this?"  ->  you type a name and pick ONE of two teach flows:
         - guided/ISOLATED: disconnect everything, record only the new device
           (cleanest data), or
         - IN-MIX ("teach on the go"): every other device keeps running; only
           the unknown device is toggled off -> baseline -> on -> recorded ->
           off -> closing baseline, and its own signal is isolated by
           baseline subtraction before being saved.
       Either way the captured signature is saved as a labelled recording ->
       scenarios are rebuilt and the model is RETRAINED in the background ->
       hot-reloaded. Training on the go: the next time that device runs, the
       system knows it.

Usage
-----
    # no hardware (exercise the whole loop with the simulated meter):
    python live.py --simulate

    # no hardware, REAL data: replay a pre-measured file through the exact
    # same pipeline, at its recorded rate (.h5 recording/scenario or csv):
    python live.py --replay ../PAC4200_reader/recordings/<file>.h5
    python live.py --replay ../../Pre_Measured/pac4200_toaster_200ms.csv

    # real meter:
    python live.py --host 192.168.168.1

    # real meter without per-order harmonics (if file numbers are unverified):
    python live.py --host 192.168.168.1 --no-harmonics

Options of note: --stride (how often to re-evaluate, s), --models-dir,
--recordings-dir (where taught devices are saved), --web-port.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
READER_DIR = os.path.join(REPO, "Scripts", "PAC4200_reader")
AGG_DIR = os.path.join(REPO, "Scripts", "Aggregator")
sys.path.insert(0, HERE)
sys.path.insert(0, READER_DIR)

import joblib
import h5py
from flask import Flask, jsonify, request, Response

import nilm_pipeline as nl
import pac_reader as pr          # reuse the verified Modbus reader + acquisition

PY = sys.executable
MIX_SCRIPT = os.path.join(AGG_DIR, "mix_measured_scenarios.py")
TRAIN_SCRIPT = os.path.join(HERE, "train.py")


# =============================================================================
# THD-computing reader wrapper
# =============================================================================
class ThdReader(pr.BaseReader):
    """Wraps the real/simulated reader and derives THD_I per phase from the
    per-order harmonic magnitudes of each sample (same formula the aggregator
    uses for scenario files: 100 * sqrt(sum h^2) / I_fundamental), so the live
    feature vector matches what the model saw in training."""

    def __init__(self, inner):
        self.inner = inner
        self.extra_channels = dict(getattr(inner, "extra_channels", {}) or {})
        if getattr(inner, "read_harmonics", False):
            for ph in ("L1", "L2", "L3"):
                self.extra_channels.setdefault(f"THD_I_{ph}", 0)

    # -- delegation -----------------------------------------------------------
    def connect(self):    self.inner.connect()
    def disconnect(self): self.inner.disconnect()
    def read_raw(self, address, count): return self.inner.read_raw(address, count)

    @property
    def is_simulated(self): return self.inner.is_simulated

    @property
    def swap(self): return getattr(self.inner, "swap", pr.SwapMode.BIG_WORD_FIRST)

    @swap.setter
    def swap(self, v):
        if hasattr(self.inner, "swap"):
            self.inner.swap = v

    @property
    def host(self): return getattr(self.inner, "host", "simulated")

    @property
    def port(self): return getattr(self.inner, "port", None)

    # -- sample ---------------------------------------------------------------
    def read_sample(self):
        s = self.inner.read_sample()
        if s is None:
            return None
        if getattr(self.inner, "read_harmonics", False):
            for ph in ("L1", "L2", "L3"):
                # a genuine THD-R I register value (probed at connect) wins
                # over the spectrum-derived estimate
                have = s.scalars.get(f"THD_I_{ph}")
                if have is not None and math.isfinite(have):
                    continue
                mag = s.h_I_mag.get(ph)
                thd = float("nan")
                if mag is not None and mag.size:
                    P = s.scalars.get(f"P_{ph}", float("nan"))
                    Q = s.scalars.get(f"Q_{ph}", float("nan"))
                    V = s.scalars.get(f"V_{ph}", float("nan"))
                    if math.isfinite(P) and math.isfinite(Q) and math.isfinite(V) and V > 1.0:
                        i_fund = math.hypot(P, Q) / V
                        if i_fund > 1e-3:
                            thd = 100.0 * float(np.sqrt(np.nansum(mag ** 2))) / i_fund
                s.scalars[f"THD_I_{ph}"] = thd
            # collapse the L1 spectrum into the five per-sample harmonic
            # features the mix model uses (nl.harm_series = the exact math
            # training applies to scenario spectra); the raw 39-order array
            # never enters the acquisition ring buffer, these scalars do
            mag = s.h_I_mag.get("L1")
            if mag is not None and mag.size:
                h3, h5, h7, hc, he = nl.harm_series(mag)[0]
                s.scalars.update(H3_I_L1=float(h3), H5_I_L1=float(h5),
                                 H7_I_L1=float(h7), HC_I_L1=float(hc),
                                 HE_I_L1=float(he))
        return s


# =============================================================================
# Replay reader -- pre-measured file in place of the meter
# =============================================================================
class ReplayReader(pr.BaseReader):
    """Plays a pre-measured file (.h5 recording/scenario or PAC4200 .csv --
    anything nilm_pipeline.load_signal accepts) through the live pipeline as if
    the meter were connected: every poll returns the next recorded sample,
    stamped with the current wall time. At the file's own sample rate the
    engine, event log, and teach loop behave exactly as with hardware."""

    def __init__(self, path: str, loop: bool = False):
        self.path = os.path.abspath(path)
        self.loop = loop
        self.finished = False
        self.read_harmonics = False           # THD_I arrives as a scalar channel
        self.extra_channels = {"THD_I_L1": 0}
        self._i = 0

        sig = nl.load_signal(self.path)
        self.sample_rate_hz = float(sig.sample_rate_hz) if sig.sample_rate_hz > 0 else 5.0
        self.label = str(sig.label or sig.name or os.path.basename(self.path))
        self._P = np.asarray(sig.P, float)
        self._Q = np.asarray(sig.Q, float)
        self._S = np.asarray(sig.S, float)
        self._PF = np.asarray(sig.PF, float)
        if sig.P_phase is not None:
            self._Pph = np.asarray(sig.P_phase, float)
        else:                                  # single-phase source (csv): all on L1
            z = np.zeros_like(self._P)
            self._Pph = np.column_stack([self._P, z, z])
        thd = np.asarray(sig.THD_I, float)
        if not np.isfinite(thd).any() and sig.harm_I is not None:
            thd = self._thd_from_harmonics(sig)
        self._thd = thd
        self._n = len(self._P)
        # per-sample harmonic features (as ThdReader emits live), so the mix
        # model sees the same channels during a replay as on the real meter
        self._hf = (nl.harm_series(sig.harm_I)
                    if sig.harm_I is not None and len(sig.harm_I) == self._n
                    else None)
        if self._n < 2:
            raise ValueError(f"{self.path}: too few samples to replay")
        self.last_index = 0              # most recently emitted sample index
        self.gt = self._load_ground_truth()

    def _thd_from_harmonics(self, sig) -> np.ndarray:
        """THD_I_L1 from the per-order spectrum, same formula as ThdReader /
        the aggregator (100 * sqrt(sum h^2) / I_fundamental). V_L1 is read from
        the file when recorded; otherwise nominal 230 V."""
        harm = np.asarray(sig.harm_I, float)
        if harm.shape[0] != len(self._P):
            return np.full(len(self._P), np.nan)
        V = None
        if self.path.lower().endswith(".h5"):
            try:
                with h5py.File(self.path, "r") as f:
                    if "measurements/V_L1" in f:
                        V = np.asarray(f["measurements/V_L1"][:], float)
            except OSError:
                V = None
        if V is None or len(V) != len(self._P):
            V = np.full(len(self._P), 230.0)
        i_fund = np.hypot(np.nan_to_num(self._P), np.nan_to_num(self._Q)) \
            / np.maximum(np.nan_to_num(V, nan=230.0), 1.0)
        energy = np.sqrt(np.nansum(harm ** 2, axis=1))
        return np.where(i_fund > 1e-3, 100.0 * energy / np.maximum(i_fund, 1e-9),
                        np.nan)

    def _load_ground_truth(self):
        """Ground truth carried by the replayed file, if any.

        * scenario .h5 (aggregator / mix_measured_scenarios output): the file's
          /ground_truth gives per-family ON state and watts for EVERY sample
          -> mode 'full' (chart overlay + presence/power scoring)
        * PAC4200 recording .h5: metadata.appliance_label names which device
          families were physically connected (mixed 'a__b' labels included)
          -> mode 'label' (expected-set comparison only)
        * anything else (csv, no metadata): None - plain live view.
        """
        if not self.path.lower().endswith(".h5"):
            return None
        try:
            with h5py.File(self.path, "r") as f:
                if ("ground_truth/P_contribution" in f
                        and "ground_truth/appliance_names" in f):
                    names = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                             for x in f["ground_truth/appliance_names"][:]]
                    Pc = np.nan_to_num(np.asarray(
                        f["ground_truth/P_contribution"][:], dtype=np.float64))
                    st = (f["ground_truth/state"][:]
                          if "ground_truth/state" in f else None)
                    if Pc.shape == (self._n, len(names)):
                        # collapse '<family>_<instance>' columns to families
                        fams: dict = {}
                        for k, nm in enumerate(names):
                            fam = nl.parse_family(nm)
                            d = fams.setdefault(fam, {
                                "W": np.zeros(self._n),
                                "on": np.zeros(self._n, dtype=bool)})
                            d["W"] = d["W"] + Pc[:, k]
                            d["on"] = d["on"] | (
                                (st[:, k] == b"on") if st is not None
                                else (np.abs(Pc[:, k]) > 3.0))
                        return {"mode": "full", "families": fams}
                lab = ""
                if "metadata" in f:
                    lab = f["metadata"].attrs.get("appliance_label", "")
                    lab = lab.decode() if isinstance(lab, (bytes, bytearray)) else str(lab)
                if lab:
                    fams = nl.parse_families(lab)
                    if fams:
                        return {"mode": "label", "label": lab, "expected": fams}
        except (OSError, KeyError):
            pass
        return None

    # -- reader interface ------------------------------------------------------
    def connect(self):
        self._i = 0
        self.finished = False

    def disconnect(self):
        pass

    def read_raw(self, address, count):
        return None                            # no register inspector for a file

    @property
    def is_simulated(self):
        return False

    @property
    def host(self):
        return f"replay:{os.path.basename(self.path)}"

    @property
    def port(self):
        return None

    @property
    def n_samples(self):
        return self._n

    def read_sample(self):
        i = self._i
        if i >= self._n:
            if self.loop:
                i = 0
            else:
                self.finished = True           # live.py freezes the dashboard
                i = self._n - 1                # hold the final state meanwhile
        self._i = i + 1
        self.last_index = i                    # aligns ground truth to "now"
        s = pr.Sample(timestamp_us=int(time.time() * 1e6))
        s.scalars = {
            "P_total": float(self._P[i]), "Q_total": float(self._Q[i]),
            "S_total": float(self._S[i]), "PF_total": float(self._PF[i]),
            "P_L1": float(self._Pph[i, 0]), "P_L2": float(self._Pph[i, 1]),
            "P_L3": float(self._Pph[i, 2]),
            "THD_I_L1": float(self._thd[i]),
        }
        if self._hf is not None:
            s.scalars.update(zip(("H3_I_L1", "H5_I_L1", "H7_I_L1",
                                  "HC_I_L1", "HE_I_L1"),
                                 map(float, self._hf[i])))
        return s


# =============================================================================
# Model manager (hot-reloadable) + device signature table
# =============================================================================
class ModelManager:
    """Loads the mix bundle (preferred) or the presence+disaggregate pair, and
    a per-device (P, Q) signature table from the single-appliance recordings.
    reload() picks up whatever a background retrain just wrote.

    Two bundle VARIANTS can coexist in the models dir:
      * 'latest'   -> model_mix.joblib           (what retraining overwrites;
                                                  the train-on-the-go model)
      * 'original' -> model_mix_original.joblib  (a frozen snapshot retraining
                                                  never touches)
    set_variant() switches between them at runtime; reload_seq increments on
    every (re)load so the live engine notices and rebuilds its device state."""

    def __init__(self, models_dir: str, recordings_dir: str):
        self.models_dir = models_dir
        self.recordings_dir = recordings_dir
        self.lock = threading.RLock()
        self.presence = None
        self.power = None
        self.appliances: list = []
        self.features: list = []
        self.window_s = 10.0
        self.on_W = 5.0
        self.metrics: dict = {}
        self.source = "none"
        self.loaded_utc = None
        self.variant = "latest"          # 'latest' | 'original'
        self.reload_seq = 0
        self.use_ih = False              # IH matching term, see note above
        self.signatures: list = []       # [{family, label, P, Q, IH}]
        # family -> steady OPERATING MODES, distilled from the signatures:
        # [{label, P, Q}] sorted by watts ('table_fan' -> low 10.6 W / high
        # 17.3 W). Small settled steps below the edge threshold are matched
        # against transitions BETWEEN these modes (fan turned from high to
        # low), which no on/off signature can represent.
        self.modes: dict = {}
        self.reload()

    def reload(self) -> dict:
        with self.lock:
            self._load_models()
            self._load_signatures()
            self.loaded_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.reload_seq += 1
            return self.info()

    def variants(self) -> list:
        out = ["latest"]
        if os.path.exists(os.path.join(self.models_dir, "model_mix_original.joblib")):
            out.append("original")
        return out

    def set_variant(self, variant: str) -> dict:
        if variant not in ("latest", "original"):
            raise ValueError(f"unknown model variant '{variant}'")
        if variant == "original" and "original" not in self.variants():
            raise ValueError("model_mix_original.joblib not found - train one first")
        with self.lock:
            self.variant = variant
        return self.reload()

    # every frozen snapshot and the live (retraining-overwritten) file it
    # shields; reset_to_original copies right over left
    ORIGINAL_PAIRS = [
        ("model_mix_original.joblib", "model_mix.joblib"),
        ("model_identify_original.joblib", "model_identify.joblib"),
        ("train_mix_metrics_original.json", "train_mix_metrics.json"),
    ]

    def reset_to_original(self) -> dict:
        """Erase every non-original (train-on-the-go) model: the frozen
        *_original snapshots are copied back over the live bundle names, so
        the system starts clean as if no retraining had ever happened. The
        snapshots themselves are never modified; recordings and scenarios are
        untouched (taught devices keep their signatures and re-enter the
        model on the next retrain)."""
        if "original" not in self.variants():
            raise RuntimeError("model_mix_original.joblib not found - there "
                               "is no frozen original snapshot to reset to")
        restored = []
        with self.lock:
            for src, dst in self.ORIGINAL_PAIRS:
                s = os.path.join(self.models_dir, src)
                if os.path.exists(s):
                    shutil.copyfile(s, os.path.join(self.models_dir, dst))
                    restored.append(dst)
            self.variant = "latest"      # latest now IS the original
        return {"restored": restored, "model": self.reload()}

    def _load_models(self):
        self.presence = self.power = None
        self.appliances, self.metrics, self.source = [], {}, "none"
        mix = os.path.join(self.models_dir, "model_mix.joblib")
        if self.variant == "original":
            orig = os.path.join(self.models_dir, "model_mix_original.joblib")
            if os.path.exists(orig):
                mix = orig
        if os.path.exists(mix):
            b = joblib.load(mix)
            self.presence, self.power = b["presence"], b["power"]
            self.appliances = list(b["appliances"])
            self.features = list(b.get("features", []) or [])
            self.window_s = float(b.get("window_s", 10.0))
            self.on_W = float(b.get("on_W", 5.0))
            self.metrics = b.get("metrics", {}) or {}
            self.source = os.path.basename(mix)
            return
        pres = os.path.join(self.models_dir, "model_presence.joblib")
        dis = os.path.join(self.models_dir, "model_disaggregate.joblib")
        if os.path.exists(pres):
            bp = joblib.load(pres)
            self.presence = bp["model"]
            self.appliances = list(bp["appliances"])
            self.features = list(bp.get("features", []) or [])
            self.window_s = float(bp.get("window_s", 10.0))
            self.on_W = float(bp.get("on_W", 5.0))
            self.metrics = bp.get("metrics", {}) or {}
            self.source = "model_presence.joblib"
            if os.path.exists(dis):
                bd = joblib.load(dis)
                if list(bd.get("appliances", [])) == self.appliances:
                    self.power = bd["model"]
                    self.source = "model_presence.joblib + model_disaggregate.joblib"

    def _load_signatures(self):
        """Steady-state (P, Q, harmonic current) per single-device recording,
        for edge matching. IH is the device's harmonic current in amps
        (THD_I/100 * I_fundamental at nominal 230 V); None when the recording
        carries no usable THD channel.

        Rotation/swing variants are excluded from the MATCHING table entirely
        (they still train the window model): the rotation motor only shifts a
        speed by ~2.6 W, which the plain-speed signature already covers within
        tolerance -- but standing_fan_high_rotate (33.0 W/52.1 var) is a
        near-exact twin of BOTH FANS ON LOW (34.0 W/52.8 var), and its
        presence let that composite single-match 'standing_fan' at d=0.07,
        unbeatable by any pair hypothesis."""
        ROT = {"rotate", "rotating", "rotation", "swing", "withswing"}
        sigs = []
        for p in sorted(glob.glob(os.path.join(self.recordings_dir, "*.h5"))):
            try:
                with h5py.File(p, "r") as f:
                    lab = f["metadata"].attrs.get("appliance_label", "")
                    lab = lab.decode() if isinstance(lab, (bytes, bytearray)) else str(lab)
                    if not lab or nl.is_mixed_label(lab):
                        continue
                    if ROT & set(lab.lower().split("_")):
                        continue
                    P = np.nan_to_num(f["measurements/P_total"][:])
                    Q = np.nan_to_num(f["measurements/Q_total"][:])
                    T = (np.asarray(f["measurements/THD_I_L1"][:], float)
                         if "measurements/THD_I_L1" in f else None)
            except (OSError, KeyError):
                continue
            on = np.abs(P) > 3.0
            if len(P) < 25 or not on.any():
                continue
            ih = None
            if T is not None and len(T) == len(P):
                fin = on & np.isfinite(T)
                if fin.sum() >= max(10, 0.3 * on.sum()):
                    ih = float(np.median(
                        T[fin] / 100.0 * np.hypot(P[fin], Q[fin]) / 230.0))
            sigs.append({"family": nl.parse_family(lab), "label": lab,
                         "P": float(np.median(P[on])), "Q": float(np.median(Q[on])),
                         "IH": ih})
        self.signatures = sigs
        self.modes = self._build_modes(sigs)

    @staticmethod
    def _build_modes(sigs) -> dict:
        """Distill per-family operating modes from the signature table.
        Recordings of the same physical setting (standing_fan_low and
        standing_fan_low_rotate differ by the ~2 W rotation motor) merge into
        one mode: sub-mode variance is noise here, and every extra pseudo-mode
        multiplies the transition pairs the small-step matcher must
        disambiguate. Families keep their modes sorted by watts."""
        # (rotate/swing variants never reach here -- _load_signatures already
        # filters them out of the matching table)
        by_fam: dict = {}
        for s in sigs:
            by_fam.setdefault(s["family"], []).append(s)
        modes = {}
        for fam, ss in by_fam.items():
            groups: list = []
            for s in sorted(ss, key=lambda x: x["P"]):
                g = next((g for g in groups
                          if abs(s["P"] - g["P"]) <= max(2.2, 0.10 * abs(g["P"]))
                          and abs(s["Q"] - g["Q"]) <= max(3.0, 0.20 * abs(g["Q"]))),
                         None)
                if g is None:
                    groups.append({"label": s["label"], "P": s["P"], "Q": s["Q"],
                                   "_n": 1})
                else:                     # running mean keeps the mode centred
                    g["P"] += (s["P"] - g["P"]) / (g["_n"] + 1)
                    g["Q"] += (s["Q"] - g["Q"]) / (g["_n"] + 1)
                    g["_n"] += 1
            for g in groups:
                g.pop("_n", None)
            modes[fam] = groups
        return modes

    # THD-derived harmonic current is an RSS DIFFERENCE of two estimates, so
    # for small steps its noise rivals the signal (a fan's whole IH is
    # ~0.008 A). Below this step magnitude the IH term is ignored -- P and Q
    # decide alone, exactly as they did on the branch's parent.
    IH_MIN_STEP = 60.0
    # The IH term is OFF by default (use_ih, --ih-matching): measured on the
    # 2026-07-13 choreo recordings it vetoed CORRECT matches twice -- the
    # coffee machine's harmonic current varies 5x across its brew cycle
    # (grinder burst 0.58 A vs recorded median 0.13 A -> tIH +8.9), and the
    # meter's ~2.3 % THD floor scales with fundamental current, so a clean
    # 930 W boiler 'gains' 0.13 A of phantom harmonics at switch-on
    # (tIH +0.69 degraded a 0.89 match below the claim bar). Its one win
    # (hair dryer at lamp watts) is preserved behind the flag for devices
    # with genuinely strong, stationary harmonics.

    def match_edge(self, dP: float, dQ: float, ih=None, q_tol_scale: float = 1.0,
                   q_noise: float = 0.0):
        """Nearest device signature for a power step; None when nothing is
        close. Elliptical distance with SEPARATE P / Q (and, when both sides
        have it, harmonic-current) tolerances: the old single 35 %-of-magnitude
        tolerance let any device near a known device's watts steal its name
        (a hair dryer at ~500 W matched the 501 W standing lamp) -- reactive
        power and harmonic content now have to agree too. And when two
        signatures of DIFFERENT families are nearly equally close, the
        confidence collapses so the caller reports 'unrecognized' instead of
        guessing a sibling: a wrong name is worse than an unknown."""
        with self.lock:
            cands = []
            for s in self.signatures:
                tol_P = max(15.0, 0.25 * abs(s["P"]))
                # Q floor stays TIGHT (8 var): reactive power is the only
                # feature separating a small fan (14-52 var) from a small
                # SMPS charger (~0 var) at equal watts, and the meter resolves
                # Q to a couple of var at these levels. A 20-var floor let a
                # 10 W laptop charger match table_fan_low (Q 14.4) at 0.43
                # conf. Q noise on LARGE loads is covered by the 5 %-of-P term.
                # q_tol_scale > 1 for probes whose Q is an ESTIMATE rather
                # than a measured step (residual matching); q_noise is the
                # MEASURED Q jitter around this specific step, so a gusty
                # evening on the fans widens the gate instead of turning a
                # real fan edge into an "unknown load"
                tol_Q = (max(8.0, 0.25 * abs(s["Q"]) + 0.05 * abs(s["P"]))
                         + 1.5 * max(0.0, q_noise)) * q_tol_scale
                terms = [(dP - s["P"]) / tol_P, (dQ - s["Q"]) / tol_Q]
                if (self.use_ih and ih is not None and math.isfinite(ih)
                        and s.get("IH") is not None
                        and math.hypot(dP, dQ) >= self.IH_MIN_STEP):
                    terms.append((ih - s["IH"]) / max(0.05, 0.35 * s["IH"]))
                # CONJUNCTIVE distance: every dimension must independently
                # agree. Averaging (RMS) let an uninformative dimension dilute
                # a real disagreement -- two resistive loads both sit at Q~0,
                # so the Q term said nothing yet still halved the distance,
                # and a 716 W toaster matched the 932 W boiler at 0.34 conf
                # instead of raising the unknown prompt.
                d = max(abs(t) for t in terms)
                if d < 1.0:
                    cands.append((d, s))
            if not cands:
                return None
            cands.sort(key=lambda x: x[0])
            best_d, best = cands[0]
            conf = 1.0 - best_d
            # ambiguity penalty against the nearest OTHER family (several
            # recordings of the same device must not penalize each other).
            # The penalty is floored when the step fits the winner WELL
            # (d <= 0.35): an excellent primary fit with a marginal second
            # family should read "probably X", not "unrecognized" -- full
            # collapse turned real fan edges into unknown-load claims, which
            # then vetoed the model on top (worse than the loose matcher it
            # replaced). A mediocre fit keeps the hard collapse: at that
            # point a wrong name IS worse than an unknown.
            for d2, s2 in cands[1:]:
                if s2["family"] != best["family"]:
                    factor = (d2 - best_d) / 0.35
                    if best_d <= 0.35:
                        factor = max(factor, 0.5)
                    conf *= min(1.0, factor)
                    break
            return {"family": best["family"], "label": best["label"],
                    "confidence": round(max(0.0, conf), 2),
                    "distance": round(best_d, 3)}

    def match_edge_pair(self, dP: float, dQ: float, q_noise: float = 0.0):
        """Explain one composite step as TWO devices switching together.

        Two devices flipped within the same detector window merge into a
        single step no single signature can match (both fans started at once:
        +46.6 W = table_fan 16 + standing_fan 31). Without this, that step
        became a permanent anonymous unknown load whose model veto then hid
        both devices for the rest of the session. A UNIQUE two-family sum
        within tolerance claims both; anything ambiguous stays unknown."""
        with self.lock:
            cands = []
            sigs = self.signatures
            for i in range(len(sigs)):
                for j in range(i + 1, len(sigs)):
                    a, b = sigs[i], sigs[j]
                    if a["family"] == b["family"]:
                        continue          # one family cannot be claimed twice
                    P2, Q2 = a["P"] + b["P"], a["Q"] + b["Q"]
                    tol_P = max(18.0, 0.22 * abs(P2))
                    # Q slack couples to P at only 2 % here (vs 5 % for single
                    # matches): the ~17 var gap between 'boiler+table_fan' and
                    # 'boiler+standing_fan' IS the discriminator, and a 5 %
                    # slack (47 var at a 950 W composite) blinds it
                    tol_Q = (max(10.0, 0.22 * abs(Q2) + 0.02 * abs(P2))
                             + 1.5 * max(0.0, q_noise))
                    d = max(abs(dP - P2) / tol_P, abs(dQ - Q2) / tol_Q)
                    if d < 1.0:
                        cands.append((d, a, b))
            if not cands:
                return None
            cands.sort(key=lambda x: x[0])
            d, a, b = cands[0]
            fams = {a["family"], b["family"]}
            # a DIFFERENT family pair fitting almost as well = ambiguous (the
            # same two families in other modes is fine -- watts are split
            # proportionally either way)
            for d2, a2, b2 in cands[1:]:
                if {a2["family"], b2["family"]} != fams and d2 - d < 0.30:
                    return None
            conf = max(0.0, 1.0 - d) * 0.85   # pair guesses stay humbler
            return {"members": [a, b], "confidence": round(conf, 2),
                    "distance": round(d, 3)}

    def match_mode_change(self, on_watts: dict, dP: float, dQ: float,
                          q_noise: float = 0.0):
        """Explain a SMALL settled step as a mode transition of a device that
        is already ON ('table fan turned from high to low'). `on_watts` maps
        each candidate family to its current per-device watt estimate, which
        anchors the FROM mode -- without the anchor, a -6 W step is equally
        'table_fan high->low' and 'standing_fan high->med' and the wrong fan
        gets its watts lowered (the exact complaint this solves).

        Returns {family, from, to, dP_sig, dQ_sig, confidence} or None when
        nothing fits or two different families fit about equally well."""
        with self.lock:
            cands = []
            for fam, w_cur in on_watts.items():
                for a in self.modes.get(fam, []):
                    # the device must currently BE in mode a -- and precisely
                    # so: claim watts are edge-measured to ~1 W, and a loose
                    # anchor lets a 30 W fan 'depart' from a neighbouring
                    # 28 W pseudo-mode, doubling the candidate transitions
                    if w_cur is None:
                        continue
                    tol_a = max(2.0, 0.10 * abs(a["P"]))
                    a_err = abs(w_cur - a["P"]) / tol_a
                    if a_err > 1.0:
                        continue
                    for b in self.modes.get(fam, []):
                        if b is a or abs(b["P"] - a["P"]) < 2.0:
                            continue
                        dp_sig = b["P"] - a["P"]
                        dq_sig = b["Q"] - a["Q"]
                        # tight dP floor (1.2 W): mode edges only fire on
                        # near-flat plateaus, so the step medians resolve to
                        # ~0.3 W -- and 1.2 W is exactly what separates the
                        # table fan's -5.8 W drop from the standing fan's
                        # -4.3 W high->med at the same moment
                        t_p = (dP - dp_sig) / max(1.2, 0.30 * abs(dp_sig))
                        t_q = (dQ - dq_sig) / (max(3.5, 0.35 * abs(dq_sig))
                                               + 1.5 * max(0.0, q_noise))
                        # the anchor error joins the distance (half weight):
                        # between two transitions that both fit the step, the
                        # one whose device actually SITS at the from-mode wins
                        d = max(abs(t_p), abs(t_q), 0.5 * a_err)
                        if d < 1.0:
                            cands.append((d, fam, a, b, dp_sig, dq_sig))
            if not cands:
                return None
            cands.sort(key=lambda x: x[0])
            d, fam, a, b, dp_sig, dq_sig = cands[0]
            # a transition of ANOTHER on-device that fits almost as well means
            # the step is genuinely ambiguous -> better no reassignment than
            # lowering the wrong fan
            for d2, fam2, *_ in cands[1:]:
                if fam2 != fam and d2 - d < 0.22:
                    return None
            return {"family": fam, "from": a, "to": b,
                    "dP_sig": dp_sig, "dQ_sig": dq_sig,
                    "confidence": round(max(0.0, 1.0 - d), 2)}

    def signature_report(self, label: str):
        """How separable is this (freshly taught) device from the rest of the
        vocabulary? Returns the nearest OTHER family in signature space and a
        rough conjunctive distance; None when there is nothing to compare."""
        fam = nl.parse_family(label)
        with self.lock:
            own = [s for s in self.signatures if s["family"] == fam]
            others = [s for s in self.signatures if s["family"] != fam]
        if not own or not others:
            return None
        newest = own[-1]
        worst = None
        for s in others:
            tol_P = max(15.0, 0.25 * abs(s["P"]))
            tol_Q = max(8.0, 0.25 * abs(s["Q"]) + 0.05 * abs(s["P"]))
            d = max(abs(newest["P"] - s["P"]) / tol_P,
                    abs(newest["Q"] - s["Q"]) / tol_Q)
            if worst is None or d < worst["distance"]:
                worst = {"family": s["family"], "label": s["label"],
                         "distance": round(d, 2),
                         "dP_W": round(newest["P"] - s["P"], 1),
                         "dQ_var": round(newest["Q"] - s["Q"], 1)}
        return worst

    def info(self) -> dict:
        with self.lock:
            m = self.metrics or {}
            acc = {}
            if "presence_macro_f1" in m:
                acc["presence_macro_f1"] = round(m["presence_macro_f1"], 3)
            if "macro_f1" in m:
                acc["presence_macro_f1"] = round(m["macro_f1"], 3)
            if "gated_overall_mae_W" in m:
                acc["power_mae_W"] = round(m["gated_overall_mae_W"], 1)
            elif "overall_mae_W" in m:
                acc["power_mae_W"] = round(m["overall_mae_W"], 1)
            return {"source": self.source, "appliances": self.appliances,
                    "window_s": self.window_s, "on_W": self.on_W,
                    "has_power_head": self.power is not None,
                    "holdout_accuracy": acc, "trained_utc": m.get("trained_utc"),
                    "n_signatures": len(self.signatures),
                    "variant": self.variant, "variants": self.variants(),
                    "loaded_utc": self.loaded_utc}


# =============================================================================
# Retrainer -- "training on the go"
# =============================================================================
class Retrainer:
    """Background rebuild of scenarios + retrain of the mix (and identify)
    models from everything in the recordings folder, then hot-reload."""

    # 64 scenarios: with 24 the random device combinations dominated model
    # quality run-to-run (set-F1 on the real mixed recordings swung by 0.1);
    # 64 stabilizes it and still retrains in well under two minutes.
    def __init__(self, models: ModelManager, scenarios_dir: str,
                 window_s: float = 10.0, on_w: float = 5.0,
                 n_scenarios: int = 64, scenario_duration: float = 300.0):
        self.models = models
        self.scenarios_dir = scenarios_dir
        self.window_s = window_s
        self.on_w = on_w
        self.n_scenarios = n_scenarios
        self.scenario_duration = scenario_duration
        self.lock = threading.RLock()
        self.state = "idle"              # idle | running | done | error
        self.step = ""
        self.log_tail = ""
        self.started = None
        self.finished = None
        self.runs = 0

    def start(self) -> bool:
        with self.lock:
            if self.state == "running":
                return False
            self.state, self.step = "running", "starting"
            self.started, self.finished = time.time(), None
        threading.Thread(target=self._run, daemon=True, name="retrain").start()
        return True

    def _sh(self, step, cmd) -> bool:
        with self.lock:
            self.step = step
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2500:]
        with self.lock:
            self.log_tail = tail
        return r.returncode == 0

    def _run(self):
        try:
            seed = int(time.time()) % 100000
            ok = self._sh("mixing recordings into scenarios", [
                PY, MIX_SCRIPT,
                "--recordings", self.models.recordings_dir,
                "--out", self.scenarios_dir,
                "--n-scenarios", str(self.n_scenarios),
                "--duration", str(self.scenario_duration),
                "--min-app", "1", "--max-app", "4", "--seed", str(seed)])
            if ok:
                ok = self._sh("training mix model (presence + power)", [
                    PY, TRAIN_SCRIPT, "--task", "mix",
                    "--data", os.path.join(self.scenarios_dir, "measured_scenario_*.h5"),
                    "--window", str(self.window_s), "--on-w", str(self.on_w),
                    "--out", self.models.models_dir])
            if ok:
                # identify refresh is best-effort; ignore its exit code.
                # non-recursive glob: retired recordings in old/ stay out,
                # matching what the scenario mixer trains on
                self._sh("training identify model", [
                    PY, TRAIN_SCRIPT, "--task", "identify",
                    "--data", os.path.join(self.models.recordings_dir, "*.h5"),
                    "--window", "10", "--stride", "5",
                    "--out", self.models.models_dir])
            with self.lock:
                self.state = "done" if ok else "error"
                self.step = "reloading model" if ok else self.step
                self.finished = time.time()
                self.runs += 1
            if ok:
                self.models.reload()
                with self.lock:
                    self.step = "model reloaded"
        except Exception as e:            # noqa: BLE001
            with self.lock:
                self.state, self.step = "error", f"exception: {e}"
                self.finished = time.time()

    def status(self) -> dict:
        with self.lock:
            return {"state": self.state, "step": self.step, "runs": self.runs,
                    "elapsed_s": round(time.time() - self.started, 1) if self.started and not self.finished else None,
                    "took_s": round(self.finished - self.started, 1) if self.started and self.finished else None,
                    "log_tail": self.log_tail}


# =============================================================================
# In-mix teach capture buffer
# =============================================================================
class MixCapture:
    """Accumulates live-buffer samples across an in-mix teach session.

    The acquisition ring buffer only holds ~5 minutes; a slow user could
    overrun it, so every wait/record loop of the in-mix flow calls collect()
    to drain new samples into this unbounded store as they arrive."""

    KEYS = ("t_ms", "P", "Q", "S", "PF", "P1", "P2", "P3", "THD")

    def __init__(self, engine):
        self.engine = engine
        self.chunks = {k: [] for k in self.KEYS}
        self.last_t = 0

    def collect(self):
        arrs = self.engine._buffer_arrays()
        if arrs is None:
            return
        sel = arrs["t_ms"] > self.last_t
        if not sel.any():
            return
        for k in self.KEYS:
            self.chunks[k].append(np.asarray(arrs[k])[sel])
        self.last_t = int(arrs["t_ms"][sel][-1])

    def arrays(self) -> dict:
        return {k: (np.concatenate(self.chunks[k]) if self.chunks[k]
                    else np.array([], dtype=float))
                for k in self.KEYS}

    def median(self, key: str, t0_ms: int, t1_ms: int) -> float:
        a = self.arrays()
        sel = (a["t_ms"] >= t0_ms) & (a["t_ms"] <= t1_ms)
        return float(np.nanmedian(a[key][sel])) if sel.any() else float("nan")

    def stats(self, t0_ms: int, t1_ms: int):
        """Channel medians (plus P noise and time centre) over [t0, t1]."""
        a = self.arrays()
        if not a["t_ms"].size:
            return None
        sel = (a["t_ms"] >= t0_ms) & (a["t_ms"] <= t1_ms)
        if int(sel.sum()) < 3:
            return None
        med = lambda k: float(np.nanmedian(a[k][sel]))   # noqa: E731
        return {"P": med("P"), "Q": med("Q"), "P1": med("P1"),
                "P2": med("P2"), "P3": med("P3"), "THD": med("THD"),
                "P_std": float(np.nanstd(a["P"][sel])),
                "t_ms": float(np.mean(a["t_ms"][sel])), "n": int(sel.sum())}


# =============================================================================
# Live NILM engine
# =============================================================================
class LiveEngine:
    """Consumes the acquisition ring buffer; every `stride_s` re-evaluates the
    model on the trailing window; detects edges; tracks who-is-on-since-when,
    the unexplained residual, and the unknown-device state."""

    def __init__(self, svc: pr.AcquisitionService, models: ModelManager,
                 retrainer: Retrainer, out_dir: str, stride_s: float = 2.0,
                 unknown_min_W: float = 30.0, unknown_frac: float = 0.15,
                 unknown_persist_s: float = 8.0, edge_min_W: float = 8.0,
                 edge_claim_conf: float = 0.30, teach_record_s: float = 45.0,
                 mode_min_W: float = 3.5, big_edge_min_W: float = 120.0,
                 min_conf: float = 0.70):
        self.svc = svc
        self.models = models
        self.retrainer = retrainer
        self.stride_s = stride_s
        self.unknown_min_W = unknown_min_W
        self.unknown_frac = unknown_frac
        self.unknown_persist_s = unknown_persist_s
        self.edge_min_W = edge_min_W
        self.edge_claim_conf = edge_claim_conf
        # confidence floor for NAMING an appliance: an edge/residual single, or
        # a window-model presence vote, below this reads as an UNKNOWN load
        # instead of a low-confidence device name ("a wrong name is worse than
        # an unknown"). Composite two-device pairs and mode changes of an
        # already-identified device keep their own (differently scaled) bars.
        self.min_conf = min_conf
        self.teach_record_s = teach_record_s
        # small settled steps in [mode_min_W, edge_min_W) are probed as MODE
        # TRANSITIONS of already-on devices (fan high -> low is a -6 W step:
        # far below the on/off edge threshold, yet decisive for which fan to
        # attribute the drop to)
        self.mode_min_W = mode_min_W
        # a family whose smallest signature is above this cannot switch ON by
        # window-model vote alone: real kilowatt-class devices always announce
        # themselves with a step. This kills the phantom "coffee machine /
        # water boiler popped up while other devices ran" reports.
        self.big_edge_min_W = big_edge_min_W
        self._teach_thread: threading.Thread | None = None
        self._teach_cancel = False
        self.guide: dict | None = None   # current guided-teach instruction

        self.lock = threading.RLock()
        self.state: dict = {}            # family -> {on, prob, power_W, since_ms}
        # edge-driven device state (Hart-style event NILM). A matched on-edge
        # CLAIMS the device on with the step's own watts; a matched off-edge
        # drops the claim and force-holds the device off until the model window
        # has flushed the old samples. Claims outrank the window model: the
        # steady-state features cannot tell "boiler + lamp" from "boiler with
        # more watts", but the +501 W step at plug-in time identifies the lamp
        # uniquely.
        self.claims: dict = {}           # family -> {W, conf, t_ms}
        self.forced_off: dict = {}       # family -> hold-off-until unix_ms
        # anonymous UNKNOWN-LOAD claims: an on-edge that matches no signature
        # used to leave its watts to the window model, which then pinned them
        # on whatever known family was closest (hair dryer -> "standing lamp").
        # Now the unmatched step itself claims the watts as an unknown load:
        # model-only switch-ons are vetoed while one is active (every real
        # switch-on produces its own edge), the residual stays unexplained,
        # and the unknown-device prompt fires instead of a wrong name.
        self.unknown_claims: list = []   # [{W, t_ms}]
        self.unknown_claim_min_W = max(15.0, edge_min_W)
        self._unknown_flush_until = 0    # model-veto after an unknown off-edge
        # every settled edge this session, matched or not. After a model
        # reload (retrain / variant switch) the history is re-matched against
        # the NEW signatures and the claims rebuilt -- a software version of
        # "unplug everything and plug it back in": a step that was
        # 'unrecognized' before a device was taught resolves to that device
        # afterwards, without touching the hardware.
        self.edge_history: deque = deque(maxlen=600)   # {t_ms, dP, dQ}
        self._model_seq = models.reload_seq
        # 5 strides (~10 s) of probability smoothing: with 3, a device whose
        # probability hovers near the threshold (coffee machine in the ~500 W
        # regime reads lamp-like) flapped in and out of "currently on"
        self.smooth: deque = deque(maxlen=5)     # recent proba vectors
        self.residual_W = 0.0
        self.total_W = 0.0
        self.explained_frac = 1.0
        self.history: deque = deque(maxlen=420)  # per-stride snapshots for the chart
        self.events: list = []           # full session event log
        self.unknown: dict | None = None
        self._unknown_first: float | None = None
        self._unknown_clear_ms: float | None = None
        self._unknown_stale_ms: float | None = None
        self._last_edge_ms = 0
        self._last_edge_dP = 0.0
        self._mode_ambig_ms = 0          # throttle for mode_ambiguous events
        self._veto_log_ms: dict = {}     # family -> last model_veto event ms
        self._last_taught_family = None  # checked against F1 after retrain
        self._teach_note = ""
        # replay ground-truth comparison: available only when the reader
        # replays a file that carries ground truth (scenario /ground_truth or
        # a labelled recording); None on real/simulated meters
        inner = getattr(svc.reader, "inner", svc.reader)
        self._gt = getattr(inner, "gt", None)
        self._gt_reader = inner
        self.gt_stats: dict = {}         # family -> confusion counts + power err
        self._gt_set_stats = {"tp": 0, "fp": 0, "fn": 0, "n": 0}
        self._gt_now: dict | None = None

        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.events_csv = os.path.join(out_dir, "events.csv")
        with open(self.events_csv, "w", newline="") as fh:
            csv.writer(fh).writerow(
                ["time_iso", "unix_ms", "kind", "device", "confidence",
                 "dP_W", "dQ_var", "P_total_W", "detail"])

        self._run_flag = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----------------------------------------------------------
    def start(self):
        self._run_flag.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="live-engine")
        self._thread.start()

    def stop(self):
        self._run_flag.clear()

    # ---- helpers ------------------------------------------------------------
    def _buffer_arrays(self):
        """Snapshot the acquisition ring buffer as channel arrays."""
        with self.svc._lock:
            buf = list(self.svc._buffer)
        if not buf:
            return None
        t_ms = np.array([b[0] for b in buf], dtype=np.int64)

        def ch(name):
            return np.array([float(b[1].get(name, np.nan)) for b in buf])

        thd = ch("THD_I_L1")
        return {"t_ms": t_ms, "P": ch("P_total"), "Q": ch("Q_total"),
                "S": ch("S_total"), "PF": ch("PF_total"),
                "P1": ch("P_L1"), "P2": ch("P_L2"), "P3": ch("P_L3"),
                "THD": thd,
                "H3": ch("H3_I_L1"), "H5": ch("H5_I_L1"), "H7": ch("H7_I_L1"),
                "HC": ch("HC_I_L1"), "HE": ch("HE_I_L1")}

    def _window_signal(self, arrs, w_samples):
        """nl.Signal over the trailing window (exact model feature path)."""
        n = len(arrs["t_ms"])
        if n < w_samples:
            return None
        sl = slice(n - w_samples, n)
        t = (arrs["t_ms"][sl] - arrs["t_ms"][sl][0]) / 1000.0
        P, Q = arrs["P"][sl], arrs["Q"][sl]
        S = arrs["S"][sl]
        S = np.where(np.isfinite(S), S, np.hypot(P, Q))
        PF = arrs["PF"][sl]
        Pph = np.column_stack([arrs["P1"][sl], arrs["P2"][sl], arrs["P3"][sl]])
        harm_ts = np.column_stack([arrs[k][sl] for k in ("H3", "H5", "H7", "HC", "HE")])
        return nl.Signal(source="live", name="live", sample_rate_hz=self.svc.sample_rate_hz,
                         t=t, P=P, Q=Q, S=S, PF=PF, THD_I=arrs["THD"][sl], P_phase=Pph,
                         harm_ts=harm_ts)

    def _log_event(self, t_ms, kind, device, conf, dP, dQ, p_total, detail=""):
        ev = {"time_iso": datetime.fromtimestamp(t_ms / 1000.0).astimezone().isoformat(timespec="milliseconds"),
              "unix_ms": int(t_ms), "kind": kind, "device": device,
              "confidence": None if conf is None else round(float(conf), 2),
              "dP_W": None if dP is None else round(float(dP), 1),
              "dQ_var": None if dQ is None else round(float(dQ), 1),
              "P_total_W": None if p_total is None else round(float(p_total), 1),
              "detail": detail}
        with self.lock:
            self.events.append(ev)
            self.events = self.events[-400:]
        with open(self.events_csv, "a", newline="") as fh:
            csv.writer(fh).writerow([ev["time_iso"], ev["unix_ms"], kind, device,
                                     ev["confidence"], ev["dP_W"], ev["dQ_var"],
                                     ev["P_total_W"], detail])

    # ---- edge detection -----------------------------------------------------
    def _detect_edge(self, arrs):
        """Settled-before vs settled-after step detector on P_total; returns the
        edge (exact sample timestamp) or None. Runs on every stride.

        Two tiers: a FULL edge (>= edge_min_W) drives the on/off claim
        machinery; a small MODE edge (>= mode_min_W but below the full
        threshold, with stricter settling) is only ever interpreted as a mode
        transition of a device that is already on -- it can never create or
        release claims or unknown loads."""
        sr = self.svc.sample_rate_hz
        need = int(8 * sr)
        P, Q, t_ms = arrs["P"], arrs["Q"], arrs["t_ms"]
        if len(P) < need:
            return None
        P8, Q8, t8 = P[-need:], Q[-need:], t_ms[-need:]
        k = int(2.5 * sr)
        pre_P, post_P = np.nanmedian(P8[:k]), np.nanmedian(P8[-k:])
        pre_Q, post_Q = np.nanmedian(Q8[:k]), np.nanmedian(Q8[-k:])
        dP, dQ = post_P - pre_P, post_Q - pre_Q
        full = abs(dP) >= self.edge_min_W or abs(dQ) >= 2 * self.edge_min_W
        if not full and abs(dP) < self.mode_min_W:
            return None
        # both sides must be settled so a ramp isn't logged sample-by-sample
        if np.nanstd(P8[:k]) > max(6.0, 0.05 * abs(pre_P)):
            return None
        if np.nanstd(P8[-k:]) > max(6.0, 0.05 * abs(post_P)):
            return None
        # a mode edge is barely above meter noise; demand near-flat plateaus
        # so P wobble cannot fabricate fan-speed changes every few strides
        if not full and (np.nanstd(P8[:k]) > max(1.5, 0.25 * abs(dP))
                         or np.nanstd(P8[-k:]) > max(1.5, 0.25 * abs(dP))):
            return None
        mid = P8[k:-k]
        if len(mid) == 0:
            return None
        j = int(np.nanargmax(np.abs(np.diff(mid)))) if len(mid) > 1 else 0
        t_edge = int(t8[k + j])
        if t_edge - self._last_edge_ms < 3000:      # debounce
            return None
        # the same physical step stays visible while it transits the 8 s
        # window (pre/post medians keep disagreeing) and can re-trigger with a
        # shifted timestamp: a near-identical dP of the same sign within the
        # transit time is the SAME event, not a new one (the lamp was logged
        # twice, 3 s apart, at 502.6 W and 501.9 W). A genuinely new step of
        # clearly different watts (boiler after lamp) still passes.
        if (t_edge - self._last_edge_ms < 9000
                and dP * self._last_edge_dP > 0
                and abs(dP - self._last_edge_dP)
                < max(10.0, 0.12 * abs(self._last_edge_dP))):
            return None
        self._last_edge_ms = t_edge
        self._last_edge_dP = float(dP)
        # harmonic-current estimate of the switching device (RSS delta of the
        # settled THD-derived harmonic currents), for the signature matcher;
        # None when THD is unavailable on either side of the step
        ih = None
        T8 = arrs["THD"][-need:]
        if (np.isfinite(T8[:k]).sum() >= max(2, k // 2)
                and np.isfinite(T8[-k:]).sum() >= max(2, k // 2)):
            ih_pre = np.nanmedian(T8[:k]) / 100.0 * math.hypot(pre_P, pre_Q) / 230.0
            ih_post = np.nanmedian(T8[-k:]) / 100.0 * math.hypot(post_P, post_Q) / 230.0
            d2 = (ih_post ** 2 - ih_pre ** 2) if dP > 0 else (ih_pre ** 2 - ih_post ** 2)
            if math.isfinite(d2):
                ih = math.sqrt(max(0.0, d2))
        # measured Q jitter around this step: the matcher widens its Q gate by
        # it, so real-world reactive noise cannot veto a correct match
        q_noise = float(max(np.nanstd(Q8[:k]), np.nanstd(Q8[-k:])))
        if not math.isfinite(q_noise):
            q_noise = 0.0
        return {"t_ms": t_edge, "dP": float(dP), "dQ": float(dQ),
                "P_after": float(post_P), "ih": ih,
                "kind": "full" if full else "mode", "q_noise": q_noise}

    # ---- full edge -> match (single, then pair) -> device state -------------
    def _handle_full_edge(self, edge, record=True):
        """Resolve a full (on/off) edge against the signature table and apply
        it: single-device match first, then a two-device COMPOSITE match (both
        fans started within one detector window), then the unknown-load path.
        `record=False` replays history after a model reload: state changes
        happen, but nothing is re-logged or re-appended."""
        m = self.models
        direction = "on" if edge["dP"] > 0 else "off"
        # an on-edge is the device's own (P, Q); an off-edge is its negative
        probe = (edge["dP"], edge["dQ"]) if direction == "on" \
            else (-edge["dP"], -edge["dQ"])
        match = (m.match_edge(*probe, ih=edge.get("ih"),
                              q_noise=edge.get("q_noise", 0.0))
                 if abs(edge["dP"]) >= self.edge_min_W else None)
        # confidence floor: a matched single appliance is only NAMED when it
        # clears self.min_conf (default 0.70). Below it -- or with no match at
        # all -- the step is reported as an unknown load, never a low-confidence
        # guess ("a wrong name is worse than an unknown").
        if match and match["confidence"] >= self.min_conf:
            dev, conf = match["family"], match["confidence"]
        else:
            dev, conf = "unrecognized", None
        # composite check for EVERY sizeable on-step, decided by raw DISTANCE:
        # whichever hypothesis explains the step more tightly wins. Gating the
        # pair on a weak single match missed the nastiest case -- both fans on
        # LOW (34 W / 53 var) is a near-twin of standing_fan HIGH alone
        # (30.4 W / 52 var), which single-matched at 0.76 and swallowed the
        # table fan. Distance compares the two stories directly: the exact
        # low+low sum (d~0.03) beats the 3.6 W-off single (d~0.24). A clear
        # margin (0.15) protects the reverse case: a lone standing_fan start
        # fits its own signature far tighter than any two-device sum.
        if direction == "on" and abs(edge["dP"]) >= 25.0:
            pair = m.match_edge_pair(*probe, q_noise=edge.get("q_noise", 0.0))
            # the pair still has to beat the best SINGLE fit on raw distance;
            # key it off the match's own confidence (>= 0.25), NOT the min_conf
            # naming decision, so raising min_conf can never turn a mediocre
            # single into a fabricated two-device claim.
            single_d = (match["distance"]
                        if (match and match["confidence"] >= 0.25) else 9.9)
            if (pair and pair["confidence"] >= self.edge_claim_conf
                    and pair["distance"] <= single_d - 0.15):
                self._apply_edge_pair(edge, pair, record=record)
                if record:
                    names = "+".join(s["family"] for s in pair["members"])
                    self._log_event(edge["t_ms"], "edge_on", names,
                                    pair["confidence"], edge["dP"], edge["dQ"],
                                    edge["P_after"],
                                    detail="composite step: two devices "
                                           "switched together - both claimed, "
                                           "watts split by signature")
                return
        # big MODE transition of an already-claimed device: the coffee
        # machine's heater duty-cycles between brew (~1206 W) and warm-hold
        # (~46 W); treating each -1160 W step as "coffee off" released the
        # claim every cycle, and the window model then re-labelled the watts
        # ("boiler drawing more"). If the step lands the claim on ANOTHER
        # known state of the same family, the device CHANGES STATE and the
        # claim survives; a genuine full-off (drop ~ claim watts) still
        # releases below.
        if dev != "unrecognized":
            with self.lock:
                c = self.claims.get(dev)
                w_cur = float(c["W"]) if c else None
            if c is not None:
                # where does the claim land after this step? On another known
                # state of the same family -> state change; near zero (no
                # state >= 8 W matches) -> genuine off, handled below. The
                # coffee heater's -1160 W lands on the 46 W warm-hold state;
                # unplugging the whole machine (-1206 W) lands on nothing.
                w_new = w_cur + float(edge["dP"])
                with m.lock:
                    states = list(m.modes.get(dev, []))
                target = next((b for b in states
                               if abs(b["P"]) >= 8.0
                               and abs(w_new - b["P"]) <= max(3.0, 0.15 * abs(b["P"]))
                               and abs(w_cur - b["P"]) > max(2.0, 0.10 * abs(b["P"]))),
                              None)
                if target is not None:
                    with self.lock:
                        c = self.claims.get(dev)
                        if c is not None:
                            c["W"] = float(target["P"])
                            c["Q"] = float(target["Q"])
                            c["conf"] = max(c["conf"], float(conf))
                            c["last_ms"] = int(edge["t_ms"])
                    if record:
                        with self.lock:
                            self.edge_history.append(
                                {"t_ms": int(edge["t_ms"]),
                                 "dP": float(edge["dP"]),
                                 "dQ": float(edge["dQ"]),
                                 "ih": edge.get("ih"),
                                 "P_after": edge.get("P_after"),
                                 "q_noise": edge.get("q_noise", 0.0)})
                        self._log_event(edge["t_ms"], "mode_change", dev, conf,
                                        edge["dP"], edge["dQ"], edge["P_after"],
                                        detail=f"state change ~{w_cur:.0f} W -> "
                                               f"~{target['P']:.0f} W "
                                               f"({target['label']}) - claim "
                                               "kept, not an off/on event")
                    return
        if (dev == "unrecognized" and direction == "off"
                and self._release_claim_pair(edge, record=record)):
            return
        if record:
            if dev != "unrecognized":
                det = "step matched to device signature"
            elif direction == "on" and abs(edge["dP"]) >= self.unknown_claim_min_W:
                det = ("step matches no known device signature - "
                       "tracking as unknown load")
            else:
                det = "step matches no known device signature"
            self._log_event(edge["t_ms"], f"edge_{direction}", dev, conf,
                            edge["dP"], edge["dQ"], edge["P_after"], detail=det)
        self._apply_edge(edge, direction, dev, conf, record=record)

    def _apply_edge_pair(self, edge, pair, record=True):
        """Claim BOTH members of a composite on-edge; the measured step watts
        are split between them in proportion to their signature watts."""
        tot = sum(abs(s["P"]) for s in pair["members"]) or 1e-6
        with self.lock:
            if record:
                self.edge_history.append({"t_ms": int(edge["t_ms"]),
                                          "dP": float(edge["dP"]),
                                          "dQ": float(edge["dQ"]),
                                          "ih": edge.get("ih"),
                                          "P_after": edge.get("P_after"),
                                          "q_noise": edge.get("q_noise", 0.0)})
            for s in pair["members"]:
                w = abs(float(edge["dP"])) * abs(s["P"]) / tot
                self.claims[s["family"]] = {"W": w, "Q": float(s["Q"]),
                                            "conf": float(pair["confidence"]),
                                            "t_ms": int(edge["t_ms"]),
                                            "last_ms": int(edge["t_ms"]),
                                            "pre_W": None}
                self.forced_off.pop(s["family"], None)

    def _release_claim_pair(self, edge, record=True) -> bool:
        """A composite OFF step whose drop equals the watt-sum of exactly two
        active claims releases both (the twin of _apply_edge_pair)."""
        drop = abs(float(edge["dP"]))
        with self.models.lock:
            ws = self.models.window_s
        flush_ms = int((ws + 5) * 1000)
        with self.lock:
            fams = list(self.claims)
            best = None
            for i in range(len(fams)):
                for j in range(i + 1, len(fams)):
                    s = self.claims[fams[i]]["W"] + self.claims[fams[j]]["W"]
                    err = abs(drop - s) / max(s, 20.0)
                    if err < 0.18 and (best is None or err < best[0]):
                        best = (err, fams[i], fams[j])
            if best is None:
                return False
            _, f1, f2 = best
            if record:
                self.edge_history.append({"t_ms": int(edge["t_ms"]),
                                          "dP": float(edge["dP"]),
                                          "dQ": float(edge["dQ"]),
                                          "ih": edge.get("ih"),
                                          "P_after": edge.get("P_after"),
                                          "q_noise": edge.get("q_noise", 0.0)})
            for f in (f1, f2):
                self.claims.pop(f, None)
                self.forced_off[f] = int(edge["t_ms"]) + flush_ms
        if record:
            self._log_event(edge["t_ms"], "edge_off", f"{f1}+{f2}", None,
                            edge["dP"], edge["dQ"], edge["P_after"],
                            detail="composite step: two claimed devices "
                                   "switched off together - both released")
        return True

    # ---- edge -> device state (claims) --------------------------------------
    def _apply_edge(self, edge, direction, dev, conf, record=True):
        """Turn a detected step into device state: on-edges claim the device ON
        (with the step's watts), off-edges release the claim and hold the device
        OFF until the model's window no longer contains the old samples.
        `record=False` when replaying history so it is not re-appended."""
        with self.models.lock:
            ws = self.models.window_s
        flush_ms = int((ws + 5) * 1000)
        with self.lock:
            if record:
                self.edge_history.append({"t_ms": int(edge["t_ms"]),
                                          "dP": float(edge["dP"]),
                                          "dQ": float(edge["dQ"]),
                                          "ih": edge.get("ih"),
                                          "P_after": edge.get("P_after"),
                                          "q_noise": edge.get("q_noise", 0.0)})
            if direction == "on":
                p_after = edge.get("P_after")
                if dev != "unrecognized" and conf is not None and conf >= self.edge_claim_conf:
                    self.claims[dev] = {"W": abs(float(edge["dP"])),
                                        "Q": float(edge["dQ"]),
                                        "conf": float(conf),
                                        "t_ms": int(edge["t_ms"]),
                                        "last_ms": int(edge["t_ms"]),
                                        "pre_W": (float(p_after) - float(edge["dP"])
                                                  if p_after is not None else None)}
                    self.forced_off.pop(dev, None)
                    return
                # unmatched switch-on: claim the watts as an UNKNOWN load so
                # the window model cannot pin them on a known family. A
                # soft-start device RAMPS (laptop charger: settled steps of
                # 10, 38, 63 W and a +28 W tail after the matched edge) and
                # every re-detection is the SAME device still climbing, not
                # another unknown -- fold it into the most recent still-
                # growing claim, NAMED or anonymous; the claim's watts become
                # the cumulative rise over its own pre-switch level.
                W = abs(float(edge["dP"]))
                grow, grow_last = None, None
                for c in list(self.claims.values()) + self.unknown_claims:
                    lm = c.get("last_ms", c.get("t_ms", 0))
                    if (int(edge["t_ms"]) - lm < 12000
                            and (grow_last is None or lm > grow_last)):
                        grow, grow_last = c, lm
                if grow is not None:
                    if p_after is not None and grow.get("pre_W") is not None:
                        W = max(W, float(p_after) - grow["pre_W"])
                    grow["W"] = max(grow["W"], W)
                    grow["last_ms"] = int(edge["t_ms"])
                elif W >= self.unknown_claim_min_W:
                    self.unknown_claims.append(
                        {"W": W, "t_ms": int(edge["t_ms"]),
                         "last_ms": int(edge["t_ms"]),
                         "pre_W": (float(p_after) - float(edge["dP"])
                                   if p_after is not None else None)})
                return
            # off-edge: release the claim it belongs to
            drop = dev if (dev != "unrecognized" and dev in self.claims) else None
            drop_unk = drop_model = None
            if drop is None:
                # signature match failed (or named an unclaimed device):
                # release whatever the drop best explains -- a named claim, an
                # unknown load, or a MODEL-tracked device. Model devices must
                # compete here: a hot coffee machine drawing 716 W (a state no
                # signature knows) was model-tracked at exactly 716 W, yet its
                # off-step got 'released' as the boiler's 967 W claim (26 %
                # error, inside the 35 % gate) because only claims were
                # candidates -- killing the boiler that was still heating.
                best_err = 0.35
                for fam, c in self.claims.items():
                    err = abs(abs(edge["dP"]) - c["W"]) / max(c["W"], 20.0)
                    if err < best_err:
                        drop, drop_unk, drop_model, best_err = fam, None, None, err
                for i, u in enumerate(self.unknown_claims):
                    err = abs(abs(edge["dP"]) - u["W"]) / max(u["W"], 20.0)
                    if err < best_err:
                        drop, drop_unk, drop_model, best_err = None, i, None, err
                for fam, v in self.state.items():
                    w = v.get("power_W")
                    if (v.get("on") and fam not in self.claims
                            and w is not None and math.isfinite(w) and w > 0):
                        err = abs(abs(edge["dP"]) - float(w)) / max(float(w), 20.0)
                        if err < best_err:
                            drop, drop_unk, drop_model, best_err = None, None, fam, err
            if drop is not None:
                released = self.claims.pop(drop, None)
                self.forced_off[drop] = int(edge["t_ms"]) + flush_ms
                # composite off: when the drop clearly exceeds the released
                # claim, it took a second claimed device down with it (boiler
                # + fan unplugged together) -- release the claim that covers
                # the remainder too, or it lingers at 16 W against a dead bus
                if released is not None:
                    rem = abs(float(edge["dP"])) - float(released["W"])
                    if rem >= 8.0 and self.claims:
                        f2 = min(self.claims, key=lambda f: abs(
                            rem - self.claims[f]["W"]) / max(self.claims[f]["W"], 20.0))
                        err2 = abs(rem - self.claims[f2]["W"]) / max(
                            self.claims[f2]["W"], 20.0)
                        if err2 < 0.35:
                            self.claims.pop(f2, None)
                            self.forced_off[f2] = int(edge["t_ms"]) + flush_ms
            elif drop_unk is not None:
                self.unknown_claims.pop(drop_unk)
                # the window still holds pre-drop samples that could tempt the
                # model into naming the leftover watts: keep the veto up
                self._unknown_flush_until = int(edge["t_ms"]) + flush_ms
            elif drop_model is not None:
                # the drop belongs to a model-tracked device: hold IT off
                # while the window flushes; every claim stays untouched
                self.forced_off[drop_model] = int(edge["t_ms"]) + flush_ms
            elif dev != "unrecognized":
                # no claim existed (device was on before the engine started),
                # but the step names it: hold it off while the window flushes
                self.forced_off[dev] = int(edge["t_ms"]) + flush_ms

    # ---- small edge -> mode transition of an already-on device --------------
    def _apply_mode_change(self, edge, record=True):
        """A settled step below the on/off threshold: try to read it as a
        device changing OPERATING MODE (table fan high -> low is -6 W). The
        step is matched against transitions between the known modes of every
        device currently on, anchored at each device's present watt estimate,
        and the winning device's claim watts are updated -- this is what
        finally attributes 'the fan was turned down' to the RIGHT fan instead
        of letting the window model shave watts off its sibling."""
        with self.lock:
            if record:
                self.edge_history.append({"t_ms": int(edge["t_ms"]),
                                          "dP": float(edge["dP"]),
                                          "dQ": float(edge["dQ"]),
                                          "ih": edge.get("ih"),
                                          "P_after": edge.get("P_after"),
                                          "q_noise": edge.get("q_noise", 0.0),
                                          "kind": "mode"})
            on_watts = {}
            for fam, c in self.claims.items():
                on_watts[fam] = float(c["W"])
            for nm, v in self.state.items():
                if (v.get("on") and nm not in on_watts
                        and v.get("power_W") is not None):
                    on_watts[nm] = float(v["power_W"])
        if not on_watts:
            return
        mc = self.models.match_mode_change(on_watts, edge["dP"], edge["dQ"],
                                           q_noise=edge.get("q_noise", 0.0))
        now_ms = int(edge["t_ms"])
        if mc is None:
            # a small unexplained step is normal churn (thermostat nudges,
            # PV clouds); log at most one 'ambiguous' note per 30 s and only
            # when at least two multi-mode devices were candidates
            multi = sum(1 for f in on_watts
                        if len(self.models.modes.get(f, [])) >= 2)
            if record and multi >= 2 and now_ms - self._mode_ambig_ms > 30000:
                self._mode_ambig_ms = now_ms
                self._log_event(now_ms, "mode_ambiguous", "-", None,
                                edge["dP"], edge["dQ"], edge.get("P_after"),
                                detail="small step matches no single device's "
                                       "mode change unambiguously - watts left "
                                       "to the window model")
            return
        fam, to = mc["family"], mc["to"]
        with self.lock:
            c = self.claims.get(fam)
            if c is not None:
                w_new = c["W"] + float(edge["dP"])
                # snap to the target mode's nameplate watts when the measured
                # arithmetic lands near it (drift-free display); keep the
                # measured value when it does not
                if abs(w_new - to["P"]) <= max(2.0, 0.15 * abs(to["P"])):
                    w_new = to["P"]
                c["W"] = max(0.5, w_new)
                c["Q"] = to["Q"]
                c["last_ms"] = now_ms
            else:
                # device was on by model vote only: the observed transition is
                # edge-grade evidence, so pin it with a claim at the new mode
                self.claims[fam] = {"W": float(to["P"]), "Q": float(to["Q"]),
                                    "conf": float(mc["confidence"]),
                                    "t_ms": now_ms, "last_ms": now_ms,
                                    "pre_W": None}
                self.forced_off.pop(fam, None)
        if record:
            frm = mc["from"]
            self._log_event(now_ms, "mode_change", fam, mc["confidence"],
                            edge["dP"], edge["dQ"], edge.get("P_after"),
                            detail=f"mode ~{frm['P']:.0f} W -> ~{to['P']:.0f} W "
                                   f"({frm['label']} -> {to['label']})")

    def _reconcile_claims(self, instant_W, now_ms):
        """Physical guard against stale claims: the claimed devices alone can
        never draw more than the meter reads RIGHT NOW. Compares against the
        settled instantaneous total (median of the last ~2.5 s), NOT the lagging
        window mean -- right after a 970 W boiler switches on, the 10 s window
        mean is still near the pre-switch level and would kill every big claim
        within one stride (that was the water-boiler regression). Claims also
        get a short grace period so the guard cannot race the step settling."""
        with self.lock:
            for fam in [f for f, until in self.forced_off.items() if now_ms >= until]:
                self.forced_off.pop(fam, None)
            if self.forced_off:
                # an off-edge is still flushing through the window: the
                # measured level is mid-transition, so a claims-vs-measured
                # comparison would kill healthy claims (that dropped the
                # standing lamp's 514 W claim 4 s after the laptop was
                # unplugged). Resume the guard once the flush is over.
                return
            while self.claims or self.unknown_claims:
                mature = {f: c for f, c in self.claims.items()
                          if now_ms - c["t_ms"] >= 6000}
                mature_unk = [i for i, u in enumerate(self.unknown_claims)
                              if now_ms - u["t_ms"] >= 6000]
                if not mature and not mature_unk:
                    break
                claimed = (sum(c["W"] for c in self.claims.values())
                           + sum(u["W"] for u in self.unknown_claims))
                if claimed <= instant_W + max(30.0, 0.10 * claimed):
                    break
                if mature_unk:           # anonymous loads go before named ones
                    i = max(mature_unk, key=lambda k: self.unknown_claims[k]["W"])
                    u = self.unknown_claims.pop(i)
                    self._log_event(now_ms, "claim_dropped", "unknown", None,
                                    -u["W"], None, instant_W,
                                    detail="unknown load exceeds measured power "
                                           "(missed off-edge?)")
                    continue
                fam = min(mature, key=lambda f: mature[f]["conf"])
                c = self.claims.pop(fam)
                self._log_event(now_ms, "claim_dropped", fam, c["conf"], -c["W"],
                                None, instant_W,
                                detail="edge claim exceeds measured power "
                                       "(missed off-edge?)")

    def _claim_residual(self, residual, now_ms, arrs, on_map, src_map):
        """NAME a persistent residual before calling it unknown: probe the
        signature table with the residual's own (P, estimated Q). Soft-start
        devices (a laptop charger plugs in at ~10 W and ramps to its 60+ W
        steady draw) never produce a switch-on edge that resembles their
        steady-state signature, and the window model drowns at mix scale --
        the residual is the only place their identity shows, so a TAUGHT
        device could stay 'unknown' forever without this path. The Q probe is
        an estimate (measured Q minus the ON devices' known vars), hence the
        relaxed Q tolerance; the min_conf naming floor still applies, and an
        ambiguous match collapses to None so the unknown prompt takes over."""
        k = max(1, int(2.5 * self.svc.sample_rate_hz))
        q_now = float(np.nanmedian(arrs["Q"][-k:]))
        if not math.isfinite(q_now):
            return None
        with self.models.lock:
            fam_q: dict = {}
            for s in self.models.signatures:
                fam_q.setdefault(s["family"], []).append(s["Q"])
        with self.lock:
            q_on = sum(c.get("Q", 0.0) for f, c in self.claims.items()
                       if on_map.get(f))
        for nm, on in on_map.items():
            if on and src_map.get(nm) == "model" and nm in fam_q:
                q_on += float(np.median(fam_q[nm]))
        q_res = q_now - q_on
        m = self.models.match_edge(residual, q_res, q_tol_scale=3.0)
        # an unusable single (weak, or naming an already-on family) cannot be
        # claimed, but its DISTANCE still sets the bar the pair must clear
        single_d = m["distance"] if m else 9.9
        if m and (m["confidence"] < self.min_conf or on_map.get(m["family"])):
            m = None
        # the residual can be TWO devices nobody claimed (engine started with
        # both fans already running on low: 34 W / 53 var reads exactly like
        # standing_fan HIGH alone). Same distance rule as the edge path: the
        # pair must explain the residual clearly more tightly than the single.
        pair = self.models.match_edge_pair(residual, q_res, q_noise=3.0)
        if (pair and pair["confidence"] >= 0.40
                and pair["distance"] <= single_d - 0.15
                and not any(on_map.get(s["family"]) for s in pair["members"])):
            tot = sum(abs(s["P"]) for s in pair["members"]) or 1e-6
            with self.lock:
                t0 = int(self._unknown_first or now_ms)
                for s in pair["members"]:
                    self.claims[s["family"]] = {
                        "W": abs(float(residual)) * abs(s["P"]) / tot,
                        "Q": float(s["Q"]),
                        "conf": float(pair["confidence"]), "t_ms": t0}
                    self.forced_off.pop(s["family"], None)
            names = "+".join(s["family"] for s in pair["members"])
            self._log_event(now_ms, "residual_matched", names,
                            pair["confidence"], residual, q_res, None,
                            detail="persistent residual matches the SUM of "
                                   "two device signatures - both claimed, "
                                   "watts split by signature")
            return names
        if not m:
            return None
        fam = m["family"]
        # a kilowatt-class residual while big UNMATCHED edges are still
        # churning is a cycling load mid-transition (coffee heater bursts),
        # not a silently-started device -- naming it would pin the watts on
        # whatever big family sits nearest (the boiler). The legitimate case
        # this path exists for (engine started while the boiler was already
        # running) has a QUIET edge history.
        with self.models.lock:
            fam_min = min((abs(s["P"]) for s in self.models.signatures
                           if s["family"] == fam), default=0.0)
        if fam_min >= self.big_edge_min_W:
            with self.lock:
                churn = any(e["kind"].startswith("edge_")
                            and e["device"] == "unrecognized"
                            and abs(e.get("dP_W") or 0.0) >= 300.0
                            and now_ms - e["unix_ms"] < 30000
                            for e in self.events[-20:])
            if churn:
                return None
        with self.lock:
            t0 = int(self._unknown_first or now_ms)
            self.claims[fam] = {"W": abs(float(residual)), "Q": float(q_res),
                                "conf": float(m["confidence"]), "t_ms": t0}
            self.forced_off.pop(fam, None)
        self._log_event(now_ms, "residual_matched", fam, m["confidence"],
                        residual, q_res, None,
                        detail="persistent residual matches this device's "
                               "steady signature - claimed (soft-start/ramp "
                               "device has no matching switch-on edge)")
        return fam

    def _rebuild_state_from_history(self, now_ms):
        """Software 'unplug everything and plug it back in': after a model
        reload (retrain finished, or variant switched) every edge this session
        is RE-MATCHED against the new signature table and the claims rebuilt
        in order. A step that read 'unrecognized' before a device was taught
        now resolves to that device; a step that matched the wrong sibling
        (table fan as standing fan) gets re-decided with the new signatures."""
        with self.lock:
            hist = list(self.edge_history)
            self.claims = {}
            self.forced_off = {}
            self.unknown_claims = []
            self._unknown_flush_until = 0
            self.smooth.clear()
        for e in hist:
            if e.get("kind") == "mode":
                self._apply_mode_change(e, record=False)
            else:
                self._handle_full_edge(e, record=False)
        with self.lock:
            on_now = sorted(self.claims)
        self._log_event(now_ms, "state_rebuilt", ", ".join(on_now) or "-", None,
                        None, None, None,
                        detail=f"re-matched {len(hist)} recorded edges against "
                               "the reloaded model's signatures")

    # ---- replay ground-truth comparison --------------------------------------
    def _gt_window(self, w_samples: int):
        """Ground-truth per-family ON state and mean watts over the model's
        trailing window, aligned to the replay position (mode 'full' only)."""
        idx = int(getattr(self._gt_reader, "last_index", 0))
        lo = max(0, idx - w_samples + 1)
        on, W = {}, {}
        for fam, d in self._gt["families"].items():
            seg = slice(lo, idx + 1)
            on[fam] = bool(d["on"][seg].mean() >= 0.5)
            W[fam] = float(np.mean(d["W"][seg]))
        return on, W

    def _update_gt(self, on_map, power_map, w_samples, instant_W, on_W):
        """Compare this stride's prediction with the replay ground truth,
        update the running score, and return the per-family ground-truth watts
        for the chart overlay (None when there is nothing to overlay)."""
        if self._gt is None:
            return None
        if self._gt["mode"] == "full":
            gt_on, gt_W = self._gt_window(w_samples)
            rows = []
            for fam in sorted(set(gt_on) | set(on_map)):
                t_on = bool(gt_on.get(fam, False))
                p_on = bool(on_map.get(fam, False))
                t_w = gt_W.get(fam, 0.0)
                p_wv = power_map.get(fam)
                p_w = float(p_wv) if (p_on and p_wv is not None
                                      and math.isfinite(p_wv)) else 0.0
                s = self.gt_stats.setdefault(
                    fam, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "err": 0.0, "n": 0})
                key = "tp" if (t_on and p_on) else \
                      "fp" if p_on else "fn" if t_on else "tn"
                s[key] += 1
                s["err"] += abs(p_w - t_w)
                s["n"] += 1
                rows.append({"device": fam, "gt_on": t_on, "pred_on": p_on,
                             "gt_W": round(t_w, 1), "pred_W": round(p_w, 1)})
            n_all = sum(s["n"] for s in self.gt_stats.values())
            correct = sum(s["tp"] + s["tn"] for s in self.gt_stats.values())
            err_sum = sum(s["err"] for s in self.gt_stats.values())
            per_dev = {f: {"presence_acc": round((s["tp"] + s["tn"]) / max(1, s["n"]), 3),
                           "mae_W": round(s["err"] / max(1, s["n"]), 1)}
                       for f, s in self.gt_stats.items()}
            with self.lock:
                self._gt_now = {
                    "mode": "full", "devices": rows,
                    "metrics": {"presence_accuracy": round(correct / max(1, n_all), 3),
                                "power_mae_W": round(err_sum / max(1, n_all), 1),
                                "per_device": per_dev}}
            return gt_W
        # mode 'label': the recording says which families were CONNECTED; only
        # the predicted device SET can be scored, and only while something
        # actually draws power (an idle connected device is not a model error)
        expected = set(self._gt["expected"])
        pred_set = {f for f, v in on_map.items() if v}
        active = abs(instant_W) > max(10.0, float(on_W))
        st = self._gt_set_stats
        if active:
            st["tp"] += len(pred_set & expected)
            st["fp"] += len(pred_set - expected)
            st["fn"] += len(expected - pred_set)
            st["n"] += 1
        f1 = 2 * st["tp"] / max(1, 2 * st["tp"] + st["fp"] + st["fn"])
        rows = []
        for fam in sorted(expected | pred_set):
            p_wv = power_map.get(fam)
            rows.append({"device": fam, "gt_on": fam in expected,
                         "pred_on": fam in pred_set, "gt_W": None,
                         "pred_W": (round(float(p_wv), 1)
                                    if p_wv is not None and math.isfinite(p_wv)
                                    else None)})
        with self.lock:
            self._gt_now = {"mode": "label", "label": self._gt.get("label", ""),
                            "devices": rows,
                            "metrics": {"set_f1": round(f1, 3),
                                        "scored_strides": st["n"],
                                        "active": active}}
        return None

    # ---- main loop ----------------------------------------------------------
    def _loop(self):
        while self._run_flag.is_set():
            t0 = time.time()
            try:
                self._step()
            except Exception as e:        # noqa: BLE001  (engine must survive)
                with self.lock:
                    self._teach_note = f"engine error: {e}"
            dt = self.stride_s - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    def _step(self):
        arrs = self._buffer_arrays()
        if arrs is None or self.svc.state != "connected":
            return
        m = self.models
        with m.lock:
            names = list(m.appliances)
            presence, power = m.presence, m.power
            ws, on_W = m.window_s, m.on_W
            seq = m.reload_seq
        if seq != self._model_seq:       # retrain finished or variant switched
            self._model_seq = seq
            self._rebuild_state_from_history(int(arrs["t_ms"][-1]))
            # how well did the retrain actually LEARN the taught device?
            fam = self._last_taught_family
            if fam:
                self._last_taught_family = None
                with m.lock:
                    f1 = (m.metrics.get("per_appliance_f1") or {}).get(fam)
                    in_vocab = fam in m.appliances
                now = int(arrs["t_ms"][-1])
                if not in_vocab:
                    self._log_event(now, "teach_warning", fam, None, None, None,
                                    None, detail="device did not enter the "
                                    "model vocabulary - check that its "
                                    "recording was saved and retrain again")
                elif f1 is not None and f1 < 0.70:
                    self._log_event(now, "teach_warning", fam, f1, None, None,
                                    None, detail=f"model learned this device "
                                    f"only weakly (held-out F1 {f1:.2f}) - "
                                    "record it again, ideally isolated and in "
                                    "every operating mode")

        # -- edge first (needs only the raw signal, works even with no model) --
        edge = self._detect_edge(arrs)
        if edge is not None and edge.get("kind") == "mode":
            # sub-threshold settled step: mode transition of an on device,
            # never an on/off event
            self._apply_mode_change(edge)
        elif edge is not None:
            self._handle_full_edge(edge)

        sr = self.svc.sample_rate_hz
        w_samples = max(1, int(round(ws * sr)))
        sig = self._window_signal(arrs, w_samples)
        if sig is None:
            return
        now_ms = int(arrs["t_ms"][-1])
        total_W = float(np.nanmean(sig.P))
        k_now = max(1, int(2.5 * sr))
        instant_W = float(np.nanmedian(arrs["P"][-k_now:]))
        # a big level change still TRANSITING the 8 s edge window belongs to
        # the edge pipeline: reconciling claims against the already-dropped
        # instant total raced the detector and killed the coffee claim
        # (1206 W) two strides before its -1160 W heater-off edge could be
        # read as a state change. Hold the guard until the transition has
        # either produced its edge or aged out of the lookback.
        k_a, k_b = int(11.5 * sr), int(9 * sr)
        in_transition = False
        if len(arrs["P"]) >= k_a and k_a > k_b:
            lvl_prev = float(np.nanmedian(arrs["P"][-k_a:-k_b]))
            in_transition = (math.isfinite(lvl_prev)
                             and abs(instant_W - lvl_prev) >= 25.0)
        if not in_transition:
            self._reconcile_claims(instant_W, now_ms)
        with self.lock:
            claims = {f: dict(c) for f, c in self.claims.items()}
            forced_off = dict(self.forced_off)
            unk_claims = [dict(u) for u in self.unknown_claims]
            unk_veto = bool(unk_claims) or now_ms < self._unknown_flush_until
            prev_on_map = {nm: v.get("on", False) for nm, v in self.state.items()}

        on_map, power_map, prob_map, src_map = {}, {}, {}, {}
        raw_model_W = 0.0        # unvetoed model estimate, for staleness only
        if presence is not None and names:
            X, _, _ = nl.aggregate_windows(sig, ws, canon=names)
            with m.lock:
                X = nl.slice_features(X, m.features or None)
            X = X[-1:].copy()
            proba = nl.presence_proba(presence, X)[0]
            with self.lock:
                # vocabulary size may change across a retrain hot-reload
                if self.smooth and len(self.smooth[-1]) != len(proba):
                    self.smooth.clear()
                self.smooth.append(proba)
                proba_s = np.median(np.vstack(list(self.smooth)), axis=0)
            watts = power.predict(X)[0] if power is not None else np.zeros(len(names))
            # what the model would explain WITHOUT any veto: the unknown-load
            # staleness check must not judge support from post-veto watts (a
            # bogus unknown claim vetoes the very devices that explain the
            # power, then looks 'supported' by the residual it manufactured)
            raw_model_W = float(sum(max(0.0, float(watts[i]))
                                    for i in range(len(names))
                                    if proba_s[i] >= 0.5))
            for i, nm in enumerate(names):
                was_on = self.state.get(nm, {}).get("on", False)
                # confidence floor with hysteresis: the window model only NAMES
                # a device it is at least min_conf sure of (default 0.70), then
                # holds it on down to min_conf - 0.20 so a near-threshold
                # probability (coffee near ~0.5 in mixes) does not flap on/off.
                # Power the model is less sure of stays in the residual and is
                # reported as unknown rather than a low-confidence name -- fast
                # events are the edge claims' job, not the window model's.
                on = bool(proba_s[i] >= (self.min_conf - 0.20 if was_on
                                         else self.min_conf))
                w = float(watts[i]) if on else 0.0
                if on and abs(w) < 0.5 and power is None:
                    w = float("nan")
                on_map[nm], power_map[nm], prob_map[nm] = on, w, float(proba_s[i])
                src_map[nm] = "model"

        # -- veto model-only switch-ons that have nothing real to explain ------
        # Every real switch-on produces its own edge. A model device newly
        # flipping ON with no edge of its own is suspect in two situations:
        # (a) an UNKNOWN load is on the mains (or still flushing out of the
        #     window) -- the model is pinning the unknown's watts on the
        #     nearest known family ("hair dryer detected as standing lamp");
        # (b) the claims already account for the measured total -- there are
        #     no unexplained watts left, so the new ON is the model
        #     re-labeling claimed power (lamp + laptop read as "coffee").
        # Devices already ON, edge-claimed devices, and devices a recent
        # on-edge named are untouched.
        headroom = (total_W - sum(c["W"] for c in claims.values())
                    - sum(u["W"] for u in unk_claims))
        with self.lock:
            recent_named = {ev["device"] for ev in self.events[-12:]
                            if ev["kind"] == "edge_on"
                            and now_ms - ev["unix_ms"] < (ws + 6) * 1000}
            recent_on_dPs = [e["dP"] for e in self.edge_history
                             if e["dP"] > 0 and e.get("kind") != "mode"
                             and now_ms - e["t_ms"] < (ws + 6) * 1000]
        if unk_veto or headroom < max(10.0, 2 * on_W):
            for nm in list(on_map):
                if (on_map[nm] and nm not in claims
                        and not prev_on_map.get(nm, False)
                        and nm not in recent_named):
                    on_map[nm] = False
                    power_map[nm] = 0.0
        # -- big devices only switch ON with a real step ------------------------
        # A kilowatt-class load physically cannot appear without an edge; a
        # window-model vote alone ("coffee machine popped up while the fans
        # ran") is always the model re-labeling someone else's watts. The ONE
        # legitimate no-edge case -- the engine started while the device was
        # already running -- is covered by the residual-claiming path, which
        # names a persistent residual against the signature table within
        # seconds. Applies regardless of headroom.
        with m.lock:
            fam_min_sig = {}
            for s in m.signatures:
                fam_min_sig[s["family"]] = min(
                    fam_min_sig.get(s["family"], float("inf")), abs(s["P"]))
        for nm in list(on_map):
            min_sig = fam_min_sig.get(nm)
            if (on_map[nm] and nm not in claims
                    and not prev_on_map.get(nm, False)
                    and min_sig is not None and min_sig >= self.big_edge_min_W
                    and nm not in recent_named
                    and not any(dp >= 0.4 * min_sig for dp in recent_on_dPs)):
                on_map[nm] = False
                power_map[nm] = 0.0
                if now_ms - self._veto_log_ms.get(nm, 0) > 60000:
                    self._veto_log_ms[nm] = now_ms
                    self._log_event(now_ms, "model_veto", nm,
                                    prob_map.get(nm), None, None, total_W,
                                    detail=f"window model voted ON but no "
                                           f">= {0.4 * min_sig:.0f} W switch-on "
                                           "step was seen - a device this size "
                                           "cannot start silently (phantom "
                                           "suppressed)")

        # -- merge edge claims over the model ----------------------------------
        # A claim forces the device ON with the watts its own switch-on step
        # measured; a fresh off-edge forces it OFF while the model window still
        # contains pre-switch samples. The model keeps authority over everything
        # unclaimed, but its watts are rescaled into what the claims (named and
        # unknown) leave of the measured total (this is what splits "boiler at
        # 1444 W" into boiler 943 W + lamp 501 W).
        for nm, c in claims.items():
            on_map[nm] = True
            power_map[nm] = c["W"]
            prob_map[nm] = max(prob_map.get(nm, 0.0), c["conf"])
            src_map[nm] = "edge"
        for nm in forced_off:
            if on_map.get(nm):
                on_map[nm] = False
                power_map[nm] = 0.0
                src_map[nm] = "edge"
        claimed_W = sum(c["W"] for f, c in claims.items() if on_map.get(f))
        unknown_W = sum(u["W"] for u in unk_claims)
        model_on = [nm for nm in on_map
                    if on_map[nm] and nm not in claims and math.isfinite(power_map.get(nm, 0.0))]
        pred_sum = sum(power_map[nm] for nm in model_on)
        remaining = max(0.0, total_W - claimed_W - unknown_W)
        if pred_sum > 1.0 and pred_sum > remaining:
            scale = remaining / pred_sum
            for nm in model_on:
                power_map[nm] = power_map[nm] * scale
        # -- snap model watts to the nearest known operating mode --------------
        # A model-tracked device's regressed watts land BETWEEN its physical
        # modes ('table_fan 13.9 W'); when exactly one mode is close, report
        # that mode's watts instead -- the leftover goes back into the
        # residual, where it is honest information rather than smeared error.
        with m.lock:
            fam_modes = {f: list(ms) for f, ms in m.modes.items()}
        for nm in model_on:
            near = [md for md in fam_modes.get(nm, [])
                    if abs(power_map[nm] - md["P"]) <= max(3.0, 0.20 * abs(md["P"]))]
            if len(near) == 1:
                power_map[nm] = float(near[0]["P"])

        # replay only: score this stride against the file's ground truth
        gt_chart = self._update_gt(on_map, power_map, w_samples, instant_W, on_W)

        explained = sum(w for nm, w in power_map.items()
                        if on_map.get(nm) and math.isfinite(w))
        residual = total_W - explained
        # -- state transitions -> events -------------------------------------
        # union: an edge claim can name a device the model vocabulary does not
        # know yet (taught but not retrained) -- it still gets live state
        track = list(dict.fromkeys(list(names) + list(on_map)))
        with self.lock:
            prev = self.state
            new_state = {}
            for nm in track:
                p = prev.get(nm, {})
                on = on_map.get(nm, False)
                since = p.get("since_ms")
                if on and not p.get("on", False):
                    if nm in claims:                         # exact switch-on moment
                        since = claims[nm]["t_ms"]
                    else:
                        since = now_ms - int(ws * 1000 / 2)  # window centre-ish
                        # a recent matching edge gives the exact switch-on moment
                        for ev in reversed(self.events[-12:]):
                            if (ev["device"] == nm and ev["kind"].startswith("edge_on")
                                    and now_ms - ev["unix_ms"] < (ws + 6) * 1000):
                                since = ev["unix_ms"]; break
                    self._log_event(since, "device_on", nm, prob_map.get(nm),
                                    None, None, total_W,
                                    detail=f"~{power_map.get(nm, 0):.0f} W"
                                           + (" (edge)" if src_map.get(nm) == "edge" else ""))
                elif not on and p.get("on", False):
                    self._log_event(now_ms - int(ws * 1000 / 2), "device_off", nm,
                                    prob_map.get(nm), None, None, total_W)
                    since = None
                new_state[nm] = {"on": on, "prob": round(prob_map.get(nm, 0.0), 3),
                                 "power_W": None if not math.isfinite(power_map.get(nm, 0.0))
                                 else round(power_map.get(nm, 0.0), 1),
                                 "since_ms": since,
                                 "src": src_map.get(nm, "model")}
            self.state = new_state
            self.total_W = round(total_W, 1)
            self.residual_W = round(residual, 1)
            self.explained_frac = round(1.0 - min(1.0, abs(residual) / max(abs(total_W), 1e-9)), 3) \
                if abs(total_W) > 1 else 1.0
            entry = {
                "t_ms": now_ms, "P_total": round(total_W, 1),
                "residual": round(residual, 1),
                "devices": {nm: (new_state[nm]["power_W"] if new_state[nm]["on"] else 0.0)
                            for nm in track}}
            if gt_chart is not None:
                entry["gt"] = {f: round(w, 1) for f, w in gt_chart.items()}
            self.history.append(entry)

        # -- unknown-device monitor -------------------------------------------
        # While a fresh claim or a forced-off is still flushing through the
        # window, the window-mean residual is transiently huge (the mean lags
        # the step by up to window_s) -- that is settling, not an unknown device.
        settling = bool(forced_off) or any(
            now_ms - c["t_ms"] < (ws + 4) * 1000 for c in claims.values())
        teaching = self._teach_thread is not None and self._teach_thread.is_alive()

        # -- stale unknown-load eviction ---------------------------------------
        # An unknown claim whose watts the measurement no longer supports must
        # not sit in "currently on" forever (its off-edge was missed, or was
        # matched to a named claim instead). The residual cannot expose this:
        # the rescale squeezes model watts into what the claims leave, which
        # MANUFACTURES residual equal to the claimed unknown power. Staleness
        # is therefore judged against the model's RAW estimate: the watts left
        # after named claims and unscaled model predictions must still make
        # room for the unknown load. Sustained lack of support (15 s) evicts
        # the smallest claim first; if a genuine unknown is ever evicted, its
        # unexplained watts re-raise the prompt through the residual monitor.
        if unk_claims and not settling:
            deficit = total_W - claimed_W - max(pred_sum, raw_model_W)
            mature_unknowns = (now_ms - min(u["t_ms"] for u in unk_claims)
                               > (ws + 6) * 1000)
            if deficit < 0.5 * unknown_W and mature_unknowns:
                if self._unknown_stale_ms is None:
                    self._unknown_stale_ms = now_ms
                elif now_ms - self._unknown_stale_ms >= 15000:
                    with self.lock:
                        if self.unknown_claims:
                            i = min(range(len(self.unknown_claims)),
                                    key=lambda k: self.unknown_claims[k]["W"])
                            u = self.unknown_claims.pop(i)
                            self._log_event(now_ms, "claim_dropped", "unknown",
                                            None, -u["W"], None, total_W,
                                            detail="measured power no longer "
                                                   "supports this unknown load "
                                                   "(stale claim)")
                    self._unknown_stale_ms = None
            else:
                self._unknown_stale_ms = None
        else:
            self._unknown_stale_ms = None
        # The tolerance must scale with how the ON power was EXPLAINED, not
        # with the raw total: model-estimated watts carry ~15 % error, but an
        # edge claim's watts were measured off the step itself, so claimed
        # power only needs a small drift budget. With boiler + lamp claimed
        # (1430 W), a flat 15 %-of-total threshold (~215 W) would hide a 60 W
        # laptop charger forever; 4 % drift budget (~57 W) does not.
        model_W = sum(power_map[nm] for nm in on_map
                      if on_map[nm] and src_map.get(nm) == "model"
                      and math.isfinite(power_map.get(nm, 0.0)))
        threshold = max(self.unknown_min_W,
                        self.unknown_frac * abs(model_W) + 0.04 * claimed_W)
        # a persisting unmatched on-edge is direct evidence, no need to wait
        # for the window residual to agree
        big_unknown = any(u["W"] >= self.unknown_min_W
                          and now_ms - u["t_ms"] >= self.unknown_persist_s * 1000
                          for u in unk_claims)
        retraining = self.retrainer.status()["state"] == "running"
        over = ((abs(residual) > threshold or big_unknown)
                and not settling and not teaching and not retraining)

        # try to NAME the persistent residual before (or while) prompting:
        # a taught soft-start device is recognized here, not by its edge
        with self.lock:
            persisted = (self.unknown is not None
                         or (self._unknown_first is not None
                             and now_ms - self._unknown_first
                             >= self.unknown_persist_s * 1000))
        if (over and persisted and abs(residual) >= self.unknown_min_W
                and self._claim_residual(residual, now_ms, arrs,
                                         on_map, src_map)):
            with self.lock:
                self._unknown_first = None
                self.unknown = None
            over = False

        with self.lock:
            if over:
                self._unknown_clear_ms = None
                if self._unknown_first is None:
                    # an unmatched on-edge pinpoints the actual switch-on
                    self._unknown_first = min(
                        [now_ms] + [u["t_ms"] for u in unk_claims])
                elif (self.unknown is None
                      and now_ms - self._unknown_first >= self.unknown_persist_s * 1000):
                    self.unknown = {"since_ms": int(self._unknown_first),
                                    "since_iso": datetime.fromtimestamp(
                                        self._unknown_first / 1000.0).astimezone()
                                        .isoformat(timespec="seconds"),
                                    "typical_W": round(residual, 1)}
                    self._log_event(self._unknown_first, "unknown_detected", "unknown",
                                    None, residual, None, total_W,
                                    detail="sustained unexplained power - "
                                           "please name this device (Teach)")
            elif self.unknown is not None and (settling or teaching or retraining):
                # transient suppression (a device switching, teach, retrain):
                # FREEZE the prompt instead of flapping cleared/detected
                self._unknown_clear_ms = None
            else:
                self._unknown_first = None
                if self.unknown is not None:
                    # residual genuinely low: clear only after it stays low,
                    # so one settled stride cannot dismiss a real unknown
                    if self._unknown_clear_ms is None:
                        self._unknown_clear_ms = now_ms
                    elif now_ms - self._unknown_clear_ms >= 6000:
                        self._log_event(now_ms, "unknown_cleared", "unknown",
                                        None, residual, None, total_W)
                        self.unknown = None
                        self._unknown_clear_ms = None
                else:
                    self._unknown_clear_ms = None
            if self.unknown is not None:
                self.unknown["typical_W"] = round(residual, 1)

    # ---- teach: two guided flows to capture exactly one device --------------
    # mode='isolated' (default): clean recording, same protocol as the manual
    # record button. Naive in-mix captures gave visibly worse models than a
    # clean isolated recording, so this stays the recommended flow:
    #   disconnect everything -> 5 s off baseline -> connect the device ->
    #   teach_record_s ON -> disconnect -> 5 s off tail -> save -> retrain.
    # mode='inmix' ("teach on the go"): when emptying the mains is impractical
    # (fridge, router, a running experiment) the other devices KEEP RUNNING and
    # only the unknown device is toggled:
    #   device off -> settled background baseline A -> device on ->
    #   teach_record_s of the mix -> device off -> closing baseline B.
    # The device's own signal is isolated by subtracting the baseline,
    # linearly interpolated A->B so slow background drift is removed too; if
    # A and B disagree beyond a drift budget some OTHER device toggled
    # mid-capture and the capture is DISCARDED instead of teaching the model a
    # polluted signature (that validation is what the naive approach lacked).
    # Phase changes are driven by the measured power itself (no confirm
    # clicks); progress/instructions are shown via snapshot()["teach_guide"].
    def teach(self, label: str, retrain: bool = True,
              mode: str = "isolated") -> dict:
        label = (label or "").strip()
        if not label:
            raise ValueError("empty device name")
        mode = (mode or "isolated").strip().lower()
        if mode not in ("isolated", "inmix"):
            raise ValueError(f"unknown teach mode '{mode}' "
                             "(use 'isolated' or 'inmix')")
        if self.svc.state != "connected":
            raise RuntimeError("meter not connected")
        with self.svc._lock:
            busy = self.svc.session is not None
        if busy:
            raise RuntimeError("a manual recording session is active - stop it first")
        with self.lock:
            if self._teach_thread is not None and self._teach_thread.is_alive():
                raise RuntimeError("a teach session is already running")
            self._teach_cancel = False
            # the unknown residual sizes the in-mix off/on thresholds
            expected_W = None
            if self.unknown is not None:
                w = abs(float(self.unknown.get("typical_W") or 0.0))
                expected_W = w if w > 0 else None
            self.unknown = None            # the guide takes over the prompt
            self._unknown_first = None
        if mode == "inmix":
            th = threading.Thread(target=self._teach_inmix_worker, daemon=True,
                                  args=(label, retrain, expected_W),
                                  name="teach-inmix")
            detail = ("in-mix capture started - other devices keep running; "
                      "follow the instructions on the dashboard")
        else:
            th = threading.Thread(target=self._teach_worker, daemon=True,
                                  args=(label, retrain), name="teach-guided")
            detail = ("guided clean recording started - follow the "
                      "instructions on the dashboard")
        with self.lock:
            self._teach_thread = th
        self._log_event(int(time.time() * 1000), "teach_recording",
                        nl.parse_family(label), None, None, None, None,
                        detail=detail)
        th.start()
        return {"scheduled": True, "guided": True, "label": label, "mode": mode,
                "on_s": self.teach_record_s, "retrain": bool(retrain)}

    def cancel_teach(self) -> bool:
        with self.lock:
            running = self._teach_thread is not None and self._teach_thread.is_alive()
            self._teach_cancel = True
        return running

    def _set_guide(self, phase: str, msg: str):
        with self.lock:
            self.guide = {"phase": phase, "msg": msg}

    def _instant_W(self):
        arrs = self._buffer_arrays()
        if arrs is None or not len(arrs["t_ms"]):
            return None
        k = max(1, int(1.5 * self.svc.sample_rate_hz))
        return float(np.nanmedian(arrs["P"][-k:]))

    def _wait_power(self, cond, hold_s, timeout_s, phase, msg_fmt) -> bool:
        """Advance when cond(instant watts) has held for hold_s seconds."""
        held, t0 = 0.0, time.time()
        while time.time() - t0 < timeout_s:
            if self._teach_cancel or not self._run_flag.is_set():
                return False
            w = self._instant_W()
            held = held + 0.5 if (w is not None and cond(w)) else 0.0
            self._set_guide(phase, msg_fmt.format(w=w if w is not None else 0.0))
            if held >= hold_s:
                return True
            time.sleep(0.5)
        return False

    def _post_teach_checks(self, label: str):
        """After a teach recording is saved: warn when the new device's
        signature sits on top of an existing family (they WILL be confused in
        mixes), and remember the family so the post-retrain hook can verify
        the model actually learned it."""
        fam = nl.parse_family(label)
        with self.lock:
            self._last_taught_family = fam
        try:
            rep = self.models.signature_report(label)
        except Exception:                 # noqa: BLE001 - advisory only
            rep = None
        if rep and rep["distance"] < 1.2:
            self._log_event(int(time.time() * 1000), "teach_warning", fam,
                            None, rep["dP_W"], rep["dQ_var"], None,
                            detail=f"signature is close to '{rep['family']}' "
                                   f"({rep['label']}: dP {rep['dP_W']:+.0f} W, "
                                   f"dQ {rep['dQ_var']:+.0f} var) - expect "
                                   "confusion in mixes; record its other "
                                   "modes or a longer run to separate them")

    def _teach_worker(self, label, retrain):
        off_W, lead_s, tail_s = 5.0, 5.0, 5.0
        on_W = max(8.0, self.edge_min_W)
        started = False
        try:
            # 1. everything off (including the new device)
            if not self._wait_power(lambda w: abs(w) < off_W, 3.0, 300.0,
                    "disconnect_all",
                    f"Step 1/5 - DISCONNECT ALL devices, including '{label}'. "
                    "Waiting for total power < 5 W (now {w:.0f} W)"):
                raise RuntimeError("timeout/cancel while waiting for all-off")
            # 2. clean session recording, 5 s off baseline
            self.svc.start_session(label)
            started = True
            for r in range(int(lead_s), 0, -1):
                if self._teach_cancel:
                    raise RuntimeError("cancelled")
                self._set_guide("off_lead", f"Step 2/5 - recording OFF baseline, "
                                            f"{r} s. Keep everything disconnected.")
                time.sleep(1.0)
            # 3. connect the device
            if not self._wait_power(lambda w: abs(w) > on_W, 1.5, 180.0,
                    "connect_now",
                    f"Step 3/5 - now CONNECT '{label}' (only this device). "
                    "Waiting for power (now {w:.0f} W)"):
                raise RuntimeError("device was not connected within 3 minutes")
            # 4. record it running
            t0 = time.time()
            while time.time() - t0 < self.teach_record_s:
                if self._teach_cancel:
                    raise RuntimeError("cancelled")
                left = self.teach_record_s - (time.time() - t0)
                w = self._instant_W() or 0.0
                self._set_guide("recording_on",
                                f"Step 4/5 - recording '{label}' at {w:.0f} W, "
                                f"{left:.0f} s left. Leave it running.")
                time.sleep(1.0)
            # 5. disconnect + off tail (if the user never disconnects, save
            # anyway - the ON data is already captured)
            if self._wait_power(lambda w: abs(w) < off_W, 1.5, 120.0,
                    "disconnect_now",
                    f"Step 5/5 - now DISCONNECT '{label}'. "
                    "Waiting for power to drop (now {w:.0f} W)"):
                for r in range(int(tail_s), 0, -1):
                    if self._teach_cancel:
                        raise RuntimeError("cancelled")
                    self._set_guide("off_tail", f"Step 5/5 - recording OFF tail, {r} s.")
                    time.sleep(1.0)
            done = self.svc.stop_session() or {}
            started = False
            self.models._load_signatures()
            self._post_teach_checks(label)
            fname = os.path.basename(str(done.get("file", "recording")))
            self._log_event(int(time.time() * 1000), "taught",
                            nl.parse_family(label), None, None, None, None,
                            detail=f"guided clean recording saved ({fname}, "
                                   f"{done.get('samples', '?')} samples)"
                                   + ("; retraining" if retrain else ""))
            with self.lock:
                self._teach_note = f"saved {fname}" + ("; retraining" if retrain else "")
            if retrain:
                self.retrainer.start()
        except Exception as e:            # noqa: BLE001
            if started:                   # discard the partial recording
                try:
                    done = self.svc.stop_session() or {}
                    f = done.get("file")
                    if f and os.path.exists(f):
                        os.remove(f)
                except Exception:
                    pass
            with self.lock:
                self._teach_note = f"teach '{label}' aborted: {e}"
            self._log_event(int(time.time() * 1000), "teach_failed",
                            nl.parse_family(label), None, None, None, None,
                            detail=f"guided recording aborted: {e}")
        finally:
            with self.lock:
                self.guide = None

    # ---- teach variant: IN-MIX capture ("teach on the go") ------------------
    def _inmix_wait(self, cap: MixCapture, cond, hold_s, timeout_s,
                    phase, msg_fmt):
        """Like _wait_power but drains samples into the capture store on every
        tick. Returns the last instantaneous watts, or None on timeout/cancel."""
        held, t0, w = 0.0, time.time(), None
        while time.time() - t0 < timeout_s:
            if self._teach_cancel or not self._run_flag.is_set():
                return None
            cap.collect()
            w = self._instant_W()
            held = held + 0.5 if (w is not None and cond(w)) else 0.0
            self._set_guide(phase, msg_fmt.format(w=w if w is not None else 0.0))
            if held >= hold_s:
                return w
            time.sleep(0.5)
        return None

    def _inmix_baseline(self, cap: MixCapture, dur_s, phase, msg_fmt) -> dict:
        """Capture a settled background baseline (channel medians over dur_s).
        A noisy baseline (some background device still ramping) gets one more
        attempt before it is accepted as-is -- the drift check between the two
        baselines is the hard guard."""
        st = None
        for _ in range(2):
            t0_ms, t0 = int(time.time() * 1000), time.time()
            while time.time() - t0 < dur_s:
                if self._teach_cancel or not self._run_flag.is_set():
                    raise RuntimeError("cancelled")
                cap.collect()
                self._set_guide(phase, msg_fmt.format(
                    left=max(0.0, dur_s - (time.time() - t0))))
                time.sleep(0.5)
            cap.collect()
            st = cap.stats(t0_ms, int(time.time() * 1000))
            if st is None:
                raise RuntimeError("no samples captured for the baseline")
            if st["P_std"] <= max(6.0, 0.05 * abs(st["P"])):
                break
        return st

    def _teach_inmix_worker(self, label, retrain, expected_W=None):
        cap = MixCapture(self)
        base_s = 6.0
        on_W_min = max(8.0, self.edge_min_W)
        try:
            ref = self._instant_W()
            if ref is None:
                raise RuntimeError("no live samples in the buffer yet")
            drop_min = max(on_W_min, 0.3 * expected_W if expected_W else 0.0)
            # 1. switch OFF only the unknown device; the background keeps running
            if self._inmix_wait(cap, lambda w: w <= ref - drop_min, 3.0, 300.0,
                    "inmix_off",
                    f"Step 1/5 - keep every OTHER device running exactly as it "
                    f"is; switch OFF only '{label}'. Waiting for a settled drop "
                    f"of >= {drop_min:.0f} W (total now {{w:.0f}} W)") is None:
                raise RuntimeError("timeout/cancel while waiting for the device "
                                   "to switch off")
            base_a = self._inmix_baseline(cap, base_s, "inmix_baseline_a",
                    "Step 2/5 - measuring the background baseline, "
                    "{left:.0f} s. Do not switch anything.")
            # 2. switch it back ON. The gate is the minimum DETECTABLE rise,
            # not a fraction of the earlier drop: a soft-start device (laptop
            # charger) comes back at ~10 W and ramps, and would time out
            # against a half-of-previous-draw threshold.
            rise_min = on_W_min
            if self._inmix_wait(cap, lambda w: w >= base_a["P"] + rise_min,
                    2.0, 300.0, "inmix_on",
                    f"Step 3/5 - now switch '{label}' back ON (leave the others "
                    f"alone). Waiting for +{rise_min:.0f} W over the "
                    f"{base_a['P']:.0f} W baseline (total now {{w:.0f}} W)") is None:
                raise RuntimeError("timeout/cancel while waiting for the device "
                                   "to switch on")
            t_on_ms = int(time.time() * 1000)
            # 3. record the mix with the device running
            t0 = time.time()
            while time.time() - t0 < self.teach_record_s:
                if self._teach_cancel:
                    raise RuntimeError("cancelled")
                cap.collect()
                left = self.teach_record_s - (time.time() - t0)
                w = self._instant_W() or 0.0
                self._set_guide("inmix_recording",
                                f"Step 4/5 - recording '{label}' inside the mix "
                                f"at {w:.0f} W total, {left:.0f} s left. Keep "
                                "every device exactly as it is.")
                time.sleep(0.5)
            cap.collect()
            t_off_req_ms = int(time.time() * 1000)
            est_W = cap.median("P", t_on_ms, t_off_req_ms) - base_a["P"]
            # 4. switch it OFF again -> closing baseline
            cur = self._instant_W()
            cur = cur if cur is not None else base_a["P"] + max(est_W, 0.0)
            drop2 = max(on_W_min, 0.5 * max(est_W, 0.0))
            if self._inmix_wait(cap, lambda w: w <= cur - drop2, 3.0, 300.0,
                    "inmix_off2",
                    f"Step 5/5 - switch '{label}' OFF again. Waiting for a "
                    f"settled drop of >= {drop2:.0f} W (total now {{w:.0f}} W)") is None:
                raise RuntimeError("timeout/cancel while waiting for the final "
                                   "switch-off")
            base_b = self._inmix_baseline(cap, base_s, "inmix_baseline_b",
                    "Step 5/5 - measuring the closing baseline, "
                    "{left:.0f} s. Do not switch anything.")
            # 5. validate: if the background changed between the two baselines,
            # another device toggled mid-capture -> the subtraction is invalid
            drift = abs(base_b["P"] - base_a["P"])
            drift_tol = max(15.0, 0.25 * max(est_W, on_W_min))
            if drift > drift_tol:
                raise RuntimeError(
                    f"background changed by {drift:.0f} W during the capture "
                    f"(budget {drift_tol:.0f} W) - another device must have "
                    "switched. Keep the other devices steady and teach again")
            iso = self._isolate_inmix(cap, base_a, base_b, t_on_ms, t_off_req_ms)
            if iso["device_W"] < on_W_min:
                raise RuntimeError(
                    f"isolated signal is only {iso['device_W']:.0f} W - no "
                    "measurable device found on top of the background")
            path = self._save_inmix_recording(label, iso, base_a, base_b, drift)
            self.models._load_signatures()
            self._post_teach_checks(label)
            fname = os.path.basename(path)
            self._log_event(int(time.time() * 1000), "taught",
                            nl.parse_family(label), None, None, None, None,
                            detail=f"in-mix capture isolated and saved ({fname}, "
                                   f"{iso['n']} samples, ~{iso['device_W']:.0f} W "
                                   f"on a {base_a['P']:.0f} W background, "
                                   f"drift {drift:.0f} W)"
                                   + ("; retraining" if retrain else ""))
            with self.lock:
                self._teach_note = (f"saved {fname} (in-mix, "
                                    f"~{iso['device_W']:.0f} W isolated)"
                                    + ("; retraining" if retrain else ""))
            if retrain:
                self.retrainer.start()
        except Exception as e:            # noqa: BLE001
            with self.lock:
                self._teach_note = f"teach '{label}' (in-mix) aborted: {e}"
            self._log_event(int(time.time() * 1000), "teach_failed",
                            nl.parse_family(label), None, None, None, None,
                            detail=f"in-mix capture aborted: {e}")
        finally:
            with self.lock:
                self.guide = None

    def _isolate_inmix(self, cap: MixCapture, base_a, base_b,
                       t_on_ms, t_off_ms) -> dict:
        """Isolated device channels = captured mix minus the background
        baseline, interpolated linearly from baseline A to baseline B so slow
        background drift (fridge duty cycle, PV) is subtracted as well."""
        a = cap.arrays()
        t = a["t_ms"].astype(np.int64)
        span = max(1.0, base_b["t_ms"] - base_a["t_ms"])
        frac = np.clip((t - base_a["t_ms"]) / span, 0.0, 1.0)

        def base(k):
            return base_a[k] + frac * (base_b[k] - base_a[k])

        P = a["P"] - base("P")
        Q = a["Q"] - base("Q")
        P1 = a["P1"] - base("P1")
        P2 = a["P2"] - base("P2")
        P3 = a["P3"] - base("P3")
        S = np.hypot(P, Q)
        PF = np.divide(P, S, out=np.ones_like(P), where=S > 1e-6)
        # THD_I: percentages do not subtract, harmonic CURRENTS of independent
        # loads add ~orthogonally -> estimate the device's harmonic current by
        # RSS subtraction and re-normalize to its own fundamental (nominal
        # 230 V cancels out of mix vs baseline, it only scales both).
        V = 230.0
        i_f_mix = np.hypot(np.nan_to_num(a["P"]), np.nan_to_num(a["Q"])) / V
        i_h_mix = a["THD"] / 100.0 * i_f_mix
        i_f_base = np.hypot(base("P"), base("Q")) / V
        i_h_base = (base_a["THD"] + frac * (base_b["THD"] - base_a["THD"])) \
            / 100.0 * i_f_base
        i_f_dev = np.hypot(P, Q) / V
        with np.errstate(invalid="ignore"):
            i_h_dev = np.sqrt(np.clip(i_h_mix ** 2 - i_h_base ** 2, 0.0, None))
            THD = np.where(np.isfinite(i_h_dev) & (i_f_dev > 1e-3),
                           100.0 * i_h_dev / np.maximum(i_f_dev, 1e-9), np.nan)
        on_sel = (t >= t_on_ms) & (t <= t_off_ms)
        device_W = float(np.nanmedian(P[on_sel])) if on_sel.any() else 0.0
        return {"t_ms": t, "P_total": P, "Q_total": Q, "S_total": S,
                "PF_total": PF, "P_L1": P1, "P_L2": P2, "P_L3": P3,
                "THD_I_L1": THD, "device_W": device_W, "n": int(len(t))}

    def _save_inmix_recording(self, label, iso, base_a, base_b, drift) -> str:
        """Write the isolated signal as a standard recorder .h5 (same layout as
        pac_reader's IncrementalHDF5Writer) so signatures, the scenario mixer,
        and identify training consume it like any clean recording."""
        channels = ["P_total", "Q_total", "S_total", "PF_total",
                    "P_L1", "P_L2", "P_L3", "THD_I_L1"]
        safe = pr._safe_label(label)
        fname = f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        os.makedirs(self.models.recordings_dir, exist_ok=True)
        path = os.path.join(self.models.recordings_dir, fname)
        t_us = iso["t_ms"].astype(np.int64) * 1000
        sr = float(self.svc.sample_rate_hz)
        with h5py.File(path, "w") as f:
            f.create_dataset("timestamp", data=t_us, compression="lzf")
            m = f.create_group("measurements")
            for ch in channels:
                m.create_dataset(ch, data=np.asarray(iso[ch], dtype=np.float32),
                                 compression="lzf")
            md = f.create_group("metadata")
            md.attrs["format_version"] = pr.FORMAT_VERSION
            md.attrs["app_version"] = pr.APP_VERSION
            md.attrs["sample_rate_hz"] = sr
            md.attrs["anchor_datetime"] = datetime.fromtimestamp(
                t_us[0] / 1e6, tz=timezone.utc).isoformat()
            md.attrs["source"] = "live_teach_inmix"
            md.attrs["appliance_label"] = label
            md.attrs["channels"] = json.dumps(channels)
            md.attrs["harmonics_enabled"] = False
            md.attrs["teach_mode"] = "in_mix"
            md.attrs["baseline_P_W"] = round(float(base_a["P"]), 1)
            md.attrs["baseline_drift_W"] = round(float(drift), 1)
            md.attrs["recording_summary"] = json.dumps({
                "appliance_label": label,
                "teach_mode": "in_mix",
                "n_samples": int(iso["n"]),
                "duration_s": round(iso["n"] / sr, 2) if sr > 0 else None,
                "device_W_median": round(float(iso["device_W"]), 1),
                "baseline_P_W": round(float(base_a["P"]), 1),
                "baseline_drift_W": round(float(drift), 1),
                "configured_sample_rate_hz": sr,
                "harmonics_enabled": False,
                "completed_utc": datetime.now(timezone.utc).isoformat()})
        return path

    # ---- snapshots -----------------------------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            on = [{"device": nm, **v,
                   "since_iso": (datetime.fromtimestamp(v["since_ms"] / 1000.0)
                                 .astimezone().isoformat(timespec="seconds")
                                 if v.get("since_ms") else None)}
                  for nm, v in sorted(self.state.items()) if v["on"]]
            return {"currently_on": on,
                    "all": self.state,
                    "total_W": self.total_W,
                    "residual_W": self.residual_W,
                    "explained_frac": self.explained_frac,
                    "unknown": self.unknown,
                    "unknown_loads": [
                        {"W": round(u["W"], 1), "since_ms": u["t_ms"],
                         "since_iso": datetime.fromtimestamp(u["t_ms"] / 1000.0)
                         .astimezone().isoformat(timespec="seconds")}
                        for u in self.unknown_claims],
                    "teach_guide": self.guide,
                    "teach_note": self._teach_note,
                    "replay_gt": self._gt_now}

    def chart(self) -> dict:
        with self.lock:
            hist = list(self.history)
        names = self.models.appliances
        out = {"t": [h["t_ms"] for h in hist],
               "P_total": [h["P_total"] for h in hist],
               "residual": [h["residual"] for h in hist],
               "devices": {nm: [h["devices"].get(nm, 0.0) or 0.0 for h in hist]
                           for nm in names}}
        if self._gt is not None and self._gt.get("mode") == "full":
            out["gt"] = {fam: [(h.get("gt") or {}).get(fam, 0.0) for h in hist]
                         for fam in self._gt["families"]}
        return out


# =============================================================================
# Flask app
# =============================================================================
def _np_clean(obj):
    """numpy scalars -> python scalars, recursively (jsonify chokes on np.bool_)."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_np_clean(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _np_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_clean(v) for v in obj]
    return obj


class NumpySafeJSONProvider(pr.SafeJSONProvider):
    def dumps(self, obj, **kwargs):
        return super().dumps(_np_clean(obj), **kwargs)


def create_app(svc: pr.AcquisitionService, engine: LiveEngine,
               models: ModelManager, retrainer: Retrainer) -> Flask:
    app = Flask(__name__)
    app.json = NumpySafeJSONProvider(app)

    @app.route("/")
    def index():
        # inject the canonical family -> color map (dark-surface steps) so the
        # dashboard chart uses the exact same device colors as the
        # measured_scenario_##_decomposition.png figures
        fam_colors = {f: c["dark"] for f, c in nl.FAMILY_COLORS.items()}
        html = LIVE_HTML.replace("__FAMILY_COLORS__", json.dumps(fam_colors))
        return Response(html, mimetype="text/html")

    @app.route("/api/status")
    def api_status():
        s = svc.status()
        s["model"] = models.info()
        s["retrain"] = retrainer.status()
        return jsonify(s)

    @app.route("/api/state")
    def api_state():
        return jsonify(engine.snapshot())

    @app.route("/api/chart")
    def api_chart():
        return jsonify(engine.chart())

    @app.route("/api/events")
    def api_events():
        since = int(request.args.get("since", 0))
        with engine.lock:
            evs = [e for e in engine.events if e["unix_ms"] > since]
        return jsonify({"events": evs[-120:]})

    @app.route("/api/teach", methods=["POST"])
    def api_teach():
        data = request.get_json(silent=True) or {}
        try:
            res = engine.teach(data.get("label", ""),
                               bool(data.get("retrain", True)),
                               str(data.get("mode", "isolated")))
            return jsonify({"ok": True, **res})
        except Exception as e:            # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/retrain", methods=["POST"])
    def api_retrain():
        return jsonify({"ok": retrainer.start(), "status": retrainer.status()})

    @app.route("/api/teach/cancel", methods=["POST"])
    def api_teach_cancel():
        return jsonify({"ok": True, "was_running": engine.cancel_teach()})

    @app.route("/api/model", methods=["POST"])
    def api_model():
        """Switch between the frozen 'original' bundle and the train-on-the-go
        'latest' bundle. The engine rebuilds its device state from the edge
        history automatically after the reload."""
        data = request.get_json(silent=True) or {}
        try:
            info = models.set_variant(str(data.get("variant", "latest")))
            return jsonify({"ok": True, "model": info})
        except Exception as e:            # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/model/reset", methods=["POST"])
    def api_model_reset():
        """Erase all non-original (train-on-the-go) models: the frozen
        *_original snapshots are copied back over the live bundles and
        hot-reloaded; the engine then re-matches its edge history against the
        restored signatures automatically."""
        if retrainer.status()["state"] == "running":
            return jsonify({"ok": False, "error": "retraining is running - "
                            "wait for it to finish first"}), 409
        try:
            res = models.reset_to_original()
            engine._log_event(int(time.time() * 1000), "model_reset", "-",
                              None, None, None, None,
                              detail="non-original models erased; restored "
                                     + ", ".join(res["restored"]))
            return jsonify({"ok": True, **res})
        except Exception as e:            # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/record/start", methods=["POST"])
    def api_record_start():
        data = request.get_json(silent=True) or {}
        try:
            info = svc.start_session(data.get("label", "appliance"))
            return jsonify({"ok": True, "session": info})
        except Exception as e:            # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/record/stop", methods=["POST"])
    def api_record_stop():
        data = request.get_json(silent=True) or {}
        done = svc.stop_session()
        started = False
        if done and (data.get("retrain") or False):
            started = retrainer.start()
        if done:
            models._load_signatures()
        return jsonify({"ok": True, "session": done, "retrain_started": started})

    @app.route("/api/connect", methods=["POST"])
    def api_connect():
        svc.request_connect()
        return jsonify({"ok": True, "state": svc.state})

    @app.route("/api/disconnect", methods=["POST"])
    def api_disconnect():
        svc.request_disconnect()
        return jsonify({"ok": True, "state": svc.state})

    @app.route("/api/sim/load", methods=["POST"])
    def api_sim_load():
        """Demo helper: change the simulated meter's load so events fire."""
        inner = getattr(svc.reader, "inner", svc.reader)
        if not getattr(inner, "is_simulated", False):
            return jsonify({"ok": False, "error": "not in simulate mode"}), 400
        data = request.get_json(silent=True) or {}
        inner.load_level = float(data.get("level", 1.0))
        return jsonify({"ok": True, "level": inner.load_level})

    return app


# =============================================================================
# Dashboard
# =============================================================================
LIVE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live NILM</title>
<style>
  :root{
    --bg:#0e1116; --panel:#171c24; --panel2:#1e2530; --line:#2a3340;
    --txt:#e6edf3; --muted:#8b98a9; --accent:#4aa8ff; --good:#3ecf8e;
    --warn:#f0b429; --bad:#ff5c5c;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       background:var(--bg);color:var(--txt);font-size:14px}
  header{display:flex;align-items:center;gap:12px;padding:12px 18px;background:var(--panel);
         border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10;flex-wrap:wrap}
  header h1{font-size:16px;margin:0;font-weight:600}
  .dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle}
  .pill{padding:4px 10px;border-radius:20px;background:var(--panel2);border:1px solid var(--line);
        font-size:12px;color:var(--muted);white-space:nowrap}
  .pill b{color:var(--txt);font-weight:600}
  .grow{flex:1}
  button{background:var(--panel2);color:var(--txt);border:1px solid var(--line);border-radius:8px;
         padding:7px 13px;font-size:13px;cursor:pointer}
  button:hover{border-color:var(--accent)}
  button.primary{background:var(--accent);border-color:var(--accent);color:#04121f;font-weight:600}
  button.warn{background:#3a2f14;border-color:#6b5716;color:#ffd97a}
  button:disabled{opacity:.4;cursor:not-allowed}
  main{display:grid;grid-template-columns:360px 1fr;gap:14px;padding:14px;align-items:start}
  @media(max-width:980px){main{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:14px}
  .panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin:0 0 10px}
  input[type=text]{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
        color:var(--txt);padding:8px 10px;font-size:13px;width:100%}
  .devcard{display:flex;align-items:center;gap:10px;background:var(--panel2);
           border:1px solid var(--line);border-radius:10px;padding:9px 12px;margin-bottom:8px}
  .devcard .nm{font-weight:600}
  .devcard .w{margin-left:auto;font-variant-numeric:tabular-nums;font-weight:600}
  .devcard .meta{font-size:11px;color:var(--muted)}
  .swatch{width:12px;height:12px;border-radius:3px;flex:none}
  .unknown{border:1px solid #6b5716;background:#241d0b;border-radius:10px;padding:12px;margin-bottom:10px}
  .unknown b{color:var(--warn)}
  .row{display:flex;gap:8px;margin-top:8px}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th,td{padding:5px 8px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
  th{color:var(--muted);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
  td.time{font-variant-numeric:tabular-nums;color:var(--muted)}
  .kind-on{color:var(--good)} .kind-off{color:var(--muted)}
  .kind-unknown{color:var(--warn)} .kind-taught{color:var(--accent)}
  canvas{width:100%;height:280px;display:block}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:12.5px}
  .kv div:nth-child(odd){color:var(--muted)}
  .muted{color:var(--muted)} .small{font-size:12px}
  #retrainbar{display:none;margin-top:8px;font-size:12px;color:var(--warn)}
  .spin{display:inline-block;width:11px;height:11px;border:2px solid var(--warn);
        border-top-color:transparent;border-radius:50%;animation:sp 1s linear infinite;
        vertical-align:-2px;margin-right:6px}
  @keyframes sp{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<header>
  <h1>Live NILM</h1>
  <span class="pill"><span id="conn-dot" class="dot" style="background:var(--bad)"></span><b id="conn-txt">connecting…</b></span>
  <span class="pill">total <b id="p-total">-</b></span>
  <span class="pill">explained <b id="explained">-</b></span>
  <span class="pill">model <b id="model-info">-</b></span>
  <span class="grow"></span>
  <span class="pill" id="sim-pill" style="display:none">simulated meter:
    <button onclick="simLoad(0)" style="padding:2px 8px">0%</button>
    <button onclick="simLoad(1)" style="padding:2px 8px">100%</button>
    <button onclick="simLoad(1.6)" style="padding:2px 8px">160%</button>
  </span>
  <button id="retrain-btn" class="warn" onclick="retrain()">Retrain now</button>
</header>

<main>
  <section>
    <div id="guide-box" class="unknown" style="display:none">
      <b>Teaching - follow the steps</b>
      <div class="small" id="guide-msg" style="margin-top:6px"></div>
      <div class="row"><button onclick="teachCancel()">Cancel</button></div>
    </div>

    <div id="unknown-box" class="unknown" style="display:none">
      <b>Unknown device detected</b>
      <div class="small" style="margin-top:4px">
        ~<span id="unk-w">?</span> W of unexplained power since <span id="unk-since">?</span>.
        What device is this? Name it, then either record it in ISOLATION
        (disconnect everything first - cleanest data) or teach it ON THE GO:
        the other devices keep running and you only toggle this one device off
        and on; its signal is isolated from the mix by baseline subtraction.
      </div>
      <div class="row">
        <input type="text" id="teach-name" placeholder="e.g. kettle">
      </div>
      <div class="row">
        <button class="primary" onclick="teach('isolated')">Teach&nbsp;(isolated)</button>
        <button class="warn" onclick="teach('inmix')">Teach&nbsp;on&nbsp;the&nbsp;go</button>
      </div>
    </div>

    <div class="panel">
      <h2>Currently on</h2>
      <div id="on-list" class="muted small">-</div>
      <div id="retrainbar"><span class="spin"></span><span id="retrain-step">retraining…</span></div>
      <div id="teach-note" class="small muted" style="margin-top:6px"></div>
    </div>

    <div class="panel">
      <h2>Model</h2>
      <select id="model-variant" onchange="setVariant(this.value)"
              style="width:100%;margin-bottom:10px;background:var(--panel2);
                     border:1px solid var(--line);border-radius:8px;
                     color:var(--txt);padding:8px 10px;font-size:13px">
        <option value="latest">train-on-the-go (latest)</option>
      </select>
      <div class="kv" id="model-kv"></div>
      <div class="small muted" id="model-devices" style="margin-top:8px"></div>
      <button class="warn" id="model-reset-btn" onclick="modelReset()"
              style="width:100%;margin-top:10px;display:none">
        Erase retrained models (start clean from original)</button>
    </div>

    <div class="panel">
      <h2>Teach a device by recording it</h2>
      <div class="small muted" style="margin-bottom:8px">
        Plug in ONLY the new device, give it a name, record ~60 s, stop - the
        model retrains automatically with the new device included.
      </div>
      <input type="text" id="rec-name" placeholder="device name, e.g. desk_lamp">
      <div class="row">
        <button class="primary" id="rec-start" onclick="recStart()">Start recording</button>
        <button id="rec-stop" onclick="recStop()" disabled>Stop&nbsp;+&nbsp;retrain</button>
      </div>
      <div class="small muted" id="rec-status" style="margin-top:6px"></div>
    </div>
  </section>

  <section>
    <div class="panel">
      <h2>Live power &amp; per-device estimate</h2>
      <canvas id="chart" width="1200" height="280"></canvas>
      <div id="legend" class="small muted" style="margin-top:6px"></div>
    </div>
    <div class="panel" id="gt-panel" style="display:none">
      <h2>Replay: prediction vs ground truth</h2>
      <div id="gt-summary" class="small muted" style="margin-bottom:8px"></div>
      <table>
        <thead><tr><th></th><th>device</th><th>truth</th><th>predicted</th>
          <th>match</th><th>truth W</th><th>predicted W</th><th>&Delta;W</th></tr></thead>
        <tbody id="gt-body"></tbody>
      </table>
    </div>
    <div class="panel">
      <h2>Event log: what switched, exactly when</h2>
      <div style="max-height:340px;overflow:auto">
        <table>
          <thead><tr><th>time</th><th>event</th><th>device</th><th>ΔP (W)</th><th>ΔQ (var)</th><th>conf</th><th>detail</th></tr></thead>
          <tbody id="ev-body"></tbody>
        </table>
      </div>
    </div>
  </section>
</main>

<script>
// family -> color, injected by live.py from nilm_pipeline.FAMILY_COLORS:
// the SAME map (dark-surface steps of the same hues) that colors the
// measured_scenario_##_decomposition.png figures, so live chart and
// decomposition plots are directly comparable per device.
const FAMILY_COLORS = __FAMILY_COLORS__;
const FALLBACK_FAMS = Object.keys(FAMILY_COLORS);
let lastEvent = 0, recActive = false;

function djb2(s){                       // same 32-bit hash as nilm_pipeline._djb2
  let h = 5381;
  for(const ch of s) h = (h * 33 + ch.codePointAt(0)) >>> 0;
  return h;
}
function col(nm){
  const fam = String(nm).replace(/_\d+$/, "");   // 'table_fan_1' -> 'table_fan'
  if(fam in FAMILY_COLORS) return FAMILY_COLORS[fam];
  // unmapped (newly taught) family: deterministic hash into the same palette,
  // matching nilm_pipeline.family_color, so both charts still agree
  return FAMILY_COLORS[FALLBACK_FAMS[djb2(fam) % FALLBACK_FAMS.length]];
}
async function j(url, opts){ const r = await fetch(url, opts); return r.json(); }
function fmtW(v){ return v==null ? "-" : (Math.abs(v)>=1000 ? (v/1000).toFixed(2)+" kW" : v.toFixed(0)+" W"); }
function hms(iso){ return iso ? iso.substring(11,19) : "-"; }
function hmsMs(iso){ return iso ? iso.substring(11,23) : "-"; }

async function pollStatus(){
  try{
    const s = await j("/api/status");
    const ok = s.state === "connected";
    document.getElementById("conn-dot").style.background = ok ? "var(--good)" : "var(--bad)";
    document.getElementById("conn-txt").textContent =
      (s.simulated ? "simulated" : s.host) + " · " + s.state + " · " + (s.effective_rate_hz||0) + " Hz";
    document.getElementById("sim-pill").style.display = s.simulated ? "" : "none";

    const m = s.model || {};
    const acc = m.holdout_accuracy || {};
    let mi = (m.appliances||[]).length + " devices";
    if(acc.presence_macro_f1 != null) mi += " · F1 " + acc.presence_macro_f1.toFixed(2);
    if(acc.power_mae_W != null) mi += " · ±" + acc.power_mae_W.toFixed(0) + " W";
    document.getElementById("model-info").textContent = m.source === "none" ? "none - teach devices!" : mi;

    const sel = document.getElementById("model-variant");
    if(document.activeElement !== sel){
      const vars = m.variants || ["latest"];
      const opts = vars.map(v => '<option value="'+v+'">'+
        (v==="latest" ? "train-on-the-go (latest)" : "original (frozen)")+
        '</option>').join("");
      if(sel.dataset.opts !== opts){ sel.innerHTML = opts; sel.dataset.opts = opts; }
      sel.value = m.variant || "latest";
    }
    document.getElementById("model-reset-btn").style.display =
      (m.variants||[]).includes("original") ? "" : "none";

    const kv = document.getElementById("model-kv");
    kv.innerHTML = "";
    const add = (k,v)=>{ kv.innerHTML += "<div>"+k+"</div><div>"+v+"</div>"; };
    add("bundle", m.source||"-");
    add("held-out presence F1", acc.presence_macro_f1!=null ? acc.presence_macro_f1.toFixed(3) : "-");
    add("held-out power MAE", acc.power_mae_W!=null ? acc.power_mae_W.toFixed(1)+" W" : "-");
    add("window", (m.window_s||"-")+" s");
    add("trained", m.trained_utc ? m.trained_utc.replace("T"," ").substring(0,19) : "-");
    add("signatures", m.n_signatures);
    document.getElementById("model-devices").textContent =
      (m.appliances||[]).length ? "knows: " + m.appliances.join(", ") : "no devices yet";

    const rt = s.retrain || {};
    const bar = document.getElementById("retrainbar");
    if(rt.state === "running"){
      bar.style.display = "block";
      document.getElementById("retrain-step").textContent =
        "training on the go: " + rt.step + (rt.elapsed_s ? " ("+rt.elapsed_s.toFixed(0)+" s)" : "");
      document.getElementById("retrain-btn").disabled = true;
    } else {
      bar.style.display = "none";
      document.getElementById("retrain-btn").disabled = false;
      if(rt.state === "error") document.getElementById("teach-note").textContent =
        "retrain FAILED - see console/log";
    }
    const sess = s.session;
    recActive = !!sess;
    document.getElementById("rec-start").disabled = !!sess;
    document.getElementById("rec-stop").disabled = !sess;
    document.getElementById("rec-status").textContent = sess ?
      ("recording '"+sess.label+"' - "+sess.samples+" samples") : "";
  }catch(e){}
}

async function pollState(){
  try{
    const st = await j("/api/state");
    document.getElementById("p-total").textContent = fmtW(st.total_W);
    document.getElementById("explained").textContent =
      st.explained_frac!=null ? (100*st.explained_frac).toFixed(0)+"%" : "-";

    const box = document.getElementById("on-list");
    let cards = st.currently_on.map(d =>
      '<div class="devcard"><span class="swatch" style="background:'+col(d.device)+'"></span>'+
      '<div><div class="nm">'+d.device+'</div><div class="meta">since '+hms(d.since_iso)+
      ' · conf '+(100*d.prob).toFixed(0)+'%'+(d.src==="edge" ? ' · edge' : '')+'</div></div>'+
      '<span class="w">'+fmtW(d.power_W)+'</span></div>').join("");
    cards += (st.unknown_loads||[]).map(u =>
      '<div class="devcard" style="border-color:#6b5716"><span class="swatch" style="background:var(--warn)"></span>'+
      '<div><div class="nm" style="color:var(--warn)">unknown device</div>'+
      '<div class="meta">since '+hms(u.since_iso)+' · switch-on matched no signature</div></div>'+
      '<span class="w">'+fmtW(u.W)+'</span></div>').join("");
    if(!cards){
      box.textContent = "nothing recognized as ON";
    } else {
      box.innerHTML = cards;
    }
    if(Math.abs(st.residual_W) > 1 && st.currently_on.length){
      box.innerHTML += '<div class="devcard" style="opacity:.7"><span class="swatch" style="background:#555"></span>'+
        '<div><div class="nm muted">unassigned residual</div></div><span class="w">'+fmtW(st.residual_W)+'</span></div>';
    }
    const gb = document.getElementById("guide-box");
    if(st.teach_guide){
      gb.style.display = "block";
      document.getElementById("guide-msg").textContent = st.teach_guide.msg;
    } else gb.style.display = "none";
    const ub = document.getElementById("unknown-box");
    if(st.unknown && !st.teach_guide){
      ub.style.display = "block";
      document.getElementById("unk-w").textContent = st.unknown.typical_W;
      document.getElementById("unk-since").textContent = hms(st.unknown.since_iso);
    } else ub.style.display = "none";
    if(st.teach_note) document.getElementById("teach-note").textContent = st.teach_note;
    renderGt(st.replay_gt);
  }catch(e){}
}

function renderGt(g){
  const gp = document.getElementById("gt-panel");
  if(!g){ gp.style.display = "none"; return; }
  gp.style.display = "block";
  const m = g.metrics || {};
  let sum;
  if(g.mode === "full"){
    sum = "scenario ground truth · since replay start: presence accuracy " +
      (m.presence_accuracy!=null ? (100*m.presence_accuracy).toFixed(0)+"%" : "-") +
      " · power MAE " + (m.power_mae_W!=null ? m.power_mae_W.toFixed(0)+" W" : "-") +
      " · dashed lines on the chart = ground truth";
  } else {
    sum = "recording '"+(g.label||"")+"' · expected-device-set F1 " +
      (m.set_f1!=null ? m.set_f1.toFixed(2) : "-") +
      " over " + (m.scored_strides||0) + " scored strides" +
      (m.active===false ? " · paused: nothing drawing power" : "");
  }
  document.getElementById("gt-summary").textContent = sum;
  document.getElementById("gt-body").innerHTML = (g.devices||[]).map(d => {
    const ok = d.gt_on === d.pred_on;
    const dw = (d.gt_W!=null && d.pred_W!=null) ? d.pred_W - d.gt_W : null;
    return '<tr><td><span class="swatch" style="background:'+col(d.device)+
      ';display:inline-block"></span></td>'+
      '<td><b>'+d.device+'</b></td>'+
      '<td>'+(d.gt_on ? 'ON' : 'off')+'</td>'+
      '<td>'+(d.pred_on ? 'ON' : 'off')+'</td>'+
      '<td class="'+(ok ? 'kind-on' : 'kind-unknown')+'">'+(ok ? '✓' : '✗')+'</td>'+
      '<td>'+(d.gt_W==null ? '-' : fmtW(d.gt_W))+'</td>'+
      '<td>'+(d.pred_W==null ? '-' : fmtW(d.pred_W))+'</td>'+
      '<td>'+(dw==null ? '-' : (dw>=0?'+':'')+fmtW(dw))+'</td></tr>';
  }).join("");
}

async function pollChart(){
  try{
    const c = await j("/api/chart");
    drawChart(c);
  }catch(e){}
}

function drawChart(c){
  const cv = document.getElementById("chart"), ctx = cv.getContext("2d");
  const W = cv.width = cv.clientWidth * (window.devicePixelRatio||1);
  const H = cv.height;
  ctx.clearRect(0,0,W,H);
  if(!c.t || c.t.length < 2) return;
  const names = Object.keys(c.devices||{});
  const n = c.t.length;
  let stacked = new Array(n).fill(0);
  const stacks = names.map(nm => {
    const s = c.devices[nm].map((v,i)=> stacked[i] += Math.max(0, v||0));
    return {nm, top: s.slice()};
  });
  // replay ground truth (same stacking order as the prediction for shared names)
  const gtNames = c.gt ? [...names.filter(nm => nm in c.gt),
                          ...Object.keys(c.gt).filter(nm => !names.includes(nm))] : [];
  let gtTop = new Array(n).fill(0);
  gtNames.forEach(nm => { const a = c.gt[nm]||[];
    for(let i=0;i<n;i++) gtTop[i] += Math.max(0, a[i]||0); });
  let ymax = Math.max(...c.P_total.map(Math.abs), ...stacked,
                      ...(gtNames.length ? gtTop : [0]), 10) * 1.15;
  let ymin = Math.min(0, ...c.P_total) * 1.15;
  const X = i => 40 + (W-50) * i/(n-1);
  const Y = v => H - 22 - (H-34) * (v - ymin) / (ymax - ymin);
  // grid
  ctx.strokeStyle = "#242d3a"; ctx.fillStyle = "#8b98a9"; ctx.font = "10px sans-serif";
  for(let g=0; g<=4; g++){
    const v = ymin + (ymax-ymin)*g/4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(40,y); ctx.lineTo(W-10,y); ctx.stroke();
    ctx.fillText(Math.round(v), 4, y+3);
  }
  // stacked device areas
  let prev = new Array(n).fill(0);
  stacks.forEach(s => {
    ctx.beginPath();
    for(let i=0;i<n;i++) ctx.lineTo(X(i), Y(s.top[i]));
    for(let i=n-1;i>=0;i--) ctx.lineTo(X(i), Y(prev[i]));
    ctx.closePath();
    ctx.fillStyle = col(s.nm) + "66";
    ctx.fill();
    prev = s.top;
  });
  // replay ground truth: dashed cumulative stack in the same device colors,
  // so each dashed line should hug the top of its device's filled area
  if(gtNames.length){
    let acc = new Array(n).fill(0);
    ctx.setLineDash([6,4]);
    gtNames.forEach(nm => {
      const a = c.gt[nm]||[];
      ctx.beginPath(); ctx.strokeStyle = col(nm); ctx.lineWidth = 1.4;
      for(let i=0;i<n;i++){ acc[i] += Math.max(0, a[i]||0); ctx.lineTo(X(i), Y(acc[i])); }
      ctx.stroke();
    });
    ctx.setLineDash([]);
  }
  // measured total
  ctx.beginPath(); ctx.strokeStyle = "#e6edf3"; ctx.lineWidth = 1.6;
  for(let i=0;i<n;i++) ctx.lineTo(X(i), Y(c.P_total[i]));
  ctx.stroke();
  // time labels
  const t0 = new Date(c.t[0]), t1 = new Date(c.t[n-1]);
  ctx.fillText(t0.toTimeString().substring(0,8), 42, H-8);
  ctx.fillText(t1.toTimeString().substring(0,8), W-70, H-8);
  document.getElementById("legend").innerHTML =
    '<span style="color:#e6edf3">━ measured total</span> · ' +
    names.map(nm=>'<span style="color:'+col(nm)+'">■ '+nm+'</span>').join(" · ") +
    (gtNames.length ? ' · <span class="muted">╌╌ ground truth (replay)</span>' : '');
}

async function pollEvents(){
  try{
    const r = await j("/api/events?since="+lastEvent);
    if(!r.events.length) return;
    const tb = document.getElementById("ev-body");
    r.events.forEach(e => {
      lastEvent = Math.max(lastEvent, e.unix_ms);
      const tr = document.createElement("tr");
      let cls = "";
      if(e.kind.includes("on") && !e.kind.includes("unknown")) cls = "kind-on";
      if(e.kind.includes("off")) cls = "kind-off";
      if(e.kind.includes("unknown")) cls = "kind-unknown";
      if(e.kind.includes("fail") || e.kind.includes("warning") || e.kind.includes("veto")) cls = "kind-unknown";
      if(e.kind === "mode_change") cls = "kind-on";
      if(e.kind === "taught" || e.kind === "residual_matched") cls = "kind-taught";
      tr.innerHTML = '<td class="time">'+hmsMs(e.time_iso)+'</td><td class="'+cls+'">'+e.kind+
        '</td><td><b>'+e.device+'</b></td><td>'+(e.dP_W==null?"":e.dP_W)+
        '</td><td>'+(e.dQ_var==null?"":e.dQ_var)+'</td><td>'+(e.confidence==null?"":e.confidence)+
        '</td><td class="muted">'+(e.detail||"")+'</td>';
      tb.insertBefore(tr, tb.firstChild);
    });
    while(tb.children.length > 150) tb.removeChild(tb.lastChild);
  }catch(e){}
}

async function teach(mode){
  const name = document.getElementById("teach-name").value.trim();
  if(!name) return alert("give the device a name first");
  const r = await j("/api/teach", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({label:name, retrain:true, mode:mode||"isolated"})});
  if(!r.ok) alert("teach failed: " + r.error);
  else document.getElementById("teach-name").value = "";
}
async function teachCancel(){ await j("/api/teach/cancel", {method:"POST"}); }
async function retrain(){ await j("/api/retrain", {method:"POST"}); }
async function setVariant(v){
  const r = await j("/api/model", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({variant:v})});
  if(!r.ok) alert("model switch failed: " + r.error);
}
async function modelReset(){
  if(!confirm("Erase all retrained (non-original) models and restore the "+
              "frozen original snapshot?\n\nRecordings are kept - taught "+
              "devices re-enter the model on the next retrain.")) return;
  const r = await j("/api/model/reset", {method:"POST"});
  if(!r.ok) alert("reset failed: " + r.error);
}
async function simLoad(l){ await j("/api/sim/load", {method:"POST",
  headers:{"Content-Type":"application/json"}, body: JSON.stringify({level:l})}); }
async function recStart(){
  const name = document.getElementById("rec-name").value.trim();
  if(!name) return alert("give the device a name first");
  const r = await j("/api/record/start", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({label:name})});
  if(!r.ok) alert("record failed: " + r.error);
}
async function recStop(){
  await j("/api/record/stop", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({retrain:true})});
}

setInterval(pollStatus, 2000); setInterval(pollState, 1000);
setInterval(pollChart, 1500); setInterval(pollEvents, 1500);
pollStatus(); pollState(); pollChart(); pollEvents();
</script>
</body>
</html>"""


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Live NILM monitor: recognize devices "
                                            "on a live PAC4200 feed and learn new "
                                            "ones on the go.")
    p.add_argument("--host", default=None, help="PAC4200 IP (omit for --simulate)")
    p.add_argument("--port", type=int, default=502)
    p.add_argument("--unit-id", type=int, default=1)
    p.add_argument("--rate", type=float, default=pr.DEFAULT_SAMPLE_RATE_HZ)
    p.add_argument("--simulate", action="store_true", help="synthetic meter (no hardware)")
    p.add_argument("--replay", default=None, metavar="FILE",
                   help="replay a pre-measured file (.h5 recording/scenario or "
                        "PAC4200 .csv) as if it were the live meter -- no "
                        "hardware needed; plays at the file's own sample rate "
                        "(--rate is ignored)")
    p.add_argument("--replay-speed", type=float, default=1.0,
                   help="replay speed factor (2 = twice as fast; note that "
                        "wall-clock windows then span 2x the recorded time, "
                        "so use 1.0 when judging accuracy)")
    p.add_argument("--replay-loop", action="store_true",
                   help="restart the replay from the beginning when it ends "
                        "(default: freeze the dashboard at the final state)")
    p.add_argument("--no-harmonics", action="store_true",
                   help="skip per-order harmonic reads (THD_I then unavailable live)")
    p.add_argument("--stride", type=float, default=2.0,
                   help="re-evaluate the model every N seconds")
    p.add_argument("--models-dir", default=os.path.join(HERE, "output"))
    p.add_argument("--recordings-dir",
                   default=os.path.join(READER_DIR, "recordings"),
                   help="where taught/recorded devices are stored (and signatures read)")
    p.add_argument("--scenarios-dir",
                   default=os.path.join(AGG_DIR, "measured_scenarios"),
                   help="where retraining writes its mixed training scenarios")
    p.add_argument("--retrain-window", type=float, default=10.0,
                   help="window (s) used when retraining on the go")
    p.add_argument("--on-w", type=float, default=5.0,
                   help="presence ON threshold (W) used when retraining")
    p.add_argument("--unknown-min-w", type=float, default=30.0,
                   help="unexplained power (W) that triggers the unknown-device prompt")
    p.add_argument("--teach-record-s", type=float, default=45.0,
                   help="teach: seconds of ON time to record (guided flow) "
                        "before retraining")
    p.add_argument("--mode-min-w", type=float, default=3.5,
                   help="settled steps above this (but below the edge "
                        "threshold) are matched as MODE changes of an "
                        "already-on device (fan high -> low)")
    p.add_argument("--big-edge-w", type=float, default=120.0,
                   help="families whose smallest signature exceeds this can "
                        "only switch ON via a real step, never by window-"
                        "model vote alone (kills phantom boiler/coffee)")
    p.add_argument("--min-conf", type=float, default=0.70,
                   help="minimum confidence to NAME a matched appliance; below "
                        "this a step/vote is reported as an unknown load "
                        "instead of a low-confidence guess (default 0.70)")
    p.add_argument("--ih-matching", action="store_true",
                   help="EXPERIMENTAL: include the harmonic-current term in "
                        "edge matching (off by default: the meter's ~2.3%% "
                        "THD floor and cycle-varying devices made it veto "
                        "correct matches; see ModelManager.IH_MIN_STEP note)")
    p.add_argument("--web-port", type=int, default=8300)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    if args.simulate and args.replay:
        p.error("--simulate and --replay are mutually exclusive")
    if not args.simulate and not args.replay and args.host is None:
        p.error("--host is required unless --simulate or --replay")

    harmonics = not args.no_harmonics
    if args.replay:
        harmonics = False                 # THD_I comes from the file as a scalar
        inner = ReplayReader(args.replay, loop=args.replay_loop)
        args.rate = inner.sample_rate_hz * max(0.1, float(args.replay_speed))
        dur = inner.n_samples / inner.sample_rate_hz
        print(f"Live NILM REPLAYING {os.path.basename(args.replay)} "
              f"('{inner.label}': {inner.n_samples} samples, {dur:.0f} s @ "
              f"{inner.sample_rate_hz:g} Hz, speed x{args.replay_speed:g}"
              f"{', looping' if args.replay_loop else ''})")
        if inner.gt is not None:
            what = ("full per-device ground truth (scenario file)"
                    if inner.gt["mode"] == "full" else
                    "expected devices from the label: "
                    + ", ".join(inner.gt["expected"]))
            print(f"  ground truth found - {what}; the dashboard compares "
                  "predictions against it live")
    elif args.simulate:
        inner = pr.SimulatedReader(extra_channels=pr.EXTENDED_CHANNELS,
                                   read_harmonics=harmonics)
        print("Live NILM with a SIMULATED meter (demo mode, no hardware).")
    else:
        inner = pr.ModbusReader(host=args.host, port=args.port, unit_id=args.unit_id,
                                extra_channels=pr.EXTENDED_CHANNELS,
                                read_harmonics=harmonics)
        print(f"Live NILM on meter {args.host}:{args.port}")

    reader = ThdReader(inner)
    session_dir = os.path.join(args.models_dir,
                               f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    svc = pr.AcquisitionService(reader, args.rate, args.recordings_dir,
                                write_harmonics=harmonics)
    models = ModelManager(args.models_dir, args.recordings_dir)
    models.use_ih = bool(args.ih_matching)
    retrainer = Retrainer(models, args.scenarios_dir,
                          window_s=args.retrain_window, on_w=args.on_w)
    engine = LiveEngine(svc, models, retrainer, session_dir,
                        stride_s=args.stride, unknown_min_W=args.unknown_min_w,
                        teach_record_s=args.teach_record_s,
                        mode_min_W=args.mode_min_w,
                        big_edge_min_W=args.big_edge_w,
                        min_conf=args.min_conf)

    info = models.info()
    if info["source"] == "none":
        print("NOTE: no trained model found - the dashboard still shows the live "
              "signal and edge events; teach/record devices and hit Retrain.")
    else:
        print(f"model: {info['source']}  ({len(info['appliances'])} devices, "
              f"held-out {info['holdout_accuracy']})")
    print(f"events + session output -> {session_dir}")

    svc.start()
    svc.request_connect()
    engine.start()

    if args.replay and not args.replay_loop:
        def _end_replay():
            while not inner.finished:
                time.sleep(0.5)
            svc.request_disconnect()
            print(f"replay of {os.path.basename(args.replay)} finished -- "
                  "dashboard frozen at the final state (Ctrl-C to quit)",
                  flush=True)
        threading.Thread(target=_end_replay, daemon=True,
                         name="replay-end").start()

    app = create_app(svc, engine, models, retrainer)
    url = f"http://127.0.0.1:{args.web_port}/"
    print(f"\nLive dashboard:  {url}\nCtrl-C to quit.\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=args.web_port, threaded=True,
                debug=False, use_reloader=False)
    finally:
        print("\nshutting down…")
        engine.stop()
        svc.shutdown()


if __name__ == "__main__":
    main()
