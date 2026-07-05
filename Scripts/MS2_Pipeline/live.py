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
       -- what is this?"  ->  you type a name  ->  the captured signature is
       saved as a labelled recording  ->  scenarios are rebuilt and the model
       is RETRAINED in the background  ->  hot-reloaded. Training on the go:
       the next time that device runs, the system knows it.

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
        if self._n < 2:
            raise ValueError(f"{self.path}: too few samples to replay")

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
        s = pr.Sample(timestamp_us=int(time.time() * 1e6))
        s.scalars = {
            "P_total": float(self._P[i]), "Q_total": float(self._Q[i]),
            "S_total": float(self._S[i]), "PF_total": float(self._PF[i]),
            "P_L1": float(self._Pph[i, 0]), "P_L2": float(self._Pph[i, 1]),
            "P_L3": float(self._Pph[i, 2]),
            "THD_I_L1": float(self._thd[i]),
        }
        return s


# =============================================================================
# Model manager (hot-reloadable) + device signature table
# =============================================================================
class ModelManager:
    """Loads the mix bundle (preferred) or the presence+disaggregate pair, and
    a per-device (P, Q) signature table from the single-appliance recordings.
    reload() picks up whatever a background retrain just wrote."""

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
        self.signatures: list = []       # [{family, label, P, Q}]
        self.reload()

    def reload(self) -> dict:
        with self.lock:
            self._load_models()
            self._load_signatures()
            self.loaded_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return self.info()

    def _load_models(self):
        self.presence = self.power = None
        self.appliances, self.metrics, self.source = [], {}, "none"
        mix = os.path.join(self.models_dir, "model_mix.joblib")
        if os.path.exists(mix):
            b = joblib.load(mix)
            self.presence, self.power = b["presence"], b["power"]
            self.appliances = list(b["appliances"])
            self.features = list(b.get("features", []) or [])
            self.window_s = float(b.get("window_s", 10.0))
            self.on_W = float(b.get("on_W", 5.0))
            self.metrics = b.get("metrics", {}) or {}
            self.source = "model_mix.joblib"
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
        """Steady-state (P, Q) per single-device recording, for edge matching."""
        sigs = []
        for p in sorted(glob.glob(os.path.join(self.recordings_dir, "*.h5"))):
            try:
                with h5py.File(p, "r") as f:
                    lab = f["metadata"].attrs.get("appliance_label", "")
                    lab = lab.decode() if isinstance(lab, (bytes, bytearray)) else str(lab)
                    if not lab or nl.is_mixed_label(lab):
                        continue
                    P = np.nan_to_num(f["measurements/P_total"][:])
                    Q = np.nan_to_num(f["measurements/Q_total"][:])
            except (OSError, KeyError):
                continue
            on = np.abs(P) > 3.0
            if len(P) < 25 or not on.any():
                continue
            sigs.append({"family": nl.parse_family(lab), "label": lab,
                         "P": float(np.median(P[on])), "Q": float(np.median(Q[on]))})
        self.signatures = sigs

    def match_edge(self, dP: float, dQ: float):
        """Nearest device signature for a power step; None when nothing is close."""
        with self.lock:
            best, best_d = None, 1.0
            for s in self.signatures:
                tol = max(20.0, 0.35 * math.hypot(s["P"], s["Q"]))
                d = math.hypot(dP - s["P"], dQ - s["Q"]) / tol
                if d < best_d:
                    best, best_d = s, d
            if best is None:
                return None
            return {"family": best["family"], "label": best["label"],
                    "confidence": round(max(0.0, 1.0 - best_d), 2)}

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
                    "loaded_utc": self.loaded_utc}


# =============================================================================
# Retrainer -- "training on the go"
# =============================================================================
class Retrainer:
    """Background rebuild of scenarios + retrain of the mix (and identify)
    models from everything in the recordings folder, then hot-reload."""

    def __init__(self, models: ModelManager, scenarios_dir: str,
                 window_s: float = 10.0, on_w: float = 5.0,
                 n_scenarios: int = 24, scenario_duration: float = 300.0):
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
# Live NILM engine
# =============================================================================
class LiveEngine:
    """Consumes the acquisition ring buffer; every `stride_s` re-evaluates the
    model on the trailing window; detects edges; tracks who-is-on-since-when,
    the unexplained residual, and the unknown-device state."""

    def __init__(self, svc: pr.AcquisitionService, models: ModelManager,
                 retrainer: Retrainer, out_dir: str, stride_s: float = 2.0,
                 unknown_min_W: float = 30.0, unknown_frac: float = 0.15,
                 unknown_persist_s: float = 8.0, edge_min_W: float = 8.0):
        self.svc = svc
        self.models = models
        self.retrainer = retrainer
        self.stride_s = stride_s
        self.unknown_min_W = unknown_min_W
        self.unknown_frac = unknown_frac
        self.unknown_persist_s = unknown_persist_s
        self.edge_min_W = edge_min_W

        self.lock = threading.RLock()
        self.state: dict = {}            # family -> {on, prob, power_W, since_ms}
        self.smooth: deque = deque(maxlen=3)     # recent proba vectors
        self.residual_W = 0.0
        self.total_W = 0.0
        self.explained_frac = 1.0
        self.history: deque = deque(maxlen=420)  # per-stride snapshots for the chart
        self.events: list = []           # full session event log
        self.unknown: dict | None = None
        self._unknown_first: float | None = None
        self._last_edge_ms = 0
        self._teach_note = ""

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
                "THD": thd}

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
        return nl.Signal(source="live", name="live", sample_rate_hz=self.svc.sample_rate_hz,
                         t=t, P=P, Q=Q, S=S, PF=PF, THD_I=arrs["THD"][sl], P_phase=Pph)

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
        edge (exact sample timestamp) or None. Runs on every stride."""
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
        if abs(dP) < self.edge_min_W and abs(dQ) < 2 * self.edge_min_W:
            return None
        # both sides must be settled so a ramp isn't logged sample-by-sample
        if np.nanstd(P8[:k]) > max(6.0, 0.05 * abs(pre_P)):
            return None
        if np.nanstd(P8[-k:]) > max(6.0, 0.05 * abs(post_P)):
            return None
        mid = P8[k:-k]
        if len(mid) == 0:
            return None
        j = int(np.nanargmax(np.abs(np.diff(mid)))) if len(mid) > 1 else 0
        t_edge = int(t8[k + j])
        if t_edge - self._last_edge_ms < 3000:      # debounce
            return None
        self._last_edge_ms = t_edge
        return {"t_ms": t_edge, "dP": float(dP), "dQ": float(dQ),
                "P_after": float(post_P)}

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

        # -- edge first (needs only the raw signal, works even with no model) --
        edge = self._detect_edge(arrs)
        if edge is not None:
            match = m.match_edge(edge["dP"], edge["dQ"]) if abs(edge["dP"]) >= self.edge_min_W else None
            direction = "on" if edge["dP"] > 0 else "off"
            if match and match["confidence"] >= 0.25:
                dev, conf = match["family"], match["confidence"]
            elif m.match_edge(-edge["dP"], -edge["dQ"]) and edge["dP"] < 0:
                neg = m.match_edge(-edge["dP"], -edge["dQ"])
                dev, conf = neg["family"], neg["confidence"]
            else:
                dev, conf = "unrecognized", None
            self._log_event(edge["t_ms"], f"edge_{direction}", dev, conf,
                            edge["dP"], edge["dQ"], edge["P_after"],
                            detail="step matched to device signature" if dev != "unrecognized"
                                   else "step matches no known device signature")

        sr = self.svc.sample_rate_hz
        w_samples = max(1, int(round(ws * sr)))
        sig = self._window_signal(arrs, w_samples)
        if sig is None:
            return
        now_ms = int(arrs["t_ms"][-1])
        total_W = float(np.nanmean(sig.P))

        on_map, power_map, prob_map = {}, {}, {}
        explained = 0.0
        if presence is not None and names:
            X, _, _ = nl.aggregate_windows(sig, ws, canon=names)
            with m.lock:
                X = nl.slice_features(X, m.features or None)
            X = X[-1:].copy()
            proba = nl.presence_proba(presence, X)[0]
            with self.lock:
                self.smooth.append(proba)
                proba_s = np.median(np.vstack(list(self.smooth)), axis=0)
            watts = power.predict(X)[0] if power is not None else np.zeros(len(names))
            for i, nm in enumerate(names):
                prev_on = self.state.get(nm, {}).get("on", False)
                # hysteresis so a 0.5-ish probability doesn't flap on/off
                on = bool(proba_s[i] >= (0.45 if prev_on else 0.55))
                w = float(watts[i]) if on else 0.0
                if on and abs(w) < 0.5 and power is None:
                    w = float("nan")
                on_map[nm], power_map[nm], prob_map[nm] = on, w, float(proba_s[i])
                if on and math.isfinite(w):
                    explained += w

        residual = total_W - explained
        # -- state transitions -> events -------------------------------------
        with self.lock:
            prev = self.state
            new_state = {}
            for nm in names:
                p = prev.get(nm, {})
                on = on_map.get(nm, False)
                since = p.get("since_ms")
                if on and not p.get("on", False):
                    since = now_ms - int(ws * 1000 / 2)      # window centre-ish
                    # a recent matching edge gives the exact switch-on moment
                    for ev in reversed(self.events[-12:]):
                        if (ev["device"] == nm and ev["kind"].startswith("edge_on")
                                and now_ms - ev["unix_ms"] < (ws + 6) * 1000):
                            since = ev["unix_ms"]; break
                    self._log_event(since, "device_on", nm, prob_map.get(nm),
                                    None, None, total_W,
                                    detail=f"~{power_map.get(nm, 0):.0f} W")
                elif not on and p.get("on", False):
                    self._log_event(now_ms - int(ws * 1000 / 2), "device_off", nm,
                                    prob_map.get(nm), None, None, total_W)
                    since = None
                new_state[nm] = {"on": on, "prob": round(prob_map.get(nm, 0.0), 3),
                                 "power_W": None if not math.isfinite(power_map.get(nm, 0.0))
                                 else round(power_map.get(nm, 0.0), 1),
                                 "since_ms": since}
            self.state = new_state
            self.total_W = round(total_W, 1)
            self.residual_W = round(residual, 1)
            self.explained_frac = round(1.0 - min(1.0, abs(residual) / max(abs(total_W), 1e-9)), 3) \
                if abs(total_W) > 1 else 1.0
            self.history.append({
                "t_ms": now_ms, "P_total": round(total_W, 1),
                "residual": round(residual, 1),
                "devices": {nm: (new_state[nm]["power_W"] if new_state[nm]["on"] else 0.0)
                            for nm in names}})

        # -- unknown-device monitor -------------------------------------------
        threshold = max(self.unknown_min_W, self.unknown_frac * abs(total_W))
        over = abs(residual) > threshold and self.retrainer.status()["state"] != "running"
        with self.lock:
            if over:
                if self._unknown_first is None:
                    self._unknown_first = now_ms
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
            else:
                self._unknown_first = None
                if self.unknown is not None:
                    self._log_event(now_ms, "unknown_cleared", "unknown", None,
                                    residual, None, total_W)
                self.unknown = None
            if self.unknown is not None:
                self.unknown["typical_W"] = round(residual, 1)

    # ---- teach: save the unknown signature as a labelled recording ----------
    def teach(self, label: str, retrain: bool = True) -> dict:
        label = (label or "").strip()
        if not label:
            raise ValueError("empty device name")
        arrs = self._buffer_arrays()
        if arrs is None:
            raise RuntimeError("no live data buffered yet")
        with self.lock:
            unk = dict(self.unknown) if self.unknown else None
        now_ms = int(arrs["t_ms"][-1])
        start_ms = (unk["since_ms"] - 4000) if unk else now_ms - 45000
        sr = self.svc.sample_rate_hz

        t = arrs["t_ms"]
        seg = t >= start_ms
        if seg.sum() < 5 * sr:
            raise RuntimeError("unknown segment too short to save")
        # baseline = what the OTHER devices were drawing just before it appeared
        base_win = (t >= start_ms - 12000) & (t < start_ms - 1000)
        def base(a):
            return float(np.nanmedian(a[base_win])) if base_win.any() else 0.0
        dP = np.nan_to_num(arrs["P"][seg] - base(arrs["P"]))
        dQ = np.nan_to_num(arrs["Q"][seg] - base(arrs["Q"]))
        dphs = [np.nan_to_num(arrs[k][seg] - base(arrs[k])) for k in ("P1", "P2", "P3")]
        S = np.hypot(dP, dQ)
        PF = np.divide(dP, S, out=np.ones_like(dP), where=S > 1e-6)

        safe = pr._safe_label(label)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.models.recordings_dir, f"{safe}_{ts}.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("timestamp", data=(t[seg] * 1000).astype(np.int64),
                             compression="lzf")
            m = f.create_group("measurements")
            m.create_dataset("P_total", data=dP.astype(np.float32), compression="lzf")
            m.create_dataset("Q_total", data=dQ.astype(np.float32), compression="lzf")
            for i, ph in enumerate(("L1", "L2", "L3")):
                m.create_dataset(f"P_{ph}", data=dphs[i].astype(np.float32),
                                 compression="lzf")
            m.create_dataset("S_total", data=S.astype(np.float32), compression="lzf")
            m.create_dataset("PF_total", data=PF.astype(np.float32), compression="lzf")
            md = f.create_group("metadata")
            md.attrs["format_version"] = pr.FORMAT_VERSION
            md.attrs["sample_rate_hz"] = float(sr)
            md.attrs["anchor_datetime"] = datetime.fromtimestamp(
                start_ms / 1000.0, tz=timezone.utc).isoformat()
            md.attrs["source"] = "live_teach_delta"
            md.attrs["appliance_label"] = label
            md.attrs["note"] = ("baseline-subtracted residual segment captured live; "
                                "other steady loads were removed by subtraction")

        n = int(seg.sum())
        self._log_event(now_ms, "taught", nl.parse_family(label), None,
                        float(np.median(dP)), float(np.median(dQ)), None,
                        detail=f"saved {n} samples to {os.path.basename(path)}"
                               + ("; retraining" if retrain else ""))
        with self.lock:
            self.unknown = None
            self._unknown_first = None
            self._teach_note = f"saved {os.path.basename(path)}"
        started = self.retrainer.start() if retrain else False
        return {"file": path, "samples": n, "retrain_started": bool(started)}

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
                    "teach_note": self._teach_note}

    def chart(self) -> dict:
        with self.lock:
            hist = list(self.history)
        names = self.models.appliances
        return {"t": [h["t_ms"] for h in hist],
                "P_total": [h["P_total"] for h in hist],
                "residual": [h["residual"] for h in hist],
                "devices": {nm: [h["devices"].get(nm, 0.0) or 0.0 for h in hist]
                            for nm in names}}


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
        return Response(LIVE_HTML, mimetype="text/html")

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
            res = engine.teach(data.get("label", ""), bool(data.get("retrain", True)))
            return jsonify({"ok": True, **res})
        except Exception as e:            # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/retrain", methods=["POST"])
    def api_retrain():
        return jsonify({"ok": retrainer.start(), "status": retrainer.status()})

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
    <div id="unknown-box" class="unknown" style="display:none">
      <b>Unknown device detected</b>
      <div class="small" style="margin-top:4px">
        ~<span id="unk-w">?</span> W of unexplained power since <span id="unk-since">?</span>.
        What device is this?
      </div>
      <div class="row">
        <input type="text" id="teach-name" placeholder="e.g. kettle">
        <button class="primary" onclick="teach()">Teach&nbsp;+&nbsp;retrain</button>
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
      <div class="kv" id="model-kv"></div>
      <div class="small muted" id="model-devices" style="margin-top:8px"></div>
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
const COLORS = ["#4aa8ff","#3ecf8e","#f0b429","#c084fc","#ff8e5c","#5cd6ff",
                "#ff5c8a","#a3e635","#e07b39","#94a3b8"];
let deviceColor = {}, lastEvent = 0, recActive = false;

function col(nm){
  if(!(nm in deviceColor)) deviceColor[nm] = COLORS[Object.keys(deviceColor).length % COLORS.length];
  return deviceColor[nm];
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
    if(!st.currently_on.length){
      box.textContent = "nothing recognized as ON";
    } else {
      box.innerHTML = st.currently_on.map(d =>
        '<div class="devcard"><span class="swatch" style="background:'+col(d.device)+'"></span>'+
        '<div><div class="nm">'+d.device+'</div><div class="meta">since '+hms(d.since_iso)+
        ' · conf '+(100*d.prob).toFixed(0)+'%</div></div>'+
        '<span class="w">'+fmtW(d.power_W)+'</span></div>').join("");
    }
    if(Math.abs(st.residual_W) > 1 && st.currently_on.length){
      box.innerHTML += '<div class="devcard" style="opacity:.7"><span class="swatch" style="background:#555"></span>'+
        '<div><div class="nm muted">unassigned residual</div></div><span class="w">'+fmtW(st.residual_W)+'</span></div>';
    }
    const ub = document.getElementById("unknown-box");
    if(st.unknown){
      ub.style.display = "block";
      document.getElementById("unk-w").textContent = st.unknown.typical_W;
      document.getElementById("unk-since").textContent = hms(st.unknown.since_iso);
    } else ub.style.display = "none";
    if(st.teach_note) document.getElementById("teach-note").textContent = st.teach_note;
  }catch(e){}
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
  let ymax = Math.max(...c.P_total.map(Math.abs), ...stacked, 10) * 1.15;
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
    names.map(nm=>'<span style="color:'+col(nm)+'">■ '+nm+'</span>').join(" · ");
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
      if(e.kind === "taught") cls = "kind-taught";
      tr.innerHTML = '<td class="time">'+hmsMs(e.time_iso)+'</td><td class="'+cls+'">'+e.kind+
        '</td><td><b>'+e.device+'</b></td><td>'+(e.dP_W==null?"":e.dP_W)+
        '</td><td>'+(e.dQ_var==null?"":e.dQ_var)+'</td><td>'+(e.confidence==null?"":e.confidence)+
        '</td><td class="muted">'+(e.detail||"")+'</td>';
      tb.insertBefore(tr, tb.firstChild);
    });
    while(tb.children.length > 150) tb.removeChild(tb.lastChild);
  }catch(e){}
}

async function teach(){
  const name = document.getElementById("teach-name").value.trim();
  if(!name) return alert("give the device a name first");
  const r = await j("/api/teach", {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify({label:name, retrain:true})});
  if(!r.ok) alert("teach failed: " + r.error);
  else document.getElementById("teach-name").value = "";
}
async function retrain(){ await j("/api/retrain", {method:"POST"}); }
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
    retrainer = Retrainer(models, args.scenarios_dir,
                          window_s=args.retrain_window, on_w=args.on_w)
    engine = LiveEngine(svc, models, retrainer, session_dir,
                        stride_s=args.stride, unknown_min_W=args.unknown_min_w)

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
