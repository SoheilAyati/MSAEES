#!/usr/bin/env python3
"""
test_teach_onthego.py -- closed-loop test of the in-mix ("teach on the go")
flow in live.py, no hardware needed.

A feeder thread plays a synthetic meter into a fake acquisition service
(steady background + the device under teach), and a driver thread acts as
the user: it watches engine.guide and toggles the simulated device to follow
the on-screen instructions with a human-ish reaction delay.

Scenario 1 (steady background): one off/on toggle must suffice, the three
estimates (off-step, ON body, residual history) agree, and the saved
recording must carry the device's level with the background's noise shrunk
out of the settled part.

Scenario 2 (background jumps +80 W mid-capture): the estimates must
disagree, the flow must ask for extra toggles, converge on the true level
via the robust median, and EXCLUDE the corrupted first stretch from the
saved recording.

Run with the MS2_Pipeline venv python:
    .venv/Scripts/python.exe test_teach_onthego.py
"""
import glob
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import types
from collections import deque

import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import live                              # noqa: E402


SR = 5.0                                 # Hz, like the real default poll rate


N_ORDERS = 39                            # orders 2..40, as the real meter
I_BG = 1                                 # order 3  -> background's harmonics
I_DEV = 3                                # order 5  -> the device's harmonics


class FakeSvc:
    """The slice of pr.AcquisitionService the teach flow touches."""

    def __init__(self):
        self.sample_rate_hz = SR
        self._lock = threading.RLock()
        self._buffer = deque(maxlen=8000)
        # the parallel per-order spectrum ring buffer the real service keeps
        self._spec_buffer = deque(maxlen=8000)
        self.state = "connected"
        self.session = None
        self.reader = types.SimpleNamespace(gt=None)


class FakeRetrainer:
    def status(self):
        return {"state": "idle"}

    def start(self):
        return False


class Feeder(threading.Thread):
    """Synthetic meter: steady background + the device under teach."""

    def __init__(self, svc, bg_W=300.0, bg_Q=50.0, dev_W=120.0, dev_Q=20.0,
                 emit_he=True, emit_spec=True):
        super().__init__(daemon=True, name="feeder")
        self.svc = svc
        self.bg_W, self.bg_Q = bg_W, bg_Q
        self.dev_W, self.dev_Q = dev_W, dev_Q
        self.dev_on = True               # the unknown device is running
        self.emit_he = emit_he           # False -> THD%-fallback path
        self.emit_spec = emit_spec       # False -> meter gives no spectrum
        self.stop_flag = False
        self.rng = np.random.default_rng(7)

    def run(self):
        while not self.stop_flag:
            bg_p = self.bg_W + self.rng.normal(0.0, 2.0)
            bg_q = self.bg_Q + self.rng.normal(0.0, 1.0)
            p, q = bg_p, bg_q
            if self.dev_on:
                p += self.dev_W + self.rng.normal(0.0, 0.5)
                q += self.dev_Q + self.rng.normal(0.0, 0.3)
            # harmonic currents add ~orthogonally: bg 8 % THD, device 5 %
            i_f_bg = math.hypot(bg_p, bg_q) / 230.0
            i_h = (0.08 * i_f_bg) ** 2
            if self.dev_on:
                i_f_dev = math.hypot(self.dev_W, self.dev_Q) / 230.0
                i_h += (0.05 * i_f_dev) ** 2
            i_f = math.hypot(p, q) / 230.0
            i_h_amp = math.sqrt(i_h)
            thd = 100.0 * i_h_amp / max(i_f, 1e-9)
            # the real meter's S_total includes the distortion component
            s = 230.0 * math.hypot(i_f, i_h_amp)
            sample = {"P_total": p, "Q_total": q, "S_total": s,
                      "PF_total": p / max(s, 1e-9),
                      "P_L1": p, "P_L2": 0.0, "P_L3": 0.0,
                      "THD_I_L1": thd}
            if self.emit_he:
                sample["HE_I_L1"] = i_h_amp
            # per-order spectrum: the background's harmonics sit on one order
            # and the device's on another, so the RSS isolation can be checked
            # exactly -- the device's order must survive, the background's
            # must cancel to ~0
            mag = np.zeros(N_ORDERS, dtype=np.float32)
            mag[I_BG] = 0.08 * i_f_bg
            if self.dev_on:
                mag[I_DEV] = 0.05 * math.hypot(self.dev_W, self.dev_Q) / 230.0
            t_ms = int(time.time() * 1000)
            with self.svc._lock:
                self.svc._buffer.append((t_ms, sample))
                if self.emit_spec:
                    self.svc._spec_buffer.append((t_ms, mag))
            time.sleep(1.0 / SR)


class Driver(threading.Thread):
    """Scripted user: follows the teach guide with a reaction delay; can
    inject a background jump a fixed time into the first ON capture."""

    def __init__(self, engine, feeder, bg_jump_W=0.0, bg_jump_after_s=4.0):
        super().__init__(daemon=True, name="driver")
        self.engine = engine
        self.feeder = feeder
        self.bg_jump_W = bg_jump_W
        self.bg_jump_after_s = bg_jump_after_s
        self.stop_flag = False
        self.phases_seen = []

    def run(self):
        last_phase, phase_t0 = None, 0.0
        jump_done = False
        while not self.stop_flag:
            g = self.engine.guide
            phase = g["phase"] if g else None
            if phase != last_phase:
                last_phase, phase_t0 = phase, time.time()
                if phase and phase not in self.phases_seen:
                    self.phases_seen.append(phase)
            if phase:
                if "off" in phase:
                    want = False
                elif "baseline" in phase:
                    want = False
                else:                    # inmix_on*, inmix_recording*
                    want = True
                if (want != self.feeder.dev_on
                        and time.time() - phase_t0 > 1.0):
                    self.feeder.dev_on = want
                if (self.bg_jump_W and not jump_done
                        and phase == "inmix_recording"
                        and time.time() - phase_t0 > self.bg_jump_after_s):
                    self.feeder.bg_W += self.bg_jump_W
                    jump_done = True
            time.sleep(0.2)


def build_engine(tmp):
    svc = FakeSvc()
    models = live.ModelManager(models_dir=os.path.join(tmp, "models"),
                               recordings_dir=os.path.join(tmp, "recordings"))
    engine = live.LiveEngine(svc, models, FakeRetrainer(),
                             out_dir=os.path.join(tmp, "out"))
    engine.teach_record_s = 8.0          # keep the test fast
    engine.inmix_base_s = 3.0
    engine._run_flag.set()               # teach loops check it; no _loop needed
    return svc, models, engine


def run_teach(engine, feeder, driver, prefill_s=3.0):
    feeder.start()
    time.sleep(prefill_s)                # the flow needs settled samples
    driver.start()
    engine.teach("test_fan", retrain=False, mode="inmix")
    t0 = time.time()
    while time.time() - t0 < 240.0:
        with engine.lock:
            th = engine._teach_thread
        if th is None or not th.is_alive():
            break
        time.sleep(0.5)
    feeder.stop_flag = driver.stop_flag = True
    feeder.join(timeout=3)
    driver.join(timeout=3)
    return engine._teach_note


def load_saved(models):
    files = sorted(glob.glob(os.path.join(models.onthego_dir, "*.h5")))
    assert files, "no recording saved in on-the-go/"
    with h5py.File(files[-1], "r") as f:
        data = {ch: f["measurements/" + ch][:] for ch in f["measurements"]
                if ch != "harmonics"}
        if "harmonics" in f["measurements"]:
            data["_I_mag_L1"] = f["measurements/harmonics/I_mag_L1"][:]
        md = dict(f["metadata"].attrs)
        summary = json.loads(md["recording_summary"])
    return data, md, summary


def scenario_steady():
    print("=== scenario 1: steady background, one toggle must suffice ===")
    tmp = tempfile.mkdtemp(prefix="teach_otg_1_")
    try:
        svc, models, engine = build_engine(tmp)
        feeder = Feeder(svc)
        driver = Driver(engine, feeder)
        # the unknown prompt has been up for a while: give the engine the
        # residual history the harvest step feeds on
        now = int(time.time() * 1000)
        engine.unknown = {"since_ms": now - 60000, "typical_W": 120.5}
        rng = np.random.default_rng(3)
        with engine.lock:
            for i in range(12):
                engine.history.append({"t_ms": now - 30000 + i * 2000,
                                       "P_total": 420.0,
                                       "residual": 120.0 + rng.normal(0, 1.0),
                                       "devices": {}})
        note = run_teach(engine, feeder, driver)
        print("teach note:", note)
        assert note.startswith("saved"), f"teach failed: {note}"
        assert "1 toggle" in note, f"expected 1 toggle, got: {note}"
        assert not any("inmix_off2" in p for p in driver.phases_seen), \
            "extra toggles were requested on a steady background"
        data, md, summary = load_saved(models)
        P = data["P_total"]
        assert md["teach_mode"] == "on_the_go"
        assert int(md["n_toggles"]) == 1
        dev_w = float(summary["device_W_median"])
        assert abs(dev_w - 120.0) <= 8.0, f"device level off: {dev_w}"
        ests = summary["device_W_estimates"]
        assert ests["residual_history"] is not None, \
            "residual-history estimate was not harvested"
        # lead must be true zeros (device off = no draw, no background)
        n_lead = int(10.0 * SR)
        assert np.all(P[:n_lead - 2] == 0.0), "off lead is not clean zeros"
        # settled body: background noise (2 W) must be shrunk out
        body = P[np.abs(P - dev_w) < 0.25 * dev_w]
        assert len(body) >= 20, "no settled body in the saved recording"
        body_std = float(np.std(body[len(body) // 3:]))
        print(f"device_W {dev_w:.1f}, settled body std {body_std:.2f} W, "
              f"noise_scale {summary['noise_scale']}")
        assert body_std < 1.5, \
            f"background noise leaked into the recording (std {body_std:.2f})"
        # the device's OWN harmonic/apparent-power signature must survive the
        # isolation: an all-NaN THD trains THD_I_mean 0 and S=hypot(P,Q)
        # trains PF ~1 - both unmatchable against the live device
        on = P > 0.5 * dev_w
        THD = data["THD_I_L1"]
        assert np.isfinite(THD[on]).sum() >= 0.8 * on.sum(), \
            "THD is missing from the saved ON stretch"
        med_thd = float(np.nanmedian(THD[on]))
        assert abs(med_thd - 5.0) <= 2.0, \
            f"device THD off: {med_thd:.1f}% (feeder truth 5%)"
        i_f_dev = math.hypot(120.0, 20.0) / 230.0
        exp_S = 230.0 * math.hypot(i_f_dev, 0.05 * i_f_dev)
        med_S = float(np.nanmedian(data["S_total"][on]))
        assert abs(med_S - exp_S) <= 4.0, \
            f"S misses the distortion VA: {med_S:.1f} vs {exp_S:.1f}"
        print(f"device THD {med_thd:.1f} %, S {med_S:.1f} VA "
              f"(expected {exp_S:.1f})")

        # the saved file MUST carry a real per-order spectrum. Without it the
        # scenario mixer zero-fills the device and the mix model is taught
        # that it has NO harmonic content -- which for a switching supply is
        # the inverse of the truth (the laptop trained at THD 0 % against a
        # real 168 % and was then read as the coffee machine live).
        assert md["harmonics_enabled"], "saved recording claims no harmonics"
        assert "_I_mag_L1" in data, "no harmonics/I_mag_L1 group was written"
        mag = data["_I_mag_L1"]
        assert mag.shape == (len(P), N_ORDERS), f"spectrum shape {mag.shape}"
        # the DEVICE's order must survive the isolation...
        body_mag = mag[on]
        got_dev = float(np.median(body_mag[:, I_DEV]))
        exp_dev = 0.05 * i_f_dev
        assert abs(got_dev - exp_dev) <= 0.15 * exp_dev, \
            f"device harmonic order lost: {got_dev:.4f} A vs {exp_dev:.4f} A"
        # ...and the BACKGROUND's order must be subtracted out
        got_bg = float(np.median(body_mag[:, I_BG]))
        assert got_bg <= 0.15 * exp_dev, \
            f"background harmonics leaked into the spectrum: {got_bg:.4f} A"
        # the off lead carries no device -> no harmonics
        assert float(np.max(mag[:n_lead - 2])) == 0.0, \
            "off lead has harmonic content"
        print(f"spectrum isolated: device order {got_dev:.4f} A "
              f"(expected {exp_dev:.4f}), background order {got_bg:.4f} A")
        print("scenario 1 PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scenario_background_jump():
    print("=== scenario 2: background jumps +80 W mid-capture ===")
    tmp = tempfile.mkdtemp(prefix="teach_otg_2_")
    try:
        svc, models, engine = build_engine(tmp)
        # emit_he=False: this scenario also covers the THD%-derived fallback
        # (a meter whose per-order spectrum reads are unavailable)
        feeder = Feeder(svc, emit_he=False)
        driver = Driver(engine, feeder, bg_jump_W=80.0, bg_jump_after_s=4.0)
        note = run_teach(engine, feeder, driver)
        print("teach note:", note)
        print("phases seen:", driver.phases_seen)
        assert note.startswith("saved"), f"teach failed: {note}"
        assert any(p.startswith("inmix_off2") for p in driver.phases_seen), \
            "no cross-check toggle was requested despite the corrupted capture"
        data, md, summary = load_saved(models)
        P = data["P_total"]
        assert int(md["n_toggles"]) >= 2
        dev_w = float(summary["device_W_median"])
        assert abs(dev_w - 120.0) <= 10.0, \
            f"level pulled off by the background jump: {dev_w}"
        # the corrupted ~200 W stretch must NOT be in the saved recording
        assert float(np.nanmax(P)) < 170.0, \
            f"corrupted stretch leaked into the recording (max {np.nanmax(P):.0f} W)"
        on = P > 0.5 * dev_w
        assert np.isfinite(data["THD_I_L1"][on]).any(), \
            "THD%-fallback path saved no THD at all"
        # the warning about disagreeing estimates must be in the event log
        kinds = [e["kind"] for e in engine.events]
        assert "teach_warning" in kinds, "no teach_warning event was logged"
        print(f"device_W {dev_w:.1f}, n_toggles {md['n_toggles']}, "
              f"max saved P {float(np.nanmax(P)):.0f} W")
        print("scenario 2 PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    scenario_steady()
    scenario_background_jump()
    print("all scenarios passed")
