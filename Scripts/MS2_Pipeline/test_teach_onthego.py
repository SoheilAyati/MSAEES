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


class FakeSvc:
    """The slice of pr.AcquisitionService the teach flow touches."""

    def __init__(self):
        self.sample_rate_hz = SR
        self._lock = threading.RLock()
        self._buffer = deque(maxlen=8000)
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

    def __init__(self, svc, bg_W=300.0, bg_Q=50.0, dev_W=120.0, dev_Q=20.0):
        super().__init__(daemon=True, name="feeder")
        self.svc = svc
        self.bg_W, self.bg_Q = bg_W, bg_Q
        self.dev_W, self.dev_Q = dev_W, dev_Q
        self.dev_on = True               # the unknown device is running
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
            thd = 100.0 * math.sqrt(i_h) / max(i_f, 1e-9)
            s = math.hypot(p, q)
            sample = {"P_total": p, "Q_total": q, "S_total": s,
                      "PF_total": p / max(s, 1e-9),
                      "P_L1": p, "P_L2": 0.0, "P_L3": 0.0,
                      "THD_I_L1": thd}
            with self.svc._lock:
                self.svc._buffer.append((int(time.time() * 1000), sample))
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
        P = f["measurements/P_total"][:]
        md = dict(f["metadata"].attrs)
        summary = json.loads(md["recording_summary"])
    return P, md, summary


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
        P, md, summary = load_saved(models)
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
        print("scenario 1 PASSED\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scenario_background_jump():
    print("=== scenario 2: background jumps +80 W mid-capture ===")
    tmp = tempfile.mkdtemp(prefix="teach_otg_2_")
    try:
        svc, models, engine = build_engine(tmp)
        feeder = Feeder(svc)
        driver = Driver(engine, feeder, bg_jump_W=80.0, bg_jump_after_s=4.0)
        note = run_teach(engine, feeder, driver)
        print("teach note:", note)
        print("phases seen:", driver.phases_seen)
        assert note.startswith("saved"), f"teach failed: {note}"
        assert any(p.startswith("inmix_off2") for p in driver.phases_seen), \
            "no cross-check toggle was requested despite the corrupted capture"
        P, md, summary = load_saved(models)
        assert int(md["n_toggles"]) >= 2
        dev_w = float(summary["device_W_median"])
        assert abs(dev_w - 120.0) <= 10.0, \
            f"level pulled off by the background jump: {dev_w}"
        # the corrupted ~200 W stretch must NOT be in the saved recording
        assert float(np.nanmax(P)) < 170.0, \
            f"corrupted stretch leaked into the recording (max {np.nanmax(P):.0f} W)"
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
