#!/usr/bin/env python3
"""Hardware-free tests for the unknown-device name suggestion.

The fixtures are artificial electrical signatures; no PAC4200, recording, or
trained estimator is required.  Run with the MS2_Pipeline environment:

    .venv/Scripts/python.exe Scripts/MS2_Pipeline/test_device_suggestion.py
"""
import os
import sys
import tempfile
import threading
import types
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import live                              # noqa: E402


class _Svc:
    sample_rate_hz = 5.0
    state = "connected"
    session = None
    _buffer = []
    _spec_buffer = []
    _lock = threading.RLock()
    reader = types.SimpleNamespace(gt=None, inner=None)

    def status(self):
        return {"state": "connected", "simulated": False,
                "host": "artificial suggestion fixture", "port": None,
                "configured_rate_hz": self.sample_rate_hz,
                "effective_rate_hz": self.sample_rate_hz,
                "harmonics": False, "total_samples": 0,
                "uptime_s": 0.0, "session": None}


class _Retrainer:
    def status(self):
        return {"state": "idle", "step": "", "elapsed_s": 0.0}

    def start(self):
        return False


def _sig(family, p, q):
    return {"family": family, "label": family, "P": float(p), "Q": float(q),
            "P_lo": float(p), "P_hi": float(p),
            "Q_lo": float(q), "Q_hi": float(q),
            "IH": None, "THD": None}


def _manager(tmp):
    mm = live.ModelManager(models_dir=os.path.join(tmp, "models"),
                           recordings_dir=os.path.join(tmp, "recordings"))
    mm.signatures = [
        _sig("table_fan", 31.0, 4.0),
        _sig("standing_fan", 31.0, 49.0),
        _sig("desk_fan", 32.0, 48.0),
    ]
    return mm


def test_reactive_power_breaks_same_watt_tie():
    with tempfile.TemporaryDirectory(prefix="suggestion_") as tmp:
        mm = _manager(tmp)
        mm.appliances = ["table_fan", "standing_fan"]
        got = mm.suggest_device_name(31.5, 47.0)
        assert got == "standing_fan_2", (
            f"same-watt artificial fan should follow Q signature; got {got}")
    print("  P/Q similarity: 31.5 W, 47 var -> standing_fan_2")


def test_hint_is_scoped_to_current_model_vocabulary():
    with tempfile.TemporaryDirectory(prefix="suggestion_") as tmp:
        mm = _manager(tmp)

        # Frozen original model: desk_fan has a recording in the shared table,
        # but is not in this bundle's appliance vocabulary and must not leak in.
        mm.variant = "original"
        mm.appliances = ["table_fan", "standing_fan"]
        original = mm.suggest_device_name(32.0, 48.0)
        assert original == "standing_fan_2", original

        # Latest trained model knows desk_fan, whose signature is now exact.
        mm.variant = "latest"
        mm.appliances = ["table_fan", "standing_fan", "desk_fan"]
        latest = mm.suggest_device_name(32.0, 48.0)
        assert latest == "desk_fan_2", latest

    print("  active vocabulary: original -> standing_fan_2; latest -> desk_fan_2")


def test_suffix_increments_but_keeps_the_same_family():
    with tempfile.TemporaryDirectory(prefix="suggestion_") as tmp:
        mm = _manager(tmp)
        mm.appliances = ["table_fan", "standing_fan"]
        mm.signatures.append(_sig("standing_fan", 31.4, 48.0))
        mm.signatures[-1]["label"] = "standing_fan_2"

        got = mm.suggest_device_name(31.5, 47.0)
        assert got == "standing_fan_3", got
        assert live.nl.parse_family(got) == "standing_fan"
        assert " " not in got
    print("  suffix: existing standing_fan_2 -> standing_fan_3 -> family standing_fan")


def test_unknown_snapshot_and_blank_textbox_default():
    with tempfile.TemporaryDirectory(prefix="suggestion_") as tmp:
        mm = _manager(tmp)
        mm.appliances = ["table_fan", "standing_fan"]
        eng = live.LiveEngine(_Svc(), mm, None, out_dir=os.path.join(tmp, "out"))
        eng.unknown = {"since_ms": 1_784_100_000_000,
                       "since_iso": "2026-07-15T20:00:00+02:00",
                       "typical_W": 31.5, "typical_Q": 47.0}

        # Exercise the actual endpoint consumed by the warning window, not
        # just the helper in isolation.
        app = live.create_app(_Svc(), eng, mm, types.SimpleNamespace())
        app.config.update(TESTING=True)
        with app.test_client() as client:
            response = client.get("/api/state")
            assert response.status_code == 200
            unknown = response.get_json()["unknown"]
            html = client.get("/").get_data(as_text=True)

        assert unknown["suggested_name"] == "standing_fan_2", unknown
        assert "typical_Q" not in unknown, "internal comparison detail leaked to UI"

        # The browser keeps manual text authoritative and uses the hint only
        # when the user leaves the field empty.
        assert "const name = typed || suggestedTeachName;" in html
        assert "Suggested: " in html
        assert "Leave the box empty to use this name" in html

    print("  warning payload + UI: underscores shown; blank uses hint; typed override")


def test_no_current_signatures_means_no_guess():
    with tempfile.TemporaryDirectory(prefix="suggestion_") as tmp:
        mm = _manager(tmp)
        mm.appliances = []
        assert mm.suggest_device_name(31.5, 47.0) is None
    print("  no active recorded devices -> no fabricated suggestion")


def serve_demo():
    """Show the real warning UI with a fixed synthetic unknown device."""
    tmp = tempfile.TemporaryDirectory(prefix="suggestion_demo_")
    mm = _manager(tmp.name)
    mm.appliances = ["table_fan", "standing_fan"]
    mm.source = "artificial_test_model"
    eng = live.LiveEngine(_Svc(), mm, _Retrainer(),
                          out_dir=os.path.join(tmp.name, "out"))
    eng.total_W = eng.residual_W = 31.5
    eng.explained_frac = 0.0
    eng.unknown = {"since_ms": 1_784_100_000_000,
                   "since_iso": "2026-07-15T20:00:00+02:00",
                   "typical_W": 31.5, "typical_Q": 47.0}
    app = live.create_app(_Svc(), eng, mm, _Retrainer())
    url = "http://127.0.0.1:8398/"
    print(f"Artificial suggestion demo: {url}")
    print("Expected hint: standing_fan_2. Press Ctrl-C to stop.")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=8398, threaded=True,
                debug=False, use_reloader=False)
    finally:
        tmp.cleanup()


if __name__ == "__main__":
    if "--demo" in sys.argv:
        serve_demo()
    else:
        print("=== unknown-device name suggestion (synthetic, no hardware) ===")
        test_reactive_power_breaks_same_watt_tie()
        test_hint_is_scoped_to_current_model_vocabulary()
        test_suffix_increments_but_keeps_the_same_family()
        test_unknown_snapshot_and_blank_textbox_default()
        test_no_current_signatures_means_no_guess()
        print("all device-suggestion scenarios passed")
