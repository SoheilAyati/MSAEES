#!/usr/bin/env python3
"""
test_thresholds.py -- the power thresholds of live.py's engine. No hardware.

Guards two real failures seen on the bench 2026-07-15:

  * a 14 W monitor was INVISIBLE: big enough for the edge detector (8 W) but
    below both unknown_claim_min_W (15 W) and unknown_min_W (30 W), so it
    could never be claimed nor prompted -- and therefore never taught, since
    the Teach button is driven by the unknown prompt.

  * a 17 W table_fan claim was IMMORTAL: the stale-claim guard allowed the
    claimed watts to exceed the meter by a flat 30 W floor, which is larger
    than the whole claim, so `17 <= 0 + 30` held with the meter reading 0.0 W
    and the phantom never cleared. Phantoms poison the residual, which is
    what the residual matcher probes with -- see test_residual_match.py.

Run with the MS2_Pipeline venv python:
    .venv/Scripts/python.exe test_thresholds.py
"""
import os
import sys
import tempfile
import threading
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import live                              # noqa: E402


class _Svc:
    """The slice of pr.AcquisitionService LiveEngine touches at construction."""
    sample_rate_hz = 5.0
    state = "connected"
    session = None
    _buffer = []
    _spec_buffer = []
    _lock = threading.RLock()
    reader = types.SimpleNamespace(gt=None, inner=None)


def build():
    tmp = tempfile.mkdtemp(prefix="thresholds_")
    mm = live.ModelManager(models_dir=os.path.join(tmp, "m"),
                           recordings_dir=os.path.join(tmp, "r"))
    return live.LiveEngine(_Svc(), mm, None, out_dir=os.path.join(tmp, "o"))


def test_small_device_is_reachable():
    """A 14 W monitor must be claimable AND promptable, or it cannot be taught."""
    eng = build()
    assert 14.0 >= eng.edge_min_W, "14 W is not even an edge"
    assert 14.0 >= eng.unknown_claim_min_W, (
        f"14 W cannot claim as an unknown load (min {eng.unknown_claim_min_W})")
    assert 14.0 >= eng.unknown_min_W, (
        f"14 W never raises the unknown prompt (min {eng.unknown_min_W}) -- "
        "so it can never be taught")
    # the two floors must not reopen the gap the monitor fell into
    assert eng.unknown_claim_min_W <= eng.unknown_min_W, (
        "a device can be prompted but not claimed: the gap is back")
    print(f"  edge {eng.edge_min_W:.0f} W <= claim {eng.unknown_claim_min_W:.0f} W "
          f"<= prompt {eng.unknown_min_W:.0f} W  -> a 14 W monitor is reachable")


def test_idle_noise_stays_silent():
    """The lowered prompt floor must still sit above the meter's idle noise."""
    eng = build()
    for noise in (0.5, 2.0, 5.0):
        assert noise < eng.unknown_min_W, (
            f"{noise} W of idle noise would raise the unknown prompt")
    print(f"  idle noise up to 5 W stays below the {eng.unknown_min_W:.0f} W prompt")


def test_phantom_is_evictable():
    """The live case: 17 W claimed while the meter reads 0.0 W."""
    eng = build()
    slack = eng._claim_slack_W(17.0)
    assert not (17.0 <= 0.0 + slack), (
        f"a 17 W phantom survives at 0 W measured (slack {slack})")
    # ...and every small device must be evictable, not just this one
    for w in (11.0, 17.0, 30.0):
        assert not (w <= 0.0 + eng._claim_slack_W(w)), \
            f"a {w} W phantom is immortal at 0 W measured"
    print(f"  17 W phantom at 0 W -> dropped (slack {slack:.1f} W)")


def test_healthy_claims_survive():
    """The tighter floor must not start killing real claims."""
    eng = build()
    cases = [
        (17.0, 17.0, True, "real 17 W fan at 17 W"),
        (17.0, 14.0, True, "17 W fan, meter drifts to 14 W"),
        (47.0, 46.0, True, "two fans 47 W at 46 W"),
        (950.0, 950.0, True, "boiler 950 W at 950 W"),
        (950.0, 900.0, True, "boiler 950 W, meter sags 50 W"),
        (1000.0, 500.0, False, "half the claimed load is gone"),
        (30.0, 0.0, False, "standing_fan phantom at 0 W"),
    ]
    for claimed, instant, want_keep, desc in cases:
        keeps = claimed <= instant + eng._claim_slack_W(claimed)
        assert keeps == want_keep, (
            f"{desc}: {'kept' if keeps else 'dropped'}, expected "
            f"{'kept' if want_keep else 'dropped'}")
    print("  drift and sag tolerated; genuinely absent load still dropped")


def test_generation_breaks_the_guard_knowingly():
    """PV is NOT supported: this records the limit rather than hiding it.

    _reconcile_claims assumes every claim consumes. With PV exporting behind a
    real load the meter reads less than the load draws, and the guard drops a
    perfectly good claim. Claims store abs(dP), so a generator cannot even be
    represented -- supporting PV needs SIGNED claims, not a wider tolerance.
    If this test ever fails, PV support has landed and the note in
    _reconcile_claims should be revisited.
    """
    eng = build()
    pv, boiler = -300.0, 950.0
    meter = pv + boiler                  # 650 W
    dropped = not (boiler <= meter + eng._claim_slack_W(boiler))
    assert dropped, ("the guard no longer drops a real claim behind PV -- "
                     "signed claims may have landed; update this test")
    print(f"  KNOWN LIMIT: PV {pv:.0f} W + boiler {boiler:.0f} W -> meter "
          f"{meter:.0f} W -> boiler claim wrongly dropped (needs signed claims)")


if __name__ == "__main__":
    print("=== a small device must be reachable ===")
    test_small_device_is_reachable()
    test_idle_noise_stays_silent()
    print("\n=== phantoms must be evictable ===")
    test_phantom_is_evictable()
    test_healthy_claims_survive()
    print("\n=== known limits ===")
    test_generation_breaks_the_guard_knowingly()
    print("\nall threshold scenarios passed")
