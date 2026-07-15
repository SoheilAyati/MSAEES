#!/usr/bin/env python3
"""
test_residual_match.py -- the harmonic-ratio (THD) gate of the residual
matcher in live.py. No hardware, no recordings: the signature table is
injected directly so the numbers below are the ones the real bench measured
on 2026-07-15.

The bug this guards: a laptop charger taught at 66 W idles at ~43 W, where
(P, Q) alone name coffee_machine_standby (46 W) at conf 0.91 -- a device
that was not even plugged in. The laptop is not a candidate at 43 W at all
(its P-term is 1.37), so the window-model arbitration could never fire. THD
does not scale with load (172 % at 43 W and at 66 W alike, vs 22 % for the
coffee machine), so it separates them where watts cannot.

Run with the MS2_Pipeline venv python:
    .venv/Scripts/python.exe test_residual_match.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import live                              # noqa: E402


# Measured on the real meter, 2026-07-15. THD is a ratio (1.72 = 172 %).
SIGS = [
    dict(family="table_fan", label="table_fan_high", P=17.1, Q=23.5,
         P_lo=17.0, P_hi=17.2, Q_lo=23.3, Q_hi=23.7, IH=0.0144, THD=0.11),
    dict(family="standing_fan", label="standing_fan_high", P=30.4, Q=51.6,
         P_lo=30.2, P_hi=30.6, Q_lo=50.9, Q_hi=52.5, IH=0.0339, THD=0.13),
    dict(family="coffee_machine", label="coffee_machine_standby", P=46.0,
         Q=-0.8, P_lo=44.0, P_hi=59.6, Q_lo=-1.0, Q_hi=-0.8, IH=0.0430,
         THD=0.22),
    dict(family="laptop", label="laptop", P=65.9, Q=-6.3,
         P_lo=65.7, P_hi=66.0, Q_lo=-6.7, Q_hi=-6.0, IH=0.4944, THD=1.72),
    dict(family="standing_lamp", label="standing_lamp_on", P=500.1, Q=-0.2,
         P_lo=499.4, P_hi=500.9, Q_lo=-0.2, Q_hi=-0.2, IH=0.0503, THD=0.02),
    dict(family="toaster", label="toaster", P=703.2, Q=-1.5,
         P_lo=702.0, P_hi=708.3, Q_lo=-1.8, Q_hi=-1.3, IH=0.0287, THD=0.01),
    dict(family="water_boiler", label="water_boiler_on", P=954.6, Q=2.6,
         P_lo=952.9, P_hi=955.8, Q_lo=2.6, Q_hi=2.6, IH=0.0978, THD=0.02),
    dict(family="coffee_machine", label="coffee_machine_run", P=1206.1,
         Q=2.5, P_lo=1196.8, P_hi=1245.7, Q_lo=2.5, Q_hi=2.6, IH=0.1282,
         THD=0.02),
]


def build():
    tmp = tempfile.mkdtemp(prefix="resid_match_")
    mm = live.ModelManager(models_dir=os.path.join(tmp, "models"),
                           recordings_dir=os.path.join(tmp, "recordings"))
    mm.signatures = [dict(s) for s in SIGS]
    mm.modes = mm._build_modes(mm.signatures)
    return mm


def name_of(mm, dP, dQ, thd=None, q_tol_scale=1.0):
    m = mm.match_edge(dP, dQ, q_tol_scale=q_tol_scale, thd=thd)
    return (m["family"] if m else None), (m["confidence"] if m else 0.0)


def test_the_bug():
    """A laptop idling at 43 W must not be named coffee_machine."""
    mm = build()
    fam, conf = name_of(mm, 43.1, -4.0, thd=1.70, q_tol_scale=3.0)
    assert fam == "laptop", f"laptop at 43 W named {fam!r} (conf {conf})"
    # named by fingerprint, not by watts -> must never read as certainty
    assert conf <= 0.85, f"off-watts match claims conf {conf}"
    print(f"  laptop idling at 43 W -> {fam} (conf {conf})")

    # without a THD reading the old (P, Q)-only behaviour is unchanged: this
    # is what the gate has to beat, and it documents why the gate exists
    fam, conf = name_of(mm, 43.1, -4.0, thd=None, q_tol_scale=3.0)
    assert fam == "coffee_machine", f"expected the old misnaming, got {fam!r}"
    print(f"  same probe with no THD available -> {fam} (conf {conf}) "
          f"[the bug, when the meter gives no spectrum]")


def test_no_regression():
    """Everything that worked before must still work."""
    mm = build()
    # a REAL coffee machine standby keeps its name
    fam, conf = name_of(mm, 46.0, -0.8, thd=0.22, q_tol_scale=3.0)
    assert fam == "coffee_machine", f"real coffee standby named {fam!r}"
    assert conf > 0.85, f"exact match should be confident, got {conf}"

    # the laptop at its taught watts stays confident
    fam, conf = name_of(mm, 65.9, -6.3, thd=1.72, q_tol_scale=3.0)
    assert fam == "laptop" and conf > 0.85, f"laptop settled -> {fam} {conf}"

    # the EDGE path passes no thd at all and must be untouched
    for dP, dQ, want in ((30.4, 51.6, "standing_fan"),
                         (17.1, 23.5, "table_fan"),
                         (954.6, 2.6, "water_boiler"),
                         (500.1, -0.2, "standing_lamp"),
                         (703.2, -1.5, "toaster"),
                         (65.9, -6.3, "laptop")):
        fam, conf = name_of(mm, dP, dQ)
        assert fam == want, f"edge {dP} W -> {fam!r}, want {want!r}"
    print("  real coffee standby, settled laptop, and every edge match: intact")


def test_gate_tolerates_meter_noise():
    """The gate must bite only on LARGE absolute THD disagreement.

    The meter has a ~2.3 % THD floor that scales with fundamental current,
    and a coffee machine's harmonics vary across its brew cycle. Neither may
    rule a signature out -- that is what killed the IH-in-amps term
    (use_ih/--ih-matching, off by default).
    """
    mm = build()
    for dP, dQ, thd, want in ((954.6, 2.6, 0.06, "water_boiler"),
                              (703.2, -1.5, 0.05, "toaster"),
                              (1206.1, 2.5, 0.10, "coffee_machine"),
                              (30.4, 51.6, 0.20, "standing_fan")):
        fam, _ = name_of(mm, dP, dQ, thd=thd, q_tol_scale=3.0)
        assert fam == want, (f"meter noise ruled out {want!r} at {dP} W "
                             f"(THD {thd}) -> got {fam!r}")
    print("  meter THD floor and brew-cycle harmonics rule nothing out")


def test_laptop_does_not_steal():
    """The widened laptop taper must not swallow other devices."""
    mm = build()
    # at another device's watts AND that device's THD, the laptop loses
    for dP, dQ, thd, want in ((30.4, 51.6, 0.13, "standing_fan"),
                              (500.1, -0.2, 0.02, "standing_lamp"),
                              (17.1, 23.5, 0.11, "table_fan")):
        fam, _ = name_of(mm, dP, dQ, thd=thd, q_tol_scale=3.0)
        assert fam == want, f"laptop stole {want!r} at {dP} W -> {fam!r}"

    # a device matched AT its own watts must outrank an off-watts taper
    fam, conf = name_of(mm, 46.0, -0.8, thd=0.22, q_tol_scale=3.0)
    assert fam == "coffee_machine", f"taper outranked an exact match: {fam!r}"
    print("  taper never outranks a device matched at its own watts")


if __name__ == "__main__":
    print("=== the bug: laptop idling below its taught watts ===")
    test_the_bug()
    print("\n=== no regression ===")
    test_no_regression()
    print("\n=== gate tolerates meter noise ===")
    test_gate_tolerates_meter_noise()
    print("\n=== laptop taper does not steal ===")
    test_laptop_does_not_steal()
    print("\nall residual-match scenarios passed")
