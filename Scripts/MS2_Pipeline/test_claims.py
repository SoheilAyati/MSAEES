#!/usr/bin/env python3
"""
test_claims.py -- the edge -> claim state machine in live.py. No hardware.

This is the regression net for the claim bookkeeping: it pins the behaviour
that WORKS today (consumers switching on/off, ramping devices, composite
steps, phantom eviction) so that reworking the sign handling for PV cannot
quietly break it.

Sign convention: consumers draw POSITIVE power, generators (pv) NEGATIVE.

Run with the MS2_Pipeline venv python:
    .venv/Scripts/python.exe test_claims.py
"""
import os
import sys
import tempfile
import threading
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import live                              # noqa: E402


# Measured on the real bench 2026-07-15. pv is the only generator.
SIGS = [
    dict(family="table_fan", label="table_fan_low", P=11.5, Q=15.1,
         P_lo=11.5, P_hi=11.6, Q_lo=15.1, Q_hi=15.2, IH=0.0084, THD=0.10),
    dict(family="table_fan", label="table_fan_high", P=17.1, Q=23.5,
         P_lo=17.0, P_hi=17.2, Q_lo=23.3, Q_hi=23.7, IH=0.0144, THD=0.11),
    # IH here is MEASURED, not guessed: 0.114 A, not the laptop's 0.494. It
    # scales with device size, so this 15 W supply has a small harmonic
    # CURRENT despite a huge harmonic RATIO -- which is why the gate is THD.
    dict(family="monitor", label="monitor", P=15.2, Q=-0.9,
         P_lo=14.8, P_hi=15.6, Q_lo=-1.2, Q_hi=-0.6, IH=0.1142, THD=1.73),
    dict(family="standing_fan", label="standing_fan_low", P=22.3, Q=36.6,
         P_lo=22.1, P_hi=22.7, Q_lo=35.6, Q_hi=38.2, IH=0.0237, THD=0.13),
    dict(family="standing_fan", label="standing_fan_high", P=30.4, Q=51.6,
         P_lo=30.2, P_hi=30.6, Q_lo=50.9, Q_hi=52.5, IH=0.0339, THD=0.13),
    dict(family="coffee_machine", label="coffee_machine_standby", P=46.0,
         Q=-0.8, P_lo=44.0, P_hi=59.6, Q_lo=-1.0, Q_hi=-0.8, IH=0.0430,
         THD=0.22),
    dict(family="laptop", label="laptop", P=65.9, Q=-6.3,
         P_lo=65.7, P_hi=66.0, Q_lo=-6.7, Q_hi=-6.0, IH=0.4944, THD=1.72),
    dict(family="coffee_machine", label="coffee_machine_run", P=1206.1,
         Q=2.5, P_lo=1196.8, P_hi=1245.7, Q_lo=2.5, Q_hi=2.6, IH=0.1282,
         THD=0.02),
    dict(family="standing_lamp", label="standing_lamp_on", P=500.1, Q=-0.2,
         P_lo=499.4, P_hi=500.9, Q_lo=-0.2, Q_hi=-0.2, IH=0.0503, THD=0.02),
    dict(family="water_boiler", label="water_boiler_on", P=954.6, Q=2.6,
         P_lo=952.9, P_hi=955.8, Q_lo=2.6, Q_hi=2.6, IH=0.0978, THD=0.02),
    dict(family="pv", label="pv_only", P=-8.5, Q=-17.4,
         P_lo=-13.3, P_hi=-4.6, Q_lo=-20.0, Q_hi=-15.0, IH=0.0060, THD=0.25),
]


class _Svc:
    sample_rate_hz = 5.0
    state = "connected"
    session = None
    _buffer = []
    _spec_buffer = []
    _lock = threading.RLock()
    reader = types.SimpleNamespace(gt=None, inner=None)


def build():
    tmp = tempfile.mkdtemp(prefix="claims_")
    mm = live.ModelManager(models_dir=os.path.join(tmp, "m"),
                           recordings_dir=os.path.join(tmp, "r"))
    mm.signatures = [dict(s) for s in SIGS]
    mm.modes = mm._build_modes(mm.signatures)
    mm.window_s = 10.0
    eng = live.LiveEngine(_Svc(), mm, None, out_dir=os.path.join(tmp, "o"))
    return eng


# a real wall-clock epoch (2026-07-15): _log_event formats it with
# datetime.astimezone(), which rejects near-epoch values on Windows
_T = [1_784_100_000_000]


def step(eng, dP, dQ, p_after, dt_ms=20_000, ih=None, q_after=None):
    """Feed one settled edge through the real matching path.

    `ih` is the step's harmonic current in amps, as the engine derives it
    from the spectrum: the switching device's own harmonics for a real
    on/off, and ~0 when nothing switched (a generator merely ramping).
    """
    _T[0] += dt_ms
    eng._handle_full_edge({"t_ms": _T[0], "dP": float(dP), "dQ": float(dQ),
                           "P_after": float(p_after), "ih": ih,
                           "Q_after": q_after,
                           "kind": "full", "q_noise": 0.0}, record=True)
    return dict(eng.claims), list(eng.unknown_claims)


def claimed_W(eng):
    return (sum(c["W"] for c in eng.claims.values())
            + sum(u["W"] for u in eng.unknown_claims))


# --------------------------------------------------------------------------
# consumers: the behaviour that already works and must keep working
# --------------------------------------------------------------------------
def test_consumer_on_off():
    eng = build()
    step(eng, +954.6, +2.6, 954.6)
    assert "water_boiler" in eng.claims, "boiler on-edge did not claim"
    assert eng.claims["water_boiler"]["W"] > 0, "a consumer must claim POSITIVE watts"
    step(eng, -954.6, -2.6, 0.0)
    assert "water_boiler" not in eng.claims, "boiler off-edge did not release"
    print("  consumer on -> claimed positive; off -> released")


def test_consumer_not_confused_with_generator():
    """A fan starting must not read as PV stopping (both are +P, +Q steps)."""
    eng = build()
    step(eng, +11.5, +15.1, 11.5)
    assert "table_fan" in eng.claims, (
        f"table_fan start was not claimed; claims={list(eng.claims)}")
    assert "pv" not in eng.claims, "a fan starting was read as PV"
    print("  fan starting is a fan, not PV stopping")


def test_ramping_device_folds_into_one_claim():
    eng = build()
    step(eng, +26.0, -2.0, 26.0)                 # laptop ramp stage 1
    step(eng, +39.0, -4.0, 65.0, dt_ms=6_000)    # stage 2, within 12 s
    total = claimed_W(eng)
    assert abs(total - 65.0) <= 12.0, (
        f"ramp stages did not fold into one claim: claimed {total} W")
    print(f"  two ramp stages -> one claim of {total:.0f} W (not double counted)")


def test_phantom_is_evicted():
    """A small stale claim must not survive a dead bus -- once the generator
    is accounted for.

    This is the live 17 W table_fan phantom. It is evictable only when there
    is no UNCLAIMED generation to hide it: an unclaimed generator that can
    export 13 W and a stale 17 W claim are the same reading at the meter, so
    while pv is untracked the guard cannot tell them apart and must not
    guess -- killing a real load (the live empty-dashboard failure) is worse
    than keeping a phantom. Claim the generator and the ambiguity is gone.
    See test_phantom_survives_unclaimed_generation_KNOWN_LIMIT.
    """
    eng = build()
    step(eng, -8.5, -17.4, -8.5, q_after=-17.4)   # pv is tracked...
    assert "pv" in eng.claims
    step(eng, +17.1, +23.5, 8.6)
    assert "table_fan" in eng.claims
    eng.claims["table_fan"]["t_ms"] -= 10_000     # mature past the grace period
    eng._reconcile_claims(-8.5, _T[0] + 10_000)   # the fan is gone; only pv left
    assert "table_fan" not in eng.claims, (
        "a 17 W phantom survived with only pv (-8.5 W) on the bus")
    print("  17 W phantom evicted once pv is claimed and the meter adds up")


def test_phantom_survives_unclaimed_generation_KNOWN_LIMIT():
    """KNOWN LIMIT: while a generator is UNCLAIMED, small phantoms survive.

    Not a tuning miss -- an unclaimed generator exporting 13 W and a stale
    13 W claim are indistinguishable at the meter. The guard allows for what
    the generator could produce, so anything smaller than that hides in the
    headroom. The alternative was killing real loads: PV behind a 14 W
    monitor left the meter at 2.6 W and the dashboard empty while two devices
    ran. Keeping a phantom is the lesser wrong, and claiming pv removes it.
    """
    eng = build()
    step(eng, +17.1, +23.5, 17.1)                 # pv taught but NOT claimed
    eng.claims["table_fan"]["t_ms"] -= 10_000
    eng._reconcile_claims(0.0, _T[0] + 10_000)
    assert "table_fan" in eng.claims, (
        "the phantom now dies with pv unclaimed -- check a real load behind "
        "exporting PV still survives")
    print("  KNOWN LIMIT: with pv unclaimed, a 17 W phantom hides in the "
          "generation headroom")


def test_healthy_claim_survives_drift():
    eng = build()
    step(eng, +954.6, +2.6, 954.6)
    eng.claims["water_boiler"]["t_ms"] -= 10_000
    eng._reconcile_claims(900.0, _T[0] + 10_000)  # meter sags 55 W
    assert "water_boiler" in eng.claims, "meter sag killed a healthy claim"
    print("  boiler claim survives a 55 W meter sag")


# --------------------------------------------------------------------------
# generation: what PV needs. These FAIL today and are the target.
# --------------------------------------------------------------------------
def test_pv_start_is_not_a_fan_stopping():
    """PV starting is dP<0, which today reads as 'a consumer turned off'."""
    eng = build()
    step(eng, -8.5, -17.4, -8.5, q_after=-17.4)
    assert "pv" in eng.claims, (
        f"PV starting was not claimed as pv; claims={list(eng.claims)}")
    assert eng.claims["pv"]["W"] < 0, (
        f"pv must claim NEGATIVE watts, got {eng.claims['pv']['W']}")
    print("  PV starting -> pv claimed at negative watts")


def test_pv_does_not_steal_a_fan_off():
    """A fan switching off (dP<0) must stay a fan, not become PV starting."""
    eng = build()
    step(eng, +30.4, +51.6, 30.4)
    assert "standing_fan" in eng.claims
    step(eng, -30.4, -51.6, 0.0)
    assert "standing_fan" not in eng.claims, "fan off did not release"
    assert "pv" not in eng.claims, "a fan switching off was read as PV starting"
    print("  fan off is a fan off, not PV starting")


def test_generation_does_not_drop_real_claims():
    """The reconcile guard must not kill a load just because PV exports."""
    eng = build()
    step(eng, +954.6, +2.6, 954.6)
    step(eng, -8.5, -17.4, 946.1, dt_ms=20_000, q_after=-14.8)
    eng.claims["water_boiler"]["t_ms"] -= 10_000
    eng._reconcile_claims(946.1, _T[0] + 10_000)   # meter = boiler + pv
    assert "water_boiler" in eng.claims, (
        "PV exporting behind the boiler dropped the boiler's claim")
    print("  boiler claim survives with PV exporting behind it")


def test_pv_stops():
    """PV stopping is a POSITIVE step -- the shape of a consumer starting."""
    eng = build()
    step(eng, -8.5, -17.4, -8.5, q_after=-17.4)
    assert "pv" in eng.claims, "PV did not claim"
    step(eng, +8.5, +17.4, 0.0)
    assert "pv" not in eng.claims, (
        f"PV stopping did not release its claim; claims={list(eng.claims)}")
    print("  PV stopping (+P step) releases the pv claim")


def test_pv_and_load_net_correctly():
    """The claims' NET must track the meter, not their magnitudes."""
    eng = build()
    step(eng, -8.5, -17.4, -8.5, q_after=-17.4)
    step(eng, +954.6, +2.6, 946.1)
    net = claimed_W(eng)
    assert abs(net - 946.1) <= 20.0, (
        f"claims net to {net:.1f} W but the meter reads 946.1 W "
        "(a generator must SUBTRACT, not add)")
    print(f"  pv + boiler claims net to {net:.0f} W, matching the meter")


def test_stale_pv_claim_does_not_evict_loads():
    """A generator claim must never be the reason a load gets dropped."""
    eng = build()
    step(eng, -8.5, -17.4, -8.5, q_after=-17.4)
    step(eng, +954.6, +2.6, 946.1)
    for c in eng.claims.values():
        c["t_ms"] -= 10_000
    eng._reconcile_claims(946.1, _T[0] + 10_000)
    assert "water_boiler" in eng.claims, "the boiler was evicted"
    assert "pv" in eng.claims, "the pv claim was evicted"
    print("  reconcile leaves both the load and the generator alone")


def test_pv_ramp_releases_the_monitor_KNOWN_LIMIT():
    """KNOWN LIMIT: PV clouding over releases the monitor's claim.

    PV ramping is (-14 W, ~0 var) -- a PERFECT (P, Q) match for a 15 W
    switching supply switching off (d=0.00). Nothing in (P, Q) separates
    them; in (P, Q) they ARE the same event.

    Harmonic current looked like the answer -- the monitor IS its harmonics
    (IH 0.114 A) while a ramp carries none -- and it was tried: gate on the
    THD ratio, compare on IH. It made things WORSE. The gate only picks
    which devices are judged; the comparison is still absolute amps, and ih
    is an RSS DIFFERENCE of two THD estimates, so for a 14 W device behind a
    731 W toaster it is mostly noise. A real monitor off-step read low, was
    ruled "impossible", and the claim STRANDED until the toaster left. A
    phantom that outlives the device is worse than this.

    So the watts decide, and PV ramping takes the monitor with it. The
    monitor comes back (model vote / residual match); a stranded claim does
    not. Fixing this properly needs evidence that does not degrade with
    device size -- the harmonic RATIO measured per-sample, not as a step
    difference.
    """
    eng = build()
    step(eng, +15.2, -0.9, 15.2, ih=0.1142)             # the monitor really starts
    assert "monitor" in eng.claims, "monitor start was not claimed"
    step(eng, -13.7, +1.0, 1.5, dt_ms=20_000, ih=0.0)   # PV clouds over
    assert "monitor" not in eng.claims, (
        "the monitor now survives a PV ramp -- if that is a real fix, check "
        "a REAL monitor off-step with a noisy ih still releases it")
    print("  KNOWN LIMIT: a PV ramp releases the monitor (watts alone decide)")


def test_monitor_really_switching_off_still_releases():
    """The mirror: a REAL monitor off-step carries the monitor's harmonics."""
    eng = build()
    step(eng, +15.2, -0.9, 15.2, ih=0.1142)
    assert "monitor" in eng.claims
    step(eng, -15.2, +0.9, 0.0, ih=0.1142)              # its harmonics leave with it
    assert "monitor" not in eng.claims, (
        "a real monitor off-step no longer releases its claim")
    print("  a real monitor off-step (ih 0.11) still releases it")


def test_ih_gate_spares_the_devices_it_regressed_before():
    """The IH term was disabled because it vetoed CORRECT matches: the coffee
    machine's harmonics vary 5x across a brew, and a clean boiler 'gains'
    phantom harmonics at switch-on. Both sit BELOW the distinctive bar, so
    the term must not touch them however wrong their ih looks.
    """
    eng = build()
    # boiler switch-on reading 0.13 A of phantom harmonics (its IH is 0.098)
    step(eng, +954.6, +2.6, 954.6, ih=0.13)
    assert "water_boiler" in eng.claims, (
        "phantom harmonics at switch-on vetoed the boiler -- the exact "
        "regression that disabled use_ih")
    # coffee machine mid-brew, harmonics 4.5x its recorded median
    eng2 = build()
    step(eng2, +1206.1, +2.5, 1206.1, ih=0.58)
    assert "coffee_machine" in eng2.claims, (
        "brew-cycle harmonics vetoed the coffee machine")
    print("  boiler and coffee machine unaffected by the IH term")


def test_fan_slowing_down_is_not_pv_starting():
    """THE regression: PV's signature is a near-twin of a fan slowing down.

    standing_fan high->low is (-8.1 W, -15.0 var); pv is (-8.5 W, -17.4 var).
    Both drop P and Q together -- a shedding inductive load and an inverter
    starting to export are the same step. Matching both polarities made the
    'pv started' reading reachable AND it fits best, so a fan changing speed
    fabricated a generator that was never plugged in (seen live at 17:39:41,
    edge_on pv dP=-8.5 dQ=-18.1, conf 0.98).

    A device we already believe is running changing MODE must beat inventing
    a new family out of nothing.
    """
    eng = build()
    step(eng, +30.4, +51.6, 30.4, q_after=51.6)      # the standing fan starts, high
    assert "standing_fan" in eng.claims
    # ...and is turned down. The REAL live step (17:50:29): the fan sits a
    # few watts off its taught mode, so the step is -10.5 W where the
    # signature predicts -8.0 -- outside match_mode_change's 2.4 W gate, so
    # the mode reading cannot save us. What settles it is the bus AFTERWARDS:
    # still +37 var, strongly inductive, because a fan is still spinning. No
    # inverter is present in that.
    step(eng, -10.5, -16.1, 22.9, dt_ms=20_000, q_after=+37.2)
    assert "pv" not in eng.claims, (
        "a fan slowing down was claimed as PV starting -- pv is not plugged in")
    assert "standing_fan" in eng.claims, "the fan lost its claim while slowing"
    # NOTE the claim's watts do NOT follow the fan down here, and that is a
    # separate (pre-existing) gap: _read_edge correctly declines to name
    # anything, so the step arrives as an unrecognized full edge -- but
    # _apply_mode_change only runs BELOW edge_min_W, and this step is 8.1 W.
    # A full-edge-sized mode change therefore leaves the claim stale. Wrong,
    # but far less wrong than inventing a generator; pinned so the day it is
    # fixed, this assert is what tells us.
    w = eng.claims["standing_fan"]["W"]
    print(f"  fan high->low stays the fan, pv not invented "
          f"(claim still {w:.0f} W -- full-edge mode changes do not update it)")


def test_unclaimed_pv_does_not_kill_a_real_load():
    """THE live failure: PV exporting behind the monitor emptied the dashboard.

    PV ~11 W behind a 14 W monitor leaves the meter at ~2.6 W. The guard read
    that as "the monitor cannot be drawing 12 W" and dropped it -- two
    devices running, nothing shown. An unclaimed generator hides load from
    the meter, so the claims may legitimately exceed it by as much as that
    generator can produce.
    """
    eng = build()
    step(eng, +15.2, -0.9, 15.2)                 # monitor on; PV also exporting
    assert "monitor" in eng.claims
    eng.claims["monitor"]["t_ms"] -= 10_000      # mature past the grace period
    eng._reconcile_claims(2.6, _T[0] + 10_000)   # meter: monitor +14, pv -11
    assert "monitor" in eng.claims, (
        "unclaimed PV behind the monitor killed its claim -- both are running")
    print("  monitor survives PV exporting behind it (meter reads 2.6 W)")


def test_headroom_vanishes_once_pv_is_claimed():
    """The headroom is for UNCERTAINTY, not a blanket loosening.

    Once PV holds a claim its watts are in the sum and the meter adds up, so
    stale claims must be caught as tightly as on a bench with no generator.
    """
    eng = build()
    step(eng, -8.5, -17.4, -8.5, q_after=-17.4)  # PV claimed
    assert "pv" in eng.claims
    before = eng._claim_slack_W(17.0)
    eng.claims.pop("pv")                          # ...and now it is not
    after = eng._claim_slack_W(17.0)
    assert after > before, (
        f"unclaimed PV must widen the slack: claimed={before} unclaimed={after}")
    assert before <= 10.0, (
        f"with PV claimed the slack must tighten back to ~edge_min_W, got {before}")
    print(f"  slack: {before:.0f} W with pv claimed -> {after:.0f} W without it")


def test_phantom_still_dies_when_no_generator_is_taught():
    """Without a generator in the table there is no uncertainty to allow for,
    so the 17 W phantom must still be evicted exactly as before."""
    eng = build()
    with eng.models.lock:
        eng.models.signatures = [s for s in eng.models.signatures
                                 if s["family"] != "pv"]
    step(eng, +17.1, +23.5, 17.1)
    eng.claims["table_fan"]["t_ms"] -= 10_000
    eng._reconcile_claims(0.0, _T[0] + 10_000)
    assert "table_fan" not in eng.claims, (
        "the 17 W phantom survived on a bench with no generator")
    print("  no generator taught -> 17 W phantom still evicted at 0 W")


def test_untaught_generation_is_not_claimed():
    """KNOWN LIMIT: an UNTAUGHT generator raises no unknown claim.

    With no signature matching either polarity there is nothing to say which
    reading is right: a -260 W step is equally "an untaught generator
    started" and "an untaught 260 W load stopped". The sign rule falls back
    to 'off', so no claim is made -- deliberately. Claiming it as generation
    would be worse than the gap: an untaught toaster that was already running
    when the engine started produces a -700 W step with no claim to release,
    and would be recorded as 700 W of GENERATION out of nothing.

    The watts are not lost -- they stay in the residual and raise the unknown
    prompt, which is how the device gets taught. Once taught, the signature
    resolves the polarity and the claim is signed correctly.
    """
    eng = build()
    step(eng, -260.0, -40.0, -260.0)
    assert not [u for u in eng.unknown_claims if u["W"] < 0], (
        "untaught generation is now claimed -- if that is deliberate, make "
        "sure an untaught load stopping cannot be read as generation")
    print("  untaught generation stays in the residual (prompt), not claimed")


CONSUMER = [test_consumer_on_off, test_consumer_not_confused_with_generator,
            test_phantom_survives_unclaimed_generation_KNOWN_LIMIT,
            test_ramping_device_folds_into_one_claim, test_phantom_is_evicted,
            test_healthy_claim_survives_drift]
CONSUMER += [test_monitor_really_switching_off_still_releases,
             test_ih_gate_spares_the_devices_it_regressed_before]
GENERATION = [test_pv_start_is_not_a_fan_stopping,
              test_unclaimed_pv_does_not_kill_a_real_load,
              test_headroom_vanishes_once_pv_is_claimed,
              test_phantom_still_dies_when_no_generator_is_taught,
              test_pv_does_not_steal_a_fan_off,
              test_generation_does_not_drop_real_claims,
              test_pv_stops, test_pv_and_load_net_correctly,
              test_stale_pv_claim_does_not_evict_loads,
              test_pv_ramp_releases_the_monitor_KNOWN_LIMIT,
              test_fan_slowing_down_is_not_pv_starting,
              test_untaught_generation_is_not_claimed]


def run(group, title):
    print(f"=== {title} ===")
    bad = []
    for t in group:
        try:
            t()
        except AssertionError as e:
            bad.append((t.__name__, str(e)))
            print(f"  FAIL {t.__name__}: {e}")
    return bad


if __name__ == "__main__":
    a = run(CONSUMER, "consumers (must always pass)")
    print()
    b = run(GENERATION, "generation / PV")
    print()
    if a:
        print(f"CONSUMER REGRESSIONS: {len(a)}")
    print(f"generation failures: {len(b)}")
    sys.exit(1 if (a or b) else 0)
