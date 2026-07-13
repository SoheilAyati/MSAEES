#!/usr/bin/env python3
"""
record_campaign.py  --  guided re-recording of the whole appliance corpus
=========================================================================

Walks you through recording every appliance (each operating mode as its own
single), the known data-gap items (coffee standby, PV on a sunny day), and a
set of choreographed validation mixes -- with ONE consistent protocol, so the
corpus stops depending on how each file happened to be recorded that day:

    all off (< 5 W)  ->  10 s OFF baseline  ->  you switch the device on
    (auto-detected)  ->  fixed ON time      ->  you switch it off
    (auto-detected)  ->  8 s OFF tail       ->  saved with the right label

Phase changes are driven by the measured power itself, exactly like the
guided teach in live.py -- you only plug things in and out when asked.
Recordings include per-order harmonics and land in the same recordings dir
live.py and the retrainer use.

Usage
-----
    python record_campaign.py --host 192.168.168.1          # full campaign
    python record_campaign.py --list                        # show the plan
    python record_campaign.py --host ... --only 12-14       # just items 12-14
    python record_campaign.py --host ... --only fan         # label filter
    python record_campaign.py --host ... --skip-existing    # resume a day
    python record_campaign.py --simulate                    # dry-run rehearsal

Afterwards: retrain (live.py dashboard "Retrain now", or train.py) and replay
the choreo mixes through live.py --replay to score the engine.
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
READER_DIR = os.path.join(REPO, "Scripts", "PAC4200_reader")
sys.path.insert(0, HERE)
sys.path.insert(0, READER_DIR)

import pac_reader as pr           # noqa: E402
from live import ThdReader        # noqa: E402  (same THD math as live/teach)

OFF_W = 5.0          # "everything off" level
LEAD_S = 10.0        # off baseline before the device
TAIL_S = 8.0         # off tail after the device
SETTLE_S = 2.0       # a change counts once it holds this long


# ---------------------------------------------------------------------------
# The plan. 'single': one device, one steady mode, fixed ON time.
# 'choreo': scripted sequence inside ONE recording (mode changes, mixes).
# Labels follow nilm_pipeline parsing: state suffixes fold into the family,
# '__' separates simultaneous devices. sunny=True items need the PV panel
# actually generating -- do them on a bright day, everything else any time.
# ---------------------------------------------------------------------------
DEFAULT_PLAN = [
    # -- fans: every speed is its own single (these ARE the mode table) ------
    dict(label="table_fan_low", kind="single", on_s=75,
         note="table fan at LOWEST speed, rotation OFF"),
    dict(label="table_fan_high", kind="single", on_s=75,
         note="table fan at HIGHEST speed, rotation OFF"),
    dict(label="standing_fan_low", kind="single", on_s=75,
         note="standing fan speed 1, rotation OFF"),
    dict(label="standing_fan_med", kind="single", on_s=75,
         note="standing fan speed 2, rotation OFF"),
    dict(label="standing_fan_high", kind="single", on_s=75,
         note="standing fan speed 3, rotation OFF"),
    dict(label="standing_fan_high_rotate", kind="single", on_s=75,
         note="standing fan speed 3 WITH rotation (teaches the wobble)"),
    # -- resistive / big loads ----------------------------------------------
    dict(label="standing_lamp_on", kind="single", on_s=75,
         note="standing lamp fully on"),
    dict(label="water_boiler_on", kind="single", on_s=90, min_delta_W=300,
         note="water boiler FILLED with water, switched on"),
    dict(label="coffee_machine_run", kind="single", on_s=150, min_delta_W=300,
         note="coffee machine BREWING -- start a real brew, let it cycle"),
    dict(label="coffee_machine_standby", kind="single", on_s=75, min_delta_W=4,
         note="right AFTER the brew: machine stays ON but idle (warm-hold). "
              "This is the ~35 W state the model kept missing in mixes"),
    # -- electronics ----------------------------------------------------------
    dict(label="laptop", kind="single", on_s=120, min_delta_W=6,
         note="laptop charger, battery NOT full (so it really draws); the "
              "soft-start ramp is part of the signature -- just plug it in"),
    # -- mode-transition singles (train + validate the small-step matcher) ---
    dict(label="table_fan_low_high_low", kind="choreo", note="table fan mode walk",
         steps=[("switch", "turn table fan ON at LOW", 6),
                ("hold", 30), ("switch", "switch table fan to HIGH", 3),
                ("hold", 30), ("switch", "switch table fan back to LOW", 3),
                ("hold", 30), ("switch", "turn the table fan OFF", 6)]),
    dict(label="standing_fan_low_med_high", kind="choreo", note="standing fan mode walk",
         steps=[("switch", "turn standing fan ON at speed 1 (LOW)", 6),
                ("hold", 30), ("switch", "switch standing fan to speed 2 (MED)", 2.5),
                ("hold", 30), ("switch", "switch standing fan to speed 3 (HIGH)", 2.5),
                ("hold", 30), ("switch", "turn the standing fan OFF", 6)]),
    # -- choreographed validation mixes (replay these through live.py) -------
    dict(label="table_fan_high__standing_fan_high", kind="choreo",
         note="the composite-edge case: BOTH fans on at the same moment",
         steps=[("switch", "turn BOTH fans ON at HIGH at the SAME time "
                           "(one flick if they share a strip)", 25),
                ("hold", 60), ("switch", "turn BOTH fans OFF together", 25)]),
    dict(label="standing_fan_high__table_fan_high_low", kind="choreo",
         note="THE complaint scenario: both high, then table fan to low",
         steps=[("switch", "turn the STANDING fan ON at HIGH", 15),
                ("hold", 20), ("switch", "turn the TABLE fan ON at HIGH", 8),
                ("hold", 30), ("switch", "switch the TABLE fan to LOW "
                                         "(leave the standing fan alone)", 3),
                ("hold", 30), ("switch", "turn the TABLE fan OFF", 5),
                ("hold", 10), ("switch", "turn the STANDING fan OFF", 15)]),
    dict(label="standing_fan_high_low__table_fan_high", kind="choreo",
         note="mirror case: the STANDING fan is the one turned down",
         steps=[("switch", "turn the STANDING fan ON at HIGH", 15),
                ("hold", 20), ("switch", "turn the TABLE fan ON at HIGH", 8),
                ("hold", 30), ("switch", "switch the STANDING fan to LOW "
                                         "(leave the table fan alone)", 3),
                ("hold", 30), ("switch", "turn BOTH fans OFF", 15)]),
    dict(label="water_boiler_on__table_fan_high", kind="choreo",
         note="big + small, staggered on",
         steps=[("switch", "turn the TABLE fan ON at HIGH", 8),
                ("hold", 15), ("switch", "turn the WATER BOILER ON", 300),
                ("hold", 60), ("switch", "turn the BOILER OFF", 300),
                ("hold", 10), ("switch", "turn the fan OFF", 8)]),
    dict(label="coffee_machine_run__standing_lamp_on", kind="choreo",
         note="include a real brew so the cycling is in the file",
         steps=[("switch", "turn the STANDING LAMP ON", 300),
                ("hold", 15), ("switch", "start a BREW on the coffee machine", 300),
                ("hold", 120), ("switch", "turn the COFFEE MACHINE fully OFF", 25),
                ("hold", 10), ("switch", "turn the LAMP OFF", 300)]),
    dict(label="water_boiler_on__standing_lamp_on__table_fan_high", kind="choreo",
         note="three devices, staggered",
         steps=[("switch", "turn the TABLE fan ON at HIGH", 8),
                ("hold", 12), ("switch", "turn the STANDING LAMP ON", 300),
                ("hold", 12), ("switch", "turn the WATER BOILER ON", 300),
                ("hold", 45), ("switch", "turn the BOILER OFF", 300),
                ("hold", 8), ("switch", "turn the LAMP OFF", 300),
                ("hold", 8), ("switch", "turn the fan OFF", 8)]),
    # -- sunny-day items (PV must actually generate!) -------------------------
    dict(label="pv_only", kind="single", on_s=180, min_delta_W=8, sunny=True,
         note="SUNNY DAY ONLY: connect the PV panel, nothing else. Total "
              "power must go clearly NEGATIVE; if it hovers near 0 W there "
              "is no sun and the recording is useless"),
    dict(label="pv__water_boiler_on", kind="choreo", sunny=True,
         note="PV generating + boiler (negative and positive mixed)",
         steps=[("switch", "connect the PV panel (sunny!)", 8),
                ("hold", 20), ("switch", "turn the WATER BOILER ON", 300),
                ("hold", 45), ("switch", "turn the BOILER OFF", 300),
                ("hold", 15), ("switch", "disconnect the PV panel", 8)]),
]


# ---------------------------------------------------------------------------
class Campaign:
    def __init__(self, svc, simulate=False, speed=1.0):
        self.svc = svc
        self.simulate = simulate
        self.speed = speed          # < 1 shrinks every wait (rehearsal)

    # -- measurement helpers --------------------------------------------------
    def _instant(self, span_s=1.5):
        with self.svc._lock:
            buf = list(self.svc._buffer)
        if not buf:
            return None
        n = max(1, int(span_s * self.svc.sample_rate_hz))
        vals = [float(b[1].get("P_total", np.nan)) for b in buf[-n:]]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else None

    def _show(self, msg):
        w = self._instant()
        ww = f"{w:7.1f} W" if w is not None else "   ...  "
        print(f"\r  [{ww}]  {msg}   ", end="", flush=True)

    def _wait(self, cond, msg, timeout_s=600.0, hold_s=SETTLE_S):
        """Advance once cond(watts) has held for hold_s. Ctrl-C aborts."""
        held, t0 = 0.0, time.time()
        while time.time() - t0 < timeout_s:
            if self.simulate and time.time() - t0 > 3.0 * self.speed:
                print()
                return True
            w = self._instant()
            held = held + 0.5 if (w is not None and cond(w)) else 0.0
            self._show(msg)
            if held >= hold_s:
                print()
                return True
            time.sleep(0.5)
        print()
        return False

    def _wait_delta(self, msg, min_delta_W, timeout_s=600.0):
        """Wait for a SETTLED change of at least min_delta_W (any direction)
        relative to the level when the prompt appeared."""
        ref = self._instant(4.0)
        if ref is None:
            ref = 0.0
        return self._wait(lambda w: abs(w - ref) >= min_delta_W, msg, timeout_s)

    def _sleep_note(self, s, msg):
        s = s * self.speed
        t0 = time.time()
        while time.time() - t0 < s:
            self._show(f"{msg} ({s - (time.time() - t0):.0f} s left)")
            time.sleep(0.5)
        print()

    # -- one item --------------------------------------------------------------
    def run_item(self, item) -> bool:
        label = item["label"]
        print(f"\n{'=' * 74}\n  {label}\n  {item.get('note', '')}\n{'=' * 74}")
        if not self._wait(lambda w: abs(w) < OFF_W,
                          f"switch EVERYTHING off (need < {OFF_W:.0f} W)"):
            print("  !! timeout waiting for all-off - item skipped")
            return False
        self.svc.start_session(label)
        try:
            self._sleep_note(LEAD_S, "recording OFF baseline - touch nothing")
            if item["kind"] == "single":
                thr = float(item.get("min_delta_W", 6.0))
                if not self._wait_delta(
                        f"now switch it ON: {item.get('note', label)}", thr):
                    raise RuntimeError("device was not switched on in time")
                self._sleep_note(float(item.get("on_s", 75)),
                                 "recording ON - leave it exactly as it is")
                if not self._wait(lambda w: abs(w) < OFF_W,
                                  "now switch it OFF again"):
                    raise RuntimeError("device was not switched off in time")
            else:
                for step in item["steps"]:
                    if step[0] == "hold":
                        self._sleep_note(float(step[1]),
                                         "recording - keep everything as it is")
                    else:                                   # ("switch", msg, dW)
                        if not self._wait_delta("NOW: " + step[1], float(step[2])):
                            raise RuntimeError(f"step not seen: {step[1]}")
                if not self._wait(lambda w: abs(w) < OFF_W,
                                  "switch everything off", timeout_s=180.0):
                    print("  (no all-off seen - saving anyway)")
            self._sleep_note(TAIL_S, "recording OFF tail - touch nothing")
            done = self.svc.stop_session() or {}
            print(f"  saved: {os.path.basename(str(done.get('file', '?')))} "
                  f"({done.get('samples', '?')} samples)")
            return True
        except (RuntimeError, KeyboardInterrupt) as e:
            done = self.svc.stop_session() or {}
            f = done.get("file")
            if f and os.path.exists(f):
                os.remove(f)
                print(f"\n  !! aborted ({e}) - partial recording deleted")
            if isinstance(e, KeyboardInterrupt):
                raise
            return False


# ---------------------------------------------------------------------------
def select_items(plan, only):
    if not only:
        return list(range(len(plan)))
    idx = []
    for part in only.split(","):
        part = part.strip()
        if "-" in part and part.replace("-", "").isdigit():
            a, b = part.split("-")
            idx += list(range(int(a) - 1, int(b)))
        elif part.isdigit():
            idx.append(int(part) - 1)
        else:
            idx += [i for i, it in enumerate(plan) if part in it["label"]]
    return sorted({i for i in idx if 0 <= i < len(plan)})


def already_recorded(label, out_dir):
    """Recorded TODAY (resume semantics): the campaign exists to REPLACE the
    old corpus, so files from earlier days must not count as done."""
    today = time.strftime("%Y%m%d")
    return bool(glob.glob(os.path.join(out_dir, f"{label}_{today}_*.h5")))


def main():
    p = argparse.ArgumentParser(description="Guided recording campaign for the "
                                            "NILM appliance corpus.")
    p.add_argument("--host", default=None, help="PAC4200 IP (e.g. 192.168.168.1)")
    p.add_argument("--port", type=int, default=502)
    p.add_argument("--unit-id", type=int, default=1)
    p.add_argument("--rate", type=float, default=pr.DEFAULT_SAMPLE_RATE_HZ)
    p.add_argument("--out", default=os.path.join(READER_DIR, "recordings"),
                   help="recordings dir (default: the shared one)")
    p.add_argument("--only", default=None,
                   help="items to record: '3', '5-8', 'fan', comma-separated")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip items that already have a recording in --out")
    p.add_argument("--list", action="store_true", help="print the plan and exit")
    p.add_argument("--simulate", action="store_true",
                   help="dry-run rehearsal with the simulated meter "
                        "(steps auto-advance)")
    p.add_argument("--fast", action="store_true",
                   help="with --simulate: shrink every wait ~20x so a "
                        "rehearsal of the whole plan takes minutes")
    args = p.parse_args()

    chosen = select_items(DEFAULT_PLAN, args.only)
    if args.list or (not args.simulate and args.host is None):
        print(f"{'#':>3}  {'kind':<7} {'sunny':<6} label")
        for i, it in enumerate(DEFAULT_PLAN):
            mark = "*" if i in chosen and args.only else " "
            done = " [recorded]" if already_recorded(it["label"], args.out) else ""
            print(f"{i + 1:>3}{mark} {it['kind']:<7} "
                  f"{'SUNNY' if it.get('sunny') else '':<6} {it['label']}{done}")
            print(f"{'':>19}{it.get('note', '')}")
        if not args.list:
            p.error("--host is required unless --simulate")
        return

    if args.simulate:
        inner = pr.SimulatedReader(extra_channels=pr.EXTENDED_CHANNELS,
                                   read_harmonics=True)
        print("DRY RUN with the simulated meter - steps auto-advance, files "
              "land in --out (delete them afterwards).")
    else:
        inner = pr.ModbusReader(host=args.host, port=args.port,
                                unit_id=args.unit_id,
                                extra_channels=pr.EXTENDED_CHANNELS,
                                read_harmonics=True)
    reader = ThdReader(inner)
    svc = pr.AcquisitionService(reader, args.rate, args.out,
                                write_harmonics=True)
    svc.start()
    svc.request_connect()
    print("connecting to the meter", end="", flush=True)
    for _ in range(40):
        if svc.state == "connected":
            break
        print(".", end="", flush=True)
        time.sleep(0.5)
    print(f" {svc.state}")
    if svc.state != "connected":
        sys.exit("could not connect - check --host / network")

    camp = Campaign(svc, simulate=args.simulate,
                    speed=0.05 if (args.fast and args.simulate) else 1.0)
    done, skipped = [], []
    try:
        for i in chosen:
            it = DEFAULT_PLAN[i]
            if args.skip_existing and already_recorded(it["label"], args.out):
                print(f"\n[{i + 1}/{len(DEFAULT_PLAN)}] {it['label']} - already "
                      "recorded, skipping")
                continue
            ans = input(f"\n[{i + 1}/{len(DEFAULT_PLAN)}] next: {it['label']}"
                        f"{'  (SUNNY-DAY item!)' if it.get('sunny') else ''}"
                        "  -- Enter=record, s=skip, q=quit > ").strip().lower()
            if ans == "q":
                break
            if ans == "s":
                skipped.append(it["label"])
                continue
            (done if camp.run_item(it) else skipped).append(it["label"])
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        svc.shutdown()

    print(f"\n{'=' * 74}\nrecorded {len(done)} item(s); skipped/failed: "
          f"{', '.join(skipped) or '-'}")
    if done:
        print("next steps:\n"
              "  1. retrain:  live.py dashboard 'Retrain now' (or train.py)\n"
              "  2. validate: python live.py --replay "
              "../PAC4200_reader/recordings/<choreo mix>.h5")


if __name__ == "__main__":
    main()
