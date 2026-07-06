#!/usr/bin/env python3
"""
verify_harmonics.py  -  find + verify the PAC4200 CURRENT harmonic source
==========================================================================
One pass, run WITH A LOAD ON (>= ~0.3 A on L1, e.g. the water boiler):

  1. reads the core registers for ground truth (I_L1..L3, V_L1),
  2. checks whether offsets 49/51/53 really are THD-R I L1..L3 (%),
  3. scans FC 0x14 files and flags every file whose order-1 fundamental
     matches a measured phase current (that IS the current-harmonic file;
     the match also settles the phase order),
  4. scans the reported plain-register block (~11007, FC 0x03 and 0x04)
     for a decaying spectrum whose fundamental matches I_L1,
  5. prints the exact lines to paste into pac_reader.py.

Background: files 101/102/103 were once guessed to be current harmonics but
return a constant index table (small integers = float32 denormals) regardless
of load - that is what poisoned the recorded harmonic datasets. Voltage L-N
files 110/116/118 decode plausibly and stay as they are.

    python verify_harmonics.py --host 192.168.168.1
"""
from __future__ import annotations
import argparse
import inspect
import math
import struct
import sys

try:
    from pymodbus.client import ModbusTcpClient
except ImportError:
    print("pymodbus not installed. Run: uv pip install pymodbus", file=sys.stderr)
    sys.exit(1)

FileRecord = None
for _mod in ("pymodbus.pdu.file_message", "pymodbus.file_message"):
    try:
        FileRecord = __import__(_mod, fromlist=["FileRecord"]).FileRecord
        break
    except Exception:
        pass


def _detect_unit_kwarg():
    try:
        params = inspect.signature(ModbusTcpClient.read_holding_registers).parameters
    except (ValueError, TypeError):
        return "device_id"
    if "device_id" in params:
        return "device_id"
    if "slave" in params:
        return "slave"
    return None


_UNIT = _detect_unit_kwarg()


def floats_be(data: bytes):
    return [struct.unpack(">f", data[i:i + 4])[0] for i in range(0, len(data) - 3, 4)]


def regs_to_float(hi, lo, swap="big"):
    if swap == "big":
        return struct.unpack(">f", struct.pack(">HH", hi & 0xFFFF, lo & 0xFFFF))[0]
    return struct.unpack(">f", struct.pack(">HH", lo & 0xFFFF, hi & 0xFFFF))[0]


def plausible(v, lo=0.0, hi=1e6):
    """finite, not a denormal bit pattern, inside [lo, hi)"""
    return math.isfinite(v) and (v == 0.0 or abs(v) > 1e-6) and lo <= v < hi


def read_holding(c, unit, addr, count):
    kw = {"count": count}
    if _UNIT:
        kw[_UNIT] = unit
    try:
        rr = c.read_holding_registers(addr, **kw)
    except Exception:            # the meter RESETS the TCP connection on some
        c.close(); c.connect()   # illegal addresses instead of replying exc 2
        return None
    return None if (rr is None or rr.isError()) else list(rr.registers)


def read_input(c, unit, addr, count):
    kw = {"count": count}
    if _UNIT:
        kw[_UNIT] = unit
    try:
        rr = c.read_input_registers(addr, **kw)
    except Exception:
        return None
    return None if (rr is None or rr.isError()) else list(rr.registers)


def read_file(c, unit, file_no, record_no, n_registers):
    fr = FileRecord(file_number=file_no, record_number=record_no,
                    record_length=2 * n_registers)
    kw = {_UNIT: unit} if _UNIT else {}
    try:
        rr = c.read_file_record([fr], **kw)
    except Exception:
        return None
    if rr is None or rr.isError():
        return None
    recs = getattr(rr, "records", None)
    if not recs:
        return None
    return getattr(recs[0], "record_data", b"")


def looks_like_spectrum(vals, fund, tol_frac=0.15, tol_abs=0.03):
    """vals[0] ~ the measured fundamental and the rest clearly smaller."""
    if not vals or not plausible(vals[0], 1e-3, 1e4):
        return False
    if abs(vals[0] - fund) > max(tol_abs, tol_frac * fund):
        return False
    rest = [v for v in vals[1:] if math.isfinite(v)]
    return bool(rest) and max(abs(v) for v in rest) < 0.6 * vals[0]


def main():
    ap = argparse.ArgumentParser(description="Verify/locate PAC4200 current-harmonic source.")
    ap.add_argument("--host", default="192.168.168.1")
    ap.add_argument("--port", type=int, default=502)
    ap.add_argument("--unit-id", type=int, default=1)
    ap.add_argument("--file-start", type=int, default=1)
    ap.add_argument("--file-end", type=int, default=256)
    ap.add_argument("--scan-plain", action="store_true",
                    help="also scan plain registers ~11007 for a spectrum "
                         "(WARNING: the meter force-closes the connection on "
                         "some illegal addresses; FC 0x14 is the confirmed "
                         "mechanism, so this is off by default)")
    args = ap.parse_args()

    c = ModbusTcpClient(args.host, port=args.port, timeout=1.5)
    if not c.connect():
        sys.exit(f"cannot connect to {args.host}:{args.port}")

    # ---- 1. ground truth ----------------------------------------------------
    core = read_holding(c, args.unit_id, 1, 48)
    if core is None:
        sys.exit("cannot read core registers 1..48")
    V1 = regs_to_float(core[0], core[1])
    I = {ph: regs_to_float(core[12 + 2 * k], core[13 + 2 * k])
         for k, ph in enumerate(("L1", "L2", "L3"))}
    print(f"ground truth:  V_L1 = {V1:.1f} V   "
          + "  ".join(f"I_{p} = {v:.3f} A" for p, v in I.items()))
    if max(I.values()) < 0.06:
        print("\n*** WARNING: no significant load current - connect a load "
              "(e.g. the water boiler)\n*** and re-run; every current check "
              "below is meaningless at idle.\n")

    # ---- 2. THD-R I candidate registers 49/51/53 -----------------------------
    print("\n[THD-R I check]  offsets 49/51/53 (family map: THD-R I L1..L3):")
    blk = read_holding(c, args.unit_id, 49, 6)
    if blk is None:
        print("  read FAILED (truly reserved on this firmware)")
    else:
        vals = [regs_to_float(blk[i], blk[i + 1]) for i in (0, 2, 4)]
        ok = all(plausible(v, 0.0, 1000.0) for v in vals)
        print(f"  decoded: L1={vals[0]:.2f}%  L2={vals[1]:.2f}%  L3={vals[2]:.2f}%"
              f"   -> {'PLAUSIBLE - pac_reader auto-enables these' if ok else 'NOT plausible'}")

    # ---- 3. FC 0x14 file scan: match fundamentals ----------------------------
    if FileRecord is None:
        print("\n[FC14] pymodbus FileRecord unavailable - skipping file scan")
        i_files = {}
    else:
        print(f"\n[FC14] scanning files {args.file_start}..{args.file_end} "
              "(order-1 fundamental + first orders):")
        i_files, v_files = {}, {}
        for fno in range(args.file_start, args.file_end + 1):
            data = read_file(c, args.unit_id, fno, 1, 16)
            if not data:
                continue
            vals = floats_be(data)
            # take order 1 at offset 1 and orders 3,5,7 at offsets 5,9,13
            for ph, amps in I.items():
                if amps >= 0.06 and looks_like_spectrum(vals, amps) and ph not in i_files:
                    i_files[ph] = fno
                    print(f"  file {fno:>3}: fundamental {vals[0]:.3f} ~ I_{ph}"
                          f" ({amps:.3f} A)  CURRENT {ph}  first orders "
                          f"{[round(v, 4) for v in vals[1:6]]}")
            if abs(vals[0] - V1) < 15.0 or abs(vals[0] - 100.0) < 2.0:
                v_files[fno] = vals[0]
        if v_files:
            print(f"  voltage-like files: { {k: round(v, 1) for k, v in v_files.items()} }")
        if not i_files:
            print("  no file matched a phase current - widen --file-end "
                  "(e.g. 1024) or check the load.")

    # ---- 4. plain-register hypothesis (~11007, stride 6; opt-in) -------------
    found_plain = None
    scanners = (("FC3", read_holding), ("FC4", read_input)) if args.scan_plain else ()
    if scanners:
        print("\n[plain regs] scanning 11000..11120 for a current spectrum "
              "(FC3 + FC4, both word orders):")
    for name, rd in scanners:
        blk = rd(c, args.unit_id, 11000, 120)
        if blk is None:
            print(f"  {name}: read failed")
            continue
        for swap in ("big", "little"):
            for start in range(0, len(blk) - 8, 1):
                f0 = regs_to_float(blk[start], blk[start + 1], swap)
                if not plausible(f0, 0.05, 1e3):
                    continue
                for stride in (2, 4, 6):
                    seq = [regs_to_float(blk[start + s * stride],
                                         blk[start + s * stride + 1], swap)
                           for s in range(4)
                           if start + s * stride + 1 < len(blk)]
                    for ph, amps in I.items():
                        if amps >= 0.06 and looks_like_spectrum(seq, amps):
                            print(f"  {name} addr {11000 + start} swap={swap} "
                                  f"stride={stride}: {[round(v, 4) for v in seq]}"
                                  f"  ~ I_{ph}")
                            found_plain = found_plain or (name, 11000 + start,
                                                          swap, stride, ph)
    if scanners and not found_plain:
        print("  nothing spectrum-like in 11000..11120")

    # ---- 5. paste-ready config ------------------------------------------------
    print("\n" + "=" * 70)
    if FileRecord is not None and i_files:
        line = ", ".join(f'"{p}": {i_files.get(p, 0)}' for p in ("L1", "L2", "L3"))
        print("PASTE into pac_reader.py:\n")
        print(f"    HARMONIC_I_FILE: Dict[str, int] = {{{line}}}")
        print("\nthen re-record each appliance with --harmonics.")
    else:
        print("Current-harmonic FC14 files not identified in this pass.")
        if found_plain:
            print(f"But the plain-register block looks live: {found_plain} - "
                  "wire it up in pac_reader (new register path).")
    c.close()


if __name__ == "__main__":
    main()
