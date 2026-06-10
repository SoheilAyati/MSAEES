#!/usr/bin/env python3
"""
find_harmonics.py  -  one-off scanner to locate the PAC4200 harmonic register
block on YOUR meter, so you can fill HARMONIC_I_BASE / HARMONIC_V_BASE in
pac_reader.py.

It reads a register range and prints the decoded 32-bit floats (big-word-first,
the PAC4200 default) for BOTH function codes:
    FC 0x03  (holding registers - what pac_reader.py uses today)
    FC 0x04  (input registers   - some firmware serves harmonics here instead)

How to use the output
---------------------
The instantaneous harmonic block is a run of plain 2-register floats, one float
per order, 1st..64th, stored contiguously per quantity. So each quantity shows
up as: one BIG value (the fundamental / order 1) followed by ~63 small, mostly
decaying values (orders 2..64), then the next quantity starts.

  * A CURRENT block:  order 1 ~ the load current in A, then small A values.
  * A VOLTAGE block:  order 1 ~ 230 (V, absolute), then small values in % .

Match what you see against the meter's own harmonic display (display 17.0 =
Harmonic I, 15.0 = Harmonic U L-N) or the meter's built-in web page. The address
printed at the start of the line where a quantity's fundamental appears is that
quantity's ORDER-1 base. In pac_reader.py you then set the base to order-1 + 2
(to start at order 2 and match N_HARMONICS = 39 -> orders 2..40).

Stay BELOW ~37200: above the instantaneous block are max/min-with-timestamp
entries (FLOAT+TS32), which are NOT 2-register floats, so this 2-register decode
will look like garbage there - ignore it.

    python find_harmonics.py --host 192.168.168.1
    python find_harmonics.py --host 192.168.168.1 --start 35000 --end 38000
"""

import argparse
import inspect
import struct
import sys

try:
    from pymodbus.client import ModbusTcpClient
except ImportError:
    print("pymodbus not installed. Run: pip install pymodbus", file=sys.stderr)
    sys.exit(1)


def _detect_unit_kwarg():
    """pymodbus renamed the unit/slave kwarg across versions (slave -> device_id
    at 3.9). Detect what the installed version accepts."""
    try:
        params = inspect.signature(
            ModbusTcpClient.read_holding_registers).parameters
    except (ValueError, TypeError):
        return "device_id"
    if "device_id" in params:
        return "device_id"
    if "slave" in params:
        return "slave"
    return None


_UNIT = _detect_unit_kwarg()


def _read(fn, addr, count, unit):
    kwargs = {"count": count}
    try:
        if _UNIT is not None:
            kwargs[_UNIT] = unit
            rr = fn(addr, **kwargs)
        else:
            rr = fn(addr, count, unit)
    except Exception:
        return None
    if rr is None or rr.isError():
        return None
    return list(rr.registers)


def _decode(regs):
    out = []
    for i in range(0, len(regs) - 1, 2):
        packed = struct.pack(">HH", regs[i] & 0xFFFF, regs[i + 1] & 0xFFFF)
        out.append(round(struct.unpack(">f", packed)[0], 3))
    return out


def main():
    ap = argparse.ArgumentParser(description="Locate PAC4200 harmonic registers.")
    ap.add_argument("--host", default="192.168.168.1")
    ap.add_argument("--port", type=int, default=502)
    ap.add_argument("--unit-id", type=int, default=1)
    ap.add_argument("--start", type=int, default=36000,
                    help="first register to scan (default 36000)")
    ap.add_argument("--end", type=int, default=37200,
                    help="last register to scan (default 37200)")
    ap.add_argument("--floats-per-line", type=int, default=16,
                    help="floats decoded per request/line (default 16 -> 32 regs)")
    args = ap.parse_args()

    regs_per_read = min(args.floats_per_line * 2, 120)  # stay under the 125 limit

    c = ModbusTcpClient(args.host, port=args.port, timeout=1.0)
    if not c.connect():
        print(f"cannot connect to {args.host}:{args.port}", file=sys.stderr)
        sys.exit(1)
    print(f"pymodbus unit kwarg = {_UNIT or 'positional'}")
    print(f"scanning {args.start}..{args.end}  "
          f"({regs_per_read} regs / line)\n")

    for label, fn in (("FC03 / holding", c.read_holding_registers),
                      ("FC04 / input  ", c.read_input_registers)):
        print(f"================= {label} =================")
        addr = args.start
        any_ok = False
        while addr < args.end:
            n = min(regs_per_read, args.end - addr)
            if n % 2:
                n -= 1
            if n < 2:
                break
            regs = _read(fn, addr, n, args.unit_id)
            if regs is None:
                print(f"  {addr:>6}: read failed")
            else:
                any_ok = True
                print(f"  {addr:>6}: {_decode(regs)}")
            addr += n
        if not any_ok:
            print("  (no successful reads on this function code)")
        print()
    c.close()


if __name__ == "__main__":
    main()
