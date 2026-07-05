#!/usr/bin/env python3
"""
probe_harmonic_registers.py -- test the plain-register harmonic hypothesis.

The FC 0x14 file-record approach only ever returned denormals (a counter/index
field, not magnitudes), so this probes the plain-register location another group
reported: current harmonics around registers 11007..11091, stride 6.

RUN WITH THE FAN RUNNING -- the harmonic registers read zero at idle, so an idle
probe tells you nothing.

It reads a span that brackets the reported block (to cover the 0-based vs 1-based
ambiguity), tries both holding (FC 0x03) and input (FC 0x04) registers, and
decodes every register pair as float32 in both word orders. Match the printout
against the meter's own Current Spectrum display:

    order 1 (fundamental) ~ 0.120 A
    order 3               ~ 0.014 A
    order 5               ~ 0.002 A
    order 7               ~ 0.001 A

Whichever start address + word order reproduces that decay is the truth; the gap
between the 0.120 row and the 0.014 row is the real stride.

    python probe_harmonic_registers.py 192.168.168.1
"""
import inspect
import struct
import sys

from pymodbus.client import ModbusTcpClient

HOST  = sys.argv[1] if len(sys.argv) > 1 else "192.168.168.1"
START = 11000        # a little before 11007 to catch 0- vs 1-based addressing
COUNT = 110          # ...through 11109: whole reported block (11007..11091) + margin
UNIT  = 1

client = ModbusTcpClient(HOST, port=502, timeout=2.0)
if not client.connect():
    sys.exit(f"cannot connect to {HOST}:502")

# pymodbus renamed the unit kwarg across versions; detect what this one takes.
params = inspect.signature(client.read_holding_registers).parameters
ukw = "device_id" if "device_id" in params else ("slave" if "slave" in params else None)
kw = {ukw: UNIT} if ukw else {}


def try_read(fn):
    try:
        rr = fn(START, count=COUNT, **kw)
        if rr is not None and not rr.isError():
            return rr
    except Exception as e:                       # noqa: BLE001
        print(f"  {fn.__name__} raised: {e}")
    return None


rr = try_read(client.read_holding_registers)
fc = "0x03 holding"
if rr is None:
    rr = try_read(client.read_input_registers)
    fc = "0x04 input"
if rr is None:
    client.close()
    sys.exit(f"both holding and input reads failed at {START} "
             f"(fan running? right address block?)")

regs = rr.registers
print(f"read OK via FC {fc}, {len(regs)} registers from {START}\n")


def as_float(hi, lo, little=False):
    a, b = (lo, hi) if little else (hi, lo)
    return struct.unpack(">f", struct.pack(">HH", a & 0xFFFF, b & 0xFFFF))[0]


print(f"{'addr':>6}  {'hi':>6} {'lo':>6}   {'float_bigword':>15} {'float_littleword':>17}")
for i in range(0, len(regs) - 1, 2):
    addr = START + i
    big = as_float(regs[i], regs[i + 1])
    lit = as_float(regs[i], regs[i + 1], little=True)
    print(f"{addr:6d}  {regs[i]:6d} {regs[i+1]:6d}   {big:15.5f} {lit:17.5f}")

client.close()