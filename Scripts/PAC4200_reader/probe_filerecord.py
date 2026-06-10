#!/usr/bin/env python3
"""
probe_filerecord.py  -  locate the PAC4200 harmonic data via FC 0x14.

What we learned from the first run:
  * Your meter ANSWERS FC 0x14 (it returned function_code 0x94 = 0x14|0x80,
    i.e. a proper Read-File-Record exception, not "Illegal Function"). So file
    records work.
  * The exception was code 2 (Illegal Data Address) because Siemens file
    records are 1-BASED: the first item in a file is at offset/record 1, never
    0. The earlier probe asked for record 0, so everything errored.

Model (from the SENTRON PAC Modbus tables): inside a file the data is a flat,
1-based register array. You read it with FC 0x14 as (file_number,
record_number = register offset, record_length = number of registers). The
whole harmonic spectrum (all quantities x 64 orders) lives in ONE file, at
offsets 1..~1152.

MODE A (default): find the file.
    python probe_filerecord.py --host 192.168.168.1

MODE B (after you know the file): dump it so you can see where each quantity's
spectrum (its fundamental, then the decaying harmonics) sits.
    python probe_filerecord.py --host 192.168.168.1 --file <N>
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

FileRecord = None
for _mod in ("pymodbus.pdu.file_message", "pymodbus.file_message"):
    try:
        FileRecord = __import__(_mod, fromlist=["FileRecord"]).FileRecord
        break
    except Exception:
        pass


def _detect_unit_kwarg():
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


def _floats_from_bytes(data: bytes):
    """Decode bytes as big-endian 32-bit floats (2 registers each)."""
    out = []
    for i in range(0, len(data) - 3, 4):
        out.append(round(struct.unpack(">f", data[i:i + 4])[0], 3))
    return out


def _read_file(client, unit, file_no, record_no, n_registers):
    # record_no is a 1-BASED register offset within the file.
    # pymodbus FileRecord takes record_length in BYTES then halves it to
    # registers, so to read N registers we pass 2*N.
    fr = FileRecord(file_number=file_no, record_number=record_no,
                    record_length=2 * n_registers)
    kwargs = {_UNIT: unit} if _UNIT else {}
    try:
        rr = client.read_file_record([fr], **kwargs)
    except Exception as e:                       # noqa: BLE001
        return None, f"exception: {e}"
    if rr is None or rr.isError():
        code = getattr(rr, "exception_code", None)
        return None, f"exc_code={code}" if code else f"error: {rr}"
    recs = getattr(rr, "records", None)
    if not recs:
        return None, "no records returned"
    return getattr(recs[0], "record_data", b""), None


def main():
    ap = argparse.ArgumentParser(description="Locate PAC4200 harmonics via FC 0x14.")
    ap.add_argument("--host", default="192.168.168.1")
    ap.add_argument("--port", type=int, default=502)
    ap.add_argument("--unit-id", type=int, default=1)
    ap.add_argument("--file-start", type=int, default=1)
    ap.add_argument("--file-end", type=int, default=256)
    ap.add_argument("--file", type=int, default=None,
                    help="MODE B: dump this file's contents across offsets")
    ap.add_argument("--window", type=int, default=40,
                    help="MODE B: registers per read (default 40 -> 20 floats)")
    ap.add_argument("--max-offset", type=int, default=1300,
                    help="MODE B: dump offsets 1..this (default 1300)")
    args = ap.parse_args()

    if FileRecord is None:
        print("Could not import FileRecord from pymodbus.", file=sys.stderr)
        sys.exit(1)

    c = ModbusTcpClient(args.host, port=args.port, timeout=1.0)
    if not c.connect():
        print(f"cannot connect to {args.host}:{args.port}", file=sys.stderr)
        sys.exit(1)
    print(f"pymodbus unit kwarg = {_UNIT or 'positional'}\n")

    # ----- MODE B: dump one file -----
    if args.file is not None:
        print(f"[FC14] dumping file {args.file}, offsets 1..{args.max_offset} "
              f"in {args.window}-register windows:")
        off = 1
        while off <= args.max_offset:
            data, err = _read_file(c, args.unit_id, args.file, off, args.window)
            if data:
                print(f"   off {off:>5}: {_floats_from_bytes(data)}")
            else:
                print(f"   off {off:>5}: {err}")
            off += args.window
        print("\nA quantity block = one big value (its fundamental: ~load amps "
              "for current,\n~230 for voltage) then ~63 small decaying values. "
              "Note the offset of each\nquantity's fundamental; order 2 is two "
              "registers later (offset + 2).")
        c.close()
        return

    # ----- MODE A: confirm + find the file -----
    print("[FC14] mechanism check - record 1 of a few low files (1-based):")
    for fno in (1, 2, 5, 90, 91):
        data, err = _read_file(c, args.unit_id, fno, 1, 4)
        if data:
            print(f"    file {fno:>3} rec 1: {_floats_from_bytes(data)}")
        else:
            print(f"    file {fno:>3} rec 1: {err}")

    print(f"\n[FC14] scanning files {args.file_start}..{args.file_end} "
          f"(record 1, 8 regs). Hits:")
    hits = []
    for fno in range(args.file_start, args.file_end + 1):
        data, err = _read_file(c, args.unit_id, fno, 1, 8)
        if data:
            hits.append(fno)
            print(f"   file {fno:>3}: {_floats_from_bytes(data)}")
    if not hits:
        print("   (no file returned data - widen --file-end, e.g. 1024)")
    else:
        print(f"\nFiles that exist: {hits}")
        print("Re-run with --file <N> on each to find the harmonic block "
              "(current in A,\nvoltage in %, the fundamental followed by "
              "decaying orders).")
    c.close()


if __name__ == "__main__":
    main()