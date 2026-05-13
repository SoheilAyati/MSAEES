#!/usr/bin/env python3
"""
NILM Project — PAC4200 Modbus TCP Reader
=========================================
Polls a Siemens SENTRON PAC4200 at the PCC over Modbus TCP at ~5 Hz, decodes
the register map into the channel set defined by the data format spec, and
writes a scenario HDF5 file compatible with `preprocessor.py`.

This script is the REAL-DATA counterpart to `aggregator.py`. Once a recording
session ends, the downstream pipeline is identical:
    pac4200_reader.py  →  scenario.h5  →  preprocessor.py  →  Milestone 2

For Milestone 1 (no PAC4200 connected yet), run with --simulate to generate
synthetic Modbus responses and verify the full file-writing path works.
For Milestone 2 (PAC4200 in the lab), drop --simulate and supply --host.

Usage
-----
    # M1: smoke test the script without hardware
    python pac_reader.py --simulate --duration 60 --output sim.h5

    # M2: record from the real PAC4200
    python pac_reader.py --host 192.168.1.50 --duration 86400 \
        --output Synthetic_Data/real_session_001/scenario.h5

    # Inspect register map being used
    python pac_reader.py --print-register-map
"""

import argparse
import json
import math
import os
import queue
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py not installed. Run: pip install h5py", file=sys.stderr)
    sys.exit(1)

try:
    from pymodbus.client import ModbusTcpClient
    from pymodbus.exceptions import ModbusException
    HAVE_PYMODBUS = True
except ImportError:
    HAVE_PYMODBUS = False  # OK in --simulate mode; required for real polling


# =============================================================================
# Constants
# =============================================================================

READER_VERSION = "0.1.0"
FORMAT_VERSION = "0.1"
N_HARMONICS = 39                  # 2nd through 40th, matching data format spec
DEFAULT_SAMPLE_RATE_HZ = 5.0


# =============================================================================
# PAC4200 register map
# =============================================================================
#
# Each register pair holds one IEEE 754 32-bit float, big-endian by default
# on PAC4200 (high word first, then low word; within each word, big-endian
# bytes). Some installations swap word order — see SwapMode below.
#
# Register addresses below are taken from the Siemens SENTRON PAC4200 Modbus
# Measured Values documentation. They are 0-based logical addresses; some
# Modbus clients expect 1-based (add 1 in pymodbus's address field if so).
#
# NOTE: For M2, verify the first few reads against the meter's display to
# confirm the addresses haven't drifted across firmware revisions.

REGISTER_MAP: Dict[str, int] = {
    # Voltage (line-to-neutral RMS)  ── 32-bit floats, 2 registers each
    "V_L1": 1,
    "V_L2": 3,
    "V_L3": 5,
    # Currents
    "I_L1": 13,
    "I_L2": 15,
    "I_L3": 17,
    "I_N":  19,
    # Per-phase apparent / active / reactive power
    "S_L1": 25, "S_L2": 27, "S_L3": 29,
    "P_L1": 31, "P_L2": 33, "P_L3": 35,
    "Q_L1": 37, "Q_L2": 39, "Q_L3": 41,
    # Power factor (signed: + = consumption, − = generation, per PAC4200 convention)
    "PF_L1": 43, "PF_L2": 45, "PF_L3": 47,
    # Total values across all three phases
    "S_total": 65, "P_total": 67, "Q_total": 69, "PF_total": 71,
    # Frequency
    "freq": 55,
    # THD (per phase, voltage and current) — percent
    "THD_V_L1": 201, "THD_V_L2": 203, "THD_V_L3": 205,
    "THD_I_L1": 207, "THD_I_L2": 209, "THD_I_L3": 211,
    # cos φ (displacement only)
    "cosphi_L1": 215, "cosphi_L2": 217, "cosphi_L3": 219,
    "cosphi_total": 221,
}

# Harmonic registers — magnitudes for V and I, per phase, orders 2..40.
# PAC4200 supports orders 1..64 natively; we read 2..40 to match the data
# format spec. Addresses are sequential starting at the base.
HARMONIC_BASE_REGISTERS = {
    "H_V_L1_mag_base": 401,   # 401..(401 + 2*64 - 1) covers orders 1..64
    "H_V_L2_mag_base": 529,
    "H_V_L3_mag_base": 657,
    "H_I_L1_mag_base": 785,
    "H_I_L2_mag_base": 913,
    "H_I_L3_mag_base": 1041,
}
# NOTE: Harmonic phase angles are not exposed by every PAC4200 firmware
# revision. We attempt to read them from these base addresses but tolerate
# absence (fill with zeros and warn).
HARMONIC_PHASE_BASE_REGISTERS = {
    "H_V_L1_phase_base": 1169,
    "H_V_L2_phase_base": 1297,
    "H_V_L3_phase_base": 1425,
    "H_I_L1_phase_base": 1553,
    "H_I_L2_phase_base": 1681,
    "H_I_L3_phase_base": 1809,
}


class SwapMode:
    """Word order within a 32-bit float register pair."""
    BIG_WORD_FIRST = "big_word_first"          # PAC4200 default
    LITTLE_WORD_FIRST = "little_word_first"    # some Siemens setups


# =============================================================================
# Sample container
# =============================================================================

@dataclass
class Sample:
    """One poll cycle's worth of decoded channel values."""
    timestamp_us: int
    scalars: Dict[str, float] = field(default_factory=dict)
    h_I_mag: Dict[str, np.ndarray] = field(default_factory=dict)   # per phase, shape (39,)
    h_I_phase: Dict[str, np.ndarray] = field(default_factory=dict)
    h_V_mag: Dict[str, np.ndarray] = field(default_factory=dict)
    h_V_phase: Dict[str, np.ndarray] = field(default_factory=dict)


# =============================================================================
# Float decoding from Modbus register pairs
# =============================================================================

def decode_float(reg_high: int, reg_low: int, swap: str = SwapMode.BIG_WORD_FIRST
                 ) -> float:
    """Two 16-bit register words → one IEEE 754 32-bit float."""
    if swap == SwapMode.BIG_WORD_FIRST:
        packed = struct.pack(">HH", reg_high, reg_low)
    else:
        packed = struct.pack(">HH", reg_low, reg_high)
    return struct.unpack(">f", packed)[0]


# =============================================================================
# Reader implementations
# =============================================================================

class PAC4200Reader:
    """Abstract interface. Two implementations follow:
       - ModbusReader: real PAC4200 over TCP (M2)
       - SimulatedReader: deterministic synthetic responses (M1 testing)"""
    def connect(self): ...
    def disconnect(self): ...
    def read_sample(self) -> Optional[Sample]: ...


class ModbusReader(PAC4200Reader):
    """Talks to a real PAC4200 over Modbus TCP. Used in M2."""

    def __init__(self, host: str, port: int = 502, unit_id: int = 1,
                 swap: str = SwapMode.BIG_WORD_FIRST,
                 read_harmonics: bool = True):
        if not HAVE_PYMODBUS:
            raise RuntimeError(
                "pymodbus not installed. Run: pip install pymodbus")
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.swap = swap
        self.read_harmonics = read_harmonics
        self.client: Optional[ModbusTcpClient] = None
        self._harmonic_phases_available = True  # auto-detected on first read

    def connect(self):
        self.client = ModbusTcpClient(self.host, port=self.port, timeout=1.0)
        if not self.client.connect():
            raise ConnectionError(f"Cannot connect to PAC4200 at {self.host}:{self.port}")

    def disconnect(self):
        if self.client is not None:
            self.client.close()

    def _read_floats(self, address: int, count: int) -> Optional[List[float]]:
        """Read `count` consecutive IEEE 754 floats starting at register `address`."""
        rr = self.client.read_holding_registers(address=address, count=count * 2,
                                                 slave=self.unit_id)
        if rr.isError():
            return None
        regs = rr.registers
        out = []
        for i in range(count):
            out.append(decode_float(regs[2 * i], regs[2 * i + 1], self.swap))
        return out

    def read_sample(self) -> Optional[Sample]:
        ts_us = int(time.time() * 1e6)
        s = Sample(timestamp_us=ts_us)

        # ----- Scalar channels -----
        # Group reads to minimise round-trips. PAC4200 supports up to ~125
        # registers per request; we batch into blocks.
        try:
            for ch, addr in REGISTER_MAP.items():
                vals = self._read_floats(addr, 1)
                if vals is None:
                    return None  # propagate the dropout
                s.scalars[ch] = vals[0]
        except (ModbusException, OSError):
            return None

        # ----- Harmonic magnitudes (orders 2..40) -----
        if self.read_harmonics:
            for key, base in HARMONIC_BASE_REGISTERS.items():
                # We want orders 2..40 → registers from base+2*1=base+2 to base+2*39=base+78
                vals = self._read_floats(base + 2, N_HARMONICS)
                if vals is None:
                    return None
                phase = key.split("_")[2]
                if "V" in key:
                    s.h_V_mag[phase] = np.asarray(vals, dtype=np.float32)
                else:
                    s.h_I_mag[phase] = np.asarray(vals, dtype=np.float32)

            # ----- Harmonic phases (optional — auto-detect availability) -----
            if self._harmonic_phases_available:
                try:
                    for key, base in HARMONIC_PHASE_BASE_REGISTERS.items():
                        vals = self._read_floats(base + 2, N_HARMONICS)
                        if vals is None:
                            self._harmonic_phases_available = False
                            break
                        phase = key.split("_")[2]
                        if "V" in key:
                            s.h_V_phase[phase] = np.asarray(vals, dtype=np.float32)
                        else:
                            s.h_I_phase[phase] = np.asarray(vals, dtype=np.float32)
                except (ModbusException, OSError):
                    self._harmonic_phases_available = False

            # Fill missing phases with zeros so downstream shapes are consistent
            for ph in ("L1", "L2", "L3"):
                s.h_I_mag.setdefault(ph, np.zeros(N_HARMONICS, dtype=np.float32))
                s.h_V_mag.setdefault(ph, np.zeros(N_HARMONICS, dtype=np.float32))
                s.h_I_phase.setdefault(ph, np.zeros(N_HARMONICS, dtype=np.float32))
                s.h_V_phase.setdefault(ph, np.zeros(N_HARMONICS, dtype=np.float32))

        return s


class SimulatedReader(PAC4200Reader):
    """Generates deterministic synthetic Modbus responses for M1 testing
    without a connected PAC4200. Values are NOT realistic loads — they're
    constants plus slow drift, designed to verify the full file-writing
    pipeline before real hardware is available."""

    def __init__(self, seed: int = 0, dropout_rate: float = 0.0,
                 read_harmonics: bool = True):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dropout_rate = dropout_rate
        self.read_harmonics = read_harmonics
        self._t0: Optional[float] = None

    def connect(self):
        self._t0 = time.time()

    def disconnect(self):
        pass

    def read_sample(self) -> Optional[Sample]:
        # Random dropout to exercise the gap-handling code path
        if self.dropout_rate > 0 and self.rng.uniform() < self.dropout_rate:
            return None

        now = time.time()
        t = now - (self._t0 or now)
        ts_us = int(now * 1e6)
        s = Sample(timestamp_us=ts_us)

        # Deterministic-ish but slowly varying scalar values
        V = 230.0 + 1.5 * math.sin(t / 31.0)
        I_per_phase = 5.0 + 1.2 * math.sin(t / 17.0)
        s.scalars = {
            "V_L1": V, "V_L2": V * 0.998, "V_L3": V * 1.002,
            "I_L1": I_per_phase, "I_L2": I_per_phase * 0.95,
            "I_L3": I_per_phase * 1.05, "I_N": I_per_phase * 0.15,
            "P_L1": 800.0, "P_L2": 750.0, "P_L3": 820.0,
            "P_total": 2370.0,
            "Q_L1": 150.0, "Q_L2": 130.0, "Q_L3": 170.0,
            "Q_total": 450.0,
            "S_L1": 813.5, "S_L2": 761.2, "S_L3": 835.7,
            "S_total": 2410.0,
            "PF_L1": 0.984, "PF_L2": 0.985, "PF_L3": 0.981,
            "PF_total": 0.983,
            "cosphi_L1": 0.984, "cosphi_L2": 0.985, "cosphi_L3": 0.981,
            "cosphi_total": 0.983,
            "freq": 50.0 + 0.02 * math.sin(t / 47.0),
            "THD_V_L1": 1.8, "THD_V_L2": 1.9, "THD_V_L3": 2.0,
            "THD_I_L1": 3.5, "THD_I_L2": 4.2, "THD_I_L3": 3.8,
        }

        if self.read_harmonics:
            for ph in ("L1", "L2", "L3"):
                # Plausible harmonic shape: 3rd dominant, decreasing with order
                orders = np.arange(2, 41)
                mag_i = (0.5 / orders ** 1.2 * I_per_phase).astype(np.float32)
                mag_v = (0.02 / orders ** 1.5 * V).astype(np.float32)
                s.h_I_mag[ph] = mag_i
                s.h_V_mag[ph] = mag_v
                # Fixed-ish phase angles
                s.h_I_phase[ph] = (0.5 * np.ones(N_HARMONICS)).astype(np.float32)
                s.h_V_phase[ph] = (0.1 * np.ones(N_HARMONICS)).astype(np.float32)

        return s


# =============================================================================
# Incremental HDF5 writer
# =============================================================================

class IncrementalHDF5Writer:
    """Appends samples to an HDF5 file with resizable datasets. Flushes at
    configurable intervals so a crash loses at most one flush window."""

    SCALAR_CHANNELS = list(REGISTER_MAP.keys())

    def __init__(self, path: str, sample_rate_hz: float,
                 anchor_datetime: datetime, expected_samples: int,
                 flush_every: int = 150):
        self.path = path
        self.sample_rate_hz = sample_rate_hz
        self.anchor_datetime = anchor_datetime
        self.flush_every = flush_every
        self._buffer: List[Sample] = []
        self._n_written = 0
        self._gap_count = 0
        self._f: Optional[h5py.File] = None
        self._init_file(expected_samples)

    def _init_file(self, hint_size: int):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._f = h5py.File(self.path, "w")
        max_shape = (None,)
        chunk = (max(64, min(hint_size, 1024)),)
        # Top-level timestamp
        self._f.create_dataset("timestamp", shape=(0,), maxshape=max_shape,
                                dtype="int64", chunks=chunk, compression="lzf")
        m = self._f.create_group("measurements")
        for ch in self.SCALAR_CHANNELS:
            m.create_dataset(ch, shape=(0,), maxshape=max_shape,
                             dtype="float32", chunks=chunk, compression="lzf")
        h = m.create_group("harmonics")
        for ph in ("L1", "L2", "L3"):
            for kind in ("I_mag", "I_phase", "V_mag", "V_phase"):
                h.create_dataset(f"{kind}_{ph}",
                                 shape=(0, N_HARMONICS),
                                 maxshape=(None, N_HARMONICS),
                                 dtype="float32",
                                 chunks=(chunk[0], N_HARMONICS),
                                 compression="lzf")
        md = self._f.create_group("metadata")
        md.attrs["format_version"] = FORMAT_VERSION
        md.attrs["reader_version"] = READER_VERSION
        md.attrs["sample_rate_hz"] = self.sample_rate_hz
        md.attrs["anchor_datetime"] = self.anchor_datetime.isoformat()
        md.attrs["source"] = "pac4200_reader"

    def add(self, sample: Optional[Sample]):
        """Append a sample, or record a gap (NaN) if sample is None."""
        if sample is None:
            self._buffer.append(None)
            self._gap_count += 1
        else:
            self._buffer.append(sample)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        n = len(self._buffer)
        ts = np.zeros(n, dtype=np.int64)
        scalars = {ch: np.full(n, np.nan, dtype=np.float32)
                   for ch in self.SCALAR_CHANNELS}
        h_arrays = {(kind, ph): np.zeros((n, N_HARMONICS), dtype=np.float32)
                    for kind in ("I_mag", "I_phase", "V_mag", "V_phase")
                    for ph in ("L1", "L2", "L3")}

        for i, s in enumerate(self._buffer):
            if s is None:
                # Gap: NaN scalars, zero harmonics. preprocessor will impute.
                ts[i] = self._estimated_ts(i)
                continue
            ts[i] = s.timestamp_us
            for ch in self.SCALAR_CHANNELS:
                scalars[ch][i] = np.float32(s.scalars.get(ch, np.nan))
            for ph in ("L1", "L2", "L3"):
                h_arrays[("I_mag", ph)][i] = s.h_I_mag.get(ph,
                    np.zeros(N_HARMONICS, dtype=np.float32))
                h_arrays[("I_phase", ph)][i] = s.h_I_phase.get(ph,
                    np.zeros(N_HARMONICS, dtype=np.float32))
                h_arrays[("V_mag", ph)][i] = s.h_V_mag.get(ph,
                    np.zeros(N_HARMONICS, dtype=np.float32))
                h_arrays[("V_phase", ph)][i] = s.h_V_phase.get(ph,
                    np.zeros(N_HARMONICS, dtype=np.float32))

        self._append("/timestamp", ts)
        for ch in self.SCALAR_CHANNELS:
            self._append(f"/measurements/{ch}", scalars[ch])
        for (kind, ph), arr in h_arrays.items():
            self._append(f"/measurements/harmonics/{kind}_{ph}", arr)
        self._n_written += n
        self._buffer.clear()
        self._f.flush()

    def _estimated_ts(self, i_in_buffer: int) -> int:
        """Synthesize a timestamp for a missing sample using buffer neighbours."""
        for j in range(i_in_buffer - 1, -1, -1):
            if self._buffer[j] is not None:
                dt_us = int(round(1e6 / self.sample_rate_hz))
                return self._buffer[j].timestamp_us + dt_us * (i_in_buffer - j)
        for j in range(i_in_buffer + 1, len(self._buffer)):
            if self._buffer[j] is not None:
                dt_us = int(round(1e6 / self.sample_rate_hz))
                return self._buffer[j].timestamp_us - dt_us * (j - i_in_buffer)
        return int(time.time() * 1e6)

    def _append(self, path: str, arr: np.ndarray):
        ds = self._f[path]
        old = ds.shape[0]
        ds.resize((old + arr.shape[0],) + arr.shape[1:])
        ds[old:old + arr.shape[0]] = arr

    def close(self, recording_summary: dict):
        self.flush()
        if self._f is None:
            return
        md = self._f["metadata"]
        md.attrs["recording_summary"] = json.dumps(recording_summary)
        self._f.close()
        self._f = None

    @property
    def n_samples(self):
        return self._n_written + len(self._buffer)

    @property
    def n_gaps(self):
        return self._gap_count


# =============================================================================
# Acquisition loop (threaded)
# =============================================================================

class AcquisitionLoop:
    """Polls the reader at the requested rate from a dedicated thread and
    drops decoded Samples onto a queue. The main thread drains the queue
    into the HDF5 writer."""

    def __init__(self, reader: PAC4200Reader, sample_rate_hz: float,
                 sample_queue: "queue.Queue"):
        self.reader = reader
        self.sample_rate_hz = sample_rate_hz
        self.queue = sample_queue
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.late_polls = 0
        self.failed_polls = 0

    def start(self):
        self.reader.connect()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 5.0):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=join_timeout)
        self.reader.disconnect()

    def _run(self):
        dt_s = 1.0 / self.sample_rate_hz
        next_tick = time.time()
        while not self._stop_event.is_set():
            next_tick += dt_s
            sample = self.reader.read_sample()
            if sample is None:
                self.failed_polls += 1
            self.queue.put(sample)
            now = time.time()
            sleep_s = next_tick - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                self.late_polls += 1  # falling behind the target rate
                next_tick = now


# =============================================================================
# Main recording orchestration
# =============================================================================

def record_session(reader: PAC4200Reader, output_path: str,
                   sample_rate_hz: float, duration_s: float,
                   anchor_datetime: datetime, flush_every: int = 150
                   ) -> dict:
    """Run a complete recording: open reader, poll, write, close."""
    expected = int(sample_rate_hz * duration_s)
    writer = IncrementalHDF5Writer(output_path, sample_rate_hz,
                                    anchor_datetime, expected, flush_every)
    q: "queue.Queue" = queue.Queue(maxsize=expected * 2 + 1000)
    loop = AcquisitionLoop(reader, sample_rate_hz, q)

    # Graceful Ctrl-C
    stop_requested = {"flag": False}
    def _sigint(_a, _b): stop_requested["flag"] = True
    old_handler = signal.signal(signal.SIGINT, _sigint)

    t_start = time.time()
    t_end = t_start + duration_s
    loop.start()
    last_report = t_start
    try:
        while time.time() < t_end and not stop_requested["flag"]:
            try:
                sample = q.get(timeout=0.5)
            except queue.Empty:
                continue
            writer.add(sample)
            now = time.time()
            if now - last_report > 5.0:
                pct = (now - t_start) / duration_s * 100
                rate = writer.n_samples / max(1e-3, now - t_start)
                print(f"  [{pct:5.1f}%]  samples={writer.n_samples}  "
                      f"gaps={writer.n_gaps}  effective_rate={rate:.2f} Hz",
                      flush=True)
                last_report = now
    finally:
        loop.stop()
        # Drain remaining queue
        while True:
            try:
                writer.add(q.get_nowait())
            except queue.Empty:
                break
        signal.signal(signal.SIGINT, old_handler)

    summary = {
        "reader_version": READER_VERSION,
        "anchor_datetime": anchor_datetime.isoformat(),
        "configured_sample_rate_hz": sample_rate_hz,
        "configured_duration_s": duration_s,
        "actual_duration_s": time.time() - t_start,
        "n_samples": writer.n_samples,
        "n_gaps": writer.n_gaps,
        "failed_polls": loop.failed_polls,
        "late_polls": loop.late_polls,
        "interrupted": stop_requested["flag"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    writer.close(summary)
    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="PAC4200 Modbus TCP reader → scenario HDF5 file. "
                    "Use --simulate to test without hardware.")
    p.add_argument("--host", default=None,
                   help="PAC4200 IP address (required unless --simulate)")
    p.add_argument("--port", type=int, default=502, help="Modbus TCP port (default 502)")
    p.add_argument("--unit-id", type=int, default=1,
                   help="Modbus slave/unit ID (default 1)")
    p.add_argument("--swap", default=SwapMode.BIG_WORD_FIRST,
                   choices=[SwapMode.BIG_WORD_FIRST, SwapMode.LITTLE_WORD_FIRST],
                   help="Word order within 32-bit float register pairs")
    p.add_argument("--rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ,
                   help=f"poll rate in Hz (default {DEFAULT_SAMPLE_RATE_HZ})")
    p.add_argument("--duration", type=float, default=60.0,
                   help="recording duration in seconds (default 60)")
    p.add_argument("--output", default="recording.h5", help="output HDF5 path")
    p.add_argument("--flush-every", type=int, default=150,
                   help="flush to disk every N samples (default 150 = 30 s at 5 Hz)")
    p.add_argument("--no-harmonics", action="store_true",
                   help="skip harmonic registers (faster polling)")
    p.add_argument("--simulate", action="store_true",
                   help="use SimulatedReader instead of real Modbus (no hardware needed)")
    p.add_argument("--sim-dropout-rate", type=float, default=0.0,
                   help="simulator: probability per sample of a dropped poll (default 0)")
    p.add_argument("--print-register-map", action="store_true",
                   help="print the PAC4200 register map being used and exit")
    args = p.parse_args()

    if args.print_register_map:
        print("PAC4200 register map (0-based addresses, 2 registers per float):")
        for ch, addr in REGISTER_MAP.items():
            print(f"  {ch:15s}  reg {addr:5d}–{addr+1:<5d}")
        print("\nHarmonic magnitude bases (registers for orders 1..64):")
        for k, b in HARMONIC_BASE_REGISTERS.items():
            print(f"  {k:20s}  reg {b}")
        print("\nHarmonic phase bases:")
        for k, b in HARMONIC_PHASE_BASE_REGISTERS.items():
            print(f"  {k:22s}  reg {b}")
        return

    if not args.simulate and args.host is None:
        p.error("--host is required unless --simulate")
    if args.simulate and not HAVE_PYMODBUS:
        print("(pymodbus not installed — that's OK for --simulate)")

    anchor = datetime.now(timezone.utc).replace(microsecond=0)

    if args.simulate:
        reader = SimulatedReader(seed=0, dropout_rate=args.sim_dropout_rate,
                                  read_harmonics=not args.no_harmonics)
    else:
        reader = ModbusReader(host=args.host, port=args.port,
                               unit_id=args.unit_id, swap=args.swap,
                               read_harmonics=not args.no_harmonics)

    print(f"Recording {args.duration} s at {args.rate} Hz "
          f"({'simulated' if args.simulate else f'live: {args.host}:{args.port}'}) "
          f"→ {args.output}")
    print(f"Press Ctrl-C to stop early; the file will be flushed and closed cleanly.")
    summary = record_session(reader, args.output, args.rate, args.duration,
                              anchor, args.flush_every)
    print()
    print("Recording summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nNext step:  python preprocessor.py --input {args.output}")


if __name__ == "__main__":
    main()