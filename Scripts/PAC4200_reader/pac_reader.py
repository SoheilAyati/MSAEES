#!/usr/bin/env python3
"""
NILM Project - PAC4200 Live Monitor & Session Recorder
======================================================
A single-process tool that:

  1. Opens ONE persistent Modbus TCP connection to a Siemens SENTRON PAC4200
     and keeps it alive (with automatic reconnect + health monitoring).
  2. Continuously polls the meter into a rolling buffer for a LIVE dashboard.
  3. Lets you start / stop a *named* recording for each appliance WITHOUT
     restarting the program or dropping the meter connection. Plug in fridge,
     record "fridge", stop; plug in LED lamp, record "led_lamp", stop; etc.
  4. Writes each recording to a scenario-layout HDF5 file that is byte-for-byte
     compatible with `preprocessor.py` (same downstream pipeline as the
     synthetic `aggregator.py` output).
  5. Serves a browser dashboard with live charts (P / Q / I), a per-appliance
     session panel, and a raw register inspector for commissioning.

The downstream pipeline is unchanged:
    pac_monitor.py  ->  <appliance>.h5  ->  preprocessor.py  ->  Milestone 2

------------------------------------------------------------------------------
REGISTER MAP NOTE (read before trusting any number)
------------------------------------------------------------------------------
The CORE register map below was verified against the Siemens SENTRON PAC4200
"Measured variables without a time stamp (function codes 0x03/0x04)" table.
On the PAC4200, registers 19..73 are: S(19/21/23), P(25/27/29), Q(31/33/35),
PF(37/39/41), THD_V L-L(43/45/47), freq(55), averages(57/59/61),
totals S/P/Q/PF(63/65/67/69), unbalance V/I(71/73).

THD *current* and cos-phi are scalars in a higher register region (varies
across firmware) and go in EXTENDED_CHANNELS once confirmed with the Register
Inspector. The per-order harmonic spectrum is NOT a register block on this
meter: it is read via Modbus FC 0x14 "Read File Record", one file per quantity
(see HARMONIC_I_FILE / HARMONIC_V_FILE). All of this is OFF by default. Siemens'
own manual says to verify the first reads against the device display - do that.

Usage
-----
    # No hardware yet (Milestone-1 style smoke test of the whole path + UI):
    python pac_reader.py --simulate

    # Real meter:
    python pac_reader.py --host 192.168.168.1

    # Record harmonics too (needs verified HARMONIC_*_FILE file numbers):
    python pac_reader.py --host 192.168.168.1 --harmonics

    # Headless long recording of a single appliance (no UI):
    python pac_reader.py --host 192.168.168.1 --headless \
        --label fridge --duration 3600 --output-dir recordings/

    # Print the register map and exit:
    python pac_reader.py --print-register-map
"""

import argparse
import inspect
import json
import math
import os
import re
import struct
import sys
import threading
import time
import webbrowser
from collections import deque
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
    from flask import Flask, jsonify, request, Response
    from flask.json.provider import DefaultJSONProvider
except ImportError:
    print("ERROR: flask not installed. Run: pip install flask", file=sys.stderr)
    sys.exit(1)

try:
    from pymodbus.client import ModbusTcpClient
    from pymodbus.exceptions import ModbusException
    try:
        from pymodbus.pdu.file_message import FileRecord   # pymodbus >= ~3.8
    except ImportError:
        from pymodbus.file_message import FileRecord        # older pymodbus
    HAVE_PYMODBUS = True
except ImportError:
    HAVE_PYMODBUS = False  # fine for --simulate; required for a real meter


# =============================================================================
# Constants
# =============================================================================

APP_VERSION = "1.0.0"
FORMAT_VERSION = "0.1"             # keep in step with the synthetic data format
N_HARMONICS = 39                   # orders 2..40, matching the synthetic spec
DEFAULT_SAMPLE_RATE_HZ = 5.0
LIVE_BUFFER_SAMPLES = 1500         # rolling buffer for the dashboard (~5 min @5Hz)


# =============================================================================
# Verified CORE register map  (Siemens offsets; FC 0x03; 32-bit float = 2 regs)
# =============================================================================
# Two contiguous blocks, deliberately split around the 49..53 reserved gap so a
# single reserved register can never fail the whole read.

CORE_BLOCK_A_START = 1
CORE_BLOCK_A_COUNT = 48            # regs 1..48  -> offsets 1..47
CORE_BLOCK_B_START = 55
CORE_BLOCK_B_COUNT = 20            # regs 55..74 -> offsets 55..73

CORE_REGISTERS: Dict[str, int] = {
    # Voltage, line-to-neutral (V)
    "V_L1": 1, "V_L2": 3, "V_L3": 5,
    # Voltage, line-to-line (V)
    "V_L12": 7, "V_L23": 9, "V_L31": 11,
    # Current (A)
    "I_L1": 13, "I_L2": 15, "I_L3": 17,
    # Apparent power per phase (VA)
    "S_L1": 19, "S_L2": 21, "S_L3": 23,
    # Active power per phase (W)
    "P_L1": 25, "P_L2": 27, "P_L3": 29,
    # Reactive power per phase (var)
    "Q_L1": 31, "Q_L2": 33, "Q_L3": 35,
    # Power factor per phase (signed)
    "PF_L1": 37, "PF_L2": 39, "PF_L3": 41,
    # THD voltage, line-to-line (%)
    "THD_V_L12": 43, "THD_V_L23": 45, "THD_V_L31": 47,
    # Frequency (Hz)
    "freq": 55,
    # 3-phase averages
    "V_avg_LN": 57, "V_avg_LL": 59, "I_avg": 61,
    # Totals over all phases
    "S_total": 63, "P_total": 65, "Q_total": 67, "PF_total": 69,
    # Amplitude unbalance (%)
    "unbalance_V": 71, "unbalance_I": 73,
}

# Channels shown as live charts on the dashboard (must be in CORE_REGISTERS).
CHART_CHANNELS = ["P_total", "P_L1", "P_L2", "P_L3",
                  "Q_total", "I_L1", "I_L2", "I_L3"]

# -----------------------------------------------------------------------------
# EXTENDED channels - addresses NOT verified, OFF by default. Confirm each one
# with the Register Inspector (compare the decoded value to the meter display)
# before enabling. Edit the addresses here once you have confirmed them.
# These are SCALARS (one float each) and flow into the file like any other
# measurement channel - they are NOT the per-order spectrum (see below).
# -----------------------------------------------------------------------------
EXTENDED_CHANNELS: Dict[str, int] = {
    # "THD_I_L1": 0, "THD_I_L2": 0, "THD_I_L3": 0,
    # "cosphi_L1": 0, "cosphi_L2": 0, "cosphi_L3": 0, "cosphi_total": 0,
}

# -----------------------------------------------------------------------------
# HARMONIC spectrum location (FC 0x14 Read File Record).
#
# On this PAC4200 the per-order spectrum is NOT a plain register block (every
# read in the 36000+ region returns Illegal Data Address). It is served via
# Modbus FC 0x14 "Read File Record": ONE FILE PER QUANTITY, with the 64 orders
# stored at 1-BASED register offsets 1, 3, 5, ... (order n at offset 2n-1).
# Order 1 is the fundamental; orders 2..40 (N_HARMONICS = 39) therefore start at
# offset 3 and span 78 registers.
#
# Fill these with the file numbers you confirm with probe_filerecord.py + the
# meter's harmonic display (display 17.0 = Harmonic I, 15.0 = Harmonic U L-N).
# A value of 0 disables that quantity (its arrays stay zero).
#
#   Voltage L-N (referred to fundamental) was identified as files 110/116/118
#   (231.9 V fundamentals, clean odd-order dominance).
#   Current was most likely files 101/102/103 (all-zero at idle) - YOU MUST
#   confirm these with a real load drawing current before trusting them, and
#   confirm which file is L1/L2/L3 against the display's phase labels.
#
# The meter gives MAGNITUDES only (current in A, voltage in % of fundamental),
# not per-order phase, so the h_*_phase datasets stay zero on real recordings.
# -----------------------------------------------------------------------------
HARMONIC_I_FILE: Dict[str, int] = {"L1": 101, "L2": 102, "L3": 103}  # VERIFY w/ load
HARMONIC_V_FILE: Dict[str, int] = {"L1": 110, "L2": 116, "L3": 118}  # U L-N, verify phase order
HARMONIC_ORDER1_OFFSET = 1   # 1-based file offset of order 1 (the fundamental)


class SwapMode:
    """Word order within a 32-bit float register pair."""
    BIG_WORD_FIRST = "big_word_first"        # PAC4200 default (high word first)
    LITTLE_WORD_FIRST = "little_word_first"


# =============================================================================
# pymodbus version adapter
# =============================================================================
# pymodbus renamed the unit/slave keyword across versions:
#   <=3.6 : positional 'slave'
#   3.7-3.8: keyword 'slave'
#   3.9+  : keyword 'device_id'   ('slave' raises TypeError)
# Detect what the installed version accepts so the same code runs everywhere.

def _detect_unit_kwarg() -> Optional[str]:
    if not HAVE_PYMODBUS:
        return None
    try:
        params = inspect.signature(ModbusTcpClient.read_holding_registers).parameters
    except (ValueError, TypeError):
        return "device_id"
    if "device_id" in params:
        return "device_id"
    if "slave" in params:
        return "slave"
    return None  # fall back to positional


_UNIT_KWARG = _detect_unit_kwarg()


def decode_float(reg_high: int, reg_low: int,
                 swap: str = SwapMode.BIG_WORD_FIRST) -> float:
    """Two 16-bit register words -> one IEEE-754 32-bit float.

    Done with struct (not pymodbus BinaryPayloadDecoder, which was removed in
    pymodbus 3.9) so decoding is independent of the pymodbus version.
    """
    if swap == SwapMode.BIG_WORD_FIRST:
        packed = struct.pack(">HH", reg_high & 0xFFFF, reg_low & 0xFFFF)
    else:
        packed = struct.pack(">HH", reg_low & 0xFFFF, reg_high & 0xFFFF)
    return struct.unpack(">f", packed)[0]


# =============================================================================
# Sample container
# =============================================================================

@dataclass
class Sample:
    """One poll cycle of decoded values."""
    timestamp_us: int
    scalars: Dict[str, float] = field(default_factory=dict)
    h_I_mag: Dict[str, np.ndarray] = field(default_factory=dict)
    h_I_phase: Dict[str, np.ndarray] = field(default_factory=dict)
    h_V_mag: Dict[str, np.ndarray] = field(default_factory=dict)
    h_V_phase: Dict[str, np.ndarray] = field(default_factory=dict)


# =============================================================================
# Readers
# =============================================================================

class BaseReader:
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def read_sample(self) -> Optional[Sample]: ...
    def read_raw(self, address: int, count: int) -> Optional[List[int]]: ...
    @property
    def is_simulated(self) -> bool:
        return False


class ModbusReader(BaseReader):
    """Real PAC4200 over Modbus TCP, using verified block reads."""

    def __init__(self, host: str, port: int = 502, unit_id: int = 1,
                 swap: str = SwapMode.BIG_WORD_FIRST,
                 address_offset: int = 0,
                 extra_channels: Optional[Dict[str, int]] = None,
                 read_harmonics: bool = False,
                 timeout: float = 1.0):
        if not HAVE_PYMODBUS:
            raise RuntimeError("pymodbus not installed. Run: pip install pymodbus")
        self.host = host
        self.port = port
        self.unit_id = unit_id
        self.swap = swap
        # address_offset lets you flip 0-based vs 1-based addressing in one place
        # if voltage etc. read as garbage (the classic Modbus off-by-one).
        self.address_offset = address_offset
        self.extra_channels = extra_channels or {}
        self.read_harmonics = read_harmonics
        self.timeout = timeout
        self.client: Optional[ModbusTcpClient] = None

    def connect(self) -> None:
        self.client = ModbusTcpClient(self.host, port=self.port,
                                      timeout=self.timeout)
        if not self.client.connect():
            raise ConnectionError(
                f"Cannot connect to PAC4200 at {self.host}:{self.port}")

    def disconnect(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

    # --- low level -----------------------------------------------------------
    def _read_block(self, address: int, count: int) -> Optional[List[int]]:
        """Read `count` raw 16-bit registers; returns the register list or None."""
        if self.client is None:
            return None
        addr = address + self.address_offset
        kwargs = {"count": count}
        if _UNIT_KWARG is not None:
            kwargs[_UNIT_KWARG] = self.unit_id
            rr = self.client.read_holding_registers(addr, **kwargs)
        else:
            rr = self.client.read_holding_registers(addr, count, self.unit_id)
        if rr is None or rr.isError():
            return None
        return list(rr.registers)

    def read_raw(self, address: int, count: int) -> Optional[List[int]]:
        try:
            return self._read_block(address, count)
        except (ModbusException, OSError, AttributeError):
            return None

    def _float_at(self, block: List[int], block_start: int, offset: int) -> float:
        i = offset - block_start
        return decode_float(block[i], block[i + 1], self.swap)

    def _read_harmonic_block(self, file_no: int) -> Optional[np.ndarray]:
        """Read N_HARMONICS harmonic magnitudes (orders 2..40) for one quantity
        from its FC 0x14 file.

        One quantity per file; order n is at 1-based register offset 2n-1, so
        orders 2..40 start at offset HARMONIC_ORDER1_OFFSET + 2 and span
        N_HARMONICS*2 registers. Returns float32[N_HARMONICS], or None on an
        unconfigured file (<=0) / failed / short read, in which case that
        quantity simply stays zero in the stored arrays.
        """
        if file_no <= 0 or self.client is None:
            return None
        offset = HARMONIC_ORDER1_OFFSET + 2          # skip order 1 (fundamental)
        n_reg = N_HARMONICS * 2
        # pymodbus FileRecord takes record_length in BYTES then halves it to
        # registers, so to read n_reg registers we pass 2*n_reg.
        fr = FileRecord(file_number=file_no, record_number=offset,
                        record_length=2 * n_reg)
        kwargs = {_UNIT_KWARG: self.unit_id} if _UNIT_KWARG else {}
        try:
            rr = self.client.read_file_record([fr], **kwargs)
        except (ModbusException, OSError, AttributeError):
            return None
        if rr is None or rr.isError():
            return None
        recs = getattr(rr, "records", None)
        if not recs:
            return None
        data = getattr(recs[0], "record_data", b"")
        if len(data) < n_reg * 2:
            return None
        out = np.empty(N_HARMONICS, dtype=np.float32)
        for k in range(N_HARMONICS):
            # big-word-first float = big-endian 4-byte group, matching `swap`
            if self.swap == SwapMode.BIG_WORD_FIRST:
                out[k] = struct.unpack(">f", data[4 * k:4 * k + 4])[0]
            else:
                b = data[4 * k:4 * k + 4]
                out[k] = struct.unpack(">f", b[2:4] + b[0:2])[0]
        return out

    # --- one full sample -----------------------------------------------------
    def read_sample(self) -> Optional[Sample]:
        ts_us = int(time.time() * 1e6)
        s = Sample(timestamp_us=ts_us)
        try:
            block_a = self._read_block(CORE_BLOCK_A_START, CORE_BLOCK_A_COUNT)
            block_b = self._read_block(CORE_BLOCK_B_START, CORE_BLOCK_B_COUNT)
        except (ModbusException, OSError, AttributeError):
            return None
        if block_a is None or block_b is None:
            return None

        for ch, off in CORE_REGISTERS.items():
            if off < CORE_BLOCK_B_START:
                s.scalars[ch] = self._float_at(block_a, CORE_BLOCK_A_START, off)
            else:
                s.scalars[ch] = self._float_at(block_b, CORE_BLOCK_B_START, off)

        # Optional, user-verified extended channels (read individually).
        for ch, off in self.extra_channels.items():
            vals = self.read_raw(off, 1)
            if vals is not None and len(vals) >= 2:
                s.scalars[ch] = decode_float(vals[0], vals[1], self.swap)

        # Optional per-order harmonic magnitudes (only with --harmonics, once the
        # file numbers are filled / verified). Read via FC 0x14, one file per
        # quantity. Magnitudes only - the PAC4200 gives no per-order phase, so
        # h_*_phase stay zero.
        if self.read_harmonics:
            for ph in ("L1", "L2", "L3"):
                im = self._read_harmonic_block(HARMONIC_I_FILE.get(ph, 0))
                if im is not None:
                    s.h_I_mag[ph] = im
                vm = self._read_harmonic_block(HARMONIC_V_FILE.get(ph, 0))
                if vm is not None:
                    s.h_V_mag[ph] = vm

        return s


class SimulatedReader(BaseReader):
    """Deterministic synthetic responses for testing the whole path with no
    hardware. Values are plausible but NOT real loads."""

    def __init__(self, seed: int = 0, dropout_rate: float = 0.0,
                 extra_channels: Optional[Dict[str, int]] = None,
                 read_harmonics: bool = False):
        self.rng = np.random.default_rng(seed)
        self.dropout_rate = dropout_rate
        self.extra_channels = extra_channels or {}
        self.read_harmonics = read_harmonics
        self._t0: Optional[float] = None
        # A simulated "appliance" that can be toggled to make the live charts move.
        self.load_level = 1.0

    @property
    def is_simulated(self) -> bool:
        return True

    def connect(self) -> None:
        self._t0 = time.time()

    def disconnect(self) -> None:
        pass

    def read_raw(self, address: int, count: int) -> Optional[List[int]]:
        # Emit consistent float pairs so the inspector shows clean values.
        out: List[int] = []
        for k in range(0, count, 2):
            packed = struct.pack(">f", 100.0 + address + k)
            hi, lo = struct.unpack(">HH", packed)
            out.extend([hi, lo])
        return out[:count]

    def read_sample(self) -> Optional[Sample]:
        if self.dropout_rate > 0 and self.rng.uniform() < self.dropout_rate:
            return None
        now = time.time()
        t = now - (self._t0 or now)
        s = Sample(timestamp_us=int(now * 1e6))

        V = 230.0 + 1.5 * math.sin(t / 31.0)
        base_I = (4.0 + 1.2 * math.sin(t / 7.0)) * self.load_level
        pf = 0.96
        P1, P2, P3 = (V * base_I * pf * k for k in (1.0, 0.96, 1.03))
        Q1, Q2, Q3 = (V * base_I * 0.28 * k for k in (1.0, 0.9, 1.1))
        S1, S2, S3 = (math.hypot(P1, Q1), math.hypot(P2, Q2), math.hypot(P3, Q3))

        s.scalars = {
            "V_L1": V, "V_L2": V * 0.998, "V_L3": V * 1.002,
            "V_L12": V * 1.732, "V_L23": V * 1.730, "V_L31": V * 1.734,
            "I_L1": base_I, "I_L2": base_I * 0.96, "I_L3": base_I * 1.03,
            "S_L1": S1, "S_L2": S2, "S_L3": S3,
            "P_L1": P1, "P_L2": P2, "P_L3": P3,
            "Q_L1": Q1, "Q_L2": Q2, "Q_L3": Q3,
            "PF_L1": pf, "PF_L2": pf, "PF_L3": pf,
            "THD_V_L12": 1.8, "THD_V_L23": 1.9, "THD_V_L31": 2.0,
            "freq": 50.0 + 0.02 * math.sin(t / 47.0),
            "V_avg_LN": V, "V_avg_LL": V * 1.732, "I_avg": base_I,
            "S_total": S1 + S2 + S3, "P_total": P1 + P2 + P3,
            "Q_total": Q1 + Q2 + Q3, "PF_total": pf,
            "unbalance_V": 0.4, "unbalance_I": 2.1,
        }
        for ch in self.extra_channels:
            s.scalars[ch] = 0.0

        # Synthetic per-order magnitudes so --simulate exercises the harmonic
        # write path the same way the real meter would (magnitudes only, no
        # phase). Orders 2..40 with odd-harmonic dominance (half-wave symmetry),
        # decaying with order and scaled by the simulated load. NOT real data.
        if self.read_harmonics:
            orders = np.arange(2, 2 + N_HARMONICS).astype(np.float32)   # 2..40
            odd = (orders % 2 == 1).astype(np.float32)
            decay = 1.0 / orders
            jitter = (1.0 + 0.05 * self.rng.standard_normal(N_HARMONICS)
                      ).astype(np.float32)
            i_mag = (0.6 * base_I * odd * decay * jitter).astype(np.float32)
            v_mag = (0.015 * V * odd * decay * jitter).astype(np.float32)
            for j, ph in enumerate(("L1", "L2", "L3")):
                scale = (1.0, 0.96, 1.03)[j]
                s.h_I_mag[ph] = np.clip(i_mag * scale, 0.0, None)
                s.h_V_mag[ph] = np.clip(v_mag * scale, 0.0, None)

        return s


# =============================================================================
# Incremental HDF5 writer (scenario layout, preprocessor-compatible)
# =============================================================================

class IncrementalHDF5Writer:
    """Appends samples to resizable datasets and flushes periodically so a crash
    loses at most one flush window. Layout matches aggregator/scenario output:

        /timestamp                         int64  (microseconds, UTC epoch)
        /measurements/<channel>            float32
        /measurements/harmonics/<k>_<ph>   float32  (n, N_HARMONICS)  [if enabled]
        /metadata  (attrs)
    """

    def __init__(self, path: str, channels: List[str], sample_rate_hz: float,
                 anchor_datetime: datetime, appliance_label: str,
                 write_harmonics: bool = False, flush_every: int = 150):
        self.path = path
        self.channels = channels
        self.sample_rate_hz = sample_rate_hz
        self.anchor_datetime = anchor_datetime
        self.appliance_label = appliance_label
        self.write_harmonics = write_harmonics
        self.flush_every = flush_every
        self._buffer: List[Optional[Sample]] = []
        self._n_written = 0
        self._gap_count = 0
        self._f: Optional[h5py.File] = None
        self._init_file()

    def _init_file(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._f = h5py.File(self.path, "w")
        chunk = (512,)
        self._f.create_dataset("timestamp", shape=(0,), maxshape=(None,),
                               dtype="int64", chunks=chunk, compression="lzf")
        m = self._f.create_group("measurements")
        for ch in self.channels:
            m.create_dataset(ch, shape=(0,), maxshape=(None,), dtype="float32",
                             chunks=chunk, compression="lzf")
        if self.write_harmonics:
            h = m.create_group("harmonics")
            for ph in ("L1", "L2", "L3"):
                for kind in ("I_mag", "I_phase", "V_mag", "V_phase"):
                    h.create_dataset(f"{kind}_{ph}", shape=(0, N_HARMONICS),
                                     maxshape=(None, N_HARMONICS), dtype="float32",
                                     chunks=(chunk[0], N_HARMONICS),
                                     compression="lzf")
        md = self._f.create_group("metadata")
        md.attrs["format_version"] = FORMAT_VERSION
        md.attrs["app_version"] = APP_VERSION
        md.attrs["sample_rate_hz"] = self.sample_rate_hz
        md.attrs["anchor_datetime"] = self.anchor_datetime.isoformat()
        md.attrs["source"] = "pac4200_monitor"
        md.attrs["appliance_label"] = self.appliance_label
        md.attrs["channels"] = json.dumps(self.channels)
        md.attrs["harmonics_enabled"] = bool(self.write_harmonics)
        if self.write_harmonics:
            # Be explicit that real PAC4200 harmonic phase is not captured, so
            # downstream code does not mistake the zero-filled phase arrays for
            # a genuine (zero) measurement.
            md.attrs["harmonic_orders"] = json.dumps(
                list(range(2, 2 + N_HARMONICS)))
            md.attrs["harmonic_phase_captured"] = False

    def add(self, sample: Optional[Sample]) -> None:
        self._buffer.append(sample)
        if sample is None:
            self._gap_count += 1
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer or self._f is None:
            return
        n = len(self._buffer)
        ts = np.zeros(n, dtype=np.int64)
        scalars = {ch: np.full(n, np.nan, dtype=np.float32) for ch in self.channels}
        h_arrays = {}
        if self.write_harmonics:
            h_arrays = {(kind, ph): np.zeros((n, N_HARMONICS), dtype=np.float32)
                        for kind in ("I_mag", "I_phase", "V_mag", "V_phase")
                        for ph in ("L1", "L2", "L3")}

        for i, smp in enumerate(self._buffer):
            if smp is None:
                ts[i] = self._estimated_ts(i)        # gap: NaN scalars, est. ts
                continue
            ts[i] = smp.timestamp_us
            for ch in self.channels:
                scalars[ch][i] = np.float32(smp.scalars.get(ch, np.nan))
            if self.write_harmonics:
                for ph in ("L1", "L2", "L3"):
                    h_arrays[("I_mag", ph)][i] = smp.h_I_mag.get(
                        ph, np.zeros(N_HARMONICS, dtype=np.float32))
                    h_arrays[("I_phase", ph)][i] = smp.h_I_phase.get(
                        ph, np.zeros(N_HARMONICS, dtype=np.float32))
                    h_arrays[("V_mag", ph)][i] = smp.h_V_mag.get(
                        ph, np.zeros(N_HARMONICS, dtype=np.float32))
                    h_arrays[("V_phase", ph)][i] = smp.h_V_phase.get(
                        ph, np.zeros(N_HARMONICS, dtype=np.float32))

        self._append("/timestamp", ts)
        for ch in self.channels:
            self._append(f"/measurements/{ch}", scalars[ch])
        for (kind, ph), arr in h_arrays.items():
            self._append(f"/measurements/harmonics/{kind}_{ph}", arr)
        self._n_written += n
        self._buffer.clear()
        self._f.flush()

    def _estimated_ts(self, i: int) -> int:
        dt_us = int(round(1e6 / self.sample_rate_hz))
        for j in range(i - 1, -1, -1):
            if self._buffer[j] is not None:
                return self._buffer[j].timestamp_us + dt_us * (i - j)
        for j in range(i + 1, len(self._buffer)):
            if self._buffer[j] is not None:
                return self._buffer[j].timestamp_us - dt_us * (j - i)
        return int(time.time() * 1e6)

    def _append(self, path: str, arr: np.ndarray) -> None:
        ds = self._f[path]
        old = ds.shape[0]
        ds.resize((old + arr.shape[0],) + arr.shape[1:])
        ds[old:old + arr.shape[0]] = arr

    def close(self, summary: dict) -> None:
        self.flush()
        if self._f is None:
            return
        self._f["metadata"].attrs["recording_summary"] = json.dumps(summary)
        self._f.close()
        self._f = None

    @property
    def n_samples(self) -> int:
        return self._n_written + len(self._buffer)

    @property
    def n_gaps(self) -> int:
        return self._gap_count


# =============================================================================
# Acquisition service - persistent connection, health, reconnect, live buffer
# =============================================================================

class AcquisitionService:
    """Owns the meter connection. Runs a background thread that:
        - keeps the connection alive (reconnect with backoff on failure),
        - polls at the requested rate into a rolling buffer for the dashboard,
        - feeds the active recording session (if any).
    Connecting/disconnecting and starting/stopping sessions never tear down the
    thread; they just change state.
    """

    def __init__(self, reader: BaseReader, sample_rate_hz: float,
                 output_dir: str, write_harmonics: bool = False):
        self.reader = reader
        self.sample_rate_hz = sample_rate_hz
        self.output_dir = output_dir
        self.write_harmonics = write_harmonics

        self._thread: Optional[threading.Thread] = None
        self._run_flag = threading.Event()
        self._lock = threading.RLock()

        # connection state
        self.state = "disconnected"      # disconnected|connecting|connected|reconnecting|error
        self.last_error = ""
        self.connected_since: Optional[float] = None
        self.last_sample_time: Optional[float] = None
        self.consecutive_failures = 0
        self.total_samples = 0
        self.total_failures = 0
        self._recent_intervals: deque = deque(maxlen=50)

        # live ring buffer for charts: list of (t_ms, {channel: value})
        self._buffer: deque = deque(maxlen=LIVE_BUFFER_SAMPLES)
        self._latest: Dict[str, float] = {}

        # recording session
        self._writer: Optional[IncrementalHDF5Writer] = None
        self.session: Optional[dict] = None       # active session info
        self.session_history: List[dict] = []      # completed sessions this run

    # ---- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._run_flag.set()
        self.state = "connecting"
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="acquisition")
        self._thread.start()

    def shutdown(self) -> None:
        self._run_flag.clear()
        if self.session is not None:
            self.stop_session()
        if self._thread:
            self._thread.join(timeout=5.0)
        try:
            self.reader.disconnect()
        except Exception:
            pass
        self.state = "disconnected"

    # ---- connection control (callable from the web thread) ------------------
    def request_connect(self) -> None:
        with self._lock:
            if self.state in ("connected", "connecting", "reconnecting"):
                return
            self.state = "connecting"
        if not (self._thread and self._thread.is_alive()):
            self.start()

    def request_disconnect(self) -> None:
        with self._lock:
            if self.session is not None:
                self.stop_session()
            try:
                self.reader.disconnect()
            except Exception:
                pass
            self.state = "disconnected"
            self.connected_since = None

    # ---- session control ----------------------------------------------------
    def start_session(self, label: str) -> dict:
        with self._lock:
            if self.session is not None:
                raise RuntimeError("a recording is already in progress")
            if self.state != "connected":
                raise RuntimeError("connect to the meter before recording")
            safe = _safe_label(label)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{safe}_{ts}.h5"
            path = os.path.join(self.output_dir, fname)
            anchor = datetime.now(timezone.utc).replace(microsecond=0)
            self._writer = IncrementalHDF5Writer(
                path, list(CORE_REGISTERS.keys()) +
                list(getattr(self.reader, "extra_channels", {}) or {}),
                self.sample_rate_hz, anchor, label,
                write_harmonics=self.write_harmonics)
            self.session = {
                "label": label, "file": path, "filename": fname,
                "started": time.time(), "samples": 0, "gaps": 0,
            }
            return dict(self.session)

    def stop_session(self) -> Optional[dict]:
        with self._lock:
            if self.session is None or self._writer is None:
                return None
            dur = time.time() - self.session["started"]
            summary = {
                "appliance_label": self.session["label"],
                "duration_s": round(dur, 2),
                "n_samples": self._writer.n_samples,
                "n_gaps": self._writer.n_gaps,
                "configured_sample_rate_hz": self.sample_rate_hz,
                "harmonics_enabled": self.write_harmonics,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._writer.close(summary)
            done = {**self.session, "duration_s": round(dur, 2),
                    "samples": self._writer.n_samples,
                    "gaps": self._writer.n_gaps, "stopped": time.time()}
            self.session_history.append(done)
            self.session = None
            self._writer = None
            return done

    # ---- main loop ----------------------------------------------------------
    def _run(self) -> None:
        backoff = 1.0
        dt = 1.0 / self.sample_rate_hz
        next_tick = time.time()
        while self._run_flag.is_set():
            # ensure connection
            if self.state in ("connecting", "reconnecting", "disconnected"):
                if self.state == "disconnected":
                    time.sleep(0.1)
                    continue
                try:
                    self.reader.disconnect()
                except Exception:
                    pass
                try:
                    self.reader.connect()
                    with self._lock:
                        self.state = "connected"
                        self.connected_since = time.time()
                        self.consecutive_failures = 0
                        self.last_error = ""
                    backoff = 1.0
                    next_tick = time.time()
                except Exception as e:  # noqa: BLE001
                    with self._lock:
                        self.last_error = str(e)
                        self.state = "reconnecting"
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 15.0)
                    continue

            if self.state != "connected":
                time.sleep(0.1)
                continue

            # poll one sample
            next_tick += dt
            sample = self.reader.read_sample()
            now = time.time()

            if sample is None:
                with self._lock:
                    self.consecutive_failures += 1
                    self.total_failures += 1
                    # feed gap to active recording so timing stays aligned
                    if self._writer is not None:
                        self._writer.add(None)
                        self.session["gaps"] = self._writer.n_gaps
                        self.session["samples"] = self._writer.n_samples
                    if self.consecutive_failures >= 10:
                        self.state = "reconnecting"
                        self.last_error = "lost connection (repeated read failures)"
            else:
                with self._lock:
                    if self.last_sample_time is not None:
                        self._recent_intervals.append(now - self.last_sample_time)
                    self.last_sample_time = now
                    self.consecutive_failures = 0
                    self.total_samples += 1
                    self._latest = sample.scalars
                    self._buffer.append((int(now * 1000), sample.scalars))
                    if self._writer is not None:
                        self._writer.add(sample)
                        self.session["samples"] = self._writer.n_samples
                        self.session["gaps"] = self._writer.n_gaps

            sleep_s = next_tick - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.time()   # fell behind; resync

    # ---- snapshots for the API ----------------------------------------------
    def effective_rate(self) -> float:
        with self._lock:
            if not self._recent_intervals:
                return 0.0
            avg = sum(self._recent_intervals) / len(self._recent_intervals)
            return 1.0 / avg if avg > 0 else 0.0

    def status(self) -> dict:
        with self._lock:
            uptime = (time.time() - self.connected_since
                      if self.connected_since else 0.0)
            return {
                "state": self.state,
                "simulated": self.reader.is_simulated,
                "host": getattr(self.reader, "host", "simulated"),
                "port": getattr(self.reader, "port", None),
                "configured_rate_hz": self.sample_rate_hz,
                "effective_rate_hz": round(self.effective_rate(), 3),
                "harmonics": self.write_harmonics,
                "uptime_s": round(uptime, 1),
                "total_samples": self.total_samples,
                "total_failures": self.total_failures,
                "consecutive_failures": self.consecutive_failures,
                "last_error": self.last_error,
                "session": dict(self.session) if self.session else None,
                "session_history": list(self.session_history),
            }

    def live(self, max_points: int = 600) -> dict:
        with self._lock:
            buf = list(self._buffer)
            latest = dict(self._latest)
        if len(buf) > max_points:
            step = max(1, len(buf) // max_points)
            buf = buf[::step]
        t = [p[0] for p in buf]
        series = {ch: [float(p[1].get(ch, float("nan"))) for p in buf]
                  for ch in CHART_CHANNELS}
        return {"t": t, "series": series, "latest": latest}


# =============================================================================
# Helpers
# =============================================================================

def _safe_label(label: str) -> str:
    label = (label or "appliance").strip().lower()
    label = re.sub(r"[^a-z0-9_-]+", "_", label).strip("_")
    return label or "appliance"


# =============================================================================
# Flask app
# =============================================================================

def _json_sanitize(obj):
    """Replace non-finite floats (NaN / +-Inf) with None, recursively.

    Real meter reads contain NaN on unpopulated channels (e.g. L2/L3 for a
    single-phase load, or PF/unbalance on an unloaded phase). Python's json
    (and Flask's jsonify) emit the bare token `NaN`, which is INVALID JSON, so
    the browser's response.json() throws and the whole live update silently
    fails - blank charts and readouts. NaN -> null keeps the payload valid;
    the frontend already renders null as "-" and skips it when drawing.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj


class SafeJSONProvider(DefaultJSONProvider):
    """JSON provider that guarantees spec-valid output (no NaN/Infinity)."""

    def dumps(self, obj, **kwargs):
        return super().dumps(_json_sanitize(obj), **kwargs)


def create_app(svc: AcquisitionService) -> Flask:
    app = Flask(__name__)
    app.json = SafeJSONProvider(app)  # NaN -> null on every endpoint

    @app.route("/")
    def index():
        return Response(DASHBOARD_HTML, mimetype="text/html")

    @app.route("/api/status")
    def api_status():
        return jsonify(svc.status())

    @app.route("/api/live")
    def api_live():
        return jsonify(svc.live())

    @app.route("/api/connect", methods=["POST"])
    def api_connect():
        svc.request_connect()
        return jsonify({"ok": True, "state": svc.state})

    @app.route("/api/disconnect", methods=["POST"])
    def api_disconnect():
        svc.request_disconnect()
        return jsonify({"ok": True, "state": svc.state})

    @app.route("/api/session/start", methods=["POST"])
    def api_session_start():
        data = request.get_json(silent=True) or {}
        label = data.get("label", "appliance")
        try:
            info = svc.start_session(label)
            return jsonify({"ok": True, "session": info})
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/session/stop", methods=["POST"])
    def api_session_stop():
        done = svc.stop_session()
        return jsonify({"ok": True, "session": done})

    @app.route("/api/inspect", methods=["POST"])
    def api_inspect():
        """Live register inspector: read raw registers and decode as floats.
        Used to confirm addresses / word order against the meter display."""
        data = request.get_json(silent=True) or {}
        try:
            address = int(data.get("address", 1))
            count = int(data.get("count", 10))
            swap = data.get("swap", SwapMode.BIG_WORD_FIRST)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "bad parameters"}), 400
        count = max(2, min(count, 100))
        # temporarily honour requested swap on the reader
        old_swap = getattr(svc.reader, "swap", SwapMode.BIG_WORD_FIRST)
        if hasattr(svc.reader, "swap"):
            svc.reader.swap = swap
        regs = svc.reader.read_raw(address, count)
        if hasattr(svc.reader, "swap"):
            svc.reader.swap = old_swap
        if regs is None:
            return jsonify({"ok": False, "error": "read failed / not connected"}), 400
        floats = []
        for i in range(0, len(regs) - 1, 2):
            floats.append({
                "address": address + i,
                "raw": [regs[i], regs[i + 1]],
                "float_big_word": round(decode_float(regs[i], regs[i + 1],
                                        SwapMode.BIG_WORD_FIRST), 4),
                "float_little_word": round(decode_float(regs[i], regs[i + 1],
                                           SwapMode.LITTLE_WORD_FIRST), 4),
            })
        return jsonify({"ok": True, "registers": regs, "floats": floats})

    return app


# =============================================================================
# Embedded dashboard (no external dependencies - works offline / air-gapped lab)
# =============================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PAC4200 Monitor</title>
<style>
  :root{
    --bg:#0e1116; --panel:#171c24; --panel2:#1e2530; --line:#2a3340;
    --txt:#e6edf3; --muted:#8b98a9; --accent:#4aa8ff; --good:#3ecf8e;
    --warn:#f0b429; --bad:#ff5c5c; --p:#4aa8ff; --q:#c084fc; --i:#3ecf8e;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       background:var(--bg);color:var(--txt);font-size:14px}
  header{display:flex;align-items:center;gap:14px;padding:12px 18px;background:var(--panel);
         border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10;flex-wrap:wrap}
  header h1{font-size:16px;margin:0;font-weight:600;letter-spacing:.3px}
  .dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle}
  .pill{padding:4px 10px;border-radius:20px;background:var(--panel2);border:1px solid var(--line);
        font-size:12px;color:var(--muted);white-space:nowrap}
  .pill b{color:var(--txt);font-weight:600}
  .grow{flex:1}
  button{background:var(--panel2);color:var(--txt);border:1px solid var(--line);border-radius:8px;
         padding:8px 14px;font-size:13px;cursor:pointer;transition:.15s}
  button:hover{border-color:var(--accent)}
  button.primary{background:var(--accent);border-color:var(--accent);color:#04121f;font-weight:600}
  button.danger{background:#3a1d22;border-color:#5e2a30;color:#ff9a9a}
  button.danger:hover{border-color:var(--bad)}
  button:disabled{opacity:.4;cursor:not-allowed}
  main{display:grid;grid-template-columns:340px 1fr;gap:16px;padding:16px;align-items:start}
  @media(max-width:900px){main{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
  .panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);
            margin:0 0 12px}
  input[type=text],input[type=number]{background:var(--panel2);border:1px solid var(--line);
        border-radius:8px;color:var(--txt);padding:8px 10px;font-size:13px;width:100%}
  label.fld{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px}
  .row{display:flex;gap:8px}
  .readout{display:grid;grid-template-columns:1fr 1fr;gap:8px 14px;margin-top:4px}
  .kv{display:flex;flex-direction:column;background:var(--panel2);border:1px solid var(--line);
      border-radius:8px;padding:8px 10px}
  .kv .k{font-size:11px;color:var(--muted)}
  .kv .v{font-size:17px;font-weight:600;font-variant-numeric:tabular-nums}
  .kv .u{font-size:11px;color:var(--muted);font-weight:400}
  .chart-card{margin-bottom:16px}
  .chart-card h3{font-size:13px;margin:0 0 4px;font-weight:600}
  .chart-card .sub{font-size:11px;color:var(--muted);margin:0 0 8px}
  canvas{width:100%;display:block;border-radius:8px;height:160px}
  #chartP{height:200px}
  .legend{display:flex;gap:14px;font-size:11px;color:var(--muted);margin-top:6px;flex-wrap:wrap}
  .legend span{display:inline-flex;align-items:center;gap:5px}
  .legend i{width:14px;height:3px;border-radius:2px;display:inline-block}
  .sess-list{margin-top:12px;font-size:12px}
  .sess-list .item{display:flex;justify-content:space-between;padding:7px 0;border-top:1px solid var(--line)}
  .sess-list .item .lbl{color:var(--txt);font-weight:600}
  .sess-list .item .meta{color:var(--muted)}
  .badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:11px;font-weight:600}
  .rec{background:#3a1d22;color:#ff9a9a}
  .muted{color:var(--muted)}
  details{margin-top:8px}
  summary{cursor:pointer;color:var(--muted);font-size:12px;padding:6px 0}
  table.insp{width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;font-variant-numeric:tabular-nums}
  table.insp th,table.insp td{text-align:right;padding:5px 8px;border-bottom:1px solid var(--line)}
  table.insp th{color:var(--muted);font-weight:500;text-align:right}
  .hint{font-size:11px;color:var(--muted);line-height:1.5;margin-top:8px}
  .err{color:var(--bad);font-size:12px;margin-top:8px;min-height:16px}
</style>
</head>
<body>
<header>
  <h1>PAC4200 Monitor</h1>
  <span class="pill"><span id="dot" class="dot"></span><b id="state">…</b></span>
  <span class="pill" id="hostpill">host: …</span>
  <span class="pill">rate: <b id="rate">-</b> Hz</span>
  <span class="pill">uptime: <b id="uptime">-</b></span>
  <span class="pill">samples: <b id="samples">0</b> · drops: <b id="drops">0</b></span>
  <span class="grow"></span>
  <button id="btnConnect" class="primary">Connect</button>
  <button id="btnDisconnect">Disconnect</button>
</header>

<main>
  <!-- LEFT: control -->
  <div>
    <div class="panel">
      <h2>Recording session</h2>
      <div id="recBanner" style="display:none" class="badge rec">● RECORDING</div>
      <label class="fld">Appliance label</label>
      <input id="label" type="text" placeholder="e.g. fridge, led_lamp, kettle" value="appliance">
      <div class="row" style="margin-top:10px">
        <button id="btnStart" class="primary grow">Start recording</button>
        <button id="btnStop" class="danger grow" disabled>Stop</button>
      </div>
      <div id="sessNow" class="hint"></div>
      <div class="sess-list" id="sessList"></div>
      <div class="hint">Each appliance is saved to its own <code>.h5</code> file in the
        output folder. Plug in an appliance, start, stop, then plug in the next -
        the meter connection stays up the whole time.</div>
    </div>

    <div class="panel" style="margin-top:16px">
      <h2>Register inspector</h2>
      <div class="hint">Confirm addresses / word order against the meter's own
        display before trusting extended channels (THD&nbsp;I, cosφ, harmonics).</div>
      <div class="row" style="margin-top:10px">
        <div style="flex:1"><label class="fld">Start addr</label>
          <input id="inspAddr" type="number" value="1"></div>
        <div style="flex:1"><label class="fld">Count</label>
          <input id="inspCount" type="number" value="20"></div>
      </div>
      <button id="btnInspect" style="margin-top:10px;width:100%">Read registers</button>
      <div id="inspErr" class="err"></div>
      <div id="inspOut"></div>
    </div>
  </div>

  <!-- RIGHT: live data -->
  <div>
    <div class="panel" style="margin-bottom:16px">
      <h2>Live readings</h2>
      <div class="readout" id="readout"></div>
    </div>

    <div class="panel chart-card">
      <h3>Active power P</h3>
      <p class="sub">Total and per-phase (W) - the primary disaggregation signal</p>
      <canvas id="chartP" height="200"></canvas>
      <div class="legend">
        <span><i style="background:var(--p)"></i>P total</span>
        <span><i style="background:#7cc4ff"></i>P L1</span>
        <span><i style="background:#9ad0ff"></i>P L2</span>
        <span><i style="background:#b9ddff"></i>P L3</span>
      </div>
    </div>

    <div class="panel chart-card">
      <h3>Reactive power Q<sub>total</sub></h3>
      <p class="sub">var - sign/shape distinguishes capacitive vs inductive loads</p>
      <canvas id="chartQ" height="160"></canvas>
      <div class="legend"><span><i style="background:var(--q)"></i>Q total</span></div>
    </div>

    <div class="panel chart-card">
      <h3>Current per phase</h3>
      <p class="sub">A</p>
      <canvas id="chartI" height="160"></canvas>
      <div class="legend">
        <span><i style="background:var(--i)"></i>I L1</span>
        <span><i style="background:#6fe0ad"></i>I L2</span>
        <span><i style="background:#9aeccb"></i>I L3</span>
      </div>
    </div>
  </div>
</main>

<script>
const $ = id => document.getElementById(id);
async function post(url, body){
  const r = await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body||{})});
  return r.json();
}
async function getj(url){ const r = await fetch(url); return r.json(); }

// ---- readouts ----
const READOUTS = [
  ["P_total","P total","W"],["Q_total","Q total","var"],
  ["S_total","S total","VA"],["PF_total","PF",""],
  ["V_avg_LN","V L-N avg","V"],["I_avg","I avg","A"],
  ["freq","Frequency","Hz"],["THD_V_L12","THD V L12","%"],
  ["P_L1","P L1","W"],["P_L2","P L2","W"],["P_L3","P L3","W"],
  ["I_L1","I L1","A"],["I_L2","I L2","A"],["I_L3","I L3","A"],
];
function buildReadout(){
  $("readout").innerHTML = READOUTS.map(([k,lab,u])=>
    `<div class="kv"><span class="k">${lab}</span>
       <span class="v" id="ro_${k}">-<span class="u"> ${u}</span></span></div>`).join("");
}
function fmt(v){
  if(v===undefined||v===null||Number.isNaN(v)) return "-";
  const a=Math.abs(v);
  if(a>=1000) return v.toFixed(0);
  if(a>=100) return v.toFixed(1);
  if(a>=1) return v.toFixed(2);
  return v.toFixed(3);
}

// ---- simple offline canvas line chart ----
function drawChart(canvas, t, seriesList){
  const dpr = window.devicePixelRatio||1;
  // Use the CSS layout size (clientWidth/clientHeight) as the source of truth.
  // The drawing buffer is CSS size * dpr. Reading canvas.height (the buffer)
  // and rewriting it each frame compounds by dpr on HiDPI screens and makes
  // the canvas grow without bound - so never do that.
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const bw = Math.round(W*dpr), bh = Math.round(H*dpr);
  if(canvas.width!==bw || canvas.height!==bh){ canvas.width=bw; canvas.height=bh; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);  // absolute, not cumulative; no compounding
  ctx.clearRect(0,0,W,H);
  const padL=46, padR=10, padT=8, padB=18;
  const plotW=W-padL-padR, plotH=H-padT-padB;

  // gather finite values
  let lo=Infinity, hi=-Infinity, any=false;
  seriesList.forEach(s=>s.data.forEach(v=>{
    if(Number.isFinite(v)){ any=true; if(v<lo)lo=v; if(v>hi)hi=v; }}));
  if(!any){ lo=0; hi=1; }
  if(lo===hi){ lo-=1; hi+=1; }
  const pad=(hi-lo)*0.1; lo-=pad; hi+=pad;
  const n = t.length;
  const x = i => padL + (n<=1?0:(i/(n-1))*plotW);
  const y = v => padT + plotH - ((v-lo)/(hi-lo))*plotH;

  // grid + y labels
  ctx.strokeStyle="#2a3340"; ctx.fillStyle="#8b98a9"; ctx.lineWidth=1;
  ctx.font="10px sans-serif"; ctx.textAlign="right"; ctx.textBaseline="middle";
  for(let g=0; g<=4; g++){
    const val = lo + (hi-lo)*g/4;
    const yy = y(val);
    ctx.beginPath(); ctx.moveTo(padL,yy); ctx.lineTo(W-padR,yy); ctx.stroke();
    ctx.fillText(fmt(val), padL-6, yy);
  }
  // time span label
  if(n>1){
    const span=(t[n-1]-t[0])/1000;
    ctx.textAlign="left"; ctx.fillText("← "+span.toFixed(0)+" s",padL+2,H-8);
  }
  // lines
  seriesList.forEach(s=>{
    ctx.strokeStyle=s.color; ctx.lineWidth=s.width||1.6; ctx.beginPath();
    let started=false;
    s.data.forEach((v,i)=>{
      if(!Number.isFinite(v)){ started=false; return; }
      const px=x(i), py=y(v);
      if(!started){ ctx.moveTo(px,py); started=true; } else ctx.lineTo(px,py);
    });
    ctx.stroke();
  });
}

// ---- polling ----
let connState="disconnected";
async function pollStatus(){
  try{
    const s = await getj("/api/status");
    connState = s.state;
    const colors={connected:"var(--good)",connecting:"var(--warn)",
      reconnecting:"var(--warn)",error:"var(--bad)",disconnected:"var(--muted)"};
    $("dot").style.background = colors[s.state]||"var(--muted)";
    $("state").textContent = s.state + (s.simulated?" (sim)":"");
    $("hostpill").textContent = "host: "+(s.host||"-");
    $("rate").textContent = s.effective_rate_hz ? s.effective_rate_hz.toFixed(2) : "-";
    $("uptime").textContent = s.uptime_s ? Math.floor(s.uptime_s)+"s" : "-";
    $("samples").textContent = s.total_samples;
    $("drops").textContent = s.total_failures;

    const recording = !!s.session;
    $("recBanner").style.display = recording ? "inline-block":"none";
    $("btnStop").disabled = !recording;
    $("btnStart").disabled = recording || s.state!=="connected";
    $("label").disabled = recording;
    if(recording){
      const d=(Date.now()/1000 - s.session.started);
      $("sessNow").innerHTML = `Recording <b>${s.session.label}</b> → `+
        `<code>${s.session.filename}</code><br>`+
        `${Math.floor(d)} s · ${s.session.samples} samples · ${s.session.gaps} gaps`;
    } else { $("sessNow").innerHTML=""; }

    if(s.session_history && s.session_history.length){
      $("sessList").innerHTML = s.session_history.slice().reverse().map(h=>
        `<div class="item"><span class="lbl">${h.label}</span>`+
        `<span class="meta">${h.samples} smp · ${h.duration_s}s · ${h.gaps} gaps</span></div>`
      ).join("");
    }
  }catch(e){ $("state").textContent="server unreachable"; }
}

async function pollLive(){
  if(connState!=="connected") return;
  try{
    const d = await getj("/api/live");
    READOUTS.forEach(([k,,u])=>{
      const el=$("ro_"+k); if(el) el.innerHTML = fmt(d.latest[k])+`<span class="u"> ${u}</span>`;
    });
    drawChart($("chartP"), d.t, [
      {data:d.series.P_total,color:"#4aa8ff",width:2},
      {data:d.series.P_L1,color:"#7cc4ff"},
      {data:d.series.P_L2,color:"#9ad0ff"},
      {data:d.series.P_L3,color:"#b9ddff"},
    ]);
    drawChart($("chartQ"), d.t, [{data:d.series.Q_total,color:"#c084fc",width:2}]);
    drawChart($("chartI"), d.t, [
      {data:d.series.I_L1,color:"#3ecf8e"},
      {data:d.series.I_L2,color:"#6fe0ad"},
      {data:d.series.I_L3,color:"#9aeccb"},
    ]);
  }catch(e){ console.error("live update failed:", e); }
}

// ---- buttons ----
$("btnConnect").onclick = ()=>post("/api/connect");
$("btnDisconnect").onclick = ()=>post("/api/disconnect");
$("btnStart").onclick = async ()=>{
  const r = await post("/api/session/start",{label:$("label").value});
  if(!r.ok) alert(r.error||"could not start");
};
$("btnStop").onclick = ()=>post("/api/session/stop");
$("btnInspect").onclick = async ()=>{
  $("inspErr").textContent=""; $("inspOut").innerHTML="reading…";
  const r = await post("/api/inspect",{address:+$("inspAddr").value,count:+$("inspCount").value});
  if(!r.ok){ $("inspOut").innerHTML=""; $("inspErr").textContent=r.error; return; }
  $("inspOut").innerHTML =
    `<table class="insp"><tr><th>addr</th><th>raw hi/lo</th>`+
    `<th>float (big word)</th><th>float (little word)</th></tr>`+
    r.floats.map(f=>`<tr><td>${f.address}</td>`+
      `<td>${f.raw[0]}/${f.raw[1]}</td>`+
      `<td>${f.float_big_word}</td><td>${f.float_little_word}</td></tr>`).join("")+
    `</table>`;
};

buildReadout();
setInterval(pollStatus, 800);
setInterval(pollLive, 800);
pollStatus();
</script>
</body>
</html>
"""


# =============================================================================
# Headless recording (no UI) - for long unattended sessions
# =============================================================================

def run_headless(svc: AcquisitionService, label: str, duration: float) -> None:
    print(f"Connecting…")
    svc.request_connect()
    # wait for connection
    t0 = time.time()
    while svc.state != "connected" and time.time() - t0 < 20:
        if svc.state in ("error",):
            break
        time.sleep(0.2)
    if svc.state != "connected":
        print(f"ERROR: could not connect ({svc.last_error})", file=sys.stderr)
        svc.shutdown()
        sys.exit(1)
    info = svc.start_session(label)
    print(f"Recording '{label}' for {duration}s -> {info['file']}")
    print("Press Ctrl-C to stop early.")
    try:
        end = time.time() + duration
        while time.time() < end:
            time.sleep(2.0)
            st = svc.status()
            sess = st["session"] or {}
            print(f"  samples={sess.get('samples',0)} gaps={sess.get('gaps',0)} "
                  f"rate={st['effective_rate_hz']}Hz state={st['state']}", flush=True)
    except KeyboardInterrupt:
        print("\nstopping…")
    done = svc.stop_session()
    svc.shutdown()
    print("\nRecording summary:")
    for k, v in (done or {}).items():
        print(f"  {k}: {v}")
    print(f"\nNext step:  python preprocessor.py --input {done['file']}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="PAC4200 live monitor + per-appliance HDF5 recorder.")
    p.add_argument("--host", default=None, help="PAC4200 IP (omit for --simulate)")
    p.add_argument("--port", type=int, default=502)
    p.add_argument("--unit-id", type=int, default=1, help="Modbus unit/device id")
    p.add_argument("--rate", type=float, default=DEFAULT_SAMPLE_RATE_HZ,
                   help=f"poll rate Hz (default {DEFAULT_SAMPLE_RATE_HZ})")
    p.add_argument("--swap", default=SwapMode.BIG_WORD_FIRST,
                   choices=[SwapMode.BIG_WORD_FIRST, SwapMode.LITTLE_WORD_FIRST])
    p.add_argument("--address-offset", type=int, default=0,
                   help="add to every register address (use -1 if values look "
                        "shifted; classic 0-based vs 1-based fix)")
    p.add_argument("--output-dir", default="recordings",
                   help="folder for per-appliance .h5 files")
    p.add_argument("--harmonics", action="store_true",
                   help="also record per-order harmonic magnitudes via FC 0x14; "
                        "requires verified file numbers in HARMONIC_I_FILE/"
                        "HARMONIC_V_FILE (use probe_filerecord.py). PAC4200 "
                        "provides magnitudes only - harmonic phase arrays stay zero.")
    p.add_argument("--simulate", action="store_true",
                   help="use a synthetic meter (no hardware needed)")
    p.add_argument("--web-port", type=int, default=8200, help="dashboard port")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--headless", action="store_true",
                   help="record one appliance with no UI (needs --label/--duration)")
    p.add_argument("--label", default="appliance", help="headless: appliance label")
    p.add_argument("--duration", type=float, default=60.0,
                   help="headless: seconds to record")
    p.add_argument("--print-register-map", action="store_true")
    args = p.parse_args()

    if args.print_register_map:
        print("PAC4200 CORE register map (VERIFIED, Siemens offsets, FC 0x03,")
        print("32-bit float = 2 regs, big word first):\n")
        for ch, off in CORE_REGISTERS.items():
            print(f"  {ch:13s}  reg {off:>3d}-{off+1:<3d}")
        print(f"\n  Block A read: regs {CORE_BLOCK_A_START}.."
              f"{CORE_BLOCK_A_START+CORE_BLOCK_A_COUNT-1} "
              f"(offsets 1..47)")
        print(f"  Block B read: regs {CORE_BLOCK_B_START}.."
              f"{CORE_BLOCK_B_START+CORE_BLOCK_B_COUNT-1} "
              f"(offsets 55..73)")
        print("\n  Regs 49,51,53 are reserved on the PAC4200 and are NOT read.")
        print("\nEXTENDED scalar channels (THD current, cosphi) are NOT in this")
        print("verified block. Confirm their addresses with the Register")
        print("Inspector in the UI before enabling them.")
        print("\nHARMONIC spectrum is read via FC 0x14 Read File Record, one")
        print(f"file per quantity ({N_HARMONICS} orders each, magnitudes only).")
        print("Verify these file numbers with probe_filerecord.py + the display:")
        for ph in ("L1", "L2", "L3"):
            print(f"  I {ph}: file {HARMONIC_I_FILE.get(ph, 0):>4d}     "
                  f"U {ph} (L-N): file {HARMONIC_V_FILE.get(ph, 0):>4d}")
        return

    if not args.simulate and args.host is None:
        p.error("--host is required unless --simulate")
    if not args.simulate and not HAVE_PYMODBUS:
        p.error("pymodbus not installed: pip install pymodbus (or use --simulate)")

    if args.simulate:
        reader: BaseReader = SimulatedReader(extra_channels=EXTENDED_CHANNELS,
                                             read_harmonics=args.harmonics)
        print("Running with a SIMULATED meter (no hardware).")
    else:
        reader = ModbusReader(host=args.host, port=args.port, unit_id=args.unit_id,
                              swap=args.swap, address_offset=args.address_offset,
                              extra_channels=EXTENDED_CHANNELS,
                              read_harmonics=args.harmonics)
        print(f"Target meter: {args.host}:{args.port}  unit={args.unit_id}  "
              f"(pymodbus unit kwarg = {_UNIT_KWARG or 'positional'})")

    if (args.harmonics and not args.simulate
            and not any(HARMONIC_I_FILE.values())
            and not any(HARMONIC_V_FILE.values())):
        print("WARNING: --harmonics is on but HARMONIC_I_FILE/HARMONIC_V_FILE are\n"
              "         all zero. The harmonics group will be created but stay\n"
              "         zero until you fill in the verified file numbers - use\n"
              "         probe_filerecord.py to find them.",
              file=sys.stderr)
    elif args.harmonics and not args.simulate:
        print("NOTE: harmonics use FC 0x14 files "
              f"I={list(HARMONIC_I_FILE.values())} "
              f"U={list(HARMONIC_V_FILE.values())}. Confirm the current files "
              "with a load on\n      and the phase order against the meter "
              "display before trusting them.", file=sys.stderr)

    svc = AcquisitionService(reader, args.rate, args.output_dir,
                             write_harmonics=args.harmonics)

    if args.headless:
        run_headless(svc, args.label, args.duration)
        return

    svc.start()
    svc.request_connect()
    app = create_app(svc)
    url = f"http://127.0.0.1:{args.web_port}/"
    print(f"\nDashboard:  {url}")
    print("Connect, label an appliance, Start/Stop recording. Ctrl-C to quit.\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=args.web_port, threaded=True,
                debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nshutting down…")
        svc.shutdown()


if __name__ == "__main__":
    main()