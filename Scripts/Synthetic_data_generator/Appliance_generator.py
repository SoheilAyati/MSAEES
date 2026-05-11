#!/usr/bin/env python3
"""
NILM Project — Appliance Generators
===================================
Single entry point for generating synthetic per-appliance traces for any
of the 8 project appliances (plus an always-on baseload). Output matches
the data format spec (5 Hz channels, harmonics 2nd-40th, ground-truth
state labels) and writes HDF5 files compatible with the storage layout.

Quick start
-----------
    # list all appliances
    python Appliance_generator.py --list

    # generate 24h of fridge data, save to fridge.h5, also show a plot
    python Appliance_generator.py --appliance fridge --output fridge.h5 --plot

    # PV on a summer solstice
    python Appliance_generator.py --appliance pv --anchor-date 2024-06-21 \
        --output pv_summer.h5 --plot

    # EV in fast-AC mode
    python Appliance_generator.py --appliance ev \
        --params '{"mode": "fast_AC_11kW"}' --output ev_fast.h5 --plot

    # override parameters
    python Appliance_generator.py --appliance fridge \
        --params '{"compressor_P_W": [200, 250]}' --seed 7

See `appliance_generators_v0.1.md` for the design behind every appliance.
"""

import argparse
import json
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_HARMONICS = 39          # harmonics 2nd through 40th
V_NOMINAL = 230.0         # nominal phase voltage (L-N) in V
FORMAT_VERSION = "0.1"
GENERATOR_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class ApplianceTrace:
    """Per-appliance trace. P, Q are the appliance's contribution to the
    aggregate active and reactive power (totals across phases for 3-phase
    appliances; the aggregator distributes to phases based on metadata)."""
    timestamp_us: np.ndarray             # int64, microseconds since Unix epoch
    P: np.ndarray                        # float32, watts (consumption > 0)
    Q: np.ndarray                        # float32, var (inductive > 0)
    state: np.ndarray                    # object/string, per-sample categorical
    harmonics_I_mag: np.ndarray          # (n_samples, N_HARMONICS), in A
    harmonics_I_phase: np.ndarray        # (n_samples, N_HARMONICS), in rad
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers — time-of-day intensity profiles
# ---------------------------------------------------------------------------

def time_of_day_intensity(hour: np.ndarray, profile_name: str) -> np.ndarray:
    """Returns an intensity curve in [0,1] given hour-of-day (0..24)."""
    h = hour % 24.0
    if profile_name == "working_hours_weekday":
        return np.exp(-((h - 13.0) ** 2) / (2 * 3.5 ** 2))
    if profile_name == "morning_evening_peaks":
        morning = np.exp(-((h - 7.5) ** 2) / 1.5)
        evening = np.exp(-((h - 19.5) ** 2) / 3.0)
        return np.maximum(morning, evening)
    if profile_name == "overnight":
        # Peaks around 02:00, low during day
        return np.exp(-((((h + 12) % 24) - 14) ** 2) / 8.0)
    if profile_name == "continuous":
        return np.ones_like(h)
    if profile_name == "random_burst":
        return np.full_like(h, 0.05)
    raise ValueError(f"Unknown profile: {profile_name}")


def solar_intensity(hour: np.ndarray, anchor_datetime: datetime,
                    latitude_deg: float = 51.0) -> np.ndarray:
    """Solar elevation as a fraction in [0,1]. Zero before sunrise / after sunset."""
    doy = anchor_datetime.timetuple().tm_yday
    decl = math.radians(23.44 * math.sin(2 * math.pi * (284 + doy) / 365.0))
    lat = math.radians(latitude_deg)
    # local solar time ~ UTC hour shifted; simplify with local noon at 13:00
    H = np.radians(15.0 * (hour - 13.0))
    sin_elev = (math.sin(lat) * math.sin(decl)
                + math.cos(lat) * math.cos(decl) * np.cos(H))
    return np.maximum(sin_elev, 0.0)


# ---------------------------------------------------------------------------
# Helpers — harmonic synthesis
# ---------------------------------------------------------------------------

def synthesize_harmonics(I_fund_A: np.ndarray,
                        profile: Dict[int, float],
                        rng: np.random.Generator,
                        phase_jitter_rad: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Given a fundamental current magnitude array and a profile
    {order: magnitude_as_fraction_of_fundamental}, produce
    (n_samples, N_HARMONICS) arrays of magnitude (A) and phase (rad)."""
    n = len(I_fund_A)
    mag = np.zeros((n, N_HARMONICS), dtype=np.float32)
    phase = np.zeros((n, N_HARMONICS), dtype=np.float32)
    for order, frac in profile.items():
        idx = order - 2
        if idx < 0 or idx >= N_HARMONICS:
            continue
        mag[:, idx] = (I_fund_A * frac).astype(np.float32)
        # Each order has a characteristic phase plus small jitter (measurement uncertainty + load drift)
        base_phase = float(rng.uniform(-np.pi, np.pi))
        phase[:, idx] = (base_phase + rng.normal(0, phase_jitter_rad, n)).astype(np.float32)
    return mag, phase


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ApplianceGenerator(ABC):
    name: str = "abstract"
    appliance_type: str = "?"  # I, II, III, IV, or special
    is_three_phase: bool = False
    DEFAULT_PARAMS: dict = {}

    def __init__(self, params: Optional[dict] = None, seed: int = 0,
                 phase: str = "L1", instance_id: int = 1):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.phase = phase if not self.is_three_phase else "all"
        self.instance_id = instance_id

    def _time_axis(self, duration_s: float, sample_rate_hz: float,
                   anchor: datetime) -> Tuple[np.ndarray, int, np.ndarray]:
        """Returns (timestamps_us, n_samples, hour_of_day)."""
        n = int(duration_s * sample_rate_hz)
        dt_us = int(round(1e6 / sample_rate_hz))
        anchor_us = int(anchor.replace(tzinfo=timezone.utc).timestamp() * 1e6)
        ts_us = np.arange(n, dtype=np.int64) * dt_us + anchor_us
        sec_since = (ts_us - anchor_us) / 1e6
        hour = sec_since / 3600.0
        return ts_us, n, hour

    def _sample_range(self, key: str) -> float:
        v = self.params[key]
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return float(self.rng.uniform(v[0], v[1]))
        return float(v)

    def _make_metadata(self, instance_params: Optional[dict] = None) -> dict:
        md = {
            "name": self.name,
            "appliance_type": self.appliance_type,
            "instance_id": self.instance_id,
            "phase": self.phase,
            "is_three_phase": self.is_three_phase,
            "seed": self.seed,
            "params": dict(self.params),
        }
        if instance_params:
            md["instance_params"] = instance_params
        return md

    @abstractmethod
    def generate(self, duration_s: float, sample_rate_hz: float,
                 anchor_datetime: datetime) -> ApplianceTrace:
        ...


# ===========================================================================
# 1. Fridge
# ===========================================================================

class FridgeGenerator(ApplianceGenerator):
    name = "fridge"
    appliance_type = "II"
    DEFAULT_PARAMS = {
        "compressor_P_W":       [120.0, 180.0],
        "compressor_Q_var":     [80.0, 140.0],
        "off_P_W":              2.0,
        "on_duration_s":        [480, 1080],
        "off_duration_s":       [1200, 2700],
        "inrush_factor":        [2.5, 4.5],
        "defrost_P_W":          [200.0, 350.0],
        "defrost_duration_s":   [600, 1200],
        "defrost_per_day":      1,
    }
    HARMONICS_ON = {3: 0.06, 5: 0.04, 7: 0.02, 9: 0.01, 11: 0.005, 13: 0.003}
    HARMONICS_OFF = {}

    def generate(self, duration_s, sample_rate_hz, anchor):
        ts, N, _ = self._time_axis(duration_s, sample_rate_hz, anchor)
        comp_P = self._sample_range("compressor_P_W")
        comp_Q = self._sample_range("compressor_Q_var")
        inrush_f = self._sample_range("inrush_factor")
        off_P = float(self.params["off_P_W"])

        P = np.full(N, off_P, dtype=np.float32)
        Q = np.zeros(N, dtype=np.float32)
        state = np.full(N, "off_control", dtype=object)

        # Schedule compressor cycles starting with a random offset
        t = -self.rng.uniform(0, 1800.0)
        while t < duration_s:
            off_dur = self._sample_range("off_duration_s")
            on_dur = self._sample_range("on_duration_s")
            t_on_start = t + off_dur
            t_on_end = t_on_start + on_dur

            i_inrush = int(t_on_start * sample_rate_hz)
            if 0 <= i_inrush < N:
                P[i_inrush] = comp_P * inrush_f
                Q[i_inrush] = comp_Q * inrush_f * 0.7
                state[i_inrush] = "compressor_starting"

            i0 = max(0, i_inrush + 1)
            i1 = min(N, int(t_on_end * sample_rate_hz))
            if i0 < i1:
                P[i0:i1] = comp_P
                Q[i0:i1] = comp_Q
                state[i0:i1] = "compressor_on"

            t = t_on_end

        # Defrost events
        for _ in range(int(self.params["defrost_per_day"])):
            t_def = self.rng.uniform(0, duration_s)
            d_P = self._sample_range("defrost_P_W")
            d_dur = self._sample_range("defrost_duration_s")
            i0 = max(0, int(t_def * sample_rate_hz))
            i1 = min(N, int((t_def + d_dur) * sample_rate_hz))
            if i0 < i1:
                P[i0:i1] = d_P
                Q[i0:i1] = 5.0
                state[i0:i1] = "defrost"

        # Harmonics — applied where compressor is active
        S = np.sqrt(P ** 2 + Q ** 2)
        I_fund = (S / V_NOMINAL).astype(np.float32)
        h_mag = np.zeros((N, N_HARMONICS), dtype=np.float32)
        h_phase = np.zeros((N, N_HARMONICS), dtype=np.float32)
        active_mask = np.isin(state, ["compressor_on", "compressor_starting"])
        if active_mask.any():
            m, p = synthesize_harmonics(I_fund[active_mask],
                                        self.HARMONICS_ON, self.rng)
            h_mag[active_mask] = m
            h_phase[active_mask] = p

        meta = self._make_metadata({
            "compressor_P_W": comp_P,
            "compressor_Q_var": comp_Q,
            "inrush_factor_used": inrush_f,
        })
        return ApplianceTrace(ts, P, Q, state, h_mag, h_phase, meta)


# ===========================================================================
# 2. Variable resistive load
# ===========================================================================

class ResistiveLoadGenerator(ApplianceGenerator):
    name = "resistive"
    appliance_type = "III"
    DEFAULT_PARAMS = {
        "P_max_W":            [500.0, 3000.0],
        "ramp_rate_W_per_s":  [50.0, 500.0],
        "active_periods":     [1, 3],
        "period_duration_s":  [60, 1800],
    }
    HARMONICS = {3: 0.005}  # essentially clean

    def generate(self, duration_s, sample_rate_hz, anchor):
        ts, N, _ = self._time_axis(duration_s, sample_rate_hz, anchor)
        P_max = self._sample_range("P_max_W")
        ramp = self._sample_range("ramp_rate_W_per_s")
        n_periods = int(round(self._sample_range("active_periods")))

        P = np.zeros(N, dtype=np.float32)
        Q = np.zeros(N, dtype=np.float32)
        state = np.full(N, "off", dtype=object)

        for _ in range(n_periods):
            t_start = self.rng.uniform(0, duration_s)
            dur = self._sample_range("period_duration_s")
            i0 = max(0, int(t_start * sample_rate_hz))
            i1 = min(N, int((t_start + dur) * sample_rate_hz))
            if i0 >= i1:
                continue
            # Piecewise-linear setpoint trajectory
            n_seg = max(1, int(dur / 30.0))  # new target every ~30 s
            targets = self.rng.uniform(0, P_max, size=n_seg)
            seg_ix = np.linspace(i0, i1, n_seg + 1).astype(int)
            current = float(P[i0 - 1]) if i0 > 0 else 0.0
            for k in range(n_seg):
                a, b = seg_ix[k], seg_ix[k + 1]
                target = targets[k]
                if b <= a:
                    continue
                t_seg = np.arange(b - a) / sample_rate_hz
                step = np.sign(target - current) * ramp * t_seg
                vals = np.clip(current + step, 0, P_max) if target >= current \
                       else np.clip(current + step, target, P_max)
                # Simpler: linear interpolation toward target
                vals = np.linspace(current, target, b - a)
                P[a:b] = vals.astype(np.float32)
                current = float(vals[-1])
            state[i0:i1] = "active"

        S = P
        I_fund = (S / V_NOMINAL).astype(np.float32)
        h_mag = np.zeros((N, N_HARMONICS), dtype=np.float32)
        h_phase = np.zeros((N, N_HARMONICS), dtype=np.float32)
        mask = state == "active"
        if mask.any():
            m, p = synthesize_harmonics(I_fund[mask], self.HARMONICS, self.rng)
            h_mag[mask] = m
            h_phase[mask] = p

        return ApplianceTrace(ts, P, Q, state, h_mag, h_phase,
                              self._make_metadata({"P_max_W": P_max,
                                                   "ramp_rate_W_per_s": ramp}))


# ===========================================================================
# 3. Hair dryer
# ===========================================================================

class HairDryerGenerator(ApplianceGenerator):
    name = "hair_dryer"
    appliance_type = "II"
    DEFAULT_PARAMS = {
        "P_high_heat_high_fan_W":  [1700, 2200],
        "P_high_heat_low_fan_W":   [1100, 1400],
        "P_low_heat_W":            [400, 550],
        "P_cold_fan_W":            [40, 80],
        "burst_count_per_day":     [0, 2],
        "burst_duration_s":        [120, 360],
    }
    HARMONICS_HEAT = {3: 0.01}
    HARMONICS_FAN = {3: 0.04, 5: 0.015, 7: 0.005}

    def generate(self, duration_s, sample_rate_hz, anchor):
        ts, N, hour = self._time_axis(duration_s, sample_rate_hz, anchor)
        P = np.zeros(N, dtype=np.float32)
        Q = np.zeros(N, dtype=np.float32)
        state = np.full(N, "off", dtype=object)

        intensity = time_of_day_intensity(hour, "morning_evening_peaks")
        n_bursts = int(round(self._sample_range("burst_count_per_day")))

        # Sample burst start times weighted by intensity
        # (rejection sampling)
        bursts = []
        attempts = 0
        while len(bursts) < n_bursts and attempts < 1000:
            cand_i = self.rng.integers(0, N)
            if self.rng.uniform() < intensity[cand_i]:
                bursts.append(cand_i)
            attempts += 1

        for i0 in bursts:
            dur = self._sample_range("burst_duration_s")
            i1 = min(N, i0 + int(dur * sample_rate_hz))
            # Within the burst: brief fan-only → heat_high → fan-only_cold
            seg_a = i0 + int(20 * sample_rate_hz)
            seg_b = i1 - int(15 * sample_rate_hz)
            seg_a = min(seg_a, i1)
            seg_b = min(max(seg_b, seg_a), i1)
            if i0 < seg_a:
                cold = self._sample_range("P_cold_fan_W")
                P[i0:seg_a] = cold
                Q[i0:seg_a] = cold * 0.3
                state[i0:seg_a] = "cold_fan"
            if seg_a < seg_b:
                # alternate heat levels
                p_high = self._sample_range("P_high_heat_high_fan_W")
                p_mid = self._sample_range("P_high_heat_low_fan_W")
                # Split this segment in 3
                third = (seg_b - seg_a) // 3
                if third > 0:
                    P[seg_a:seg_a + third] = p_mid
                    P[seg_a + third:seg_a + 2 * third] = p_high
                    P[seg_a + 2 * third:seg_b] = p_mid
                    Q[seg_a:seg_b] = P[seg_a:seg_b] * 0.04
                    state[seg_a:seg_b] = "heat_high"
            if seg_b < i1:
                cold = self._sample_range("P_cold_fan_W")
                P[seg_b:i1] = cold
                Q[seg_b:i1] = cold * 0.3
                state[seg_b:i1] = "cold_fan"

        I_fund = (np.sqrt(P ** 2 + Q ** 2) / V_NOMINAL).astype(np.float32)
        h_mag = np.zeros((N, N_HARMONICS), dtype=np.float32)
        h_phase = np.zeros((N, N_HARMONICS), dtype=np.float32)
        mask_heat = state == "heat_high"
        mask_cold = state == "cold_fan"
        if mask_heat.any():
            m, p = synthesize_harmonics(I_fund[mask_heat], self.HARMONICS_HEAT, self.rng)
            h_mag[mask_heat] = m
            h_phase[mask_heat] = p
        if mask_cold.any():
            m, p = synthesize_harmonics(I_fund[mask_cold], self.HARMONICS_FAN, self.rng)
            h_mag[mask_cold] = m
            h_phase[mask_cold] = p

        return ApplianceTrace(ts, P, Q, state, h_mag, h_phase,
                              self._make_metadata({"n_bursts": len(bursts)}))


# ===========================================================================
# 4. PC
# ===========================================================================

class PCGenerator(ApplianceGenerator):
    name = "pc"
    appliance_type = "IV"
    DEFAULT_PARAMS = {
        "standby_P_W":          [2.0, 5.0],
        "idle_P_W":             [30.0, 70.0],
        "active_P_W":           [80.0, 200.0],
        "peak_P_W":             [250.0, 400.0],
        "PF_active":            [0.75, 0.9],
        "peak_fraction":        [0.02, 0.10],
        "weekend_active_prob":  [0.0, 0.5],
        "is_weekday":           True,
    }
    HARMONICS = {3: 0.20, 5: 0.10, 7: 0.05, 9: 0.03, 11: 0.02, 13: 0.01}

    def generate(self, duration_s, sample_rate_hz, anchor):
        ts, N, hour = self._time_axis(duration_s, sample_rate_hz, anchor)
        idle_P = self._sample_range("idle_P_W")
        active_P = self._sample_range("active_P_W")
        peak_P = self._sample_range("peak_P_W")
        standby_P = self._sample_range("standby_P_W")
        peak_frac = self._sample_range("peak_fraction")
        pf = self._sample_range("PF_active")

        is_weekday = bool(self.params.get("is_weekday", True))
        intensity = time_of_day_intensity(hour, "working_hours_weekday")
        if not is_weekday:
            intensity *= self._sample_range("weekend_active_prob")

        P = np.full(N, standby_P, dtype=np.float32)
        state = np.full(N, "standby", dtype=object)

        # For each sample, decide state by intensity threshold + random
        u = self.rng.uniform(0, 1, size=N)
        active_mask = u < intensity * 0.8
        peak_mask = active_mask & (self.rng.uniform(0, 1, size=N) < peak_frac)

        P[active_mask] = idle_P
        state[active_mask] = "idle"

        # Within active sessions, sometimes go to "active" (heavier load)
        heavy_mask = active_mask & (self.rng.uniform(0, 1, size=N) < 0.3)
        P[heavy_mask] = active_P
        state[heavy_mask] = "active"

        # Peak events
        P[peak_mask] = peak_P
        state[peak_mask] = "peak"

        # Smooth P with short rolling mean to remove single-sample flicker
        P = _rolling_mean(P, max(1, int(sample_rate_hz * 3)))

        Q = P * np.tan(np.arccos(pf)).astype(np.float32)

        I_fund = (np.sqrt(P ** 2 + Q ** 2) / V_NOMINAL).astype(np.float32)
        h_mag = np.zeros((N, N_HARMONICS), dtype=np.float32)
        h_phase = np.zeros((N, N_HARMONICS), dtype=np.float32)
        active = P > standby_P * 1.5
        if active.any():
            m, p = synthesize_harmonics(I_fund[active], self.HARMONICS, self.rng)
            h_mag[active] = m
            h_phase[active] = p

        return ApplianceTrace(ts, P.astype(np.float32), Q.astype(np.float32),
                              state, h_mag, h_phase,
                              self._make_metadata({"PF": pf}))


def _rolling_mean(x, w):
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same").astype(x.dtype)


# ===========================================================================
# 5. Washing machine
# ===========================================================================

class WashingMachineGenerator(ApplianceGenerator):
    name = "washing_machine"
    appliance_type = "II"
    DEFAULT_PARAMS = {
        "program":              "warm",     # hot | warm | cold | eco
        "heat_P_W":             [1800, 2300],
        "motor_P_W":            [100, 250],
        "spin_max_P_W":         [300, 600],
        "cycle_probability":    0.5,        # chance there's a wash this day
    }
    HARMONICS_HEAT = {3: 0.01}
    HARMONICS_MOTOR = {3: 0.05, 5: 0.02, 7: 0.01}

    def generate(self, duration_s, sample_rate_hz, anchor):
        ts, N, hour = self._time_axis(duration_s, sample_rate_hz, anchor)
        P = np.zeros(N, dtype=np.float32)
        Q = np.zeros(N, dtype=np.float32)
        state = np.full(N, "off", dtype=object)

        # Single cycle per day with probability
        if self.rng.uniform() < self.params["cycle_probability"]:
            # Start mid-day or evening
            t_start_hour = float(self.rng.choice([11.0, 13.0, 18.0, 20.0])) \
                           + self.rng.uniform(-0.5, 0.5)
            t_start = t_start_hour * 3600.0
            i = int(t_start * sample_rate_hz)
            i = max(0, min(N - 1, i))

            heat_P = self._sample_range("heat_P_W")
            motor_P = self._sample_range("motor_P_W")
            spin_P_max = self._sample_range("spin_max_P_W")
            program = self.params["program"]
            heat_dur_s = {"hot": 1500, "warm": 900, "cold": 0, "eco": 600}.get(program, 900)

            i = self._fill(P, Q, state, i, 90, "fill", N, sample_rate_hz, 50, 20)
            if heat_dur_s > 0:
                i = self._fill(P, Q, state, i, heat_dur_s, "heat", N, sample_rate_hz,
                               heat_P, 50)
            i = self._motor_phase(P, Q, state, i, 1800, "wash_agitate", N, sample_rate_hz, motor_P)
            i = self._fill(P, Q, state, i, 60, "pump_out", N, sample_rate_hz, 300, 150)
            i = self._fill(P, Q, state, i, 120, "fill", N, sample_rate_hz, 50, 20)
            i = self._motor_phase(P, Q, state, i, 900, "rinse", N, sample_rate_hz, motor_P)
            i = self._fill(P, Q, state, i, 60, "pump_out", N, sample_rate_hz, 300, 150)
            i = self._spin(P, Q, state, i, 480, "spin", N, sample_rate_hz, spin_P_max)
            i = self._fill(P, Q, state, i, 60, "pump_drain_final", N, sample_rate_hz, 300, 150)

        I_fund = (np.sqrt(P ** 2 + Q ** 2) / V_NOMINAL).astype(np.float32)
        h_mag = np.zeros((N, N_HARMONICS), dtype=np.float32)
        h_phase = np.zeros((N, N_HARMONICS), dtype=np.float32)
        heat_mask = state == "heat"
        motor_mask = np.isin(state, ["wash_agitate", "rinse", "spin", "pump_out", "pump_drain_final"])
        if heat_mask.any():
            m, p = synthesize_harmonics(I_fund[heat_mask], self.HARMONICS_HEAT, self.rng)
            h_mag[heat_mask] = m
            h_phase[heat_mask] = p
        if motor_mask.any():
            m, p = synthesize_harmonics(I_fund[motor_mask], self.HARMONICS_MOTOR, self.rng)
            h_mag[motor_mask] = m
            h_phase[motor_mask] = p

        return ApplianceTrace(ts, P, Q, state, h_mag, h_phase, self._make_metadata())

    def _fill(self, P, Q, state, i, dur_s, label, N, rate, p_W, q_var):
        i1 = min(N, i + int(dur_s * rate))
        P[i:i1] = p_W
        Q[i:i1] = q_var
        state[i:i1] = label
        return i1

    def _motor_phase(self, P, Q, state, i, dur_s, label, N, rate, p_motor):
        """Motor cycles on/off ~10 s on / ~10 s off through the phase."""
        i1 = min(N, i + int(dur_s * rate))
        block = int(10 * rate)
        cursor = i
        on = True
        while cursor < i1:
            end = min(cursor + block, i1)
            if on:
                P[cursor:end] = p_motor
                Q[cursor:end] = p_motor * 0.7
                state[cursor:end] = label
            else:
                P[cursor:end] = 5.0
                Q[cursor:end] = 1.0
                state[cursor:end] = label + "_idle"
            cursor = end
            on = not on
        return i1

    def _spin(self, P, Q, state, i, dur_s, label, N, rate, p_max):
        i1 = min(N, i + int(dur_s * rate))
        n = i1 - i
        if n <= 0:
            return i1
        # Ramp up over first 30%, hold, ramp down last 20%
        profile = np.linspace(0.3, 1.0, max(1, int(n * 0.3)))
        hold = np.full(int(n * 0.5), 1.0)
        down = np.linspace(1.0, 0.4, n - len(profile) - len(hold))
        prof = np.concatenate([profile, hold, down])[:n]
        P[i:i1] = (prof * p_max).astype(np.float32)
        Q[i:i1] = (P[i:i1] * 0.7).astype(np.float32)
        state[i:i1] = label
        return i1


# ===========================================================================
# 6. EV charging (multiple modes)
# ===========================================================================

class EVGenerator(ApplianceGenerator):
    name = "ev"
    appliance_type = "III"
    DEFAULT_PARAMS = {
        "mode":                 "slow_AC",   # slow_AC | fast_AC_11kW | fast_AC_22kW | smart_modulated
        "battery_capacity_kWh": [40.0, 100.0],
        "start_SoC":            [0.10, 0.60],
        "charger_quality":      "mid",       # budget | mid | premium
        "session_probability":  0.7,
    }
    MODE_POWER = {
        "slow_AC": 3700.0,
        "fast_AC_11kW": 11000.0,
        "fast_AC_22kW": 22000.0,
        "smart_modulated": 11000.0,
    }
    HARMONICS_BY_QUALITY = {
        "budget":  {3: 0.08, 5: 0.06, 7: 0.04, 9: 0.02},
        "mid":     {3: 0.03, 5: 0.025, 7: 0.015, 9: 0.005},
        "premium": {3: 0.01, 5: 0.01, 7: 0.005},
    }

    def generate(self, duration_s, sample_rate_hz, anchor):
        ts, N, hour = self._time_axis(duration_s, sample_rate_hz, anchor)
        P = np.zeros(N, dtype=np.float32)
        Q = np.zeros(N, dtype=np.float32)
        state = np.full(N, "off", dtype=object)

        if self.rng.uniform() > self.params["session_probability"]:
            return ApplianceTrace(ts, P, Q, state,
                                  np.zeros((N, N_HARMONICS), dtype=np.float32),
                                  np.zeros((N, N_HARMONICS), dtype=np.float32),
                                  self._make_metadata({"session": False}))

        mode = self.params["mode"]
        P_target = self.MODE_POWER[mode]
        capacity_kWh = self._sample_range("battery_capacity_kWh")
        start_SoC = self._sample_range("start_SoC")
        energy_needed_Wh = capacity_kWh * 1000 * (1.0 - start_SoC)
        # Assume PF 0.98 -> Q small but positive
        pf = 0.98

        if mode == "smart_modulated":
            # Daytime + evening, modulated
            t_plug = self.rng.uniform(8.0, 18.0) * 3600.0
            session_dur = energy_needed_Wh / P_target * 3600.0
            self._smart_session(P, Q, state, t_plug, session_dur, P_target, pf,
                                N, sample_rate_hz)
        else:
            # Overnight: plug in 21:00–23:00
            t_plug = self.rng.uniform(21.0, 23.0) * 3600.0
            # Realistic session duration
            session_dur = energy_needed_Wh / P_target * 3600.0
            self._steady_session(P, Q, state, t_plug, session_dur, P_target, pf,
                                 N, sample_rate_hz)

        I_fund = (np.sqrt(P ** 2 + Q ** 2) / V_NOMINAL).astype(np.float32)
        h_mag = np.zeros((N, N_HARMONICS), dtype=np.float32)
        h_phase = np.zeros((N, N_HARMONICS), dtype=np.float32)
        active_mask = P > 100
        if active_mask.any():
            quality = self.params.get("charger_quality", "mid")
            profile = self.HARMONICS_BY_QUALITY[quality]
            m, p = synthesize_harmonics(I_fund[active_mask], profile, self.rng)
            h_mag[active_mask] = m
            h_phase[active_mask] = p

        return ApplianceTrace(ts, P, Q, state, h_mag, h_phase,
                              self._make_metadata({"mode": mode,
                                                   "session_duration_h": session_dur/3600.0}))

    def _steady_session(self, P, Q, state, t_plug, dur_s, P_target, pf,
                        N, rate):
        i0 = max(0, int(t_plug * rate))
        i1 = min(N, int((t_plug + dur_s) * rate))
        if i0 >= i1:
            return
        # Ramp-up over ~3 s
        ramp_n = int(3 * rate)
        ramp_n = min(ramp_n, i1 - i0)
        P[i0:i0 + ramp_n] = np.linspace(0, P_target, ramp_n)
        # Constant power
        cv_n = int(min(0.85 * (i1 - i0 - ramp_n), i1 - i0 - ramp_n))
        if cv_n > 0:
            P[i0 + ramp_n:i0 + ramp_n + cv_n] = P_target
        # Taper at the end (CV phase)
        taper_start = i0 + ramp_n + cv_n
        if taper_start < i1:
            n_taper = i1 - taper_start
            P[taper_start:i1] = np.linspace(P_target, P_target * 0.15, n_taper)
        Q[i0:i1] = P[i0:i1] * np.tan(np.arccos(pf))
        state[i0:i1] = "charging"

    def _smart_session(self, P, Q, state, t_plug, dur_s, P_max, pf, N, rate):
        i0 = max(0, int(t_plug * rate))
        i1 = min(N, int((t_plug + dur_s * 1.5) * rate))  # smart sessions take longer
        if i0 >= i1:
            return
        # Setpoint follows a slowly varying signal in [0.2, 1.0] of P_max
        # Implement as a sum of two sines with random phases
        n = i1 - i0
        t = np.arange(n) / rate
        setpoint = 0.6 + 0.3 * np.sin(2 * np.pi * t / 1200 + self.rng.uniform(0, np.pi)) \
                       + 0.1 * np.sin(2 * np.pi * t / 300 + self.rng.uniform(0, np.pi))
        setpoint = np.clip(setpoint, 0.2, 1.0)
        P[i0:i1] = (setpoint * P_max).astype(np.float32)
        Q[i0:i1] = P[i0:i1] * np.tan(np.arccos(pf))
        state[i0:i1] = "smart_charging"


# ===========================================================================
# 7. PV
# ===========================================================================

class PVGenerator(ApplianceGenerator):
    name = "pv"
    appliance_type = "special"
    is_three_phase = True
    DEFAULT_PARAMS = {
        "panel_count":         3,
        "panel_Wp":            [300.0, 400.0],
        "latitude_deg":        51.0,
        "cloud_intensity":     [0.0, 0.6],   # fraction of day under cloud
        "inverter_THD_pct":    [1.0, 5.0],
        "Q_setpoint_var":      0.0,
    }
    HARMONICS = {3: 0.005, 5: 0.02, 7: 0.015, 11: 0.005, 13: 0.003}

    def generate(self, duration_s, sample_rate_hz, anchor):
        ts, N, hour = self._time_axis(duration_s, sample_rate_hz, anchor)
        panel_count = int(self.params["panel_count"])
        panel_Wp = self._sample_range("panel_Wp")
        lat = float(self.params["latitude_deg"])
        peak_W = panel_count * panel_Wp

        clear_sky = solar_intensity(hour, anchor, lat)

        # Cloud cover: 1-minute autocorrelated multiplier in [0.3, 1.0],
        # with occasional drops
        cloud_intensity = self._sample_range("cloud_intensity")
        cloud = self._cloud_process(N, sample_rate_hz, cloud_intensity)

        # P is NEGATIVE for generation
        P_mag = clear_sky * cloud * peak_W
        P = -P_mag.astype(np.float32)
        Q = np.full(N, float(self.params["Q_setpoint_var"]), dtype=np.float32)
        state = np.where(P_mag > 5, "generating", "off")

        # Harmonics
        I_fund = (np.abs(P) / (V_NOMINAL * panel_count)).astype(np.float32)  # rough per-phase
        h_mag = np.zeros((N, N_HARMONICS), dtype=np.float32)
        h_phase = np.zeros((N, N_HARMONICS), dtype=np.float32)
        active_mask = P_mag > 5
        if active_mask.any():
            # PV phase is characteristically different (near zero relative to fundamental zero crossing)
            m, p = synthesize_harmonics(I_fund[active_mask], self.HARMONICS, self.rng,
                                        phase_jitter_rad=0.02)
            # Bias phases near 0 (PV switching-aligned)
            p = p * 0.2  # tighter phase clustering
            h_mag[active_mask] = m
            h_phase[active_mask] = p

        return ApplianceTrace(ts, P, Q, state, h_mag, h_phase,
                              self._make_metadata({"peak_W": peak_W,
                                                   "cloud_intensity": cloud_intensity}))

    def _cloud_process(self, N, rate, intensity):
        """Slow autocorrelated cloud cover in [0.2, 1.0]."""
        # AR(1) at coarse resolution then upsampled
        coarse_rate = 1.0 / 30.0  # one value per 30 seconds
        n_coarse = int(N / rate * coarse_rate) + 2
        x = np.zeros(n_coarse)
        x[0] = 1.0
        for k in range(1, n_coarse):
            x[k] = 0.95 * x[k - 1] + self.rng.normal(0, 0.05)
        x = np.clip(x, 1.0 - intensity, 1.0)
        # Upsample
        coarse_t = np.linspace(0, N - 1, n_coarse)
        fine_t = np.arange(N)
        return np.interp(fine_t, coarse_t, x).astype(np.float32)


# ===========================================================================
# 8. Synchronous machine
# ===========================================================================

class SynchronousMachineGenerator(ApplianceGenerator):
    name = "synchronous"
    appliance_type = "special"
    is_three_phase = True
    DEFAULT_PARAMS = {
        "mode_schedule":        None,        # list of (start_h, end_h, mode, P, Q)
        "operating_hours":      [9.0, 17.0],
        "transition_rate_W_per_s": [10, 100],
    }
    MODES = {
        "motor_underexcited":      (+1, +1),
        "motor_overexcited":       (+1, -1),
        "generator_underexcited":  (-1, +1),
        "generator_overexcited":   (-1, -1),
    }
    HARMONICS = {5: 0.04, 7: 0.025, 11: 0.01, 13: 0.005}

    def generate(self, duration_s, sample_rate_hz, anchor):
        ts, N, hour = self._time_axis(duration_s, sample_rate_hz, anchor)
        P = np.zeros(N, dtype=np.float32)
        Q = np.zeros(N, dtype=np.float32)
        state = np.full(N, "off", dtype=object)

        op_start, op_end = self.params["operating_hours"]
        sched = self.params.get("mode_schedule")
        if sched is None:
            # Build a default schedule cycling through all four quadrants
            modes = list(self.MODES.keys())
            sched = []
            cur = op_start
            while cur < op_end - 0.5:
                end = min(op_end, cur + self.rng.uniform(0.5, 1.5))
                m = modes[len(sched) % 4]
                P_set = self.rng.uniform(300, 1500)
                Q_set = self.rng.uniform(200, 1000)
                sched.append((cur, end, m, P_set, Q_set))
                cur = end

        rate_W = self._sample_range("transition_rate_W_per_s")

        for (h0, h1, mode, P_set, Q_set) in sched:
            i0 = max(0, int(h0 * 3600 * sample_rate_hz))
            i1 = min(N, int(h1 * 3600 * sample_rate_hz))
            if i0 >= i1:
                continue
            sP, sQ = self.MODES[mode]
            target_P = sP * P_set
            target_Q = sQ * Q_set
            # Smooth transition
            n_trans = min(int(abs(target_P - P[i0]) / rate_W * sample_rate_hz), i1 - i0)
            if n_trans > 0:
                P[i0:i0 + n_trans] = np.linspace(P[i0 - 1] if i0 > 0 else 0,
                                                  target_P, n_trans)
                Q[i0:i0 + n_trans] = np.linspace(Q[i0 - 1] if i0 > 0 else 0,
                                                  target_Q, n_trans)
            P[i0 + n_trans:i1] = target_P
            Q[i0 + n_trans:i1] = target_Q
            state[i0:i1] = mode

        I_fund = (np.sqrt(P ** 2 + Q ** 2) / (V_NOMINAL * 3)).astype(np.float32)
        h_mag = np.zeros((N, N_HARMONICS), dtype=np.float32)
        h_phase = np.zeros((N, N_HARMONICS), dtype=np.float32)
        mask = state != "off"
        if mask.any():
            m, p = synthesize_harmonics(I_fund[mask], self.HARMONICS, self.rng)
            h_mag[mask] = m
            h_phase[mask] = p

        return ApplianceTrace(ts, P, Q, state, h_mag, h_phase,
                              self._make_metadata({"schedule": [list(s) for s in sched]}))


# ===========================================================================
# 9. Baseload (always-on)
# ===========================================================================

class BaseloadGenerator(ApplianceGenerator):
    name = "baseload"
    appliance_type = "IV"
    DEFAULT_PARAMS = {
        "P_W": [15.0, 30.0],
        "Q_var": [3.0, 10.0],
        "noise_W": 0.5,
    }
    HARMONICS = {3: 0.08, 5: 0.04, 7: 0.02}

    def generate(self, duration_s, sample_rate_hz, anchor):
        ts, N, _ = self._time_axis(duration_s, sample_rate_hz, anchor)
        p_base = self._sample_range("P_W")
        q_base = self._sample_range("Q_var")
        noise = float(self.params.get("noise_W", 0.5))
        P = (p_base + self.rng.normal(0, noise, N)).astype(np.float32)
        Q = np.full(N, q_base, dtype=np.float32)
        state = np.full(N, "on", dtype=object)
        I_fund = (np.sqrt(P ** 2 + Q ** 2) / V_NOMINAL).astype(np.float32)
        h_mag, h_phase = synthesize_harmonics(I_fund, self.HARMONICS, self.rng)
        return ApplianceTrace(ts, P, Q, state, h_mag, h_phase,
                              self._make_metadata({"P_base_W": p_base}))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: Dict[str, type] = {
    "fridge":          FridgeGenerator,
    "resistive":       ResistiveLoadGenerator,
    "hair_dryer":      HairDryerGenerator,
    "pc":              PCGenerator,
    "washing_machine": WashingMachineGenerator,
    "ev":              EVGenerator,
    "pv":              PVGenerator,
    "synchronous":     SynchronousMachineGenerator,
    "baseload":        BaseloadGenerator,
}


# ---------------------------------------------------------------------------
# HDF5 export
# ---------------------------------------------------------------------------

def save_trace_hdf5(path: str, trace: ApplianceTrace, sample_rate_hz: float,
                    anchor_datetime: datetime, tier: str = "single_appliance"):
    if not HAVE_H5PY:
        print(f"[warn] h5py not installed; cannot write {path}. Install with: pip install h5py",
              file=sys.stderr)
        return
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamp", data=trace.timestamp_us, compression="lzf")
        m = f.create_group("measurements")
        m.create_dataset("P", data=trace.P, compression="lzf")
        m.create_dataset("Q", data=trace.Q, compression="lzf")
        m.create_dataset("harmonics_I_mag", data=trace.harmonics_I_mag, compression="lzf")
        m.create_dataset("harmonics_I_phase", data=trace.harmonics_I_phase, compression="lzf")
        g = f.create_group("ground_truth")
        # Convert state strings to fixed-length to store in HDF5
        states_bytes = np.array([str(s) for s in trace.state], dtype="S32")
        g.create_dataset("state", data=states_bytes, compression="lzf")
        g.create_dataset("P_contribution", data=trace.P, compression="lzf")
        md = f.create_group("metadata")
        md.attrs["format_version"] = FORMAT_VERSION
        md.attrs["generator_version"] = GENERATOR_VERSION
        md.attrs["sample_rate_hz"] = sample_rate_hz
        md.attrs["anchor_datetime"] = anchor_datetime.isoformat()
        md.attrs["tier"] = tier
        md.attrs["appliance_metadata"] = json.dumps(trace.metadata)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_trace(trace: ApplianceTrace, sample_rate_hz: float, name: str = ""):
    if not HAVE_MPL:
        print("[warn] matplotlib not installed; skipping plot", file=sys.stderr)
        return
    t_h = (trace.timestamp_us - trace.timestamp_us[0]) / 1e6 / 3600.0
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t_h, trace.P, lw=0.6)
    axes[0].set_ylabel("P (W)")
    axes[0].set_title(f"{name} — P, Q, harmonic 3rd magnitude")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t_h, trace.Q, lw=0.6, color="orange")
    axes[1].set_ylabel("Q (var)")
    axes[1].grid(alpha=0.3)
    # 3rd harmonic = index 1
    if trace.harmonics_I_mag.shape[1] > 1:
        axes[2].plot(t_h, trace.harmonics_I_mag[:, 1], lw=0.6, color="green")
        axes[2].set_ylabel("|I_3rd| (A)")
    axes[2].set_xlabel("hour of day")
    axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate synthetic per-appliance traces (NILM project, M1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available appliances: " + ", ".join(GENERATORS.keys()),
    )
    p.add_argument("--appliance", help="appliance name; --list to see all")
    p.add_argument("--list", action="store_true", help="list appliances and exit")
    p.add_argument("--duration", type=float, default=86400.0,
                   help="duration in seconds (default 86400 = 24 h)")
    p.add_argument("--rate", type=float, default=5.0, help="sample rate Hz (default 5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--anchor-date", default="2024-01-01",
                   help="anchor date YYYY-MM-DD (UTC, 00:00:00)")
    p.add_argument("--phase", default="L1", help="L1/L2/L3 for single-phase (ignored for 3ph)")
    p.add_argument("--instance-id", type=int, default=1)
    p.add_argument("--params", default=None,
                   help="JSON dict of parameter overrides for the generator")
    p.add_argument("--output", default=None,
                   help="HDF5 output file (default: <appliance>.h5)")
    p.add_argument("--no-save", action="store_true", help="skip writing HDF5")
    p.add_argument("--plot", action="store_true", help="show a quick plot of P, Q, 3rd harmonic")
    args = p.parse_args()

    if args.list:
        print("Available appliances:")
        for k, cls in GENERATORS.items():
            print(f"  {k:18s}  type {cls.appliance_type:8s}  "
                  f"{'3-phase' if cls.is_three_phase else 'single-phase'}")
        return

    if not args.appliance:
        p.error("--appliance is required (or use --list)")
    if args.appliance not in GENERATORS:
        p.error(f"unknown appliance {args.appliance}; see --list")

    anchor = datetime.strptime(args.anchor_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    params = json.loads(args.params) if args.params else None
    cls = GENERATORS[args.appliance]
    gen = cls(params=params, seed=args.seed, phase=args.phase, instance_id=args.instance_id)
    trace = gen.generate(args.duration, args.rate, anchor)

    print(f"Generated {args.appliance}: N={len(trace.P)} samples, "
          f"P range [{trace.P.min():.1f}, {trace.P.max():.1f}] W, "
          f"Q range [{trace.Q.min():.1f}, {trace.Q.max():.1f}] var, "
          f"unique states: {sorted(set(str(s) for s in trace.state))}")

    if not args.no_save:
        out = args.output or f"{args.appliance}.h5"
        save_trace_hdf5(out, trace, args.rate, anchor)
        print(f"Saved: {out}")

    if args.plot:
        plot_trace(trace, args.rate, name=args.appliance)


if __name__ == "__main__":
    main()