#!/usr/bin/env python3
"""
NILM Project - Scenario Aggregator
===================================
Combines per-appliance HDF5 files (from Appliance_generator.py) into a
single scenario file that mimics what a Siemens PAC4200 would log at the
Point of Common Coupling.

This is a SYNTHETIC-DATA-ONLY step. In a real deployment the PAC4200
already provides aggregated signals - the aggregator simulates that
physics so the downstream preprocessing/ML pipeline is identical for
real and synthetic data.

Output is CLEAN aggregate data. Noise injection and Modbus-dropout
simulation (to simulate real PAC4200 output) live in the preprocessing
step, not here.

Usage
-----
    # Aggregate explicit list of files
    python scenario_aggregator.py \
        --inputs fridge.h5 pc.h5 hair_dryer.h5 pv.h5 baseload.h5 \
        --output scenario_001.h5 --tier train --seed 42

    # Aggregate every .h5 file in a directory
    python scenario_aggregator.py \
        --input-dir ./appliances --output scenario_001.h5 --tier train

    # Inspect aggregated output without writing
    python scenario_aggregator.py --inputs *.h5 --inspect
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py not installed. Run: pip install h5py", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V_NOMINAL = 230.0           # nominal phase voltage (L-N), V
F_NOMINAL = 50.0            # nominal frequency, Hz
N_HARMONICS = 39            # harmonics 2nd through 40th
FORMAT_VERSION = "0.1"
AGGREGATOR_VERSION = "0.1.0"

# Small fixed voltage harmonic content (typical of stiff grid)
# These would emerge from grid impedance × load current harmonics in reality.
# For an "infinite bus" approximation we put small constant fractions of V_NOMINAL.
V_HARMONIC_FRACTIONS = {3: 0.005, 5: 0.015, 7: 0.010, 11: 0.005, 13: 0.003}


# ---------------------------------------------------------------------------
# Loading per-appliance files
# ---------------------------------------------------------------------------

def load_appliance(path: str) -> dict:
    """Load a per-appliance HDF5 file produced by Appliance_generator.py."""
    with h5py.File(path, "r") as f:
        out = {
            "path":      path,
            "timestamp": f["timestamp"][:],
            "P":         f["measurements/P"][:],
            "Q":         f["measurements/Q"][:],
            "h_mag":     f["measurements/harmonics_I_mag"][:],
            "h_phase":   f["measurements/harmonics_I_phase"][:],
            "state":     [s.decode() for s in f["ground_truth/state"][:]],
            "sample_rate_hz": float(f["metadata"].attrs["sample_rate_hz"]),
            "anchor_datetime": str(f["metadata"].attrs["anchor_datetime"]),
        }
        out["metadata"] = json.loads(str(f["metadata"].attrs["appliance_metadata"]))
    return out


def validate_alignment(appliances: List[dict], allow_anchor_mismatch: bool = False):
    """Check that all loaded appliances share the same timeline."""
    if not appliances:
        raise ValueError("No appliance files supplied.")
    ref = appliances[0]
    for app in appliances[1:]:
        if len(app["timestamp"]) != len(ref["timestamp"]):
            raise ValueError(
                f"Length mismatch: {ref['path']} has {len(ref['timestamp'])} samples, "
                f"{app['path']} has {len(app['timestamp'])}.")
        if app["sample_rate_hz"] != ref["sample_rate_hz"]:
            raise ValueError(
                f"Sample rate mismatch: {ref['path']} = {ref['sample_rate_hz']} Hz, "
                f"{app['path']} = {app['sample_rate_hz']} Hz.")
        if app["anchor_datetime"] != ref["anchor_datetime"]:
            msg = (f"Anchor datetime mismatch: {ref['path']} = {ref['anchor_datetime']}, "
                   f"{app['path']} = {app['anchor_datetime']}. "
                   "Appliances are not contemporaneous (e.g. PV solar position is "
                   "computed for a different date).")
            if allow_anchor_mismatch:
                print(f"WARNING: {msg}", file=sys.stderr)
            else:
                raise ValueError(msg + " Re-generate all appliances with the same "
                                 "--anchor-date, or pass --allow-anchor-mismatch.")


# ---------------------------------------------------------------------------
# Aggregation math
# ---------------------------------------------------------------------------

def aggregate_power_per_phase(appliances: List[dict], N: int) -> Tuple[Dict, Dict]:
    """Sum P and Q across all appliances, respecting phase assignment.
    3-phase appliances distribute equally across L1/L2/L3."""
    P = {p: np.zeros(N, dtype=np.float64) for p in ("L1", "L2", "L3")}
    Q = {p: np.zeros(N, dtype=np.float64) for p in ("L1", "L2", "L3")}
    for app in appliances:
        if app["metadata"]["is_three_phase"]:
            for p in ("L1", "L2", "L3"):
                P[p] += app["P"] / 3.0
                Q[p] += app["Q"] / 3.0
        else:
            ph = app["metadata"]["phase"]
            if ph not in P:
                raise ValueError(f"Unknown phase '{ph}' in {app['path']}")
            P[ph] += app["P"]
            Q[ph] += app["Q"]
    return P, Q


def aggregate_harmonics_per_phase(appliances: List[dict], N: int
                                  ) -> Dict[str, Dict[str, np.ndarray]]:
    """Sum current harmonics as COMPLEX vectors per phase. The same harmonic
    order from two appliances doesn't simply add in magnitude - phases
    interact. This is the correct physics and what makes harmonic-based
    NILM disaggregation non-trivial when multiple appliances run together."""
    complex_per_phase = {
        p: np.zeros((N, N_HARMONICS), dtype=np.complex128) for p in ("L1", "L2", "L3")
    }
    for app in appliances:
        contrib = app["h_mag"] * np.exp(1j * app["h_phase"])
        if app["metadata"]["is_three_phase"]:
            for p in ("L1", "L2", "L3"):
                complex_per_phase[p] += contrib / 3.0
        else:
            complex_per_phase[app["metadata"]["phase"]] += contrib
    return {
        p: {"mag": np.abs(complex_per_phase[p]).astype(np.float32),
            "phase": np.angle(complex_per_phase[p]).astype(np.float32)}
        for p in ("L1", "L2", "L3")
    }


def synthesize_voltage(N: int, sample_rate_hz: float, rng: np.random.Generator
                      ) -> Dict[str, np.ndarray]:
    """Generate per-phase RMS voltage as nominal + small slow drift.
    Stiff infinite-bus assumption - voltage is independent of load."""
    V = {}
    for phase in ("L1", "L2", "L3"):
        # Slow random walk on 30-second scale, bounded to ±2 V
        coarse_n = max(2, int(N / sample_rate_hz / 30) + 2)
        x = np.zeros(coarse_n)
        for k in range(1, coarse_n):
            x[k] = 0.95 * x[k - 1] + rng.normal(0, 0.3)
        x = np.clip(x, -2.0, 2.0)
        coarse_t = np.linspace(0, N - 1, coarse_n)
        fine = np.interp(np.arange(N), coarse_t, x)
        # Plus tiny per-sample jitter
        V[phase] = (V_NOMINAL + fine + rng.normal(0, 0.05, N)).astype(np.float32)
    return V


def synthesize_voltage_harmonics(V_rms_per_phase: Dict[str, np.ndarray],
                                rng: np.random.Generator
                                ) -> Dict[str, Dict[str, np.ndarray]]:
    """Generate small voltage harmonics (V_h_mag, V_h_phase) per phase.
    For an idealized stiff grid these would be near zero; we put realistic
    small fractions so THD_V has a defined value."""
    out = {}
    N = len(next(iter(V_rms_per_phase.values())))
    for phase, V_rms in V_rms_per_phase.items():
        mag = np.zeros((N, N_HARMONICS), dtype=np.float32)
        ph = np.zeros((N, N_HARMONICS), dtype=np.float32)
        for order, frac in V_HARMONIC_FRACTIONS.items():
            idx = order - 2
            mag[:, idx] = (V_rms * frac).astype(np.float32)
            base = rng.uniform(-np.pi, np.pi)
            ph[:, idx] = (base + rng.normal(0, 0.02, N)).astype(np.float32)
        out[phase] = {"mag": mag, "phase": ph}
    return out


def synthesize_frequency(N: int, sample_rate_hz: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Frequency as nominal 50 Hz with slow drift in ±0.05 Hz."""
    coarse_n = max(2, int(N / sample_rate_hz / 60) + 2)
    x = np.zeros(coarse_n)
    for k in range(1, coarse_n):
        x[k] = 0.97 * x[k - 1] + rng.normal(0, 0.01)
    x = np.clip(x, -0.05, 0.05)
    coarse_t = np.linspace(0, N - 1, coarse_n)
    return (F_NOMINAL + np.interp(np.arange(N), coarse_t, x)).astype(np.float32)


def compute_phase_currents(P: Dict[str, np.ndarray], Q: Dict[str, np.ndarray],
                          V: Dict[str, np.ndarray],
                          harmonics: Dict[str, Dict[str, np.ndarray]]
                          ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute fundamental current and true-RMS current (incl. harmonics) per phase."""
    I_fund = {}
    I_rms = {}
    for p in ("L1", "L2", "L3"):
        S_fund = np.sqrt(P[p] ** 2 + Q[p] ** 2)
        I_fund_p = np.where(V[p] > 1.0, S_fund / V[p], 0.0)
        h_sq_sum = (harmonics[p]["mag"] ** 2).sum(axis=1)
        I_fund[p] = I_fund_p.astype(np.float32)
        I_rms[p] = np.sqrt(I_fund_p ** 2 + h_sq_sum).astype(np.float32)
    return I_fund, I_rms


def compute_neutral_current(I_rms: Dict[str, np.ndarray]) -> np.ndarray:
    """Approximate I_N magnitude for an unbalanced 3-phase 4-wire system.
    Uses |I_L1 + a²·I_L2 + a·I_L3| with a = e^(j·120°). Zero when balanced."""
    a = np.exp(1j * 2 * np.pi / 3)
    I_complex = I_rms["L1"] + (a ** 2) * I_rms["L2"] + a * I_rms["L3"]
    return np.abs(I_complex).astype(np.float32)


def compute_apparent_power(V: Dict[str, np.ndarray],
                          I_rms: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {p: (V[p] * I_rms[p]).astype(np.float32) for p in ("L1", "L2", "L3")}


def compute_power_factors(P: Dict[str, np.ndarray], Q: Dict[str, np.ndarray],
                         S: Dict[str, np.ndarray]
                         ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """PF (true) = P / S; cos φ (displacement) = P / sqrt(P² + Q²).
    Uses np.divide's where= to skip division when |S| is below 1 VA (idle bus)."""
    PF = {}
    cos_phi = {}
    for p in ("L1", "L2", "L3"):
        ones = np.ones_like(S[p])
        PF[p] = np.divide(P[p], S[p], out=ones.copy(), where=(S[p] > 1.0)).astype(np.float32)
        S_disp = np.sqrt(P[p] ** 2 + Q[p] ** 2)
        cos_phi[p] = np.divide(P[p], S_disp, out=ones.copy(),
                               where=(S_disp > 1.0)).astype(np.float32)
    return PF, cos_phi


def compute_thd(harmonics_per_phase: Dict[str, Dict[str, np.ndarray]],
               fundamental_rms: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """THD per phase as a percentage: 100 × sqrt(sum H²) / fundamental_RMS.
    Returns 0 where fundamental is below 0.01 A (no measurable current)."""
    THD = {}
    for p in ("L1", "L2", "L3"):
        h_rms = np.sqrt((harmonics_per_phase[p]["mag"] ** 2).sum(axis=1))
        zeros = np.zeros_like(fundamental_rms[p])
        THD[p] = np.divide(100.0 * h_rms, fundamental_rms[p], out=zeros,
                           where=(fundamental_rms[p] > 0.01)).astype(np.float32)
    return THD


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def aggregate(input_files: List[str], scenario_seed: int = 0,
              allow_anchor_mismatch: bool = False) -> dict:
    """Load all per-appliance files and produce the full aggregate scenario."""
    appliances = [load_appliance(p) for p in input_files]
    validate_alignment(appliances, allow_anchor_mismatch=allow_anchor_mismatch)

    ref = appliances[0]
    ts = ref["timestamp"]
    N = len(ts)
    sample_rate = ref["sample_rate_hz"]
    anchor = ref["anchor_datetime"]
    rng = np.random.default_rng(scenario_seed)

    # Power flows
    P_phase, Q_phase = aggregate_power_per_phase(appliances, N)
    P_total = sum(P_phase.values())
    Q_total = sum(Q_phase.values())

    # Voltage and frequency (independent of load - stiff bus)
    V_phase = synthesize_voltage(N, sample_rate, rng)
    V_harmonics = synthesize_voltage_harmonics(V_phase, rng)
    freq = synthesize_frequency(N, sample_rate, rng)

    # Current - fundamental and true RMS
    I_harmonics = aggregate_harmonics_per_phase(appliances, N)
    I_fund, I_rms = compute_phase_currents(P_phase, Q_phase, V_phase, I_harmonics)
    I_neutral = compute_neutral_current(I_rms)

    # Derived quantities
    S_phase = compute_apparent_power(V_phase, I_rms)
    S_total = sum(S_phase.values())
    PF_phase, cosphi_phase = compute_power_factors(P_phase, Q_phase, S_phase)
    ones_total = np.ones_like(S_total)
    PF_total = np.divide(P_total, S_total, out=ones_total.copy(),
                         where=(S_total > 1.0)).astype(np.float32)
    S_disp_total = np.sqrt(P_total ** 2 + Q_total ** 2)
    cosphi_total = np.divide(P_total, S_disp_total, out=ones_total.copy(),
                             where=(S_disp_total > 1.0)).astype(np.float32)

    THD_I = compute_thd(I_harmonics, I_fund)
    THD_V = compute_thd(V_harmonics, V_phase)

    return {
        "timestamp": ts,
        "sample_rate_hz": sample_rate,
        "anchor_datetime": anchor,
        "scenario_seed": scenario_seed,
        "appliances": appliances,
        # Measurements (what a PAC4200 would expose)
        "V": V_phase,
        "I_rms": I_rms,
        "I_fundamental": I_fund,
        "I_N": I_neutral,
        "P_phase": {p: P_phase[p].astype(np.float32) for p in P_phase},
        "Q_phase": {p: Q_phase[p].astype(np.float32) for p in Q_phase},
        "S_phase": S_phase,
        "P_total": P_total.astype(np.float32),
        "Q_total": Q_total.astype(np.float32),
        "S_total": S_total.astype(np.float32),
        "PF_phase": PF_phase,
        "PF_total": PF_total,
        "cosphi_phase": cosphi_phase,
        "cosphi_total": cosphi_total,
        "freq": freq,
        "THD_V": THD_V,
        "THD_I": THD_I,
        "I_harmonics": I_harmonics,
        "V_harmonics": V_harmonics,
    }


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_scenario(path: str, agg: dict, tier: str = "train"):
    """Write the aggregated scenario to HDF5 in the format spec layout."""
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamp", data=agg["timestamp"], compression="lzf")

        # /measurements - clean aggregate data, format-matched to PAC4200 outputs
        m = f.create_group("measurements")
        for phase in ("L1", "L2", "L3"):
            m.create_dataset(f"V_{phase}", data=agg["V"][phase], compression="lzf")
            m.create_dataset(f"I_{phase}", data=agg["I_rms"][phase], compression="lzf")
            m.create_dataset(f"P_{phase}", data=agg["P_phase"][phase], compression="lzf")
            m.create_dataset(f"Q_{phase}", data=agg["Q_phase"][phase], compression="lzf")
            m.create_dataset(f"S_{phase}", data=agg["S_phase"][phase], compression="lzf")
            m.create_dataset(f"PF_{phase}", data=agg["PF_phase"][phase], compression="lzf")
            m.create_dataset(f"cosphi_{phase}", data=agg["cosphi_phase"][phase],
                             compression="lzf")
            m.create_dataset(f"THD_V_{phase}", data=agg["THD_V"][phase], compression="lzf")
            m.create_dataset(f"THD_I_{phase}", data=agg["THD_I"][phase], compression="lzf")
        m.create_dataset("I_N", data=agg["I_N"], compression="lzf")
        m.create_dataset("P_total", data=agg["P_total"], compression="lzf")
        m.create_dataset("Q_total", data=agg["Q_total"], compression="lzf")
        m.create_dataset("S_total", data=agg["S_total"], compression="lzf")
        m.create_dataset("PF_total", data=agg["PF_total"], compression="lzf")
        m.create_dataset("cosphi_total", data=agg["cosphi_total"], compression="lzf")
        m.create_dataset("freq", data=agg["freq"], compression="lzf")

        # /measurements/harmonics - per-phase mag and phase, V and I
        h = m.create_group("harmonics")
        for phase in ("L1", "L2", "L3"):
            h.create_dataset(f"I_mag_{phase}", data=agg["I_harmonics"][phase]["mag"],
                             compression="lzf")
            h.create_dataset(f"I_phase_{phase}", data=agg["I_harmonics"][phase]["phase"],
                             compression="lzf")
            h.create_dataset(f"V_mag_{phase}", data=agg["V_harmonics"][phase]["mag"],
                             compression="lzf")
            h.create_dataset(f"V_phase_{phase}", data=agg["V_harmonics"][phase]["phase"],
                             compression="lzf")

        # /ground_truth - only available because we control generation
        gt = f.create_group("ground_truth")
        appliance_names = [a["metadata"]["name"] + f"_{a['metadata']['instance_id']}"
                           for a in agg["appliances"]]
        gt.create_dataset("appliance_names",
                          data=np.array(appliance_names, dtype="S32"))
        # P contributions: (N, n_appliances)
        P_contrib = np.stack([a["P"] for a in agg["appliances"]], axis=1).astype(np.float32)
        Q_contrib = np.stack([a["Q"] for a in agg["appliances"]], axis=1).astype(np.float32)
        gt.create_dataset("P_contribution", data=P_contrib, compression="lzf")
        gt.create_dataset("Q_contribution", data=Q_contrib, compression="lzf")
        # States: (N, n_appliances) as fixed-length bytes
        states = np.stack([np.array([str(s) for s in a["state"]], dtype="S32")
                           for a in agg["appliances"]], axis=1)
        gt.create_dataset("state", data=states, compression="lzf")
        # Per-appliance metadata as JSON attributes
        for i, app in enumerate(agg["appliances"]):
            gt.attrs[f"appliance_{i}_metadata"] = json.dumps(app["metadata"])

        # /metadata
        md = f.create_group("metadata")
        md.attrs["format_version"] = FORMAT_VERSION
        md.attrs["aggregator_version"] = AGGREGATOR_VERSION
        md.attrs["sample_rate_hz"] = agg["sample_rate_hz"]
        md.attrs["anchor_datetime"] = agg["anchor_datetime"]
        md.attrs["tier"] = tier
        md.attrs["scenario_seed"] = agg["scenario_seed"]
        md.attrs["n_appliances"] = len(agg["appliances"])
        md.attrs["n_samples"] = len(agg["timestamp"])
        md.attrs["duration_seconds"] = float(len(agg["timestamp"]) / agg["sample_rate_hz"])


# ---------------------------------------------------------------------------
# Inspection helper
# ---------------------------------------------------------------------------

def print_summary(agg: dict):
    """Print a brief summary of the aggregated scenario for sanity checking."""
    N = len(agg["timestamp"])
    print(f"Scenario summary:")
    print(f"  Samples: {N} @ {agg['sample_rate_hz']} Hz "
          f"({N/agg['sample_rate_hz']/3600:.2f} h)")
    print(f"  Anchor:  {agg['anchor_datetime']}")
    print(f"  Appliances ({len(agg['appliances'])}):")
    for a in agg["appliances"]:
        name = a["metadata"]["name"]
        phase = a["metadata"]["phase"]
        p3 = "3-phase" if a["metadata"]["is_three_phase"] else f"phase={phase}"
        print(f"    - {name:18s} ({p3:9s})  "
              f"P range [{a['P'].min():7.1f}, {a['P'].max():8.1f}] W")
    print(f"  Aggregate P_total:  [{agg['P_total'].min():7.1f}, "
          f"{agg['P_total'].max():8.1f}] W   mean = {agg['P_total'].mean():7.1f} W")
    print(f"  Aggregate Q_total:  [{agg['Q_total'].min():7.1f}, "
          f"{agg['Q_total'].max():8.1f}] var mean = {agg['Q_total'].mean():7.1f} var")
    print(f"  Per-phase P means:  L1 = {agg['P_phase']['L1'].mean():.1f} W, "
          f"L2 = {agg['P_phase']['L2'].mean():.1f} W, "
          f"L3 = {agg['P_phase']['L3'].mean():.1f} W")
    print(f"  Per-phase RMS I:    L1 = {agg['I_rms']['L1'].mean():.3f} A, "
          f"L2 = {agg['I_rms']['L2'].mean():.3f} A, "
          f"L3 = {agg['I_rms']['L3'].mean():.3f} A")
    print(f"  Neutral I mean:     {agg['I_N'].mean():.3f} A")
    print(f"  THD_I per phase:    L1 = {agg['THD_I']['L1'].mean():.2f}%, "
          f"L2 = {agg['THD_I']['L2'].mean():.2f}%, "
          f"L3 = {agg['THD_I']['L3'].mean():.2f}%")
    # Invariant: sum of contributions should equal P_total (within float32 precision)
    P_sum_contrib = sum(a["P"] for a in agg["appliances"])
    invariant_err = np.abs(P_sum_contrib - agg["P_total"]).max()
    P_scale = max(1.0, np.abs(agg["P_total"]).max())
    rel_err = invariant_err / P_scale
    status = "OK" if rel_err < 1e-5 else "WARNING - should be ~ float32 precision"
    print(f"  Invariant check: max |sum(P_contrib) - P_total| = {invariant_err:.3e} W "
          f"({rel_err*100:.2e}% of peak)  [{status}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Aggregate per-appliance HDF5 files into a single scenario file "
                    "matching what a Siemens PAC4200 would log at the PCC.")
    p.add_argument("--inputs", nargs="*", default=None,
                   help="explicit list of input .h5 files")
    p.add_argument("--input-dir", default=None,
                   help="directory containing per-appliance .h5 files (loads all *.h5)")
    p.add_argument("--output", default="scenario.h5",
                   help="output HDF5 path (default scenario.h5)")
    p.add_argument("--tier", default="train",
                   choices=["train", "easy", "normal", "hard", "adversarial"])
    p.add_argument("--seed", type=int, default=0,
                   help="scenario-level seed (for synthesized V, freq, voltage harmonics)")
    p.add_argument("--inspect", action="store_true",
                   help="print summary instead of (or in addition to) writing the file")
    p.add_argument("--no-save", action="store_true", help="skip writing the file")
    p.add_argument("--allow-anchor-mismatch", action="store_true",
                   help="downgrade anchor-datetime mismatch from error to warning")
    args = p.parse_args()

    if not args.inputs and not args.input_dir:
        p.error("either --inputs or --input-dir is required")

    if args.input_dir:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5")))
        if not files:
            p.error(f"no .h5 files found in {args.input_dir}")
    else:
        files = args.inputs

    print(f"Aggregating {len(files)} per-appliance files:")
    for f in files:
        print(f"  - {f}")

    agg = aggregate(files, scenario_seed=args.seed,
                    allow_anchor_mismatch=args.allow_anchor_mismatch)

    print()
    print_summary(agg)

    if not args.no_save:
        write_scenario(args.output, agg, tier=args.tier)
        size_mb = os.path.getsize(args.output) / 1e6
        print(f"\nSaved: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()