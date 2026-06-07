"""
nilm_io.py  --  Milestone 2 I/O layer
=====================================

Loading utilities that hide the difference between the three data sources we
have in Milestone 2, so every downstream stage (event detection, feature
extraction, clustering, classification) sees the same simple objects:

  1. Mixed synthetic scenarios   Synthetic_Data/Mixed/scenario_*.h5
        - full /measurements (per-phase + total P,Q,S,PF,THD,...)
        - /measurements/harmonics/{I,V}_{mag,phase}_L{1,2,3}  (T x 39, orders 2..40)
        - /ground_truth/{appliance_names, state, P_contribution, Q_contribution}
        - /preprocessed/{cleaned, features}   (built in Milestone 1)

  2. Single-appliance synthetic   Synthetic_Data/Single/*.h5
        - /measurements/{P, Q, harmonics_I_mag, harmonics_I_phase}
        - /ground_truth/{state, P_contribution}
        - one appliance, 24 h, clean -> ideal for building a labelled
          per-appliance training set.

  3. Real pre-measured            Pre_Measured/pac4200_*_200ms.csv
        - single phase (L1 only): p_total_w, q_total_var, s_total_va,
          pf_total, thd_i_l1_percent, thd_u_l1_percent, i_l1_a, u_l1_n_v
        - NO per-order harmonics, NO per-phase channels.

The asymmetry in (3) is the load-bearing design fact for synthetic->real
transfer: a model that is supposed to run on real PAC4200 data may only use
the channels that the real data actually contains.  See COMMON_CHANNELS.

Author: MS2 starter code (Ayati / Steffgen NILM project)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError("h5py is required: pip install h5py") from exc


# --------------------------------------------------------------------------
# Harmonic order bookkeeping
# --------------------------------------------------------------------------
# Harmonic arrays store orders 2..40 -> column index = order - 2.
HARMONIC_ORDERS = np.arange(2, 41)            # length 39


def order_index(order: int) -> int:
    """Column index of a given harmonic order inside the (T, 39) arrays."""
    if not 2 <= order <= 40:
        raise ValueError(f"harmonic order {order} outside stored range 2..40")
    return order - 2


# Channels that exist in BOTH synthetic and real data.  Any model intended to
# transfer to the real PAC4200 may only consume features derived from these.
COMMON_CHANNELS = ("P_total", "Q_total", "S_total", "PF_total", "THD_I_L1")


# --------------------------------------------------------------------------
# Container objects
# --------------------------------------------------------------------------
@dataclass
class Scenario:
    """A loaded mixed synthetic scenario (the real NILM target)."""
    timestamp: np.ndarray                       # int64 us since epoch
    sample_rate_hz: float
    meas: dict                                  # name -> 1-D float array
    harm: dict                                  # name -> (T, 39) float array
    appliance_names: list                       # length N_app
    states: np.ndarray                          # (T, N_app) <U.. strings
    P_contribution: np.ndarray                  # (T, N_app)
    Q_contribution: np.ndarray                  # (T, N_app)
    metadata: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return self.timestamp.shape[0]

    @property
    def hours(self) -> np.ndarray:
        """Hour-of-day axis (0..24) for plotting."""
        t0 = self.timestamp[0]
        return (self.timestamp - t0) / 1e6 / 3600.0


@dataclass
class SingleAppliance:
    """A loaded single-appliance synthetic file."""
    name: str
    timestamp: np.ndarray
    sample_rate_hz: float
    P: np.ndarray
    Q: np.ndarray
    harm_I_mag: np.ndarray                       # (T, 39)
    harm_I_phase: np.ndarray                     # (T, 39)
    state: np.ndarray                            # (T,) strings
    metadata: dict = field(default_factory=dict)


@dataclass
class RealRun:
    """A loaded real PAC4200 measurement run (single device, single phase)."""
    device_name: str
    run_id: str
    t_seconds: np.ndarray                        # seconds from start of run
    sample_rate_hz: float
    P: np.ndarray                                # p_total_w
    Q: np.ndarray                                # q_total_var
    S: np.ndarray                                # s_total_va
    PF: np.ndarray                               # pf_total
    THD_I: np.ndarray                            # thd_i_l1_percent
    THD_V: np.ndarray                            # thd_u_l1_percent
    I: np.ndarray                                # i_l1_a
    V: np.ndarray                                # u_l1_n_v


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
_SCALAR_MEAS = [
    "V_L1", "V_L2", "V_L3", "I_L1", "I_L2", "I_L3", "I_N",
    "P_L1", "P_L2", "P_L3", "P_total",
    "Q_L1", "Q_L2", "Q_L3", "Q_total",
    "S_L1", "S_L2", "S_L3", "S_total",
    "PF_L1", "PF_L2", "PF_L3", "PF_total",
    "cosphi_L1", "cosphi_L2", "cosphi_L3", "cosphi_total",
    "THD_V_L1", "THD_V_L2", "THD_V_L3",
    "THD_I_L1", "THD_I_L2", "THD_I_L3", "freq",
]


def _decode(arr) -> list:
    return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]


def load_scenario(path: str, prefer_cleaned: bool = True) -> Scenario:
    """Load a mixed synthetic scenario file.

    If ``prefer_cleaned`` and a /preprocessed/cleaned group exists, the scalar
    measurement channels are taken from there (this is what Milestone 1 built
    and what real-data preprocessing will also produce).
    """
    with h5py.File(path, "r") as f:
        ts = f["timestamp"][:]
        meta = {k: _attr(v) for k, v in f["metadata"].attrs.items()}
        sr = float(meta.get("sample_rate_hz", 5.0))

        cleaned = f.get("preprocessed/cleaned") if prefer_cleaned else None
        meas = {}
        for ch in _SCALAR_MEAS:
            if cleaned is not None and ch in cleaned:
                meas[ch] = cleaned[ch][:].astype(np.float32)
            elif ch in f["measurements"]:
                meas[ch] = f["measurements"][ch][:].astype(np.float32)

        harm = {}
        hg = f.get("measurements/harmonics")
        if hg is not None:
            for key in hg.keys():
                harm[key] = hg[key][:].astype(np.float32)

        gt = f["ground_truth"]
        names = _decode(gt["appliance_names"][:])
        states = gt["state"][:].astype("U32")
        Pc = gt["P_contribution"][:].astype(np.float32)
        Qc = gt["Q_contribution"][:].astype(np.float32)

    return Scenario(ts, sr, meas, harm, names, states, Pc, Qc, meta)


def load_single(path: str) -> SingleAppliance:
    """Load a single-appliance synthetic file."""
    with h5py.File(path, "r") as f:
        ts = f["timestamp"][:]
        meta = {k: _attr(v) for k, v in f["metadata"].attrs.items()}
        sr = float(meta.get("sample_rate_hz", 5.0))
        name = meta.get("appliance_metadata", "{}")
        try:
            name = json.loads(name).get("name", "unknown")
        except Exception:
            name = "unknown"
        m = f["measurements"]
        P = m["P"][:].astype(np.float32)
        Q = m["Q"][:].astype(np.float32)
        him = m["harmonics_I_mag"][:].astype(np.float32)
        hip = m["harmonics_I_phase"][:].astype(np.float32)
        st = f["ground_truth"]["state"][:].astype("U32")
    return SingleAppliance(name, ts, sr, P, Q, him, hip, st, meta)


def load_real_csv(path: str) -> RealRun:
    """Load a real PAC4200 CSV (semicolon-separated, German decimal already dot)."""
    import pandas as pd

    df = pd.read_csv(path, sep=";")
    t = pd.to_datetime(df["timestamp_iso"], utc=True, format="ISO8601")
    t_s = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    dt = np.median(np.diff(t_s)) if len(t_s) > 1 else 0.2
    sr = 1.0 / dt if dt > 0 else 5.0

    def col(name):
        return df[name].to_numpy(dtype=np.float32)

    return RealRun(
        device_name=str(df["device_name"].iloc[0]),
        run_id=str(df["run_id"].iloc[0]),
        t_seconds=t_s.astype(np.float32),
        sample_rate_hz=float(sr),
        P=col("p_total_w"),
        Q=col("q_total_var"),
        S=col("s_total_va"),
        PF=col("pf_total"),
        THD_I=col("thd_i_l1_percent"),
        THD_V=col("thd_u_l1_percent"),
        I=col("i_l1_a"),
        V=col("u_l1_n_v"),
    )


# --------------------------------------------------------------------------
# Ground-truth event derivation (synthetic only)
# --------------------------------------------------------------------------
def ground_truth_events(scn: Scenario, min_dP: float = 5.0):
    """Derive a list of ground-truth switching events from per-appliance states.

    An event is any sample at which an appliance changes state *and* the
    associated change in its power contribution exceeds ``min_dP`` watts (so we
    do not count negligible electronic flutter).

    Returns
    -------
    list of dict: {idx, appliance, from_state, to_state, dP}
    """
    events = []
    for a, name in enumerate(scn.appliance_names):
        st = scn.states[:, a]
        change = np.where(st[1:] != st[:-1])[0] + 1
        for i in change:
            dP = float(scn.P_contribution[i, a] - scn.P_contribution[i - 1, a])
            if abs(dP) >= min_dP:
                events.append(dict(idx=int(i), appliance=name,
                                   from_state=st[i - 1], to_state=st[i], dP=dP))
    events.sort(key=lambda e: e["idx"])
    return events


def _attr(v):
    if isinstance(v, (bytes, bytearray)):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    return v


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "Synthetic_Data/Mixed/scenario_normal.h5"
    scn = load_scenario(p)
    print(f"Loaded {p}")
    print(f"  {scn.n_samples} samples @ {scn.sample_rate_hz} Hz, {len(scn.appliance_names)} appliances")
    print(f"  P_total: {scn.meas['P_total'].min():.1f} .. {scn.meas['P_total'].max():.1f} W")
    ev = ground_truth_events(scn)
    print(f"  ground-truth events: {len(ev)}")
