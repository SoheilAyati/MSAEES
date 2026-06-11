#!/usr/bin/env python3
"""
mix_measured_scenarios.py  --  mix real PAC4200 recordings into scenarios
=========================================================================
Turns the single-appliance recordings made with the PAC4200 monitor
(Scripts/PAC4200_reader/recordings/*.h5) into aggregate SCENARIO files with
ground truth, by reusing the existing synthetic-data aggregator unchanged.

Why an adapter is needed
------------------------
`aggregator.py` consumes *per-appliance* files in the Appliance_generator.py
layout:  measurements/P, measurements/Q, measurements/harmonics_I_mag,
measurements/harmonics_I_phase, ground_truth/state, and a JSON
metadata.appliance_metadata (name, instance_id, phase, is_three_phase). It also
requires every appliance in a scenario to share the same length, sample rate and
anchor datetime.

The PAC4200 recorder writes a different layout (P_total/Q_total, per-phase
P_L1..3, harmonics/I_mag_L1.., metadata.appliance_label) and each recording has a
different length and anchor. This script bridges that gap:

  * P  <- P_total,  Q <- Q_total          (single-appliance total power)
  * harmonics_I_mag/_phase <- harmonics/I_mag_L1 / I_phase_L1   (the live phase)
  * ground_truth/state <- "on" where |P| > on_W else "off"
  * appliance_metadata <- {name=<family>, phase="L1", is_three_phase=False, ...}
  * every recording is LOOPED (tiled) to one common duration and given an
    identical timestamp axis + anchor, so the aggregator's alignment check passes.

It then builds several scenarios, each mixing a random subset of appliance
*families* (one variant per family), and calls aggregator.aggregate +
aggregator.write_scenario to produce ground-truth scenario .h5 files that are
byte-compatible with the MS2 pipeline (load as h5_scenario, with /ground_truth).

Notes for these specific recordings
------------------------------------
  * All appliances were recorded on L1 (kept on L1 here, as requested).
  * The per-order current harmonics in the recordings are all zero (the harmonic
    FC-0x14 file numbers were never verified), so mixed THD_I will be ~0. P/Q and
    per-phase power are real.
  * The PV recordings are ~0 W (no generation captured); a scenario containing PV
    just gets a near-zero PV contribution.

Usage
-----
    python mix_measured_scenarios.py \
        --recordings ../PAC4200_reader/recordings \
        --out ./measured_scenarios \
        --n-scenarios 6 --duration 300 --min-app 2 --max-app 4 --seed 0
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py not installed. Run: pip install h5py", file=sys.stderr)
    sys.exit(1)

# Reuse the existing aggregator (same dir by default; AGG_DIR can override the
# location when running this script from elsewhere).
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("AGG_DIR", HERE))
sys.path.insert(0, HERE)
import aggregator as agg  # noqa: E402

N_HARMONICS = 39


# ---------------------------------------------------------------------------
# Appliance "family" grouping (so we never sum two variants of one device)
# ---------------------------------------------------------------------------
_FAMILY_PREFIXES = ("stand_cooler", "table_fan", "table_pv", "laptop")


def family(label: str) -> str:
    """Collapse a recording label to its base appliance family."""
    l = (label or "appliance").lower()
    for fam in _FAMILY_PREFIXES:
        if l.startswith(fam):
            return fam
    return re.split(r"[_\d]", l)[0] or l


# ---------------------------------------------------------------------------
# Read a PAC4200 recording
# ---------------------------------------------------------------------------
def load_pac(path: str) -> dict:
    with h5py.File(path, "r") as h:
        m = h["measurements"]
        P = np.nan_to_num(np.asarray(m["P_total"][:], dtype=np.float64))
        Q = np.nan_to_num(np.asarray(m["Q_total"][:], dtype=np.float64))
        him = m["harmonics/I_mag_L1"][:] if "harmonics/I_mag_L1" in m else None
        hip = m["harmonics/I_phase_L1"][:] if "harmonics/I_phase_L1" in m else None
        sr = float(h["metadata"].attrs.get("sample_rate_hz", 5.0))
        lab = h["metadata"].attrs.get("appliance_label", os.path.basename(path))
        lab = lab.decode() if isinstance(lab, (bytes, bytearray)) else lab
    n = len(P)
    if him is None:
        him = np.zeros((n, N_HARMONICS), np.float32)
    if hip is None:
        hip = np.zeros((n, N_HARMONICS), np.float32)
    return dict(P=P, Q=Q, h_mag=np.asarray(him, np.float32),
                h_phase=np.asarray(hip, np.float32), sr=sr, label=str(lab), n=n,
                family=family(str(lab)))


def _tile_to(arr: np.ndarray, N: int) -> np.ndarray:
    """Loop `arr` along axis 0 until it is exactly length N."""
    if arr.shape[0] == 0:
        return np.zeros((N,) + arr.shape[1:], dtype=arr.dtype)
    reps = int(np.ceil(N / arr.shape[0]))
    return np.concatenate([arr] * reps, axis=0)[:N]


# ---------------------------------------------------------------------------
# Write one PAC recording as an Appliance_generator-format file
# ---------------------------------------------------------------------------
def write_appliance(path: str, rec: dict, N: int, sr: float,
                    anchor: datetime, name: str, instance_id: int,
                    phase: str = "L1", on_W: float = 3.0) -> None:
    P = _tile_to(rec["P"], N).astype(np.float32)
    Q = _tile_to(rec["Q"], N).astype(np.float32)
    hm = _tile_to(rec["h_mag"], N).astype(np.float32)
    hp = _tile_to(rec["h_phase"], N).astype(np.float32)
    state = np.where(np.abs(P) > on_W, b"on", b"off").astype("S32")

    dt_us = int(round(1e6 / sr))
    anchor_us = int(anchor.replace(tzinfo=timezone.utc).timestamp() * 1e6)
    ts = np.arange(N, dtype=np.int64) * dt_us + anchor_us

    meta = {"name": name, "appliance_type": "measured", "instance_id": int(instance_id),
            "phase": phase, "is_three_phase": False, "seed": 0,
            "source_label": rec["label"]}
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamp", data=ts, compression="lzf")
        m = f.create_group("measurements")
        m.create_dataset("P", data=P, compression="lzf")
        m.create_dataset("Q", data=Q, compression="lzf")
        m.create_dataset("harmonics_I_mag", data=hm, compression="lzf")
        m.create_dataset("harmonics_I_phase", data=hp, compression="lzf")
        g = f.create_group("ground_truth")
        g.create_dataset("state", data=state, compression="lzf")
        g.create_dataset("P_contribution", data=P, compression="lzf")
        md = f.create_group("metadata")
        md.attrs["format_version"] = "0.2"
        md.attrs["generator_version"] = "pac_adapter_0.1"
        md.attrs["sample_rate_hz"] = sr
        md.attrs["anchor_datetime"] = anchor.isoformat()
        md.attrs["tier"] = "measured_single"
        md.attrs["appliance_metadata"] = json.dumps(meta)


# ---------------------------------------------------------------------------
# Optional decomposition plot
# ---------------------------------------------------------------------------
def _plot_scenario(path: str, out_png: str) -> None:
    """Save a stacked-area plot of per-appliance ground truth vs the aggregate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # matplotlib is optional
        print(f"  (plot skipped, matplotlib unavailable: {e})")
        return
    with h5py.File(path, "r") as h:
        sr = float(h["metadata"].attrs["sample_rate_hz"])
        Pt = h["measurements/P_total"][:]
        Pc = h["ground_truth/P_contribution"][:]
        names = [x.decode() if isinstance(x, (bytes, bytearray)) else x
                 for x in h["ground_truth/appliance_names"][:]]
    t = np.arange(len(Pt)) / sr
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(t, np.clip(Pc, 0, None).T, labels=names, alpha=0.85)
    neg = np.clip(Pc, None, 0)
    if (neg < 0).any():
        ax.stackplot(t, neg.T, alpha=0.85)
    ax.plot(t, Pt, color="black", lw=1.4, label="P_total")
    ax.set_xlabel("time (s)"); ax.set_ylabel("P (W)")
    ax.set_title(os.path.basename(path) + " - per-appliance ground truth vs aggregate")
    ax.legend(loc="upper right", ncol=3, fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()


# ---------------------------------------------------------------------------
# Build scenarios
# ---------------------------------------------------------------------------
def build(args) -> None:
    files = sorted(glob.glob(os.path.join(args.recordings, "*.h5")))
    if not files:
        sys.exit(f"no .h5 recordings found in {args.recordings}")
    recs = [load_pac(p) for p in files]

    pool: dict = {}
    for r in recs:
        pool.setdefault(r["family"], []).append(r)
    fam_names = sorted(pool)
    print(f"{len(recs)} recordings in {len(fam_names)} families: "
          + ", ".join(f"{f}({len(pool[f])})" for f in fam_names))

    N = int(round(args.duration * args.rate))
    anchor = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)  # shared anchor
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    lo = max(2, min(args.min_app, len(fam_names)))
    hi = min(args.max_app, len(fam_names))
    manifest = []
    for k in range(args.n_scenarios):
        n_app = int(rng.integers(lo, hi + 1)) if hi > lo else lo
        chosen = [str(x) for x in rng.choice(fam_names, size=n_app, replace=False)]
        tmp = tempfile.mkdtemp(prefix=f"scn{k}_")
        try:
            app_files = []
            for i, fam in enumerate(chosen):
                variants = pool[fam]
                rec = variants[int(rng.integers(0, len(variants)))]
                ap = os.path.join(tmp, f"{fam}_{i + 1}.h5")
                write_appliance(ap, rec, N, args.rate, anchor, name=fam, instance_id=i + 1)
                app_files.append(ap)
            a = agg.aggregate(app_files, scenario_seed=int(rng.integers(0, 1_000_000)))
            out = os.path.join(args.out, f"measured_scenario_{k:02d}.h5")
            agg.write_scenario(out, a, tier="measured")
            if args.plot:
                _plot_scenario(out, os.path.join(
                    args.out, f"measured_scenario_{k:02d}_decomposition.png"))
            p_peak = float(np.abs(a["P_total"]).max())
            manifest.append({"scenario": os.path.basename(out), "appliances": chosen,
                             "duration_s": args.duration, "n_samples": N,
                             "P_total_peak_W": round(p_peak, 1)})
            print(f"  [{k:02d}] measured_scenario_{k:02d}.h5  <-  {chosen}"
                  f"   (P_total peak {p_peak:.0f} W)")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote {len(manifest)} scenarios + manifest.json to {args.out}")


def main():
    p = argparse.ArgumentParser(description="Mix real PAC4200 recordings into "
                                            "ground-truth scenario files.")
    p.add_argument("--recordings", default=os.path.join(HERE, "..", "PAC4200_reader",
                                                         "recordings"),
                   help="dir of PAC4200 single-appliance recordings (*.h5)")
    p.add_argument("--out", default=os.path.join(HERE, "measured_scenarios"),
                   help="output dir for the scenario .h5 files")
    p.add_argument("--n-scenarios", type=int, default=6)
    p.add_argument("--duration", type=float, default=300.0,
                   help="common scenario length in seconds (each recording is looped to fit)")
    p.add_argument("--rate", type=float, default=5.0, help="sample rate Hz")
    p.add_argument("--min-app", type=int, default=2, help="min appliances per scenario")
    p.add_argument("--max-app", type=int, default=4, help="max appliances per scenario")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true",
                   help="also save a per-scenario decomposition PNG (needs matplotlib)")
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
