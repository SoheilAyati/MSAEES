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
  * The L1 per-order current harmonics are REAL (file number verified
    2026-07-06) and are summed by the aggregator, so mixed scenarios carry a
    genuine spectrum + THD_I. Recordings made before that date, and in-mix
    taught devices (no isolatable spectrum), contribute zeros.
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

# Label -> family parsing is shared with the MS2 pipeline (single source of
# truth), so 'standing_fan_high_no_rotation' and 'standing_fan_low_rotation'
# collapse to the same family here AND at training/inference time.
sys.path.insert(0, os.path.join(HERE, "..", "MS2_Pipeline"))
from nilm_pipeline import parse_family as family, is_mixed_label, family_color  # noqa: E402

N_HARMONICS = 39
MIN_SAMPLES = 25            # skip recordings shorter than ~5 s @ 5 Hz


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


def _make_schedule(N: int, sr: float, rng) -> np.ndarray:
    """Random ON/OFF usage schedule (bool mask) with realistic block lengths.

    Without this, every appliance is ON for the whole scenario (the looped
    recording), the model never sees per-device OFF periods inside a mix, and
    it learns 'everything is always on' -- exactly the false-alarm failure the
    real mixed recordings exposed. Alternating ON (30-120 s) and OFF (15-90 s)
    blocks give every window a random device subset instead.
    """
    sched = np.zeros(N, dtype=bool)
    pos = 0
    on = bool(rng.uniform() < 0.6)
    while pos < N:
        dur_s = rng.uniform(30, 120) if on else rng.uniform(15, 90)
        L = max(1, int(dur_s * sr))
        if on:
            sched[pos:pos + L] = True
        pos += L
        on = not on
    if not sched.any():                 # guarantee the appliance runs at all
        sched[: max(1, int(45 * sr))] = True
    return sched


# ---------------------------------------------------------------------------
# Write one PAC recording as an Appliance_generator-format file
# ---------------------------------------------------------------------------
def write_appliance(path: str, rec: dict, N: int, sr: float,
                    anchor: datetime, name: str, instance_id: int,
                    phase: str = "L1", on_W: float = 3.0, rng=None,
                    schedule: bool = True) -> None:
    P = _tile_to(rec["P"], N).astype(np.float32)
    Q = _tile_to(rec["Q"], N).astype(np.float32)
    hm = _tile_to(rec["h_mag"], N).astype(np.float32)
    hp = _tile_to(rec["h_phase"], N).astype(np.float32)
    if schedule and rng is not None:
        mask = _make_schedule(N, sr, rng)
        P = P * mask
        Q = Q * mask
        hm = hm * mask[:, None]
        hp = hp * mask[:, None]
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
    """Save a detailed decomposition figure: the stacked aggregate view on top,
    then one ground-truth panel per appliance showing its own P_contribution,
    the ON intervals as shading, and duty-cycle / typical-watts stats. Colors
    come from nilm_pipeline.family_color so every chart in the project (incl.
    the live dashboard) shows a given device family in the same hue."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # matplotlib is optional
        print(f"  (plot skipped, matplotlib unavailable: {e})")
        return
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
    with h5py.File(path, "r") as h:
        sr = float(h["metadata"].attrs["sample_rate_hz"])
        Pt = h["measurements/P_total"][:]
        Pc = h["ground_truth/P_contribution"][:]          # (N, n_app)
        names = [x.decode() if isinstance(x, (bytes, bytearray)) else x
                 for x in h["ground_truth/appliance_names"][:]]
        states = h["ground_truth/state"][:]                # (N, n_app) b"on"/b"off"
    n_app = len(names)
    t = np.arange(len(Pt)) / sr
    colors = [family_color(nm, "light") for nm in names]

    fig, axes = plt.subplots(
        n_app + 1, 1, sharex=True,
        figsize=(12, 3.4 + 1.15 * n_app),
        gridspec_kw={"height_ratios": [2.4] + [1.0] * n_app, "hspace": 0.12})
    axes = np.atleast_1d(axes)

    # -- top: aggregate vs stacked ground truth ------------------------------
    ax = axes[0]
    ax.stackplot(t, np.clip(Pc, 0, None).T, labels=names, colors=colors,
                 alpha=0.85, linewidth=0)
    neg = np.clip(Pc, None, 0)
    if (neg < 0).any():
        ax.stackplot(t, neg.T, colors=colors, alpha=0.85, linewidth=0)
    ax.plot(t, Pt, color=INK, lw=1.5, label="P_total (aggregate)")
    ax.set_ylabel("P (W)", color=INK2)
    ax.set_title(os.path.basename(path) + " - aggregate vs per-appliance ground truth",
                 fontsize=11, color=INK)
    top = max(float(np.abs(Pt).max()),
              float(np.clip(Pc, 0, None).sum(axis=1).max()), 1.0)
    ax.set_ylim(top=top * 1.45)          # headroom so the legend clears the data
    leg = ax.legend(loc="upper right", ncol=min(4, n_app + 1), fontsize=8,
                    frameon=True, framealpha=0.9, edgecolor=GRID)
    leg.get_frame().set_facecolor("#ffffff")

    # -- one ground-truth panel per appliance --------------------------------
    for i, (nm, c) in enumerate(zip(names, colors)):
        axd = axes[i + 1]
        p = Pc[:, i]
        on = states[:, i] == b"on"
        # ON intervals as a background wash: the ground-truth schedule is
        # visible even where the device draws few watts
        if on.any():
            axd.fill_between(t, 0, 1, where=on, transform=axd.get_xaxis_transform(),
                             color=c, alpha=0.10, linewidth=0)
        axd.fill_between(t, 0, p, color=c, alpha=0.45, linewidth=0)
        axd.plot(t, p, color=c, lw=1.3)
        duty = 100.0 * float(on.mean())
        on_w = float(np.median(np.abs(p[on]))) if on.any() else 0.0
        peak = float(np.abs(p).max())
        axd.set_ylabel(nm, rotation=0, ha="right", va="center",
                       fontsize=9, color=INK)
        axd.text(0.995, 0.92, f"on {duty:.0f}% of the time · ~{on_w:.0f} W when on · peak {peak:.0f} W",
                 transform=axd.transAxes, ha="right", va="top",
                 fontsize=7.5, color=INK2)
        ymax = max(peak, 1.0)
        axd.set_ylim(min(0.0, float(p.min()) * 1.1), ymax * 1.35)

    for ax_ in axes:
        ax_.grid(alpha=0.6, color=GRID, linewidth=0.7)
        ax_.tick_params(colors=INK2, labelsize=8)
        for sp in ax_.spines.values():
            sp.set_color(GRID)
    axes[-1].set_xlabel("time (s)", color=INK2)
    axes[-1].set_xlim(t[0], t[-1])
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Build scenarios
# ---------------------------------------------------------------------------
def build(args) -> None:
    files = sorted(glob.glob(os.path.join(args.recordings, "*.h5")))
    if args.exclude:
        import fnmatch
        files = [f for f in files
                 if not fnmatch.fnmatch(os.path.basename(f), args.exclude)]
    if not files:
        sys.exit(f"no .h5 recordings found in {args.recordings}")

    recs = []
    for p in files:
        r = load_pac(p)
        # mixed recordings ('a__b__c' = several devices at once) have no
        # per-device ground truth -> not usable as single-appliance sources;
        # they serve as real test inputs for infer/live instead.
        if is_mixed_label(r["label"]):
            print(f"  skip (mixed recording, keep for testing): {os.path.basename(p)}")
            continue
        if r["n"] < MIN_SAMPLES:
            print(f"  skip (too short, {r['n']} samples): {os.path.basename(p)}")
            continue
        if float(np.abs(r["P"]).max()) < 1.0:
            print(f"  note: '{r['label']}' never draws power (max |P| < 1 W) - "
                  f"kept, but it can only teach 'always off'")
        recs.append(r)
    if not recs:
        sys.exit("no usable single-appliance recordings after filtering")

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

    lo = max(1, min(args.min_app, len(fam_names)))
    hi = min(args.max_app, len(fam_names))

    # ---- plan scenario compositions, then guarantee coverage ---------------
    # Every family must appear in >= min_cover scenarios; otherwise a grouped
    # train/test split can end up with a device ONLY in the held-out set (the
    # model then never learns it) or only in training (its score is untestable).
    plans = []
    for _ in range(args.n_scenarios):
        n_app = int(rng.integers(lo, hi + 1)) if hi > lo else lo
        plans.append([str(x) for x in rng.choice(fam_names, size=n_app, replace=False)])
    min_cover = min(4, args.n_scenarios)
    for fam in fam_names:
        cover = sum(fam in pl for pl in plans)
        while cover < min_cover:
            candidates = [pl for pl in plans if fam not in pl]
            if not candidates:
                break
            pl = min(candidates, key=len)
            if len(pl) < hi:
                pl.append(fam)
            else:                       # swap out the most-covered member
                counts = {f: sum(f in q for q in plans) for f in pl}
                pl[pl.index(max(counts, key=counts.get))] = fam
            cover = sum(fam in pl for pl in plans)

    manifest = []
    for k in range(args.n_scenarios):
        chosen = plans[k]
        tmp = tempfile.mkdtemp(prefix=f"scn{k}_")
        try:
            app_files = []
            for i, fam in enumerate(chosen):
                variants = pool[fam]
                rec = variants[int(rng.integers(0, len(variants)))]
                ap = os.path.join(tmp, f"{fam}_{i + 1}.h5")
                write_appliance(ap, rec, N, args.rate, anchor, name=fam,
                                instance_id=i + 1, rng=rng,
                                schedule=not args.no_schedule)
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
    p.add_argument("--exclude", default=None, metavar="GLOB",
                   help="skip recordings whose filename matches this pattern "
                        "(e.g. 'test_*' to hold test recordings out of training)")
    p.add_argument("--no-schedule", action="store_true",
                   help="keep every appliance ON for the whole scenario instead "
                        "of the random usage schedule")
    p.add_argument("--plot", action="store_true",
                   help="also save a per-scenario decomposition PNG (needs matplotlib)")
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
