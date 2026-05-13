#!/usr/bin/env python3
"""
NILM HDF5 Plotter
=================
Plots one generated/aggregated NILM .h5 file in the style of the example figure,
but without the RMS-current subplot.

Default usage:
    1) Set H5_PATH and PLOT_DIR below.
    2) Run:
        python plot_nilm_h5.py

CLI override:
    python plot_nilm_h5.py --h5 ./Mixed/scenario_normal.h5 --outdir ./plots
    python plot_nilm_h5.py --h5 ./Mixed/scenario_normal.h5 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# User-editable defaults
# ---------------------------------------------------------------------------

H5_PATH = Path(r"./Synthetic_Data/Mixed/scenario_normal.h5")
PLOT_DIR = Path(r".\Synthetic_Data\Mixed\plots")
PLOT_FILENAME = "nilm_h5_overview.png"

# 432000 samples/day at 5 Hz is fine, but plotting every sample can be slow.
# None = automatic downsampling to roughly MAX_PLOT_POINTS.
# for efficient sampling use None = None. If you desire max plot points None = 1
PLOT_STRIDE: int | None = 1
MAX_PLOT_POINTS = 80_000

FIGSIZE = (16, 11)
DPI = 150


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _read_dataset(group: h5py.Group, name: str, default=None):
    return group[name][:] if name in group else default


def _decode_names(arr: np.ndarray) -> list[str]:
    names: list[str] = []
    for x in arr:
        if isinstance(x, bytes):
            names.append(x.decode("utf-8", errors="replace"))
        else:
            names.append(str(x))
    return names


def _time_hours(timestamp_us: np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamp_us)
    if ts.size == 0:
        return ts.astype(float)
    return (ts - ts[0]) / 1e6 / 3600.0


def _choose_stride(n_samples: int, requested_stride: int | None) -> int:
    if requested_stride is not None and requested_stride > 0:
        return int(requested_stride)
    return max(1, int(np.ceil(n_samples / MAX_PLOT_POINTS)))


def load_h5_for_plot(path: Path, stride: int | None = None) -> dict:
    """Load either an aggregated scenario HDF5 or a single-appliance HDF5."""
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(path, "r") as f:
        if "timestamp" not in f:
            raise KeyError("Missing required dataset: /timestamp")
        n = len(f["timestamp"])
        step = _choose_stride(n, stride)
        sl = slice(None, None, step)

        timestamp = f["timestamp"][sl]
        t_h = _time_hours(timestamp)

        if "measurements" not in f:
            raise KeyError("Missing required group: /measurements")
        m = f["measurements"]

        # Aggregated scenario layout from aggregator.py
        if "P_total" in m:
            P_total = m["P_total"][sl]
            Q_total = _read_dataset(m, "Q_total", np.zeros_like(P_total))[sl]
            P_phase = {
                ph: _read_dataset(m, f"P_{ph}", np.full_like(P_total, np.nan))[sl]
                for ph in ("L1", "L2", "L3")
            }
        # Single-appliance layout from Appliance_generator.py
        elif "P" in m:
            P_total = m["P"][sl]
            Q_total = _read_dataset(m, "Q", np.zeros_like(P_total))[sl]
            P_phase = {"L1": P_total, "L2": np.zeros_like(P_total), "L3": np.zeros_like(P_total)}
        else:
            raise KeyError("Could not find /measurements/P_total or /measurements/P")

        appliance_names: list[str] = []
        P_contrib = None
        states = None
        if "ground_truth" in f:
            gt = f["ground_truth"]
            if "appliance_names" in gt:
                appliance_names = _decode_names(gt["appliance_names"][:])
            if "P_contribution" in gt:
                raw = gt["P_contribution"]
                P_contrib = raw[sl]
                if P_contrib.ndim == 1:
                    P_contrib = P_contrib[:, None]
            if "state" in gt:
                states = gt["state"][sl]
                if states.ndim == 1:
                    states = states[:, None]

        if P_contrib is not None and not appliance_names:
            appliance_names = [f"appliance_{i+1}" for i in range(P_contrib.shape[1])]

        meta = {}
        if "metadata" in f:
            meta = {k: f["metadata"].attrs[k] for k in f["metadata"].attrs.keys()}

    return {
        "path": path,
        "stride": step,
        "t_h": t_h,
        "P_total": P_total,
        "Q_total": Q_total,
        "P_phase": P_phase,
        "P_contrib": P_contrib,
        "appliance_names": appliance_names,
        "states": states,
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def signed_stackplot(ax, x: np.ndarray, y: np.ndarray, labels: list[str]) -> None:
    """Stack positive and negative appliance contributions separately.

    This avoids misleading fills when PV/generator traces are negative.
    """
    if y is None or y.size == 0:
        ax.text(0.5, 0.5, "No ground_truth/P_contribution found", ha="center", va="center",
                transform=ax.transAxes)
        return

    pos_base = np.zeros(y.shape[0], dtype=float)
    neg_base = np.zeros(y.shape[0], dtype=float)

    for i in range(y.shape[1]):
        yi = y[:, i].astype(float)
        label = labels[i] if i < len(labels) else f"appliance_{i+1}"

        yi_pos = np.where(yi > 0, yi, 0.0)
        yi_neg = np.where(yi < 0, yi, 0.0)

        if np.any(yi_pos):
            ax.fill_between(x, pos_base, pos_base + yi_pos, step="pre", alpha=0.85, label=label)
            pos_base += yi_pos
        if np.any(yi_neg):
            # Do not duplicate the legend entry if the appliance has positive and negative sections.
            neg_label = label if not np.any(yi_pos) else "_nolegend_"
            ax.fill_between(x, neg_base, neg_base + yi_neg, step="pre", alpha=0.85, label=neg_label)
            neg_base += yi_neg


def plot_overview(data: dict, out_path: Path | None = None, show: bool = False) -> Path | None:
    t = data["t_h"]
    P_total = data["P_total"]
    Q_total = data["Q_total"]
    P_phase = data["P_phase"]
    P_contrib = data["P_contrib"]
    appliance_names = data["appliance_names"]

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0, 1.35]})

    # 1) Aggregate at PCC: P_total and Q_total with twin y-axis
    ax = axes[0]
    ax_q = ax.twinx()
    p_line, = ax.plot(t, P_total, linewidth=0.7, label="P_total")
    q_line, = ax_q.plot(t, Q_total, linewidth=0.7, label="Q_total", color="tab:orange", alpha=0.85)
    ax.axhline(0, linewidth=0.7, color="black", alpha=0.5)
    ax.set_title("Aggregate at PCC")
    ax.set_ylabel("P_total (W)", color=p_line.get_color())
    ax_q.set_ylabel("Q_total (var)", color=q_line.get_color())
    ax.grid(True, alpha=0.3)

    # 2) Per-phase active-power imbalance
    ax = axes[1]
    for ph in ("L1", "L2", "L3"):
        ax.plot(t, P_phase[ph], linewidth=0.65, label=f"P_{ph}")
    ax.axhline(0, linewidth=0.7, color="black", alpha=0.5)
    ax.set_title("Per-phase imbalance")
    ax.set_ylabel("P per phase (W)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3) Ground-truth appliance contributions + aggregate P_total overlay
    ax = axes[2]
    signed_stackplot(ax, t, P_contrib, appliance_names)
    ax.plot(t, P_total, color="black", linewidth=0.8, label="P_total")
    ax.axhline(0, linewidth=0.7, color="black", alpha=0.5)
    ax.set_title("Ground truth: per-appliance breakdown")
    ax.set_ylabel("P contribution (W)")
    ax.set_xlabel("hour of day")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=5, fontsize=7, framealpha=0.9)

    # Common x-limits and a little status text
    if t.size:
        axes[-1].set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
    fig.suptitle(data["path"].name + (f"  |  plotted every {data['stride']} sample(s)" if data["stride"] > 1 else ""),
                 y=0.995, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    saved = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        saved = out_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a NILM scenario/single-appliance HDF5 file.")
    parser.add_argument("--h5", type=Path, default=H5_PATH, help="Path to .h5 file")
    parser.add_argument("--outdir", type=Path, default=PLOT_DIR, help="Directory for plot output")
    parser.add_argument("--outfile", default=PLOT_FILENAME, help="Output image filename")
    parser.add_argument("--stride", type=int, default=PLOT_STRIDE, help="Plot every n-th sample")
    parser.add_argument("--show", action="store_true", help="Show plot window instead of only saving")
    parser.add_argument("--no-save", action="store_true", help="Do not save PNG")
    args = parser.parse_args()

    data = load_h5_for_plot(args.h5, stride=args.stride)
    out_path = None if args.no_save else args.outdir / args.outfile
    saved = plot_overview(data, out_path=out_path, show=args.show)

    print(f"Loaded: {args.h5}")
    print(f"Samples plotted: {len(data['t_h'])} (stride={data['stride']})")
    if saved is not None:
        print(f"Saved plot: {saved}")


if __name__ == "__main__":
    main()
