#!/usr/bin/env python3
"""
NILM Project - Preprocessing Pipeline
======================================
Universal preprocessor for scenario HDF5 files. Operates identically on:
  - Synthetic scenarios (output of scenario_aggregator.py)
  - Real scenarios (output of pac4200_reader.py - Milestone 2)

Reads `/measurements`, produces `/preprocessed` with cleaned channels
and derived features. Does NOT modify `/measurements` or `/ground_truth`;
they remain as the audit trail.

The script makes no assumption about the input source - it handles whatever
data quality issues are present (gaps, NaN, outliers) and produces a
consistent feature set regardless.

Usage
-----
    # Preprocess a single scenario
    python preprocessor.py --input scenario_normal.h5

    # Preprocess all scenarios in a directory
    python preprocessor.py --input-dir ./scenarios

    # Enable smoothing (recommended for real PAC4200 data, not for synthetic)
    python preprocessor.py --input real_pac4200.h5 --smooth-window 5

    # Strict mode: error out instead of imputing
    python preprocessor.py --input scenario_001.h5 --strict
"""

import argparse
import glob
import json
import os
import sys
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREPROCESSOR_VERSION = "0.1.0"

# Physical bounds for outlier detection.
# (lo, hi) - values outside this range are clipped (or flagged) by the
# outlier step. Bounds are deliberately generous so legitimate extremes
# (EV fast charging at 22 kW, PV generation, synchronous machine in
# generator mode) are not flagged.
PHYSICAL_BOUNDS_1D: Dict[str, Tuple[float, float]] = {
    "V_L1": (180.0, 270.0), "V_L2": (180.0, 270.0), "V_L3": (180.0, 270.0),
    "I_L1": (0.0, 200.0),   "I_L2": (0.0, 200.0),   "I_L3": (0.0, 200.0),
    "I_N":  (0.0, 200.0),
    "P_L1": (-30000.0, 30000.0),
    "P_L2": (-30000.0, 30000.0),
    "P_L3": (-30000.0, 30000.0),
    "P_total": (-50000.0, 50000.0),
    "Q_L1": (-15000.0, 15000.0),
    "Q_L2": (-15000.0, 15000.0),
    "Q_L3": (-15000.0, 15000.0),
    "Q_total": (-30000.0, 30000.0),
    "S_L1": (0.0, 50000.0),  "S_L2": (0.0, 50000.0),
    "S_L3": (0.0, 50000.0),  "S_total": (0.0, 100000.0),
    "PF_L1": (-1.0, 1.0), "PF_L2": (-1.0, 1.0),
    "PF_L3": (-1.0, 1.0), "PF_total": (-1.0, 1.0),
    "cosphi_L1": (-1.0, 1.0), "cosphi_L2": (-1.0, 1.0),
    "cosphi_L3": (-1.0, 1.0), "cosphi_total": (-1.0, 1.0),
    "THD_V_L1": (0.0, 50.0), "THD_V_L2": (0.0, 50.0), "THD_V_L3": (0.0, 50.0),
    "THD_I_L1": (0.0, 200.0), "THD_I_L2": (0.0, 200.0), "THD_I_L3": (0.0, 200.0),
    "freq": (47.0, 53.0),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessingConfig:
    """Knobs for the preprocessing pipeline. Defaults are tuned for synthetic
    data; for real PAC4200 data you'll likely want smooth_window >= 3."""
    max_gap_to_impute: int = 5           # samples; longer gaps left as NaN
    outlier_mode: str = "clip"           # "clip" or "flag"
    smooth_window: int = 1               # rolling mean window in samples (1 = off)
    smooth_channels: Tuple[str, ...] = ("P_total", "Q_total")
    feature_short_window_s: float = 5.0  # short rolling window
    feature_long_window_s: float = 30.0  # long rolling window
    compute_event_features: bool = True
    compute_imbalance_features: bool = True
    compute_distortion_features: bool = True
    strict: bool = False                 # if True, raise on validation errors


@dataclass
class PreprocessingReport:
    """What happened during preprocessing. Logged to /preprocessed/report."""
    n_samples: int = 0
    sample_rate_hz: float = 0.0
    channels_processed: List[str] = field(default_factory=list)
    nan_counts: Dict[str, int] = field(default_factory=dict)
    inf_counts: Dict[str, int] = field(default_factory=dict)
    outliers_clipped: Dict[str, int] = field(default_factory=dict)
    gaps_imputed: int = 0
    gaps_left_as_nan: int = 0
    features_built: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "preprocessor_version": PREPROCESSOR_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_samples": self.n_samples,
            "sample_rate_hz": self.sample_rate_hz,
            "channels_processed": self.channels_processed,
            "nan_counts": self.nan_counts,
            "inf_counts": self.inf_counts,
            "outliers_clipped": self.outliers_clipped,
            "gaps_imputed": self.gaps_imputed,
            "gaps_left_as_nan": self.gaps_left_as_nan,
            "features_built": self.features_built,
            "elapsed_seconds": self.elapsed_seconds,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_scenario(f: h5py.File, config: PreprocessingConfig,
                      report: PreprocessingReport):
    """Check the file has what we need. Populates report.warnings."""
    if "measurements" not in f:
        raise KeyError("Scenario file is missing /measurements group.")
    if "timestamp" not in f:
        raise KeyError("Scenario file is missing /timestamp dataset.")

    ts = f["timestamp"][:]
    if len(ts) < 2:
        raise ValueError("Need at least 2 samples to preprocess.")

    # Monotonicity
    dt = np.diff(ts)
    if (dt <= 0).any():
        msg = (f"Timestamps are not strictly increasing "
               f"({(dt <= 0).sum()} violations).")
        if config.strict:
            raise ValueError(msg)
        report.warnings.append(msg)

    # Sample rate from metadata if present, otherwise estimate
    if "metadata" in f and "sample_rate_hz" in f["metadata"].attrs:
        report.sample_rate_hz = float(f["metadata"].attrs["sample_rate_hz"])
    else:
        # Estimate from median dt (dt is in microseconds)
        median_dt_us = float(np.median(dt[dt > 0]))
        report.sample_rate_hz = 1e6 / median_dt_us
        report.warnings.append(
            f"No sample_rate_hz in metadata; estimated {report.sample_rate_hz:.2f} Hz "
            f"from median timestamp interval.")

    report.n_samples = len(ts)


def list_channels_to_clean(f: h5py.File) -> List[str]:
    """Channels present in /measurements that we'll process."""
    out = []
    m = f["measurements"]
    for name in m:
        if isinstance(m[name], h5py.Dataset) and m[name].ndim == 1:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# NaN / Inf handling
# ---------------------------------------------------------------------------

def count_nan_inf(arr: np.ndarray) -> Tuple[int, int]:
    nan = int(np.isnan(arr).sum())
    inf = int(np.isinf(arr).sum())
    return nan, inf


def impute_short_gaps(arr: np.ndarray, max_gap: int
                     ) -> Tuple[np.ndarray, int, int]:
    """Linear interpolation over runs of NaN with length <= max_gap.
    Returns (imputed_array, n_gaps_imputed, n_gaps_left)."""
    out = arr.astype(np.float64, copy=True)  # promote for safe interp
    nan_mask = np.isnan(out)
    if not nan_mask.any():
        return out.astype(arr.dtype, copy=False), 0, 0

    # Find runs of NaN
    diff = np.diff(nan_mask.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    imputed = 0
    left = 0
    for s, e in zip(starts, ends):
        run_len = e - s
        if run_len <= max_gap and s > 0 and e < len(out):
            # Linear interp between out[s-1] and out[e]
            left_val = out[s - 1]
            right_val = out[e]
            if np.isnan(left_val) or np.isnan(right_val):
                left += 1
                continue
            out[s:e] = np.linspace(left_val, right_val, run_len + 2)[1:-1]
            imputed += 1
        else:
            left += 1
    return out.astype(arr.dtype, copy=False), imputed, left


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

def handle_outliers(arr: np.ndarray, bounds: Tuple[float, float],
                   mode: str) -> Tuple[np.ndarray, int]:
    """Clip or flag values outside bounds. Returns (processed_array, n_outliers)."""
    lo, hi = bounds
    mask = (arr < lo) | (arr > hi)
    mask &= np.isfinite(arr)  # don't count NaN/inf as outliers here
    n_outliers = int(mask.sum())
    if n_outliers == 0:
        return arr, 0
    if mode == "clip":
        return np.clip(arr, lo, hi).astype(arr.dtype, copy=False), n_outliers
    elif mode == "flag":
        out = arr.copy()
        out[mask] = np.nan
        return out, n_outliers
    else:
        raise ValueError(f"Unknown outlier_mode '{mode}'")


# ---------------------------------------------------------------------------
# Optional smoothing
# ---------------------------------------------------------------------------

def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean. Edges padded with the original signal."""
    if window <= 1:
        return arr
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(arr.astype(np.float64), kernel, mode="same")
    # Avoid edge artifacts from convolution: keep original at edges
    half = window // 2
    smooth[:half] = arr[:half]
    smooth[-half:] = arr[-half:]
    return smooth.astype(arr.dtype, copy=False)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def rolling_stat(arr: np.ndarray, window_samples: int, stat: str) -> np.ndarray:
    """Centered rolling mean or std over `window_samples` samples."""
    if window_samples < 2:
        return arr.copy() if stat == "mean" else np.zeros_like(arr)
    w = window_samples
    if stat == "mean":
        kernel = np.ones(w, dtype=np.float64) / w
        out = np.convolve(arr.astype(np.float64), kernel, mode="same")
        return out.astype(np.float32, copy=False)
    elif stat == "std":
        # E[x^2] - E[x]^2
        mean_x = rolling_stat(arr, w, "mean").astype(np.float64)
        mean_x2 = rolling_stat(arr ** 2, w, "mean").astype(np.float64)
        var = np.clip(mean_x2 - mean_x ** 2, 0, None)
        return np.sqrt(var).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown stat '{stat}'")


def build_features(cleaned: Dict[str, np.ndarray], sample_rate_hz: float,
                   config: PreprocessingConfig
                   ) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Compute derived features from cleaned channels."""
    features: Dict[str, np.ndarray] = {}
    names: List[str] = []

    short_w = max(1, int(config.feature_short_window_s * sample_rate_hz))
    long_w = max(1, int(config.feature_long_window_s * sample_rate_hz))

    # ----- Event detection features -----
    if config.compute_event_features and "P_total" in cleaned:
        P = cleaned["P_total"]
        dP = np.gradient(P) * sample_rate_hz  # W per second
        features["dP_total"] = dP.astype(np.float32)
        features["abs_dP_total"] = np.abs(dP).astype(np.float32)
        features["P_total_rolling_std_short"] = rolling_stat(P, short_w, "std")
        features["P_total_rolling_mean_short"] = rolling_stat(P, short_w, "mean")
        features["P_total_rolling_mean_long"] = rolling_stat(P, long_w, "mean")
        names += ["dP_total", "abs_dP_total",
                  "P_total_rolling_std_short",
                  "P_total_rolling_mean_short",
                  "P_total_rolling_mean_long"]

    # ----- Per-phase rolling means -----
    for ph in ("L1", "L2", "L3"):
        k = f"P_{ph}"
        if k in cleaned:
            features[f"{k}_rolling_mean_short"] = rolling_stat(cleaned[k], short_w, "mean")
            names.append(f"{k}_rolling_mean_short")

    # ----- Distortion features -----
    if config.compute_distortion_features:
        thd_i = [cleaned[k] for k in ("THD_I_L1", "THD_I_L2", "THD_I_L3") if k in cleaned]
        thd_v = [cleaned[k] for k in ("THD_V_L1", "THD_V_L2", "THD_V_L3") if k in cleaned]
        if thd_i:
            mean_thd_i = np.mean(np.stack(thd_i, axis=0), axis=0)
            features["mean_THD_I"] = rolling_stat(mean_thd_i, short_w, "mean").astype(np.float32)
            names.append("mean_THD_I")
        if thd_v:
            mean_thd_v = np.mean(np.stack(thd_v, axis=0), axis=0)
            features["mean_THD_V"] = rolling_stat(mean_thd_v, short_w, "mean").astype(np.float32)
            names.append("mean_THD_V")

    # ----- Phase-imbalance features -----
    if config.compute_imbalance_features:
        per_phase_P = [cleaned[k] for k in ("P_L1", "P_L2", "P_L3") if k in cleaned]
        if len(per_phase_P) == 3:
            P_stack = np.stack(per_phase_P, axis=0)
            P_max = P_stack.max(axis=0)
            P_min = P_stack.min(axis=0)
            P_mean = P_stack.mean(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                imbalance = np.where(np.abs(P_mean) > 1.0,
                                     (P_max - P_min) / np.abs(P_mean),
                                     0.0)
            features["P_phase_imbalance"] = imbalance.astype(np.float32)
            names.append("P_phase_imbalance")

        per_phase_I = [cleaned[k] for k in ("I_L1", "I_L2", "I_L3") if k in cleaned]
        if len(per_phase_I) == 3 and "I_N" in cleaned:
            I_stack = np.stack(per_phase_I, axis=0)
            I_mean_phase = I_stack.mean(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                neutral_ratio = np.where(I_mean_phase > 0.01,
                                         cleaned["I_N"] / I_mean_phase, 0.0)
            features["I_neutral_to_phase_ratio"] = neutral_ratio.astype(np.float32)
            names.append("I_neutral_to_phase_ratio")

    return features, names


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def read_measurements(f: h5py.File, channels: List[str]) -> Dict[str, np.ndarray]:
    """Read named 1D channels from /measurements into a dict."""
    return {c: f[f"measurements/{c}"][:] for c in channels}


def write_preprocessed(f: h5py.File, cleaned: Dict[str, np.ndarray],
                       features: Dict[str, np.ndarray],
                       report: PreprocessingReport):
    """Write cleaned channels and features to /preprocessed."""
    if "preprocessed" in f:
        del f["preprocessed"]
    g = f.create_group("preprocessed")

    cleaned_g = g.create_group("cleaned")
    for name, arr in cleaned.items():
        cleaned_g.create_dataset(name, data=arr, compression="lzf")

    if features:
        feat_g = g.create_group("features")
        for name, arr in features.items():
            feat_g.create_dataset(name, data=arr, compression="lzf")

    g.attrs["report"] = json.dumps(report.to_dict())
    g.attrs["preprocessor_version"] = PREPROCESSOR_VERSION


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess_file(path: str, config: PreprocessingConfig) -> PreprocessingReport:
    """Preprocess one scenario file. Modifies it in-place."""
    report = PreprocessingReport()
    t0 = time.time()

    with h5py.File(path, "r+") as f:
        validate_scenario(f, config, report)
        channels = list_channels_to_clean(f)
        raw = read_measurements(f, channels)

        cleaned: Dict[str, np.ndarray] = {}
        for ch_name, arr in raw.items():
            # NaN / inf counts
            nan, inf = count_nan_inf(arr)
            if nan: report.nan_counts[ch_name] = nan
            if inf: report.inf_counts[ch_name] = inf

            # inf → NaN so gap imputation can handle uniformly
            work = arr.astype(np.float32, copy=True)
            work[~np.isfinite(work)] = np.nan

            # Impute short gaps
            work, imp, left = impute_short_gaps(work, config.max_gap_to_impute)
            report.gaps_imputed += imp
            report.gaps_left_as_nan += left

            # Outlier handling within physical bounds
            if ch_name in PHYSICAL_BOUNDS_1D:
                work_finite = np.where(np.isnan(work), 0.0, work).astype(np.float32)
                work_finite, n_out = handle_outliers(
                    work_finite, PHYSICAL_BOUNDS_1D[ch_name], config.outlier_mode)
                # restore NaN positions (we don't want clipping to mask real gaps)
                work_finite[np.isnan(work)] = np.nan
                work = work_finite
                if n_out:
                    report.outliers_clipped[ch_name] = n_out

            # Optional smoothing for selected channels
            if config.smooth_window > 1 and ch_name in config.smooth_channels:
                # Skip if all NaN to avoid issues
                if not np.isnan(work).all():
                    finite = np.where(np.isnan(work), 0.0, work)
                    smoothed = rolling_mean(finite, config.smooth_window)
                    smoothed[np.isnan(work)] = np.nan
                    work = smoothed

            cleaned[ch_name] = work.astype(np.float32, copy=False)
            report.channels_processed.append(ch_name)

        # Feature engineering
        features, feat_names = build_features(cleaned, report.sample_rate_hz, config)
        report.features_built = feat_names

        report.elapsed_seconds = time.time() - t0
        write_preprocessed(f, cleaned, features, report)

    return report


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(path: str, report: PreprocessingReport):
    print(f"\n  ── Preprocessed: {path} ──")
    print(f"  Samples: {report.n_samples} @ {report.sample_rate_hz} Hz "
          f"({report.n_samples / report.sample_rate_hz / 3600:.2f} h)")
    print(f"  Channels processed: {len(report.channels_processed)}")
    print(f"  Features built:     {len(report.features_built)}")
    if report.nan_counts:
        n_total = sum(report.nan_counts.values())
        print(f"  NaN values found:   {n_total} across {len(report.nan_counts)} channel(s)")
        for ch, n in list(report.nan_counts.items())[:5]:
            print(f"    - {ch}: {n}")
    if report.inf_counts:
        print(f"  Inf values found:   "
              f"{sum(report.inf_counts.values())} across {len(report.inf_counts)} channel(s)")
    if report.outliers_clipped:
        n_total = sum(report.outliers_clipped.values())
        print(f"  Outliers clipped:   {n_total} across "
              f"{len(report.outliers_clipped)} channel(s)")
        for ch, n in list(report.outliers_clipped.items())[:5]:
            print(f"    - {ch}: {n}")
    if report.gaps_imputed or report.gaps_left_as_nan:
        print(f"  Gaps: {report.gaps_imputed} imputed, "
              f"{report.gaps_left_as_nan} too long, left as NaN")
    if report.warnings:
        print(f"  Warnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"    ! {w}")
    print(f"  Elapsed: {report.elapsed_seconds:.2f} s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Universal preprocessor for NILM scenario HDF5 files. "
                    "Operates identically on synthetic and real PAC4200 scenarios. "
                    "Writes results to /preprocessed in each file.")
    p.add_argument("--input", help="single scenario .h5 file to preprocess")
    p.add_argument("--input-dir", help="directory containing scenario .h5 files")
    p.add_argument("--pattern", default="*.h5",
                   help="glob pattern within --input-dir (default *.h5)")
    p.add_argument("--max-gap-impute", type=int, default=5,
                   help="impute NaN runs up to this many samples; longer left as NaN")
    p.add_argument("--outlier-mode", choices=["clip", "flag"], default="clip",
                   help="clip to physical bounds or flag (NaN) values out of bounds")
    p.add_argument("--smooth-window", type=int, default=1,
                   help="rolling-mean window in samples for P_total/Q_total (1 = no smoothing)")
    p.add_argument("--short-window-s", type=float, default=5.0,
                   help="short rolling window for features, seconds")
    p.add_argument("--long-window-s", type=float, default=30.0,
                   help="long rolling window for features, seconds")
    p.add_argument("--strict", action="store_true",
                   help="error out instead of warning on validation issues")
    args = p.parse_args()

    if not args.input and not args.input_dir:
        p.error("specify either --input or --input-dir")

    config = PreprocessingConfig(
        max_gap_to_impute=args.max_gap_impute,
        outlier_mode=args.outlier_mode,
        smooth_window=args.smooth_window,
        feature_short_window_s=args.short_window_s,
        feature_long_window_s=args.long_window_s,
        strict=args.strict,
    )

    if args.input:
        files = [args.input]
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if not files:
            p.error(f"no files matched {args.pattern} in {args.input_dir}")

    print(f"Preprocessing {len(files)} scenario file(s)")
    print(f"  max_gap_to_impute: {config.max_gap_to_impute}")
    print(f"  outlier_mode:      {config.outlier_mode}")
    print(f"  smooth_window:     {config.smooth_window}")
    print(f"  feature windows:   short={config.feature_short_window_s}s, "
          f"long={config.feature_long_window_s}s")

    for path in files:
        try:
            report = preprocess_file(path, config)
            print_report(path, report)
        except Exception as e:
            print(f"\n  ── FAILED: {path} ──\n  {type(e).__name__}: {e}",
                  file=sys.stderr)
            if args.strict:
                raise


if __name__ == "__main__":
    main()