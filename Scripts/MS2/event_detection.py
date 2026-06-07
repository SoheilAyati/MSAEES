"""
event_detection.py  --  Milestone 2, Stage 2
============================================

Event detection is the entry point of event-based NILM (Hart, 1992): find the
sample indices at which an appliance switches on/off or changes state, because
each such edge carries a (delta P, delta Q, delta harmonics) signature that
identifies the appliance responsible.

Two detectors are provided:

  detect_edges()      Hart-style thresholded edge detector on the derivative of
                      P_total.  Fast, interpretable, the classical baseline.
                      Operates on dP/dt which Milestone 1 already exposes as the
                      `dP_total` feature.

  detect_cusum()      Two-sided CUSUM change-point detector.  More robust to
                      slow ramps and to noisy real data than a raw threshold,
                      at the cost of one extra parameter.

Both return a list of Event records.  evaluate_events() scores a detection
against the ground-truth events from nilm_io.ground_truth_events() using a
tolerance window, giving precision / recall / F1 -- the headline event-detection
metric for the milestone.

Optional: if the `ruptures` package is installed, detect_ruptures() exposes
PELT segmentation as a third method for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Event:
    idx: int            # sample index of the edge
    dP: float           # signed power step (W), positive = load turning on
    direction: str      # "on" | "off"
    score: float = 0.0  # detector-specific confidence


# --------------------------------------------------------------------------
# Hart-style edge detector
# --------------------------------------------------------------------------
def detect_edges(P, sample_rate_hz, threshold_W=20.0, min_gap_s=2.0,
                 smooth_samples=3):
    """Detect switching edges as large samples of the discrete derivative of P.

    Parameters
    ----------
    P : 1-D array of active power (W), typically P_total.
    threshold_W : minimum |delta P| to count as an event.
    min_gap_s : events closer than this are merged (keep the largest).
    smooth_samples : centred moving average applied before differencing to
        suppress single-sample noise (set 1 to disable).
    """
    P = np.asarray(P, dtype=float)
    if smooth_samples > 1:
        k = np.ones(smooth_samples) / smooth_samples
        P = np.convolve(P, k, mode="same")

    dP = np.diff(P, prepend=P[0])
    cand = np.where(np.abs(dP) >= threshold_W)[0]

    # Merge runs / nearby candidates: within min_gap, keep the index of max |dP|.
    events = []
    min_gap = max(1, int(round(min_gap_s * sample_rate_hz)))
    i = 0
    while i < len(cand):
        j = i
        while j + 1 < len(cand) and cand[j + 1] - cand[j] <= min_gap:
            j += 1
        group = cand[i:j + 1]
        k = group[np.argmax(np.abs(dP[group]))]
        step = float(dP[k])
        events.append(Event(int(k), step, "on" if step > 0 else "off",
                            score=abs(step)))
        i = j + 1
    return events


# --------------------------------------------------------------------------
# Two-sided CUSUM change-point detector
# --------------------------------------------------------------------------
def detect_cusum(P, sample_rate_hz, drift_W=5.0, threshold_W=60.0,
                 min_gap_s=2.0):
    """Two-sided cumulative-sum detector.

    Accumulates deviations of P from a running level; flags a change when the
    accumulator crosses ``threshold_W``, then resets.  ``drift_W`` is the
    allowance that prevents slow noise from accumulating.
    """
    P = np.asarray(P, dtype=float)
    g_pos = g_neg = 0.0
    level = P[0]
    last_evt = -10 ** 9
    min_gap = max(1, int(round(min_gap_s * sample_rate_hz)))
    events = []
    for i in range(1, len(P)):
        diff = P[i] - level
        g_pos = max(0.0, g_pos + diff - drift_W)
        g_neg = min(0.0, g_neg + diff + drift_W)
        fired = None
        if g_pos > threshold_W:
            fired = "on"
        elif g_neg < -threshold_W:
            fired = "off"
        if fired and (i - last_evt) >= min_gap:
            step = float(P[i] - level)
            events.append(Event(int(i), step, fired, score=abs(g_pos if fired == "on" else g_neg)))
            last_evt = i
            g_pos = g_neg = 0.0
            level = P[i]
        else:
            # slow level tracking so steady operation does not drift the detector
            level += 0.01 * diff
    return events


def detect_ruptures(P, penalty=200.0, model="l2"):
    """PELT change-point detection (requires the optional `ruptures` package)."""
    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError("pip install ruptures to use detect_ruptures()")
    P = np.asarray(P, dtype=float).reshape(-1, 1)
    algo = rpt.Pelt(model=model).fit(P)
    bkps = algo.predict(pen=penalty)[:-1]
    events = []
    for i in bkps:
        step = float(np.mean(P[i:i + 5]) - np.mean(P[max(0, i - 5):i]))
        events.append(Event(int(i), step, "on" if step > 0 else "off", score=abs(step)))
    return events


# --------------------------------------------------------------------------
# Evaluation against ground truth
# --------------------------------------------------------------------------
def evaluate_events(detected, truth_events, sample_rate_hz, tolerance_s=1.0):
    """Match detected events to ground-truth events within +/- tolerance.

    Greedy nearest matching, each truth event used at most once.

    Returns dict with tp, fp, fn, precision, recall, f1, and the mean absolute
    timing error of matched events (seconds).
    """
    det_idx = sorted(e.idx for e in detected)
    tru_idx = sorted(e["idx"] for e in truth_events)
    tol = tolerance_s * sample_rate_hz

    used = np.zeros(len(tru_idx), dtype=bool)
    tru = np.array(tru_idx)
    tp = 0
    timing_err = []
    for d in det_idx:
        if len(tru) == 0:
            break
        k = int(np.argmin(np.abs(tru - d)))
        if not used[k] and abs(tru[k] - d) <= tol:
            used[k] = True
            tp += 1
            timing_err.append(abs(tru[k] - d) / sample_rate_hz)
    fp = len(det_idx) - tp
    fn = len(tru_idx) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1,
                n_detected=len(det_idx), n_truth=len(tru_idx),
                mean_timing_error_s=float(np.mean(timing_err)) if timing_err else None)
