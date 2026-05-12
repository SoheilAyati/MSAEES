# NILM Project — Preprocessor Specification

**Version:** 0.1 (draft)    
**Milestone:** 1 & 2    
**Companion to:** `data_format.md`, `appliance_generators.md`, `aggregator.md`
**Owner:** Soheil Ayati, Marc Steffgen    
**Last updated:** 2026-05-12

---

## 1. Purpose and scope

The preprocessor is the **universal data hygiene and feature-engineering step** in the NILM pipeline. It reads a scenario HDF5 file's `/measurements` group, produces `/preprocessed` with cleaned channels and derived features, and leaves `/measurements` and `/ground_truth` untouched as the audit trail.

The same script runs on:

- **Synthetic scenarios** produced by `aggregator.py` in Milestone 1.
- **Real scenarios** produced by a Modbus poller logging a Siemens PAC4200 at the PCC in Milestone 2.

This is the architectural payoff of Milestone 1: by the time M2 starts, the preprocessing and downstream ML pipeline are fully built and validated on synthetic data. Only the data acquisition step (synthetic aggregator vs Modbus poller) changes between milestones; preprocessing does not.

**In scope:** validation of input data, handling of NaN/inf values, imputation of short gaps, clipping or flagging of out-of-bounds values, optional rolling-mean smoothing, computation of derived features for ML, and structured reporting of all of the above.

**Out of scope:** signal corruption / noise injection (deliberately not implemented — see §10); feature extraction beyond the 12 features defined here (Milestone 2 may add more); event detection or classification (Milestone 2).

---

## 2. Position in the pipeline

```
                          ┌─────────────────────────┐
                          │ Source A (synthetic):   │
                          │   aggregator.py         │
                          └───────────┬─────────────┘
                                      │
                                      │  scenario_*.h5
                                      │  with /measurements
                                      │  and /ground_truth
                                      │
                          ┌─────────────────────────┐
                          │ Source B (real, M2):    │
                          │   pac4200_modbus_poller │
                          └───────────┬─────────────┘
                                      │
                                      │  scenario_*.h5
                                      │  with /measurements
                                      │  only (no ground truth)
                                      ▼
                          ┌─────────────────────────┐
                          │  PREPROCESSOR (this doc)│  ◄── identical
                          │  preprocessor.py        │       behaviour
                          └───────────┬─────────────┘       for A and B
                                      │
                                      │  same file, with
                                      │  /preprocessed populated
                                      ▼
                          ┌─────────────────────────┐
                          │  Milestone 2: feature   │
                          │  extraction, event      │
                          │  detection, ML          │
                          └─────────────────────────┘
```

The preprocessor makes no assumptions about its input source. It handles whatever data quality issues are present (gaps, NaN, outliers) and produces a consistent feature set regardless. For synthetic data with no quality issues, preprocessing is mostly feature engineering; for real data, it does both cleaning and feature engineering. Same code path either way.

---

## 3. Inputs and outputs

### 3.1 Input

A scenario HDF5 file (from the aggregator or future Modbus poller) satisfying:

- Has a top-level `/timestamp` dataset (1D, microseconds since Unix epoch).
- Has a `/measurements` group containing at least the channels listed in §A.
- Has `/metadata` with a `sample_rate_hz` attribute. If missing, the preprocessor estimates from median timestamp interval and warns.

`/ground_truth` is **not required**. Real PAC4200 scenarios won't have it; the preprocessor never reads from or writes to it.

### 3.2 Output

The same file, modified in place, with a new `/preprocessed` group:

```
/preprocessed
├── /cleaned                ← cleaned versions of all 1D /measurements channels
│   ├── V_L1, V_L2, V_L3
│   ├── I_L1, I_L2, I_L3, I_N
│   ├── P_*, Q_*, S_*, PF_*, cosphi_*
│   ├── THD_V_*, THD_I_*
│   └── freq
├── /features               ← derived features for ML
│   ├── dP_total, abs_dP_total
│   ├── P_total_rolling_{mean,std}_{short,long}
│   ├── P_{L1,L2,L3}_rolling_mean_short
│   ├── mean_THD_I, mean_THD_V
│   ├── P_phase_imbalance
│   └── I_neutral_to_phase_ratio
└── attrs/report            ← JSON-encoded report of what happened
```

`/measurements`, `/ground_truth`, and `/metadata` are explicitly **not touched**. Re-running the preprocessor replaces only `/preprocessed`. This makes the operation idempotent and the original measurements always recoverable.

Harmonic arrays (`/measurements/harmonics/*`, 2D) are also not currently cleaned by the preprocessor (deferred decision, see §10.4). They are read by Milestone 2 directly from `/measurements/harmonics/` when needed.

---

## 4. Operations

The preprocessor performs the following operations in order for each scenario file. Every operation can be tuned via `PreprocessingConfig` (§5).

### 4.1 Validation

Checks performed:

| Check | Failure mode |
|---|---|
| `/measurements` group exists | hard error |
| `/timestamp` dataset exists | hard error |
| ≥ 2 samples | hard error |
| Timestamps strictly increasing | warning (or hard error in `--strict`) |
| `sample_rate_hz` present in metadata | warning; estimate from median dt |

Validation results are appended to `report.warnings`. In `--strict` mode all warnings become hard errors.

### 4.2 NaN and inf handling

Per channel:

1. Count `NaN` values; record per channel in `report.nan_counts`.
2. Count `inf` values; record per channel in `report.inf_counts`.
3. Convert all `inf` to `NaN` so gap imputation can handle both uniformly.

These counts are useful diagnostics — high NaN counts on real PAC4200 data usually mean Modbus connectivity problems; high inf counts mean either driver bugs or impossible derived computations (division by zero).

### 4.3 Gap detection and imputation

A "gap" is a run of consecutive NaN values in a single channel.

- For runs of length ≤ `max_gap_to_impute` (default 5 samples = 1 second at 5 Hz): linear interpolation between the values immediately before and after the gap.
- For longer runs: left as NaN, counted in `report.gaps_left_as_nan`, and downstream features computed on the channel will inherit the NaN.

**Why 5 samples (1 s) for the default threshold:**
- Real PAC4200 Modbus dropouts are typically single-sample or very short bursts caused by network jitter. 1 s covers >95% of realistic dropouts.
- Appliance state changes happen on second-to-minute scales; linear interpolation across <1 s is unlikely to mask a real event.
- Longer gaps (≥ 5 s) usually indicate connectivity loss, not measurement noise, and should not be silently invented.

Edge cases: gaps at the very start or end of the channel cannot be imputed (no value on one side); these are counted as left-as-NaN.

### 4.4 Outlier handling

For each 1D channel with defined physical bounds (§A), any finite value outside the bounds is treated as an outlier. Two modes:

- `clip` (default): replace the value with the nearest bound. Preserves a usable signal for downstream features.
- `flag`: replace the value with `NaN`. Useful for downstream code that wants to ignore unreliable samples explicitly.

**Why clip rather than flag by default:** rolling-window features (means, stds, gradients) become unusable wherever NaN propagates. For most quality issues (spurious single-sample spikes), clipping preserves the surrounding context. Use `flag` only when you specifically want to mask out bad samples.

NaN positions are explicitly preserved through outlier handling — clipping does not turn NaN into bound values. This means gap-imputation status survives outlier detection.

### 4.5 Optional smoothing

Disabled by default (`smooth_window = 1` = no-op).

When enabled, applies a centered rolling-mean filter of window `smooth_window` samples to channels listed in `smooth_channels` (default: `P_total`, `Q_total`). Edge samples are left untouched to avoid convolution artifacts.

**When to enable:**
- Real PAC4200 data at high poll rates often shows quantization-level jitter on P and Q. A 3–5 sample rolling mean removes it without affecting event detection.
- Synthetic data is already clean; smoothing is unnecessary and may slightly delay detected events.

**Channels that should never be smoothed:**
- `freq` (slow drift; rolling mean is meaningless)
- Phase angles or harmonic phases (averaging angles is mathematically wrong)
- `state` ground truth (not a measurement)

### 4.6 Feature engineering

12 features computed from the cleaned channels and written to `/preprocessed/features/`. Each feature is a single 1D array, same length as the input. See §6 for the full catalogue.

---

## 5. Configuration

The preprocessor accepts a `PreprocessingConfig` dataclass:

```python
@dataclass
class PreprocessingConfig:
    max_gap_to_impute: int = 5
    outlier_mode: str = "clip"               # or "flag"
    smooth_window: int = 1                   # 1 = off
    smooth_channels: Tuple[str, ...] = ("P_total", "Q_total")
    feature_short_window_s: float = 5.0
    feature_long_window_s: float = 30.0
    compute_event_features: bool = True
    compute_imbalance_features: bool = True
    compute_distortion_features: bool = True
    strict: bool = False
```

**Defaults are tuned for synthetic data.** For real PAC4200 data the recommended deviation is:

| Parameter | Synthetic default | Real PAC4200 recommended |
|---|---|---|
| `max_gap_to_impute` | 5 | 5 (same) |
| `outlier_mode` | clip | clip |
| `smooth_window` | 1 (off) | 3 |
| `feature_short_window_s` | 5.0 | 5.0 |
| `feature_long_window_s` | 30.0 | 30.0 |
| `strict` | false | false |

`--strict` is mainly useful for CI / regression testing where any data quality issue should fail loudly.

---

## 6. Feature catalogue

The 12 derived features fall into four functional groups. Each is designed to support a specific Milestone 2 task.

### 6.1 Event-detection features

| Feature | Computation | Purpose |
|---|---|---|
| `dP_total` | `gradient(P_total) * sample_rate_hz` (W/s) | signed power-change rate; peaks at appliance switching |
| `abs_dP_total` | `abs(dP_total)` | symmetric event indicator; used for threshold-based event detection |
| `P_total_rolling_std_short` | std over 5 s window | detects bursts of activity vs steady state |

These are the inputs to event-based NILM. The classical Hart-1992 approach detects events as edges in P; `dP_total` is exactly that edge signal.

### 6.2 Steady-state features

| Feature | Computation | Purpose |
|---|---|---|
| `P_total_rolling_mean_short` | mean over 5 s | smoothed P for steady-state matching |
| `P_total_rolling_mean_long` | mean over 30 s | very smooth P for slow-cycle appliances (fridge) |
| `P_L1_rolling_mean_short` | per-phase 5 s mean | per-phase steady-state |
| `P_L2_rolling_mean_short` | same for L2 | |
| `P_L3_rolling_mean_short` | same for L3 | |

These are the inputs to steady-state NILM methods (clustering, HMM). The two windows let ML choose the temporal scale matching each appliance.

### 6.3 Distortion features

| Feature | Computation | Purpose |
|---|---|---|
| `mean_THD_I` | mean of THD_I_{L1,L2,L3}, smoothed over 5 s | aggregate distortion summary; rises when nonlinear loads (PCs, EV chargers) are active |
| `mean_THD_V` | same for THD_V | grid voltage quality indicator |

These provide a single scalar summary of harmonic content that NILM can use without consuming the full 39-bin harmonic spectrum. The full spectrum is still accessible at `/measurements/harmonics/`.

### 6.4 Imbalance features

| Feature | Computation | Purpose |
|---|---|---|
| `P_phase_imbalance` | `(max(P_phase) − min(P_phase)) / abs(mean(P_phase))` | how unbalanced the load is; helps phase-aware NILM |
| `I_neutral_to_phase_ratio` | `I_N / mean(I_L1, I_L2, I_L3)` | neutral-current dominance; indicates strong single-phase loading |

These are particularly useful for distinguishing 3-phase appliances (PV, synchronous machine, fast-AC EV) which produce zero imbalance contribution from single-phase appliances which produce maximum imbalance contribution. The angle-4 (multi-feature fusion) approach uses both.

### 6.5 Feature design rationale

- Every feature is a 1D array of the same length as the input — keeps downstream indexing simple.
- All features are `float32` — sufficient for ML, half the storage of `float64`.
- Centered rolling windows (not trailing) are used so feature values align with the event they describe, not its end.
- No features depend on `/ground_truth` — the preprocessor uses only `/measurements`. This guarantees the same feature set is produced for real and synthetic data.

---

## 7. Report structure

Every preprocessing run produces a JSON-encoded report stored as an attribute on `/preprocessed`:

```json
{
  "preprocessor_version": "0.1.0",
  "timestamp_utc": "2026-05-12T20:30:00+00:00",
  "n_samples": 432000,
  "sample_rate_hz": 5.0,
  "channels_processed": ["V_L1", "V_L2", ...],
  "nan_counts": {"P_total": 75, "Q_total": 50},
  "inf_counts": {"P_total": 1},
  "outliers_clipped": {"P_total": 1, "V_L1": 2},
  "gaps_imputed": 106,
  "gaps_left_as_nan": 1,
  "features_built": ["dP_total", ...],
  "elapsed_seconds": 1.45,
  "warnings": []
}
```

**Why store the report in the file:** during Milestone 2, when ML behavior on one scenario differs from another, the report tells you immediately whether the difference is in the data, the preprocessing, or the model. Without the embedded report this triage requires re-running preprocessing.

---

## 8. CLI

```
python preprocessor.py [options]

Required (one of):
  --input PATH               single scenario file
  --input-dir DIR            directory of scenario files

Optional:
  --pattern GLOB             glob within --input-dir (default "*.h5")
  --max-gap-impute INT       gap imputation threshold (default 5 samples)
  --outlier-mode {clip,flag} default clip
  --smooth-window INT        rolling-mean window in samples (default 1 = off)
  --short-window-s FLOAT     short feature window in seconds (default 5)
  --long-window-s FLOAT      long feature window in seconds (default 30)
  --strict                   raise on validation issues instead of warning
```

Batch processing (e.g., the full 22-scenario dataset) is a single command:

```bash
python preprocessor.py --input-dir ./scenarios
```

The script reports per-file results as it goes and continues to the next file on errors (unless `--strict`).

---

## 9. Behaviour for real PAC4200 data (M2)

Documenting the expected behaviour now so M2 has no surprises.

The Modbus poller in M2 will produce scenario files with:

- `/measurements/{V,I,P,Q,...}` populated by polling PAC4200 registers at ~5 Hz.
- `/measurements/harmonics/*` populated from the PAC4200 harmonic registers (it natively exposes orders 1 through 64).
- No `/ground_truth`. (Optional: a "ground truth" via sub-meters can be added in M2 if the lab budget allows; the preprocessor doesn't need it.)
- Same `/metadata` schema with `sample_rate_hz`, `anchor_datetime`, etc.

Expected differences from synthetic data:

| Aspect | Synthetic | Real PAC4200 |
|---|---|---|
| Sample regularity | exact 200 ms spacing | jitter of ±10–50 ms typical |
| Missing samples | none | occasional (Modbus retries, network blips) |
| Noise floor | none in aggregator output | ~0.2% of full-scale per channel |
| Outliers | none unless injected for testing | rare, mostly transient meter glitches |
| `is_finite` everywhere | yes | mostly; rare driver bugs produce inf |
| Voltage harmonics | small synthetic constants | actual measured values, typically 1–3% THD_V |

The preprocessor handles all of these without code changes — only configuration changes (`--smooth-window 3` recommended for real data).

---

## 10. Open issues / deferred decisions

1. **No noise injection in M1.** Some NILM projects deliberately corrupt synthetic data to test preprocessing robustness. We've deferred this because the corruption parameters (noise amplitude, dropout rate, etc.) cannot be calibrated without real PAC4200 measurements to compare against. M2 can revisit this once real data is available — at that point a `pac4200_simulator.py` module that takes clean synthetic data and adds calibrated realism becomes useful and well-defined.

2. **Smoothing default is off.** Conservative — better to feed slightly noisy data to ML than to over-smooth and lose event edges. The user explicitly enables smoothing with `--smooth-window N` when needed. Considered but rejected: auto-detecting smoothing need based on per-channel noise statistics, since this would change preprocessing behavior across runs in non-obvious ways.

3. **PF sign convention is [-1, +1], not [0, +1].** Matches PAC4200's convention where sign indicates direction of real power flow (positive = consumption, negative = generation/export). Differs slightly from the original data-format spec §6 which described unsigned PF; the preprocessor reflects what the meter actually does. The data format spec should be updated to match in a future revision.

4. **Harmonic arrays are not cleaned.** `/measurements/harmonics/*` (2D, 39 columns per phase) is currently left untouched. Cleaning would require deciding how to impute multi-dimensional gaps and define physical bounds per harmonic order — both non-trivial. Deferred until M2 demonstrates a need (e.g., real PAC4200 harmonic data has missing values that block feature extraction).

5. **No imputation of ground-truth states.** Even if `/ground_truth/state` happens to have NaN-like values (it shouldn't, but real datasets sometimes do), the preprocessor leaves it alone. Ground truth must remain a verbatim record of what the data source claims, not a cleaned interpretation. Add a separate `ground_truth_cleaner` module if needed in M2.

6. **Feature set is fixed at 12.** Milestone 2 may discover that additional features are needed (e.g., FFT-derived spectral entropy, dominant-harmonic indicators, slot-based statistics). The framework is extensible — adding a feature is one function definition plus one entry in `build_features()`. Document new features in this spec when they're added.

7. **No frequency-domain preprocessing.** The preprocessor only operates in the time domain. Spectral features (FFT slices, wavelet coefficients) are deferred to Milestone 2's feature extraction stage, where they're properly scoped to the modeling approach.

---

## Appendix A — Physical bounds for outlier handling

Bounds are deliberately generous: legitimate extreme operating conditions (EV fast charging at 22 kW, PV peak generation, synchronous machine in all four quadrants) are all inside the bounds. Outliers indicate measurement errors or data corruption, not unusual load behavior.

| Channel | Lo | Hi | Units | Rationale |
|---|---|---|---|---|
| `V_L*` | 180 | 270 | V | ±18% of nominal 230 V — well outside any normal grid condition |
| `I_L*`, `I_N` | 0 | 200 | A | covers ≥40 kW single-phase loads; >200 A indicates sensor fault |
| `P_L*` | −30 000 | +30 000 | W | per-phase room for fast AC at 22 kW (~9.6 kW/phase) |
| `P_total` | −50 000 | +50 000 | W | ditto for all three phases combined |
| `Q_L*` | −15 000 | +15 000 | var | covers typical reactive loads with margin |
| `Q_total` | −30 000 | +30 000 | var |   |
| `S_L*` | 0 | 50 000 | VA | apparent power is always non-negative |
| `S_total` | 0 | 100 000 | VA |   |
| `PF_*` | −1 | +1 | — | signed PF (PAC4200 convention) |
| `cosphi_*` | −1 | +1 | — | by definition |
| `THD_V_*` | 0 | 50 | % | normal residential <8%; outside 50% is sensor fault |
| `THD_I_*` | 0 | 200 | % | THD_I can exceed 100% with very nonlinear loads |
| `freq` | 47 | 53 | Hz | ±3 Hz from nominal 50; outside this the grid has failed |

Bounds are stored in the module-level `PHYSICAL_BOUNDS_1D` dictionary and can be overridden by editing the source — no CLI override is provided to discourage casual changes.

---

## Appendix B — Sample run on dirty data

Input: a synthetic scenario file with deliberately injected issues — 50 single dropouts, 5 burst dropouts (3 samples each), one 10-sample long gap, 1 inf value, 3 out-of-bounds outliers, plus calibrated 0.3% Gaussian noise on V_L1.

Output report:

```
Samples: 432000 @ 5.0 Hz (24.00 h)
Channels processed: 34
Features built:     12
NaN values found:   125 across 2 channel(s)
  - P_total: 75
  - Q_total: 50
Inf values found:   1 across 1 channel(s)
Outliers clipped:   3 across 2 channel(s)
  - P_total: 1
  - V_L1: 2
Gaps: 106 imputed, 1 too long, left as NaN
Elapsed: 1.45 s
```

All injected issues correctly accounted for: 75 NaN on P_total (50 singles + 5×3 bursts + 10-sample long = 75 ✓); 106 short gaps imputed (50+50+5+1-from-inf = 106 ✓); 1 long gap left (the 10-sample gap, since 10 > max_gap_to_impute=5 ✓); 3 outliers clipped at physical bounds ✓.

This demonstrates the preprocessor is M2-ready: it handles the kinds of data quality issues that real PAC4200 acquisition will surface, with no code changes from its synthetic-data operation.