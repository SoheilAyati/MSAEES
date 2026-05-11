# NILM Project — Data Format Specification

**Version:** 0.1 (draft)    
**Milestone:** 1    
**Owners:** Soheil Ayati, Marc Steffgen   
**Last updated:** 2026-05-11

---

## 1. Purpose and scope

This document defines the on-disk data format for synthetic (and future real) measurements in the NILM project. Every channel, rate, and structural decision is justified either by a documented capability of the **Siemens SENTRON PAC4200** power monitoring device, or by an explicit NILM requirement, or by an advantage that synthetic data offers over real datasets.

**In scope:** channels, sampling rates, timing reference, scenario duration, file format, naming, metadata schema, sign conventions, units, numeric precision.

**Out of scope:** appliance generator internals (separate doc), preprocessing pipeline details (separate doc), ML feature extraction (Milestone 2).

---

## 2. Conceptual model

Each scenario file represents 24 hours of measurements at a single point of common coupling (PCC), as if a PAC4200 were continuously polling the bus over Modbus TCP.

Two parallel views are stored:

- **`/measurements`** — the aggregate signal the meter would actually see. This is what a real NILM algorithm operates on.
- **`/ground_truth`** — the per-appliance breakdown. The sum of all per-appliance contributions equals the aggregate signal (within numerical precision). This view is only possible because we control the data generation; real datasets do not have it.

Plus metadata describing the scenario.

---

## 3. PAC4200 capabilities this spec is grounded in

| Capability | Value | Source |
|---|---|---|
| Internal sampling | 170 samples per cycle (~8.5 kHz at 50 Hz) | Siemens datasheet |
| Voltage / current measurement | True-RMS, sinusoidal or distorted | Siemens datasheet |
| Connection types supported | 3P4W, 3P3W, 3P4WB, 3P3WB, 1P2W | Siemens manual |
| Harmonics measured | 1st through 64th, V and I per phase, mag + phase | Siemens manual |
| THD | Per phase, V and I | Siemens datasheet |
| Power quantities | P, Q, S, PF, cos φ per phase and total | Siemens datasheet |
| Frequency range | 45–65 Hz | Siemens datasheet |
| Accuracy | Class 0.2 per IEC 61557-12 | Siemens datasheet |
| Communication | Modbus TCP (standard) | Siemens datasheet |
| Realistic poll cycle (full register block) | ~200 ms | Modbus TCP practical bound |
| Native load-profile storage | 15-min fixed/rolling block, 40 days buffer | Siemens manual |

**What is NOT possible with the PAC4200 over Modbus TCP:** streaming raw waveform samples at 8.5 kHz. The device exposes processed values (RMS, P, Q, harmonics) at whatever rate the client can poll Modbus, not waveform-level data.

This is the load-bearing constraint behind the sampling-rate decision in section 5.

---

## 4. Channels

### 4.1 Timestamp

| Name | Unit | Type | Rate | Why |
|---|---|---|---|---|
| `timestamp` | µs since Unix epoch (UTC) | `int64` / `datetime64[us]` | 5 Hz | µs precision because Modbus reply latency varies; ms could lose ordering on close events. UTC because no DST ambiguity. |

### 4.2 Voltage (raw electrical, RMS line-to-neutral)

| Name | Unit | Type | Rate |
|---|---|---|---|
| `V_L1`, `V_L2`, `V_L3` | V | `float32` | 5 Hz |

**Why included:** voltage is the reference signal; needed for power calculations and to detect grid disturbances/dips.
**Why this rate:** PAC4200 internally updates per cycle (20 ms); Modbus TCP poll cycle for a full register block is ~200 ms (5 Hz). 5 Hz captures all voltage dynamics relevant to load monitoring.

### 4.3 Current (raw electrical, RMS, per phase + neutral)

| Name | Unit | Type | Rate |
|---|---|---|---|
| `I_L1`, `I_L2`, `I_L3`, `I_N` | A | `float32` | 5 Hz |

**Why included:** current is the primary load signature; the heart of NILM. When an appliance switches, current changes first.
**Note:** PAC4200 *calculates* `I_N` from the three phase currents rather than measuring it directly. Documented here so it isn't treated as an independent measurement.
**Why this rate:** same as voltage.

### 4.4 Power

| Name | Unit | Type | Rate |
|---|---|---|---|
| `P_L1`, `P_L2`, `P_L3`, `P_total` | W | `float32` | 5 Hz |
| `Q_L1`, `Q_L2`, `Q_L3`, `Q_total` | var | `float32` | 5 Hz |
| `S_L1`, `S_L2`, `S_L3`, `S_total` | VA | `float32` | 5 Hz |
| `PF_L1`, `PF_L2`, `PF_L3`, `PF_total` | — | `float32` | 5 Hz |
| `cosphi_L1`, `cosphi_L2`, `cosphi_L3`, `cosphi_total` | — | `float32` | 5 Hz |

**Why P:** primary NILM disaggregation feature; every appliance has a characteristic P signature.
**Why Q:** separates inductive (motors, transformers), resistive (heaters), and capacitive loads. The (P, Q) feature plane is the classical Hart-1992 NILM signature space; without Q you can't distinguish a fridge from an oven at similar wattage.
**Why S:** derivable from P and Q, but stored to match PAC4200 native output and for fast access.
**Why PF and cos φ:** distinguishes power-electronic loads (low PF, distorted current) from linear loads. PF is total (includes distortion); cos φ is displacement only. Storing both is cheap and disambiguating.
**Why this rate:** same as voltage.

### 4.5 Distortion

| Name | Unit | Type | Rate |
|---|---|---|---|
| `freq` | Hz | `float32` | 5 Hz |
| `THD_V_L1`, `THD_V_L2`, `THD_V_L3` | % | `float32` | 5 Hz |
| `THD_I_L1`, `THD_I_L2`, `THD_I_L3` | % | `float32` | 5 Hz |

**Why `freq`:** context channel for grid state and PV-grid interaction analysis. Not used directly for NILM disaggregation.
**Why THD:** fast distortion summary; cheap to compute; coarse appliance feature. Sharp THD changes also serve as cheap event triggers.
**Why this rate:** same as voltage.

### 4.6 Harmonics

| Name pattern | Unit | Type | Rate |
|---|---|---|---|
| `H_V_L{1,2,3}_n_mag`, n = 2..40 | V | `float32` | 5 Hz |
| `H_V_L{1,2,3}_n_phase`, n = 2..40 | rad | `float32` | 5 Hz |
| `H_I_L{1,2,3}_n_mag`, n = 2..40 | A | `float32` | 5 Hz |
| `H_I_L{1,2,3}_n_phase`, n = 2..40 | rad | `float32` | 5 Hz |

**Column count:** 39 harmonics × 3 phases × 2 (V, I) × 2 (mag, phase) = **468 columns**.

**Why harmonics 2 through 40:** each appliance has a harmonic fingerprint. Power electronics and motors produce characteristic harmonic patterns. PAC4200 exposes up to the 64th, but appliance harmonic content decays rapidly above the 25th — truncating at 40 keeps headroom while saving ~38% storage on this section.

**Why phases retained (not just magnitudes):** the same harmonic order from two different sources (e.g. PV inverter and motor) has different phase relationships relative to the fundamental. Phases are diagnostic for source attribution — directly relevant to angle 3 (PV-aware disaggregation) and angle 4 (multi-feature fusion).

**Why 5 Hz (not 1 Hz):**
- *Alignment.* All measured channels at one rate means one timestamp column applies to everything. No resampling needed for ML feature construction.
- *Switching transients.* Inrush currents on motor start-up contain disproportionate harmonic energy that decays within ~500 ms. 1 Hz risks missing it; 5 Hz gives ~5 samples across the transient.
- *Multi-state transitions.* Washing-machine and EV-mode transitions complete in 2–5 s; harmonic signatures change between phases. 1 Hz can smear two phases together.
- *Adversarial scenarios.* Distinguishing near-simultaneous events needs joint feature vectors at one common rate.
- *Storage is acceptable.* See appendix A.

### 4.7 Ground truth (synthetic-data advantage)

| Name pattern | Unit | Type | Rate |
|---|---|---|---|
| `state_<appliance>` | enum (string) | `category` | 5 Hz |
| `P_contribution_<appliance>` | W | `float32` | 5 Hz |

For each appliance instance in the scenario, two channels: the discrete state (e.g. `"off"`, `"spin_phase"`, `"fast_charge"`) and its active-power contribution to the aggregate signal.

**Why:** real datasets don't have per-sample ground truth. This is the central synthetic-data advantage and is what makes:
- Supervised ML in M2 trivial (every sample has a correct label).
- Probabilistic disaggregation (angle 2) directly trainable.
- Per-appliance error metrics computable without manual annotation.

**Why same rate as measurements:** so labels and features align without resampling.

**Invariant:** `sum(P_contribution_<a>) ≈ P_total` for every timestep, within numerical precision plus injected noise.

---

## 5. Sampling rate summary

| Channel group | Rate | Anchor for the rate |
|---|---|---|
| Voltage, current, power (P, Q, S, PF, cos φ), distortion (freq, THD), harmonics (2nd–40th, mag + phase) | **5 Hz (200 ms)** | Realistic Modbus TCP poll cycle for full PAC4200 register block |
| Ground truth (states + contributions) | **5 Hz** | Aligned with measurements |
| Metadata | **Once per file** | Static per scenario |

No event log channel. Events can be derived post-hoc from continuous channels using change-point detection on P_total in Milestone 2.

---

## 6. Sign conventions

Defined once. Never deviated from.

- **Active power P:** positive = consumption from grid; negative = generation (PV, synchronous machine in generator mode).
- **Reactive power Q:** positive = inductive (lagging); negative = capacitive (leading).
- **Apparent power S:** always non-negative (magnitude).
- **Power factor PF:** in [0, 1], unsigned.
- **cos φ:** in [-1, 1], signed (sign indicates leading/lagging).

---

## 7. Phase reference for harmonics

All harmonic phases are referred to the zero crossing of `V_L1` fundamental. Required for cross-appliance comparison when appliances sit on different phases.

---

## 8. Phase assignment for single-phase appliances

Most household appliances (fridge, hair dryer, PCs) are single-phase. At scenario generation time, each appliance instance is assigned to a specific phase (L1, L2, or L3) and that assignment is recorded in metadata.

**Why this matters:** phase assignment is a free disambiguation feature. Two identical hair dryers on different phases are distinguishable in the data; an algorithm that uses phase information will outperform one that doesn't.

---

## 9. Timing reference and scenario duration

### 9.1 Atomic unit: 24-hour scenarios

Each file covers exactly 24 hours, starting at 00:00:00 UTC of a synthetic anchor date.

**Why 24 hours:**
- PV has a strong diurnal cycle. Angle 3 (PV-aware NILM) requires a full day to make sense.
- Fridge cycles ~30–60 min → 24 h contains ~30 cycles, statistically meaningful.
- Washing machine cycles ~1–2 h with long idle periods → 24 h captures one cycle in realistic context.
- EV charging is typically overnight → requires a full 24 h.
- PC and hair-dryer usage follows morning/evening time-of-day patterns.
- These daily patterns are themselves NILM signatures; algorithms that exploit prior knowledge of when appliances run will disaggregate better.

**Why not 1 hour:** too short — half the appliances never run in a 1 h window, no PV cycle, no overnight EV charging context.

**Why not 1 week:** files become ~1.5 GB each, iteration is slow, one seed per file gives less variety than seven 24 h files. Weekly patterns (weekday vs weekend) are recoverable across multiple 24 h files with appropriate weekday metadata.

### 9.2 Anchor datetimes

Each scenario has an anchor datetime in ISO 8601 UTC stored in metadata. Anchor dates vary across scenarios so that solar elevation and weekday context vary — useful for PV realism and for generalization testing (angle 5).

### 9.3 Total dataset for Milestone 1

| Purpose | Count | Notes |
|---|---|---|
| Training scenarios (mixed easy/normal, varied parameters) | 10 days | ML training in M2 |
| Benchmark — easy tier | 3 days | Sparse, non-overlapping events |
| Benchmark — normal tier | 3 days | Realistic concurrent events |
| Benchmark — hard tier | 3 days | Concurrent events, similar-power appliances |
| Benchmark — adversarial tier | 3 days | EV mirroring hair dryer + PCs; PV cancelling fridge |
| **Total** | **22 days** | ~5 GB compressed |

**Why this much:** training set large enough for M2 ML; three independent days per benchmark tier means meaningful test statistics rather than single-shot evaluation; adversarial tier kept small but deliberate.

---

## 10. File format

### 10.1 Choice: HDF5 (`.h5`)
Use https://myhdf5.hdfgroup.org/ to see and visualize .h5 files.
| Option | Pros | Cons | Verdict |
|---|---|---|---|
| HDF5 | Hierarchical (measurements + ground truth + metadata in one file); efficient mixed types; built-in compression; self-describing | Slightly heavier dependency than CSV/Parquet | **Chosen** |
| Parquet | Excellent columnar storage; great pandas/Polars integration | No native hierarchical structure; would split metadata into separate file | Rejected |
| CSV | Universally readable | No types, no compression, slow, no nested structures | Rejected |

### 10.2 Internal structure

```
scenario_<id>.h5
├── /timestamp                              (1D, 5 Hz)
├── /measurements                           (group)
│   ├── voltage      (2D: time × {V_L1, V_L2, V_L3})
│   ├── current      (2D: time × {I_L1, I_L2, I_L3, I_N})
│   ├── power        (2D: time × {P_L1..P_total, Q_L1..Q_total, S_*, PF_*, cosphi_*})
│   ├── distortion   (2D: time × {freq, THD_V_L1..3, THD_I_L1..3})
│   └── harmonics    (2D: time × 468 harmonic columns)
├── /ground_truth                           (group)
│   ├── states           (2D: time × N_appliances, categorical)
│   └── contributions    (2D: time × N_appliances, W)
├── /metadata                               (group of attributes)
│   ├── tier                 ("easy" | "normal" | "hard" | "adversarial" | "train")
│   ├── seed                 (int)
│   ├── generator_version    (semver string, e.g. "0.1.0")
│   ├── format_version       ("0.1")
│   ├── anchor_datetime      (ISO 8601 UTC string)
│   ├── duration_seconds     (86400)
│   ├── sample_rate_hz       (5)
│   └── appliances           (subgroup)
│       └── <appliance_name>
│           ├── phase             ("L1" | "L2" | "L3" | "all")
│           ├── parameters        (JSON-encoded dict)
│           └── instance_id       (int)
└── /preprocessed                           (group, populated after preprocessing)
    └── (mirrors /measurements structure with cleaned data)
```

### 10.3 Compression and chunking

- **Compression:** LZF (fast decompression, modest ratio). Trades 2× larger files for ~3× faster read vs gzip — appropriate for ML training loops that re-read often.
- **Chunk size:** 3600 samples per chunk (= 12 minutes at 5 Hz). Balances random access against compression efficiency.

---

## 11. File naming

```
scenario_{tier}_{anchor_date}_{seed}.h5
```

Examples:
- `scenario_train_20240101_42.h5`
- `scenario_hard_20240315_137.h5`
- `scenario_adversarial_20240601_999.h5`

`tier` ∈ {train, easy, normal, hard, adversarial}.
`anchor_date` in `YYYYMMDD`.
`seed` is the random seed (int).

The name alone identifies the file uniquely without opening it.

---

## 12. Numeric precision

- **All measurement and ground-truth channels:** `float32`. PAC4200 accuracy is class 0.2 (~0.2%), giving ~3 significant digits; `float32` provides ~7 significant digits, well beyond meter accuracy. `float64` doubles storage with no real benefit.
- **Timestamps:** `int64` microseconds since Unix epoch, displayed as `datetime64[us]`.
- **States:** pandas categorical (compact integer-backed string enum).

---

## 13. Versioning

- `format_version` (this document): bumped on any breaking change to file structure or channel names. Currently `0.1`.
- `generator_version`: bumped per appliance generator release using semantic versioning.

Files always carry both, so any reader can determine compatibility.

---

## 14. Open questions / deferred decisions

These are noted here so they don't get lost, but they are owned by other documents:

- **Noise injection** (Gaussian per channel + occasional spikes, calibrated to class-0.2 accuracy) — specified in preprocessing pipeline doc, not here.
- **Simulated Modbus dropout** (occasional missing samples in raw view, filled in preprocessed view) — preprocessing doc.
- **Phase-imbalance handling** for 3-phase appliances — appliance generator doc.

---

## Appendix A: Storage budget

At 5 Hz over 24 h = 432 000 samples per channel.

| Section | Columns | Raw size (float32) |
|---|---|---|
| Voltage | 3 | 5.2 MB |
| Current | 4 | 6.9 MB |
| Power (P, Q, S, PF, cos φ each × 4) | 20 | 34.6 MB |
| Distortion (freq + 6 × THD) | 7 | 12.1 MB |
| Harmonics (468) | 468 | 808 MB |
| Ground truth (~10 appliance × 2) | ~20 | 34.6 MB |
| **Total raw** | ~520 | **~900 MB / day** |
| **After LZF compression (~25% ratio)** | | **~225 MB / day** |
| **22-day dataset, compressed** | | **~5 GB** |