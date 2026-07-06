# NILM Project - Data Format Specification

**Version:** 0.4
**Milestone:** 1 & 2
**Owners:** Soheil Ayati, Marc Steffgen
**Last updated:** 2026-07-05

---

## 1. Purpose and scope

This document defines the on-disk data formats actually written by the project code. There are four HDF5 layouts (single-appliance, scenario/aggregate, PAC4200 recording, preprocessed additions) plus one CSV format for the early hand-measured device runs. Every layout was verified against the writer code:

| Layout | Writer | Section |
|---|---|---|
| (a) Single-appliance | `Scripts/Synthetic_data_generator/Appliance_generator.py` (`save_trace_hdf5`) and the adapter in `Scripts/Aggregator/mix_measured_scenarios.py` (`write_appliance`) | 4 |
| (b) Scenario / aggregate | `Scripts/Aggregator/aggregator.py` (`write_scenario`), also used by `mix_measured_scenarios.py` | 5 |
| (c) PAC4200 recording | `Scripts/PAC4200_reader/pac_reader.py` (`IncrementalHDF5Writer`) | 6 |
| (d) Preprocessed additions | `Scripts/Preprocessor/preprocessor.py` (`write_preprocessed`) | 7 |
| CSV (Pre_Measured) | external logging tool, files in `Pre_Measured/*.csv` | 8 |

**In scope:** dataset names, shapes, dtypes, metadata attributes, units, sign conventions, sampling rate, compression.
**Out of scope:** generator internals (`02_appliance_generator.md`), aggregation math (`03_aggregator.md`), preprocessing behaviour (`04_preprocessor.md`), the reader itself (`05_pac4200_reader.md`).

---

## 2. Common invariants

These hold across all four HDF5 layouts:

- **Timestamps:** top-level `/timestamp` dataset, `int64`, microseconds since the Unix epoch, UTC. Microsecond precision because Modbus reply latency varies; UTC because it has no DST ambiguity.
- **Measurement data:** `float32`. The PAC4200 is accuracy class 0.2 (about 3 significant digits); `float32` gives about 7, so `float64` would only double storage.
- **Strings:** fixed-length bytes, dtype `S32` (state labels, appliance names).
- **Compression:** LZF on every dataset (fast decompression, modest ratio; appropriate for ML loops that re-read files often).
- **Nominal sample rate:** 5 Hz (200 ms). This is what a Modbus TCP client realistically sustains for a full PAC4200 register block, and it is fast enough to catch inrush transients (1-2 samples) and multi-state transitions.
- **Harmonics:** 39 orders, 2 through 40. Harmonic arrays have shape `(N, 39)`; order n sits at column n-2. Magnitudes in ampere (current) and phases in radians. Note the PAC4200 itself only provides magnitudes (see section 6). (The Milestone 1 report mentioned truncation at order 15; that statement is outdated. All writers store orders 2..40; rationale in `09_design_rationale.md` section 2.3.)
- **Version caveat:** `format_version` is per-writer and intentionally not unified: `"0.2"` for single-appliance files, `"0.1"` for scenario files and PAC4200 recordings. Check `metadata` attrs, not the filename, when a consumer needs to distinguish layouts.

### 2.1 Sign conventions

- **Active power P:** positive = consumption from grid; negative = generation (PV, synchronous machine in generator mode).
- **Reactive power Q:** positive = inductive (lagging); negative = capacitive (leading).
- **Apparent power S:** non-negative magnitude.
- **Power factor PF:** signed, in [-1, +1]. The PAC4200 reports signed PF (sign indicates direction of real power flow) and the preprocessor bounds match that.
- **cos phi:** signed, in [-1, +1] (displacement factor, fundamental only).
- **I_N:** non-negative magnitude.

---

## 3. File format choice: HDF5

Use https://myhdf5.hdfgroup.org/ to inspect and visualize `.h5` files.

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| HDF5 | Hierarchical (measurements + ground truth + metadata in one file); mixed types; built-in compression; self-describing; supports incremental resizable writes (needed by the live recorder) | Heavier dependency than CSV | **Chosen** |
| Parquet | Excellent columnar storage | No native hierarchy; metadata would need a sidecar file | Rejected |
| CSV | Universally readable | No types, no compression, no nesting | Rejected |

---

## 4. Layout (a): single-appliance file

Produced by `Appliance_generator.py` for synthetic traces, and by the `mix_measured_scenarios.py` adapter when it rewrites a real PAC4200 recording into this shape. This is the only layout `aggregator.py` accepts as input.

```
<appliance>.h5
|-- /timestamp                       int64 (N,)   microseconds UTC
|-- /measurements
|   |-- P                            float32 (N,)     W, contribution to aggregate
|   |-- Q                            float32 (N,)     var
|   |-- harmonics_I_mag              float32 (N, 39)  A, orders 2..40
|   `-- harmonics_I_phase            float32 (N, 39)  rad
|-- /ground_truth
|   |-- state                        S32 (N,)         per-sample state label
|   `-- P_contribution               float32 (N,)     identical to /measurements/P
`-- /metadata                        (attrs)
    |-- format_version               "0.2"
    |-- generator_version            "0.1.0" (or "pac_adapter_0.1" from the mix adapter)
    |-- sample_rate_hz               float
    |-- anchor_datetime              ISO 8601 UTC string
    |-- tier                         "single_appliance" (or "measured_single")
    `-- appliance_metadata           JSON: name, appliance_type, instance_id,
                                     phase ("L1"|"L2"|"L3"|"all"), is_three_phase,
                                     seed, params (+ instance_params or source_label)
```

For three-phase appliances (PV, synchronous machine), P and Q are the totals across all phases; the aggregator distributes them equally to L1/L2/L3 based on `is_three_phase`.

---

## 5. Layout (b): scenario / aggregate file

Produced by `aggregator.py::write_scenario`. This is what the MS2 pipeline trains on. `mix_measured_scenarios.py` writes the same layout with `tier="measured"`.

All measurement channels are **1D per-channel datasets** (not 2D column blocks):

```
scenario.h5  /  measured_scenario_NN.h5
|-- /timestamp                       int64 (N,)
|-- /measurements
|   |-- V_L1, V_L2, V_L3             float32 (N,)  RMS line-to-neutral, V
|   |-- I_L1, I_L2, I_L3             float32 (N,)  true-RMS current, A
|   |-- I_N                          float32 (N,)  neutral current magnitude, A
|   |-- P_L1..L3, P_total            float32 (N,)  W
|   |-- Q_L1..L3, Q_total            float32 (N,)  var
|   |-- S_L1..L3, S_total            float32 (N,)  VA
|   |-- PF_L1..L3, PF_total          float32 (N,)  true PF (includes distortion)
|   |-- cosphi_L1..L3, cosphi_total  float32 (N,)  displacement factor
|   |-- THD_V_L1..L3                 float32 (N,)  %, per phase (line-to-neutral)
|   |-- THD_I_L1..L3                 float32 (N,)  %
|   |-- freq                         float32 (N,)  Hz
|   `-- harmonics/
|       |-- I_mag_{L1,L2,L3}         float32 (N, 39)
|       |-- I_phase_{L1,L2,L3}       float32 (N, 39)
|       |-- V_mag_{L1,L2,L3}         float32 (N, 39)
|       `-- V_phase_{L1,L2,L3}       float32 (N, 39)
|-- /ground_truth
|   |-- appliance_names              S32 (n_app,)   e.g. "fridge_1"
|   |-- P_contribution               float32 (N, n_app)
|   |-- Q_contribution               float32 (N, n_app)
|   |-- state                        S32 (N, n_app)
|   `-- attrs: appliance_<i>_metadata  (JSON per appliance)
`-- /metadata                        (attrs)
    |-- format_version               "0.1"
    |-- aggregator_version           "0.1.0"
    |-- sample_rate_hz, anchor_datetime
    |-- tier                         train|easy|normal|hard|adversarial|measured
    |-- scenario_seed, n_appliances, n_samples, duration_seconds
```

**Invariant:** `sum over appliances of P_contribution[:, a]` equals `P_total` at every sample, within float32 precision. The aggregator prints this check on every run.

---

## 6. Layout (c): PAC4200 recording

Produced by `pac_reader.py` per recording session. It shares the scenario skeleton (top-level `/timestamp`, per-channel 1D float32 datasets under `/measurements`) so `preprocessor.py` runs on it unchanged, but the channel set is the meter's **verified core register map**, which differs from layout (b):

- **Present:** `V_L1..L3`, `V_L12`, `V_L23`, `V_L31` (line-to-line voltages), `I_L1..L3`, `S_L1..L3`, `P_L1..L3`, `Q_L1..L3`, `PF_L1..L3`, `THD_V_L12`, `THD_V_L23`, `THD_V_L31`, `freq`, `V_avg_LN`, `V_avg_LL`, `I_avg`, `S_total`, `P_total`, `Q_total`, `PF_total`, `unbalance_V`, `unbalance_I`.
- **Absent:** `I_N`, `cosphi_*`, per-phase `THD_I_*`, per-phase `THD_V_L1..L3`. THD voltage on this meter's core block is **line-to-line**, not line-to-neutral. THD current and cos phi live in unverified register regions and are only added once confirmed (see `05_pac4200_reader.md`).
- **No `/ground_truth` group.** Real aggregate measurements have no per-appliance breakdown; the label of what was plugged in is in `appliance_label`.
- **Gaps:** a failed poll is stored as a NaN sample with an estimated timestamp, so gaps are visible to the preprocessor.

```
<label>_<YYYYmmdd_HHMMSS>.h5
|-- /timestamp                       int64 (N,)  resizable, chunked (512,)
|-- /measurements/<channel>          float32 (N,) for each core channel above
|-- /measurements/harmonics/         only when recorded with --harmonics
|   |-- I_mag_{L1,L2,L3}             float32 (N, 39)
|   |-- I_phase_{L1,L2,L3}           float32 (N, 39)  ALWAYS ZERO (see below)
|   |-- V_mag_{L1,L2,L3}             float32 (N, 39)
|   `-- V_phase_{L1,L2,L3}           float32 (N, 39)  ALWAYS ZERO
`-- /metadata                        (attrs)
    |-- format_version               "0.1"
    |-- app_version                  "1.0.0"
    |-- sample_rate_hz, anchor_datetime
    |-- source                       "pac4200_monitor"
    |-- appliance_label              the session label
    |-- channels                     JSON list of channel names
    |-- harmonics_enabled            bool
    |-- harmonic_orders              JSON [2..40]        (only with harmonics)
    |-- harmonic_phase_captured      False               (only with harmonics)
    `-- recording_summary            JSON, written on close: appliance_label,
                                     duration_s, n_samples, n_gaps,
                                     configured_sample_rate_hz,
                                     harmonics_enabled, completed_utc
```

The PAC4200 exposes per-order harmonic **magnitudes only** (via Modbus FC 0x14 file records). The `*_phase` datasets exist to keep the array shape identical to layout (b) but are always zero on real recordings; `harmonic_phase_captured=False` makes that explicit so downstream code does not mistake them for measured zeros.

---

## 7. Layout (d): preprocessed additions

`preprocessor.py` modifies a layout (b) or (c) file in place, adding one group and leaving `/measurements`, `/ground_truth`, and `/metadata` untouched as the audit trail. Re-running replaces only `/preprocessed`.

```
/preprocessed
|-- /cleaned/<channel>               float32 (N,)  one per 1D input channel
|-- /features/<name>                 float32 (N,)  the 12 derived features
`-- attrs
    |-- report                       JSON (counts of NaN/inf/outliers/gaps, etc.)
    `-- preprocessor_version         "0.1.0"
```

The feature catalogue and the report schema are specified in `04_preprocessor.md`. 2D harmonic arrays are not cleaned; consumers read them from `/measurements/harmonics/` directly.

---

## 8. CSV format: Pre_Measured device runs

`Pre_Measured/pac4200_*_200ms.csv` are early single-device measurements (toaster, hair dryer stage 1, fluorescent tube, LED lamp, USB charger, mixer) logged at 200 ms from the PAC4200 before the HDF5 recorder existed. Format:

- Separator: `;` (semicolon). Decimal point is `.`.
- One row per sample; single phase (L1) plus totals only; no per-order harmonics.
- Columns: `timestamp_iso` (ISO 8601, Z suffix), `device_name`, `run_id`, `sample_interval_ms`, `u_l1_n_v`, `i_l1_a`, `p_total_w`, `s_total_va`, `s_calc_va`, `q_total_var`, `pf_total`, `frequency_hz`, `thd_u_l1_percent`, `thd_i_l1_percent`, `block_time_difference_ms`.
- `pf_total` and `thd_i_l1_percent` contain the literal token `NaN` when the load draws no current; parsers must handle it.

These files drive the "common channel" (no-harmonics, single-phase) transfer experiments in MS2.

---

## 9. File naming

| Layout | Pattern | Example |
|---|---|---|
| Single-appliance (synthetic) | `<appliance>.h5` (CLI `--output` overrides) | `fridge.h5` |
| Scenario (synthetic) | `--output`, conventionally `scenario_<tier>_*.h5` | `scenario_hard_20240315_137.h5` |
| Scenario (measured mix) | `measured_scenario_NN.h5` + `manifest.json` | `measured_scenario_03.h5` |
| PAC4200 recording | `<safe_label>_<YYYYmmdd_HHMMSS>.h5` | `table_fan_med_20260701_135036.h5` |

PAC4200 recording labels follow the convention `<device>_<setting>`; simultaneous multi-device recordings join labels with a double underscore (`water_boiler_on__table_lamp_on`); see `05_pac4200_reader.md` section 10.

---

## Appendix A: Storage budget

At 5 Hz over 24 h = 432 000 samples per channel (float32 = 4 bytes):

| Section (layout b) | Datasets | Raw size |
|---|---|---|
| Scalar channels | 34 | 59 MB |
| Harmonics (12 arrays x 39 cols) | 468 columns | 808 MB |
| Ground truth (10 appliances) | ~21 columns | 36 MB |
| **Total raw** | | **~900 MB / day** |
| After LZF | | roughly 25-50% of raw, depending on harmonic variability |

Short measured scenarios (300 s) and PAC4200 recordings (minutes each) are far smaller, typically well under 1 MB.
