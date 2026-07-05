# NILM Project - Scenario Aggregator Specification

**Version:** 0.2
**Milestone:** 1 & 2
**Companion to:** `01_data_format.md`, `02_appliance_generator.md`
**Owners:** Soheil Ayati, Marc Steffgen
**Last updated:** 2026-07-05

---

## 1. Purpose and scope

The aggregator (`Scripts/Aggregator/aggregator.py`) combines per-appliance HDF5 files (produced by `Appliance_generator.py`) into a single **scenario file** that mimics what a Siemens PAC4200 would log at the Point of Common Coupling.

The aggregator is a **synthetic-data-only step**. In a real deployment the PAC4200 already provides aggregated signals; the aggregator simulates that physics so the downstream preprocessing/ML pipeline is identical for real and synthetic data.

**In scope:** combining per-appliance P, Q, and harmonic contributions into PCC-level aggregate signals; synthesizing voltage, frequency, and voltage harmonics; deriving secondary quantities (PF, cos phi, THD, S, I_N); writing the scenario file with both `/measurements` (PAC4200 view) and `/ground_truth` (per-appliance breakdown).

**Out of scope:** noise injection, Modbus dropout simulation, quantization; cleaning of corrupted signals (preprocessing); feature extraction (Milestone 2).

This document also covers `mix_measured_scenarios.py` (section 10), the companion script that feeds **real PAC4200 recordings** through the same aggregation path.

---

## 2. Position in the pipeline

```
   per-appliance generators           (clean, deterministic per-appliance traces)
   Appliance_generator.py
              |   N x .h5 files
              v
   AGGREGATOR (this doc)              synthetic-only; simulates physics at the PCC
   aggregator.py
              |   scenario_*.h5
              v
   preprocessor.py                    cleaning + 12 features, /preprocessed in same file
              |
              v
   Milestone 2 ML
```

The split exists because per-appliance ground truth has to be tracked separately for ML supervision. That is the entire reason the synthetic-data approach gives an advantage over real datasets like UK-DALE.

---

## 3. Inputs and outputs

### 3.1 Inputs

A list of per-appliance HDF5 files (layout (a) in `01_data_format.md`), all of which must satisfy:

- Same `sample_rate_hz` (typically 5 Hz).
- Same number of samples.
- Same `anchor_datetime`, otherwise PV solar position is computed for a different date than other appliances see, which is physically wrong.

The aggregator validates these on load and errors out on mismatch by default. The CLI flag `--allow-anchor-mismatch` downgrades the anchor check to a warning, but is almost never the right choice.

### 3.2 Output

A single scenario HDF5 file (layout (b) in `01_data_format.md`) containing:

- `/measurements` : what a PAC4200 would log at the PCC. The single observable signal that downstream NILM operates on.
- `/ground_truth` : the per-appliance breakdown (states plus per-appliance P and Q contributions).
- `/metadata` : scenario-level attributes (tier, seed, anchor datetime, format version, etc.).
- Top-level `/timestamp` shared across all channels.

`/preprocessed` does not exist at this stage; the preprocessing step adds it.

---

## 4. Aggregation math

### 4.1 Power per phase

Each appliance contributes to one or all three phases depending on its `is_three_phase` metadata flag:

- **Single-phase appliance** (fridge, PC, hair dryer, etc.): contributes its full P and Q to the phase recorded in its metadata (`L1`, `L2`, or `L3`).
- **3-phase appliance** (PV, synchronous machine): distributes P and Q equally across L1, L2, L3 (each phase gets one third).

```
P_phase[p] = sum of P_appliance_i  where appliance_i is on phase p (or distributes to p)
Q_phase[p] = same for Q
P_total    = P_L1 + P_L2 + P_L3
Q_total    = Q_L1 + Q_L2 + Q_L3
```

**Why equal distribution for 3-phase appliances:** in a balanced 3-phase appliance (PV inverter, generator), power is shared symmetrically across phases. We assume perfect symmetry; real installations have small imbalances (~1%) that we don't capture.

### 4.2 Voltage synthesis

The PAC4200 measures voltage at the PCC; in real installations voltage is largely set by the grid, not the load (the "infinite/stiff bus" assumption). The aggregator therefore synthesizes voltage independently of the aggregated load:

- Nominal 230 V phase-to-neutral on each phase.
- Slow random walk (~30 s correlation time, clipped to a plus/minus 2 V envelope): normal grid voltage drift.
- Per-sample Gaussian jitter (sigma = 0.05 V): fast variation.

Each phase has an independent realization, so they are not correlated with each other. Real installations show small correlated drift; our model neglects this.

**Why this matters for NILM:** voltage is mostly context, but it is required to derive currents from powers (I = S / V) and to compute true power factor (PF = P / S with S = V x I_rms).

### 4.3 Frequency synthesis

Nominal 50 Hz with a slow random walk (~60 s correlation, clipped to plus/minus 0.05 Hz). Real grid frequency varies similarly under normal operation. Frequency is a context channel, useful for grid-stability and PV-interaction analysis, not for disaggregation directly.

### 4.4 Current harmonics: complex vector sum

This is the only mathematically non-trivial step. The same harmonic order from two different appliances does not simply add in magnitude; the phases interact:

```
I_3rd_total   = |I_3rd_A| * exp(j * phi_3rd_A)  +  |I_3rd_B| * exp(j * phi_3rd_B)
|I_3rd_total| = |this complex sum|
```

If the two contributions are in phase they add constructively; if 180 degrees out of phase, they cancel. This **harmonic interaction effect** is what makes harmonic-based NILM disaggregation genuinely hard when multiple appliances run simultaneously, and exactly what a multi-feature fusion approach has to deal with.

Per phase, for each harmonic order n in [2, 40]:

```
complex_n[p] = sum over appliances a of (h_mag_a[n] * exp(j * h_phase_a[n]))
               x (1 if a is on phase p; 1/3 if a is 3-phase; 0 otherwise)

|h_n[p]|     = abs(complex_n[p])
arg(h_n[p])  = angle(complex_n[p])
```

Magnitude-only summation would be a noticeable physical error and would mask interaction effects that real PAC4200 measurements show.

### 4.5 Voltage harmonics

For a true infinite bus, voltage harmonics would be zero. Real installations have small voltage harmonics because grid impedance times load current harmonics produces a voltage drop with harmonic content. We approximate this with small fixed fractions of nominal voltage:

| Order | Fraction of V_nominal |
|---|---|
| 3rd | 0.5% |
| 5th | 1.5% |
| 7th | 1.0% |
| 11th | 0.5% |
| 13th | 0.3% |

Each order has a fixed random phase angle per scenario plus small per-sample jitter (sigma 0.02 rad). THD_V lands around 2% on each phase, which matches typical EN 50160 measurements on a stiff grid.

### 4.6 Per-phase current

Per-phase RMS current is computed from the fundamental plus harmonics in quadrature, which is what the PAC4200 reports as true-RMS current:

```
I_fund[p]  = sqrt(P_phase[p]^2 + Q_phase[p]^2) / V_phase[p]     (0 where V <= 1 V)
I_h_rms[p] = sqrt(sum over n of |h_n[p]|^2)
I_rms[p]   = sqrt(I_fund[p]^2 + I_h_rms[p]^2)
```

`I_rms[p]` is what gets stored as `I_L1`, `I_L2`, `I_L3`.

### 4.7 Neutral current

For a 3-phase 4-wire system with single-phase loads, the neutral carries the residual unbalance:

```
I_N = | I_L1 * 1 + I_L2 * a^2 + I_L3 * a |     where a = exp(j * 120 deg)
```

This evaluates to zero when the three phase currents are equal (balanced load) and grows with imbalance.

**Approximation:** the formula uses each phase's RMS current as if it were a positive real value at the reference phase angle. The exact PAC4200 calculation uses the instantaneous phase angles of each phase current at every sample; we approximate with the imbalance-only contribution, which is correct in magnitude but loses zero-sequence triplen-harmonic neutral content. Acceptable for this project.

### 4.8 Derived secondary quantities

All derived per phase plus as totals:

| Channel | Formula | Notes |
|---|---|---|
| `S` | V_phase x I_rms_phase | apparent power |
| `PF` (true) | P / S | includes distortion |
| `cosphi` (displacement) | P / sqrt(P^2 + Q^2) | fundamental only |
| `THD_I` | 100 x I_h_rms / I_fund | percent |
| `THD_V` | 100 x V_h_rms / V_rms | percent |

Division is guarded: PF and cos phi are set to 1.0 where the denominator is at or below 1 (S below 1 VA, or sqrt(P^2 + Q^2) below 1); THD is set to 0 where the fundamental is below 0.01. This avoids both NaN and division warnings on an idle bus.

---

## 5. Sign and phase conventions

Inherited from `01_data_format.md` section 2.1:

- **Active power P:** positive = consumption; negative = generation (PV active hours, synchronous machine in generator mode).
- **Reactive power Q:** positive = inductive (lagging); negative = capacitive (leading). The synchronous machine can produce all four quadrants.
- **Apparent power S:** always non-negative.
- **PF and cos phi:** the guarded-division outputs land in [-1, +1]; idle-bus samples read 1.0 by convention.
- **I_N:** non-negative magnitude.

For single-phase appliances, P and Q on the un-assigned phases are zero.

---

## 6. Tier handling

The aggregator accepts `--tier {train | easy | normal | hard | adversarial}`, which gets recorded in the scenario file's `/metadata` `tier` attribute. (`mix_measured_scenarios.py` additionally writes `tier="measured"`.)

The tier label does not change what the aggregator computes; it is a label for downstream filtering. The content difference between tiers comes from how the per-appliance files were generated before aggregation:

| Tier | Generation strategy | Expected M2 performance |
|---|---|---|
| `train` | Realistic mixed activations; varied seeds for instance diversity | training data; not evaluated |
| `easy` | Sparse non-overlapping events; widely separated power magnitudes | F1 > 0.95 |
| `normal` | Realistic everyday operation; some natural overlap | F1 ~ 0.85 |
| `hard` | Concurrent events; similar-power overlaps; smart EV; cloudy PV | F1 ~ 0.65-0.80 |
| `adversarial` | Net-zero scenarios; signature collisions; PV cancelling load | F1 ~ 0.50; deliberately stress-tests |

The aggregator itself is tier-agnostic.

---

## 7. File format

Output structure follows layout (b) in `01_data_format.md`:

```
scenario_*.h5
|-- /timestamp                             (1D int64, microseconds since Unix epoch)
|-- /measurements
|   |-- V_L1, V_L2, V_L3                   (1D float32)
|   |-- I_L1, I_L2, I_L3, I_N              (1D float32)
|   |-- P_L1..P_total, Q_L1..Q_total       (1D float32)
|   |-- S_L1..S_total                      (1D float32)
|   |-- PF_L1..PF_total, cosphi_*          (1D float32)
|   |-- THD_V_L1..3, THD_I_L1..3           (1D float32)
|   |-- freq                               (1D float32)
|   `-- harmonics/
|       |-- I_mag_{L1,L2,L3}               (2D float32, N x 39)
|       |-- I_phase_{L1,L2,L3}             (2D float32, N x 39)
|       |-- V_mag_{L1,L2,L3}               (2D float32, N x 39)
|       `-- V_phase_{L1,L2,L3}             (2D float32, N x 39)
|-- /ground_truth
|   |-- appliance_names                    (1D S32, n_app entries)
|   |-- P_contribution                     (2D float32, N x n_app)
|   |-- Q_contribution                     (2D float32, N x n_app)
|   |-- state                              (2D S32, N x n_app)
|   `-- appliance_<i>_metadata             (attributes, JSON-encoded)
`-- /metadata
    |-- format_version, aggregator_version (string attrs)
    |-- sample_rate_hz                     (float attr)
    |-- anchor_datetime                    (ISO 8601 string attr)
    |-- tier                               (string attr)
    |-- scenario_seed                      (int attr)
    |-- n_appliances, n_samples            (int attrs)
    `-- duration_seconds                   (float attr)
```

- Compression: LZF on all datasets.
- Float precision: float32 throughout.

---

## 8. CLI

```
python aggregator.py [options]

Required (one of):
  --inputs FILE [FILE ...]      explicit list of per-appliance .h5 files
  --input-dir DIR               directory; loads all *.h5 inside

Output / tagging:
  --output PATH                 output HDF5 path (default: scenario.h5)
  --tier {train,easy,normal,hard,adversarial}   tier label (default: train)
  --seed INT                    scenario-level seed for V/freq synthesis (default: 0)

Behaviour:
  --inspect                     print summary (the summary is printed on every run)
  --no-save                     skip writing the file
  --allow-anchor-mismatch       downgrade anchor-mismatch from error to warning
```

The scenario seed is independent of the per-appliance seeds; it only affects the synthesized voltage, frequency, and voltage-harmonic content. Different scenario seeds applied to the same per-appliance files give different (but valid) realizations of the same load profile.

---

## 9. Validation invariants

The aggregator prints a summary on every run that includes the **conservation invariant**:

```
max | sum over a of P_contribution[t, a]  -  P_total[t] |   <   1e-5 x max |P_total|
```

In words: the sum of per-appliance contributions must equal the aggregate P_total at every sample, within float32 precision. Anything more than 1e-5 of peak power indicates a phase-distribution bug and the summary prints a WARNING.

Other sanity checks worth examining in the summary output:

- **Per-phase RMS I should be vaguely similar** across L1/L2/L3 if appliances are distributed across phases. One phase being 10x larger than the others indicates all single-phase appliances ended up on one phase.
- **THD_I should sit in the 2-8% range** during periods of normal load. Below 1% suggests harmonics are not being aggregated correctly; above 15% suggests excessive distortion is being generated.
- **Neutral I mean should track imbalance.** Zero indicates a perfectly balanced load (rare in residential settings).

---

## 10. mix_measured_scenarios.py: real recordings into training scenarios

`Scripts/Aggregator/mix_measured_scenarios.py` turns the real single-device PAC4200 recordings (made with `pac_reader.py`, see `05_pac4200_reader.md`) into aggregate **scenario files with ground truth**, by reusing the synthetic aggregator unchanged. This is how the measured-data training corpus for the MS2 pipeline is built.

### 10.1 Why an adapter is needed

`aggregator.py` consumes per-appliance files in the `Appliance_generator.py` layout (`measurements/P`, `measurements/Q`, `harmonics_I_mag`, `harmonics_I_phase`, `ground_truth/state`, JSON `appliance_metadata`), all sharing one length, rate and anchor. The PAC4200 recorder writes a different layout (`P_total`, per-phase `P_L1..3`, `harmonics/I_mag_L1`, `appliance_label`), and every recording has a different length and anchor. The adapter bridges the gap per recording:

- `P <- P_total`, `Q <- Q_total` (single-appliance totals; the values are real).
- `harmonics_I_mag / _phase <- harmonics/I_mag_L1 / I_phase_L1` (the live phase); missing harmonic groups become zeros.
- `ground_truth/state <- "on"` where `|P| > 3 W`, else `"off"`.
- `appliance_metadata <- {name=<family>, phase="L1", is_three_phase=False, source_label=<original label>}`. All lab recordings were made on L1 and are kept on L1.
- Every recording is **tiled (looped)** along time to one common length `N = duration x rate` and given an identical timestamp axis and shared anchor (2026-06-10 12:00 UTC), so the aggregator's alignment check passes.
- The rewritten files carry `format_version "0.2"`, `generator_version "pac_adapter_0.1"`, `tier "measured_single"`; they are written to a temporary directory and deleted after aggregation.

### 10.2 Scenario construction

1. **Scan and filter.** All `*.h5` in `--recordings` are loaded. Recordings whose label contains a double underscore (`a__b`, several devices at once) are skipped: they have no per-device ground truth and serve as real test inputs for inference instead. Recordings shorter than 25 samples (~5 s at 5 Hz) are skipped. Recordings that never draw power (max |P| < 1 W, e.g. the PV files with no captured generation) are kept but flagged: they can only teach "always off".
2. **Family grouping.** Labels collapse to appliance families via `nilm_pipeline.parse_family` (single source of truth shared with the MS2 pipeline), so `standing_fan_high_no_rotation` and `standing_fan_low_rotation` are the same `standing_fan` family. One variant per family is drawn per scenario.
3. **Composition planning with coverage guarantee.** Each of the `--n-scenarios` scenarios mixes a random subset of `--min-app` to `--max-app` families. Plans are then adjusted so every family appears in at least `min(4, n_scenarios)` scenarios; otherwise a grouped train/test split could leave a device only in the held-out set (never learned) or only in training (untestable).
4. **Random ON/OFF schedules.** Unless `--no-schedule` is given, each appliance's tiled trace is multiplied by a random usage schedule: alternating ON blocks of 30-120 s and OFF blocks of 15-90 s (60% chance of starting ON, and at least ~45 s ON is guaranteed). Without this, every appliance would be ON for the whole looped scenario and the model would learn "everything is always on", exactly the false-alarm failure the real mixed recordings exposed.
5. **Aggregate and write.** The adapter files are fed through `aggregator.aggregate` (fresh random scenario seed each time) and `aggregator.write_scenario(tier="measured")`. Voltage, frequency and voltage harmonics are synthesized as for synthetic scenarios.

Caveat inherited from the recordings: the per-order current harmonics in the existing lab recordings are all zero (the FC 0x14 file numbers were not verified at recording time), so mixed THD_I is ~0. P, Q and per-phase power are real.

### 10.3 CLI

```
python mix_measured_scenarios.py [options]

--recordings DIR      dir of PAC4200 single-appliance recordings
                      (default ../PAC4200_reader/recordings, relative to the script)
--out DIR             output dir (default measured_scenarios next to the script)
--n-scenarios INT     default 6
--duration SECONDS    common scenario length; each recording is looped to fit (default 300)
--rate HZ             default 5
--min-app INT         min appliances per scenario (default 2)
--max-app INT         max appliances per scenario (default 4)
--seed INT            default 0
--exclude GLOB        skip recordings whose filename matches (e.g. 'test_*'
                      to hold test recordings out of training)
--no-schedule         keep every appliance ON for the whole scenario
--plot                also save a per-scenario decomposition PNG (needs matplotlib)
```

### 10.4 Outputs

- `measured_scenario_NN.h5` : one scenario per plan, layout (b), `tier="measured"`.
- `measured_scenario_NN_decomposition.png` : optional stacked-area plot of per-appliance ground truth vs the aggregate (`--plot`).
- `manifest.json` : per scenario: filename, appliance families, duration, sample count, peak P_total.

### 10.5 Coupling note

The script imports `aggregator.py` from its own directory (override with the `AGG_DIR` environment variable) and `parse_family` / `is_mixed_label` from `Scripts/MS2_Pipeline/nilm_pipeline.py` via a relative path (`../MS2_Pipeline`). Moving either folder breaks the import; keep `Aggregator`, `PAC4200_reader` and `MS2_Pipeline` as siblings under `Scripts/`, or set `AGG_DIR` accordingly.

---

## Appendix A: Mapping per-appliance channels to aggregate output

For reference when debugging which per-appliance file contributed what:

| Per-appliance channel | Aggregate destination | Transformation |
|---|---|---|
| `measurements/P` | `measurements/P_{phase}` and `P_total` | summed; respects `is_three_phase` |
| `measurements/Q` | `measurements/Q_{phase}` and `Q_total` | same |
| `measurements/harmonics_I_mag` + `..._I_phase` | `measurements/harmonics/I_{mag,phase}_{phase}` | complex sum, then re-decomposed |
| `ground_truth/state` | `ground_truth/state[:, appliance_idx]` | preserved as a column |
| `ground_truth/P_contribution` | `ground_truth/P_contribution[:, appliance_idx]` | preserved as a column |
| `metadata` attrs | `ground_truth.attrs/appliance_<i>_metadata` | JSON-encoded |
| (no per-appliance equivalent) | `measurements/V_*` | synthesized fresh per scenario |
| (no per-appliance equivalent) | `measurements/freq` | synthesized fresh per scenario |
| (no per-appliance equivalent) | `measurements/harmonics/V_*` | synthesized fresh per scenario |
| (no per-appliance equivalent) | `measurements/I_N` | derived from I_phase via 3-phase balance |
| (no per-appliance equivalent) | `measurements/S_*`, `PF_*`, `cosphi_*`, `THD_*` | derived from V, I, P, Q, harmonics |
