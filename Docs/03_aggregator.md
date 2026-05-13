# NILM Project — Scenario Aggregator Specification

**Version:** 0.1 (draft)    
**Milestone:** 1    
**Companion to:** `data_format.md`, `appliance_generator.md`    
**Owner:** Soheil Ayati, Marc Steffgen    
**Last updated:** 2026-05-12

---

## 1. Purpose and scope

The aggregator combines per-appliance HDF5 files (produced by `Appliance_generator.py`) into a single **scenario file** that mimics what a Siemens PAC4200 would log at the Point of Common Coupling.

The aggregator is a **synthetic-data-only step**. In a real deployment the PAC4200 already provides aggregated signals — the aggregator simulates that physics so the downstream preprocessing/ML pipeline is identical for real and synthetic data. If a real PAC4200 were wired up in the lab tomorrow, only the aggregator step would be replaced; preprocessing and Milestone 2 stay the same.

**In scope:** combining per-appliance P, Q, and harmonic contributions into PCC-level aggregate signals; synthesizing voltage, frequency, and voltage harmonics; deriving secondary quantities (PF, cos φ, THD, S, I_N); writing the scenario file with both `/measurements` (PAC4200 view) and `/ground_truth` (per-appliance breakdown).

**Out of scope:** noise injection, Modbus dropout simulation, quantization (these live in the preprocessing step's "Job A — corrupt the clean signal"); cleaning of corrupted signals (preprocessing's "Job B — clean it up"); feature extraction (Milestone 2).

---

## 2. Position in the pipeline

```
                ┌─────────────────────────────┐
                │  per-appliance generators   │   (clean, deterministic
                │  Appliance_generator.py     │    per-appliance traces)
                └──────────────┬──────────────┘
                               │  N × .h5 files
                               ▼
                ┌─────────────────────────────┐
                │   AGGREGATOR (this doc)     │   ◄── synthetic-only;
                │   aggregator.py             │       simulates physics
                └──────────────┬──────────────┘       at the PCC
                               │  scenario_*.h5
                               ▼
                ┌─────────────────────────────┐
                │   preprocessing (next)      │   Job A: corrupt to
                │                             │   match real PAC4200.
                │                             │   Job B: clean.
                └──────────────┬──────────────┘
                               │  /preprocessed in same file
                               ▼
                ┌─────────────────────────────┐
                │   Milestone 2 ML            │
                └─────────────────────────────┘
```

The split exists because per-appliance ground truth has to be tracked separately for ML supervision (it's the entire reason the synthetic-data approach gives an advantage over real datasets like UK-DALE).

---

## 3. Inputs and outputs

### 3.1 Inputs

A list of per-appliance HDF5 files, all of which must satisfy:

- Same `sample_rate_hz` (typically 5 Hz).
- Same number of samples (typically 432 000 for 24 h).
- Same `anchor_datetime` — otherwise PV solar position is computed for a different date than other appliances see, which is physically wrong.

The aggregator validates these on load and errors out on mismatch by default. The CLI flag `--allow-anchor-mismatch` downgrades the anchor check to a warning, but is almost never the right choice.

### 3.2 Output

A single scenario HDF5 file containing three top-level groups:

- `/measurements` — what a PAC4200 would log at the PCC. The single observable signal that downstream NILM operates on.
- `/ground_truth` — the per-appliance breakdown (states + per-appliance P, Q contributions). Only available because we control generation.
- `/metadata` — scenario-level attributes (tier, seed, anchor datetime, format version, etc.).

Plus the top-level `/timestamp` dataset shared across all channels.

The output file structure intentionally matches the layout in `data_format.md` §10.2, with the exception that `/preprocessed` is empty at this stage (populated by the preprocessing step).

---

## 4. Aggregation math

### 4.1 Power per phase

Each appliance contributes to one or all three phases depending on its `is_three_phase` flag:

- **Single-phase appliance** (fridge, PC, hair dryer, etc.): contributes its full P and Q to the phase recorded in its metadata (`L1`, `L2`, or `L3`).
- **3-phase appliance** (PV, synchronous machine, fast-AC EV): distributes P and Q equally across L1, L2, L3 (each phase gets one-third).

```
P_phase[p] = Σ P_appliance_i  where appliance_i is on phase p (or distributes to p)
Q_phase[p] = same for Q
P_total    = P_L1 + P_L2 + P_L3
Q_total    = Q_L1 + Q_L2 + Q_L3
```

**Why equal distribution for 3-phase appliances:** in a balanced 3-phase appliance (PV inverter, generator), instantaneous power is shared symmetrically across phases. For modeling at this synthetic level we assume perfect symmetry; real installations have small imbalances (~1%) that we don't capture.

### 4.2 Voltage synthesis

PAC4200 measures voltage at the PCC; in real installations voltage is largely set by the grid, not the load (the "infinite/stiff bus" assumption). The aggregator therefore synthesizes voltage independently of the aggregated load:

- Nominal 230 V phase-to-neutral on each phase.
- Slow random walk (~30 s correlation time, ±2 V envelope) — represents normal grid voltage drift.
- Per-sample Gaussian jitter (σ = 0.05 V) — represents fast variation.

Each phase has an independent realization, so they're not correlated to each other. Real installations show small correlated drift; our model neglects this.

**Why this matters for NILM:** voltage is mostly context (not directly used for disaggregation), but it's required to derive currents from powers (I = S / V) and to compute true power factor (PF = P / S where S = V × I_rms).

### 4.3 Frequency synthesis

Nominal 50 Hz with a slow random walk (~60 s correlation, ±0.05 Hz envelope). Real grid frequency varies similarly under normal operation.

Frequency is a context channel; it's not used for NILM disaggregation directly but is useful for detecting grid-stability events and PV-interaction analysis (angle 3).

### 4.4 Current harmonics — complex vector sum

This is the only mathematically non-trivial step. The same harmonic order from two different appliances does not simply add in magnitude — the phases interact:

```
I_3rd_total = |I_3rd_A| · exp(j · φ_3rd_A)  +  |I_3rd_B| · exp(j · φ_3rd_B)
|I_3rd_total| = |this complex sum|
```

If the two contributions are in phase (φ_A ≈ φ_B), they add constructively; if 180° out of phase, they cancel. This is the **harmonic interaction effect** that makes harmonic-based NILM disaggregation genuinely hard when multiple appliances run simultaneously — and exactly what the angle-4 (multi-feature fusion) approach has to deal with.

Per phase, for each harmonic order n ∈ [2, 40]:

```
complex_n[p] = Σ_a   (h_mag_a[n] · exp(j · h_phase_a[n]))
                     × (1 if a is on phase p else 1/3 if a is 3-phase else 0)

|h_n[p]|     = abs(complex_n[p])
arg(h_n[p]) = angle(complex_n[p])
```

Magnitude-only summation would be a noticeable physical error and would mask interaction effects that real PAC4200 measurements show.

### 4.5 Voltage harmonics

For a true infinite bus, voltage harmonics would be zero. Real installations have small voltage harmonics because grid impedance × load current harmonics produces a voltage drop with harmonic content. We approximate this with small fixed fractions of nominal voltage:

| Order | Fraction of V_nominal |
|---|---|
| 3rd | 0.5% |
| 5th | 1.5% |
| 7th | 1.0% |
| 11th | 0.5% |
| 13th | 0.3% |

Each order has a fixed phase angle per scenario plus small jitter. This is realistic for a stiff residential or industrial grid; THD_V will land around 2% on each phase, which matches typical EN 50160 measurements.

### 4.6 Per-phase current

Per-phase RMS current is computed from the fundamental + harmonics in quadrature, which is what the PAC4200 reports as true-RMS current:

```
I_fund[p]    = sqrt(P_phase[p]² + Q_phase[p]²) / V_phase[p]
I_h_rms[p]  = sqrt(Σ_n |h_n[p]|²)
I_rms[p]    = sqrt(I_fund[p]² + I_h_rms[p]²)
```

I_rms[p] is what gets stored as `I_L1`, `I_L2`, `I_L3`.

### 4.7 Neutral current

For a 3-phase 4-wire system with single-phase loads, the neutral carries the residual unbalance:

```
I_N = | I_L1 · 1 + I_L2 · a² + I_L3 · a |     where a = exp(j · 120°)
```

This evaluates to zero when the three phase currents are equal (balanced load) and grows with imbalance.

**Approximation in this aggregator:** the formula uses each phase's RMS current as if it were a positive real value at the reference phase angle. The exact PAC4200 calculation uses the instantaneous phase angles of each phase current at every sample — we approximate this with the imbalance-only contribution, which is correct in magnitude but loses harmonic-frequency neutral content (zero-sequence triplen harmonics). For this project's purposes the approximation is acceptable.

### 4.8 Derived secondary quantities

All derived per phase + as totals:

| Channel | Formula | Notes |
|---|---|---|
| `S` | V_phase × I_rms_phase | apparent power |
| `PF` (true) | P / S | includes distortion |
| `cos φ` (displacement) | P / sqrt(P² + Q²) | fundamental only |
| `THD_I` | 100 · I_h_rms / I_fund | percentage |
| `THD_V` | 100 · V_h_rms / V_fund | percentage |

Division is guarded — where the denominator is below threshold (P below 1 W, current below 0.01 A) the output is set to a sentinel (1.0 for PF/cos φ; 0 for THD). This avoids both NaN and division warnings.

---

## 5. Sign and phase conventions

Inherited from `data_format.md` §6:

- **Active power P:** positive = consumption from grid; negative = generation (PV active hours, synchronous machine in generator mode).
- **Reactive power Q:** positive = inductive (lagging); negative = capacitive (leading). The synchronous machine can produce all four quadrants.
- **Apparent power S:** always non-negative (magnitude).
- **Power factor PF:** in [0, 1], unsigned.
- **cos φ:** in [-1, 1], signed (sign indicates leading/lagging).
- **I_N:** non-negative (magnitude).

For single-phase appliances, P/Q on the un-assigned phases is zero.

---

## 6. Tier handling

The aggregator accepts `--tier {train | easy | normal | hard | adversarial}`, which gets recorded in the scenario file's `/metadata/tier` attribute.

The tier label does not change what the aggregator computes — it's a label for downstream filtering. The *content* difference between tiers comes from how the per-appliance files were generated before aggregation:

| Tier | Generation strategy | Expected M2 performance |
|---|---|---|
| `train` | Realistic mixed activations; varied seeds for instance diversity | training data; not evaluated |
| `easy` | Sparse non-overlapping events; widely separated power magnitudes | F1 > 0.95 |
| `normal` | Realistic everyday operation; some natural overlap | F1 ~ 0.85 |
| `hard` | Concurrent events; similar-power overlaps; smart EV; cloudy PV | F1 ~ 0.65–0.80 |
| `adversarial` | Net-zero scenarios; signature collisions; PV cancelling load | F1 ~ 0.50 — deliberately stress-tests |

The full tier strategy is documented separately and orchestrated by a per-tier generation wrapper script. The aggregator itself is tier-agnostic.

---

## 7. File format

Output structure follows `data_format.md` §10.2:

```
scenario_*.h5
├── /timestamp                              (1D int64, µs since Unix epoch)
├── /measurements
│   ├── V_L1, V_L2, V_L3                    (1D float32)
│   ├── I_L1, I_L2, I_L3, I_N               (1D float32)
│   ├── P_L1..P_total, Q_L1..Q_total        (1D float32)
│   ├── S_L1..S_total                       (1D float32)
│   ├── PF_L1..PF_total, cosphi_*           (1D float32)
│   ├── THD_V_L1..3, THD_I_L1..3            (1D float32)
│   ├── freq                                (1D float32)
│   └── harmonics/
│       ├── I_mag_{L1,L2,L3}                (2D float32, N × 39)
│       ├── I_phase_{L1,L2,L3}              (2D float32, N × 39)
│       ├── V_mag_{L1,L2,L3}                (2D float32, N × 39)
│       └── V_phase_{L1,L2,L3}              (2D float32, N × 39)
├── /ground_truth
│   ├── appliance_names                     (1D bytes, N_app entries)
│   ├── P_contribution                      (2D float32, N × N_app)
│   ├── Q_contribution                      (2D float32, N × N_app)
│   ├── state                               (2D bytes, N × N_app)
│   └── appliance_<i>_metadata              (attributes, JSON-encoded)
└── /metadata
    ├── format_version, aggregator_version  (string attrs)
    ├── sample_rate_hz                      (float attr)
    ├── anchor_datetime                     (ISO 8601 string attr)
    ├── tier                                (string attr)
    ├── scenario_seed                       (int attr)
    ├── n_appliances, n_samples             (int attrs)
    └── duration_seconds                    (float attr)
```

- Compression: LZF on all datasets (fast decompression, suitable for ML training loops).
- Float precision: float32 throughout (PAC4200 class 0.2 accuracy fits easily in float32).

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
  --inspect                     print summary
  --no-save                     skip writing the file
  --allow-anchor-mismatch       downgrade anchor-mismatch from error to warning
```

The scenario seed is independent of the per-appliance seeds — it only affects the synthesized voltage, frequency, and voltage-harmonic content. Different scenario seeds applied to the same per-appliance files give different (but valid) realizations of the same load profile.

---

## 9. Validation invariants

The aggregator prints a summary on every run that includes the **conservation invariant**:

```
max | Σ P_contribution[t, a]  -  P_total[t] |   <   1e-5 × max |P_total|
```

Expressed in words: the sum of per-appliance contributions must equal the aggregate P_total at every sample, within float32 precision. A residual on the order of millivolts (or in this context, ~10⁻³ W on an 11 kW signal) is normal; anything more than 10⁻⁵ × peak power indicates a phase-distribution bug.

Other sanity checks that should be examined in the summary output:

- **`Per-phase RMS I` should be vaguely similar** across L1/L2/L3 if appliances are distributed across phases. One phase being 10× larger than the others indicates all single-phase appliances ended up on one phase (unbalanced).
- **`THD_I` should sit in 2–8% range** during periods of normal load. Below 1% suggests harmonics aren't being aggregated correctly; above 15% suggests something is generating excessive distortion.
- **`Neutral I mean` should track imbalance.** Zero indicates perfectly balanced load (rare in residential); 50+ A would indicate severe imbalance worth flagging.

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