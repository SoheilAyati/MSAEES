# NILM Project - Design Rationale and Feature Justification

**Version:** 1.0
**Milestone:** 1, 2, and the live extension (paper-writing companion)
**Companion to:** all other documents in `Docs/`; this one collects the *why* behind every load-bearing decision in one place, with the evidence and the literature anchors, so the scientific paper can be written from it directly.
**Owners:** Soheil Ayati, Marc Steffgen
**Last updated:** 2026-07-06

---

## 1. Project thesis

NILM (Non-Intrusive Load Monitoring, Hart 1992 [1]) recovers per-appliance information from one aggregate measurement at the Point of Common Coupling (PCC). Beyond the basic pipeline, this project deliberately targets two challenges that separate good NILM systems from passing demonstrations:

1. **PV-aware disaggregation.** Behind-the-meter generation can pull the aggregate negative (the *signal eclipse* [2]). We address it with signed power channels end to end, a PV generator model with an inverter-specific harmonic signature, and four-quadrant support (the synchronous machine model exercises all four P/Q quadrants).
2. **Multi-feature fusion.** Different appliances are most distinguishable in different feature domains [3]: steady-state (P, Q, PF), harmonic (per-order current spectra, THD), and transient (switching steps). We record and fuse all three rather than choosing one.

Everything below is a decision in service of these two goals plus one architectural rule: **the same downstream code must run on synthetic and real data.** Only the acquisition step differs (synthetic aggregator vs live PAC4200 reader); preprocessing, features, training, and inference are source-agnostic.

---

## 2. Acquisition decisions

### 2.1 Why a Siemens PAC4200, and why 5 Hz

The PAC4200 is the meter installed in the lab, accuracy class 0.2, with Modbus TCP access. It samples internally at 8.5 kHz but does **not** stream raw waveform over Modbus; it exposes processed values (RMS, P, Q, PF, THD, per-order harmonics). A full poll of the verified register block takes about 200 ms in practice, so **5 Hz is the rate the meter can actually sustain**, not an arbitrary choice. It is also sufficient: appliance behaviour lives on second-to-minute scales, and at 5 Hz an inrush transient still registers as 1-2 elevated samples and a multi-state cycle (washing machine agitation at roughly 10 s on / 10 s off) is fully resolved.

Consequence for method choice: with no raw waveform, high-frequency NILM techniques (V-I trajectories, sub-cycle transients, EMI signatures) are out of reach by construction. The feature space is *low-frequency steady-state + harmonic magnitudes + 5 Hz step transients*, and the whole feature design (section 5) follows from that constraint.

### 2.2 Why harmonics at all

Active power alone cannot separate devices of similar wattage. The clearest evidence is our own pre-measured fingerprint table (real PAC4200, six devices, `Pre_Measured/*.csv`):

| Device | P (W) | Q (var) | PF | THD_I (%) | Character |
|---|---|---|---|---|---|
| Toaster | 1400 | ~4 | 1.00 | 2 | pure resistive |
| Hair dryer (stage 1) | 139 | -9 | 0.90 | 47 | heater + small motor |
| Fluorescent tube | 39 | 78 | 0.45 | 10 | inductive ballast |
| LED lamp | 6.7 | -5 | 0.74 | 48 | capacitive driver |
| USB charger (+phone) | 10 | -0.7 | 0.54 | 151 | switch-mode, very distorted |
| Mixer | 17 | 8 | 0.36 | 93 | universal motor |

The LED lamp (6.7 W) and the USB charger (10 W) are indistinguishable in P; THD_I separates them by a factor of three (48 % vs 151 %). The fluorescent tube and the mixer sit at low PF for *different physical reasons* (displacement vs distortion), which only the harmonic channels resolve. This is the project's multi-feature-fusion thesis demonstrated on real hardware, and it matches the literature: odd low-order harmonic currents are established NILM discriminators [3][5].

Physics behind it: rectifier front-ends (SMPS in PCs, LED drivers, chargers) draw current in narrow peaks around the voltage crest, producing strong odd harmonics (3rd, 5th, 7th); motors add slot harmonics (5th, 7th); resistive heaters draw sinusoidal current and are nearly clean. The harmonic spectrum is therefore a *load-technology fingerprint* that survives when wattage overlaps.

### 2.3 Why orders 2 through 40, magnitudes and phases

- **39 orders (2..40):** the PAC4200 exposes up to the 64th order, but NILM-relevant content concentrates in the low odd orders; order 40 at 50 Hz is already 2 kHz, far above where appliance signatures carry energy. 39 orders keep generous headroom for spectral-shape features (centroid, energy) while capping the storage cost of the harmonic arrays, which dominate the file size (see `01_data_format.md` Appendix A).
  **Correction note for the paper:** the Milestone 1 report stated truncation at order 15; the as-built code stores orders 2..40 (`N_HARMONICS = 39` in generator, aggregator, and reader). The code and `01_data_format.md` are authoritative.
- **Phases are synthesized but not measurable.** Synthetic files carry per-order phase because the aggregation math needs it (section 3.2), and because inverter phase clustering is a physically meaningful PV signature. The real PAC4200 provides **magnitudes only** (Modbus FC 0x14 file records). Recordings keep zero-filled phase arrays for shape compatibility and set `harmonic_phase_captured = False` so downstream code cannot mistake them for measured zeros. Harmonic-phase features are therefore synthetic-only, and no model intended for the real meter uses them.
- **Harmonics at the full 5 Hz rate:** stored at the same rate as P and Q so switching transients are temporally aligned across all channels; a slower harmonic rate would break window-level feature fusion.

### 2.4 Why HDF5, float32, LZF, microsecond UTC timestamps

| Decision | Reason |
|---|---|
| HDF5 | one self-describing file holds measurements + per-appliance ground truth + metadata; mixed dtypes; built-in compression; resizable datasets for the incremental live recorder. Parquet has no native hierarchy (ground truth would need a sidecar); CSV has no types, nesting, or compression. |
| float32 | the meter is accuracy class 0.2 (about 3 significant digits); float32 carries about 7, so float64 would only double storage. |
| LZF compression | fast decompression matters more than ratio for ML loops that re-read files often. |
| int64 microseconds, UTC | Modbus reply latency jitters at the millisecond scale, so second resolution is too coarse; UTC avoids DST ambiguity in 24 h scenarios. |

### 2.5 The recorded channel inventory: what we record and why

Every channel in the scenario layout exists for a stated consumer. This is the answer to "do we record all we need, and why these":

| Channel group | Why recorded | What consumes it |
|---|---|---|
| `P_L1..L3, P_total` (signed) | the headline NILM signal; signed so PV export is representable (thesis 1) | all tasks; event features; edge detector |
| `Q_L1..L3, Q_total` (signed) | separates inductive / capacitive / resistive technology at equal wattage; second axis of the classical Hart (P, Q) plane | identification and aggregate features; edge matching (dP, dQ) |
| `S_*`, `PF_*` | apparent power and true power factor (includes distortion); PF is a magnitude-free shape descriptor | `S_mean`, `PF_mean` features |
| `cosphi_*` | displacement factor (fundamental only); the *difference* between PF and cos phi isolates distortion from phase shift | analysis; not yet a model feature (real meter does not expose it in the verified block) |
| `THD_I_L1..L3` | scalar distortion summary; the cheapest harmonic feature and the one that transfers to simple meters | `THD_I_mean` / `THDI_mean` features |
| `THD_V_*`, `V_L1..L3`, `freq` | context and plausibility channels: voltage is needed to derive current from power (I = S/V), frequency and THD_V validate grid conditions; neither is an appliance signature on a stiff grid | preprocessing bounds checks, aggregation math, commissioning |
| `I_L1..L3`, `I_N` | true-RMS currents; I_N grows with single-phase imbalance, separating three-phase from single-phase loads | preprocessor imbalance features |
| `harmonics/I_mag_*` (39 orders) | the load-technology fingerprint (section 2.2) | `h3, h5, h7, h_centroid, h_energy` features; live THD derivation |
| `harmonics/I_phase_*` | required for physically correct aggregation (complex summation); PV inverter phase clustering | aggregator; synthetic-only features |
| `harmonics/V_*` | realism of the synthetic grid model (about 2 % THD_V per EN 50160); kept for completeness on recordings | none directly |
| `/ground_truth` (names, P/Q contribution, state) | per-sample supervision; the entire reason for the synthetic-first strategy | training targets, evaluation |

Real-meter deltas (verified against the live device): the PAC4200 core block has no `I_N`, no `cosphi_*`, no per-phase `THD_I_*` (derived from the FC 0x14 spectrum instead), and THD_V is line-to-line. Any model meant to run on the real meter restricts itself to the common channel subset; this *feature gap* is a first-class design constraint (section 5.1).

---

## 3. Why synthetic-first, and why the aggregation physics matters

### 3.1 Synthetic data with embedded ground truth

Public datasets (UK-DALE, REDD) provide aggregates, but per-appliance ground truth requires intrusive sub-metering, is fixed at whatever appliance set was recorded, and offers no control over difficulty. Generating the data ourselves gives:

- **Per-sample supervision** (`P_contribution` per appliance) rather than event labels, enabling direct regression training for disaggregation.
- **Instance diversity on demand:** every generator parameter given as `[lo, hi]` is re-sampled per seed, so different seeds are different appliance *instances*. This is what makes the honest evaluation protocol (train on some instances, test on held-out ones, section 7) possible at all.
- **Difficulty control:** tiers from easy (sparse, non-overlapping) to adversarial (net-zero, PV cancelling load) that no real dataset offers.
- **PV at will:** the signal-eclipse case can be made common instead of waiting for sunny days.

The generators produce *clean* signatures by design; noise, dropout, and quantization belong to the acquisition layer, so the preprocessor learns to handle them where they actually occur (real recordings) instead of chasing invented corruption parameters.

### 3.2 Why the aggregator sums harmonics as complex vectors

The same harmonic order from two appliances does not add in magnitude; phases interact, and contributions can cancel. Magnitude-only summation would systematically overestimate aggregate harmonics and hide exactly the interaction effect that makes harmonic NILM hard with concurrent loads. The complex vector sum per order and phase is the one mathematically non-trivial piece of the aggregator, and it is what makes the synthetic training distribution honest about that difficulty. (Full math: `03_aggregator.md` section 4.4.)

### 3.3 Semi-synthetic measured mixes: the bridge to the real lab

Real multi-device ground truth would need one meter per device. Instead, `mix_measured_scenarios.py` composes *real single-device recordings* into aggregate scenarios with exact ground truth by reusing the Milestone 1 aggregator unchanged: same physics, same file format, tier `measured`. Two of its choices carry method weight:

- **Random ON/OFF schedules per appliance.** Without them every looped recording is ON for the whole scenario and the model learns "everything is always on"; the schedules create the OFF windows a presence classifier needs. This failure mode was actually observed before the fix, which is worth reporting.
- **Coverage guarantee.** Composition plans are adjusted so every device family appears in at least `min(4, n_scenarios)` scenarios; otherwise a grouped train/test split can leave a device only in the held-out set (never learned) or only in training (untestable).

This mirrors the established practice of synthesizing aggregates from sub-metered real data [6], with the difference that our mixer reuses the identical physical aggregation code path as the synthetic pipeline.

---

## 4. Preprocessing decisions (summary; details in `04_preprocessor.md`)

- **Universal, source-agnostic:** one script for synthetic and real files; for clean synthetic data it is effectively feature engineering, for real data it also repairs gaps and outliers. Same code path is the architectural payoff of Milestone 1.
- **Gap imputation threshold 5 samples (1 s):** real Modbus dropouts are short bursts; appliance states change on second-to-minute scales, so interpolating under 1 s cannot mask a real event; anything longer indicates connectivity loss and must not be silently invented.
- **Clip outliers to physical bounds rather than flag:** NaN poisons every rolling-window feature it touches; clipping preserves usable context. Bounds are generous (EV at 22 kW, PV export, four-quadrant machine all inside), so a clip means sensor fault, not unusual load.
- **Originals never modified:** `/preprocessed` is added next to `/measurements`; re-running replaces only itself. The embedded JSON report makes every cleaning action auditable, which is what lets a paper claim data provenance.

---

## 5. Feature design: the core of the method

Three feature sets exist, one per model input type. The guiding constraints: (a) 5 Hz processed values only (section 2.1), (b) real-meter transferability (the *common* subset must exist on a PAC4200 CSV), (c) few, physically interpretable columns, because training sets are small (hundreds to a few thousand windows) and the live retrain loop must finish in under two minutes (section 8).

### 5.1 Identification features (single-device windows)

`FEATURES_COMMON` (9 columns) works on every source including the early CSV runs; `FEATURES_HARM` (5 more) needs per-order harmonics. Steady-state values are the **median over the middle 20-80 % of the window**, so switch-on/off transients at the edges do not pollute the steady-state summary; the min/max/std columns then capture those dynamics deliberately, on their own terms.

| Feature | Physical meaning | What it separates | Why included |
|---|---|---|---|
| `P_mean` | steady-state active power | the primary magnitude axis (Hart [1]) | strongest single discriminator when wattages differ |
| `Q_mean` | steady-state reactive power, signed | motors/ballasts (+Q) vs electronics (-Q) vs heaters (~0) | second axis of the classical (P, Q) signature plane |
| `S_mean` | apparent power | total current draw incl. distortion | lets the model see S without deriving it |
| `PF_mean` | true power factor | fluorescent 0.45 vs toaster 1.00 at any wattage | magnitude-free shape descriptor; survives instance-to-instance power variation |
| `QP_ratio` | tan phi (load angle) | same as PF but unbounded and signed | scale-free; robust when P is small and PF is ill-conditioned |
| `THD_I_mean` | current distortion (%) | LED 48 % vs USB charger 151 % at ~same watts | the cheapest harmonic feature; available (derived) on the real meter |
| `P_std` | within-window variability | compressor cycling / agitation vs constant loads | temporal texture that medians erase |
| `P_min`, `P_max` | window envelope | inrush peaks, burst floors, PV eclipse depth | catches events shorter than the window mean can see |
| `h3, h5, h7` | 3rd/5th/7th harmonic current magnitude (A) | SMPS (strong 3rd) vs motor (5th/7th slot) vs resistive (clean) | the canonical NILM orders [5]; lowest orders carry the most appliance energy |
| `h_centroid` | spectral centre of mass over orders 2..40 | fast-decaying vs flat spectra in one scalar | compresses the 39-bin shape; top-ranked feature in our baseline importance analysis |
| `h_energy` | root-sum-square of all 39 magnitudes | overall distortion in amps (not %) | complements THD (which is relative to the fundamental) |

**Why not all 39 harmonic orders as raw features?** Three reasons. Dimensionality: identification currently trains on 171 active windows across 7 device families; 39 extra correlated columns invite overfitting where 5 summaries do not. Physics: appliance harmonic energy concentrates in the low odd orders; high orders sit near the meter's resolution floor and add noise. Transferability and interpretability: the summary features have physical names a paper can discuss, and the measured baseline confirms they carry the signal (top importances: `h_centroid`, `PF_mean`, `QP_ratio`, `h7`, `h5`; macro-F1 0.998 on clean synthetic single-appliance windows). The full spectrum remains stored in every file, so richer spectral models remain possible without re-recording.

**THD fallback rule:** THD_I is used as measured when the channel exists, and otherwise derived from the stored harmonic spectrum relative to the fundamental current estimated at 230 V. This single rule keeps the feature comparable across synthetic scenarios (measured THD channel), real recordings (derived from the FC 0x14 spectrum), and CSV runs (measured), which is what makes cross-source transfer experiments valid.

### 5.2 Aggregate features (disaggregation / presence / mix)

`AGG_FEATURES`, 17 columns per non-overlapping window. The list is **append-only**: a model bundle stores the feature list it was trained with, and inference slices any newer, wider matrix down to that length, so old models keep working as the feature set grows. That contract is why order matters and is documented in code.

| # | Feature | Why it is in the set |
|---|---|---|
| 1-4 | `Ptot_mean, Ptot_std, Ptot_min, Ptot_max` | level, activity, and envelope of the aggregate; min/max catch brief events and PV eclipse depth inside the window |
| 5 | `Qtot_mean` | reactive composition of the running mix; changes when a motor joins even if P barely moves |
| 6-8 | `PL1_mean, PL2_mean, PL3_mean` | **phase localisation**: a single-phase appliance moves exactly one phase, a balanced three-phase appliance moves all three equally; this narrows the candidate set before any signature matching |
| 9 | `PF_mean` | distortion/displacement share of the mix |
| 10 | `THDI_mean` | rises when nonlinear loads are active; separates "1500 W of heater" from "1500 W of heater + PC" |
| 11 | `hour` | time-of-day prior; synthetic appliances have diurnal usage models (EV overnight, PV solar bell, PC working hours). *Honest caveat:* computed from file start, so it is a true clock hour only for the 24 h midnight-anchored synthetic scenarios; on 300 s measured scenarios it is effectively constant and carries no information. Kept for the synthetic corpus, harmless elsewhere. |
| 12 | `Qtot_std` | reactive dynamics (compressor/agitator cycling) |
| 13 | `QP_ratio` | aggregate load angle, scale-free |
| 14 | `Stot_mean` | apparent magnitude including distortion current |
| 15-17 | `Pstep_max, Qstep_at_Pstep, n_steps` | **event features** (added 2026-07-06): the largest settled power step inside the window (pre/post medians around the sharpest sample-to-sample change), the reactive step at the same instant, and the step count. Rationale: steady-state sums cannot tell "boiler + lamp" from "boiler drawing more", but the switch-on step is the joining device's (dP, dQ) signature, i.e. Hart's edge model [1] embedded as window features. Settled medians rather than raw diffs so inrush spikes do not masquerade as load steps. |

This set operationalises multi-feature fusion at the aggregate level: steady-state (1-5, 9, 13, 14), harmonic (10), contextual (6-8, 11), transient (12, 15-17).

### 5.3 Raw sequences (neural path)

The MLP path deliberately does **not** use the features above: it flattens the raw `[P, Q, THD_I]` samples of each window (3 x window-samples) and learns from waveform shape. This gives the classical-vs-neural comparison a clean interpretation: same windows, same targets, hand-crafted summaries vs learned representation.

### 5.4 Preprocessor features (per-sample, MS1)

The 12 per-sample features (`dP_total`, rolling means/stds at 5 s and 30 s, THD summaries, imbalance ratios) are the MS1 groundwork aimed at event detection and steady-state matching; the MS2 window features above are computed independently from the cleaned channels. Full catalogue and per-feature purpose: `04_preprocessor.md` section 6. The two-window design (5 s / 30 s) exists so downstream methods can pick the temporal scale matching each appliance (fast switch vs fridge cycle).

### 5.5 Completeness check: do we record everything the method needs?

| Feature domain the literature uses | Our coverage | Gap / consequence |
|---|---|---|
| Steady-state P, Q, S, PF | full, per phase + total, signed | none |
| Harmonic magnitudes | orders 2..40 at 5 Hz, real meter verified (file 113 on L1) | L2/L3 file numbers unverified (all lab loads are on L1); re-run `tools/verify_harmonics.py` with load on those phases if needed |
| Harmonic phase | synthetic only | not measurable on this meter over Modbus; phase features excluded from transferable models by design |
| Transients / events | 5 Hz steps (`Pstep_max` etc., edge detector in live) | sub-cycle transients (true inrush waveform) impossible over Modbus; acknowledged limitation, not a missing recording |
| Contextual (time, phase) | hour + per-phase P | hour is file-relative (see 5.2) |
| V-I trajectory / waveform features | not available | needs raw waveform; out of scope for a Modbus meter |

Conclusion for the paper: within the physical envelope of a Modbus-connected class 0.2 panel meter, the recorded channel set is complete; every unrecorded quantity is unmeasurable on this hardware rather than overlooked.

---

## 6. Windowing decisions

| Parameter | Value | Reason |
|---|---|---|
| Window (synthetic scenarios) | 30 s | long enough for a stable steady-state median over the middle 60 %, short enough that most appliance states (fridge on-period 8-18 min, hair-dryer burst 2-6 min) span several windows; matches the preprocessor's long rolling window |
| Window (measured scenarios / live) | 10 s | measured scenarios are 300 s with ON blocks of 30-120 s; 30 s windows would give too few rows and smear short blocks; 10 s keeps tens of windows per scenario and 2-3 windows per ON block |
| Stride (identify, short CSV runs) | 5 s (window 10 s) | the 80 s CSV runs yield too few non-overlapping 30 s windows; overlap multiplies training rows without new recordings |
| Active threshold | 5 W | above meter noise and standby floors, below the smallest device of interest (LED lamp 6.7 W); windows below it train nothing |
| Presence ON threshold (`--on-w`) | 15 W synthetic, 5 W measured | synthetic default suppresses sub-state flutter; the measured fleet includes an 11 W table fan, so the threshold drops below it |

Rule of thumb documented for users: make the window about as long as the shortest event you care about; the window is stored in the model bundle so train and inference can never disagree.

---

## 7. Model family decisions

### 7.1 Why classical models first (Random Forest default)

- **Small-data regime.** Current training sets are 171 windows (identify) to about 2000 windows (mix). Tree ensembles on engineered features are the strongest, most stable choice at this scale; deep models would be fitting noise.
- **No scaling or tuning sensitivity.** RF is invariant to feature scale and monotone transforms, robust to the correlated features a physical feature set inevitably contains, and works out of the box with `class_weight="balanced"` for the inherent class imbalance (always-on baseload vs rare hair-dryer bursts).
- **Interpretability.** Feature importances connect results back to the physics story (harmonic and PF features ranking on top *is* a finding). This also serves the EU AI Act analysis (section 7.4).
- **Training speed is a functional requirement, not a convenience.** The live teach-and-retrain loop rebuilds scenarios and retrains mix + identify in 30-90 s (measured: 26 s in the validated run). That user experience is only possible because the models train in seconds. A deep model would break training-on-the-go.
- Multi-output tasks wrap the base estimator in `MultiOutputRegressor` / `MultiOutputClassifier` (one estimator per appliance): simple, parallel, and it lets per-appliance metrics fall out naturally.

### 7.2 Why LightGBM as the second family

Gradient-boosted trees are the consistently strongest family on tabular data, and LightGBM specifically is fast, memory-light, and familiar to the team from a previous load-forecasting project built under an interpretability constraint. It provides an independent second opinion on the same features: if RF and LGBM agree, the result is a property of the features, not the model. Measured baseline: RF 0.998 vs LGBM 0.994 macro-F1 on clean single-appliance windows, confirming feature quality rather than model choice drives performance.

**Why presence forces RF internally even when `--model lgbm` is passed:** an appliance that is always-on (baseload) or always-off in the training data yields a single-class target column; per-output LightGBM cannot fit a single-class output, RF handles it gracefully. This is an implementation-level robustness decision worth one sentence in the paper.

### 7.3 Why the neural path is an MLP (for now)

`deep_models.py` trains a scikit-learn MLP (hidden layers 256/128, scaled inputs) on raw `[P, Q, THD_I]` window waveforms. Choices: no PyTorch dependency (runs in every project environment), same windows and targets as the classical path (clean comparison), saved under separate filenames so classical and neural bundles coexist. It is explicitly the *neural baseline*; a CNN/LSTM (seq2point [4], neural NILM [6]) is the known next step and is expected to help exactly where the window models fail (washing-machine multi-minute cycles).

### 7.4 Regulatory framing (EU AI Act)

A lab NILM monitor is informational, not a safety component in the operation of critical infrastructure, so it falls outside the Act's high-risk category and deep learning is permissible; the classical baseline nevertheless hedges the classification question, since it is what we would be restricted to if the scope ever moved toward grid operation. Full analysis with sources: `06_milestone2_plan.md` section 2. GDPR (occupancy inference from load data) is a separate consideration and does not constrain model class.

### 7.5 Task structure: why four tasks and why `mix` exists

`identify` (single-device windows -> label), `disaggregate` (aggregate -> watts per appliance), `presence` (aggregate -> multi-label ON/OFF), and `mix` = presence + disaggregation trained on the same windows and saved as one bundle. `mix` exists because deployment needs both answers at once and gating the regressed watts by predicted presence (OFF -> 0 W) is both more accurate (gated MAE 2.9 W vs raw 3.3 W on the current model) and the honest error of what the dashboard actually displays. The **appliance vocabulary is dynamic**: derived from the training data (`scan_canon`) and stored in the bundle, which is what lets a newly taught device enter the model with zero code changes.

---

## 8. Evaluation methodology decisions

These choices are what make the reported numbers defensible in a paper:

1. **Grouped held-out splits.** When several instances exist (seeds, files), `GroupShuffleSplit` keeps whole instances apart, so scores measure generalisation to unseen appliance instances, not memorisation of rows from the same recording. With one instance per class it falls back to a stratified row split *and the metrics record which split was used* (the current identify metrics honestly say "stratified rows"; this is the known optimism to state alongside the 0.955).
2. **Honest multi-label macro-F1.** An appliance absent from the held-out data and never falsely predicted scores `null`, not 0, and is excluded from the macro average: nothing to detect plus nothing falsely detected is not a failure. Support counts are reported next to every F1.
3. **Gated MAE.** The end-to-end error of presence-gated power, i.e. of the number actually displayed, is reported alongside raw regression MAE.
4. **Label-set accuracy on real mixes.** Multi-device recordings (`a__b__c` filenames) have no per-sample ground truth but a known device set; inference parses the expected set from the label and reports set precision/recall/F1, misses, and false alarms. This turns every casually recorded mix into a legitimate test case.
5. **Metrics travel with the model.** Every bundle embeds its held-out metrics and vocabulary, so every inference output and the live dashboard display the model's provenance; results cannot silently outlive the model that produced them.
6. **Tiered difficulty benchmark** (easy / normal / hard / adversarial) as the results narrative for synthetic evaluation, with the adversarial tier reported as a stress test, not a failure.

Current headline numbers (all held-out, from the tracked metrics files and validated runs):

| Result | Value | Data |
|---|---|---|
| Identify, 7 real device families (RF, full features) | macro-F1 0.955 | 171 active windows, stratified split (single session per device) |
| Mix presence (RF, 17 agg features) | macro-F1 0.916 | 1920 windows from the measured-scenario corpus, grouped split (whole scenarios held out) |
| Mix power / gated power MAE | 3.3 W / 2.9 W | same |
| Earlier 7-family mix model (2026-07-02) | presence F1 0.90, gated MAE 12 W | before event features and 30-scenario corpus |
| Real multi-device recordings (never trained on) | 4 of 9 perfect device sets (set-F1 1.0), incl. a 3-device mix | failures concentrate in PV mixes (PV recorded at ~0 W) and the two small fans (11-34 W overlap) |
| Clean synthetic single-appliance upper bound | RF macro-F1 0.998, LGBM 0.994 | 1957 windows, 9 appliances |
| Event detection baseline (synthetic normal tier) | Hart edge P=0.95 / R=0.27; CUSUM P=0.30 / R=0.88 | 385 significant events; complementary profiles motivate combining |
| Live teach loop (simulate) | unknown flagged after 8 s, retrained in 26 s, re-recognised at 0.95 confidence, 97.9 % power explained | full closed loop |

---

## 9. Live-system decisions (beyond MS2; details in `08_live_nilm.md`)

- **Edge claims outrank the window model.** Steady-state window features cannot distinguish "boiler + lamp" from "boiler drawing more", but the +501 W step at plug-in identifies the lamp uniquely; a matched on-edge therefore claims the device ON with the step's own watts, and unclaimed model watts are rescaled into the remaining measured total. A physical guard (claims must not exceed the measured total) catches missed off-edges.
- **Hysteresis + median smoothing on presence** (ON at p >= 0.55, OFF at <= 0.45, median over 3 strides) so borderline windows do not flap; standard debouncing, cheap, and keeps the event log meaningful.
- **Guided teach protocol.** Naive in-mix signature capture with baseline subtraction measurably degraded models, so teaching an unknown device walks the user through the same isolate-record protocol as a manual clean recording (OFF baseline, isolated ON capture, OFF tail), auto-advancing on measured power. Design lesson worth reporting: for training data quality, a constant-baseline subtraction is not enough; the background must either be removed physically (guided isolation) or measured redundantly.
- **In-mix teach by cross-checked estimates, not by trust in one baseline.** The convenient flow (other devices keep running, the unknown is toggled off/on once and stays running) failed in its first form because the background was modeled as a constant between two 8 s medians: everything the background did during the capture leaked into the "isolated" signal, and even a steady background left its noise in the saved waveform, inflating the variance features the identifier trains on. The rewrite treats the toggle as three independent measurements of the same watts (off-step delta, settled ON level minus baseline, and the engine's own residual history since the unknown appeared) and accepts a single toggle only when they agree; disagreement requests up to two more toggles and takes the robust median, excluding outlier stretches from the saved recording. Before saving, background noise is shrunk out of the settled part (fluctuations scaled to the device-only share sqrt(var_mix - var_baseline)) while the switch-on transient is kept as measured. Design lesson: redundant cheap measurements plus a consistency gate beat a single carefully validated capture.
- **Residual as a first-class output.** Explained-power fraction is displayed continuously and drives unknown-device detection (sustained residual above max(30 W, 15 % of total)). The system says "I do not know" instead of forcing every watt into a known class.
- **THD parity between live and training:** live THD_I is derived from the FC 0x14 spectrum with the same formula the aggregator uses, so the live feature vector matches the training distribution; without this, the deployed model would silently see a shifted feature.

---

## 10. Known limitations and threats to validity (state these in the paper)

1. **PV was never recorded generating** (indoor panel, ~0 W): PV presence/disaggregation on real data is untrained and untestable; the synthetic PV results do not transfer. Fix: record actual generation.
2. **Identify's 0.955 is a within-session estimate** (one recording session per device family, stratified split). Grouped evaluation needs several sessions per device.
3. **Washing machine** needs temporal context beyond any single window; the documented failure motivates the CNN/LSTM step.
4. **Harmonic phase features are synthetic-only** (meter limitation), and lab recordings before 2026-07-06 carry zero harmonics (wrong FC 0x14 file number, since fixed and guarded against denormal poisoning).
5. **`hour` is file-relative**, informative only on the 24 h synthetic corpus.
6. **The four tier files currently on disk are identical copies** (all seed 42); the tier benchmark needs properly differentiated regeneration before the tier curve is reported.
7. **EV fast-AC is modelled single-phase**; three-phase EV distribution is a known generator limitation.
8. **Measured scenarios are semi-synthetic** (looped recordings, random schedules, synthetic voltage): real simultaneous-device physics (voltage sag interaction, harmonic phase interaction between real devices) is only exercised by the untrained `a__b` test recordings.

---

## 11. Suggested paper experiments (all runnable from the current code)

1. **Feature ablation:** common vs full (harmonics) on identify; 14-feature vs 17-feature (event features) on mix; per-feature importance plots from the RF bundles.
2. **Model comparison:** RF vs LGBM vs MLP on identical windows/targets (the outputs are saved side by side by design).
3. **Window-length sweep** (10/30/60/120 s) against the washing-machine failure case.
4. **Tier curve:** presence/disaggregation F1 across easy/normal/hard/adversarial after regenerating differentiated tiers.
5. **Synthetic-to-real transfer:** train identify on synthetic with common features, test on the six CSV fingerprints; report the gap.
6. **Live closed loop:** the 26 s teach-retrain-recognise run as a system result, plus set-F1 on the nine held-out real mixes.

---

## References

- [1] G. W. Hart, "Nonintrusive appliance load monitoring," *Proc. IEEE* 80(12), 1992. doi:10.1109/5.192069
- [2] Y. Wu et al., "DualNILM: Energy Injection Identification Enabled Disaggregation with Deep Multi-Task Learning," arXiv:2508.14600, 2025.
- [3] S. R. Sahoo et al., "A feature fusion technique for improved non-intrusive load monitoring," *Energy Informatics* 3(1):13, 2020. doi:10.1186/s42162-020-00112-w
- [4] C. Zhang et al., "Sequence-to-point learning with neural networks for NILM," *AAAI* 2018. arXiv:1612.09106
- [5] N. Tsironi et al., "Odd Harmonic Distortion Contribution on a SVM NILM Approach," *IEEE SyNERGY MED* 2022. doi:10.1109/SyNERGYMED55767.2022.9941416
- [6] J. Kelly, W. Knottenbelt, "Neural NILM: deep neural networks applied to energy disaggregation," *BuildSys* 2015. arXiv:1507.06594
- [7] Siemens AG, *SENTRON PAC4200 Power Monitoring Device Manual* (`Datasheets/manual_pac4200_system_manual.pdf`).
- EU AI Act sources: see `06_milestone2_plan.md` section 12.
