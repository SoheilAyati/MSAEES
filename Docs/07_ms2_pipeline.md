# NILM Project: MS2 Pipeline Reference

**Version:** 1.3
**Milestone:** 2 (+ live extensions)
**Location:** `Scripts/MS2_Pipeline/`
**Companion to:** `06_milestone2_plan.md` (the plan), `08_live_nilm.md` (live monitor + training-on-the-go). This document describes what was actually built.
**Owners:** Soheil Ayati, Marc Steffgen
**Last updated:** 2026-07-05

---

## 1. Purpose and scope

`Scripts/MS2_Pipeline/` is the self-contained Milestone-2 toolkit. It does the analysis/ML half of NILM: take a signal file in, train a model or produce per-appliance results out. Two entry points cover everything:

```
train.py    labelled data in   ->  a trained model (+ metrics)
infer.py    one signal in      ->  results (CSV + JSON + plots)
```

Plus a shared library (`nilm_pipeline.py`), the neural-network module (`deep_models.py`), a corpus generator (`generate_corpus.py`), the **live monitor** (`live.py`, see `08_live_nilm.md`), and a point-and-click UI (`app.py`). Everything the pipeline needs lives in this folder.

The pipeline supports **four tasks** (identify, disaggregate, presence, and **mix** = presence + disaggregate in one bundle) and **three model families** (Random Forest, LightGBM, MLP neural network).

**The appliance vocabulary is dynamic.** `train.py` derives the device list from whatever the training data contains (`nilm_pipeline.scan_canon`) and stores it in the model bundle under the key `"appliances"`; inference and the live monitor read it back. Teaching a new device and retraining is therefore enough to grow the model, with no code change. `scan_canon()` reads `/ground_truth/appliance_names` from every training scenario file and collapses each name to its family base. The legacy `CANON` list (9 synthetic appliances + 4 measured families, `laptop, stand_cooler, table_fan, table_pv`, appended so the synthetic indices stay stable) is only the fallback used when no explicit vocabulary is available.

**Recording label conventions** (`nilm_pipeline.strip_timestamp` / `parse_family` / `parse_families`):
- trailing recorder timestamps are stripped from filenames (`strip_timestamp`);
- state/setting suffixes are collapsed (`parse_family`): words like `high`, `low`, `on`, `rotation` are stripped, so `standing_fan_high_no_rotation` becomes family `standing_fan`;
- session prefixes such as `test_` are stripped: `test_water_boiler_on` becomes `water_boiler`;
- a double underscore separates SIMULTANEOUS devices (`parse_families` splits on `"__"`): `pv__water_boiler_on` is a real two-device mix. Such recordings are excluded from single-appliance training and serve as *test* inputs; inference parses the expected device set from the name and reports set-accuracy against it.

---

## 2. Folder layout

| File | Role |
|---|---|
| `app.py` | Streamlit UI for everything (`streamlit run app.py`) |
| `live.py` | **live monitor**: real-time recognition + event log + teach/retrain on the go (`08_live_nilm.md`) |
| `train.py` | training pipeline: learn a model from labelled data |
| `infer.py` | inference pipeline: run a trained model on one signal |
| `nilm_pipeline.py` | shared library: file loading, label-to-family parsing, feature/sequence extraction, dynamic vocabulary |
| `deep_models.py` | neural-network (MLP) training + inference |
| `generate_corpus.py` | build a multi-seed synthetic corpus (calls the MS1 generator + aggregator) |
| `requirements.txt` | Python dependencies |
| `README.md` | short usage guide |
| `output/` | trained models, metrics, per-run inference results, live session logs |

---

## 3. Quick start

**UI (recommended):**
```bash
pip install -r requirements.txt          # or: uv pip install -r requirements.txt
streamlit run app.py                     # opens in browser, five tabs
```

**Command line:**
```bash
# 1. make a corpus (multiple seeds = multiple appliance instances)
python generate_corpus.py --seeds 101 102 103 --outdir corpus

# 2. train (examples)
python train.py --task identify     --data <single-appliance-h5-folder> --features common
python train.py --task disaggregate --data "corpus\scenario_*.h5" --model lgbm
python train.py --task presence     --data "corpus\scenario_*.h5"
python train.py --task presence     --data "corpus\scenario_*.h5" --model mlp   # neural net
python train.py --task mix          --data "..\Aggregator\measured_scenarios\measured_scenario_*.h5" --window 10 --on-w 5

# 3. infer (the model file knows its own task)
python infer.py --input <signal.csv|.h5> --model output\model_identify.joblib
python infer.py --input <scenario.h5>    --model output\model_disaggregate.joblib
python infer.py --input <scenario.h5>    --model output\model_presence.joblib
python infer.py --input <recording.h5>   --model output\model_mix.joblib
```

---

## 4. Input data

`nilm_pipeline.load_signal(path)` auto-detects the file type and returns a uniform `Signal` object, so the rest of the code never special-cases the source. Dispatch:

- **`.csv`** (real PAC4200 run): read with separator `;`, columns `timestamp_iso, p_total_w, q_total_var, s_total_va, pf_total, thd_i_l1_percent, device_name`. Single phase; the label is the `device_name` column.
- **`.h5`**: routed by the presence of a `/ground_truth` group. With `/ground_truth` (containing `appliance_names` and `P_contribution`) it loads as a **scenario** (aggregate signal + per-appliance truth); without it, as a **single-appliance / single-recording** file whose label is the filename or metadata label.

| Source | Detect | Channels | Ground truth? |
|---|---|---|---|
| Real PAC4200 run | `.csv` | single phase: P, Q, S, PF, THD_I | no (device name only) |
| Single-appliance synthetic or recorded | `.h5` without `/ground_truth` | P, Q, current harmonics (if recorded) | label = appliance |
| Scenario (aggregate) | `.h5` with `/ground_truth` | per-phase + total P, Q, S, PF, THD, per-order harmonics | per-sample per-appliance power |

The `Signal` object exposes `P, Q, S, PF, THD_I` (always), `P_phase` and `harm_I` (when available), and `gt_names`/`gt_P` (scenarios only). Non-finite values are converted to NaN on load.

---

## 5. Core concepts

### 5.1 Windowing: `window`, `stride`, `on-threshold`

The signal is 5 Hz. Models need fixed-size chunks, so the pipeline slides a **window** (length, seconds) along the signal and computes one feature row per window. **Stride** (seconds) is how far it jumps between windows; stride < window means overlapping windows. A window is "active" if its mean |P| exceeds **on-threshold** (W).

- Identification uses overlapping windows (window/stride configurable; e.g. 10/5 for short real runs, 30/30 for scenarios).
- Disaggregation, presence, and mix use **non-overlapping** windows of length `window` (stride is not used there).
- Rule of thumb: make the window about as long as the shortest event you care about. Keep `window` consistent between train and infer for `identify` (for the other tasks the window is stored in the model and reused automatically).

### 5.2 Feature sets (classical models)

For single-device windows, the **steady state** is the median over the middle 20-80 % of the window, so switch-on transients at the edges don't pollute the summary statistics. THD_I is used as measured when the file provides it, and otherwise derived from the per-order harmonic magnitudes relative to the fundamental current estimated at 230 V.

| Set | Columns | Use |
|---|---|---|
| `FEATURES_COMMON` (9) | `P_mean, Q_mean, S_mean, PF_mean, QP_ratio, THD_I_mean, P_std, P_min, P_max` | works on real PAC4200 CSVs too (no per-order harmonics needed) |
| `FEATURES_HARM` (5) | `h3, h5, h7, h_centroid, h_energy` (3rd/5th/7th harmonic magnitudes, spectral centroid, harmonic energy) | needs per-order harmonics |
| `FEATURES_FULL` (14) | `FEATURES_COMMON + FEATURES_HARM` | harmonic-capable `.h5` only; higher ceiling |

For aggregate windows (disaggregate / presence / mix), `AGG_FEATURES` currently has **17 columns, in this exact order**:

```
Ptot_mean, Ptot_std, Ptot_min, Ptot_max, Qtot_mean,
PL1_mean, PL2_mean, PL3_mean, PF_mean, THDI_mean, hour,
Qtot_std, QP_ratio, Stot_mean,
Pstep_max, Qstep_at_Pstep, n_steps
```

The last three (added 2026-07-06) are **event features**: the largest settled power step inside the window, the reactive step at the same instant, and the number of steps. Steady-state sums cannot tell "boiler + lamp" from "boiler drawing more", but the switch-on step identifies the joining device.

The order matters and new features are only ever **appended**: a model bundle stores the feature list it was trained with, and `slice_features()` trims a freshly built (possibly wider) feature matrix down to that length at inference time, so old models keep working after the feature set grows.

### 5.3 Raw sequences (neural model)

The MLP path does **not** use the summary features above. `aggregate_sequences()` flattens the raw `[P, Q, THD_I]` samples inside each non-overlapping window into one vector (3 x window-samples), so the network learns from the waveform shape itself.

---

## 6. Tasks

| Task | Question it answers | Training data | Output per window |
|---|---|---|---|
| **identify** | "what single appliance is this?" | single-appliance / single-device files (label per file, mapped to its family) | one appliance label |
| **disaggregate** | "how much power does each appliance draw?" | scenario `.h5` with ground truth | per-appliance power (W) |
| **presence** | "which appliances are ON right now?" | scenario `.h5` with ground truth | multi-label on/off per appliance |
| **mix** | both at once: "which devices are ON *and* how many watts each?" | scenario `.h5` with ground truth | on/off + P(on) + gated watts per appliance |

`mix` trains a presence classifier and a power regressor on the same windows and saves them as ONE bundle (`model_mix.joblib`), together with the gated power MAE. Inference gates the power by presence (a device predicted OFF contributes 0 W) and additionally reports the *unexplained residual* (measured total minus the sum of estimates). This is the bundle the live monitor uses.

**When to use which:**
- `identify` is for *single-device* signals: the real PAC4200 device CSVs, or single-appliance files. On a mixed scenario it can only name the one appliance each window most resembles (a known limitation; the infer plot shows this against ground truth).
- `disaggregate` is the right tool for a mixed scenario when you want **power per appliance**.
- `presence` is the right tool when you want **which appliances are on** (multi-label), e.g. a Gantt timeline. Presence "on" is defined as |appliance power| > `--on-w` per window (default 15 W).
- `mix` is the deployment bundle: one file for the live monitor.

---

## 7. Model families

| `--model` | Type | Input | Tasks | Notes |
|---|---|---|---|---|
| `rf` (default) | Random Forest | summary features | all | robust, fast, interpretable |
| `lgbm` | LightGBM | summary features | identify, disaggregate | needs `lightgbm`; presence forces RF internally (handles always-on classes) |
| `mlp` | Neural network (scikit-learn MLP) | **raw waveform** | disaggregate, presence | the neural path; no extra dependency |

`deep_models.py` implements the `mlp` option: an `MLPRegressor` (disaggregate) or `MLPClassifier` (presence) with hidden layers (256, 128) inside a scaling pipeline, trained on the raw windowed `[P, Q, THD_I]` vectors from section 5.3. Models are saved as `model_<task>_mlp.joblib` with `train_<task>_mlp_metrics.json`, so they **never overwrite** the classical RF/LGBM bundles and the two can be compared side by side. A true PyTorch CNN/LSTM is a future upgrade (needs a torch environment) for full temporal/sequence modelling.

---

## 8. Training pipeline: `train.py`

```
python train.py --data <files|globs|dirs> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--data` | (required) | files, globs, or directories |
| `--task` | `identify` | `identify` / `disaggregate` / `presence` / `mix` |
| `--model` | `rf` | `rf` / `lgbm` / `mlp` |
| `--features` | `auto` | identify only: `auto` / `common` / `full` (`common` = real-PAC4200 compatible) |
| `--window` | 30 | window length (s); 10 s recommended for measured scenarios |
| `--stride` | 30 | stride (s); identify only |
| `--on-threshold` | 5 | "active window" power floor (W) |
| `--on-w` | 15 | presence/mix: \|appliance power\| above this counts as ON (use 5 W for the measured devices; the table fan draws 11 W) |
| `--raw-labels` | off | identify: keep full recording labels instead of collapsing to device families |
| `--out` | `output` | where models + metrics are written |

Every bundle stores its **held-out metrics** (`bundle["metrics"]`) and its **appliance vocabulary** (`bundle["appliances"]`), so inference outputs and the live dashboard can display the model's accuracy and device list without hunting for the metrics JSON. Multi-label F1 is macro-averaged **only over appliances that occur** in the held-out data (an appliance that is absent and never falsely predicted scores `null`, not 0).

**Honest evaluation.** When the data contains several instances (e.g. multiple seeds in sub-folders), the held-out split keeps **whole instances apart** (`GroupShuffleSplit`), so the reported score reflects generalisation to appliance instances the model never saw, not memorisation. With only one instance per class it falls back to a stratified row split (and says so).

**Outputs** (in `--out`):

| File | From |
|---|---|
| `model_identify.joblib` / `model_disaggregate.joblib` / `model_presence.joblib` | classical models |
| `model_mix.joblib` | mix bundle: presence classifier + power regressor + gated MAE in one file |
| `model_disaggregate_mlp.joblib` / `model_presence_mlp.joblib` | neural models (kept separate so they don't overwrite) |
| `train_<task>_metrics.json` (and `_mlp` variants) | held-out scores |
| `train_identify_confusion.png` | identify only |

Files are streamed one at a time, so memory stays small even on a large corpus.

---

## 9. Inference pipeline: `infer.py`

```
python infer.py --input <file> --model <model.joblib> [--window 30 --stride 30 --on-threshold 5] [--out DIR]
```

The model file remembers its own task, its vocabulary, and (for disaggregate/presence/mix/mlp) its window, so you only pass the signal and the model. **Each run is saved to its own timestamped folder** `output/infer_<name>_<YYYYmmdd_HHMMSS>/`, so nothing is overwritten and runs can be compared.

| Task | Files written | Plot |
|---|---|---|
| identify | `predictions.csv`, `summary.json` | `identify_timeline.png`: signal with the area under it coloured by predicted appliance; a hatched overlay marks the model's 2nd-best guess; if the scenario has ground truth, a truth panel is drawn below for comparison |
| disaggregate | `disaggregation.csv`, `summary.json` (energy kWh/appliance, MAE if GT) | `disaggregation.png`: predicted vs true power for the top appliances |
| presence | `presence.csv` (incl. per-device probabilities), `summary.json` (fraction on, mean confidence, F1 if GT) | `presence_timeline.png`: Gantt of predicted presence, with ground-truth presence below when available |
| mix | `mix_timeline.csv` (on/off + probability + watts per device, measured total, residual, `explained_power_fraction`), `summary.json` | `mix_timeline.png`: stacked per-device power under the measured total + presence Gantt (+ truth Gantt if GT) |

**Accuracy is always part of the result.** Every summary carries the model's held-out metrics; with ground truth it adds F1/MAE against the truth; and for **labelled real recordings without ground truth** (including `a__b__c` multi-device mixes) it parses the expected device set from the filename and reports `label_set_accuracy`: expected vs detected devices, misses, false alarms, set-F1. The plot title shows the headline numbers.

For short real-device CSV runs use a small window (e.g. `--window 10 --stride 5`); the 80 s runs give too few 30 s windows.

---

## 10. Corpus generation: `generate_corpus.py`

```
python generate_corpus.py --seeds 101 102 103 --outdir corpus [--duration 86400] [--base-date 2024-03-01]
```

For each seed it runs the Milestone-1 generator (`Appliance_generator.py`, 9 appliances + baseload) and the aggregator, producing `corpus/seed_<NNN>/<appliance>.h5` (single-appliance files, for `identify`) and `corpus/scenario_seed<NNN>.h5` (aggregate + ground truth, for `disaggregate`/`presence`/`mix`). The phase assignment is a **fixed map** (e.g. fridge on L1, resistive on L2, hair_dryer on L3, pv/synchronous on all phases): instance variety comes from the seed, not the phase. Defaults: `--duration 86400` (24 h) and `--base-date 2024-03-01`.

Different seeds = different appliance *instances*, which is what enables honest train-on-some / test-on-held-out evaluation. Roughly 230 MB per 24 h seed; use fewer seeds or `--duration 43200` for quick tests. The corpus folder is gitignored, regenerate it locally.

Note: intermittent appliances (EV, washing machine) don't run every day, so a single seed may lack them; train across several seeds so every appliance is represented "on".

---

## 11. The UI: `app.py`

`streamlit run app.py` opens a browser app with five tabs:

- **Live** starts/stops the live monitor (`live.py`): real-time recognition of what is ON (watts + confidence), event log with exact switch times, unknown-device teach-and-retrain loop. Opens its own dashboard; see `08_live_nilm.md`. A dropdown picks a replay file when no meter is reachable.
- **Infer** picks a signal file and a trained model (dropdowns auto-list Pre_Measured, Synthetic_Data, corpus, PAC4200 recordings, measured scenarios, and `output/` models), sets window/stride, runs; the result graphs render inline and are saved to a fresh timestamped folder.
- **Train** chooses task (mix/identify/disaggregate/presence), model, window, ON threshold; trains and shows the held-out metrics (+ confusion matrix for identify).
- **Generate corpus** enters seeds and builds a synthetic corpus.
- **Aggregate (measured)** mixes real PAC4200 recordings into ground-truth training scenarios (random per-appliance ON/OFF schedules, family coverage guaranteed; `test_*` and `a__b` mixed recordings handled automatically).

The UI is a thin wrapper: each button shells out to the same `live.py` / `train.py` / `infer.py` / `generate_corpus.py` / `mix_measured_scenarios.py` CLIs with the working directory set to `Scripts/MS2_Pipeline`, and displays their outputs. Anything the UI does is reproducible from the command line.

---

## 12. Environment

Core (any Python >= 3.10): `numpy, pandas, scikit-learn, h5py, matplotlib, joblib`. Optional: `lightgbm` (for `--model lgbm`), `streamlit` (for the UI), `flask`/`pymodbus` (for live/replay via the reader). Install with `pip install -r requirements.txt`.

**Recommended:** a `uv` virtual environment on **Python 3.12**, because `lightgbm`, `streamlit` (via `pyarrow`) and friends have no Python 3.14 wheels yet:
```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```
The `mlp` (neural-network) path uses scikit-learn only, no PyTorch, so it runs in any of these environments.

---

## 13. Known limitations and guidance

- **Single-label identify on scenarios** can only name one appliance per window; for "which appliances are in a mixed scenario" use `presence` or `mix`. The identify plot's ground-truth panel makes the gap visible.
- **Washing machine** is the hardest synthetic appliance: its low-power agitate/rinse phases are buried under other loads, and a single window can't see its full multi-minute cycle. A longer `--window` (60-120 s) helps; full recovery needs a temporal/sequence model.
- **The 4 tier files** (`Synthetic_Data/Mixed/scenario_{easy,normal,hard,adversarial}.h5`, gitignored) are identical copies (all seed 42); a real difficulty curve needs differentiated tiers.
- **PV magnitude varies** a lot run-to-run, so disaggregated PV can be over/under-estimated when the test instance differs from training (a distribution-shift effect).
- **`mlp` is an MLP, not a CNN/LSTM**: it sees the raw waveform but, at a 30 s window, still not the washing machine's full cycle. It's the neural baseline; a PyTorch CNN/LSTM is the next step.
- **sklearn version warnings** when loading a model trained in a different scikit-learn version are harmless; retrain locally to silence them.

---

## 14. End-to-end example

```bash
# environment
uv venv --python 3.12 .venv && .venv\Scripts\activate && uv pip install -r requirements.txt

# data: 25 seeds, hold out 115 for testing
python generate_corpus.py --seeds 101 102 103 ... 125 --outdir corpus

# train presence on everything except the test seed (classical and neural)
python train.py --task presence --data "corpus\scenario_seed1[0-1][0-9].h5" --window 30
python train.py --task presence --data "corpus\scenario_seed1[0-1][0-9].h5" --window 30 --model mlp

# test on the held-out seed
python infer.py --input corpus\scenario_seed115.h5 --model output\model_presence.joblib
python infer.py --input corpus\scenario_seed115.h5 --model output\model_presence_mlp.joblib
```

Compare the two `presence_timeline.png` outputs (RF vs MLP) against the ground-truth panel: a clean classical-vs-neural result for the report.
