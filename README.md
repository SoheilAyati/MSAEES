# NILM Project

Non-Intrusive Load Monitoring (NILM) recovers per-appliance power consumption from a single aggregate measurement at the Point of Common Coupling. This project develops a complete NILM pipeline targeting two challenges that distinguish good systems from passing demonstrations: **PV-aware disaggregation** (handling bidirectional power flow when behind-the-meter generation creates a *signal-eclipse* effect) and **multi-feature fusion** (combining steady-state, harmonic, and transient features rather than choosing one).

Course: *Modeling, Simulation and Automation of Electrical Energy Systems*, TH Köln.
Team: Soheil Ayati (11153003), Marc Steffgen (11149043).

## Repository layout

```
Datasheets/                     PAC4200 manuals and register documentation
                                (manual_pac4200_system_manual.pdf is the English
                                system manual, manual_pac4200_de-DE_de-DE.pdf the German one)
Docs/                           full specifications, one document per component (see index below)
Pre_Measured/                   early real PAC4200 CSV captures (toaster etc.), used as test inputs
Scripts/
  Aggregator/                   combines per-appliance traces into scenario files;
                                mix_measured_scenarios.py does the same for real recordings
  MS2_Pipeline/                 the ML side: train / infer / live monitor / Streamlit UI
  PAC4200_reader/               live meter monitor + session recorder (Modbus TCP);
                                recordings/ holds the measured .h5 sessions,
                                tools/ holds one-off Modbus probe scripts
  Preprocessor/                 universal data cleaning and feature engineering
  Synthetic_data_generator/     per-appliance synthetic HDF5 trace generator
  Visualisation/                plotting helpers for generated and recorded data
Synthetic_Data/                 generated scenario datasets (gitignored, reproducible)
```

## What's built

- **`Scripts/Synthetic_data_generator/`** generates synthetic per-appliance HDF5 traces (8 appliances + baseload, each with its own state machine, harmonic content, and time-of-day usage pattern), designed to match what a Siemens SENTRON PAC4200 power meter delivers in the lab.

- **`Scripts/Aggregator/`** combines per-appliance traces into a scenario file, performing the physics the meter does at the PCC: phase distribution, complex harmonic summation, voltage synthesis. `mix_measured_scenarios.py` does the same for *real* PAC4200 recordings, turning single-device sessions into ground-truth training mixes via random on/off schedules.

- **`Scripts/Preprocessor/`** is the universal cleaning and feature-engineering stage: validation, NaN/Inf handling, gap imputation, outlier clipping, and 12 derived features for downstream ML.

- **`Scripts/PAC4200_reader/`** talks to the real PAC4200 over Modbus TCP: a browser dashboard (port 8200 by default), a per-appliance session recorder, per-order harmonics via FC 0x14, and a simulation mode so everything runs without hardware. One-off register-probing utilities live in `Scripts/PAC4200_reader/tools/` (`probe_filerecord.py`, `probe_harmonic_registers.py`, `find_harmonics.py`). Recordings are organised as `recordings/*.h5` (current single-device and multi-device mix sessions), `recordings/old/` (a retired June 10 session), and `recordings/test/` (held-out `test_*` recordings). See `Docs/05_pac4200_reader.md`.

- **`Scripts/MS2_Pipeline/`** is the ML side: `train.py` / `infer.py` / a Streamlit UI covering four tasks: **identify**, **disaggregate**, **presence**, and **mix** (presence + per-device power in one bundle), with Random Forest, LightGBM, and a scikit-learn MLP (`deep_models.py`) as model families. On top of that sits **`live.py`**, the live NILM monitor: it shows which devices are ON right now (watts + confidence), logs every switch event with exact timestamps, and *learns unknown devices on the go*: you name an unknown load, it captures the signature, retrains in the background, and hot-reloads the models. No meter at hand? `--replay` plays any pre-measured file (`.h5` recording or `Pre_Measured/*.csv`) through the same live pipeline. See `Docs/07_ms2_pipeline.md` and `Docs/08_live_nilm.md`.

## Quickstart

### Synthetic path (generator to trained model)

```bash
# 1. Generate per-appliance traces (repeat for each appliance)
python Scripts/Synthetic_data_generator/Appliance_generator.py \
    --appliance fridge --phase L1 --seed 11 \
    --anchor-date 2024-06-21 \
    --output Synthetic_Data/scenario_01/fridge.h5

# 2. Aggregate into a scenario
python Scripts/Aggregator/aggregator.py \
    --input-dir Synthetic_Data/scenario_01 \
    --output Synthetic_Data/scenario_01/scenario.h5 \
    --tier train --seed 42

# 3. Preprocess
python Scripts/Preprocessor/preprocessor.py \
    --input Synthetic_Data/scenario_01/scenario.h5
```

(`Scripts/MS2_Pipeline/generate_corpus.py` automates steps 1-2 across many seeds.)

### Measured path (real PAC4200 to live monitor)

```bash
# 1. Record single devices with the PAC4200 monitor.
#    Default mode opens a web dashboard on http://127.0.0.1:8200/ with a record button:
python Scripts/PAC4200_reader/pac_reader.py --host 192.168.168.1

#    Or record a fixed duration without any UI:
python Scripts/PAC4200_reader/pac_reader.py --host 192.168.168.1 --headless \
    --label water_boiler_on --duration 120 \
    --output-dir Scripts/PAC4200_reader/recordings

# 2. Mix the single-device recordings into ground-truth training scenarios
python Scripts/Aggregator/mix_measured_scenarios.py \
    --recordings Scripts/PAC4200_reader/recordings \
    --out Scripts/Aggregator/measured_scenarios \
    --n-scenarios 30 --duration 300 --min-app 2 --max-app 4 --seed 11

# 3. Train the mix model (presence + per-device power in one bundle)
cd Scripts/MS2_Pipeline
python train.py --task mix --data "../Aggregator/measured_scenarios/measured_scenario_*.h5" \
    --window 10 --on-w 5

# 4. Offline check on a multi-device recording, then go live
python infer.py --input "../PAC4200_reader/recordings/<some_recording>.h5" \
    --model output/model_mix.joblib
python live.py --host 192.168.168.1        # or --simulate / --replay <file>
```

### UI

```bash
cd Scripts/MS2_Pipeline
streamlit run app.py
```

One browser app with five tabs (Live, Infer, Train, Generate corpus, Aggregate (measured)) that runs all of the above for you.

## Requirements

Python >= 3.10 (3.12 recommended). Core packages: `numpy`, `h5py`, `pymodbus`, `flask` for the reader; `scikit-learn`, `pandas`, `matplotlib`, `joblib`, `streamlit` (plus optional `lightgbm`) for the ML pipeline. The complete pipeline list is in `Scripts/MS2_Pipeline/requirements.txt`:

```bash
pip install -r Scripts/MS2_Pipeline/requirements.txt
```

## Documentation index

| Doc | Contents |
|---|---|
| `Docs/01_data_format.md` | the HDF5 data format shared by all components |
| `Docs/02_appliance_generator.md` | synthetic per-appliance generator |
| `Docs/03_aggregator.md` | scenario aggregation (synthetic and measured) |
| `Docs/04_preprocessor.md` | cleaning and feature engineering |
| `Docs/05_pac4200_reader.md` | PAC4200 live monitor and recorder |
| `Docs/06_milestone2_plan.md` | Milestone 2 plan |
| `Docs/07_ms2_pipeline.md` | ML pipeline reference (train / infer / tasks / models) |
| `Docs/08_live_nilm.md` | live NILM monitor and training-on-the-go |
| `Docs/MS1/`, `Docs/MS2/` | milestone report PDFs |

## Data and version control

Large generated datasets are **gitignored and reproducible** from the scripts above: `Synthetic_Data/Mixed`, the synthetic corpus (`Scripts/MS2_Pipeline/corpus`), mixed measured scenarios (`Scripts/Aggregator/measured_scenarios*`), and per-run outputs (`output/infer_*/`, `output/live_*/`). What *is* tracked in `Scripts/MS2_Pipeline/output/`: the trained model bundles (`model_identify.joblib`, `model_mix.joblib`, `model_presence.joblib`), their held-out metrics (`train_*_metrics.json`, `train_identify_confusion.png`), and the `accept3_*` acceptance-result folders.
