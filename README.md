# NILM Project

Non-Intrusive Load Monitoring (NILM) recovers per-appliance power consumption from a single aggregate measurement at the Point of Common Coupling. This project develops a complete NILM pipeline targeting two challenges that distinguish good systems from passing demonstrations: **PV-aware disaggregation** (handling bidirectional power flow when behind-the-meter generation creates a *signal-eclipse* effect) and **multi-feature fusion** (combining steady-state, harmonic, and transient features rather than choosing one).

Course: *Modeling, Simulation and Automation of Electrical Energy Systems*, TH Köln.     
Team: Soheil Ayati (11153003), Marc Steffgen (11149043).

## What's built

A complete data pipeline for three-phase synthetic data designed to match what a Siemens SENTRON PAC4200 power meter delivers in the lab:

- **`Scripts/Synthetic_data_generator/`** — generates synthetic per-appliance HDF5 traces (8 appliances + baseload, each with its own state machine, harmonic content, and time-of-day usage pattern).
- **`Scripts/Aggregator/`** — combines per-appliance traces into a scenario file, performing the physics the meter does at the PCC (phase distribution, complex harmonic summation, voltage synthesis). `mix_measured_scenarios.py` does the same for *real* PAC4200 recordings (random on/off schedules → ground-truth training mixes).
- **`Scripts/Preprocessor/`** — universal data cleaning and feature engineering: validation, NaN/Inf handling, gap imputation, outlier clipping, and 12 derived features for downstream ML.
- **`Scripts/PAC4200_reader/`** — live monitor + per-appliance session recorder for the real PAC4200 over Modbus TCP (browser dashboard, harmonics via FC 0x14). Includes a simulation mode.
- **`Scripts/MS2_Pipeline/`** — the ML side: train / infer / UI for **identify**, **disaggregate**, **presence**, and **mix** (presence + power in one bundle), plus **`live.py`** — the live NILM monitor that shows which devices are ON right now (watts + confidence), logs every switch event with exact timestamps, and *learns unknown devices on the go* (teach → auto-retrain → hot reload). See `Docs/07_ms2_pipeline.md` and `Docs/08_live_nilm.md`.

Detailed specifications for each component live in `Docs/`.

## Requirements

Python ≥ 3.10, with `numpy`, `h5py`, and `pymodbus`:

```bash
pip install numpy h5py pymodbus
```

## Running the pipeline

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

# (Later) Record from a real PAC4200
python Scripts/Reader/pac4200_reader.py \
    --host 192.168.1.50 --duration 86400 \
    --output Synthetic_Data/real_001/scenario.h5
```

Generated HDF5 files are not committed (the full dataset is ~500 MB compressed); they are reproducible from the scripts above.
