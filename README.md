# NILM Project

Non-Intrusive Load Monitoring (NILM) recovers per-appliance power consumption from a single aggregate measurement at the Point of Common Coupling. This project develops a complete NILM pipeline targeting two challenges that distinguish good systems from passing demonstrations: **PV-aware disaggregation** (handling bidirectional power flow when behind-the-meter generation creates a *signal-eclipse* effect) and **multi-feature fusion** (combining steady-state, harmonic, and transient features rather than choosing one).

Course: *Modeling, Simulation and Automation of Electrical Energy Systems*, TH Köln.     
Team: Soheil Ayati (11153003), Marc Steffgen (11149043).

## What's built

A complete data pipeline for three-phase synthetic data designed to match what a Siemens SENTRON PAC4200 power meter delivers in the lab:

- **`Scripts/Synthetic_data_generator/`** — generates synthetic per-appliance HDF5 traces (8 appliances + baseload, each with its own state machine, harmonic content, and time-of-day usage pattern).
- **`Scripts/Aggregator/`** — combines per-appliance traces into a scenario file, performing the physics the meter does at the PCC (phase distribution, complex harmonic summation, voltage synthesis).
- **`Scripts/Preprocessor/`** — universal data cleaning and feature engineering: validation, NaN/Inf handling, gap imputation, outlier clipping, and 12 derived features for downstream ML.
- **`Scripts/Reader/`** — polls a real PAC4200 over Modbus TCP and writes scenario files structurally identical to the aggregator output, so downstream code is unchanged. Includes a simulation mode for testing without hardware.

Detailed specifications for each component live in `Docs/`.

## What's next

- Connect the real PAC4200 in the lab and record actual measurements.
- Tune preprocessing parameters against real data.
- Build the ML side: event detection, feature extraction, appliance classification, and disaggregation.

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
