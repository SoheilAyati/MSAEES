# MS2 Pipeline

The ML half of the NILM project, self-contained in this folder. Everything takes a `.csv` (real PAC4200 run) or `.h5` (synthetic / recorded) file and just works.

```
app.py     Streamlit UI for all of the below (recommended)
live.py    LIVE monitor: what is ON right now, exact switch times,
           unknown-device teach + retrain on the go   (Docs/08_live_nilm.md)
infer.py   signal file + trained model  ->  results (csv + json + plot + accuracy)
train.py   labelled files               ->  trained model (+ held-out metrics)
```

Four tasks are supported: **identify** (single-device window classifier), **disaggregate** (per-appliance power regression), **presence** (multi-label on/off), and **mix** (presence + power in one bundle, used by the live monitor). Three model families: Random Forest (`rf`, default), LightGBM (`lgbm`), and a scikit-learn MLP neural network (`mlp`, via `deep_models.py`) trained on the raw windowed waveform.

The full reference lives in `Docs/07_ms2_pipeline.md`; the live monitor in `Docs/08_live_nilm.md`. This file is the practical quick guide.

## UI (recommended)

```bash
uv pip install -r requirements.txt
uv run streamlit run app.py
```

Opens in the browser with five tabs: **Live**, **Infer**, **Train**, **Generate corpus**, **Aggregate (measured)**. Everything below is the command-line equivalent the UI runs for you.

## Files

| File | Role |
|---|---|
| `app.py` | Streamlit UI for everything below (`streamlit run app.py`) |
| `live.py` | live monitor: connect the meter (or `--replay` a pre-measured file), see per-device watts and confidence live, event log with exact timestamps, teach unknown devices and retrain automatically |
| `train.py` | training pipeline: learn a model from labelled data (tasks: identify / disaggregate / presence / mix) |
| `infer.py` | inference pipeline: run a trained model on one signal file |
| `nilm_pipeline.py` | shared library: `load_signal()`, label-to-family parsing, feature extraction, dynamic appliance vocabulary |
| `deep_models.py` | neural-network (`--model mlp`) training and inference on the raw waveform |
| `generate_corpus.py` | build a multi-seed synthetic corpus (calls the MS1 generator + aggregator) |
| `requirements.txt` | Python dependencies |
| `output/` | trained models, metrics, per-run inference results, live session logs |

## Quick start (measured devices, end to end)

```bash
# 0. record single devices with the PAC4200 monitor (../PAC4200_reader/pac_reader.py)
#    naming: '<device>_<setting>' e.g. water_boiler_on, standing_fan_high_no_rotation
#    a recording of SEVERAL devices at once uses '__': pv__water_boiler_on

# 1. mix recordings into ground-truth training scenarios (random on/off schedules)
python ../Aggregator/mix_measured_scenarios.py --recordings ../PAC4200_reader/recordings \
       --out ../Aggregator/measured_scenarios --n-scenarios 30 --duration 300 \
       --min-app 2 --max-app 4 --seed 11

# 2. train the MIX model (presence + per-device power in one bundle)
python train.py --task mix --data "../Aggregator/measured_scenarios/measured_scenario_*.h5" \
       --window 10 --on-w 5

# 3. offline check on a real multi-device recording; the expected devices are
#    parsed from the '__' name and a set-accuracy is reported automatically
python infer.py --input "../PAC4200_reader/recordings/water_boiler_on__table_lamp_on_<ts>.h5" \
       --model output/model_mix.joblib

# 4. go LIVE: recognition + event log + unknown-device teach/retrain loop
python live.py --host 192.168.168.1          # or: --simulate

# 4b. no meter reachable? REPLAY a pre-measured file through the same live
#     pipeline (dashboard, events, teach loop) at its recorded rate:
python live.py --replay "../PAC4200_reader/recordings/water_boiler_on__table_lamp_on_<ts>.h5"
python live.py --replay "../../Pre_Measured/pac4200_toaster_200ms.csv" --replay-speed 2
```

Also available: `--task identify` (single-device window classifier), `--task disaggregate` and `--task presence` (the mix model's two halves separately), and `--model mlp` (neural net on the raw waveform; saved as `model_<task>_mlp.joblib`, never overwriting the classical model).

## Pipeline 1: `infer.py` (signal in, result out)

The model file remembers its own task, so you only pass the signal and the model.

- **identify**: for each window it predicts the appliance. Writes `predictions.csv` (per-window label + confidence), `summary.json` (time per appliance, most-likely device), `identify_timeline.png`.
- **disaggregate**: predicts every appliance's power per window. Writes `disaggregation.csv` (power vs time), `summary.json` (energy kWh per appliance; MAE if the file has ground truth), `disaggregation.png`.
- **presence**: multi-label on/off per appliance. Writes `presence.csv` (with per-device probabilities), `summary.json`, `presence_timeline.png` (Gantt).
- **mix**: on/off + probability + gated watts per device, plus the unexplained residual. Writes `mix_timeline.csv`, `summary.json`, `mix_timeline.png`.

Outputs go to a fresh timestamped folder `output/infer_<name>_<timestamp>/` by default (`--out` to change), so runs never overwrite each other.

## Pipeline 2: `train.py` (data in, model out)

- `--task identify` learns *window to appliance name*. Each input file's label is its appliance (h5 single-appliance) or `device_name` (csv). Saves a classifier (`model_identify.joblib`) + confusion matrix + metrics.
- `--task disaggregate` learns *aggregate window to per-appliance power* from scenario `.h5` files containing `/ground_truth`. Saves a multi-output regressor + per-appliance MAE.
- `--task presence` learns *aggregate window to multi-label on/off*.
- `--task mix` trains presence + power on the same windows and saves them as ONE bundle (`model_mix.joblib`); this is what `live.py` uses.

Key options: `--model rf|lgbm|mlp`, `--features auto|common|full`, `--window`/`--stride` (seconds), `--on-w` (presence ON threshold, W), `--out DIR`. Full CLI table in `Docs/07_ms2_pipeline.md`.

The appliance vocabulary is **dynamic**: `train.py` derives the device list from the training data and stores it in the bundle, so teaching a new device and retraining is enough to grow the model.

**Evaluation is honest:** if the data has several instances (e.g. multiple seeds in sub-folders), the held-out split keeps *whole instances* apart, so the score reflects generalisation to appliance instances the model never saw, not memorisation.

## csv vs h5, and the `--features` choice

Real PAC4200 CSVs are single-phase and have **no per-order harmonics**, so:

- `--features common` (P, Q, S, PF, Q/P, THD_I, P stats) works on **both** csv and h5; use this for anything that must run on the real meter.
- `--features full` adds 3rd/5th/7th harmonic features (synthetic h5 only): higher ceiling, but won't transfer to the csv.
- `--features auto` (default) picks full if every input has harmonics, else common.

THD_I is measured on csv/scenario files and derived from harmonics on single-appliance files, so it is comparable across sources.

## Validation results (on the data in this repo)

Synthetic corpus (multi-seed, via `generate_corpus.py`):

- **Identification** (common features, tested on held-out seeds): macro-F1 **0.95**, accuracy **0.99**. Real toaster CSV predicted as **resistive** (correct).
- **Disaggregation** (LightGBM, held-out scenario): overall MAE **26 W**; best on baseload/pc/hair_dryer/resistive (1-4 W), hardest on **PV (83 W)** and **synchronous (96 W)**, the continuously-variable / four-quadrant loads. On a different-generation scenario MAE rises to ~63 W (PV over-predicted, a magnitude distribution shift; motivates PV-aware harmonic-phase features).

Measured devices (mix model trained on `mix_measured_scenarios.py` output, see `Docs/08_live_nilm.md`):

- **Mix model**, held-out: presence macro-F1 **0.90**, gated power MAE **12 W**.

## Readiness for the real device

Record real devices with `../PAC4200_reader/pac_reader.py` (or the live monitor's teach loop) and retrain the mix model as labelled recordings accumulate; the quick start above is exactly that path. The synthetic disaggregator additionally benefits from per-appliance ground truth (sub-meters) on real data; until then, real-data ground truth comes from `mix_measured_scenarios.py` mixes of single-device recordings.

## Notes

- The `mlp` model (`deep_models.py`) is the neural-network path: a scikit-learn MLP on the raw windowed [P, Q, THD_I] waveform, no PyTorch dependency. A PyTorch CNN/LSTM for full temporal modelling is a future upgrade.
- Recommended environment: a `uv` virtual environment on Python 3.12 (`uv venv --python 3.12 .venv`), because `lightgbm` and `streamlit` have no Python 3.14 wheels yet.
