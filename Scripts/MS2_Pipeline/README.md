# MS2 Pipeline

Clean pipelines for Milestone 2 + the live system. Everything takes a `.csv`
(real PAC4200 run) or `.h5` (synthetic / recorded) file and just works.

```
app.py     Streamlit UI for all of the below (recommended)
live.py    LIVE monitor: what is ON right now, exact switch times,
           unknown-device teach + retrain on the go   (Docs/08_live_nilm.md)
infer.py   signal file + trained model  ->  results (csv + json + plot + accuracy)
train.py   labelled files               ->  trained model (+ held-out metrics)
```

Everything needed is in this folder; nothing imports the old `Scripts/MS2`.

## UI (recommended)

```bash
uv pip install -r requirements.txt
uv run streamlit run app.py
```

Opens in the browser with five tabs: **Live**, **Infer**, **Train**,
**Generate corpus**, **Aggregate (measured)**.
Everything below is the command-line equivalent the UI runs for you.

## Files

| File | Role |
|---|---|
| `app.py` | **Streamlit UI** for everything below (`streamlit run app.py`) |
| `live.py` | **live monitor** — connect the meter, see per-device watts/confidence live, event log with exact timestamps, teach unknown devices and retrain automatically |
| `generate_corpus.py` | make a multi-seed synthetic corpus (calls the MS1 generator + aggregator) |
| `nilm_pipeline.py` | shared library: `load_signal()`, label→family parsing, feature extraction |
| `train.py` | TRAINING pipeline — learn a model from labelled data |
| `infer.py` | INFERENCE pipeline — run a trained model on one signal file |
| `output/` | trained models, metrics, and example results |

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

# 3. offline check on a real multi-device recording — the expected devices are
#    parsed from the '__' name and a set-accuracy is reported automatically
python infer.py --input "../PAC4200_reader/recordings/water_boiler_on__table_lamp_on_<ts>.h5" \
       --model output/model_mix.joblib

# 4. go LIVE: recognition + event log + unknown-device teach/retrain loop
python live.py --host 192.168.168.1          # or: --simulate
```

Also available: `--task identify` (single-device window classifier),
`--task disaggregate`, `--task presence` (the mix model's two halves separately),
`--model mlp` (neural net on the raw waveform).

## Pipeline 1 — `infer.py` (signal in → result out)

The model file remembers its own task, so you only pass the signal and the model.

- **identify** → for each 30 s window it predicts the appliance. Writes
  `predictions.csv` (per-window label + confidence), `summary.json` (time per
  appliance, most-likely device), `identify_timeline.png`.
- **disaggregate** → predicts every appliance's power per window. Writes
  `disaggregation.csv` (power vs time), `summary.json` (energy kWh per
  appliance; MAE if the file has ground truth), `disaggregation.png`.

Outputs go to `output/<input-name>/` by default (`--out` to change).

## Pipeline 2 — `train.py` (data in → model out)

- `--task identify` — learns *window → appliance name*. Each input file's label
  is its appliance (h5 single-appliance) or `device_name` (csv). Saves a
  classifier (`model_identify.joblib`) + confusion matrix + metrics.
- `--task disaggregate` — learns *aggregate window → per-appliance power* from
  scenario `.h5` files containing `/ground_truth`. Saves a multi-output
  regressor (`model_disaggregate.joblib`) + per-appliance MAE.

Key options: `--model rf|lgbm`, `--features auto|common|full`,
`--window`/`--stride` (seconds), `--out DIR`.

**Evaluation is honest:** if the data has several instances (e.g. multiple
seeds in sub-folders), the held-out split keeps *whole instances* apart, so the
score reflects generalisation to appliances the model never saw — not memorising.

## csv vs h5, and the `--features` choice

Real PAC4200 CSVs are single-phase and have **no per-order harmonics**, so:

- `--features common` (P, Q, S, PF, Q/P, THD_I, P stats) → works on **both**
  csv and h5; use this for anything that must run on the real meter.
- `--features full` adds 3rd/5th/7th harmonic features (synthetic h5 only) →
  higher ceiling, but won't transfer to the csv.
- `--features auto` (default) picks full if every input has harmonics, else common.

THD_I is measured on csv/scenario files and derived from harmonics on
single-appliance files, so it is comparable across sources.

## Validation results (on the data in this repo)

Trained on a 6-seed synthetic corpus (see `../MS2/generate_corpus.py`):

- **Identification** (common features, tested on held-out seeds):
  macro-F1 **0.95**, accuracy **0.99**. Real toaster CSV → predicted
  **resistive** (correct).
- **Disaggregation** (LightGBM, held-out scenario): overall MAE **26 W**;
  best on baseload/pc/hair_dryer/resistive (1–4 W), hardest on **PV (83 W)**
  and **synchronous (96 W)** — the continuously-variable / four-quadrant loads.
  On a different-generation scenario MAE rises to ~63 W (PV over-predicted —
  a magnitude distribution shift; motivates PV-aware harmonic-phase features).

## Readiness for the real device

When the PAC4200 records real data, save it as CSV (same columns as
`Pre_Measured/`) and run `infer.py` directly with the **common-feature**
identifier. Retrain on real labelled switch events as they accumulate. The
disaggregator needs per-appliance ground truth (sub-meters) before it can be
retrained on real data; until then it runs on synthetic.

## Notes

- Deep learning is intentionally omitted (no PyTorch wheels for Python 3.14).
  These classical models are the deliverable; a seq2point reference lives in
  `../MS2/deep_seq2point.py` for a future torch environment.
- The older `Scripts/MS2/` folder (exploration + step-by-step demos) can be
  removed if you only want the clean pipelines.
