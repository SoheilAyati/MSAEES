# MS2 Pipeline

Two clean pipelines for Milestone 2. Both take a `.csv` (real PAC4200 run) or
`.h5` (synthetic) file and just work — no other setup.

```
infer.py   signal file + trained model  ->  results (csv + json + plot)
train.py   labelled files               ->  trained model (+ metrics)
```

Everything needed is in this folder; nothing imports the old `Scripts/MS2`.

## Files

| File | Role |
|---|---|
| `nilm_pipeline.py` | shared library: `load_signal()` (auto-detects csv/h5), feature extraction. Both scripts import it. |
| `train.py` | TRAINING pipeline — learn a model from labelled data |
| `infer.py` | INFERENCE pipeline — run a trained model on one signal file |
| `output/` | trained models, metrics, and example results |

## Quick start

```bash
# install once (Python 3.10+):  numpy pandas scikit-learn h5py matplotlib joblib  (lightgbm optional)

# ---- TRAIN ----
# appliance identification (real-meter-compatible features):
python train.py --data <folder-of-single-appliance-h5> --task identify --features common

# power disaggregation (needs scenario .h5 with /ground_truth):
python train.py --data <folder-of-scenario-h5> --task disaggregate --model lgbm

# ---- INFER ----
python infer.py --input <signal.csv|.h5> --model output/model_identify.joblib
python infer.py --input <scenario.h5>    --model output/model_disaggregate.joblib
```

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
