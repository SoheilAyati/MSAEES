# NILM Project — MS2 Pipeline Reference

**Version:** 1.0
**Milestone:** 2
**Location:** `Scripts/MS2_Pipeline/`
**Companion to:** `06_milestone2_plan.md` (the plan) — this document describes what was actually built.
**Owners:** Soheil Ayati, Marc Steffgen
**Last updated:** 2026-06-07

---

## 1. Purpose and scope

`Scripts/MS2_Pipeline/` is the self-contained Milestone-2 toolkit. It does the analysis/ML half of NILM: take a signal file in, train a model or produce per-appliance results out. Two entry points cover everything:

```
train.py    labelled data in   ->  a trained model (+ metrics)
infer.py    one signal in      ->  results (CSV + JSON + plots)
```

Plus a shared library (`nilm_pipeline.py`), the neural-network module (`deep_models.py`), a corpus generator (`generate_corpus.py`), and a point-and-click UI (`app.py`). Nothing here depends on the old `Scripts/MS2/` exploration folder.

The pipeline supports **three tasks** (identify, disaggregate, presence) and **three model families** (Random Forest, LightGBM, MLP neural network).

---

## 2. Folder layout

| File | Role |
|---|---|
| `app.py` | Streamlit UI for everything (`streamlit run app.py`) |
| `train.py` | training pipeline — learn a model from labelled data |
| `infer.py` | inference pipeline — run a trained model on one signal |
| `nilm_pipeline.py` | shared library: file loading + feature/sequence extraction |
| `deep_models.py` | neural-network (MLP) training + inference |
| `generate_corpus.py` | build a multi-seed synthetic corpus (calls the MS1 generator + aggregator) |
| `requirements.txt` | Python dependencies |
| `README.md` | short usage guide |
| `output/` | trained models, metrics, and per-run inference results |

---

## 3. Quick start

**UI (recommended):**
```bash
pip install streamlit          # or: uv pip install streamlit
streamlit run app.py           # opens in browser: Infer / Train / Generate tabs
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

# 3. infer (the model file knows its own task)
python infer.py --input <signal.csv|.h5> --model output\model_identify.joblib
python infer.py --input <scenario.h5>    --model output\model_disaggregate.joblib
python infer.py --input <scenario.h5>    --model output\model_presence.joblib
```

---

## 4. Input data

`nilm_pipeline.load_signal(path)` auto-detects the file type and returns a uniform `Signal` object, so the rest of the code never special-cases the source.

| Source | Detect | Channels | Ground truth? |
|---|---|---|---|
| Real PAC4200 run | `.csv` | single phase: P, Q, S, PF, THD_I, THD_V, I, V | no (device name only) |
| Single-appliance synthetic | `.h5` with `/measurements/P` | P, Q, current harmonics | per-sample state (label = appliance) |
| Scenario (aggregate) synthetic | `.h5` with `/measurements/P_total` + `/ground_truth` | per-phase + total P,Q,S,PF,THD, 39-order harmonics | per-sample per-appliance power |

The `Signal` object exposes `P, Q, S, PF, THD_I` (always), `P_phase` and `harm_I` (synthetic only), and `gt_names`/`gt_P` (scenarios only). Non-finite values are converted to NaN on load.

The 9 canonical appliances (`nilm_pipeline.CANON`): `baseload, ev, fridge, hair_dryer, pc, pv, resistive, synchronous, washing_machine`.

---

## 5. Core concepts

### 5.1 Windowing — `window`, `stride`, `on-threshold`

The signal is 5 Hz. Models need fixed-size chunks, so the pipeline slides a **window** (length, seconds) along the signal and computes one feature row per window. **Stride** (seconds) is how far it jumps between windows; stride < window means overlapping windows. A window is "active" if its mean |P| exceeds **on-threshold** (W).

- Identification uses overlapping windows (window/stride configurable; e.g. 10/5 for short real runs, 30/30 for scenarios).
- Disaggregation and presence use **non-overlapping** windows of length `window` (stride is not used there).
- Rule of thumb: make the window about as long as the shortest event you care about. Keep `window` consistent between train and infer for `identify` (for disaggregate/presence the window is stored in the model and reused automatically).

### 5.2 Feature sets (classical models)

| Set | Columns | Use |
|---|---|---|
| `FEATURES_COMMON` | P_mean, Q_mean, S_mean, PF_mean, Q/P ratio, THD_I, P_std, P_min, P_max | works on real PAC4200 CSVs too (no per-order harmonics) |
| `FEATURES_FULL` | COMMON + 3rd/5th/7th harmonic magnitudes, spectral centroid, harmonic energy | synthetic `.h5` only; higher ceiling |
| `AGG_FEATURES` | Ptot mean/std/min/max, Qtot mean, per-phase P means, PF, THD_I, hour-of-day | aggregate features for disaggregate/presence |

THD_I is measured on csv/scenario data and **derived from harmonics** on single-appliance files, so it is comparable across all sources.

### 5.3 Raw sequences (neural model)

The MLP path does **not** use the summary features above. `aggregate_sequences()` flattens the raw `[P, Q, THD_I]` samples inside each non-overlapping window into one vector (3 × window-samples), so the network learns from the waveform shape itself.

---

## 6. Tasks

| Task | Question it answers | Training data | Output per window |
|---|---|---|---|
| **identify** | "what single appliance is this?" | single-appliance / single-device files (label per file) | one appliance label |
| **disaggregate** | "how much power does each appliance draw?" | scenario `.h5` with ground truth | per-appliance power (W) |
| **presence** | "which appliances are ON right now?" | scenario `.h5` with ground truth | multi-label on/off per appliance |

**When to use which:**
- `identify` is for *single-device* signals — the real PAC4200 device CSVs, or single-appliance synthetic files. On a mixed scenario it can only name the one appliance each window most resembles (a known limitation; the infer plot shows this against ground truth).
- `disaggregate` is the right tool for a mixed scenario when you want **power per appliance**.
- `presence` is the right tool for a mixed scenario when you want **which appliances are on** (multi-label), e.g. a Gantt timeline. Presence "on" is defined as |appliance power| > 15 W per window.

---

## 7. Model families

| `--model` | Type | Input | Tasks | Notes |
|---|---|---|---|---|
| `rf` (default) | Random Forest | summary features | identify, disaggregate, presence | robust, fast, interpretable |
| `lgbm` | LightGBM | summary features | identify, disaggregate | needs `lightgbm`; presence forces RF internally (handles always-on classes) |
| `mlp` | Neural network (scikit-learn MLP) | **raw waveform** | disaggregate, presence | the "deep-learning" path; no extra dependency |

The `mlp` model is a multi-layer perceptron trained on the raw windowed waveform via `deep_models.py`. It is the neural-network path that runs without PyTorch. A true PyTorch CNN/LSTM is a future upgrade (needs a torch environment) for full temporal/sequence modelling.

---

## 8. Training pipeline — `train.py`

```
python train.py --data <files|globs|dirs> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--task` | identify | identify / disaggregate / presence |
| `--model` | rf | rf / lgbm / mlp |
| `--features` | auto | identify only: auto / common / full |
| `--window` | 30 | window length (s) |
| `--stride` | 30 | stride (s); identify only |
| `--on-threshold` | 5 | "active window" power floor (W) |
| `--out` | `output` | where models + metrics are written |

**Honest evaluation.** When the data contains several instances (e.g. multiple seeds in sub-folders), the held-out split keeps **whole instances apart** (`GroupShuffleSplit`), so the reported score reflects generalisation to appliances the model never saw — not memorisation. With only one instance per class it falls back to a stratified row split (and says so).

**Outputs** (in `--out`):

| File | From |
|---|---|
| `model_identify.joblib` / `model_disaggregate.joblib` / `model_presence.joblib` | classical models |
| `model_disaggregate_mlp.joblib` / `model_presence_mlp.joblib` | neural models (kept separate so they don't overwrite) |
| `train_<task>_metrics.json` (and `_mlp` variants) | held-out scores |
| `train_identify_confusion.png` | identify only |

Files are streamed one at a time, so memory stays small even on a large corpus.

---

## 9. Inference pipeline — `infer.py`

```
python infer.py --input <file> --model <model.joblib> [--window 10 --stride 5]
```

The model file remembers its own task and (for disaggregate/presence/mlp) its window, so you only pass the signal and the model. **Each run is saved to its own timestamped folder** `output/infer_<name>_<YYYYmmdd_HHMMSS>/`, so nothing is overwritten and runs can be compared.

| Task | Files written | Plot |
|---|---|---|
| identify | `predictions.csv`, `summary.json` | `identify_timeline.png` — signal with the area under it **coloured by predicted appliance**; a hatched overlay marks the model's 2nd-best guess; if the scenario has ground truth, a **truth panel** is drawn below for comparison |
| disaggregate | `disaggregation.csv`, `summary.json` (energy kWh/appliance, MAE if GT) | `disaggregation.png` — predicted vs true power for the top appliances |
| presence | `presence.csv`, `summary.json` (fraction on, F1 if GT) | `presence_timeline.png` — **Gantt** of predicted presence, with ground-truth presence below when available |

For short real-device CSV runs use a small window (e.g. `--window 10 --stride 5`); the 80 s runs give too few 30 s windows.

---

## 10. Corpus generation — `generate_corpus.py`

```
python generate_corpus.py --seeds 101 102 103 --outdir corpus [--duration 86400]
```

For each seed it runs the Milestone-1 generator (9 appliances + baseload) and aggregator, producing `corpus/seed_<NNN>/<appliance>.h5` (single-appliance files, for `identify`) and `corpus/scenario_seed<NNN>.h5` (aggregate + ground truth, for `disaggregate`/`presence`). Different seeds = different appliance *instances*, which is what enables honest train-on-some / test-on-held-out evaluation. ~230 MB per seed (24 h); use fewer seeds or `--duration 43200` for quick tests.

Note: intermittent appliances (EV, washing machine) don't run every day, so a single seed may lack them — train across several seeds so every appliance is represented "on".

---

## 11. The UI — `app.py`

`streamlit run app.py` opens a browser app with three tabs:

- **Infer** — pick a signal file and a trained model (dropdowns auto-list Pre_Measured, Synthetic_Data, corpus, and `output/` models), set window/stride, run; the result graphs render inline and are saved to a fresh timestamped folder.
- **Train** — choose task, model, features, window; trains and shows the held-out metrics (+ confusion matrix for identify).
- **Generate corpus** — enter seeds and build a corpus.

The UI is a thin wrapper: each button runs the same `train.py` / `infer.py` / `generate_corpus.py` and displays their outputs.

---

## 12. Environment

Core (any Python ≥ 3.10): `numpy, pandas, scikit-learn, h5py, matplotlib, joblib`. Optional: `lightgbm` (for `--model lgbm`), `streamlit` (for the UI). Install with `pip install -r requirements.txt`.

**Recommended:** a `uv` virtual environment on **Python 3.12**, because `lightgbm`, `streamlit` (via `pyarrow`) and friends have no Python 3.14 wheels yet:
```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```
The `mlp` (neural-network) path uses scikit-learn only — no PyTorch — so it runs in any of these environments.

---

## 13. Known limitations and guidance

- **Single-label identify on scenarios** can only name one appliance per window; for "which appliances are in a mixed scenario" use `presence` (or `disaggregate`). The identify plot's ground-truth panel makes the gap visible.
- **Washing machine** is the hardest appliance: its low-power agitate/rinse phases are buried under other loads, and a single window can't see its full multi-minute cycle. A longer `--window` (60–120 s) helps; full recovery needs a temporal/sequence model.
- **The 4 tier files** (`Synthetic_Data/Mixed/scenario_{easy,normal,hard,adversarial}.h5`) are identical copies (all seed 42); a real difficulty curve needs differentiated tiers.
- **PV magnitude varies** a lot run-to-run, so disaggregated PV can be over/under-estimated when the test instance differs from training (a distribution-shift effect).
- **`mlp` is an MLP, not a CNN/LSTM** — it sees the raw waveform but, at a 30 s window, still not the washing machine's full cycle. It's the neural baseline; a PyTorch CNN/LSTM is the next step.
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

Compare the two `presence_timeline.png` outputs (RF vs MLP) against the ground-truth panel — a clean classical-vs-neural result for the report.
