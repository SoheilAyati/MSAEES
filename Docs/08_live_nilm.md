# NILM Project — Live NILM & Training-on-the-go Reference

**Version:** 1.0
**Milestone:** 3
**Location:** `Scripts/MS2_Pipeline/live.py`
**Companion to:** `07_ms2_pipeline.md` (offline train/infer), `05_pac4200_reader.md` (the meter reader it builds on)
**Owners:** Soheil Ayati, Marc Steffgen
**Last updated:** 2026-07-02

---

## 1. What it does

`live.py` closes the loop between the PAC4200 and the MS2 models. It connects to
the meter, evaluates the trained **mix** model (presence + disaggregation) on a
sliding window of the live signal, and serves a dashboard that continuously
answers:

1. **Which devices are ON right now** — with estimated watts, a confidence
   percentage, and *since when*.
2. **What switched, exactly when** — an event log with millisecond timestamps,
   fed by a step detector on P/Q whose edges are matched against per-device
   signatures (and persisted to `events.csv`).
3. **How trustworthy the answer is** — the model's held-out accuracy (presence
   F1, power MAE), the live per-device confidence, and the *explained power*
   fraction (how much of the measured total the per-device estimates add up to).

And when the system does **not** know the answer, it learns:

4. **Unknown device → teach → retrain on the go.** Sustained unexplained power
   raises a prompt ("Unknown device, ~180 W since 14:32:05 — what is this?").
   You type a name; the captured signature is saved as a labelled recording;
   scenarios are rebuilt and the mix + identify models are **retrained in the
   background** (≈30–90 s) and hot-reloaded. The next time that device runs,
   the system recognizes it.

```
                 ┌────────────────────────── live dashboard ──────────────────────────┐
PAC4200 ─Modbus─▶ AcquisitionService ─▶ ring buffer ─▶ LiveEngine ─▶ who is ON (W, %) │
                 (from pac_reader.py)                  │   ├─▶ edge detector ─▶ event log
                                                       │   ├─▶ residual monitor ─▶ unknown?
                                                       │   └─▶ teach ─▶ recording ─▶ Retrainer
                                                       │                  (scenarios + train.py)
                                                       └◀───────── hot reload ◀───────┘
```

## 2. Running it

```bash
cd Scripts/MS2_Pipeline

# real meter (harmonics on by default → live THD_I matches training):
python live.py --host 192.168.168.1

# no hardware — exercise the whole loop against the simulated meter:
python live.py --simulate

# via the Streamlit app: the "Live" tab starts/stops it for you
streamlit run app.py
```

Dashboard: `http://127.0.0.1:8300/` (`--web-port` to change). Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--stride` | 2.0 | re-evaluate the model every N seconds |
| `--models-dir` | `output` | where `model_mix.joblib` / `model_identify.joblib` live |
| `--recordings-dir` | `../PAC4200_reader/recordings` | taught devices are saved here; signatures are read from here |
| `--scenarios-dir` | `../Aggregator/measured_scenarios` | retraining writes its training scenarios here |
| `--retrain-window` / `--on-w` | 10 s / 5 W | training parameters used by the on-the-go retrain |
| `--unknown-min-w` | 30 W | unexplained power that triggers the unknown prompt |
| `--no-harmonics` | off | skip FC-0x14 harmonic reads (live THD_I then unavailable) |

## 3. How recognition works

- Every `--stride` seconds the engine takes the trailing `window_s` (stored in
  the model bundle, default 10 s) of the ring buffer, builds the **same
  aggregate feature row used in training** (`nilm_pipeline.aggregate_windows`),
  and asks the mix bundle: presence probabilities per device + regressed watts.
  Presence output is median-smoothed over 3 strides and passed through a small
  hysteresis (on ≥ 0.55, off ≤ 0.45) so borderline windows don't flap.
- **THD parity:** `ThdReader` derives THD_I per phase from the live per-order
  harmonic magnitudes with the same formula the aggregator uses for scenario
  files, so the live feature vector matches the training distribution.
- **Exact times** come from the edge detector: settled-median before vs after
  each step on P_total (debounced, both sides must be quiet). Each edge's
  (ΔP, ΔQ) is matched to the nearest device signature (steady-state medians of
  every single-device recording). A presence transition adopts the matching
  edge's timestamp when one exists, so `device_on water_boiler` is logged at
  the actual switching moment, not at window resolution.
- **Residual** = measured window mean − Σ (watts of devices predicted ON). It
  is shown live as *explained %*, and drives unknown detection: residual above
  `max(--unknown-min-w, 15 % of total)` for ≥ 8 s ⇒ unknown-device prompt.

## 4. Teaching (training on the go)

Two paths, both end in an automatic background retrain + hot reload:

1. **Teach from the unknown prompt.** The engine captures the segment since the
   unknown appeared and subtracts the pre-event baseline, so the *other* running
   devices are removed; the delta trace is written as a normal labelled
   recording (`metadata.source = live_teach_delta`).
2. **Record it clean** (side panel). Plug in only the new device, name it,
   record ~60 s, stop. This uses the same session writer as the PAC4200
   monitor, including harmonics — the higher-fidelity path when you can
   isolate the device.

The **Retrainer** then runs, exactly like the CLI: `mix_measured_scenarios.py`
(random ON/OFF schedules per appliance, coverage-guaranteed mixes) →
`train.py --task mix` → `train.py --task identify`, then reloads the bundles
and the signature table. The device vocabulary is derived from the data each
time, so new devices simply appear. Progress is shown in the dashboard; typical
duration ≈ 30–90 s.

## 5. Outputs

- `output/live_<timestamp>/events.csv` — every event with ISO milliseconds:
  `edge_on/edge_off` (exact step, matched device, ΔP/ΔQ, confidence),
  `device_on/device_off` (presence transitions), `unknown_detected/cleared`,
  `taught`.
- Taught/recorded devices → `--recordings-dir` (same format as the PAC4200
  monitor's recordings).
- Retrained models + metrics → `--models-dir` (same files as `train.py`).

## 6. Validated behaviour (2026-07-02, simulate mode + measured recordings)

- Mix model on the 7 measured device families: held-out presence macro-F1
  **0.90**, gated power MAE **12 W**.
- Real multi-device recordings (`a__b__c` names, never used in training):
  4 of 9 perfect device sets (F1 1.0) including the 3-device coffee-machine
  mix; failures concentrate in (a) PV-shifted mixes — the PV recordings contain
  ~0 W of generation, so PV distortion is missing from training — and (b) the
  two small fans running simultaneously (11–34 W, overlapping P/Q signatures).
- Full unknown→teach→retrain loop in simulate mode: unknown flagged after 8 s,
  taught, retrained in **26 s**, recognized afterwards at 0.95 confidence with
  97.9 % of power explained; switch-off edge matched to the taught signature at
  millisecond resolution.

**Known gaps:** record the PV actually generating (the current `pv_only`
recording is ~0 W, so PV can neither be trained nor detected); more variant
recordings of the two fans would separate them better; simulate mode's ±800 W
baseline oscillation is not a trained device, so it (correctly) shows up as
unknown — it demos the teach loop, not accuracy.
