# NILM Project: Live NILM and Training-on-the-go Reference

**Version:** 1.1
**Milestone:** 3
**Location:** `Scripts/MS2_Pipeline/live.py`
**Companion to:** `07_ms2_pipeline.md` (offline train/infer), `05_pac4200_reader.md` (the meter reader it builds on)
**Owners:** Soheil Ayati, Marc Steffgen
**Last updated:** 2026-07-05

---

## 1. What it does

`live.py` closes the loop between the PAC4200 and the MS2 models. It connects to
the meter, evaluates the trained **mix** model (presence + disaggregation) on a
sliding window of the live signal, and serves a dashboard that continuously
answers:

1. **Which devices are ON right now**: estimated watts, a confidence
   percentage, and *since when*.
2. **What switched, exactly when**: an event log with millisecond timestamps,
   fed by a step detector on P/Q whose edges are matched against per-device
   signatures (and persisted to `events.csv`).
3. **How trustworthy the answer is**: the model's held-out accuracy (presence
   F1, power MAE), the live per-device confidence, and the *explained power*
   fraction (how much of the measured total the per-device estimates add up to).

And when the system does **not** know the answer, it learns:

4. **Unknown device, teach, retrain on the go.** Sustained unexplained power
   raises a prompt ("Unknown device, ~180 W since 14:32:05, what is this?").
   You type a name; the captured signature is saved as a labelled recording;
   scenarios are rebuilt and the mix + identify models are **retrained in the
   background** (about 30-90 s) and hot-reloaded. The next time that device
   runs, the system recognizes it.

```
                    +--------------------- live dashboard ----------------------+
PAC4200 --Modbus--> | AcquisitionService --> ring buffer --> LiveEngine         |
                    | (from pac_reader.py)                     |                |
                    |                who is ON (W, %) <--------+                |
                    |                event log <-- edge detector <--+           |
                    |                unknown? <-- residual monitor <-+          |
                    |                                             |             |
                    |    teach --> labelled recording --> Retrainer             |
                    |              (mix_measured_scenarios + train.py)          |
                    |                                             |             |
                    |    models <------------ hot reload <--------+             |
                    +-----------------------------------------------------------+
```

## 2. Running it

```bash
cd Scripts/MS2_Pipeline

# real meter (harmonics on by default, so live THD_I matches training):
python live.py --host 192.168.168.1

# no hardware: exercise the whole loop against the simulated meter:
python live.py --simulate

# no meter reachable: REPLAY a pre-measured file through the exact same
# pipeline (.h5 recording/scenario or Pre_Measured .csv), at its recorded rate:
python live.py --replay ../PAC4200_reader/recordings/coffee_machine_run__standing_fan_high__standing_lamp_on_20260702_143215.h5

# via the Streamlit app: the "Live" tab starts/stops it for you
# (a dropdown there picks the replay file)
streamlit run app.py
```

Dashboard: `http://127.0.0.1:8300/` (`--web-port` to change). Options:

| Option | Default | Meaning |
|---|---|---|
| `--host` / `--port` / `--unit-id` | none / 502 / 1 | PAC4200 Modbus TCP address (omit `--host` for `--simulate`) |
| `--stride` | 2.0 | re-evaluate the model every N seconds |
| `--web-port` | 8300 | dashboard port |
| `--models-dir` | `output` | where `model_mix.joblib` / `model_identify.joblib` live |
| `--recordings-dir` | `../PAC4200_reader/recordings` | taught devices are saved here; signatures are read from here |
| `--scenarios-dir` | `../Aggregator/measured_scenarios` | retraining writes its training scenarios here |
| `--retrain-window` | 10 | window (s) used by the on-the-go retrain |
| `--on-w` | 5 | presence ON threshold (W) used when retraining |
| `--unknown-min-w` | 30 | unexplained power (W) that triggers the unknown prompt |
| `--no-harmonics` | off | skip FC-0x14 harmonic reads (live THD_I then unavailable) |
| `--simulate` | off | synthetic meter, no hardware needed |
| `--replay FILE` | none | play a pre-measured file instead of a meter: each poll returns the next recorded sample, so recognition, edges, and the teach loop run exactly as live. THD_I comes from the file (scalar channel, or derived from its harmonic spectrum). At the end the dashboard freezes at the final state |
| `--replay-speed` | 1.0 | replay faster (2 = twice as fast). Wall-clock windows then span speed times the recorded time; keep 1.0 when judging accuracy |
| `--replay-loop` | off | wrap around at the end of the file instead of freezing |

## 3. How recognition works

- Every `--stride` seconds (default 2.0) the engine takes the trailing
  `window_s` (stored in the model bundle, default 10 s) of the ring buffer,
  builds the **same aggregate feature row used in training**
  (`nilm_pipeline.aggregate_windows`), and asks the mix bundle: presence
  probabilities per device + regressed watts. Presence output is
  median-smoothed over the last 3 strides and passed through a small
  hysteresis (ON at probability >= 0.55, OFF at <= 0.45) so borderline
  windows don't flap.
- **THD parity:** `ThdReader` derives THD_I per phase from the live per-order
  harmonic magnitudes with the same formula the aggregator uses for scenario
  files, so the live feature vector matches the training distribution.
- **Exact times** come from the edge detector, which runs on every stride over
  the last 8 s of the buffer: it compares the settled median of the first
  2.5 s against the settled median of the last 2.5 s of that look-back. A step
  counts as an edge only if |dP| >= 8 W (or |dQ| >= 16 var), both sides are
  quiet (low standard deviation, so a ramp is not logged sample-by-sample),
  and at least 3 s have passed since the previous edge (debounce). The edge
  timestamp is the exact sample of the largest jump inside the look-back.
  Each edge's (dP, dQ) is matched to the nearest device signature
  (steady-state medians of every single-device recording). A presence
  transition adopts the matching edge's timestamp when one exists, so
  `device_on water_boiler` is logged at the actual switching moment, not at
  window resolution.
- **Residual** = measured window mean minus the sum of watts of devices
  predicted ON. It is shown live as *explained %*, and drives unknown
  detection: residual above `max(--unknown-min-w, 15 % of total)` for 8 s or
  more raises the unknown-device prompt.

## 4. Teaching (training on the go)

Two paths, both end in an automatic background retrain + hot reload:

1. **Teach from the unknown prompt.** The engine captures the segment since the
   unknown appeared and subtracts the pre-event baseline, so the *other* running
   devices are removed; the delta trace is written as a normal labelled
   recording, marked `metadata.source = live_teach_delta`.
2. **Record it clean** (side panel). Plug in only the new device, name it,
   record about 60 s, stop. This uses the same session writer as the PAC4200
   monitor, including harmonics: the higher-fidelity path when you can
   isolate the device.

The **Retrainer** then runs, exactly like the CLI chain:
`mix_measured_scenarios.py` (random ON/OFF schedules per appliance,
coverage-guaranteed mixes), then `train.py --task mix`, then
`train.py --task identify`, then it hot-reloads the bundles and the signature
table. The device vocabulary is derived from the data each time, so new
devices simply appear. Progress is shown in the dashboard; typical duration
is 30-90 s.

## 5. Outputs

- `output/live_<timestamp>/events.csv` with columns
  `time_iso, unix_ms, kind, device, confidence, dP_W, dQ_var, P_total_W, detail`.
  Event kinds: `edge_on`/`edge_off` (exact step, matched device, dP/dQ,
  confidence), `device_on`/`device_off` (presence transitions),
  `unknown_detected`/`unknown_cleared`, `taught`.
- Taught/recorded devices go to `--recordings-dir` (same format as the PAC4200
  monitor's recordings).
- Retrained models + metrics go to `--models-dir` (same files as `train.py`).

The `output/live_*/` session folders are gitignored; the retrained model
bundles and metrics in `output/` are tracked.

## 6. Validated behaviour (2026-07-02, simulate mode + measured recordings)

- Mix model on the 7 measured device families: held-out presence macro-F1
  **0.90**, gated power MAE **12 W**.
- Real multi-device recordings (`a__b__c` names, never used in training):
  4 of 9 perfect device sets (F1 1.0) including the 3-device coffee-machine
  mix; failures concentrate in (a) PV-shifted mixes, because the PV recordings
  contain ~0 W of generation so PV distortion is missing from training, and
  (b) the two small fans running simultaneously (11-34 W, overlapping P/Q
  signatures).
- Full unknown-teach-retrain loop in simulate mode: unknown flagged after 8 s,
  taught, retrained in **26 s**, recognized afterwards at 0.95 confidence with
  97.9 % of power explained; switch-off edge matched to the taught signature at
  millisecond resolution.

**Known gaps:** record the PV actually generating (the current `pv_only`
recording is ~0 W, so PV can neither be trained nor detected); more variant
recordings of the two fans would separate them better; simulate mode's +/-800 W
baseline oscillation is not a trained device, so it (correctly) shows up as
unknown: it demos the teach loop, not accuracy.
