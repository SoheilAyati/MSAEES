# NILM Project: Live NILM and Training-on-the-go Reference

**Version:** 1.2
**Milestone:** 3
**Location:** `Scripts/MS2_Pipeline/live.py`
**Companion to:** `07_ms2_pipeline.md` (offline train/infer), `05_pac4200_reader.md` (the meter reader it builds on), `09_design_rationale.md` (design rationale, incl. the live-system decisions)
**Owners:** Soheil Ayati, Marc Steffgen
**Last updated:** 2026-07-06

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
- **Edge claims drive state (2026-07-06).** A matched on-edge does not just
  timestamp events anymore: it CLAIMS the device ON with the step's own watts,
  and a matched off-edge releases the claim (holding the device off while the
  model window still contains pre-switch samples). Claims outrank the window
  model because steady-state features cannot tell "boiler + lamp" from
  "boiler drawing more" -- but the +501 W step at plug-in time identifies the
  lamp uniquely. Watts of unclaimed model-ON devices are rescaled into
  whatever the claims leave of the measured total, which is what splits a
  1444 W window into boiler 943 W + lamp 501 W. A physical guard drops the
  weakest claim whenever the claimed devices alone would exceed the measured
  total (missed off-edge). Claimed devices show `edge` next to their
  confidence on the dashboard.
- **Residual** = measured window mean minus the sum of watts of devices
  predicted ON. It is shown live as *explained %*, and drives unknown
  detection: residual above `max(--unknown-min-w, 15 % of total)` for 8 s or
  more raises the unknown-device prompt.

### Model variants (2026-07-06)

Two mix bundles can coexist in the models dir and the dashboard's Model panel
has a dropdown to switch between them at runtime:

- **train-on-the-go (latest)** -> `model_mix.joblib`: what every retrain
  (teach loop or Retrain button) overwrites.
- **original (frozen)** -> `model_mix_original.joblib`: a blessed snapshot that
  retraining never touches. Freeze one by copying:
  `cp model_mix.joblib model_mix_original.joblib`.

### State rebuild after a model reload (software "re-plug")

Whenever the model reloads (retrain finished, or the variant was switched),
the engine RE-MATCHES every edge recorded this session against the new
signature table and rebuilds the claims in order (`state_rebuilt` event).
This is a software version of unplugging everything and plugging it back in:
an edge that read `unrecognized` before a device was taught resolves to that
device afterwards, and a step that matched the wrong sibling gets re-decided -
no need to physically power-cycle the appliances.

## 4. Teaching (training on the go)

Three paths, all end in an automatic background retrain + hot reload:

1. **Teach from the unknown prompt (GUIDED / isolated, 2026-07-06).** Naming
   the unknown starts a guided walk-through of the same protocol as the
   manual record button. The dashboard shows one step at a time, and each
   step advances automatically from the measured power (no confirm clicks):
   1. *Disconnect ALL devices* (including the unknown one) - waits for total
      power < 5 W;
   2. records a 10 s OFF baseline;
   3. *Connect only the new device* - waits for power to appear;
   4. records it running for `--teach-record-s` (default 45) seconds;
   5. *Disconnect it* - then records an 8 s OFF tail.
   The result is a normal clean session recording (same writer as the manual
   path, harmonics included) and the retrain starts automatically. A Cancel
   button aborts and discards the partial file; if the device is never
   disconnected in step 5, the recording is saved anyway (the ON data is
   already captured). The unknown prompt is suppressed while a teach runs.
2. **Teach ON THE GO (in-mix).** When emptying the mains is impractical
   (fridge, router, a running experiment), the other devices keep running
   and the unknown device is toggled off and back on once; it stays running
   when the flow ends. Naive baseline subtraction (one 8 s baseline median,
   endpoint drift check) taught visibly worse models: anything the
   background did during the capture leaked into the "isolated" signal, and
   even a steady background left its noise in the saved waveform, inflating
   the signature's variance features. The current flow therefore treats the
   capture as one of several independent measurements and cross-checks them:
   1. *Switch OFF only the unknown device* - the settled step delta of that
      toggle is a model-free measurement of its running draw;
   2. records a settled background baseline;
   3. *Switch it back ON* - records the mix for `--teach-record-s` seconds
      (extended while the signal still ramps or cycles);
   4. three estimates of the settled draw must agree: the off-step delta,
      the settled ON level minus the baseline, and the engine's own residual
      history since the unknown was detected (free evidence, no user
      action). If they agree, one toggle was enough. If not, a background
      device changed mid-capture: the flow asks for up to two more quick
      off/on toggles and takes the robust median across all estimates;
      stretches whose level disagrees with the consensus are excluded from
      the saved recording instead of poisoning it.
   Before saving, the background's noise is shrunk out of the settled part
   of the subtracted signal (fluctuations scaled to the device-only share
   `sqrt(var_mix - var_baseline)`), so the training features (P_std,
   P_min/max, THD stats) describe the device, not the background. A cycling
   device keeps its swings; a steady device comes out as flat as a guided
   recording. The switch-on transient is kept exactly as measured. The
   device's own current-THD is recovered by RSS subtraction of harmonic
   currents (mix minus baseline; the per-sample spectrum energy is used
   directly when the meter delivers it, the THD%-derived estimate is the
   fallback), and the saved S/PF are rebuilt INCLUDING that distortion
   component: with plain `S = hypot(P, Q)` a high-THD electronic load
   (laptop: 66 W but 135 VA, PF 0.49, THD_I ~175 %) would train with PF ~1
   and half its real VA and the window model would never match it live --
   this was exactly why in-mix-taught devices kept being mistaken for the
   coffee machine. If the harmonic reads are unavailable during a capture, a
   teach warning is raised instead of silently saving a THD-less recording.
   The saved file follows the campaign-single shape (10 s off lead, ON
   stretch(es) separated by short off gaps, 8 s off tail) and carries the
   per-estimate watts and toggle count in its metadata.
3. **Record it clean** (side panel). Plug in only the new device, name it,
   record about 60 s, stop. This uses the same session writer as the PAC4200
   monitor, including harmonics: the higher-fidelity path when you can
   isolate the device.

### Naming a residual: the harmonic-ratio gate (2026-07-15)

A device is taught at whatever it happened to draw during the capture, but a
switching supply's draw depends on battery and CPU state: the laptop taught
at 66 W idles at ~43 W. There the signature table does not merely find it
ambiguous, it does not find it at all (its P-term is 1.37, past the 1.0
cut), so the nearest watt-twin wins outright -- `coffee_machine_standby`
(46 W) was named at conf 0.91, with an empty contender list, on a bench with
no coffee machine plugged in. No confidence floor helps: the window-model
arbitration only runs when the match is ambiguous, and this match was
confident and alone.

The residual probe therefore also carries the residual's own **harmonic
ratio**: its harmonic current (measured spectrum energy, RSS-minus the known
harmonic currents of everything already ON) over its own fundamental. Unlike
watts, that ratio does not scale with load -- measured on this bench:

| device | THD ratio |
|---|---|
| toaster / lamp / boiler / coffee brewing | 0.01 - 0.02 |
| fans | 0.10 - 0.13 |
| coffee machine standby | 0.22 |
| **laptop** | **1.72** (8x the nearest other family) |

`match_edge` uses it only when a probe passes `thd=` (the residual path; the
edge path is untouched). It does two things:

* **rules out** a signature whose ratio disagrees. The gate bites only on
  LARGE ABSOLUTE disagreement -- the tolerance floors at 0.20 and otherwise
  follows the SMALLER of the two ratios -- so the meter's ~2.3 % THD floor
  and the coffee machine's cycle-varying harmonics (both moves of a few
  percentage points) can never veto a correct match. That fragility is what
  disabled the older IH-in-amps term (`--ih-matching`, still off).
* lets a device whose fingerprint is **distinctive** (ratio >= 0.5) and
  matches tightly **taper** down toward idle, so it is recognised across its
  operating range rather than only at the watts it was taught at. Such a
  match is capped at conf 0.85: another switching supply of similar size
  would show the same ratio, so a device matched AT its own watts always
  outranks it, and the unknown prompt can still win.

Covered by `test_residual_match.py`.

### Power thresholds, and why PV does not work yet (2026-07-15)

The engine's floors were tuned for large positive loads, which left a gap
small devices fell straight into. A 14 W monitor is big enough for the edge
detector (`--edge-min-w` 8 W) but used to be below BOTH the unknown-claim
floor (15 W) and the unknown prompt (`--unknown-min-w`, 30 W) -- so it could
never be claimed, never prompted, and therefore never taught, because the
Teach button is driven by the prompt. The floors are now ordered
`edge <= claim <= prompt` (8 / 8 / 12 W) and a test asserts that ordering so
the gap cannot silently reopen. The prompt floor only bites on a quiet
bench: with load running the threshold rises with it (`--unknown-frac`), and
the meter idles at ~0.4 W with `--unknown-persist-s` of persistence required,
so noise does not reach it.

The stale-claim guard had the same disease. Its tolerance floored at a flat
30 W -- larger than the entire claim of every small device here -- which made
them IMMORTAL: a 17 W `table_fan` claim satisfied `17 <= 0 + 30` with the
meter reading 0.0 W and stayed "on" forever once its off-step was masked by a
big device switching at the same moment. The floor now anchors to the
smallest step the edge detector can see (`_claim_slack_W`). Big claims are
unaffected -- they are carried by the proportional 10 % term.

### Generation (PV): signed claims (2026-07-15)

Claims are **signed**: a consumer's watts are positive, a generator's
negative. They used to store `abs(dP)`, which threw away the one feature that
identifies generation -- PV exporting 18 W was claimed as an 18 W *consumer*,
and `_reconcile_claims` then killed the claim on sight because the meter read
-16 W. (Seen live: `residual_matched pv` immediately followed by
`claim_dropped pv - edge claim exceeds measured power`.)

**The sign of a step cannot say whether a device started or stopped.** A
consumer starting and a generator STOPPING are the same event at the meter --
both are +P -- and so are a consumer stopping and a generator starting. The
old rule (`"on" if dP > 0 else "off"`) simply assumed every device consumes,
so PV starting to export read as a device switching OFF and its -8.5 W step
was probed as +8.5 W and pinned on the table fan, force-holding a fan nobody
touched. `_read_edge` now matches BOTH readings and lets the better fit
decide:

* **START** -- some family's signature equals the step itself
* **STOP** -- some family's signature equals the step negated

Where they fit equally well (a fan starting looks exactly like PV stopping),
the tie goes to the reading that resolves a device already believed to be
running: a device that is not on cannot stop. With no START reading at all
the STOP reading stands unchallenged, which is how a device that was already
running when the engine started still releases its watts.

Consequences threaded through the bookkeeping: the drop-matcher compares
`dP + claim_W` (a claim of W ends with a step of -W, so a 950 W boiler stops
at -950 and PV stops at +18); ramp-folding only merges steps of the same
polarity; `_reconcile_claims` sums signed claims, so PV's export offsets the
loads exactly as it does at the meter, and only CONSUMERS are eviction
candidates -- dropping a generator would RAISE the claimed net and the loop
would evict every load to fix an over-claim caused by nothing.

**Known limit:** an UNTAUGHT generator raises no unknown claim. With no
signature matching either polarity, a -260 W step is equally "an untaught
generator started" and "an untaught 260 W load stopped", and guessing
generation would let an untaught toaster that was already running when the
engine started be recorded as 700 W of generation out of nothing. The watts
stay in the residual and raise the unknown prompt, which is how the device
gets taught; once taught, the signature resolves the polarity. Pinned by
`test_claims.py`.

**PV produces two different kinds of step, and only one is an event.**
Plugging the inverter IN raises both P and Q (its ~-16 var is a fixed
property of the inverter, present whenever it is connected). After that only
P moves, as the sun does -- so a generation RAMP is a step of `(dP, dQ ~ 0)`,
and that is *exactly* what a small switching supply switching off looks like.
Measured live: a PV ramp of `(-13.7 W, +1.0 var)` matched `monitor off` at
**d=0.00** -- a perfect fit to a device nobody touched. No (P, Q) rule can
separate them, because in (P, Q) they are the same event.

The harmonic current can. A device TAKES ITS HARMONICS WITH IT when it
switches, so its off-step carries them; a ramp carries none, because nothing
switched. `match_edge`'s IH term (and `_ih_contradicts`, which guards the
release path the same way) therefore applies whenever a signature's harmonic
RATIO is distinctive -- at any step size, without `--ih-matching`.

The gate is the RATIO, never the absolute amps, and this is the trap: **IH
scales with device size**. The monitor has enormous distortion (THD 173 %)
but draws only 15 W, so its harmonic CURRENT is just 0.114 A -- *below* the
coffee machine's 0.128 A at THD 2 %. No IH threshold can judge the monitor
without also judging the coffee machine, whose harmonics vary 5x across a
brew and which is precisely what disabled `--ih-matching`. THD separates the
two by 86x. On this bench exactly one family clears THD_DISTINCTIVE
(monitor, 1.73); coffee machine, boiler, lamp, fans and PV are all exempt,
so the term cannot repeat the regressions that shelved it.

**Still open for PV:**

* The guided teach waits for `abs(w) < 5 W` to call the mains empty, which
  never happens while PV generates more than 5 W. An unswitchable PV is a
  reason to use the in-mix teach, which needs no all-off step.
* `explained_frac` divides by `|total|`, so when load nearly cancels
  generation the total approaches zero and the figure becomes meaningless.
* PV's output tracks the sun across a huge range while its signature is
  taught at one operating point (recorded -4.6..-13.3 W settled, observed
  live at -30 W -- past the range, so no match). This is the same
  variable-output problem as the laptop, but PV cannot use the harmonic
  taper: its THD is 0.25, indistinguishable from `coffee_machine_standby`'s
  0.22. What DOES identify PV uniquely is its sign -- no consumer generates.

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
