# NILM Project - PAC4200 Live Monitor & Session Recorder

**Version:** 1.0 (documents `pac_reader.py` as built, app version 1.0.0)
**Milestone:** 2
**Companion to:** `01_data_format.md`, `03_aggregator.md`, `04_preprocessor.md`
**Owners:** Soheil Ayati, Marc Steffgen
**Last updated:** 2026-07-05

---

## 1. What it is

`Scripts/PAC4200_reader/pac_reader.py` is a single-process tool that:

1. Opens **one persistent Modbus TCP connection** to a Siemens SENTRON PAC4200 and keeps it alive (automatic reconnect with health monitoring).
2. Continuously polls the meter into a rolling buffer for a **live browser dashboard**.
3. Lets you start and stop **named recording sessions** per appliance without restarting the program or dropping the meter connection: plug in the fridge, record "fridge", stop; plug in the lamp, record "table_lamp_on", stop; and so on.
4. Writes each recording to an HDF5 file (layout (c) in `01_data_format.md`) that runs through `preprocessor.py` and the MS2 pipeline unchanged.
5. Serves the dashboard with live P / Q / I charts, a per-appliance session panel, and a raw register inspector for commissioning.

Dependencies: `h5py` and `flask` are required; `pymodbus` is required for a real meter but optional with `--simulate`. Earlier revisions of this document described scripts named `pac4200_reader.py` / `pac_monitor.py` with a queue-based two-thread design and holding-register harmonics; none of that matches the code. This document is authoritative for the as-built tool.

---

## 2. Architecture

One process, two responsibilities:

- **Acquisition thread** (`AcquisitionService`, a single daemon thread): owns the meter connection, polls one full sample per tick at the configured `--rate` (default 5 Hz), and pushes each sample into
  - a rolling live buffer, a `deque(maxlen=1500)` (about 5 minutes at 5 Hz) that feeds the dashboard charts, and
  - the active recording session's incremental HDF5 writer, if a session is running.
- **Flask web thread**: serves the dashboard and JSON API on `127.0.0.1:8200`. API handlers only flip state on the acquisition service (connect, disconnect, start/stop session); they never talk Modbus themselves, except the register inspector which issues a raw read through the same reader.

There is **no queue and no second acquisition thread**. Writing is cheap because the writer buffers in memory and flushes every 150 samples (about 30 s at 5 Hz), so acquisition and file I/O share one thread without timing problems.

**Connection health and reconnect.** Connection states are `disconnected`, `connecting`, `connected`, `reconnecting`, `error`. On connect failure the thread retries with exponential backoff starting at 1 s and capped at 15 s. While connected, every failed poll increments a consecutive-failure counter; **10 consecutive failures** flip the state to `reconnecting` and the backoff loop takes over. Starting/stopping recordings never tears the connection down.

**Timing.** The poll loop targets fixed ticks (`next_tick += 1/rate`); if a poll overruns, the loop resynchronizes rather than accumulating drift. The effective rate (from the last 50 inter-sample intervals) is shown in the dashboard header.

**Gap handling.** A failed poll during a recording is stored as a **NaN sample** with a timestamp extrapolated from the nearest successful poll. Gaps are therefore visible to `preprocessor.py` as NaN runs (imputed if 5 samples or shorter), and counted as `n_gaps` in the recording summary. Values are never repeated or interpolated at acquisition time.

---

## 3. Modbus protocol details

### 3.1 Connection and decoding

Modbus TCP, default port 502, unit id 1, 1 s request timeout. Measured values are IEEE 754 32-bit floats spanning two consecutive 16-bit holding registers, decoded with `struct` (independent of the pymodbus version; the code also adapts to the pymodbus 3.x `slave` vs `device_id` keyword rename automatically).

- `--swap big_word_first` (default, correct for the PAC4200): high word first.
- `--swap little_word_first`: low word first.
- `--address-offset N` adds N to every register address; use `-1` if all values look shifted (the classic 0-based vs 1-based Modbus off-by-one).

First commissioning check: `V_L1` should read about 230 V. Values like 1e-40 or 1e38 mean wrong word order or wrong addresses.

### 3.2 Verified core register map (FC 0x03)

The core map was verified against the Siemens SENTRON PAC4200 "Measured variables without a time stamp (function codes 0x03/0x04)" table and against the meter display. Each channel is one 32-bit float (2 registers) at these Siemens offsets:

| Channel | Offset | | Channel | Offset |
|---|---|---|---|---|
| V_L1 / V_L2 / V_L3 (L-N, V) | 1 / 3 / 5 | | PF_L1 / PF_L2 / PF_L3 (signed) | 37 / 39 / 41 |
| V_L12 / V_L23 / V_L31 (L-L, V) | 7 / 9 / 11 | | THD_V_L12 / L23 / L31 (%, **line-to-line**) | 43 / 45 / 47 |
| I_L1 / I_L2 / I_L3 (A) | 13 / 15 / 17 | | freq (Hz) | 55 |
| S_L1 / S_L2 / S_L3 (VA) | 19 / 21 / 23 | | V_avg_LN / V_avg_LL / I_avg | 57 / 59 / 61 |
| P_L1 / P_L2 / P_L3 (W) | 25 / 27 / 29 | | S_total / P_total / Q_total / PF_total | 63 / 65 / 67 / 69 |
| Q_L1 / Q_L2 / Q_L3 (var) | 31 / 33 / 35 | | unbalance_V / unbalance_I (%) | 71 / 73 |

Each poll reads the whole map in **two block reads**: offsets 1..48 (block A) and 55..74 (block B). The split deliberately skips the reserved registers 49..53 so a single reserved register can never fail the whole read.

**Not in the core map:** `I_N`, `cosphi_*`, per-phase `THD_I_*`, and per-phase line-to-neutral `THD_V_L1..L3`. THD current and cos phi are scalars in a higher, firmware-dependent register region. `EXTENDED_CHANNELS` (an empty dict in the source) is the hook for them: confirm an address with the register inspector against the meter display, add it there, and it flows into the live data and recordings like any other scalar channel.

`--print-register-map` prints the full map, block layout, and harmonic file numbers, then exits.

### 3.3 Harmonics: Modbus FC 0x14 file records, opt-in

Per-order harmonics on this PAC4200 are **not** a plain register block (every read in the 36000+ region returns Illegal Data Address). They are served via **Modbus FC 0x14 "Read File Record"**: one file per quantity, with the 64 orders stored at 1-based register offsets 1, 3, 5, ... (order n at offset 2n-1). Orders 2..40 (`N_HARMONICS = 39`) start at offset 3 and span 78 registers per read.

Configured file numbers in the source (verify per meter with `tools/probe_filerecord.py` and the meter's harmonic displays, 17.0 = Harmonic I, 15.0 = Harmonic U L-N):

| Quantity | L1 | L2 | L3 | Notes |
|---|---|---|---|---|
| `HARMONIC_I_FILE` (current, A) | **113** | 0 | 0 | L1 VERIFIED 2026-07-06 against a live load: fundamental matched I_L1 (0.078 A) exactly, orders consecutive, even orders ~0, odd orders decaying; derived THD_I_L1 = 10.6 % for the table fan. L2/L3 cannot be identified while no current flows on those phases (all appliances are on L1); re-run `tools/verify_harmonics.py` with a load there if needed. The old guess 101/102/103 returns an index table (float32 denormals), which poisoned every earlier harmonic recording |
| `HARMONIC_V_FILE` (voltage L-N, % of fundamental) | 110 | 116 | 118 | identified via 231.9 V fundamentals and clean odd-order dominance. File 123 mirrors 113 with odd orders only; 120/126/128 mirror the voltage files (not used) |

`_read_harmonic_block()` now rejects blocks whose nonzero values are all denormals, so a wrong file number can no longer poison recordings. In addition, the reader probes offsets **49/51/53** (THD-R I L1..L3 in the PAC3200/PAC4200 family map, previously treated as a reserved gap) once per connect and enables `THD_I_L1..L3` scalar channels automatically when all three decode as plausible percentages -- on this meter/firmware they read NaN, so they stay disabled and live THD_I is derived from the file-113 spectrum instead.

A file number of 0 disables that quantity (its arrays stay zero). Harmonic reading is **off by default**; enable with `--harmonics`.

**Magnitudes only.** The meter provides per-order magnitudes, not per-order phase. The `I_phase_*` / `V_phase_*` datasets exist only to keep the array shape compatible with synthetic scenarios; they are always zero, and the file records `harmonic_phase_captured = False` in metadata so downstream code cannot mistake them for real zeros.

---

## 4. Web dashboard and API

The dashboard is a single embedded HTML page (no external assets, works in an air-gapped lab) served at `http://127.0.0.1:8200/` (`--web-port` to change). The browser opens automatically unless `--no-browser` is given. Panels:

- **Header:** connection state, host, effective poll rate, uptime, total samples and dropped polls, Connect / Disconnect buttons.
- **Recording session panel:** appliance label field, Start/Stop, live sample/gap counters for the active session, and a history of completed sessions in this run.
- **Live readings** (updated every 800 ms) and three canvas charts fed by `CHART_CHANNELS`: an Active-power chart (P_total, P_L1, P_L2, P_L3), a Q_total chart, and a per-phase current chart (I_L1, I_L2, I_L3).
- **Register inspector:** read an arbitrary register span and see raw words plus the decoded float in both word orders. This is the commissioning tool for confirming extended-channel addresses.

JSON API:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/status` | GET | connection state, rates, counters, active session, session history |
| `/api/live` | GET | chart series (downsampled to at most 600 points) + latest values |
| `/api/connect` | POST | request connection |
| `/api/disconnect` | POST | stop any session, disconnect |
| `/api/session/start` | POST `{label}` | start a named recording (requires connected state; one at a time) |
| `/api/session/stop` | POST | finalize the recording file |
| `/api/inspect` | POST `{address, count, swap}` | raw register read, count clamped to 2..100 |

Real meter reads contain NaN on unpopulated channels (e.g. L2/L3 for a single-phase load). Python's default JSON encoder would emit the bare token `NaN`, which is invalid JSON and silently breaks the frontend, so the app installs a `SafeJSONProvider` that serializes every non-finite float as `null` on every endpoint.

---

## 5. Recording file format

Each session writes `<safe_label>_<YYYYmmdd_HHMMSS>.h5` into `--output-dir` (default `recordings`, **relative to the current working directory**, so run the tool from `Scripts/PAC4200_reader`). The label is sanitized to lowercase `[a-z0-9_-]`.

The `IncrementalHDF5Writer` appends to resizable, LZF-compressed datasets with chunk size `(512,)` and flushes every 150 samples, so a crash loses at most one flush window and even an interrupted recording is a valid file.

```
<label>_<timestamp>.h5
|-- /timestamp                     int64 (N,)   microseconds, UTC epoch
|-- /measurements/<channel>        float32 (N,) one per core channel (section 3.2)
|-- /measurements/harmonics/       only when --harmonics
|   `-- {I_mag,I_phase,V_mag,V_phase}_{L1,L2,L3}   float32 (N, 39)
`-- /metadata (attrs)
    |-- format_version "0.1", app_version "1.0.0"
    |-- sample_rate_hz, anchor_datetime (UTC ISO 8601, session start)
    |-- source "pac4200_monitor", appliance_label
    |-- channels (JSON list), harmonics_enabled (bool)
    |-- harmonic_orders (JSON [2..40]) and harmonic_phase_captured (False)   [with --harmonics]
    `-- recording_summary (JSON, written on close):
        appliance_label, duration_s, n_samples, n_gaps,
        configured_sample_rate_hz, harmonics_enabled, completed_utc
```

Failed polls appear as NaN rows with estimated timestamps (section 2). There is no `/ground_truth` group; the appliance identity lives in `appliance_label`. Downstream: `python preprocessor.py --input <file>` runs unchanged, and `mix_measured_scenarios.py` (see `03_aggregator.md` section 10) converts these recordings into ground-truth training scenarios.

---

## 6. CLI reference

```
python pac_reader.py [options]

--host IP               PAC4200 address; required unless --simulate
--port INT              Modbus TCP port (default 502)
--unit-id INT           Modbus unit/device id (default 1)
--rate HZ               poll rate (default 5.0)
--swap MODE             big_word_first (default) | little_word_first
--address-offset INT    added to every register address (default 0; try -1)
--output-dir DIR        folder for per-appliance .h5 files (default "recordings")
--harmonics             also record per-order harmonic magnitudes via FC 0x14;
                        needs verified HARMONIC_I_FILE / HARMONIC_V_FILE numbers
--simulate              synthetic meter, no hardware or pymodbus needed
--web-port INT          dashboard port (default 8200)
--no-browser            do not auto-open the browser
--headless              record one appliance with no UI (uses --label/--duration)
--label NAME            headless: appliance label (default "appliance")
--duration SECONDS      headless: seconds to record (default 60)
--print-register-map    print the verified map and harmonic files, then exit
```

Typical invocations:

```bash
# Smoke-test the whole path + UI without hardware
python pac_reader.py --simulate

# Real meter with dashboard
python pac_reader.py --host 192.168.168.1

# Record harmonics too (after verifying file numbers)
python pac_reader.py --host 192.168.168.1 --harmonics

# Unattended long recording, no UI
python pac_reader.py --host 192.168.168.1 --headless --label fridge --duration 3600
```

**Simulation mode** replaces the Modbus reader with a deterministic synthetic meter (plausible V/I/P/Q with slow sinusoidal drift; with `--harmonics`, odd-order-dominant magnitude spectra that scale with the simulated load). It exercises the full file-writing and UI path but is not a source of training data; the synthetic generator/aggregator path serves that purpose.

**Headless mode** connects, records one labelled session for `--duration` seconds (Ctrl-C stops early), prints progress every 2 s, writes the same file format, and prints the recording summary plus the follow-up preprocessor command.

---

## 7. Commissioning tools (`Scripts/PAC4200_reader/tools/`)

- **`probe_filerecord.py`** : the maintained commissioning tool. Locates the PAC4200 harmonic data via FC 0x14. Mode A scans candidate file numbers to find files that answer; Mode B (`--file N`) dumps a file so you can see where each quantity's spectrum (a large fundamental followed by decaying orders) sits. Key findings baked into `pac_reader.py`: Siemens file records are 1-based (record 0 always errors), voltage L-N spectra were identified as files 110/116/118; the 101/102/103 guess for current was later disproven with real loads. Use this whenever a new meter or firmware revision is commissioned.
- **`verify_harmonics.py`** : one-pass verification, run WITH A LOAD ON. Reads the core registers for ground truth, checks the THD-R I candidate registers 49/51/53, scans FC 0x14 files for one whose order-1 fundamental matches a measured phase current (that is the current-harmonic file, and the match settles the phase order), scans the reported plain-register block (~11007), and prints the exact `HARMONIC_I_FILE` line to paste into `pac_reader.py`.
- **`find_harmonics.py`** : historical register-block scanner (FC 0x03 and 0x04) from when the harmonics were assumed to be a plain register run. Superseded by the file-record approach; kept for reference.
- **`probe_harmonic_registers.py`** : historical probe of a plain-register location around 11007..11091 reported by another group. Also superseded; kept for reference. Its README-style docstring documents why the register hypotheses failed.

The old `script.py` helper was deleted; the accurate parts of the former `Docs/Recorder_Readme.md` are folded into this document.

---

## 8. Practical recording workflow

Session labels double as ground-truth labels for the MS2 pipeline, so naming is load-bearing:

- **Single device:** `<device>_<setting>`, e.g. `table_fan_med`, `standing_fan_high_no_rotation`, `water_boiler_on`, `coffee_machine_run`. The MS2 `parse_family` helper strips trailing setting tokens, so all `standing_fan_*` variants collapse to the `standing_fan` family.
- **Simultaneous multi-device mixes:** join labels with a **double underscore**, e.g. `water_boiler_on__table_lamp_on`, `pv__standing_fan_high`. The double underscore is how `mix_measured_scenarios.py` and the MS2 pipeline recognize a mixed recording (no per-device ground truth; used as real test input for inference, never as training source).
- **Held-out test runs:** prefix with `test_` and keep them in `recordings/test/` so `mix_measured_scenarios.py --exclude 'test_*'` (or simply pointing at the main folder) keeps them out of training.
- **Retired recordings** (old label scheme such as `stand_cooler_*`, aborted runs) live in `recordings/old/`.

Recommended procedure per appliance: connect once, then for each device plug it in, set its state, start a session with a precise label, record at least 30-60 s of steady operation (include the switch-on if possible), stop, and move on. The meter connection stays up across all sessions.

**Manuals:** `Datasheets/manual_pac4200_system_manual.pdf` (English system manual, includes the Modbus measured-variables tables) and `Datasheets/manual_pac4200_de-DE_de-DE.pdf` (German). Siemens' own guidance applies: verify the first reads against the device display before trusting any number.

---

## 9. Commissioning checklist

1. **Connectivity.** `ping <host>` succeeds; `python pac_reader.py --host <ip>` reaches state `connected`.
2. **Float decoding.** `V_L1` about 230 V, `freq` about 50 Hz. Garbage values: try `--swap little_word_first`, then `--address-offset -1`.
3. **Cross-check against the display.** Compare 3-5 channels (V, I, P_total) with the meter's own display; class 0.2 accuracy means they should agree closely.
4. **Effective rate.** Watch the header pill; if the effective rate sits well below the configured `--rate`, lower the rate to what the link sustains.
5. **Extended channels (optional).** Use the register inspector to confirm THD-I / cos-phi addresses for the installed firmware before adding them to `EXTENDED_CHANNELS`.
6. **Harmonics (optional).** Run `tools/verify_harmonics.py` with a load drawing current; it identifies the current files, the phase order, and the THD-R I registers in one pass and prints the `HARMONIC_I_FILE` line to paste into `pac_reader.py`. Then record with `--harmonics`. Remember: magnitudes only.

---

## 10. Known limitations

1. **One meter per process.** Multi-meter setups need multiple instances and separate output merging.
2. **Core channel set differs from synthetic scenarios.** No I_N, cos phi, or per-phase THD; THD_V is line-to-line. Models meant to transfer between synthetic and real data must restrict themselves to the common channels.
3. **Harmonic phases are not measurable** on this meter over Modbus; harmonic-phase features remain synthetic-only.
4. **Timestamps come from the recording host's clock** at poll time; NTP synchronization is assumed, millisecond-scale drift accepted.
5. **Recordings do not auto-split.** One session = one file; long campaigns are run as multiple sessions.
