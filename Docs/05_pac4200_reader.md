# NILM Project — PAC4200 Modbus Reader Specification

**Version:** 0.1 (draft)  
**Milestone:** 1 & 2 
**Companion to:** `data_format.md`, `appliance_generator.md`, `aggregator.md`, `preprocessor.md`  
**Owner:** Soheil Ayati, Marc Steffgen  
**Last updated:** 2026-05-13

---

## 1. Purpose and scope

The PAC4200 reader is the **real-data counterpart to the aggregator**. It polls a Siemens SENTRON PAC4200 at the Point of Common Coupling over Modbus TCP at ~5 Hz, decodes the meter's register map into the channel set defined by the data format spec, and writes a scenario HDF5 file that is byte-format-compatible with what `aggregator.py` produces.

Downstream of this script, the pipeline is identical to the synthetic path: `preprocessor.py` runs unchanged, feature engineering is the same, and Milestone 2 ML treats real and synthetic scenarios as interchangeable inputs.

```
Synthetic:                                            Real (M2):
  generators → aggregator                              PAC4200 → pac4200_reader
                       \                                              /
                        \                                            /
                         ▼                                          ▼
                                    scenario_*.h5
                                          │
                                          ▼
                                    preprocessor.py
                                          │
                                          ▼
                                    Milestone 2 ML
```

This is the architectural payoff of building everything else in Milestone 1: by the time M2 starts, only the data-acquisition step changes; the rest of the codebase is already validated against synthetic data.

**In scope:** Modbus TCP polling of PAC4200 holding registers; decoding IEEE 754 floats from register pairs; incremental HDF5 writing with crash safety; threaded acquisition with rate-limited polling; graceful shutdown; recording diagnostics; a simulation mode for testing the file pipeline without hardware.

**Out of scope:** signal cleaning, smoothing, gap imputation, feature engineering — all of those live in `preprocessor.py` and run identically on the output of this reader. Aggregating across multiple meters is also out of scope; one reader instance polls one PAC4200.

---

## 2. Position in the pipeline

The reader sits at the boundary between hardware and software. Its job is to convert "registers on the wire" into "a scenario HDF5 file on disk." Everything upstream is electrical engineering (the meter and its wiring); everything downstream is the software pipeline.

| Stage | M1 (today) | M2 (lab) |
|---|---|---|
| Data acquisition | `aggregator.py` from synthetic appliances | `pac4200_reader.py` from real PAC4200 |
| Cleaning + features | `preprocessor.py` | `preprocessor.py` (no change) |
| Classification + ML | (in development) | (in development) |

The reader supports a `--simulate` flag in M1 that fabricates Modbus responses on the fly. This is not a substitute for real hardware testing — it exists to verify the file-writing path and threading model before the meter is connected.

---

## 3. Inputs and outputs

### 3.1 Inputs

The script accepts no input files. It takes its parameters from the command line:

- Modbus connection: host, port, slave/unit ID, byte-order swap mode.
- Recording parameters: sample rate (Hz), duration (seconds), output file path.
- Optional: harmonic reading (on by default), flush interval, simulation mode and its dropout rate.

### 3.2 Output

A single scenario HDF5 file with the same layout as `aggregator.py` produces, with two intentional differences:

- **No `/ground_truth` group.** Real measurements have no per-appliance breakdown. (If sub-metering is added in the lab in M2, a separate ground-truth merging step can populate this group from sub-meter recordings.)
- **No voltage/current synthesis.** Everything in `/measurements` came directly from PAC4200 registers — no derived or synthesized channels.

The full output structure is:

```
recording_*.h5
├── /timestamp                              (1D int64, µs since Unix epoch)
├── /measurements
│   ├── V_L1, V_L2, V_L3                    (1D float32, RMS phase voltages)
│   ├── I_L1, I_L2, I_L3, I_N               (1D float32, RMS currents)
│   ├── P_L1..P_total, Q_L1..Q_total        (1D float32)
│   ├── S_L1..S_total                       (1D float32)
│   ├── PF_L1..PF_total, cosphi_*           (1D float32)
│   ├── THD_V_L1..3, THD_I_L1..3            (1D float32)
│   ├── freq                                (1D float32)
│   └── harmonics/
│       ├── I_mag_{L1,L2,L3}                (2D float32, N × 39)
│       ├── I_phase_{L1,L2,L3}              (2D float32, N × 39)
│       ├── V_mag_{L1,L2,L3}                (2D float32, N × 39)
│       └── V_phase_{L1,L2,L3}              (2D float32, N × 39)
└── /metadata
    ├── format_version                      ("0.1")
    ├── reader_version                      ("0.1.0")
    ├── sample_rate_hz                      (configured rate)
    ├── anchor_datetime                     (UTC ISO 8601, recording start)
    ├── source                              ("pac4200_reader")
    └── recording_summary                   (JSON, see §9)
```

Datasets are written with resizable shapes during recording (chunked, LZF-compressed). Once the session ends, the file is structurally identical to a fixed-shape aggregator output.

---

## 4. Modbus protocol details

### 4.1 Connection

PAC4200 exposes Modbus TCP on port 502 by default. The reader uses `pymodbus` (sync client) with a 1-second per-request timeout. The reader does not attempt automatic reconnection within a single recording session — a connection loss is treated as a sustained sequence of failed polls, each becoming a gap in the output file. Manual restart is required.

### 4.2 Float encoding

PAC4200 measured values are encoded as IEEE 754 32-bit floats spanning two consecutive 16-bit Modbus holding registers. The reader decodes them with the convention:

- **Big-word-first (default for PAC4200):** high word first, then low word; within each word, big-endian bytes.
- **Little-word-first (some Siemens installations):** low word first, then high word; bytes still big-endian within each word.

Selectable at runtime via `--swap big_word_first` or `--swap little_word_first`. The first thing to verify in M2 is that `V_L1` reads ~230 V; a value like 1.23×10⁻⁴⁰ or 1.5×10³⁸ indicates wrong byte order or wrong addresses.

### 4.3 Register grouping

The reader issues one Modbus request per channel (2 registers each) for scalar values, and batch requests of 39 floats (78 registers) for harmonic blocks. This sums to ~37 requests per poll cycle for the full channel set with harmonics, ~25 requests without.

Potential optimization (deferred to M2 if needed): group consecutive scalar channels into a single multi-register read. PAC4200 supports up to ~125 registers per request, so most of the scalar channels could be fetched in a single round-trip. Not done in v0.1 to keep the code straightforward.

### 4.4 Effective polling rate

Theoretical 5 Hz (200 ms per poll cycle) assumes a fast local network with no Modbus retries. Realistic installations land at 3–5 Hz with the full register block. The reader does not adapt its target rate — it logs `late_polls` (cycles where the previous poll exceeded its 200 ms budget) and `failed_polls` (cycles that returned no data) in the recording summary. If `late_polls / n_samples` exceeds ~5%, lower the configured `--rate` to match what the link actually sustains.

---

## 5. Register map

Channel-to-register mapping is hardcoded in the script as the `REGISTER_MAP` dictionary. Addresses are 0-based logical addresses per the Siemens SENTRON PAC4200 Modbus Measured Values documentation. The full map is printable via `--print-register-map`.

**Scalar channels** (34 in total, each occupying 2 registers):

| Channel group | Registers | Notes |
|---|---|---|
| V_L1, V_L2, V_L3 | 1, 3, 5 | line-to-neutral RMS, V |
| I_L1, I_L2, I_L3, I_N | 13, 15, 17, 19 | RMS, A. I_N is calculated by the meter |
| S_L1..L3, P_L1..L3, Q_L1..L3 | 25..41 | per-phase apparent/active/reactive power |
| PF_L1, PF_L2, PF_L3 | 43, 45, 47 | signed (PAC4200 convention) |
| S_total, P_total, Q_total, PF_total | 65, 67, 69, 71 | three-phase totals |
| freq | 55 | Hz |
| THD_V_L1..L3, THD_I_L1..L3 | 201..211 | percent |
| cosphi_L1..L3, cosphi_total | 215..221 | displacement only |

**Harmonic blocks** (sequential floats, orders 1..64 per phase):

| Group | Base register | Orders read |
|---|---|---|
| H_V_L1_mag, L2, L3 | 401, 529, 657 | 2..40 (39 values per phase) |
| H_I_L1_mag, L2, L3 | 785, 913, 1041 | 2..40 |
| H_V_L1_phase, L2, L3 | 1169, 1297, 1425 | 2..40 (optional, see §6.2) |
| H_I_L1_phase, L2, L3 | 1553, 1681, 1809 | 2..40 (optional, see §6.2) |

PAC4200 natively exposes harmonics up to the 64th, but appliance signatures decay above the 25th, so 40 is the cutoff defined by the data format spec.

---

## 6. Channel availability and conventions

### 6.1 Signed power factor

PAC4200 reports power factor as a signed value where the sign indicates direction of real power flow (positive = consumption, negative = export/generation). This matches what gets stored in `/measurements/PF_*`. It differs from the data format spec §6 which originally described unsigned PF; the reader reflects what the meter actually does, and the preprocessor's outlier bounds were already updated to match ([-1, +1]).

### 6.2 Harmonic phases — auto-detection

Some PAC4200 firmware revisions expose harmonic magnitudes over Modbus but not harmonic phases. The reader handles this without manual configuration:

1. On the first poll where harmonics are read, the reader attempts to fetch phase registers.
2. If the read fails (timeout or protocol error), the reader marks harmonic phases as unavailable and stops trying for the rest of the session.
3. Subsequent samples write zero arrays for the missing phase channels.
4. The recording summary's `failed_polls` count includes the initial probe failures.

This means the output file always has the same shape — 12 harmonic datasets, each (N × 39) — regardless of whether phases were actually measurable. Downstream code doesn't need to special-case missing-phase scenarios.

### 6.3 Calculated channels

Two of the channels stored in `/measurements/` are calculated by the PAC4200 rather than directly sensed:

- **I_N** is not measured directly; the meter computes it from the three phase currents.
- **S, P, Q totals** are computed from per-phase values inside the meter.

For our purposes these are treated as if measured. The PAC4200 documentation defines exactly what algorithm produces each, so they're consistent across firmware versions.

---

## 7. Acquisition architecture

### 7.1 Threading model

The reader uses two threads:

1. **Acquisition thread.** Polls the PAC4200 at the configured rate, drops samples onto a thread-safe queue.
2. **Main thread.** Drains the queue, accumulates samples in a write buffer, flushes to HDF5 every `flush_every` samples (default 150 = 30 s at 5 Hz).

This separation matters for long recordings. If acquisition and write were on the same thread, a slow disk flush would cause the next poll to be late, propagating into clock drift. Threading isolates the two timing budgets.

### 7.2 Why a queue and not direct writes

The queue provides bounded backpressure. If the writer falls behind, the queue grows; if it grows past the configured cap, samples will block at `queue.put()` and the acquisition thread will fall behind — this becomes a `late_polls` count. This is preferable to silently dropping samples or running out of memory.

### 7.3 Graceful shutdown

`SIGINT` (Ctrl-C) sets a stop flag rather than crashing. The main loop notices, drains pending samples from the queue, flushes the writer, writes the recording summary, and closes the file. This guarantees that even an interrupted recording produces a valid HDF5 file.

---

## 8. Gap handling at acquisition time

When a poll fails (timeout, Modbus error, connection drop), the reader does not interpolate or repeat the previous value. Instead, it writes a sample of NaN values with a synthesized timestamp computed by extrapolating from the nearest successful poll's timestamp.

**Why NaN, not the last known value:**

- The preprocessor's gap-detection logic relies on finding NaN runs. Repeating values would hide the gap from the report.
- Imputation should be explicit and centralized in the preprocessor, not duplicated at acquisition time.
- The recording summary's `n_gaps` count gives a true picture of network/meter reliability.

This means a recording with frequent dropouts will land in the preprocessor as a sequence of NaN runs; the preprocessor imputes runs ≤ 5 samples (1 second at 5 Hz) and leaves longer ones flagged. No code change between handling synthetic and real recordings.

---

## 9. Recording diagnostics

Every recording produces a `recording_summary` dictionary that is written as a JSON-encoded attribute on `/metadata`. The summary captures everything needed to diagnose acquisition issues:

| Key | Type | Meaning |
|---|---|---|
| `reader_version` | string | semver, e.g. "0.1.0" |
| `anchor_datetime` | ISO 8601 string | UTC start of recording |
| `configured_sample_rate_hz` | float | what was requested |
| `configured_duration_s` | float | what was requested |
| `actual_duration_s` | float | wall-clock duration from start to end |
| `n_samples` | int | total samples in the output file |
| `n_gaps` | int | how many of those were NaN-filled gaps |
| `failed_polls` | int | poll cycles that returned no data |
| `late_polls` | int | poll cycles that exceeded their time budget |
| `interrupted` | bool | whether Ctrl-C ended the recording early |
| `completed_utc` | ISO 8601 string | when the recording finished |

**Why this matters:** during M2, ML behavior may differ between recording sessions. The summary lets you triage immediately whether the difference is in the data quality (e.g. `failed_polls` of 20% vs 0.5%) or in something else (model, parameters, preprocessing config).

The preprocessor's report (stored separately at `/preprocessed/attrs/report`) provides the complementary view: what the cleaning step did. Together they give a complete audit trail from "registers on the wire" to "features fed to ML."

---

## 10. Simulation mode

`--simulate` replaces the real `ModbusReader` with a `SimulatedReader` that produces deterministic synthetic responses (~constants with slow sinusoidal drift on voltage, current, frequency, and harmonics decaying as 1/order^1.2).

**What simulation mode is for:**

- Verifying the file-writing path produces a valid scenario file.
- Stress-testing the threading model under high poll rates.
- Exercising the gap-handling path via `--sim-dropout-rate 0.1`.
- CI / smoke tests that don't depend on lab hardware.

**What simulation mode is NOT for:**

- Producing data for Milestone 2 ML training. The values are not physically realistic — they are constants. The aggregator path (`aggregator.py` running on appliance generators) is the proper source of synthetic training data.
- Estimating real PAC4200 timing characteristics. Simulation polls return instantly; real ones take 5–20 ms each.

The recording summary records `"source": "pac4200_reader"` in either mode, but the file contents make the difference obvious — simulated recordings have flat P_total ≈ 2.37 kW, no realistic appliance activity.

---

## 11. Configuration

| CLI flag | Default | Purpose |
|---|---|---|
| `--host` | required (unless simulate) | PAC4200 IP address |
| `--port` | 502 | Modbus TCP port |
| `--unit-id` | 1 | Modbus slave ID |
| `--swap` | `big_word_first` | float byte/word order |
| `--rate` | 5.0 | poll rate in Hz |
| `--duration` | 60.0 | recording duration in seconds |
| `--output` | `recording.h5` | output file path |
| `--flush-every` | 150 | samples per disk flush |
| `--no-harmonics` | false | skip harmonic reads (~30% faster polling) |
| `--simulate` | false | use SimulatedReader instead of real Modbus |
| `--sim-dropout-rate` | 0.0 | simulator: probability per sample of a dropped poll |
| `--print-register-map` | — | print register map and exit |

The defaults are chosen to match the data format spec (5 Hz, full channel set). For sustained recordings on a slow link, lower `--rate` and increase `--flush-every` proportionally.

---

## 12. M2 verification checklist

When the PAC4200 is connected for Milestone 2, the following checks should happen before recording anything that matters:

1. **Connectivity.** `ping <host>` succeeds, and `python pac4200_reader.py --host <ip> --duration 5` produces a non-empty file.

2. **Float decoding.** Inspect the first samples: `V_L1` should read ~230 V (±10), `freq` should read ~50 Hz (±0.5), `P_total` should be plausible for the lab's actual load (typically 50–500 W if the lab is idle). If any value reads as 0, NaN, or extreme exponents (10⁻⁴⁰ or 10³⁸), swap byte order with `--swap little_word_first` and try again.

3. **Register addresses.** Cross-check 3–5 channels against the PAC4200 display. If a stored `V_L1` differs from the display's L1 voltage by more than the meter's class-0.2 accuracy, the address may be wrong for this firmware revision. Verify against the Siemens documentation for the installed firmware.

4. **Effective rate.** Run a 60-second recording at the default 5 Hz. Inspect `recording_summary.n_samples` — expect ~300 (with `late_polls` close to zero). If samples are 200 or fewer, the link is too slow for 5 Hz; reduce `--rate` to 3 or 2.

5. **Harmonic availability.** Check whether `/measurements/harmonics/I_mag_L1` contains non-zero values and whether `..._I_phase_L1` is non-zero or all-zero. All-zero phases indicate the firmware doesn't expose them; document this in the recording summary and proceed (downstream code handles either case).

6. **Sub-meter ground truth (optional).** If sub-meters are installed on individual appliances, a separate script (not in M1 scope) will need to combine the PAC4200 recording with sub-meter recordings to populate `/ground_truth`. Without sub-meters, real recordings remain unlabeled; ML can still train on them as test data using transfer from synthetic-only training.

---

## 13. Open issues / deferred decisions

1. **Single-meter scope.** The reader handles one PAC4200 per process. Multi-meter setups (e.g., one meter per phase or per circuit) require running multiple instances and merging the output files. Not implemented; deferred until M2 confirms whether this is needed.

2. **No automatic reconnection.** A sustained network drop ends the useful part of the recording, even though the file remains valid. Auto-reconnect with backoff is feasible but adds complexity; deferred until M2 reveals whether real installations actually need it.

3. **Scalar reads are not grouped.** Each of the 34 scalar channels is fetched in its own Modbus request. PAC4200 supports multi-register reads of up to ~125 registers, which would let most of the scalar channels be read in 1–2 round-trips instead of 17. Worth implementing if the achievable poll rate is too low.

4. **No back-channel for ground-truth labels.** Real PAC4200 recordings have no per-appliance breakdown. If lab sub-meters become available in M2, a `ground_truth_merger.py` script will read sub-meter recordings and populate `/ground_truth` in the PAC4200 file. Designing that script is deferred to when sub-meter outputs are concretely available.

5. **Simulated values are not load-like.** The `SimulatedReader` produces constants plus slow drift. It does not consume the appliance generators' output. A more elaborate simulation mode could replay an aggregator file as if it were live Modbus data, but the existing synthetic→aggregator path already serves that purpose more directly.

6. **Time synchronization.** Timestamps are sourced from the local system clock at the moment each poll completes. PAC4200 also has its own internal clock accessible via Modbus, but reading it adds polling cost. We assume the recording host's clock is synchronized via NTP and accept ms-scale drift. For sub-cycle event timing (not needed at our 5 Hz rate) this would need revisiting.

7. **HDF5 file is opened in 'w' mode.** Each recording overwrites the output file. No append mode — long recordings should not be split across multiple invocations. If a recording needs to be split (e.g., 7-day campaign in 24h chunks), each invocation writes its own file and a downstream script concatenates them.

---

## Appendix A — Verification sequence run on simulation mode (M1)

The script has been validated end-to-end in simulation mode prior to PAC4200 availability. The verification sequence ran:

1. `python pac4200_reader.py --simulate --duration 10 --rate 5 --output sim.h5`
   - Wrote a 51-sample file, ~zero gaps, full channel set including harmonics.
   - File structure matches `aggregator.py` output minus `/ground_truth`.
2. `python preprocessor.py --input sim.h5`
   - Processed all 34 scalar channels, built 12 features, no warnings.
3. `python pac4200_reader.py --simulate --duration 10 --sim-dropout-rate 0.1 --output sim_gaps.h5`
   - 51 samples written; 6 of them gaps (one in 10 dropout rate).
4. `python preprocessor.py --input sim_gaps.h5`
   - Detected 6 NaNs in each of 34 channels (204 total), imputed all of them as short gaps. No code change from synthetic-data preprocessing.

These four steps confirm that the M1→M2 architectural promise holds: the same preprocessor.py operates identically on the reader's output and on the aggregator's output, with gap handling exercised at acquisition time and during cleaning.