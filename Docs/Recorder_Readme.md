# PAC4200 Live Monitor + Per-Appliance HDF5 Recorder

A single-file tool (`pac_monitor.py`) that holds one persistent Modbus TCP
connection to a Siemens SENTRON PAC4200, shows a live dashboard with charts,
and records each appliance to its own `.h5` file — without restarting the
script between appliances. Output files match the scenario layout your
`preprocessor.py` already reads.

## Install

```bash
pip install pymodbus h5py flask numpy
```

(Tested against pymodbus 3.13, h5py 3.16, flask 3.1. The float decode and the
unit-id keyword are version-detected at runtime, so older pymodbus 2.x/3.x also
works.)

## Run

Try it with no hardware first:

```bash
python pac_monitor.py --simulate
```

Then point it at the real meter:

```bash
python pac_monitor.py --host 192.168.168.11
```

A browser tab opens at `http://127.0.0.1:8200`. In the UI:

1. **Connect** — opens the persistent connection; charts start streaming.
2. Type an appliance label (e.g. `fridge`), press **Start recording**.
3. Swap the appliance on the meter, **Stop**, relabel, **Start** again.
   The connection stays up the whole time; each session is a separate file.

Files land in `./recordings/` as `<label>_<timestamp>.h5`.

Useful flags: `--rate` (Hz, default 5), `--output-dir`, `--web-port`,
`--unit-id`, `--no-browser`. Headless recording without the UI:
`--headless --label fridge --duration 600`.

## Verify the registers (do this once, ~2 min)

The **core** register map below is taken from the Siemens PAC4200 Modbus
documentation and is what the script reads by default:

| Channel | Reg | Channel | Reg | Channel | Reg |
|---|---|---|---|---|---|
| V_L1/L2/L3 | 1/3/5 | S_L1/L2/L3 | 19/21/23 | freq | 55 |
| V_L12/L23/L31 | 7/9/11 | P_L1/L2/L3 | 25/27/29 | V_avg LN/LL | 57/59 |
| I_L1/L2/L3 | 13/15/17 | Q_L1/L2/L3 | 31/33/35 | I_avg | 61 |
| PF_L1/L2/L3 | 37/39/41 | THD_V LL | 43/45/47 | S/P/Q/PF total | 63/65/67/69 |

Registers 49/51/53 are reserved on the PAC4200 and are not read.

Open the **Register Inspector** panel at the bottom of the dashboard, read a
few registers, and compare the decoded value to the same value on the meter's
physical display:

- If `V_L1` (reg 1) reads ~230 V under **big-word** decoding, the map and word
  order are correct. You're done.
- If every value is shifted by one register, your meter uses 1-based
  addressing — restart with `--address-offset -1`.
- If values are garbage under big-word but clean under little-word, restart
  with `--swap little_word_first`.

## THD current / cosφ / harmonics — not enabled yet

These are **not** in the basic measured-values block; they sit in a
higher register region that varies by PAC4200 firmware. I did not ship guessed
addresses for them, because a wrong address silently records plausible-looking
nonsense. The earlier version of your script had guessed THD_I/harmonic
addresses that decode to the wrong quantities — those are removed here.

To add them: use the Register Inspector to find the address where THD_I_L1
matches the meter display, then uncomment and fill in the entries in the
`EXTENDED_CHANNELS` dict near the top of `pac_monitor.py`. They'll be recorded
automatically on the next session.

Note: for a single appliance like your 6.3 W LED lamp you can also derive total
THD_I from the recorded power factor and displacement cosφ:
`THD_I = sqrt(1/(PF/cosφ)² − 1)`. That's exact and needs no extra registers —
useful as a cross-check once you've verified a THD_I register.

## Output file structure

```
/timestamp                      int64, microseconds (UTC epoch)
/measurements/<channel>         float32, one per core channel above
/measurements/harmonics/...     float32 (n, 39), only if harmonics enabled
/metadata                       attrs: appliance_label, sample_rate_hz,
                                anchor_datetime, source, recording_summary
```

Gaps (dropped polls) are written as `NaN` with an interpolated timestamp, so
`preprocessor.py`'s gap-imputation has something to do on real data — which it
didn't on the synthetic files.
