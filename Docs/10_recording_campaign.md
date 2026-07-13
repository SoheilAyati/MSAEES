# 10 – Recording campaign: rebuilding the corpus consistently

Date started: 2026-07-13 · Script: `Scripts/MS2_Pipeline/record_campaign.py`

## Why re-record

The 2026-07-06 corpus made the system work, but three of its gaps now bound
every metric, and the recordings were made ad-hoc (varying lengths, devices
switched at varying moments, some pairs switched simultaneously):

1. **PV is untrainable** – `pv_only` captured ~0 W generation. Every
   `pv__*` mix loses set-F1 for a reason no model can fix.
2. **Coffee standby is invisible** – the only coffee single shows the
   ~1180 W brew; in real mixes the machine idles at ~35 W warm-hold and the
   model misses it (4 of the 6 coffee mixes fail exactly this way).
3. **Fan modes were never recorded as *transitions*** – the live engine now
   matches small settled steps (−5 W) against mode changes; recordings of the
   actual `high→low` walk validate that path and teach the window model
   intra-family steps.

One protocol for every item removes the per-file quirks: fixed off-baseline,
fixed ON time, auto-detected switch moments, harmonics always on.

## How to record

```bash
cd Scripts/MS2_Pipeline
python record_campaign.py --list                     # see the plan
python record_campaign.py --host 192.168.168.1       # run it (guided)
python record_campaign.py --host ... --only 20-21    # sunny-day items only
python record_campaign.py --host ... --skip-existing # resume the same day
python record_campaign.py --simulate --fast          # rehearse the flow
```

The script waits for the measured power itself – you only plug things in and
out when the prompt says so. Ctrl-C aborts an item and deletes the partial
file. Recordings land in the shared `Scripts/PAC4200_reader/recordings/`.

**Before you start:** move the previous top-level recordings into a dated
subfolder (e.g. `recordings/old_06-07-2026b/`) *after* the campaign is
complete and validated – until then the old files keep the system working.

## The plan (~60–75 min, plus one sunny slot)

| # | Item | Kind | Time | Why |
|---|------|------|------|-----|
| 1–2 | `table_fan_low/high` | single 75 s | mode table for the small-step matcher |
| 3–5 | `standing_fan_low/med/high` | single 75 s | 3 clean speeds, rotation OFF |
| 6 | `standing_fan_high_rotate` | single 75 s | teaches the rotation wobble |
| 7 | `standing_lamp_on` | single 75 s | |
| 8 | `water_boiler_on` | single 90 s | filled with water |
| 9 | `coffee_machine_run` | single 150 s | a real brew, cycling included |
| 10 | `coffee_machine_standby` | single 75 s | **gap #2** – warm-hold right after the brew |
| 11 | `laptop` | single 120 s | battery not full; soft-start ramp included |
| 12 | `table_fan_low_high_low` | choreo | mode walk – trains + validates transitions |
| 13 | `standing_fan_low_med_high` | choreo | mode walk |
| 14 | `table_fan_high__standing_fan_high` | choreo | both fans ON **at the same moment** (composite-edge test) |
| 15 | `standing_fan_high__table_fan_high_low` | choreo | **the complaint scenario**: both high → table fan to low |
| 16 | `standing_fan_high_low__table_fan_high` | choreo | mirror: standing fan is the one turned down |
| 17 | `water_boiler_on__table_fan_high` | choreo | big + small, staggered |
| 18 | `coffee_machine_run__standing_lamp_on` | choreo | brew next to a 500 W lamp |
| 19 | `water_boiler_on__standing_lamp_on__table_fan_high` | choreo | three devices |
| 20 | `pv_only` | single 180 s | **gap #1** – SUNNY DAY, total must go clearly negative |
| 21 | `pv__water_boiler_on` | choreo | SUNNY DAY – negative + positive mixed |

Items 14–19 are *validation* mixes: the mixer never trains on `__` files;
they score the model (`infer.py` set-accuracy) and the engine
(`live.py --replay <file>`).

## After the campaign

1. Retrain: dashboard **Retrain now** (or `train.py --task mix` +
   `--task identify`). The feature policy is automatic – with a fully
   harmonic corpus you can also try `--agg-features harm` and compare
   real-mix set-F1 before adopting it (base-17 won 0.767 vs 0.674 on the
   2026-07-06 corpus).
2. Replay items 14–16 through `live.py --replay` – expect the composite
   `edge_on` for both fans and a `mode_change` event naming the right fan.
3. If everything scores, freeze the new originals (copy the three
   `*_original` files) and archive the old top-level recordings.
