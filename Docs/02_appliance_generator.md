# NILM Project - Appliance Generator Specification

**Version:** 0.3 (as built)
**Milestone:** 1
**Companion to:** `01_data_format.md`
**Owners:** Soheil Ayati, Marc Steffgen
**Last updated:** 2026-07-05

---

## 1. Purpose

Documents the appliance generators implemented in `Scripts/Synthetic_data_generator/Appliance_generator.py`. The script is a single entry point for generating synthetic per-appliance traces for 9 appliances. Each generator produces a clean, deterministic signature given a seed and writes a single-appliance HDF5 file (layout (a) in `01_data_format.md`), which the aggregator then combines into scenarios.

Module constants: `N_HARMONICS = 39` (orders 2 through 40), `V_NOMINAL = 230.0` V, `FORMAT_VERSION = "0.2"`, `GENERATOR_VERSION = "0.1.0"`.

---

## 2. Common framework

### 2.1 Generator interface

Every appliance is a subclass of `ApplianceGenerator`:

```python
class ApplianceGenerator(ABC):
    name: str                 # e.g. "fridge", "pc"
    appliance_type: str       # "I" | "II" | "III" | "IV" | "special"
    is_three_phase: bool
    DEFAULT_PARAMS: dict

    def __init__(self, params=None, seed=0, phase="L1", instance_id=1): ...

    def generate(self, duration_s, sample_rate_hz, anchor_datetime) -> ApplianceTrace
```

`ApplianceTrace` carries:
- `timestamp_us` : int64 microseconds since Unix epoch (UTC)
- `P`, `Q` : float32 active and reactive power contribution per sample. For three-phase appliances these are totals across phases; the aggregator splits them.
- `state` : per-sample categorical state label
- `harmonics_I_mag`, `harmonics_I_phase` : `(N, 39)` current-harmonic magnitude (A) and phase (rad), orders 2 through 40
- `metadata` : name, type, instance_id, phase, is_three_phase, seed, params, plus per-run sampled `instance_params`

For three-phase generators the `phase` argument is ignored and recorded as `"all"`.

### 2.2 Conventions

- P in watts, Q in var; sign convention per `01_data_format.md` section 2.1 (P > 0 consumption, P < 0 generation; Q > 0 inductive).
- Harmonics are current harmonics. Voltage harmonics are synthesized by the aggregator, not by appliances.
- Harmonic synthesis (`synthesize_harmonics`): each generator carries a profile `{order: fraction_of_fundamental}`. The fundamental current is derived as `I = sqrt(P^2 + Q^2) / 230`. Each order gets one random characteristic base phase per run (uniform in [-pi, pi]) plus small per-sample jitter (default sigma 0.05 rad).

### 2.3 Time-of-day model

`time_of_day_intensity(hour, profile_name)` returns an activation-intensity curve in [0, 1]. Implemented presets:

- `working_hours_weekday` : Gaussian bump centred on 13:00 (sigma 3.5 h)
- `morning_evening_peaks` : peaks around 07:30 and 19:30
- `overnight` : peaks around 02:00, low during the day
- `continuous` : constant 1.0
- `random_burst` : constant low value (0.05), i.e. bursts equally likely all day

PV does not use this helper; it has a separate `solar_intensity(hour, anchor_datetime, latitude_deg)` that computes solar elevation from day-of-year and latitude (clipped at zero before sunrise / after sunset, local noon simplified to 13:00).

### 2.4 Parameter randomization for generalization

Parameters given as a two-element list `[lo, hi]` are ranges: the generator's seeded RNG draws one value per run (`_sample_range`). Scalars are used as-is. Two runs with different seeds therefore produce different appliance "instances" of the same category, which is what enables train-on-some-instances, test-on-held-out-instances generalization experiments in MS2. The values actually drawn are recorded in `metadata.instance_params`.

### 2.5 Where noise does NOT live

Generators produce clean signatures; measurement noise, Modbus dropout, and quantization belong downstream. The one deliberate exception is the baseload generator, which adds small Gaussian jitter (`noise_W`, default 0.5 W) so the always-on floor is not perfectly flat.

---

## 3. Per-appliance specifications (as implemented)

### 3.1 Fridge (`fridge`, Type II, single-phase)

States: `off_control`, `compressor_starting`, `compressor_on`, `defrost`.

Behaviour: compressor cycles continuously with per-cycle sampled off duration then on duration; the first sample of each on period is an inrush spike (`P = compressor_P_W * inrush_factor`, `Q = compressor_Q_var * inrush_factor * 0.7`, one sample at 5 Hz). Defrost events (`defrost_per_day`) overwrite a random window with resistive heating (Q = 5 var).

| Parameter | Default |
|---|---|
| `compressor_P_W` | [120, 180] W |
| `compressor_Q_var` | [80, 140] var |
| `off_P_W` | 2.0 W (electronics floor) |
| `on_duration_s` | [480, 1080] |
| `off_duration_s` | [1200, 2700] |
| `inrush_factor` | [2.5, 4.5] |
| `defrost_P_W` | [200, 350] W |
| `defrost_duration_s` | [600, 1200] |
| `defrost_per_day` | 1 |

Harmonics while compressor active: 3rd 6%, 5th 4%, 7th 2%, 9th 1%, 11th 0.5%, 13th 0.3% of fundamental. Off and defrost: none.

### 3.2 Variable resistive load (`resistive`, Type III, single-phase)

States: `off`, `active`. During each active period the setpoint moves piecewise-linearly between random targets in [0, `P_max_W`] (a new target roughly every 30 s). Q is zero.

| Parameter | Default |
|---|---|
| `P_max_W` | [500, 3000] W |
| `ramp_rate_W_per_s` | [50, 500] (sampled and recorded; the implementation interpolates linearly between targets) |
| `active_periods` | [1, 3] per day |
| `period_duration_s` | [60, 1800] |

Harmonics: 3rd at 0.5% (essentially clean).

### 3.3 Hair dryer (`hair_dryer`, Type II, single-phase)

States as implemented: `off`, `cold_fan`, `heat_high` only (the richer six-level state table from the original design was collapsed). Each burst is structured as roughly 20 s `cold_fan`, then a heat segment split into thirds at mid / high / mid power, then about 15 s `cold_fan`. Burst start times are rejection-sampled against the `morning_evening_peaks` intensity curve.

| Parameter | Default |
|---|---|
| `P_high_heat_high_fan_W` | [1700, 2200] W |
| `P_high_heat_low_fan_W` | [1100, 1400] W |
| `P_low_heat_W` | [400, 550] W (defined but not used by the current burst logic) |
| `P_cold_fan_W` | [40, 80] W (Q = 0.3 P in fan-only; Q = 0.04 P in heat) |
| `burst_count_per_day` | [0, 2] |
| `burst_duration_s` | [120, 360] |

Harmonics: heat 3rd 1%; fan-only 3rd 4%, 5th 1.5%, 7th 0.5%.

### 3.4 PC (`pc`, Type IV, single-phase)

States as implemented: `standby`, `idle`, `active`, `peak` (there is no separate `off` or `sleep` state; the floor is `standby`). Per-sample state selection is driven by the `working_hours_weekday` intensity curve (scaled by `weekend_active_prob` when `is_weekday` is false): samples become `idle` with probability `0.8 * intensity`, 30% of those escalate to `active`, and `peak_fraction` of them to `peak`. P is then smoothed with a 3-second rolling mean to remove single-sample flicker; Q follows from the sampled power factor.

| Parameter | Default |
|---|---|
| `standby_P_W` | [2, 5] W |
| `idle_P_W` | [30, 70] W |
| `active_P_W` | [80, 200] W |
| `peak_P_W` | [250, 400] W |
| `PF_active` | [0.75, 0.9] |
| `peak_fraction` | [0.02, 0.10] |
| `weekend_active_prob` | [0.0, 0.5] |
| `is_weekday` | true |

Harmonics (SMPS signature, applied where P > 1.5x standby): 3rd 20%, 5th 10%, 7th 5%, 9th 3%, 11th 2%, 13th 1%. Multiple instances can be created via `--instance-id` and `--phase`.

### 3.5 Washing machine (`washing_machine`, Type II, single-phase)

States: `off`, `fill`, `heat`, `wash_agitate` (+ `wash_agitate_idle`), `pump_out`, `rinse` (+ `rinse_idle`), `spin`, `pump_drain_final`. One cycle per day with probability `cycle_probability`, starting near 11:00, 13:00, 18:00 or 20:00 (plus or minus 30 min). Fixed sequence and durations: fill 90 s, heat (program dependent), agitate 1800 s, pump 60 s, fill 120 s, rinse 900 s, pump 60 s, spin 480 s (ramp up 30%, hold 50%, ramp down), final drain 60 s. During agitate/rinse the motor cycles roughly 10 s on / 10 s off (idle floor 5 W), producing the characteristic P oscillation.

| Parameter | Default |
|---|---|
| `program` | `"warm"`; heat duration: hot 1500 s, warm 900 s, cold 0 s, eco 600 s |
| `heat_P_W` | [1800, 2300] W |
| `motor_P_W` | [100, 250] W (Q = 0.7 P in motor phases) |
| `spin_max_P_W` | [300, 600] W |
| `cycle_probability` | 0.85 |

Harmonics: heat 3rd 1%; motor phases 3rd 5%, 5th 2%, 7th 1%.

### 3.6 EV charging (`ev`, Type III, single-phase in code)

Modes and target power: `slow_AC` 3700 W, `fast_AC_11kW` 11 000 W, `fast_AC_22kW` 22 000 W, `smart_modulated` up to 11 000 W. States: `off`, `charging`, `smart_charging`. A session occurs with probability `session_probability`; session energy comes from `battery_capacity_kWh * (1 - start_SoC)`. Conventional modes plug in between 21:00 and 23:00, soft-start over ~3 s, hold constant power for ~85% of the session, then taper to 15% (CV phase). Smart mode plugs in between 08:00 and 18:00 and follows a slowly varying setpoint (sum of two sines, clipped to [0.2, 1.0] of max). PF is fixed at 0.98 (Q small and positive).

| Parameter | Default |
|---|---|
| `mode` | `"slow_AC"` |
| `battery_capacity_kWh` | [40, 100] |
| `start_SoC` | [0.10, 0.60] |
| `charger_quality` | `"mid"` (budget / mid / premium) |
| `session_probability` | 0.9 |

Harmonics by charger quality: budget 3rd 8%, 5th 6%, 7th 4%, 9th 2%; mid 3rd 3%, 5th 2.5%, 7th 1.5%, 9th 0.5%; premium 3rd 1%, 5th 1%, 7th 0.5%.

Note: the generator class is single-phase (`is_three_phase = False`), so even the fast-AC modes currently land on one phase when aggregated. Treat fast-AC three-phase distribution as a known limitation.

### 3.7 PV (`pv`, special, three-phase, P < 0)

States: `off`, `generating` (generating where the magnitude exceeds 5 W). P is negative during generation: clear-sky envelope from `solar_intensity()` times a cloud process (AR(1) at 30 s resolution, clipped to [1 - cloud_intensity, 1]) times peak power `panel_count * panel_Wp`. Q is a constant setpoint.

| Parameter | Default |
|---|---|
| `panel_count` | 3 |
| `panel_Wp` | [300, 400] W |
| `latitude_deg` | 51.0 (Gummersbach) |
| `cloud_intensity` | [0.0, 0.6] |
| `inverter_THD_pct` | [1.0, 5.0] (declared; the fixed harmonic profile below is what is actually applied) |
| `Q_setpoint_var` | 0.0 |

Harmonics: 3rd 0.5%, 5th 2%, 7th 1.5%, 11th 0.5%, 13th 0.3%, with deliberately tight phase clustering near zero (jitter 0.02 rad, phases scaled by 0.2) to model PWM-inverter switching. This phase signature is the feature that distinguishes PV generation from a reduction in load.

### 3.8 Synchronous machine (`synchronous`, special, three-phase, four-quadrant)

States are the four operating modes: `motor_underexcited` (+P, +Q), `motor_overexcited` (+P, -Q), `generator_underexcited` (-P, +Q), `generator_overexcited` (-P, -Q), plus `off`. If no `mode_schedule` is supplied, a default schedule cycles through all four quadrants during `operating_hours` in blocks of 0.5 to 1.5 h with setpoints |P| in 300-1500 W and |Q| in 200-1000 var, with ramped transitions.

| Parameter | Default |
|---|---|
| `mode_schedule` | None (auto-build); otherwise list of (start_h, end_h, mode, P_set, Q_set) |
| `operating_hours` | [9.0, 17.0] |
| `transition_rate_W_per_s` | [10, 100] |

Harmonics (slot harmonics): 5th 4%, 7th 2.5%, 11th 1%, 13th 0.5%.

### 3.9 Baseload (`baseload`, Type IV, single-phase)

Always `on`. Constant P with small Gaussian noise, constant Q. Keeps the always-on floor realistic (router, standby devices).

| Parameter | Default |
|---|---|
| `P_W` | [15, 30] W |
| `Q_var` | [3, 10] var |
| `noise_W` | 0.5 |

Harmonics: 3rd 8%, 5th 4%, 7th 2% (mixed small SMPS loads).

---

## 4. CLI

```
python Appliance_generator.py [options]

--appliance NAME       one of: fridge, resistive, hair_dryer, pc,
                       washing_machine, ev, pv, synchronous, baseload
--list                 list appliances and exit
--duration SECONDS     default 86400 (24 h)
--rate HZ              default 5.0
--seed INT             default 42
--anchor-date DATE     YYYY-MM-DD, UTC midnight; default 2024-01-01
--phase L1|L2|L3       single-phase appliances only; default L1
--instance-id INT      default 1
--params JSON          parameter overrides, e.g. '{"mode": "fast_AC_11kW"}'
--output PATH          default <appliance>.h5
--no-save              skip writing HDF5
--plot                 quick plot of P, Q and the 3rd-harmonic magnitude
```

Examples:

```bash
python Appliance_generator.py --list
python Appliance_generator.py --appliance fridge --output fridge.h5 --plot
python Appliance_generator.py --appliance pv --anchor-date 2024-06-21 --output pv_summer.h5
python Appliance_generator.py --appliance ev --params '{"mode": "fast_AC_11kW"}'
python Appliance_generator.py --appliance fridge --params '{"compressor_P_W": [200, 250]}' --seed 7
```

Output files use layout (a) of `01_data_format.md` and are the direct inputs to `aggregator.py`. All appliances in one scenario must be generated with the same `--duration`, `--rate` and `--anchor-date` or the aggregator's alignment check fails.

---

## Appendix A: Type-vs-feature quick reference

Each appliance's distinguishing features, for MS2 feature selection:

| Appliance | Best in time domain | Best in (P, Q) plane | Best in harmonics |
|---|---|---|---|
| Fridge | cyclic on/off + inrush spike | inductive Q during compressor | motor 3rd/5th |
| Resistive load | smooth ramp dynamics | along P axis (Q near 0) | weak (clean) |
| Hair dryer | short bursts, discrete levels | along P axis | weak |
| PC | working-hours envelope | low-PF cluster | strong 3rd/5th/7th (SMPS) |
| Washing machine | multi-phase sequence, 10 s motor cycling | jumps between (P, Q) regions | motor vs heater contrast |
| EV | long flat sessions, mode-specific | near-unity-PF cluster | low; quality-dependent |
| PV | diurnal, negative P | P < 0, Q near 0 | low; tight phase clustering |
| Synchronous machine | slow setpoint changes | spans all four quadrants | slot harmonics (5th/7th) |
| Baseload | constant floor | small fixed point | moderate 3rd |
