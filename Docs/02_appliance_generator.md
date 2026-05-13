# NILM Project — Appliance Generator Specification

**Version:** 0.2    
**Milestone:** 1    
**Companion to:** `data_format.md`    
**Owners:** Soheil Ayati, Marc Steffgen   
**Last updated:** 2026-05-13

---

## 1. Purpose

Defines the physical, electrical, and behavioural model for each of the 8 appliances in the NILM project. Each appliance generator produces a clean, deterministic signature given a seed; measurement noise is added later in the preprocessing pipeline.

---

## 2. Common framework

### 2.1 Generator interface

Every appliance is a Python class implementing the same interface:

```python
class ApplianceGenerator:
    name: str                   # e.g. "fridge", "pc"
    instance_id: int            # 1, 2, 3 ... for multi-instance appliances
    phase: str                  # "L1" | "L2" | "L3" | "all" (3-phase)
    appliance_type: str         # "I" | "II" | "III" | "IV"

    def __init__(self, params: dict, seed: int): ...

    def generate(
        self,
        duration_s: float,
        sample_rate_hz: float,
        anchor_datetime: datetime,
    ) -> ApplianceTrace:
        """Returns a deterministic trace given the seed."""
```

`ApplianceTrace` carries:
- `state[t]` — categorical per-sample state label
- `P[t]`, `Q[t]` — active and reactive power contribution to aggregate, per sample
- `harmonics_I[t, n]` — magnitude and phase of current harmonics 2–40 (the appliance's contribution; aggregated by summing complex harmonic vectors across appliances)
- `metadata` — parameters, instance_id, phase

### 2.2 Conventions

- All P in watts, Q in var, time in UTC microseconds.
- Sign convention follows `data_format.md` §6 (P>0 = consumption, P<0 = generation; Q>0 = inductive).
- Harmonics are *current* harmonics; voltage harmonics emerge from the grid model, not from appliances directly. (For NILM purposes, current harmonics are what matters.)
- Harmonic phase is referred to V_L1 fundamental zero crossing — appliance generators receive a reference phase at construction so single-phase appliances on L2/L3 shift accordingly.

### 2.3 Time-of-day model

A shared helper produces a probability-of-activation curve `p(t)` from a small set of profile presets:

- `working_hours_weekday` — peak 09:00–17:00, low overnight
- `morning_evening_peaks` — peaks 07:00–09:00 and 18:00–22:00
- `overnight` — peaks 22:00–06:00
- `random_burst` — uniform; one or two bursts per day at random times
- `continuous` — always on (no time-of-day modulation)
- `solar` — bell curve sunrise to sunset, modulated by anchor date and latitude

Each appliance generator declares which profile drives its top-level state machine.

### 2.4 Parameter randomization for generalization

Each generator accepts `params` with ranges, not fixed values. The seed picks a sample inside the ranges. Two scenarios generated with different seeds therefore see different appliance "instances" of the same category — different fridge compressor sizes, different EV charging rates, etc. This is what enables generalization testing in M2 (train on some sampled instances, test on held-out ones).

### 2.5 Where noise does NOT live

Generators produce **clean** signatures. Don't add:
- Measurement noise (added by preprocessing per channel)
- Modbus dropout (added by preprocessing)
- Quantization to class-0.2 accuracy (added by preprocessing)

This keeps generators deterministic and keeps the per-appliance-contribution invariant clean.

---

## 3. Per-appliance specifications

### 3.1 Fridge

**Type:** II (FSM with cycling)   
**Recommended as first implementation** — exercises most of the format on bounded complexity.   
**Phase:** single-phase   

**States**

| State | Typical P (W) | Typical Q (var) | Typical PF | Notes |
|---|---|---|---|---|
| `off_control` | 1–3 | <1 | ~1 | electronics only, always |
| `compressor_on` | 120–180 | 80–140 | 0.7–0.85 | inductive, the dominant signature |
| `compressor_starting` | 400–800 | 300–600 | 0.5–0.7 | 100–300 ms inrush, then settles |
| `defrost` | 200–350 | <30 | ~1 | resistive heater; rare (~1–2× per day) |

**State transitions**
- Top-level driver: `continuous`. Compressor cycles regardless of time of day.
- `off_control → compressor_starting`: every `T_off_min` to `T_off_max` (typically 20–45 min). Sampled per cycle.
- `compressor_starting → compressor_on`: duration `T_inrush` ≈ 100–300 ms.
- `compressor_on → off_control`: duration `T_on_min` to `T_on_max` (typically 8–18 min).
- `* → defrost`: low-probability scheduled event, ~1–2 times per 24h, duration 10–20 min.

**Harmonic signature (current)**
- Compressor on: motor harmonics, primarily 3rd (~5–8% of fundamental), 5th (~3–5%), 7th (~1–3%). Decreasing with order. Phases roughly fixed relative to fundamental due to single-phase induction motor characteristics.
- Inrush: broadband transient harmonic content during starting; spike on 3rd and 5th up to ~15%.
- Defrost: resistive, near-zero harmonics.

**Tunable parameters**
- `compressor_power_W`: range [120, 180]
- `compressor_Q_var`: range [80, 140]
- `on_duration_s`: range [480, 1080]
- `off_duration_s`: range [1200, 2700]
- `inrush_factor`: range [2.5, 4.5] × steady P
- `defrost_per_day`: 0–2
- `phase_assignment`: L1/L2/L3 (random unless fixed)

**Implementation notes**
- The inrush is the trickiest part. At 5 Hz, a 200 ms inrush is one sample. Model it as a single sample with P ≈ inrush_factor × steady_P, then the next sample is steady_P. Inrush samples should still respect the sign convention.
- Defrost is rare but very distinctive in (P, Q) space — high P, near-zero Q. Adversarial scenarios should exploit this.

---

### 3.2 Variable resistive load

**Type:** III (continuously variable)   
**Phase:** single-phase   
**Primary purpose:** lab calibration device, also doubles as a "smooth ramp" test case   

**States**
- Effectively one state with a continuously varying setpoint.

| State | Typical P (W) | Q (var) | PF |
|---|---|---|---|
| `active` | 0 to `P_max` (variable) | ~0 | ~1.00 |

**Behavior**
- Drive top-level activation with `random_burst` or `working_hours_weekday`.
- When active, P follows a slowly varying setpoint trajectory: e.g. piecewise-linear ramps between random target values within [0, P_max].
- Ramp rate: 50–500 W/s.

**Harmonic signature**
- Negligible. Purely resistive, sinusoidal current. Optionally add a tiny 3rd harmonic (~0.5%) to be realistic.

**Tunable parameters**
- `P_max`: range [500, 3000]
- `ramp_rate_W_per_s`: range [50, 500]
- `active_periods_per_day`: 0–4
- `period_duration_s`: range [60, 1800]

**Implementation notes**
- This is the easiest one. Implement second, after the fridge, to validate that the framework handles Type-III continuously-variable correctly.

---

### 3.3 Hair dryer

**Type:** II (multi-state)   
**Phase:** single-phase   

**States**

| State | Typical P (W) | Typical Q (var) | Typical PF |
|---|---|---|---|
| `off` | 0 | 0 | — |
| `cold_fan_low` | 30–60 | 15–25 | 0.85 |
| `cold_fan_high` | 50–90 | 25–40 | 0.85 |
| `heat_low_fan_low` | 350–500 | 30–50 | 0.99 |
| `heat_low_fan_high` | 400–550 | 35–55 | 0.99 |
| `heat_high_fan_low` | 1100–1400 | 60–90 | 0.99 |
| `heat_high_fan_high` | 1700–2200 | 80–120 | 0.99 |

**State transitions**
- Top-level driver: `morning_evening_peaks`. When activated, the appliance enters a usage burst.
- Within a burst: random walk between states (user adjusting settings). Typical burst structure: 30 s fan-only → 60–120 s heat_high → 30 s fan_only_cold.
- Inrush on motor start: brief (~50 ms) spike, mostly captured in one sample at 5 Hz.

**Harmonic signature (current)**
- Heater-dominated states (heat_*): mostly clean, slight 3rd harmonic (<2%).
- Fan-only states (cold_*): motor harmonics, 3rd ~3–5%, 5th ~1–2%.
- Mixed: weighted average.

**Tunable parameters**
- `P_max`: range [1700, 2200]
- `usage_minutes_per_day`: 0–15 (most days zero in some households)
- `burst_count_per_day`: typically 0 or 1, occasionally 2

**Implementation notes**
- Multi-state with rapid transitions makes this a good test of state-machine plumbing.
- Adversarial scenarios: hair_dryer (heat_high_fan_high) at ~2 kW has similar P signature to EV slow-charge mode; distinguishing them needs harmonics or duration.

---

### 3.4 PCs (2 or 3 instances)

**Type:** IV (permanent/standby) with Type II overlay (active/idle/sleep)   
**Phase:** single-phase (each PC independently assigned to L1/L2/L3)   

**States**

| State | Typical P (W) | Typical Q (var) | Typical PF |
|---|---|---|---|
| `off` | 0 | 0 | — |
| `standby` | 2–5 | 1–2 | 0.6–0.8 |
| `sleep` | 5–15 | 3–8 | 0.6–0.8 |
| `idle` | 30–70 | 15–35 | 0.6–0.85 |
| `active` | 80–200 | 30–80 | 0.75–0.9 |
| `peak` | 250–400 | 80–150 | 0.8–0.95 |

**State transitions**
- Top-level driver: `working_hours_weekday`. Outside working hours: mostly `off` or `standby` with occasional brief activity.
- During working hours: random walk between `idle` / `active` / `peak`. Typical pattern: 70% idle, 25% active, 5% peak.
- State changes on ~30 s to ~5 min timescales (CPU load fluctuations).

**Harmonic signature (current)**
- Strong switched-mode PSU signature: pronounced 3rd (10–25% of fundamental), 5th (5–15%), 7th (3–8%). Harmonic content increases with load.
- Phases of harmonics are characteristic of the SMPS topology — relatively stable.

**Tunable parameters**
- `PSU_quality`: ∈ {low, medium, high} — affects harmonic magnitudes and PF.
- `idle_P_mean`: range [30, 70]
- `active_P_mean`: range [80, 200]
- `peak_P_mean`: range [250, 400]
- `peak_fraction`: range [0.02, 0.10] — how often a workday session goes into peak
- `weekend_active_probability`: range [0.0, 0.5] — usage on non-weekdays

**Implementation notes**
- Generate 2 or 3 independent PC instances per scenario, each on a different phase if possible — this provides a natural phase-disambiguation test case.
- Important target: PCs should show meaningful daily and weekday/weekend variation, which is what makes them distinguishable from always-on loads.

---

### 3.5 Washing machine

**Type:** II/III (complex multi-phase FSM)   
**Phase:** single-phase   

**States**

| State | Typical P (W) | Typical Q (var) | Typical PF | Typical duration |
|---|---|---|---|---|
| `off` | 0 | 0 | — | — |
| `fill` | 30–80 | 10–30 | 0.85 | 1–3 min |
| `heat` | 1800–2300 | 30–80 | ~1.00 | 5–25 min (depends on program) |
| `wash_agitate` | 100–250 (cycling on/off) | 60–150 | 0.7–0.85 | 20–40 min |
| `pump_out` | 200–400 | 80–200 | 0.7–0.8 | 1–2 min |
| `rinse` | 100–250 (cycling) | 60–150 | 0.7–0.85 | 10–20 min |
| `spin` | 300–600 (rising) | 200–400 | 0.7–0.8 | 5–10 min |
| `pump_drain_final` | 200–400 | 80–200 | 0.7–0.8 | 1–2 min |

**State sequence**
Cycle proceeds: `off → fill → heat → wash_agitate → pump_out → fill → rinse → pump_out → fill → rinse → pump_out → spin → pump_drain_final → off`. Some programs skip `heat` (cold wash) or repeat rinse.

Top-level driver: `random_burst`, but realistically clustered to mid-day (10:00–14:00) or evenings (18:00–21:00).

**Within agitate/rinse phases**
- Motor cycles on/off every 5–15 s with ~50% duty cycle, producing rapid P oscillation.
- At 5 Hz, these oscillations are clearly visible (25–75 samples per cycle).

**Harmonic signature (current)**
- `heat`: clean, near-zero harmonics (resistive).
- `wash_agitate`, `rinse`: motor harmonics during active sub-phases (3rd ~5%, 5th ~2%), near-zero between.
- `spin`: rises with motor speed; broadband content from variable-speed controller (if equipped).

**Tunable parameters**
- `program`: ∈ {hot, warm, cold, eco} — affects whether/how long `heat` runs
- `heat_P_W`: range [1800, 2300]
- `motor_P_W`: range [100, 250]
- `spin_max_P_W`: range [300, 600]
- `cycles_per_week`: range [1, 7]

**Implementation notes**
- This is the most state-heavy generator. Implement after one Type-IV (PCs) and one continuous Type-III (resistive) are working — you'll have battle-tested the framework by then.
- The phase transitions are valuable benchmark events — every phase change is an event with a known timestamp and known before/after signature.

---

### 3.6 EV charging

**Type:** III (continuously variable, especially smart mode)   
**Phase:** single-phase for slow AC; 3-phase for fast AC   

**Modes**

| Mode | P range (W) | Phases | PF | Notes |
|---|---|---|---|---|
| `slow_AC` (Mode 2 / Mode 3, 16A) | ~3700 | L1 only | 0.97–1.00 | constant power once started |
| `fast_AC_11kW` | ~11000 | all three | 0.97–1.00 | balanced 3-phase |
| `fast_AC_22kW` | ~22000 | all three | 0.97–1.00 | balanced 3-phase |
| `smart_modulated` | 0–11000 | varies | 0.97–1.00 | follows grid/solar signal; ramps and step-changes |

**Behavior**
- Plug-in event: P ramps from 0 to setpoint over ~1–5 s (no large inrush — modern chargers soft-start).
- Steady charging: constant P for most of the session.
- CV phase (last ~10–15% of charge): P tapers down over 10–30 min as battery approaches full.
- Smart mode: P responds to a notional setpoint signal that varies over the day; can include responses to PV (charging more when sun is up) and to grid signals (reducing during peaks).

**Top-level driver**
- Conventional charging: `overnight` (start 21:00–23:00, end 04:00–07:00).
- Smart mode: combination of `solar` (daytime charging when PV available) and `overnight` (top-up).

**Harmonic signature (current)**
- Modern chargers: clean. 5th and 7th harmonics typically <3%. PF near unity.
- "Budget" chargers (parameterizable): can have 3rd ~5–10%, 5th ~5–8%.
- Smart mode ramps produce transient harmonic content during step-changes.

**Tunable parameters**
- `mode`: ∈ {slow_AC, fast_AC_11kW, fast_AC_22kW, smart_modulated}
- `battery_capacity_kWh`: range [40, 100]
- `start_SoC`: range [0.10, 0.60]
- `charger_quality`: ∈ {budget, mid, premium}
- `charging_probability_per_day`: range [0.3, 1.0]
- For smart mode: `pv_responsive` (bool), `tariff_responsive` (bool)

**Implementation notes**
- The mode toggle is the main lever — each mode is essentially a different appliance from a NILM perspective.
- Smart mode is the angle-1 / angle-3 / angle-4 powerhouse: continuously variable, PV-correlated, hard to distinguish from net load swings without harmonic features. Build it last, but build it well.

---

### 3.7 PV input (3 panels behind one AC converter)

**Type:** Special — negative load (generation)   
**Phase:** 3-phase (typical for a 3-panel installation in Germany with 3-phase grid feed-in)   
**Sign:** P < 0 throughout active hours   

**Behavior**
- Sign of P is negative during generation (per data format spec §6).
- Magnitude follows a bell curve from sunrise to sunset, peak around solar noon.
- Peak power: 3 panels × ~350–400 Wp each = ~1000–1200 W peak under ideal conditions.
- Modulated by:
  - Daily envelope (sin-squared from sunrise to sunset, scaled by anchor date / season).
  - Cloud cover: random multiplicative noise process. Realistic model: 1-minute autocorrelated random walk in [0.3, 1.0], with occasional drops to [0.1, 0.3] for cloud passage events lasting 1–10 minutes.
- Reactive power: typically 0 (unity-PF inverter), but modern inverters can be configured to inject Q on demand — parameterize this.

**Harmonic signature (current)**
- Inverter-specific. Generally low-distortion modern inverters: 3rd <1%, 5th 1–3%, 7th <2%.
- Critically: harmonic *phase* relationships differ from motor loads. PV harmonics tend to be in-phase with the fundamental zero crossing because they come from PWM inverter switching, not from magnetic non-linearity. This is the key disambiguation feature for angle 3 (PV-aware NILM).

**Top-level driver**
- `solar` — daily envelope from anchor date.

**Tunable parameters**
- `panel_count`: fixed at 3 for the lab setup
- `panel_Wp`: range [300, 400]
- `latitude_deg`: ~51 (Gummersbach)
- `cloud_intensity`: range [0.0, 0.8] — fraction of day under cloud
- `inverter_THD_pct`: range [1.0, 5.0]
- `Q_setpoint_var`: range [-200, 200] — for VAr support testing

**Implementation notes**
- This is the centerpiece for angle 3. Get the phase relationship of harmonics right — it's the feature that lets a smart algorithm distinguish PV generation from a reduction in load.
- Anchor-date-driven solar elevation: compute peak P from a simple solar position formula, not a hardcoded curve. Lets generalization tests cover summer vs winter days.

---

### 3.8 Synchronous machine

**Type:** Special — four-quadrant device   
**Phase:** 3-phase (synchronous machines are inherently 3-phase)   
**Sign:** P and Q can both be positive or negative independently   

**Operating modes**

| Mode | P sign | Q sign | Typical use |
|---|---|---|---|
| `motor_underexcited` | + | + | absorbs P and Q |
| `motor_overexcited` | + | – | absorbs P, supplies Q (power-factor correction) |
| `generator_underexcited` | – | + | supplies P, absorbs Q |
| `generator_overexcited` | – | – | supplies P and Q |

This four-quadrant capability is what makes the synchronous machine pedagogically interesting and a distinctive NILM signature: it is the only appliance on the list that occupies all four quadrants of the (P, Q) plane.

**Operating ranges** (typical lab machine)
- |P|: 100 – 2000 W
- |Q|: 100 – 1500 var

**Behavior**
- Operates in slowly varying setpoints — the lab user changes excitation and shaft load occasionally, not continuously.
- Transitions between modes are smooth (excitation ramps) on second-to-minute timescales.

**Harmonic signature (current)**
- Slot harmonics: 5th (3–5%), 7th (2–4%) — characteristic of stator slot count.
- Harmonic content varies with operating point: higher in motor mode at full load, lower in light operation.
- Distinct from PV inverter and from induction-motor signatures because of phase relationships and harmonic-order distribution.

**Top-level driver**
- `working_hours_weekday` (it's a lab device).

**Tunable parameters**
- `mode_schedule`: list of (start_time, end_time, mode, P_setpoint, Q_setpoint)
- `transition_rate_W_per_s`: range [10, 100]
- `slot_count`: affects harmonic profile

**Implementation notes**
- Worth being deliberate about (P, Q) trajectories so the four-quadrant behavior is actually demonstrated in benchmark scenarios. Otherwise this appliance just looks like "varied motor."
- Hardest from a "realism" perspective but quite tractable as a generator — a sequence of (P, Q) setpoints with smooth interpolation between them does the job.

---

## 4. Aggregation

Each scenario file contains one instance of each appliance type (except PCs, where 2–3 instances are normal). The aggregate signal at the PCC is the sum of all per-appliance traces:

```
P_total[t] = sum over appliances of P_a[t]
Q_total[t] = sum over appliances of Q_a[t]
I_harmonic_n[t] = vector sum (in complex form) of all per-appliance harmonic_n
```

For 3-phase aggregation, single-phase appliances contribute only to their assigned phase; 3-phase appliances (synchronous machine, fast AC EV charging, PV) contribute to all three.

Voltage is *not* affected by load in this synthetic model — we assume an infinite bus. V_L{1,2,3} are generated independently (nominal 230 V_rms ± small slow variations).

---

## 5. Implementation order

1. **Fridge** (Type II, well-bounded) — exercises state machine, transients, inductive Q, motor harmonics.
2. **Variable resistive load** (Type III, simplest) — validates continuous-setpoint generation and harmonic-free output.
3. **Hair dryer** (Type II, multi-state) — validates richer state machines and time-of-day driver.
4. **PCs** (Type IV with overlay) — validates multi-instance and per-phase assignment.
5. **Washing machine** (complex FSM) — validates long-sequence state machines and within-phase fast oscillations.
6. **PV** (bidirectional, solar-driven) — validates sign convention and inverter harmonic phase model.
7. **EV** (multiple modes including smart) — validates mode-switching and PV-correlated control signal.
8. **Synchronous machine** (four-quadrant) — validates full (P, Q) plane coverage.

Each appliance should round-trip through the format → storage → preprocessing → plot pipeline before moving to the next. If something is wrong in the format spec, the fridge will surface it.

---

## 6. Open issues for joint review

1. Confirm the appliance list and instance counts (PCs = 2 or 3? More than one washing machine?).
2. Confirm phase assignment policy: randomized per scenario, or fixed (e.g. fridge always on L1)?
3. Confirm whether the lab synchronous machine has documented operating ranges to match — otherwise use the typical values above.
4. Whether to include a "permanent baseload" appliance (router, modem, standby lights) as a Type IV — keeps the always-on floor realistic. Recommended: yes, add a minimal `baseload` generator at ~10–30 W constant.

---

## Appendix A: Type-vs-feature quick reference

For Milestone 2's characterization deliverable. Each appliance's distinguishing features:

| Appliance | Best in time domain | Best in (P,Q) plane | Best in harmonics |
|---|---|---|---|
| Fridge | cyclic on/off pattern | inductive Q during compressor | motor 3rd/5th |
| Resistive load | smooth ramp dynamics | along P axis (Q≈0) | weak (clean) |
| Hair dryer | short bursts, multiple discrete levels | along P axis | weak |
| PC | working-hours envelope | low PF cluster | strong 3rd, 5th, 7th (SMPS) |
| Washing machine | distinctive multi-phase sequence | jumps between (P,Q) regions | motor + heater contrast |
| EV | long flat sessions, mode-specific | unity-PF cluster | low; phase signatures |
| PV | diurnal, negative P | P<0, Q≈0 cluster | low; inverter phase signature |
| Synchronous machine | slow setpoint changes | spans all four quadrants | slot harmonics, distinctive |

This table is what tells Milestone 2 which feature extractors to apply where.