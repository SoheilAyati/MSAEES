# Lab Report: Software Improvements Identified During Live Testing

**Course:** Modeling, Simulation and Automation of Electrical Energy Systems, TH Koeln

**Team:** Soheil Ayati (11153003), Marc Steffgen (11149043)

**System under test:** live NILM monitor (16.07.2026)

**Date:** 2026-07-22

## Context

Our live NILM system disaggregates per-appliance power from a single PAC4200
measurement, logs every switch event with exact timestamps, and learns unknown
devices on the go. It performed well in the live evaluation. This report
collects the concrete weaknesses we observed during those tests and the
improvements we would prioritise next. Each point states what the software does
today, why the current behaviour is a limitation, and the proposed change.

## 1. A graphical event timeline

**Today.** Switch events are written to `events.csv`
(`time_iso, kind, device, confidence, dP_W, dQ_var, P_total_W`), and the live
dashboard shows them as a scrolling text log. The measured power is charted, but
the ON/OFF history of individual devices is not.

**Limitation.** During and after a test it is hard to reconstruct what ran when.
Reading a text log line by line does not show overlap between devices, missed
edges, or a claim that flickered, all of which are obvious at a glance on a
timeline.

**Improvement.** Render a per-device Gantt-style timeline (one lane per device,
coloured ON intervals against a shared time axis) directly from `events.csv`,
alongside the existing power chart. Because the event log already carries the
exact edge timestamps and matched device, no new data is needed, only the view.
This would also serve as a post-test artifact for the report and for debugging
misclassifications.

## 2. Faster and more precise corrections

**Today.** When a device's steady state changes shape, the engine can briefly
misread it before a second evidence source corrects it. In the live test at
about 16:01 the PV export was released as OFF for roughly ten seconds, then the
residual monitor re-matched its signature and restored it to ON.

**Limitation.** The root cause is documented: a PV generation ramp is a step of
`(dP, dQ` near `0)`, which is indistinguishable in the (P, Q) plane from a small
switching supply turning off, so the off-edge wrongly released the PV claim. The
correction is real but slow, because it waits for the residual monitor's
multi-second window to accumulate before re-matching.

**Improvement.** Move the discriminating evidence to the moment of the edge
instead of waiting for the residual monitor. A true switch carries the device's
harmonic current with it; a generation ramp carries none. Applying that
harmonic and sign test at edge time (and adding a short hold before a generator
claim is released) would shorten the ten-second correction to a fraction of a
second and avoid the visible flicker, rather than detecting and undoing the
mistake after the fact.

## 3. Generic load-class labels for unknown devices

**Today.** Teaching an unknown load requires the user to type a specific name
(`standing_fan_high`, `coffee_machine_standby`, and so on). The signature table
and the identify model are then keyed to that exact family.

**Limitation.** In a real installation the operator often does not know the
device family, only that something new is drawing power. Requiring a specific
name makes the system brittle to naming and unhelpful when the device is simply
unfamiliar.

**Improvement.** Offer an automatic electrical-class label derived from features
we already measure, so an unknown load can be reported as, for example,
*resistive*, *inductive*, *capacitive*, or *non-linear (switch-mode)* without a
device name. The sign of Q separates inductive from capacitive; power factor and
THD_I separate a clean resistive heater from a distorted electronic supply. Our
own fingerprint table already shows these classes are cleanly separable on this
hardware (for example THD_I of about 2 percent for the boiler versus about 170
percent for the laptop). The user could still refine the auto-label into a
specific name later, but the system would be useful immediately.

## 4. Fully asynchronous, hands-off training

**Today.** The in-mix teach removed the need to empty the mains, but it still
asks the user for one manual action: switch the unknown device off and back on
once so the settled step gives a model-free measurement of its draw.

**Limitation.** Any required user action interrupts an experiment and is
impossible for loads that cannot be toggled (a running fridge, an inverter, a
sealed installation). It also prevents unattended, continuous learning.

**Improvement.** Learn passively from the residual history the engine already
records, with no toggle required. When unexplained power appears and stays,
capture its settled level, Q, and harmonic signature over time and teach a
provisional signature from that alone, cross-checking against any natural
on/off transitions that happen to occur. The manual toggle would become an
optional way to improve confidence, not a prerequisite, making training a
background process the user never has to trigger.

## 5. Recognition across a device's operating range

**Today.** A device is taught at whatever it happened to draw during the
capture. A switching supply or PV inverter whose output varies with load or
sunlight can then fall outside the taught range and stop matching (the laptop
taught at 66 W idles at 43 W; PV taught around -13 W was observed at -30 W).

**Limitation.** Variable-output loads and generation are exactly the cases NILM
finds hardest, and a single taught operating point does not cover them.

**Improvement.** Teach a range rather than a point: record the device across
several operating levels, or let a distinctive harmonic fingerprint taper the
match toward idle (already done for high-THD loads). For PV specifically, the
unique and load-independent discriminator is the sign of power, which no
consumer produces, so generation should be recognised by sign across its whole
range rather than by a taught wattage.

## Summary

The system met the goals of the exercise. The improvements above target the
gaps we felt during live operation: better visibility (timeline), faster
self-correction (edge-time harmonic evidence), lower operator burden (generic
labels and hands-off learning), and robustness to the hardest loads
(variable-output devices and generation). None require new sensing hardware;
all build on features the pipeline already records.
