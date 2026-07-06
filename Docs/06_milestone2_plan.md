# NILM Project - Milestone 2 Plan

**Time-series analysis · Event detection · Feature extraction · Clustering · Classification / Identification · ML application**

**Version:** 0.1 (draft)
**Milestone:** 2
**Companion to:** `01_data_format.md`, `02_appliance_generator.md`, `03_aggregator.md`, `04_preprocessor.md`, `05_pac4200_reader.md`
**Starter code:** `Scripts/MS2/` (removed; superseded by `Scripts/MS2_Pipeline/`)
**Owners:** Soheil Ayati, Marc Steffgen
**Last updated:** 2026-06-07

> **Status note (2026-07-05):** this is the original Milestone 2 planning document, kept for reference. The starter-code folder it describes (formerly `Scripts/MS2`) was reorganized into `Scripts/MS2_Pipeline`, and the module names listed below were superseded during that rewrite. For the system as actually built, see `07_ms2_pipeline.md` (training/inference pipeline) and `08_live_nilm.md` (live monitor); the consolidated design/feature rationale for the paper is `09_design_rationale.md`.

---

## 1. Purpose and scope

Milestone 1 delivered the data pipeline: synthetic generation, aggregation at the PCC, preprocessing, the PAC4200 reader, and 12 engineered features - the same code path for synthetic and real data. Milestone 2 is the **analysis and learning** stage. Given a scenario file (synthetic now, real PAC4200 shortly), recover *what appliances are doing* from the aggregate.

**In scope (this milestone):**

1. **Time-series / exploratory analysis** - characterise the aggregate and per-appliance signals.
2. **Event detection** - find the timestamps where appliances switch or change state.
3. **Feature extraction** - turn events/segments into appliance-discriminating feature vectors.
4. **Clustering** - group signatures without labels (separability check + unlabelled real data).
5. **Classification / identification** - supervised appliance recognition from features.
6. **ML application** - train, evaluate per tier, and run a first synthetic->real transfer test.

**Out of scope (deferred to Milestone 3):** full *disaggregation* of the aggregate into a per-appliance power time series for every sample, lab utilization analysis, and the "transfer - what to do with the output" question (energy breakdowns, feedback, dashboards). MS2 produces the detectors, features and classifiers that MS3's disaggregation is built on.

**Inputs.** MS2 reads the files MS1 produces, unchanged:

- `/preprocessed/cleaned/*` and `/preprocessed/features/*` (the 12 MS1 features),
- `/measurements/*` including `/measurements/harmonics/*`,
- `/ground_truth/*` (synthetic only - the supervision signal).

---

## 2. Model-class decision: the EU AI Act and why deep learning is allowed here

We previously built a 24-hour-ahead German load-forecasting system under the constraint that, as a critical-infrastructure application, it could not use opaque/"stochastic" models - only interpretable ones (LightGBM, scikit-learn). The natural question: **does the same constraint bind this NILM project?**

**What the AI Act actually says.** Annex III, §2 classifies as *high-risk* only AI systems "intended to be used as **safety components** in the **management and operation** of critical digital infrastructure, road traffic, or the supply of water, gas, heating or electricity." A *safety component* is one whose failure or malfunction could cause physical damage to infrastructure or harm to persons. Crucially, systems that are "merely **supportive, informational, organisational or optimisation-oriented**" are **explicitly excluded** from the high-risk category.

**Where NILM lands.** This project monitors and analyses appliance loads behind a single meter in a lab. It does not control switchgear, dispatch generation, protect equipment, or operate any part of the grid. Its output is informational (which appliance ran, how much energy it used). It therefore falls in the **excluded** "informational / monitoring" category, **not** the high-risk critical-infrastructure category. This is the opposite of the national load-forecasting case, which can plausibly be a component in grid management/operation and so sits much closer to (or inside) the high-risk line.

**Decision for MS2.** Deep learning is permissible. The recommended strategy is a **classical -> deep-learning progression**, not an either/or:

| Step | Models | Why |
|---|---|---|
| Baseline | Random Forest, **LightGBM**, SVM (scikit-learn) on engineered features | Interpretable, strong with limited data, fast; *also* the models we would be restricted to if the project were ever re-scoped as critical infrastructure - so this baseline hedges the legal question. |
| Main | seq2point / seq2seq CNNs, multi-task nets | Higher ceiling, exploits the per-sample synthetic ground truth, handles overlapping loads better. |
| Compare | Both, on the same tiered benchmark | The strongest result for the report: "interpretable baseline vs deep model," with the trade-off quantified. |

**Caveats.** This is a reading of the regulation, not legal advice - confirm the classification with the course supervisor before relying on it. Separately, NILM can reveal occupancy/behaviour, which is a **GDPR / data-protection** consideration; it is independent of the AI-Act model-class question and does not restrict model choice. Sources are listed in §12.

---

## 3. Data inventory and the synthetic-vs-real feature gap

| Source | Files | Channels | Ground truth | Role in MS2 |
|---|---|---|---|---|
| Mixed synthetic | `Synthetic_Data/Mixed/scenario_{easy,normal,hard,adversarial}.h5` | full (per-phase + total P,Q,S,PF,cosφ,THD, 39-order harmonics ×3 phases) | yes (per-sample states + P/Q contributions for 9 appliances) | the real NILM target; train + quantitative eval |
| Single appliance | `Synthetic_Data/Single/*.h5` (9 files) | P, Q, current harmonics | yes (per-sample state) | clean labelled signatures -> first classifier + clustering validation |
| Real pre-measured | `Pre_Measured/pac4200_*_200ms.csv` (6 devices) | **single phase only**: P, Q, S, PF, THD_I, THD_V, I, V | no (only `device_name`) | validate methods on real signals; transfer target |

**The feature gap is a first-class design constraint.** Real PAC4200 CSV runs are single-phase and carry **no per-order harmonics** and **no per-phase split**. Any model meant to run on real data may therefore only use the intersection of channels:

```
COMMON_CHANNELS = P_total, Q_total, S_total, PF_total, THD_I_L1
```

So we maintain **two feature sets**: a *rich* set (per-phase + 39-order harmonics) for synthetic-only experiments, and a *common* set (the five channels above + derived ratios) for anything that has to transfer to the lab meter. `Scripts/MS2/feature_extraction.py` defines `COMMON_FEATURE_COLUMNS` for exactly this.

**Action item before training.** The generator supports varied seeds and anchor dates (`02_appliance_generator.md` §2.4). Generate **multiple** scenarios per tier with different seeds so we can train on some appliance *instances* and test on held-out ones - the generalisation test the data was designed for. Today we have one file per tier; that is enough to build and smoke-test the pipeline but not to claim generalisation.

**Measured real fingerprints** (steady-state ON medians, from `run_ms2_demo.py`):

| Device | P (W) | Q (var) | PF | THD_I (%) | Character |
|---|---|---|---|---|---|
| Toaster | 1400 | ~4 | 1.00 | 2 | pure resistive |
| Hair dryer (Föhn St.1) | 139 | −9 | 0.90 | 47 | heater + small motor |
| Fluorescent tube | 39 | 78 | 0.45 | 10 | inductive ballast |
| LED lamp 6.3 W | 6.7 | −5 | 0.74 | 48 | capacitive driver |
| USB charger (+phone) | 10 | −0.7 | 0.54 | 151 | switch-mode, very distorted |
| Mixer | 17 | 8 | 0.36 | 93 | universal motor |

These six already demonstrate the project thesis: PF and THD_I separate devices of *similar wattage* (LED vs USB charger: 6.7 W vs 10 W but THD 48 % vs 151 %).

---

## 4. Stage 1 - Time-series & exploratory analysis

Before modelling, characterise the signals. Concrete tasks:

- **Aggregate overview.** P_total / Q_total / S_total over 24 h, per phase and total; mark where P_total < 0 (the PV *signal-eclipse*, 59 590 samples in `scenario_normal`).
- **Diurnal structure.** Overlay hour-of-day; show working-hours (PC), morning/evening (hair dryer), overnight (EV), solar bell (PV).
- **Periodicity.** Autocorrelation / FFT of P_total and of the fridge contribution -> recover the ~30-60 min compressor cycle; this is itself a NILM feature.
- **Signature planes.** Scatter in the (P, Q) plane coloured by the dominant ground-truth appliance - reproduce the classical Hart space and see which appliances overlap. Add (ΔP, ΔQ) at events.
- **Harmonic structure.** Mean current-harmonic spectra (orders 2-40) per appliance state; contrast PC/SMPS (strong 3rd/5th/7th) vs toaster/resistive (flat) vs PV inverter (phase-distinct).
- **Distortion & imbalance.** Distributions of THD_I, `P_phase_imbalance`, `I_neutral_to_phase_ratio` (MS1 features) - these separate 3-phase (PV, synchronous, fast EV) from single-phase loads.
- **Real data.** Segment each CSV into baseline / ON / OFF (the documented 20 s-40 s-20 s protocol) and tabulate steady-state fingerprints (table in §3).

**Deliverable:** an EDA notebook + figures; the per-appliance fingerprint table.

---

## 5. Stage 2 - Event detection

An *event* is a sample where the aggregate changes because an appliance switched or changed state. Events are the entry point of event-based NILM (Hart 1992): each edge carries a (ΔP, ΔQ, Δharmonics) signature.

**Methods** (`Scripts/MS2/event_detection.py`):

| Method | Idea | Strength | Weakness |
|---|---|---|---|
| Hart edge | threshold the derivative `dP_total` (an MS1 feature) | high precision, exact timing | misses overlapping/concurrent events |
| CUSUM | two-sided cumulative-sum change-point | high recall, robust to slow drift | over-segments without tuning |
| PELT (`ruptures`, optional) | optimal multi-change-point segmentation | principled, multivariate-capable | slower; penalty tuning |

**Enhancements planned:** multi-channel detection (stack ΔP, ΔQ, ΔTHD_I); **per-phase** detection on P_L1/L2/L3 to localise the appliance to a phase; matched on/off pairing (Hart's two-edge appliance model); debounce around the 200 ms inrush.

**Evaluation.** Derive ground-truth events from `/ground_truth/state` transitions (filter to |ΔP| ≥ 50 W so we score *significant aggregate* switching, not sub-state flutter that is invisible at the PCC). Score detected vs truth with a ±1.5 s tolerance -> precision / recall / F1 + timing error.

**Baseline already measured** on `scenario_normal` (385 significant events):

| Detector | Detected | Precision | Recall | F1 |
|---|---|---|---|---|
| Hart edge (40 W) | 109 | **0.95** | 0.27 | 0.42 |
| CUSUM (smoothed, 250 W) | 1142 | 0.30 | **0.88** | 0.44 |

The complementary error profiles (Hart precise / CUSUM thorough) are exactly why **combining** them - Hart for confident edges, CUSUM to recover missed ones - is the first improvement.

**Real-data check.** Run the detector on each CSV and confirm it finds the documented switch-on / switch-off (e.g. toaster: 2 edges around the 40 s ON window). This validates the detector against a *known* event schedule.

---

## 6. Stage 3 - Feature extraction

For each event (or fixed window over an ON region) build a feature vector. Families map to appliances per `02_appliance_generator.md` Appendix A.

| Family | Features | Best for |
|---|---|---|
| Steady-state | mean/median P, Q, S, PF, cosφ; **Q/P ratio** | fridge (inductive Q), resistive (Q≈0), PV (P<0) |
| Harmonic | magnitudes of 3rd/5th/7th, spectral centroid & energy, THD_I | PC/SMPS, USB charger, fluorescent - high distortion |
| Transient | **inrush ratio** (peak/steady), rise time, power variability (std) | motor start (fridge), washing-machine spin ramp |
| Contextual | duration, time-of-day, **phase (L1/L2/L3)** | EV (long overnight), hair dryer (short morning/evening) |

**Output schema:** one row per event/window, feature columns + `label` (synthetic: dominant appliance from ground truth; real: `device_name`) + `start_idx`/`end_idx`. Builders provided: `build_window_table_from_singles()` (balanced windows for ML), `features_for_real_run()` (real fingerprints in the **common** space). Harmonic arrays store orders 2-40, so order *n* is column *n−2* (`order_index()` helper).

**Why windows, not only events:** windowing ON regions gives many, class-balanced samples (an always-on appliance would otherwise be one giant segment). The starter demo builds 1957 windows across 9 appliances this way.

---

## 7. Stage 4 - Clustering

Two motivations: (a) a **separability check** - clean clusters predict easy classification; overlap tells us which appliances will be confused and which feature domain to add; (b) **unlabelled real data** - clustering is how we discover recurring device signatures when no labels exist.

**Algorithms** (`Scripts/MS2/clustering.py`): KMeans, GMM, DBSCAN/HDBSCAN (density, finds noise), agglomerative. Work in the standardised feature space and specifically in the (P, Q) + harmonic subspace.

**Validation:** ARI, NMI, homogeneity, completeness (vs known labels) and silhouette (label-free). Visualise with PCA (always) / t-SNE / UMAP.

**Baseline measured** (KMeans, k = 9, single-appliance windows): ARI 0.45, NMI 0.66, homogeneity 0.62, completeness 0.71, silhouette 0.64. Interpretation: EV (high power) and resistive separate cleanly; baseload/fridge/synchronous overlap near the origin -> they need harmonic/transient features or supervision to split. The PCA plot (`demo_output/feature_space_pca.png`) shows this directly.

---

## 8. Stage 5 - Classification / identification

**Classical baseline** (`Scripts/MS2/classification.py`): Random Forest, LightGBM, SVM, kNN inside an impute->scale->model pipeline with balanced class weights.

- **Generalisation protocol (important):** train on some seeds/instances, test on **held-out seeds** (uses the generator's parameter randomisation). Use *group* hold-out so windows from the same operating interval never split across train/test (otherwise scores are inflated by leakage).
- **Baseline measured** on clean single-appliance windows: RF 5-fold macro-F1 = **0.998 ± 0.004**, LightGBM = 0.994 ± 0.007; top features `h_centroid`, `PF_mean`, `Q/P ratio`, `h7`, `h5` - i.e. the harmonic + (P,Q) **fusion** the project is built around. This is the *clean-signature upper bound*; per-event identification inside a busy mixed scenario is harder and is the real test.

**Deep learning** (`Scripts/MS2/deep_seq2point.py`, reference impl., needs `torch`):

| Model | Input -> output | Source |
|---|---|---|
| seq2point CNN | window of aggregate -> one appliance's power at the midpoint | Zhang et al. 2018 |
| seq2seq / denoising autoencoder | window -> window of appliance power | Kelly & Knottenbelt 2015 |
| Multi-task (DualNILM-style) | aggregate -> loads **+** PV-injection flag | project ref [2] |

Train directly on per-sample `P_contribution` (the synthetic advantage). Recommended inputs `[P_total, Q_total, THD_I_L1]` (multi-feature fusion); restrict to `COMMON_CHANNELS` for any model intended to transfer to the lab meter.

**Metrics.** Classification: macro-F1, per-class precision/recall, confusion matrix. Power estimation (bridges to MS3): MAE (W), Signal Aggregate Error (SAE), Normalised Disaggregation Error (NDE), energy-based F1.

**Tiered evaluation** - reuse the tiers already built (`03_aggregator.md` §6) as the headline benchmark:

| Tier | Expected F1 | Tests |
|---|---|---|
| easy | > 0.95 | sparse, non-overlapping |
| normal | ~0.85 | realistic overlap |
| hard | 0.65-0.80 | concurrent, similar-power, smart EV, cloudy PV |
| adversarial | ~0.50 | net-zero, signature collisions, PV cancelling load |

Reporting performance against these four tiers is a ready-made results story.

---

## 9. Stage 6 - ML application & synthetic->real transfer (bridge to MS3)

**End-to-end pipeline:** event detection -> per-event features -> classifier (label the event's appliance), and in parallel the windowed classifier for continuous identification. Wire both through the same `nilm_io` loaders so synthetic and real inputs are interchangeable.

**Synthetic->real transfer test.** Train a classifier on synthetic data using **only `COMMON_FEATURE_COLUMNS`**, then apply it to the six real device fingerprints. Classes only partially overlap (hair dryer is in both; toaster ≈ resistive; LED/USB/mixer ≈ power-electronic/motor), so expect this to be partial - the goal is to measure the synthetic-to-real gap and motivate domain adaptation. This directly sets up **Milestone 3** (full disaggregation + "what to do with the output": per-appliance energy, utilisation, feedback).

---

## 10. Starter code in the repository

Everything below was in the former `Scripts/MS2/` starter folder, runs on the data already present, and is sklearn-only except the optional deep module.

| File | Stage | Contents |
|---|---|---|
| `nilm_io.py` | - | loaders for scenario / single / real CSV; `ground_truth_events()`; `COMMON_CHANNELS`; harmonic-order helper |
| `event_detection.py` | 2 | `detect_edges` (Hart), `detect_cusum`, `detect_ruptures`, `evaluate_events` |
| `feature_extraction.py` | 3 | steady-state/harmonic/transient features; window & segment table builders; real fingerprints; `COMMON_FEATURE_COLUMNS` |
| `clustering.py` | 4 | KMeans/GMM/DBSCAN/agglomerative + ARI/NMI/silhouette + 2-D embeddings |
| `classification.py` | 5 | RF / LightGBM / SVM pipelines, CV, confusion matrix, feature importance |
| `deep_seq2point.py` | 5 | reference seq2point CNN (PyTorch) + windowing + training loop |
| `run_ms2_demo.py` | 2-5 | end-to-end demo; writes plots + `real_fingerprints.csv` to `demo_output/` |

**Run:** `python Scripts/MS2/run_ms2_demo.py` (needs `numpy h5py pandas scikit-learn matplotlib`; `lightgbm` optional; `torch` only for the deep module).

---

## 11. Suggested work split and timeline

| Week | Work | Lead |
|---|---|---|
| 1 | Generate multi-seed scenarios (≥5/tier); EDA notebook + figures | both |
| 2 | Event detection: combine Hart+CUSUM, per-phase, tune & evaluate per tier | one |
| 2-3 | Feature extraction: finalise rich + common feature sets; build tables | other |
| 3 | Clustering study + separability report | one |
| 4 | Classical classification + generalisation (held-out seeds) per tier | other |
| 5 | seq2point training; classical-vs-deep comparison | both |
| 6 | Synthetic->real transfer test; write MS2 report | both |
| parallel | Connect PAC4200, verify registers/byte-order, record lab runs (`05_pac4200_reader.md` §12) | both |

**First-week checklist:** (1) `python Scripts/MS2/run_ms2_demo.py` reproduces the baseline numbers; (2) generate ≥5 seeds per tier; (3) produce the (P,Q) and harmonic-spectrum EDA figures; (4) commit an MS2 results notebook.

---

## 12. Risks and open questions

1. **Single seed today.** Generalisation claims need multiple seeds/anchor dates - generate them first.
2. **Real data is single-phase, no harmonics.** Limits transfer features to the common subset; per-phase and harmonic models stay synthetic-only until the lab meter provides them.
3. **Overlapping events** depress event-detection recall - combine detectors and exploit per-phase/Q channels.
4. **Class imbalance** (always-on vs rare appliances like hair dryer) - windowing + balanced weights help; consider per-class sampling.
5. **Adversarial tier is meant to be hard** (~0.50 F1) - report it as a stress test, not a failure.
6. **AI Act classification** - confirm with the supervisor (§2); revisit if the project scope changes toward grid operation.

---

## References

- G. W. Hart, "Nonintrusive appliance load monitoring," *Proc. IEEE* 80(12), 1992. doi:10.1109/5.192069
- C. Zhang, M. Zhong, Z. Wang, N. Goddard, C. Sutton, "Sequence-to-point learning with neural networks for NILM," *AAAI* 2018. arXiv:1612.09106
- J. Kelly, W. Knottenbelt, "Neural NILM: deep neural networks applied to energy disaggregation," *BuildSys* 2015. arXiv:1507.06594
- Y. Wu et al., "DualNILM: Energy Injection Identification Enabled Disaggregation with Deep Multi-Task Learning," arXiv:2508.14600, 2025.
- S. R. Sahoo et al., "A feature fusion technique for improved NILM," *Energy Informatics* 3(1):13, 2020. doi:10.1186/s42162-020-00112-w
- N. Tsironi et al., "Odd Harmonic Distortion Contribution on a SVM NILM Approach," *IEEE SyNERGY MED* 2022. doi:10.1109/SyNERGYMED55767.2022.9941416
- EU AI Act - Annex III (high-risk use cases): https://artificialintelligenceact.eu/annex/3/
- EU AI Act - Commission draft guidelines on high-risk classification (critical infrastructure scope & exclusions): McCann FitzGerald summary, https://www.mccannfitzgerald.com/knowledge/construction-and-infrastructure/critical-infrastructure-spotlight-eu-ai-act-draft-guidelines-on-high-risk-ai-classification
- EU AI Act for the energy sector (safety-component test): Baker Botts, https://www.bakerbotts.com/thought-leadership/publications/2026/march/the-eu-ai-act
