# Milestone 2 — starter code

Analysis & learning stage of the NILM pipeline. Full plan: `Docs/06_milestone2_plan.md`.

Modules (each maps to one MS2 stage):

| File | Stage |
|---|---|
| `nilm_io.py` | loaders (scenario / single / real CSV), ground-truth events, COMMON_CHANNELS |
| `event_detection.py` | Hart edge + CUSUM (+ optional ruptures) + F1 evaluation |
| `feature_extraction.py` | steady-state / harmonic / transient features; window & real-fingerprint tables |
| `clustering.py` | KMeans / GMM / DBSCAN / agglomerative + ARI/NMI/silhouette + 2-D embedding |
| `classification.py` | Random Forest / LightGBM / SVM pipelines + CV + confusion matrix |
| `deep_seq2point.py` | reference seq2point CNN (PyTorch) — the deep-learning option |
| `run_ms2_demo.py` | end-to-end demo over the data in the repo |

## Run

```bash
pip install numpy h5py pandas scikit-learn matplotlib   # lightgbm optional; torch only for deep_seq2point
python run_ms2_demo.py                                   # writes plots + tables to demo_output/
```

## Baseline numbers (from `run_ms2_demo.py`, scenario_normal + single-appliance files)

- Event detection — Hart: P=0.95 R=0.27 F1=0.42; CUSUM: P=0.30 R=0.88 F1=0.44 (complementary → combine).
- Clustering (KMeans, k=9): ARI=0.45 NMI=0.66 silhouette=0.64.
- Classification (clean single-appliance windows): RF 5-fold macro-F1=0.998, LightGBM=0.994.
  Top features: harmonic centroid, PF, Q/P ratio, 7th & 5th harmonics — i.e. (P,Q)+harmonic fusion.

These are starting points: generate multiple seeds per tier and evaluate per tier (easy→adversarial) for the report.
