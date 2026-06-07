"""
classification.py  --  Milestone 2, Stage 5 (classical models)
==============================================================

Supervised appliance identification from the per-segment feature table.

These are the interpretable, low-data, "AI-Act-safe" models (scikit-learn +
optional LightGBM).  They are the Milestone 2 baseline and -- if the project is
ever treated as critical infrastructure -- the only models that would be used.
The deep-learning counterpart lives in deep_seq2point.py.

Key functions
-------------
train_classifier()     fit RF / LightGBM / SVM with a standardising pipeline.
evaluate()             accuracy + macro-F1 + per-class report + confusion matrix.
group_holdout_eval()   train/test split that keeps whole operating intervals (or
                       whole seeds, when available) apart, so we measure
                       generalisation to unseen appliance *instances* rather than
                       memorisation -- the generalisation goal of the data design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


def make_model(kind="rf"):
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=0,
                                     class_weight="balanced")
    elif kind == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    elif kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("pip install lightgbm to use kind='lgbm'") from exc
        clf = LGBMClassifier(n_estimators=400, learning_rate=0.05,
                             num_leaves=31, random_state=0, verbose=-1)
    else:
        raise ValueError(kind)
    return make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), clf)


def train_classifier(df, features, kind="rf", test_size=0.3, random_state=0):
    """Simple stratified train/test split; returns model + metrics."""
    X = df[features].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if np.min(np.unique(y, return_counts=True)[1]) >= 2 else None)
    model = make_model(kind).fit(Xtr, ytr)
    yp = model.predict(Xte)
    metrics = dict(
        accuracy=float(accuracy_score(yte, yp)),
        macro_f1=float(f1_score(yte, yp, average="macro")),
        report=classification_report(yte, yp, zero_division=0),
        labels=sorted(set(y)),
        confusion=confusion_matrix(yte, yp, labels=sorted(set(y))).tolist(),
    )
    return model, metrics


def cv_score(df, features, kind="rf", n_splits=5):
    """Stratified k-fold macro-F1 (more honest than a single split)."""
    X = df[features].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    counts = np.unique(y, return_counts=True)[1]
    k = int(min(n_splits, counts.min()))
    if k < 2:
        return None
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    sc = cross_val_score(make_model(kind), X, y, cv=skf,
                         scoring="f1_macro")
    return dict(mean_macro_f1=float(sc.mean()), std=float(sc.std()), folds=k)


def feature_importance(model, features):
    """Pull RF/LGBM impurity importances out of the pipeline, if present."""
    clf = model.steps[-1][1]
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        return None
    return pd.Series(imp, index=features).sort_values(ascending=False)
