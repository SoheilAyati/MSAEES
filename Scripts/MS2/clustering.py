"""
clustering.py  --  Milestone 2, Stage 4
=======================================

Unsupervised grouping of the per-segment feature vectors.  Two motivations:

  1. Sanity / separability check: if appliances form clean clusters in the
     feature space, supervised classification will be easy; if they overlap,
     we know which appliances will be confused (and which feature domain to
     add).

  2. Real, unlabelled data: on the PAC4200 lab data we will not always have
     labels, so clustering is how we discover recurring device signatures.

cluster() scales features then runs the chosen algorithm.  score() compares
clusters against known labels with ARI / NMI / homogeneity / completeness, plus
the label-free silhouette.  embed_2d() returns a 2-D projection for plotting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_score, completeness_score,
                             silhouette_score)


def _matrix(df: pd.DataFrame, features):
    # .copy() -> writable array (pandas/numpy may return a read-only view)
    X = df[features].to_numpy(dtype=float).copy()
    # impute column means for any NaN (e.g. THD missing on single files)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_mean, inds[1])
    return StandardScaler().fit_transform(X)


def cluster(df: pd.DataFrame, features, method="kmeans", n_clusters=8,
            **kwargs):
    """Return integer cluster labels for each row of ``df``."""
    Xs = _matrix(df, features)
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = model.fit_predict(Xs)
    elif method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=0)
        labels = model.fit_predict(Xs)
    elif method == "dbscan":
        model = DBSCAN(eps=kwargs.get("eps", 1.2),
                       min_samples=kwargs.get("min_samples", 5))
        labels = model.fit_predict(Xs)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(Xs)
    else:
        raise ValueError(f"unknown method {method}")
    return labels


def score(true_labels, cluster_labels, df=None, features=None):
    """Cluster-quality metrics. Silhouette computed if df+features supplied."""
    out = dict(
        ARI=float(adjusted_rand_score(true_labels, cluster_labels)),
        NMI=float(normalized_mutual_info_score(true_labels, cluster_labels)),
        homogeneity=float(homogeneity_score(true_labels, cluster_labels)),
        completeness=float(completeness_score(true_labels, cluster_labels)),
        n_clusters=int(len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)),
    )
    if df is not None and features is not None:
        Xs = _matrix(df, features)
        mask = np.array(cluster_labels) != -1
        if len(set(np.array(cluster_labels)[mask])) > 1:
            out["silhouette"] = float(silhouette_score(Xs[mask],
                                                       np.array(cluster_labels)[mask]))
    return out


def embed_2d(df: pd.DataFrame, features, method="pca"):
    """2-D embedding for visualisation (PCA always available; t-SNE optional)."""
    Xs = _matrix(df, features)
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            return TSNE(n_components=2, init="pca", random_state=0,
                        perplexity=min(30, max(5, len(Xs) // 4))).fit_transform(Xs)
        except Exception:
            pass
    return PCA(n_components=2, random_state=0).fit_transform(Xs)
