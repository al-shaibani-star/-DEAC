# -*- coding: utf-8 -*-
"""Post-processing: merge tiny clusters into nearest large cluster."""
import numpy as np


def postprocess_tiny_clusters(y_pred: np.ndarray, Z: np.ndarray,
                               min_size_ratio: float = 0.005) -> np.ndarray:
    """Merge clusters smaller than min_size_ratio * n into nearest large cluster.

    Parameters
    ----------
    y_pred : np.ndarray
        Cluster labels.
    Z : np.ndarray
        Feature matrix (for centroid distance).
    min_size_ratio : float
        Minimum cluster size as fraction of n.

    Returns
    -------
    np.ndarray
        Updated cluster labels with tiny clusters merged.
    """
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_pred)
    if n <= 0:
        return y_pred
    min_size = max(3, int(n * min_size_ratio))

    labels, counts = np.unique(y_pred, return_counts=True)
    tiny = labels[counts < min_size]
    large = labels[counts >= min_size]
    if tiny.size == 0 or large.size == 0:
        return y_pred

    centers = []
    kept = []
    for l in large:
        idx = np.where(y_pred == l)[0]
        if idx.size > 0:
            centers.append(Z[idx].mean(axis=0))
            kept.append(int(l))
    if not centers:
        return y_pred

    centers = np.asarray(centers, dtype=np.float32)
    kept = np.asarray(kept, dtype=int)
    new_labels = y_pred.copy()

    idx_tiny = np.where(np.isin(y_pred, tiny))[0]
    if idx_tiny.size > 0:
        P = Z[idx_tiny].astype(np.float32)
        d2 = np.sum(P * P, axis=1, keepdims=True) + np.sum(centers * centers, axis=1) - 2.0 * (P @ centers.T)
        new_labels[idx_tiny] = kept[np.argmin(d2, axis=1)]

    uniq = np.unique(new_labels)
    return np.searchsorted(uniq, new_labels).astype(int)
