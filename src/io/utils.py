# -*- coding: utf-8 -*-
"""Utility functions: seed, clipping, scoring helpers, penalties."""
from typing import List, Tuple

import numpy as np


def set_global_seed(seed: int) -> np.random.RandomState:
    """Create a reproducible random state."""
    return np.random.RandomState(int(seed))


def clip_int(x: float, lo: int, hi: int) -> int:
    """Clip and round to integer."""
    return int(np.clip(int(round(float(x))), lo, hi))


def clip_float(x: float, lo: float, hi: float) -> float:
    """Clip to float range."""
    return float(np.clip(float(x), lo, hi))


def safe_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and std, handling NaN values."""
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def silhouette01_from_value(sil: float) -> float:
    """Normalize silhouette score from [-1, 1] to [0, 1]."""
    if not np.isfinite(sil):
        return 0.0
    return float(np.clip((sil + 1.0) / 2.0, 0.0, 1.0))


def modularity01_from_value(Q: float) -> float:
    """Normalize modularity from [-1, 1] to [0, 1]."""
    if not np.isfinite(Q):
        return 0.0
    return float(np.clip((Q + 1.0) / 2.0, 0.0, 1.0))


def best_map_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute best-mapping accuracy (Hungarian algorithm)."""
    from scipy.optimize import linear_sum_assignment
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)
    cost = np.zeros((len(classes_true), len(classes_pred)), dtype=np.int64)
    for i, ct in enumerate(classes_true):
        mask_t = (y_true == ct)
        for j, cp in enumerate(classes_pred):
            cost[i, j] = np.sum(mask_t & (y_pred == cp))
    row_ind, col_ind = linear_sum_assignment(-cost)
    matched = cost[row_ind, col_ind].sum()
    return float(matched / len(y_true))


def cluster_balance_score(y_pred: np.ndarray) -> Tuple[float, int, int]:
    """Compute entropy-based balance score + singleton/tiny counts."""
    labels, counts = np.unique(y_pred, return_counts=True)
    C = int(len(labels))
    if C <= 1:
        n_singletons = int(np.sum(counts == 1)) if counts.size else 0
        n_tiny = int(np.sum(counts <= 2)) if counts.size else 0
        return 0.0, n_singletons, n_tiny
    p = counts.astype(np.float64) / max(1.0, float(counts.sum()))
    ent = -np.sum(p * np.log(p + 1e-12))
    ent_max = np.log(float(C) + 1e-12)
    balance01 = float(np.clip(ent / (ent_max + 1e-12), 0.0, 1.0))
    n_singletons = int(np.sum(counts == 1))
    n_tiny = int(np.sum(counts <= 2))
    return balance01, n_singletons, n_tiny


def default_c_max(n: int) -> int:
    """Upper bound on cluster count to prevent over-fragmentation."""
    n = int(max(20, n))
    cmax = int(np.clip(np.sqrt(float(n)), 8, 80))
    return int(max(6, cmax))


def compute_target_band(target_c: int, band_alpha: float) -> Tuple[int, int]:
    """Compute target cluster count band."""
    target_c = int(max(2, target_c))
    a = float(np.clip(band_alpha, 0.01, 0.80))
    lo = int(np.ceil((1.0 - a) * target_c))
    hi = int(np.floor((1.0 + a) * target_c))
    lo = max(2, lo)
    hi = max(lo, hi)
    return lo, hi


def target_c_penalty(C: int, target_c: int, band_alpha: float, lam_c_target: float) -> float:
    """Quadratic penalty for cluster count outside target band."""
    if target_c <= 0 or lam_c_target <= 0.0:
        return 0.0
    lo, hi = compute_target_band(target_c, band_alpha)
    if C < lo:
        return float(lam_c_target) * ((lo - C) / max(1, lo)) ** 2
    if C > hi:
        return float(lam_c_target) * ((C - hi) / max(1, hi)) ** 2
    return 0.0
