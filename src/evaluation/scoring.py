# -*- coding: utf-8 -*-
"""Evaluation metrics: internal scoring, external metrics, RunResult dataclass."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.io.utils import (
    modularity01_from_value, silhouette01_from_value,
    cluster_balance_score, default_c_max, target_c_penalty, best_map_acc,
)


def internal_score_components(
    Q: float, SIL: float, y_pred: np.ndarray, ncomp: int, n: int,
    w_mod: float, w_sil: float, w_bal: float,
    lam_c: float, lam_singleton: float, lam_tiny: float, lam_components: float,
    target_c: int, band_alpha: float, lam_c_target: float,
) -> Tuple[float, float, float, float, float, float]:
    """Compute internal fitness score with penalties."""
    C = int(np.unique(y_pred).size)
    mod01 = modularity01_from_value(Q)
    sil01 = silhouette01_from_value(SIL)
    bal01, n_sing, n_tiny = cluster_balance_score(y_pred)
    pen = 0.0
    if ncomp > 1:
        pen += lam_components * float(ncomp - 1)
    cmax = default_c_max(n)
    if C > cmax:
        pen += lam_c * float(C - cmax) / float(max(1, cmax))
    pen += lam_singleton * (n_sing / float(max(1, n)))
    pen += lam_tiny * (n_tiny / float(max(1, n)))
    pen += target_c_penalty(C, target_c, band_alpha, lam_c_target)
    score = (w_mod * mod01) + (w_sil * sil01) + (w_bal * bal01) - pen
    return float(score), float(mod01), float(sil01), float(bal01), float(pen), float(C)


def evaluate_run(Z: np.ndarray, y_true: Optional[np.ndarray], y_pred: np.ndarray,
                 C: int, Q: float, sil_metric: str) -> Tuple[float, float, float, float, float]:
    """Compute external metrics (ARI, NMI, ACC) and internal metrics (SIL, MOD)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    if y_true is not None:
        if C < 2:
            ARI = NMI = ACC = 0.0
        else:
            ARI = float(adjusted_rand_score(y_true, y_pred))
            NMI = float(normalized_mutual_info_score(y_true, y_pred))
            ACC = float(best_map_acc(y_true, y_pred))
    else:
        ARI = NMI = ACC = float("nan")

    SIL = float("nan")
    if 2 <= C < len(y_pred):
        try:
            SIL = float(silhouette_score(Z, y_pred, metric=sil_metric))
        except:
            pass
    MOD = float(Q) if np.isfinite(Q) else float("nan")
    return ARI, NMI, ACC, SIL, MOD


@dataclass
class RunResult:
    """Container for a single pipeline run's results."""
    seed: int
    mode: str
    k: int
    dr_dim: int
    prune_q: float
    resolution: float
    C: int
    edges: int
    n_components: int
    comp_min: int
    comp_med: float
    comp_max: int
    degenerate: bool
    ARI: float
    NMI: float
    ACC: float
    SIL: float
    MOD: float
    bal01: float
    n_singletons: int
    n_tiny: int
    internal_score: float
    internal_pen: float
    time_total_sec: float
    time_opt_sec: float = 0.0
