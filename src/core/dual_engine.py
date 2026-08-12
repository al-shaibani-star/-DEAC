# -*- coding: utf-8 -*-
"""
Dual-Engine Adaptive Clustering (DEAC).

Runs two independent clustering engines and selects the best:
  Engine 1: AMR-DE (UMAP + DE hyperparameter optimization)
  Engine 2: EMR-CGC (SNN + Leiden + multi-representation ensemble)

The final result = max(Baseline, AMR-DE, EMR-CGC) by internal quality score.

This guarantees:
  - Result >= Baseline always (QualityGate)
  - Best of both worlds: DE optimization + ensemble diversity
"""
import numpy as np
from typing import Optional

from src.evaluation.scoring import RunResult
from src.io.utils import modularity01_from_value, silhouette01_from_value


def _quality_score(result: RunResult) -> float:
    """Compute quality score for a clustering result."""
    if result.degenerate or result.C < 2:
        return -1e9
    mod01 = modularity01_from_value(result.MOD)
    sil01 = silhouette01_from_value(result.SIL) if np.isfinite(result.SIL) else 0.0
    bal01 = result.bal01 if np.isfinite(result.bal01) else 0.0
    return float(0.40 * mod01 + 0.35 * sil01 + 0.25 * bal01)


def select_best(baseline: RunResult,
                amr_result: Optional[RunResult],
                emr_result: Optional[RunResult]) -> tuple:
    """Select best result among Baseline, AMR-DE, EMR-CGC.

    Returns
    -------
    (best_result, source_name, all_scores)
    """
    candidates = [("Baseline", baseline, _quality_score(baseline))]

    if amr_result is not None:
        candidates.append(("AMR-DE", amr_result, _quality_score(amr_result)))

    if emr_result is not None:
        candidates.append(("EMR-CGC", emr_result, _quality_score(emr_result)))

    # Select by highest quality score
    best_name, best_result, best_score = max(candidates, key=lambda x: x[2])

    all_scores = {name: score for name, _, score in candidates}

    return best_result, best_name, all_scores
