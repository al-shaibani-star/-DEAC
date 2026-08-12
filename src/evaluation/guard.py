# -*- coding: utf-8 -*-
"""
3-Layer Adaptive QualityGate.

Prevents DE from producing solutions that are internally better
but externally worse (over-fragmentation protection).

Layer 1: Adaptive C-ratio threshold (scales with dataset size)
Layer 2: Moderate C-ratio with small score improvement check
Layer 3: Baseline quality score > DE quality score

QualityGate is disabled for degenerate baselines (C < 2).

Key: C-ratio adapts to dataset size because larger datasets
naturally need more clusters, and baseline C can be far from
ground truth (e.g., ISOLET: baseline C=8, true C=26).
"""
import numpy as np

from config.settings import QG_W_MOD, QG_W_SIL, QG_W_BAL, QG_SCORE_DIFF
from src.io.utils import modularity01_from_value, silhouette01_from_value
from src.evaluation.scoring import RunResult


def quality_score(result: RunResult) -> float:
    """Compute QualityGate score (quality without penalties).

    Score = 0.40*MOD01 + 0.35*SIL01 + 0.25*BAL01

    Returns -1e9 for degenerate results.
    """
    if result.degenerate or result.C < 2:
        return -1e9
    mod01 = modularity01_from_value(result.MOD)
    sil01 = silhouette01_from_value(result.SIL) if np.isfinite(result.SIL) else 0.0
    bal01 = result.bal01 if np.isfinite(result.bal01) else 0.0
    return float(QG_W_MOD * mod01 + QG_W_SIL * sil01 + QG_W_BAL * bal01)


def _adaptive_c_ratio(n_samples: int) -> float:
    """Compute adaptive C-ratio threshold based on dataset size.

    Larger datasets naturally support more clusters, so the QualityGate
    should be more permissive.

    Examples:
        n=400   -> ratio=4.0  (small dataset, strict)
        n=2000  -> ratio=6.3  (medium, moderate)
        n=6006  -> ratio=11.0 (large, permissive)
    """
    return max(4.0, np.sqrt(n_samples / 50.0))


def apply_quality_gate(baseline: RunResult, optimized: RunResult,
                       n_samples: int = 0) -> bool:
    """Apply 3-Layer Adaptive QualityGate.

    Parameters
    ----------
    baseline : RunResult
        Baseline clustering result.
    optimized : RunResult
        DE-optimized clustering result.
    n_samples : int
        Dataset size for adaptive C-ratio.

    Returns
    -------
    bool
        True if baseline should be used instead of optimized.
    """
    # Degenerate baseline — always prefer DE
    if baseline.degenerate or baseline.C < 2:
        return False

    base_score = quality_score(baseline)
    de_score = quality_score(optimized)

    # Adaptive C-ratio threshold
    c_ratio_l1 = _adaptive_c_ratio(n_samples) if n_samples > 0 else 4.0
    c_ratio_l2 = max(2.0, c_ratio_l1 * 0.5)

    # Layer 1: Extreme over-fragmentation (adaptive)
    if optimized.C >= c_ratio_l1 * baseline.C:
        return True

    # Layer 2: Moderate over-fragmentation with small score improvement
    if optimized.C > c_ratio_l2 * baseline.C and (de_score - base_score) < QG_SCORE_DIFF:
        return True

    # Layer 3: DE's internal quality is worse
    if base_score > de_score:
        return True

    return False
