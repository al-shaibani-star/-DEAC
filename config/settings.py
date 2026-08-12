# -*- coding: utf-8 -*-
"""
Configuration: All hyperparameters and constants for AMR-GraphClust.
Adaptive Multi-Representation Graph Clustering.
"""

# ============================================================
# Search Space Bounds
# ============================================================
K_LO, K_HI = 5, 80          # KNN neighbors
D_LO, D_HI = 5, 100         # PCA dimensions
Q_LO, Q_HI = 0.0, 0.40      # Edge pruning quantile
R_LO, R_HI = 0.01, 0.90     # Louvain resolution

BOUNDS = [(K_LO, K_HI), (D_LO, D_HI), (Q_LO, Q_HI), (R_LO, R_HI)]

# ============================================================
# Fitness Penalties
# ============================================================
LAM_C = 1.5           # Over-fragmentation penalty
LAM_SING = 2.5        # Singleton cluster penalty
LAM_TINY = 1.2        # Tiny cluster penalty
LAM_COMP = 3.0        # Disconnected component penalty
TARGET_C = 0          # Target cluster count (0 = disabled)
BAND_ALPHA = 0.15     # Target band width
LAM_C_TARGET = 3.0    # Target penalty weight

# ============================================================
# AMR-DE Parameters (Adaptive Multi-Representation DE)
# ============================================================
AMR_DE_MAXITER = 8      # DE generations per restart
AMR_DE_POPSIZE = 5      # DE population multiplier
AMR_N_RESTARTS = 2      # Number of independent restarts

# ============================================================
# Experiment Settings
# ============================================================
SEEDS = 5
SUBSAMPLE = 800
SIL_EVERY = 8
SIL_SAMPLE = 400

# ============================================================
# Fitness Weights
# ============================================================
W_MOD = 0.50          # Modularity weight
W_SIL = 0.40          # Silhouette weight (increased -- correlates better with ARI)
W_BAL = 0.10          # Balance weight

# ============================================================
# QualityGate Weights (no penalties)
# ============================================================
QG_W_MOD = 0.40
QG_W_SIL = 0.35
QG_W_BAL = 0.25
QG_SCORE_DIFF = 0.06   # Layer 2: score diff threshold

# ============================================================
# Baseline Parameters
# ============================================================
METRIC = "cosine"
GRAPH_MODE = "symmetric"
BASELINE_K = 35
BASELINE_DR_DIM = 40
BASELINE_PRUNE_Q = 0.10
BASELINE_RESOLUTION = 0.25

# ============================================================
# Output
# ============================================================
OUTPUT_BASE = "results"
