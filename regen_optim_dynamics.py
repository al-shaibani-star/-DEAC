# -*- coding: utf-8 -*-
"""
Regenerate the Optimization Dynamics figure with REAL Differential Evolution
convergence + parameter-trajectory data, correctly labelled "DE" (the previous
images were empty and mislabelled "GWO"/"LevGWO" — recycled from another repo).

Captures per-generation best fitness and best (k, q, r) via a scipy
differential_evolution callback, for MFeat and COIL20 (the two datasets shown
in the paper figure). Overwrites the four PNGs in place (filenames unchanged so
the .tex needs no edit).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.io.data_loading import load_csv
from src.io.utils import set_global_seed
from src.core.fast_amr_de import _quick_evaluate_rep, _build_fitness_4d
from config.settings import (K_LO, K_HI, D_LO, D_HI, Q_LO, Q_HI, R_LO, R_HI,
                             METRIC, SUBSAMPLE)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figures")

DATASETS = {"MFeat": "mfeat_full.csv", "COIL20": "COIL20.csv"}
SEEDS = [1001, 1002, 1003, 1004, 1005]
BOUNDS = [(K_LO, K_HI), (D_LO, D_HI), (Q_LO, Q_HI), (R_LO, R_HI)]
MAXITER = 12
POPSIZE = 5

REP_CONFIGS = [
    ("pca", {}),
    ("umap", {"n_neighbors": 15, "min_dist": 0.1}),
    ("umap", {"n_neighbors": 10, "min_dist": 0.0}),
    ("umap", {"n_neighbors": 30, "min_dist": 0.2}),
    ("tsne", {"perplexity": 30}),
]


def best_representation(X_sub, seed):
    best_score, best_Z, best_rep = -1e9, None, "pca"
    for rep_type, params in REP_CONFIGS:
        score, Z = _quick_evaluate_rep(
            X_sub, seed, rep_type, METRIC, "symmetric", 3,
            n_neighbors=params.get("n_neighbors", 15),
            min_dist=params.get("min_dist", 0.1),
            perplexity=params.get("perplexity", 30))
        if Z is not None and score > best_score:
            best_score, best_Z, best_rep = score, Z, rep_type
    return best_Z, best_rep


def run_de_logged(Z, seed):
    """Run DE with a callback; return best-so-far (score_per_gen, params_per_gen).

    Fitness is made deterministic (silhouette every eval, full sample) so the
    best-so-far convergence curve is monotonic and meaningful.
    """
    fit = _build_fitness_4d(Z, seed, METRIC, "symmetric",
                            sil_every=1, sil_sample=0, min_deg_guard=3)
    evals = []  # (fitness, x) for every evaluation, in order

    def logged(v):
        vv = np.asarray(v, np.float64)
        f = fit(vv)
        evals.append((float(f), vv.copy()))
        return f

    n_dims = len(BOUNDS)
    pop_total = POPSIZE * n_dims
    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    lb = np.array([b[0] for b in BOUNDS]); ub = np.array([b[1] for b in BOUNDS])
    init_pop = qmc.scale(sampler.random(n=pop_total), lb, ub)
    differential_evolution(
        logged, BOUNDS, maxiter=MAXITER, init=init_pop, seed=seed,
        polish=False, updating="deferred", workers=1, tol=0)

    # group evaluations into generations of size pop_total; best-so-far per gen
    n_gen = max(1, len(evals) // pop_total)
    scores, params = [], []
    best_f, best_x = np.inf, evals[0][1]
    for g in range(n_gen):
        chunk = evals[g * pop_total:(g + 1) * pop_total]
        for f, x in chunk:
            if f < best_f:
                best_f, best_x = f, x
        scores.append(-best_f)          # best-so-far score (monotonic)
        params.append(best_x.copy())
    return np.array(scores), np.array(params)


def main():
    for name, csv in DATASETS.items():
        print(f"[{name}] loading...")
        X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
        rng = set_global_seed(1001)
        n = X.shape[0]
        sub = min(SUBSAMPLE, n)
        idx = rng.choice(n, size=sub, replace=False) if sub < n else np.arange(n)
        X_sub = np.asarray(X, np.float32)[idx]

        Z, rep = best_representation(X_sub, 1001)
        print(f"[{name}] best rep = {rep}; running DE over {len(SEEDS)} seeds...")

        all_curves = {}
        traj = None
        for s in SEEDS:
            sc, pr = run_de_logged(Z, s)
            all_curves[s] = sc
            if s == 1001:
                traj = pr
            print(f"   seed {s}: {len(sc)} gens, final score={sc[-1]:.4f}")

        out_dir = os.path.join(FIG_DIR, name)
        os.makedirs(out_dir, exist_ok=True)

        # ---- Convergence curve ----
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        colors = plt.cm.tab10(np.linspace(0, 1, len(SEEDS)))
        for (s, sc), c in zip(all_curves.items(), colors):
            ax.plot(range(1, len(sc) + 1), sc, "-o", ms=3, lw=1.4,
                    color=c, label=f"Seed {s}")
        ax.set_xlabel("DE generation")
        ax.set_ylabel("Best fitness score (higher = better)")
        ax.set_title(f"DE Convergence Curve — {name}", fontweight="bold")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"de_convergence_{name}.png"), dpi=200)
        plt.close(fig)

        # ---- Parameter trajectory (seed 1001) ----
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        gens = range(1, len(traj) + 1)
        norm = lambda col, lo, hi: (traj[:, col] - lo) / (hi - lo)
        ax.plot(gens, norm(0, K_LO, K_HI), "-o", ms=3, lw=1.4, label="$k$ (neighbors)")
        ax.plot(gens, norm(2, Q_LO, Q_HI), "-s", ms=3, lw=1.4, label="$q$ (pruning)")
        ax.plot(gens, norm(3, R_LO, R_HI), "-^", ms=3, lw=1.4, label="$r$ (resolution)")
        ax.set_xlabel("DE generation")
        ax.set_ylabel("Normalized parameter value")
        ax.set_title(f"DE Parameter Trajectory — {name}", fontweight="bold")
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"param_trajectory_{name}.png"), dpi=200)
        plt.close(fig)
        print(f"[{name}] saved 2 figures.")


if __name__ == "__main__":
    main()
