# -*- coding: utf-8 -*-
"""
Fast AMR-DE: Two-stage approach for speed.

Stage 1: Quick scan — evaluate PCA, UMAP, t-SNE with default params.
         Pick the best representation.
Stage 2: DE optimizes graph params (k, q, r) on the FIXED best representation.

This is 10-50x faster than full AMR-DE because UMAP/t-SNE is computed
once per representation, not per fitness evaluation.

4D search space: [k, d', q, r] (same as original, but on best representation)
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from src.io.utils import (
    set_global_seed, clip_int, clip_float,
    modularity01_from_value, silhouette01_from_value,
    cluster_balance_score, default_c_max, target_c_penalty,
)
from src.graph.construction import (
    _vectorized_dist_to_weight, components_stats_from_edges,
    keep_graph_connected_by_bridging,
)
from src.graph.clustering import cluster_graph
from config.settings import (
    K_LO, K_HI, D_LO, D_HI, Q_LO, Q_HI, R_LO, R_HI,
    LAM_C, LAM_SING, LAM_TINY, LAM_COMP,
    TARGET_C, BAND_ALPHA, LAM_C_TARGET,
)


@dataclass
class FastAMRResult:
    """Result from Fast AMR-DE."""
    best_k: int
    best_d: int
    best_q: float
    best_r: float
    best_rep: str  # "pca", "umap", or "tsne"
    best_n_neighbors: int
    best_min_dist: float
    best_perplexity: float
    opt_time: float
    best_fitness: float
    stage1_scores: dict = field(default_factory=dict)
    restart_results: List[dict] = field(default_factory=list)


def _quick_evaluate_rep(X_sub, seed, rep_type, metric, graph_mode,
                         min_deg_guard, n_neighbors=15, min_dist=0.1,
                         perplexity=30):
    """Quickly evaluate a representation with default graph params."""
    from src.io.representation import make_representation
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    try:
        Z = make_representation(X_sub, seed=seed, dr_dim=40,
                                rep_type=rep_type, n_neighbors=n_neighbors,
                                min_dist=min_dist, perplexity=perplexity)

        from sklearn.neighbors import NearestNeighbors
        n = Z.shape[0]
        k = min(35, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=1)
        nn.fit(Z)
        dists, idxs = nn.kneighbors(Z, return_distance=True)
        dists, idxs = dists[:, 1:], idxs[:, 1:]
        weights = _vectorized_dist_to_weight(dists.ravel(), metric).reshape(dists.shape)

        dir_w = {}
        for i in range(n):
            for t in range(k):
                dir_w[(i, int(idxs[i, t]))] = float(weights[i, t])

        edges_map = {}
        for (i, j), w in dir_w.items():
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in edges_map:
                edges_map[(a, b)] = 0.5 * (edges_map[(a, b)] + w)
            else:
                edges_map[(a, b)] = w

        edges = [(a, b, w) for (a, b), w in edges_map.items()]
        edges = keep_graph_connected_by_bridging(Z, edges, metric=metric)

        y_pred, C, degen, m, Q, _ = cluster_graph(n=n, edges=edges,
                                                    seed=seed, resolution=0.25)
        if C < 2 or degen:
            return -1e9, Z

        mod01 = modularity01_from_value(Q)
        sil = float(silhouette_score(Z, y_pred, metric=metric,
                                      sample_size=min(300, n)))
        sil01 = silhouette01_from_value(sil)
        ch = float(calinski_harabasz_score(Z, y_pred))
        ch01 = float(np.clip(ch / 1000.0, 0.0, 1.0))

        score = 0.30 * mod01 + 0.40 * sil01 + 0.30 * ch01
        return float(score), Z
    except Exception:
        return -1e9, None


def _build_fitness_4d(Z_fixed, seed, metric, graph_mode, sil_every,
                       sil_sample, min_deg_guard):
    """Build 4D fitness on pre-computed representation Z_fixed."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                                  davies_bouldin_score)

    rng = set_global_seed(seed)
    n = Z_fixed.shape[0]
    cmax = default_c_max(n)
    counter = {"n": 0}
    base_scale = np.log10(max(100, n)) / 2.0
    MAX_K = min(140, n - 1)

    # Pre-compute KNN once (for max k)
    nn = NearestNeighbors(n_neighbors=min(MAX_K + 1, n - 1),
                          metric=metric, n_jobs=1)
    nn.fit(Z_fixed)
    d_all, i_all = nn.kneighbors(Z_fixed, return_distance=True)

    def fitness(vec):
        try:
            k = clip_int(vec[0], K_LO, K_HI)
            dr_dim = clip_int(vec[1], D_LO, D_HI)  # ignored (Z is fixed)
            prune_q = clip_float(vec[2], Q_LO, Q_HI)
            resolution = clip_float(vec[3], R_LO, R_HI)

            k = min(k, d_all.shape[1] - 1)
            dists = d_all[:, 1:k + 1]
            idxs = i_all[:, 1:k + 1]
            weights = _vectorized_dist_to_weight(dists.ravel(), metric).reshape(dists.shape)

            dir_w = {}
            for i in range(n):
                for t in range(k):
                    dir_w[(i, int(idxs[i, t]))] = float(weights[i, t])
            if not dir_w:
                return 1e9

            edges_map = {}
            for (i, j), w in dir_w.items():
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in edges_map:
                    edges_map[(a, b)] = 0.5 * (edges_map[(a, b)] + w)
                else:
                    edges_map[(a, b)] = w

            if prune_q > 0:
                wvals = np.asarray(list(edges_map.values()), dtype=np.float32)
                thr = float(np.quantile(wvals, prune_q))
                deg = np.zeros(n, dtype=np.int32)
                for a, b in edges_map:
                    deg[a] += 1
                    deg[b] += 1
                edges_map = {(a, b): w for (a, b), w in edges_map.items()
                             if w >= thr or deg[a] <= min_deg_guard or deg[b] <= min_deg_guard}

            edges = [(a, b, w) for (a, b), w in edges_map.items()]
            edges = keep_graph_connected_by_bridging(Z_fixed, edges, metric=metric)
            ncomp = components_stats_from_edges(n, edges)[0]

            y_pred, C, degen, m, Q, _ = cluster_graph(n=n, edges=edges,
                                                       seed=seed, resolution=resolution)
            if m <= 0 or C < 2 or degen:
                return 1e9

            counter["n"] += 1
            mod01 = modularity01_from_value(Q)
            bal01, n_sing, n_tiny = cluster_balance_score(y_pred)

            sil01 = ch01 = db01 = 0.0
            do_extra = (counter["n"] == 1 or
                        (sil_every > 0 and counter["n"] % sil_every == 0)) and \
                       2 <= C < min(n, 120)
            if do_extra:
                try:
                    if sil_sample > 0 and n > sil_sample:
                        idx = rng.choice(n, size=sil_sample, replace=False)
                        Z_s, y_s = Z_fixed[idx], y_pred[idx]
                    else:
                        Z_s, y_s = Z_fixed, y_pred
                    sil01 = silhouette01_from_value(
                        float(silhouette_score(Z_s, y_s, metric=metric)))
                    ch = float(calinski_harabasz_score(Z_s, y_s))
                    ch01 = float(np.clip(ch / 1000.0, 0.0, 1.0))
                    db = float(davies_bouldin_score(Z_s, y_s))
                    db01 = float(np.clip(1.0 - db / 5.0, 0.0, 1.0))
                except:
                    pass

            pen = 0.0
            if ncomp > 1:
                pen += LAM_COMP * base_scale * (ncomp - 1)
            if C > cmax:
                pen += LAM_C * base_scale * (C - cmax) / max(1, cmax)
            pen += LAM_SING * base_scale * (n_sing / max(1, n))
            pen += LAM_TINY * base_scale * (n_tiny / max(1, n))
            pen += target_c_penalty(C, TARGET_C, BAND_ALPHA, LAM_C_TARGET * base_scale)

            score = (0.30 * mod01 + 0.25 * sil01 + 0.20 * ch01 +
                     0.15 * db01 + 0.10 * bal01) - pen
            return -float(score)
        except:
            return 1e9

    return fitness


def run_fast_amr_de(X, y, seed, metric="cosine", subsample=800,
                     de_maxiter=15, de_popsize=8, n_restarts=3,
                     sil_every=4, sil_sample=400, graph_mode="symmetric",
                     min_deg_guard=3, verbose=0,
                     max_dr_dim=0, max_k=0) -> FastAMRResult:
    """Fast AMR-DE: select best representation first, then optimize graph params.

    10-50x faster than full AMR-DE.
    """
    rng = set_global_seed(seed)
    n = int(X.shape[0])

    if subsample > 0 and subsample < n:
        idx = rng.choice(n, size=subsample, replace=False)
        X_sub = np.asarray(X, dtype=np.float32)[idx]
    else:
        X_sub = np.asarray(X, dtype=np.float32)

    n_sub = X_sub.shape[0]
    if n_sub < 30:
        return FastAMRResult(35, 40, 0.10, 0.25, "pca", 15, 0.1, 30.0, 0.0, 1e9)

    t0 = time.time()

    # ============================================================
    # Stage 1: Quick representation scan
    # ============================================================
    if verbose:
        print("  [Stage 1] Scanning representations...", flush=True)

    rep_configs = [
        ("pca", {}),
        ("umap", {"n_neighbors": 15, "min_dist": 0.1}),
        ("umap", {"n_neighbors": 10, "min_dist": 0.0}),
        ("umap", {"n_neighbors": 30, "min_dist": 0.2}),
        ("tsne", {"perplexity": 30}),
    ]

    best_rep_score = -1e9
    best_rep_type = "pca"
    best_rep_params = {}
    best_Z = None
    stage1_scores = {}

    for rep_type, params in rep_configs:
        label = rep_type
        if params:
            label += f"({','.join(f'{k}={v}' for k,v in params.items())})"

        score, Z = _quick_evaluate_rep(
            X_sub, seed, rep_type, metric, graph_mode, min_deg_guard,
            n_neighbors=params.get("n_neighbors", 15),
            min_dist=params.get("min_dist", 0.1),
            perplexity=params.get("perplexity", 30),
        )
        stage1_scores[label] = float(score)
        if verbose:
            print(f"    {label}: score={score:.4f}")

        if score > best_rep_score and Z is not None:
            best_rep_score = score
            best_rep_type = rep_type
            best_rep_params = params
            best_Z = Z

    if best_Z is None:
        # Fallback to PCA
        from src.io.representation import make_representation
        best_Z = make_representation(X_sub, seed=seed, dr_dim=40, rep_type="pca")
        best_rep_type = "pca"

    if verbose:
        print(f"  [Stage 1] Best: {best_rep_type} (score={best_rep_score:.4f})")

    # ============================================================
    # Stage 2: Multi-restart DE on fixed representation
    # ============================================================
    d_hi = min(D_HI, max(D_LO, max_dr_dim)) if max_dr_dim > 0 else D_HI
    k_hi = min(K_HI, max(K_LO, max_k)) if max_k > 0 else K_HI

    bounds = [(K_LO, k_hi), (D_LO, d_hi), (Q_LO, Q_HI), (R_LO, R_HI)]

    fit = _build_fitness_4d(best_Z, seed, metric, graph_mode,
                            sil_every, sil_sample, min_deg_guard)

    all_restart_results = []
    global_best = None
    global_best_score = 1e9

    for restart in range(n_restarts):
        restart_seed = seed + restart * 1000
        n_dims = len(bounds)
        pop_total = de_popsize * n_dims

        # LHS initialization
        sampler = qmc.LatinHypercube(d=n_dims, seed=restart_seed)
        lhs = sampler.random(n=pop_total)
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        init_pop = qmc.scale(lhs, lb, ub)

        if verbose:
            print(f"  [Stage 2] Restart {restart+1}/{n_restarts} "
                  f"(pop={pop_total})...", flush=True)

        res = differential_evolution(
            lambda v: fit(np.asarray(v, np.float64)),
            bounds, maxiter=de_maxiter, init=init_pop,
            seed=restart_seed, polish=False,
            updating="deferred", workers=1,
        )

        # NM refinement
        try:
            nm = minimize(lambda v: fit(np.asarray(v, np.float64)),
                          res.x, method="Nelder-Mead",
                          options={"maxiter": 50, "maxfev": 60})
            if nm.fun < res.fun:
                best_x, best_f = nm.x, nm.fun
            else:
                best_x, best_f = res.x, res.fun
        except:
            best_x, best_f = res.x, res.fun

        restart_info = {
            "restart": restart,
            "fitness": float(best_f),
            "k": clip_int(best_x[0], K_LO, k_hi),
            "d": clip_int(best_x[1], D_LO, d_hi),
            "q": clip_float(best_x[2], Q_LO, Q_HI),
            "r": clip_float(best_x[3], R_LO, R_HI),
        }
        all_restart_results.append(restart_info)

        if verbose:
            print(f"    -> fitness={best_f:.4f}, k={restart_info['k']}, "
                  f"d={restart_info['d']}, r={restart_info['r']:.2f}")

        if best_f < global_best_score:
            global_best_score = best_f
            global_best = best_x.copy()

    opt_time = time.time() - t0
    if verbose:
        print(f"  [Fast AMR-DE] Done in {opt_time:.1f}s (rep={best_rep_type})")

    return FastAMRResult(
        best_k=clip_int(global_best[0], K_LO, k_hi),
        best_d=clip_int(global_best[1], D_LO, d_hi),
        best_q=clip_float(global_best[2], Q_LO, Q_HI),
        best_r=clip_float(global_best[3], R_LO, R_HI),
        best_rep=best_rep_type,
        best_n_neighbors=best_rep_params.get("n_neighbors", 15),
        best_min_dist=best_rep_params.get("min_dist", 0.1),
        best_perplexity=best_rep_params.get("perplexity", 30),
        opt_time=opt_time,
        best_fitness=float(global_best_score),
        stage1_scores=stage1_scores,
        restart_results=all_restart_results,
    )
