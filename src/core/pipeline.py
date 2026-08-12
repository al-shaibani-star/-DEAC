# -*- coding: utf-8 -*-
"""Pipeline: run_pipeline_once orchestrates the full clustering flow."""
import time
from typing import Optional

import numpy as np

from config.settings import LAM_C, LAM_SING, LAM_TINY, LAM_COMP, TARGET_C, BAND_ALPHA, LAM_C_TARGET
from src.io.representation import make_representation
from src.graph.construction import build_knn_graph, components_stats_from_edges, keep_graph_connected_by_bridging
from src.graph.clustering import cluster_graph
from src.graph.postprocessing import postprocess_tiny_clusters
from src.evaluation.scoring import evaluate_run, RunResult, internal_score_components
from src.io.utils import cluster_balance_score


def run_pipeline_once(
    X, y: Optional[np.ndarray], seed: int, metric: str,
    k: int, dr_dim: int, prune_q: float, resolution: float,
    w_mod: float, w_sil: float, w_bal: float,
    graph_mode: str = "symmetric", keep_connected: int = 1, min_deg_guard: int = 3,
    min_cluster_size_ratio: float = 0.005,
    use_umap: bool = False,
    rep_type: str = "pca",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: float = 30.0,
) -> RunResult:
    """Execute the complete clustering pipeline once.

    Flow: PCA/UMAP/t-SNE -> KNN Graph -> Louvain -> Post-process -> Evaluate
    """
    t0 = time.time()
    sil_metric = "cosine" if metric == "cosine" else "euclidean"
    prune_q = float(np.clip(prune_q, 0.0, 0.40))

    # Step 1: Dimensionality Reduction (PCA, UMAP, or t-SNE)
    Z = make_representation(X, seed=seed, dr_dim=dr_dim, use_umap=use_umap,
                            rep_type=rep_type, n_neighbors=n_neighbors,
                            min_dist=min_dist, perplexity=perplexity)

    # Step 2: Graph Construction
    n, edges = build_knn_graph(Z, k=k, metric=metric, graph_mode=graph_mode,
                               prune_q=prune_q, min_deg_guard=min_deg_guard)
    if keep_connected == 1:
        edges = keep_graph_connected_by_bridging(Z, edges, metric=metric)

    ncomp, cmin, cmed, cmax = components_stats_from_edges(n, edges)

    # Step 3: Graph Clustering (Louvain)
    y_pred, C, degenerate, m_edges, Q, _comms = cluster_graph(
        n=n, edges=edges, seed=seed, resolution=resolution
    )

    # Step 4: Post-processing (merge tiny clusters)
    if (not degenerate) and (C > 1) and (min_cluster_size_ratio > 0):
        y_pred2 = postprocess_tiny_clusters(y_pred, Z, min_size_ratio=min_cluster_size_ratio)
        C2 = int(np.unique(y_pred2).size)
        if C2 != C:
            y_pred = y_pred2
            C = C2
            try:
                import networkx as nx
                from networkx.algorithms.community.quality import modularity as nx_modularity
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for i, j, w in edges:
                    if i != j:
                        G.add_edge(int(i), int(j), weight=float(w))
                comms = [set(np.where(y_pred == l)[0]) for l in np.unique(y_pred)]
                Q = float(nx_modularity(G, comms, weight="weight"))
            except:
                pass

    # Step 5: Evaluation
    ARI, NMI, ACC, SIL, MOD = evaluate_run(Z, y, y_pred, C=C, Q=Q, sil_metric=sil_metric)
    bal01, n_sing, n_tiny = cluster_balance_score(y_pred)
    score, *_, pen, _ = internal_score_components(
        Q=MOD, SIL=SIL, y_pred=y_pred, ncomp=ncomp, n=n,
        w_mod=w_mod, w_sil=w_sil, w_bal=w_bal,
        lam_c=LAM_C, lam_singleton=LAM_SING, lam_tiny=LAM_TINY, lam_components=LAM_COMP,
        target_c=TARGET_C, band_alpha=BAND_ALPHA, lam_c_target=LAM_C_TARGET,
    )

    return RunResult(
        seed=int(seed), mode="baseline",
        k=int(k), dr_dim=int(dr_dim), prune_q=float(prune_q), resolution=float(resolution),
        C=int(C), edges=int(m_edges),
        n_components=int(ncomp), comp_min=int(cmin), comp_med=float(cmed), comp_max=int(cmax),
        degenerate=bool(degenerate),
        ARI=float(ARI), NMI=float(NMI), ACC=float(ACC), SIL=float(SIL), MOD=float(MOD),
        bal01=float(bal01), n_singletons=int(n_sing), n_tiny=int(n_tiny),
        internal_score=float(score), internal_pen=float(pen),
        time_total_sec=float(time.time() - t0),
    )
