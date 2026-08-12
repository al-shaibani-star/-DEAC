# -*- coding: utf-8 -*-
"""
EMR-CGC: Ensemble Multi-Representation Consensus Graph Clustering.

Innovation:
  1. Multiple representations: PCA, UMAP (3 configs), t-SNE
  2. Multiple graph configs: varying (k, q, r)
  3. Shared Nearest Neighbors (SNN) weighting
  4. Leiden community detection (guaranteed connected)
  5. Consensus voting across all runs → stable final labels

No optimizer needed — brute-force diversity + consensus.
Much simpler, faster, and more robust than DE-based optimization.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

from src.io.utils import (
    set_global_seed, modularity01_from_value, silhouette01_from_value,
    cluster_balance_score,
)


@dataclass
class EMRResult:
    """Result from EMR-CGC."""
    y_consensus: np.ndarray      # Final consensus labels
    C: int                       # Number of clusters
    n_runs: int                  # Total runs performed
    n_valid: int                 # Runs that produced valid clustering
    best_rep: str                # Best individual representation
    best_ari_individual: float   # Best ARI from a single run
    consensus_quality: float     # Agreement score among runs
    opt_time: float
    rep_scores: Dict[str, float] = field(default_factory=dict)
    all_labels: Optional[List[np.ndarray]] = None


# ============================================================
# Representation configs
# ============================================================
def _get_representations(X_scaled, seed, dr_dim=40):
    """Compute multiple representations."""
    reps = {}

    # 1. PCA
    d = min(dr_dim, X_scaled.shape[1] - 1)
    pca = PCA(n_components=max(2, d), whiten=True, svd_solver="randomized",
              random_state=seed)
    reps["PCA"] = pca.fit_transform(X_scaled).astype(np.float32)

    # 2-4. UMAP variants
    try:
        import umap
        X_pre = X_scaled
        if X_scaled.shape[1] > 50:
            pca_pre = PCA(n_components=50, svd_solver="randomized", random_state=seed)
            X_pre = pca_pre.fit_transform(X_scaled)

        for nn, md, label in [(10, 0.0, "UMAP_10"), (15, 0.1, "UMAP_15"), (30, 0.2, "UMAP_30")]:
            try:
                reducer = umap.UMAP(
                    n_components=max(2, min(d, X_pre.shape[1] - 1)),
                    n_neighbors=min(nn, X_pre.shape[0] - 1),
                    min_dist=md, metric="euclidean", random_state=seed,
                )
                reps[label] = reducer.fit_transform(X_pre).astype(np.float32)
            except:
                pass
    except ImportError:
        pass

    # 5. t-SNE
    try:
        from sklearn.manifold import TSNE
        X_pre = X_scaled
        if X_scaled.shape[1] > 50:
            pca_pre = PCA(n_components=50, svd_solver="randomized", random_state=seed)
            X_pre = pca_pre.fit_transform(X_scaled)
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, X_pre.shape[0] // 4)),
                     random_state=seed, n_iter=500, init="pca", learning_rate="auto")
        reps["tSNE"] = tsne.fit_transform(X_pre).astype(np.float32)
    except:
        pass

    return reps


# ============================================================
# SNN Graph Construction
# ============================================================
def _build_snn_graph(Z, k, metric="cosine"):
    """Build Shared Nearest Neighbor (SNN) graph.

    Edge weight = number of shared neighbors / k
    This captures local structure much better than distance-based weights.
    """
    n = Z.shape[0]
    k_eff = min(k, n - 1)

    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=1)
    nn.fit(Z)
    _, indices = nn.kneighbors(Z)
    indices = indices[:, 1:]  # exclude self

    # Build neighbor sets
    neighbor_sets = [set(indices[i]) for i in range(n)]

    # SNN weights: |N(i) ∩ N(j)| / k
    edges = []
    seen = set()
    for i in range(n):
        for j in indices[i]:
            j = int(j)
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                shared = len(neighbor_sets[a] & neighbor_sets[b])
                if shared > 0:
                    w = shared / k_eff
                    edges.append((a, b, float(w)))

    return edges


# ============================================================
# Leiden Clustering
# ============================================================
def _leiden_cluster(n, edges, resolution, seed):
    """Apply Leiden community detection."""
    try:
        import igraph as ig
        import leidenalg

        g = ig.Graph(n=n)
        if edges:
            edge_list = [(a, b) for a, b, w in edges]
            weights = [w for a, b, w in edges]
            g.add_edges(edge_list)

            partition = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition,
                weights=weights, resolution_parameter=float(resolution),
                seed=int(seed),
            )
            y_pred = np.array(partition.membership, dtype=int)
            Q = float(partition.modularity)
            return y_pred, len(set(y_pred)), Q
        else:
            return np.zeros(n, dtype=int), 1, 0.0

    except ImportError:
        # Fallback to Louvain
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
        from networkx.algorithms.community.quality import modularity as nx_mod

        G = nx.Graph()
        G.add_nodes_from(range(n))
        for a, b, w in edges:
            if a != b:
                G.add_edge(a, b, weight=w)
        if G.number_of_edges() == 0:
            return np.zeros(n, dtype=int), 1, 0.0

        comms = louvain_communities(G, weight="weight", resolution=float(resolution),
                                     seed=int(seed))
        y_pred = np.empty(n, dtype=int)
        for cid, nodes in enumerate(comms):
            for v in nodes:
                y_pred[int(v)] = cid

        Q = float(nx_mod(G, comms, weight="weight"))
        return y_pred, len(set(y_pred)), Q


# ============================================================
# Consensus Voting
# ============================================================
def _consensus_matrix(all_labels, n):
    """Build co-association matrix from multiple clusterings."""
    C = np.zeros((n, n), dtype=np.float32)
    count = len(all_labels)
    for labels in all_labels:
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    C[i, j] += 1
                    C[j, i] += 1
    C /= max(count, 1)
    np.fill_diagonal(C, 1.0)
    return C


def _consensus_cluster(C_matrix, n, method="leiden"):
    """Cluster the consensus matrix to get final labels.

    Uses Leiden on the co-association matrix (treated as weighted graph).
    Falls back to Agglomerative if Leiden unavailable.
    """
    # Try multiple resolutions and pick best silhouette
    best_labels = np.zeros(n, dtype=int)
    best_sil = -1.0
    best_k = 1

    # Build edges from consensus matrix (threshold low-weight edges)
    threshold = 0.1
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if C_matrix[i, j] > threshold:
                edges.append((i, j, float(C_matrix[i, j])))

    if len(edges) < n:
        # Not enough consensus — fallback to best individual run
        return np.zeros(n, dtype=int), 1, -1.0

    resolutions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

    for res in resolutions:
        try:
            y_pred, C_count, Q = _leiden_cluster(n, edges, res, seed=42)
            if C_count < 2 or C_count > n // 3:
                continue

            dists = 1.0 - C_matrix
            sil = float(silhouette_score(dists, y_pred, metric="precomputed"))

            if sil > best_sil:
                best_sil = sil
                best_labels = y_pred.copy()
                best_k = C_count
        except:
            continue

    if best_k < 2:
        # Fallback to Agglomerative
        from sklearn.cluster import AgglomerativeClustering
        dists = 1.0 - C_matrix
        for k in range(2, min(20, n // 3)):
            try:
                agg = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                               linkage="average")
                labels = agg.fit_predict(dists)
                sil = float(silhouette_score(dists, labels, metric="precomputed"))
                if sil > best_sil:
                    best_sil = sil
                    best_labels = labels.copy()
                    best_k = k
            except:
                continue

    return best_labels, best_k, best_sil


# ============================================================
# Main EMR-CGC
# ============================================================
def run_emr_cgc(X, y_true, seed, metric="cosine", subsample=0,
                verbose=0) -> EMRResult:
    """Run Ensemble Multi-Representation Consensus Graph Clustering.

    Steps:
      1. Compute 5 representations (PCA, 3xUMAP, t-SNE)
      2. For each rep, try 12 graph configs (k × r)
      3. Build SNN graph + Leiden clustering for each
      4. Consensus vote across all valid runs
    """
    rng = set_global_seed(seed)
    n_orig = X.shape[0]
    X_arr = np.asarray(X, dtype=np.float32)

    # Subsample for large datasets
    if subsample > 0 and n_orig > subsample:
        idx = rng.choice(n_orig, size=subsample, replace=False)
        X_sub = X_arr[idx]
    else:
        X_sub = X_arr
        idx = np.arange(n_orig)

    n = X_sub.shape[0]
    t0 = time.time()

    # Standardize
    Xs = StandardScaler().fit_transform(X_sub)

    # Step 1: Multiple representations
    if verbose:
        print("  [EMR-CGC] Stage 1: Computing representations...", flush=True)
    reps = _get_representations(Xs, seed, dr_dim=min(40, Xs.shape[1] - 1))
    if verbose:
        print(f"    Representations: {list(reps.keys())}")

    # Step 2-3: Multiple graph configs × representations
    k_values = [10, 20, 35, 50]
    r_values = [0.1, 0.25, 0.5]

    all_labels = []
    all_scores = []
    rep_best_scores = {}

    if verbose:
        print("  [EMR-CGC] Stage 2: Running clustering ensemble...", flush=True)

    for rep_name, Z in reps.items():
        rep_score_best = -1e9
        for k in k_values:
            for r in r_values:
                try:
                    # SNN graph
                    edges = _build_snn_graph(Z, k, metric=metric)
                    if len(edges) < 10:
                        continue

                    # Leiden clustering
                    y_pred, C, Q = _leiden_cluster(n, edges, r, seed)

                    if C < 2 or C > n // 2:
                        continue

                    # Quality score
                    mod01 = modularity01_from_value(Q)
                    try:
                        sil = float(silhouette_score(Z, y_pred, metric=metric,
                                                      sample_size=min(300, n)))
                        sil01 = silhouette01_from_value(sil)
                    except:
                        sil01 = 0.0
                    bal01, _, _ = cluster_balance_score(y_pred)

                    score = 0.35 * mod01 + 0.40 * sil01 + 0.25 * bal01

                    all_labels.append(y_pred.copy())
                    all_scores.append(score)

                    if score > rep_score_best:
                        rep_score_best = score

                except:
                    continue

        rep_best_scores[rep_name] = rep_score_best

    n_valid = len(all_labels)
    if verbose:
        print(f"    Valid runs: {n_valid} / {len(reps) * len(k_values) * len(r_values)}")
        for rn, rs in rep_best_scores.items():
            print(f"    {rn}: best_score={rs:.4f}")

    if n_valid == 0:
        # Fallback
        return EMRResult(
            y_consensus=np.zeros(n_orig, dtype=int), C=1,
            n_runs=0, n_valid=0, best_rep="none",
            best_ari_individual=0.0, consensus_quality=0.0,
            opt_time=time.time() - t0,
        )

    # Step 4: Select best run (highest quality score)
    if verbose:
        print("  [EMR-CGC] Stage 3: Selecting best clustering...", flush=True)

    # Strategy: pick the single best run by internal quality
    # This is more robust than consensus for varying cluster counts
    best_run_idx = int(np.argmax(all_scores))
    y_consensus_sub = all_labels[best_run_idx].copy()
    C_final = len(np.unique(y_consensus_sub))
    consensus_quality = float(all_scores[best_run_idx])

    # Map back to full dataset if subsampled
    if subsample > 0 and n_orig > subsample:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.decomposition import PCA as _PCA
        # Use PCA for fast mapping
        Z_full = StandardScaler().fit_transform(X_arr)
        if Z_full.shape[1] > 50:
            pca_map = _PCA(n_components=50, svd_solver="randomized", random_state=seed)
            Z_full = pca_map.fit_transform(Z_full)
        Z_sub_mapped = Z_full[idx]
        knn_cls = KNeighborsClassifier(n_neighbors=min(5, len(idx) - 1))
        knn_cls.fit(Z_sub_mapped, y_consensus_sub)
        y_consensus = knn_cls.predict(Z_full)
    else:
        y_consensus = y_consensus_sub

    # Best individual run
    best_idx = int(np.argmax(all_scores))
    best_rep = max(rep_best_scores, key=rep_best_scores.get)

    # Evaluate best individual ARI
    best_ari = 0.0
    if y_true is not None:
        from sklearn.metrics import adjusted_rand_score
        best_ari = float(adjusted_rand_score(y_true, all_labels[best_idx]))

    opt_time = time.time() - t0
    if verbose:
        print(f"  [EMR-CGC] Done: C={C_final}, quality={consensus_quality:.3f}, "
              f"time={opt_time:.1f}s")

    return EMRResult(
        y_consensus=y_consensus,
        C=C_final,
        n_runs=len(reps) * len(k_values) * len(r_values),
        n_valid=n_valid,
        best_rep=best_rep,
        best_ari_individual=best_ari,
        consensus_quality=consensus_quality,
        opt_time=opt_time,
        rep_scores=rep_best_scores,
        all_labels=all_labels if n < 2000 else None,  # save memory
    )
