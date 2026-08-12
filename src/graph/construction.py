# -*- coding: utf-8 -*-
"""Graph construction: KNN graph building with sklearn, edge pruning, bridging."""
from typing import Dict, List, Tuple

import numpy as np


def _dist_to_weight(d: float, metric: str) -> float:
    """Convert distance to edge weight."""
    if metric == "cosine":
        cos_sim = 1.0 - float(d)
        return float((1.0 + cos_sim) / 2.0)
    return float(np.exp(-float(d)))


def _vectorized_dist_to_weight(dists: np.ndarray, metric: str) -> np.ndarray:
    """Vectorized distance-to-weight conversion."""
    if metric == "cosine":
        cos_sim = 1.0 - dists
        return (1.0 + cos_sim) / 2.0
    return np.exp(-dists)


def _build_edges_from_knn(
    n: int, k: int, dists: np.ndarray, idxs: np.ndarray,
    metric: str, graph_mode: str, prune_q: float, min_deg_guard: int,
) -> List[Tuple[int, int, float]]:
    """Build edge list from KNN distances and indices."""
    weights = _vectorized_dist_to_weight(dists.ravel(), metric).reshape(dists.shape)

    sources = np.repeat(np.arange(n), k)
    targets = idxs.ravel()
    w_vals = weights.ravel()

    dir_w: Dict[Tuple[int, int], float] = {}
    for idx in range(len(sources)):
        dir_w[(int(sources[idx]), int(targets[idx]))] = float(w_vals[idx])

    if not dir_w:
        return []

    edges_map: Dict[Tuple[int, int], float] = {}
    if graph_mode == "mutual":
        for (i, j), w_ij in dir_w.items():
            w_ji = dir_w.get((j, i), None)
            if w_ji is None:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges_map[(a, b)] = 0.5 * (w_ij + w_ji)
    else:
        for (i, j), w_ij in dir_w.items():
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in edges_map:
                edges_map[(a, b)] = 0.5 * (edges_map[(a, b)] + w_ij)
            else:
                edges_map[(a, b)] = w_ij

    if not edges_map:
        return []

    # Edge pruning
    if prune_q > 0.0:
        wvals = np.asarray(list(edges_map.values()), dtype=np.float32)
        thr = float(np.quantile(wvals, prune_q))
        deg = np.zeros(n, dtype=np.int32)
        for (a, b), _w in edges_map.items():
            deg[a] += 1
            deg[b] += 1
        pruned: Dict[Tuple[int, int], float] = {}
        for (a, b), w in edges_map.items():
            keep = (w >= thr)
            if not keep:
                if deg[a] <= min_deg_guard or deg[b] <= min_deg_guard:
                    keep = True
            if keep:
                pruned[(a, b)] = w
        edges_map = pruned

    return [(a, b, float(w)) for (a, b), w in edges_map.items()]


def build_knn_graph(
    Z: np.ndarray, k: int, metric: str, graph_mode: str,
    prune_q: float, min_deg_guard: int, n_jobs: int = 1,
) -> Tuple[int, List[Tuple[int, int, float]]]:
    """Build KNN graph from feature matrix Z."""
    from sklearn.neighbors import NearestNeighbors

    n = Z.shape[0]
    if n < 3:
        return n, []
    k = int(np.clip(k, 5, min(140, n - 1)))
    prune_q = float(np.clip(prune_q, 0.0, 0.40))
    min_deg_guard = int(np.clip(min_deg_guard, 0, max(0, k)))

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=int(n_jobs))
    nn.fit(Z)
    dists, idxs = nn.kneighbors(Z, return_distance=True)
    dists, idxs = dists[:, 1:], idxs[:, 1:]

    edges = _build_edges_from_knn(n, k, dists, idxs, metric, graph_mode, prune_q, min_deg_guard)
    return n, edges


def components_stats_from_edges(n: int, edges: List[Tuple[int, int, float]]) -> Tuple[int, int, float, int]:
    """Compute connected component statistics."""
    try:
        import networkx as nx
    except ImportError:
        return 1, n, float(n), n
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, w in edges:
        if a != b:
            G.add_edge(int(a), int(b), weight=float(w))
    comps = [len(c) for c in nx.connected_components(G)]
    if not comps:
        return n, 1, 1.0, 1
    comps_sorted = sorted(comps)
    return int(len(comps_sorted)), int(comps_sorted[0]), float(np.median(comps_sorted)), int(comps_sorted[-1])


def keep_graph_connected_by_bridging(
    Z: np.ndarray, edges: List[Tuple[int, int, float]],
    metric: str, max_bridges: int = 4000,
) -> List[Tuple[int, int, float]]:
    """Add bridge edges to connect disconnected components."""
    try:
        import networkx as nx
    except ImportError:
        return edges
    n = Z.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, w in edges:
        if a != b:
            G.add_edge(int(a), int(b), weight=float(w))
    comps = list(nx.connected_components(G))
    if len(comps) <= 1:
        return edges
    reps = [int(next(iter(c))) for c in comps]
    if len(reps) <= 1:
        return edges

    bridges: List[Tuple[int, int, float]] = []
    connected = [reps[0]]
    remaining = set(reps[1:])

    while remaining and len(bridges) < max_bridges:
        best_pair = None
        best_dist = None
        for u in connected:
            zu = Z[u]
            nu = float(np.linalg.norm(zu) + 1e-12)
            for v in list(remaining):
                zv = Z[v]
                if metric == "cosine":
                    nv = float(np.linalg.norm(zv) + 1e-12)
                    du = float(1.0 - float(np.dot(zu, zv)) / (nu * nv))
                else:
                    du = float(np.linalg.norm(zu - zv))
                if best_dist is None or du < best_dist:
                    best_dist = du
                    best_pair = (u, v)
        if best_pair is None or best_dist is None:
            break
        u, v = best_pair
        w = _dist_to_weight(best_dist, metric=metric)
        a, b = (u, v) if u < v else (v, u)
        bridges.append((a, b, float(w)))
        connected.append(v)
        remaining.discard(v)

    seen = {(min(a, b), max(a, b)) for a, b, _ in edges}
    out = list(edges)
    for a, b, w in bridges:
        key = (min(a, b), max(a, b))
        if key not in seen and a != b:
            seen.add(key)
            out.append((a, b, float(w)))
    return out
