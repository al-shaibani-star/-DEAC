# -*- coding: utf-8 -*-
"""Graph clustering: Louvain community detection."""
from typing import List, Tuple

import numpy as np


def cluster_graph(n: int, edges: List[Tuple[int, int, float]], seed: int,
                  resolution: float) -> Tuple[np.ndarray, int, bool, int, float, list]:
    """Apply Louvain community detection to the graph.

    Returns
    -------
    y_pred : np.ndarray
        Cluster assignments.
    C : int
        Number of clusters.
    degenerate : bool
        True if clustering is degenerate.
    m : int
        Number of edges.
    Q : float
        Modularity score.
    communities : list
        List of community sets.
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    from networkx.algorithms.community.quality import modularity as nx_modularity

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, j, w in edges:
        if i != j:
            G.add_edge(int(i), int(j), weight=float(w))
    m = G.number_of_edges()
    if m == 0:
        return np.arange(n, dtype=int), n, True, 0, float("nan"), [{i} for i in range(n)]

    communities = louvain_communities(G, weight="weight", resolution=float(resolution), seed=int(seed))

    y_pred = np.empty(n, dtype=int)
    for cid, nodes in enumerate(communities):
        for v in nodes:
            y_pred[int(v)] = int(cid)

    C = int(np.unique(y_pred).size)
    degenerate = (C < 2) or (C > max(200, n // 2))
    Q = float("nan")
    try:
        Q = float(nx_modularity(G, communities, weight="weight"))
    except Exception:
        pass
    return y_pred, C, degenerate, m, Q, communities
