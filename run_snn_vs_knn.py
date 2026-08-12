# -*- coding: utf-8 -*-
"""
Ablation for the Discussion claim: how much does swapping KNN+Louvain for
SNN+Leiden raise mean ARI? Matched settings (same UMAP representation, same
k and resolution); only the graph type and community detector change.

Prints the real mean-ARI difference over the 12 paper datasets.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.io.data_loading import load_csv
from src.io.representation import make_representation
from src.core.emr_cgc import _build_snn_graph, _leiden_cluster
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
DATASETS = [
    ("Wine", "Wine.csv"), ("Glass", "Glass.csv"), ("Olivetti", "olivetti_full.csv"),
    ("CNAE-9", "CNAE9.csv"), ("COIL-20", "COIL20.csv"), ("Optdigits", "Optdigits.csv"),
    ("MFeat", "mfeat_full.csv"), ("Segmentation", "Segmentation.csv"),
    ("ISOLET", "isolet_full.csv"), ("Satimage", "Satimage.csv"),
    ("USPS", "USPS.csv"), ("PenDigits", "PenDigits.csv"),
]
SEEDS = [1001, 1002, 1003]
K, R, METRIC = 35, 0.25, "cosine"


def knn_louvain(Z, k, r, seed):
    """KNN graph (weight=(1+cos)/2) + Louvain."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
    n = Z.shape[0]
    keff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=keff + 1, metric=METRIC, n_jobs=1).fit(Z)
    dist, idx = nn.kneighbors(Z)
    dist, idx = dist[:, 1:], idx[:, 1:]           # cosine distance = 1 - cos_sim
    G = nx.Graph(); G.add_nodes_from(range(n))
    for i in range(n):
        for t in range(keff):
            j = int(idx[i, t])
            w = 1.0 - dist[i, t] / 2.0            # (1 + cos)/2 = 1 - dist/2
            a, b = (i, j) if i < j else (j, i)
            if a != b:
                G.add_edge(a, b, weight=float(w))
    if G.number_of_edges() == 0:
        return np.zeros(n, dtype=int)
    comms = louvain_communities(G, weight="weight", resolution=float(r), seed=int(seed))
    y = np.empty(n, dtype=int)
    for cid, nodes in enumerate(comms):
        for v in nodes:
            y[int(v)] = cid
    return y


def snn_leiden(Z, k, r, seed):
    edges = _build_snn_graph(Z, k, metric=METRIC)
    y, C, Q = _leiden_cluster(Z.shape[0], edges, r, seed)
    return y


def main():
    rows = []
    for name, csv in DATASETS:
        X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
        dr = max(2, min(40, X.shape[1] - 1))
        kl, sl = [], []
        for s in SEEDS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Z = make_representation(X, seed=s, dr_dim=dr, rep_type="umap")
                y_kl = knn_louvain(Z, K, R, s)
                y_sl = snn_leiden(Z, K, R, s)
            kl.append(adjusted_rand_score(y, y_kl))
            sl.append(adjusted_rand_score(y, y_sl))
        a_kl, a_sl = float(np.mean(kl)), float(np.mean(sl))
        rows.append((name, a_kl, a_sl))
        print(f"{name:<13} KNN+Louvain={a_kl:.3f}  SNN+Leiden={a_sl:.3f}  delta={a_sl-a_kl:+.3f}")

    m_kl = np.mean([r[1] for r in rows])
    m_sl = np.mean([r[2] for r in rows])
    print("-" * 60)
    print(f"{'MEAN':<13} KNN+Louvain={m_kl:.3f}  SNN+Leiden={m_sl:.3f}  "
          f"delta={m_sl-m_kl:+.3f} = {100*(m_sl-m_kl):+.1f} ARI points")


if __name__ == "__main__":
    main()
