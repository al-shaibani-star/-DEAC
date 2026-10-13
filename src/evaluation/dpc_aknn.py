# -*- coding: utf-8 -*-
"""
Faithful reimplementation of DPC-AKNN (DPCAKNN): "Density peak clustering
algorithm based on adaptive K-nearest neighbors for manifold data", Yane Wang &
Mary Jane Samonte, Discover Applied Sciences 8:165 (2026),
doi:10.1007/s42452-025-08068-w (Section 3).

Reading of the paper:
  * Features min-max normalised to [0,1] (Step 1).
  * Adaptive K is parameter-free: we use the Natural-Neighbor (NaN) eigenvalue --
    grow the neighbor order r until the count of points with no reverse r-NN
    stabilises; that r is the dataset's adaptive k (the paper's "adaptive K
    according to the natural distribution").
  * Def 1: KNN(x_i) = the k nearest neighbours of x_i (local, size k), by ED.
  * Def 4 (Eqs 9-10): en_rho_i = mean over j in KNN(i) of dist(KNN(x_j))/k_j^2,
    where dist(KNN(x_j)) is the summed pairwise ED among x_j's kNN; rho_i =
    |KNN(i)| + exp(-en_rho_i)  (|KNN| = k, so density is ranked by the exp term).
  * Def 5 geodesic distance: shortest path on the symmetric kNN graph (ED weights),
    infinite if unreachable; used for the relative distance and for assignment.
  * delta_i = min geodesic distance to a higher-density point (max finite geodesic
    for the global peak).
  * Centres = top-C by gamma = rho*delta (C given -> known-C protocol, matching how
    the deep baselines are given the true cluster count).
  * Assignment (Step 6): each non-centre goes to the centre with the largest shared
    neighbours SNN (Eq 8, local kNN overlap); if it shares none, to the nearest
    centre by geodesic distance.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors


def natural_neighbor_k(X, kmax=None):
    n = X.shape[0]
    kmax = kmax or min(n - 1, max(5, int(np.sqrt(n))))
    nn = NearestNeighbors(n_neighbors=kmax + 1).fit(X)
    _, idx = nn.kneighbors(X)
    idx = idx[:, 1:]  # drop self
    rnn = np.zeros(n, dtype=int)
    prev = n
    r = 0
    for r in range(1, kmax + 1):
        np.add.at(rnn, idx[:, r - 1], 1)
        nozero = int(np.sum(rnn == 0))
        if nozero == prev:
            break
        prev = nozero
    return max(3, r)


def dpc_aknn(X, C, seed=1001, k=None):
    X = np.asarray(X, dtype=np.float64)
    mn, mx = X.min(0), X.max(0)
    Xn = (X - mn) / np.where(mx > mn, mx - mn, 1.0)
    n = Xn.shape[0]
    if k is None:
        k = natural_neighbor_k(Xn)
    k = int(min(max(3, k), n - 1))

    D = cdist(Xn, Xn, metric="euclidean")
    # Def 1: local k nearest neighbours
    knn_idx = np.argsort(D, axis=1)[:, 1:k + 1]
    knn = [set(knn_idx[i].tolist()) for i in range(n)]

    # symmetric kNN graph -> geodesic (Def 5)
    nng = NearestNeighbors(n_neighbors=k + 1).fit(Xn)
    G = nng.kneighbors_graph(mode="distance")
    G = G.maximum(G.T)
    geo = shortest_path(csr_matrix(G), method="D", directed=False)
    fm = geo[np.isfinite(geo)].max() if np.isfinite(geo).any() else 1.0
    g = np.where(np.isfinite(geo), geo, fm * 2.0)

    # Def 4: local density (Eqs 9-10)
    distknn = np.array([D[np.ix_(knn_idx[j], knn_idx[j])].sum() / 2.0 for j in range(n)])
    kj2 = float(k) ** 2
    en_rho = np.array([np.mean(distknn[knn_idx[i]] / kj2) for i in range(n)])
    rho = k + np.exp(-en_rho)

    # relative distance on geodesics
    order = np.argsort(-rho)
    delta = np.zeros(n)
    for idx_, i in enumerate(order):
        higher = order[:idx_]
        delta[i] = g[i, higher].min() if len(higher) else g[i].max()

    # centres: top-C by gamma (known-C)
    centers = list(np.argsort(-(rho * delta))[:C])
    labels = -np.ones(n, dtype=int)
    for ci, c in enumerate(centers):
        labels[c] = ci

    # assignment by SNN, else nearest geodesic centre
    for i in range(n):
        if labels[i] >= 0:
            continue
        snn = [len(knn[i] & knn[c]) for c in centers]
        labels[i] = int(np.argmax(snn)) if max(snn) > 0 else int(np.argmin([g[i, c] for c in centers]))
    return labels


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.io.data_loading import load_csv
    from sklearn.metrics import adjusted_rand_score
    X, y, _ = load_csv(sys.argv[1], label_col=-1)
    C = len(np.unique(y))
    lab = dpc_aknn(X, C=C)
    print(f"n={X.shape[0]} C={C} k={natural_neighbor_k((X-X.min(0))/np.where(X.max(0)>X.min(0),X.max(0)-X.min(0),1))} "
          f"found={len(np.unique(lab))} ARI={adjusted_rand_score(y, lab):.3f}")
