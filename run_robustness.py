# -*- coding: utf-8 -*-
"""
R2.5(c): robustness of the DEAC gate to controlled stressors, on synthetic data
(sklearn make_classification) where ground truth is known. Four axes, varied one
at a time around a base (n=2000, d=20 all-informative, C=10, class_sep=1.0,
balanced): class imbalance, cluster separation (density), irrelevant dimensions,
and sample size n. For each condition we run the three engines + the deployable
gate and record ARI/C.

Single-unit mode: `python run_robustness.py <cond_idx> <seed>` (subprocess-isolated).
Writes results_robustness/c<idx>_s<seed>.json. List conditions: `... list`.
"""
import json, os, sys, time, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.datasets import make_classification
from src.io.representation import make_representation
from src.graph.construction import build_knn_graph, keep_graph_connected_by_bridging
from src.graph.clustering import cluster_graph
from src.core.fast_amr_de import run_fast_amr_de
from src.core.emr_cgc import run_emr_cgc
from src.io.utils import cluster_balance_score, modularity01_from_value, silhouette01_from_value
from sklearn.metrics import adjusted_rand_score, silhouette_score
import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_mod

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_robustness")
METRIC = "cosine"
ENG = ["Baseline", "AMR-DE", "EMR-CGC"]
BASE = dict(n=2000, d=20, C=10, sep=1.0, weights=None)


def imbalance_weights(C, ratio):
    # geometric weights with max/min = ratio, normalised
    w = np.geomspace(1.0, ratio, C); return list(w / w.sum())


def conditions():
    L = []
    L.append(("base", dict(BASE)))
    for r in [3, 10, 30]:
        c = dict(BASE); c["weights"] = ("imb", r); L.append((f"imbalance x{r}", c))
    for s in [0.5, 0.75, 1.5, 2.0]:
        c = dict(BASE); c["sep"] = s; L.append((f"separation {s}", c))
    for d in [70, 220, 520]:
        c = dict(BASE); c["d"] = d; L.append((f"irrelevant d={d}", c))
    for n in [1000, 5000, 10000]:
        c = dict(BASE); c["n"] = n; L.append((f"n={n}", c))
    return L


def gen(cfg, seed):
    w = cfg["weights"]
    weights = imbalance_weights(cfg["C"], w[1]) if isinstance(w, tuple) else None
    X, y = make_classification(n_samples=cfg["n"], n_features=cfg["d"], n_informative=20,
                               n_redundant=0, n_repeated=0, n_classes=cfg["C"],
                               n_clusters_per_class=1, class_sep=cfg["sep"], weights=weights,
                               flip_y=0.0, random_state=seed)
    return X.astype(np.float32), y


def cluster_on_rep(Z, k, q, r, seed):
    n, e = build_knn_graph(Z, k=int(k), metric=METRIC, graph_mode="symmetric", prune_q=float(q), min_deg_guard=3)
    e = keep_graph_connected_by_bridging(Z, e, metric=METRIC)
    y, C, degen, m, Q, _ = cluster_graph(n=n, edges=e, seed=seed, resolution=float(r))
    return np.asarray(y), e


def internal(labels, Zref, Gref, seed):
    u = np.unique(labels)
    if len(u) < 2:
        return -1.0
    comms = [set(np.where(labels == l)[0]) for l in u]
    try: Q = float(nx_mod(Gref, comms, weight="weight"))
    except Exception: Q = 0.0
    try: sil = float(silhouette_score(Zref, labels, metric=METRIC, sample_size=min(1000, len(labels)), random_state=seed))
    except Exception: sil = 0.0
    b, _, _ = cluster_balance_score(labels)
    return 0.40 * modularity01_from_value(Q) + 0.35 * silhouette01_from_value(sil) + 0.25 * b


def run_one(idx, seed):
    label, cfg = conditions()[idx]
    X, y = gen(cfg, seed); n = X.shape[0]; dr = max(2, min(40, X.shape[1] - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Zb = make_representation(X, seed=seed, dr_dim=dr, rep_type="umap")
        yB, eB = cluster_on_rep(Zb, 35, 0.1, 0.25, seed)
        amr = run_fast_amr_de(X, y, seed=seed, metric=METRIC, subsample=min(800, n),
                              de_maxiter=8, de_popsize=5, n_restarts=2, max_dr_dim=X.shape[1] - 1, max_k=n - 1)
        Zde = make_representation(X, seed=seed, dr_dim=min(dr, amr.best_d), rep_type=amr.best_rep,
                                  n_neighbors=amr.best_n_neighbors, min_dist=amr.best_min_dist, perplexity=amr.best_perplexity)
        yDE, _ = cluster_on_rep(Zde, amr.best_k, amr.best_q, amr.best_r, seed)
        yEMR = np.asarray(run_emr_cgc(X, y, seed=seed, metric=METRIC, subsample=0).y_consensus)
        G = nx.Graph(); G.add_nodes_from(range(n))
        for a, b, w in eB:
            if a != b: G.add_edge(int(a), int(b), weight=float(w))
        cand = {"Baseline": yB, "AMR-DE": yDE, "EMR-CGC": yEMR}
        ari = {e: float(adjusted_rand_score(y, cand[e])) for e in ENG}
        C = {e: int(len(np.unique(cand[e]))) for e in ENG}
        sc = {e: internal(cand[e], Zb, G, seed) for e in ENG}
    rho = max(4.0, np.sqrt(n / 50.0)); CB = C["Baseline"]
    elig = [e for e in ENG if e == "Baseline" or C[e] < rho * CB]
    gate = max(elig, key=lambda e: sc[e])
    return {"idx": idx, "label": label, "seed": int(seed), "n": int(n), "d": int(X.shape[1]),
            "C_true": int(len(np.unique(y))), "ari": ari, "C": C,
            "gate_pick": gate, "gate_ari": ari[gate], "oracle_ari": max(ari.values())}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    a = sys.argv[1:]
    if a and a[0] == "list":
        for i, (lab, _) in enumerate(conditions()): print(i, lab)
        return
    if len(a) == 2 and a[0].isdigit() and a[1].isdigit():
        idx, seed = int(a[0]), int(a[1])
        ck = os.path.join(OUT_DIR, f"c{idx}_s{seed}.json")
        if os.path.exists(ck): print(f"[cache] c{idx} s{seed}", flush=True); return
        ts = time.time(); res = run_one(idx, seed); json.dump(res, open(ck, "w"), indent=2)
        print(f"[ok] c{idx}({res['label']}) s{seed} gate={res['gate_ari']:.3f} "
              f"base={res['ari']['Baseline']:.3f} ({time.time()-ts:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
