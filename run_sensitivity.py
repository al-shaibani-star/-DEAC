# -*- coding: utf-8 -*-
"""
R2.2 data capture: for each (dataset, seed) record each engine's internal-score
COMPONENTS (MOD01, SIL01, BAL01), cluster count C, and ARI -- everything needed to
re-derive the QualityGate's pick under ANY (floor, c, weight) setting offline, so
we can run a sensitivity analysis and a dev/eval split without labels leaking.

Single-unit mode: `python run_sensitivity.py <name> <seed>` (used by the driver,
one fresh subprocess per unit -- same isolation fix as the canonical run).
Writes results_sensitivity/<name>_s<seed>.json.
"""
import json, os, sys, time, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.io.data_loading import load_csv
from src.io.representation import make_representation
from src.graph.construction import build_knn_graph, keep_graph_connected_by_bridging
from src.graph.clustering import cluster_graph
from src.core.fast_amr_de import run_fast_amr_de
from src.core.emr_cgc import run_emr_cgc
from src.io.utils import (cluster_balance_score, modularity01_from_value,
                          silhouette01_from_value)
from sklearn.metrics import adjusted_rand_score, silhouette_score
import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_mod

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_sensitivity")
DATASETS = dict([
    ("Wine", "Wine.csv"), ("Glass", "Glass.csv"), ("Olivetti", "olivetti_full.csv"),
    ("CNAE-9", "CNAE9.csv"), ("COIL-20", "COIL20.csv"), ("Optdigits", "Optdigits.csv"),
    ("MFeat", "mfeat_full.csv"), ("Segmentation", "Segmentation.csv"),
    ("ISOLET", "isolet_full.csv"), ("Satimage", "Satimage.csv"),
    ("USPS", "USPS.csv"), ("PenDigits", "PenDigits.csv"),
])
CAP, METRIC = 100000, "cosine"
ENG = ["Baseline", "AMR-DE", "EMR-CGC"]


def cluster_on_rep(Z, k, q, r, seed):
    n, e = build_knn_graph(Z, k=int(k), metric=METRIC, graph_mode="symmetric", prune_q=float(q), min_deg_guard=3)
    e = keep_graph_connected_by_bridging(Z, e, metric=METRIC)
    y, C, degen, m, Q, _ = cluster_graph(n=n, edges=e, seed=seed, resolution=float(r))
    return np.asarray(y), e


def components(labels, Z_ref, G_ref, seed):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {"mod01": 0.0, "sil01": 0.0, "bal01": 0.0}
    comms = [set(np.where(labels == l)[0]) for l in uniq]
    try:
        Q = float(nx_mod(G_ref, comms, weight="weight"))
    except Exception:
        Q = 0.0
    try:
        sil = float(silhouette_score(Z_ref, labels, metric=METRIC, sample_size=min(1000, len(labels)), random_state=seed))
    except Exception:
        sil = 0.0
    bal01, _, _ = cluster_balance_score(labels)
    return {"mod01": modularity01_from_value(Q), "sil01": silhouette01_from_value(sil), "bal01": float(bal01)}


def run_one(name, csv, seed):
    X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
    if X.shape[0] > CAP:
        from sklearn.model_selection import train_test_split
        _, X, _, y = train_test_split(X, y, test_size=CAP, stratify=y, random_state=seed)
    n = X.shape[0]; dr = max(2, min(40, X.shape[1] - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Zb = make_representation(X, seed=seed, dr_dim=dr, rep_type="umap")
        yB, edgesB = cluster_on_rep(Zb, 35, 0.1, 0.25, seed)
        amr = run_fast_amr_de(X, y, seed=seed, metric=METRIC, subsample=min(800, n),
                              de_maxiter=8, de_popsize=5, n_restarts=2, max_dr_dim=X.shape[1] - 1, max_k=n - 1)
        Zde = make_representation(X, seed=seed, dr_dim=min(dr, amr.best_d), rep_type=amr.best_rep,
                                  n_neighbors=amr.best_n_neighbors, min_dist=amr.best_min_dist, perplexity=amr.best_perplexity)
        yDE, _ = cluster_on_rep(Zde, amr.best_k, amr.best_q, amr.best_r, seed)
        yEMR = np.asarray(run_emr_cgc(X, y, seed=seed, metric=METRIC, subsample=0).y_consensus)
        G_ref = nx.Graph(); G_ref.add_nodes_from(range(n))
        for a, b, wt in edgesB:
            if a != b:
                G_ref.add_edge(int(a), int(b), weight=float(wt))
        cand = {"Baseline": yB, "AMR-DE": yDE, "EMR-CGC": yEMR}
        comp = {e: components(cand[e], Zb, G_ref, seed) for e in ENG}
        ari = {e: float(adjusted_rand_score(y, cand[e])) for e in ENG}
        C = {e: int(len(np.unique(cand[e]))) for e in ENG}
    return {"name": name, "seed": int(seed), "n": int(n), "C_true": int(len(np.unique(y))),
            "components": comp, "ari": ari, "C": C}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    a = sys.argv[1:]
    if len(a) == 2 and a[1].isdigit():
        name, seed = a[0], int(a[1])
        ck = os.path.join(OUT_DIR, f"{name}_s{seed}.json")
        if os.path.exists(ck):
            print(f"[cache] {name} s{seed}", flush=True); return
        ts = time.time(); res = run_one(name, DATASETS[name], seed)
        json.dump(res, open(ck, "w"), indent=2)
        print(f"[ok] {name} s{seed} ({time.time()-ts:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
