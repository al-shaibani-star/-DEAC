# -*- coding: utf-8 -*-
"""
Canonical results generator for the DEAC manuscript (IJIES resubmission).

Runs the CURRENT three-engine pipeline (Baseline UMAP+Louvain, AMR-DE, EMR-CGC)
on the 12 benchmark datasets over 5 fixed seeds (1001-1005) on FULL data, and
records -- per (dataset, seed, engine) -- ARI, NMI, ACC, cluster count C, and the
label-free internal quality score, plus the deployable gate pick and the oracle
pick. Everything the manuscript reports (main table, multi-metric, cluster-count,
ablation, gate-vs-oracle) is derivable from these files, so figures and tables
can be regenerated from ONE source.

Checkpointed per (dataset, seed) to results_canonical/<name>_s<seed>.json; re-runs
skip completed files. Mirrors run_gate_vs_oracle.py exactly for the engine calls,
gate scoring (0.40 MOD + 0.35 SIL + 0.25 BAL) and guard rho = max(4, sqrt(n/50)).
"""
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.io.data_loading import load_csv
from src.io.representation import make_representation
from src.graph.construction import build_knn_graph, keep_graph_connected_by_bridging
from src.graph.clustering import cluster_graph
from src.core.fast_amr_de import run_fast_amr_de
from src.core.emr_cgc import run_emr_cgc
from src.io.utils import (cluster_balance_score, modularity01_from_value,
                          silhouette01_from_value, best_map_acc)
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_mod

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_canonical")
DATASETS = [
    ("Wine", "Wine.csv"), ("Glass", "Glass.csv"), ("Olivetti", "olivetti_full.csv"),
    ("CNAE-9", "CNAE9.csv"), ("COIL-20", "COIL20.csv"), ("Optdigits", "Optdigits.csv"),
    ("MFeat", "mfeat_full.csv"), ("Segmentation", "Segmentation.csv"),
    ("ISOLET", "isolet_full.csv"), ("Satimage", "Satimage.csv"),
    ("USPS", "USPS.csv"), ("PenDigits", "PenDigits.csv"),
]
SEEDS = [1001, 1002, 1003, 1004, 1005]
CAP = 100000
METRIC = "cosine"
ENGINES = ["Baseline", "AMR-DE", "EMR-CGC"]


def cluster_on_rep(Z, k, q, r, seed):
    n, edges = build_knn_graph(Z, k=int(k), metric=METRIC, graph_mode="symmetric",
                               prune_q=float(q), min_deg_guard=3)
    edges = keep_graph_connected_by_bridging(Z, edges, metric=METRIC)
    y, C, degen, m, Q, _ = cluster_graph(n=n, edges=edges, seed=seed, resolution=float(r))
    return np.asarray(y), edges


def internal_quality(labels, Z_ref, G_ref, seed):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return -1.0
    comms = [set(np.where(labels == l)[0]) for l in uniq]
    try:
        Q = float(nx_mod(G_ref, comms, weight="weight"))
    except Exception:
        Q = 0.0
    mod01 = modularity01_from_value(Q)
    try:
        sil = float(silhouette_score(Z_ref, labels, metric=METRIC,
                                     sample_size=min(1000, len(labels)), random_state=seed))
    except Exception:
        sil = 0.0
    sil01 = silhouette01_from_value(sil)
    bal01, _, _ = cluster_balance_score(labels)
    return 0.40 * mod01 + 0.35 * sil01 + 0.25 * bal01


def run_one(name, csv, seed):
    X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
    if X.shape[0] > CAP:
        from sklearn.model_selection import train_test_split
        _, X, _, y = train_test_split(X, y, test_size=CAP, stratify=y, random_state=seed)
    n = X.shape[0]
    dr = max(2, min(40, X.shape[1] - 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # --- Baseline: UMAP + Louvain (k=35, q=0.1, r=0.25) ---
        Zb = make_representation(X, seed=seed, dr_dim=dr, rep_type="umap")
        yB, edgesB = cluster_on_rep(Zb, 35, 0.1, 0.25, seed)

        # --- AMR-DE ---
        amr = run_fast_amr_de(X, y, seed=seed, metric=METRIC, subsample=min(800, n),
                              de_maxiter=8, de_popsize=5, n_restarts=2,
                              max_dr_dim=X.shape[1] - 1, max_k=n - 1)
        Zde = make_representation(X, seed=seed, dr_dim=min(dr, amr.best_d),
                                  rep_type=amr.best_rep,
                                  n_neighbors=amr.best_n_neighbors,
                                  min_dist=amr.best_min_dist, perplexity=amr.best_perplexity)
        yDE, _ = cluster_on_rep(Zde, amr.best_k, amr.best_q, amr.best_r, seed)

        # --- EMR-CGC ---
        emr = run_emr_cgc(X, y, seed=seed, metric=METRIC, subsample=0)
        yEMR = np.asarray(emr.y_consensus)

        # --- common reference graph on baseline rep (for modularity) ---
        G_ref = nx.Graph(); G_ref.add_nodes_from(range(n))
        for a, b, w in edgesB:
            if a != b:
                G_ref.add_edge(int(a), int(b), weight=float(w))

        cand = {"Baseline": yB, "AMR-DE": yDE, "EMR-CGC": yEMR}
        ari = {k: float(adjusted_rand_score(y, v)) for k, v in cand.items()}
        nmi = {k: float(normalized_mutual_info_score(y, v)) for k, v in cand.items()}
        acc = {k: float(best_map_acc(y, v)) for k, v in cand.items()}
        C = {k: int(len(np.unique(v))) for k, v in cand.items()}
        scores = {k: internal_quality(v, Zb, G_ref, seed) for k, v in cand.items()}

    # over-fragmentation guard (baseline always eligible)
    rho = max(4.0, np.sqrt(n / 50.0))
    CB = C["Baseline"]
    eligible = [k for k in cand if k == "Baseline" or C[k] < rho * CB]
    gate_pick = max(eligible, key=lambda k: scores[k])
    oracle_pick = max(cand, key=lambda k: ari[k])

    C_true = int(len(np.unique(y)))
    res = {
        "name": name, "seed": int(seed), "n": int(n), "d": int(X.shape[1]),
        "C_true": C_true, "rho": float(rho),
        "ari": ari, "nmi": nmi, "acc": acc, "C": C, "scores": scores,
        "gate_pick": gate_pick, "oracle_pick": oracle_pick,
        "amr_params": {"rep": amr.best_rep, "k": int(amr.best_k), "d": int(amr.best_d),
                       "q": float(amr.best_q), "r": float(amr.best_r)},
    }
    return res


def run_unit(name, csv, seed):
    ck = os.path.join(OUT_DIR, f"{name}_s{seed}.json")
    if os.path.exists(ck):
        print(f"[cache] {name} s{seed}", flush=True)
        return
    ts = time.time()
    res = run_one(name, csv, seed)
    with open(ck, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[ok]  {name:<13} s{seed}  "
          f"B={res['ari']['Baseline']:.3f} DE={res['ari']['AMR-DE']:.3f} "
          f"EMR={res['ari']['EMR-CGC']:.3f}  gate={res['gate_pick']:<8} "
          f"oracle={res['oracle_pick']:<8}  ({time.time()-ts:.0f}s)", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    args = sys.argv[1:]
    dmap = dict(DATASETS)
    # Single-unit mode (used by the subprocess driver): "<name> <seed>".
    # Runs ONE (dataset, seed) in this fresh interpreter, then exits -- this
    # isolation is the fix for EMR-CGC collapsing after long sequential runs.
    if len(args) == 2 and args[1].isdigit():
        name, seed = args[0], int(args[1])
        run_unit(name, dmap[name], seed)
        return
    # Legacy in-process mode (kept for debugging; NOT used for the canonical run
    # because accumulated state across many units degrades EMR-CGC).
    wanted = args if args else [d[0] for d in DATASETS]
    t0 = time.time()
    for name, csv in DATASETS:
        if name not in wanted:
            continue
        for seed in SEEDS:
            try:
                run_unit(name, csv, seed)
            except Exception as e:
                import traceback
                print(f"[ERR] {name} s{seed}: {e}")
                traceback.print_exc()
    print(f"\nDONE in {time.time()-t0:.0f}s. Checkpoints in {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
