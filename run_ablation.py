# -*- coding: utf-8 -*-
"""
R2.3 (a,b,c): controlled single-variable pipeline ablation. Four configs, each a
SINGLE clustering at the same budget (k=35, r=0.25), so each pairwise difference
isolates ONE factor:
  C1 PCA  + KNN + Louvain   vs C2 UMAP + KNN + Louvain   -> representation (a)
  C2 UMAP + KNN + Louvain   vs C3 UMAP + KNN + Leiden     -> community detector (b)
  C3 UMAP + KNN + Leiden    vs C4 UMAP + SNN + Leiden      -> edge weighting (c)
Single-unit mode: `python run_ablation.py <name> <seed>` (subprocess-isolated).
Writes results_ablation/<name>_s<seed>.json with per-config ARI and C.
"""
import json, os, sys, time, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.io.data_loading import load_csv
from src.io.representation import make_representation
from src.graph.construction import build_knn_graph, keep_graph_connected_by_bridging
from src.graph.clustering import cluster_graph
from src.core.emr_cgc import _build_snn_graph, _leiden_cluster
from sklearn.metrics import adjusted_rand_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_ablation")
DATASETS = dict([
    ("Wine", "Wine.csv"), ("Glass", "Glass.csv"), ("Olivetti", "olivetti_full.csv"),
    ("CNAE-9", "CNAE9.csv"), ("COIL-20", "COIL20.csv"), ("Optdigits", "Optdigits.csv"),
    ("MFeat", "mfeat_full.csv"), ("Segmentation", "Segmentation.csv"),
    ("ISOLET", "isolet_full.csv"), ("Satimage", "Satimage.csv"),
    ("USPS", "USPS.csv"), ("PenDigits", "PenDigits.csv"),
])
CAP, METRIC, K, Q, R = 100000, "cosine", 35, 0.1, 0.25


def knn_edges(Z, seed):
    n, e = build_knn_graph(Z, k=K, metric=METRIC, graph_mode="symmetric", prune_q=Q, min_deg_guard=3)
    e = keep_graph_connected_by_bridging(Z, e, metric=METRIC)
    return n, e


def louvain(n, e, seed):
    y, C, deg, m, Q_, _ = cluster_graph(n=n, edges=e, seed=seed, resolution=R)
    return np.asarray(y)


def leiden(n, e, seed):
    y, C, Q_ = _leiden_cluster(n, e, R, seed)
    return np.asarray(y)


def run_one(name, csv, seed):
    X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
    if X.shape[0] > CAP:
        from sklearn.model_selection import train_test_split
        _, X, _, y = train_test_split(X, y, test_size=CAP, stratify=y, random_state=seed)
    n = X.shape[0]; dr = max(2, min(40, X.shape[1] - 1))
    out = {"name": name, "seed": int(seed), "n": int(n), "C_true": int(len(np.unique(y))), "cfg": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Zp = make_representation(X, seed=seed, dr_dim=dr, rep_type="pca")
        Zu = make_representation(X, seed=seed, dr_dim=dr, rep_type="umap")
        configs = {
            "PCA+KNN+Louvain":  lambda: louvain(*knn_edges(Zp, seed), seed),
            "UMAP+KNN+Louvain": lambda: louvain(*knn_edges(Zu, seed), seed),
            "UMAP+KNN+Leiden":  lambda: leiden(*knn_edges(Zu, seed), seed),
            "UMAP+SNN+Leiden":  lambda: leiden(n, _build_snn_graph(Zu, K, metric=METRIC), seed),
        }
        for name_cfg, fn in configs.items():
            try:
                yy = np.asarray(fn())
                out["cfg"][name_cfg] = {"ari": float(adjusted_rand_score(y, yy)),
                                        "C": int(len(np.unique(yy)))}
            except Exception as e:
                out["cfg"][name_cfg] = {"ari": None, "C": None, "err": str(e)[:120]}
    return out


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
        s = " ".join(f"{k.split('+')[0][0]}{k.split('+')[2][0]}={v['ari']:.3f}" if v['ari'] is not None else f"{k}=ERR"
                     for k, v in res["cfg"].items())
        print(f"[ok] {name} s{seed} {s} ({time.time()-ts:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
