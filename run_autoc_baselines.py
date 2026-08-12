# -*- coding: utf-8 -*-
"""
Fair classical-baseline comparison with AUTOMATIC cluster-count selection.

The paper's original Table tab:benchmarks handed the classical baselines the
cluster count C* discovered by DEAC -- a leakage a Q1 reviewer would reject.
Here each baseline instead discovers C by an unsupervised criterion: C* is the
value (over a label-free grid) that maximizes the KMeans silhouette on the same
UMAP representation, then applied to KMeans/Spectral/Agg/GMM; HDBSCAN finds C on
its own. DEAC still discovers C via its QualityGate. No ground-truth C is used.

Runs 12 datasets x 5 seeds (1001-1005). Checkpoints to results_autoc/.
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
from src.evaluation.benchmarks import (
    run_kmeans, run_spectral, run_agglomerative, run_gmm, run_hdbscan)
from config.settings import BASELINE_DR_DIM, METRIC

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_autoc")

DATASETS = [
    ("Wine", "Wine.csv"), ("Glass", "Glass.csv"), ("Olivetti", "olivetti_full.csv"),
    ("CNAE-9", "CNAE9.csv"), ("COIL-20", "COIL20.csv"), ("Optdigits", "Optdigits.csv"),
    ("MFeat", "mfeat_full.csv"), ("Segmentation", "Segmentation.csv"),
    ("ISOLET", "isolet_full.csv"), ("Satimage", "Satimage.csv"),
    ("USPS", "USPS.csv"), ("PenDigits", "PenDigits.csv"),
]
SEEDS = [1001, 1002, 1003, 1004, 1005]
METHODS = ["KMeans", "Spectral", "Agglomerative", "GMM", "HDBSCAN"]


def k_grid(n):
    hi = min(50, max(10, n // 10))
    base = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    return [k for k in base if 2 <= k <= hi]


def select_k_silhouette(Z, seed):
    """C* = argmax KMeans silhouette over a label-free grid (no ground truth)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n = Z.shape[0]
    # subsample for silhouette evaluation on large data
    rng = np.random.RandomState(seed)
    if n > 2000:
        sidx = rng.choice(n, 2000, replace=False)
    else:
        sidx = np.arange(n)
    best_k, best_s = 2, -np.inf
    for k in k_grid(n):
        try:
            labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(Z)
            if len(np.unique(labels[sidx])) < 2:
                continue
            s = silhouette_score(Z[sidx], labels[sidx], metric=METRIC)
            if s > best_s:
                best_s, best_k = s, k
        except Exception:
            continue
    return best_k


def run_one(name, csv):
    X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
    n, d = X.shape
    dr = max(2, min(BASELINE_DR_DIM, d - 1))
    print(f"\n{'='*70}\n{name}: n={n} d={d} C_true={len(np.unique(y))}\n{'='*70}")
    per = {m: [] for m in METHODS}
    sel_ks = []
    for seed in SEEDS:
        Z = make_representation(X, seed=seed, dr_dim=dr, rep_type="umap")
        kstar = select_k_silhouette(Z, seed)
        sel_ks.append(kstar)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = [
                run_kmeans(Z, y, kstar, seed, METRIC),
                run_spectral(Z, y, kstar, seed, METRIC),
                run_agglomerative(Z, y, kstar, METRIC),
                run_gmm(Z, y, kstar, seed, METRIC),
                run_hdbscan(Z, y, seed, METRIC),
            ]
        for r in res:
            per[r.method].append({"ARI": r.ARI, "NMI": r.NMI, "ACC": r.ACC, "C": r.C})
        print(f"  seed {seed}: C*={kstar} | " +
              " ".join(f"{r.method[:4]}={r.ARI:.3f}(C={r.C})" for r in res))

    summary = {"name": name, "n": int(n), "d": int(d),
               "C_true": int(len(np.unique(y))),
               "C_selected_mean": float(np.mean(sel_ks)), "methods": {}}
    for m in METHODS:
        a = per[m]
        summary["methods"][m] = {
            "ARI_mean": float(np.mean([x["ARI"] for x in a])),
            "ARI_std": float(np.std([x["ARI"] for x in a])),
            "NMI_mean": float(np.mean([x["NMI"] for x in a])),
            "ACC_mean": float(np.mean([x["ACC"] for x in a])),
            "C_mean": float(np.mean([x["C"] for x in a])),
        }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {name} (mean C*={np.mean(sel_ks):.1f}, true={summary['C_true']})")
    return summary


def main():
    wanted = sys.argv[1:] if len(sys.argv) > 1 else [d[0] for d in DATASETS]
    t0 = time.time()
    for name, csv in DATASETS:
        if name not in wanted:
            continue
        ck = os.path.join(OUT_DIR, f"{name}.json")
        if os.path.exists(ck):
            print(f"[CACHE] {name} -- skipping")
            continue
        try:
            run_one(name, csv)
        except Exception as e:
            import traceback
            print(f"[ERROR] {name}: {e}")
            traceback.print_exc()
    print(f"\nTotal time: {time.time()-t0:.1f}s | Output: {OUT_DIR}/")


if __name__ == "__main__":
    main()
