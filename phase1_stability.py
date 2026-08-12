# -*- coding: utf-8 -*-
"""
Phase 1 / Approach A: stability-based label-free engine selection.

For each engine we measure how REPRODUCIBLE its clustering is under data
perturbation: draw B subsamples (80%), re-cluster with the same engine, and
score stability as the mean ARI between each subsample clustering and the full
reference (restricted to the subsample). The most stable engine is selected.
No ground-truth labels are used for selection; ARI to ground truth is only for
evaluation. Checkpointed to results_stability/.
"""
import json, os, sys, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.io.data_loading import load_csv
from src.io.representation import make_representation
from src.graph.construction import build_knn_graph, keep_graph_connected_by_bridging
from src.graph.clustering import cluster_graph
from src.core.fast_amr_de import run_fast_amr_de
from src.core.emr_cgc import run_emr_cgc
from sklearn.metrics import adjusted_rand_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_stability")
DATASETS = [("Wine","Wine.csv"),("Glass","Glass.csv"),("Olivetti","olivetti_full.csv"),
    ("CNAE-9","CNAE9.csv"),("COIL-20","COIL20.csv"),("Optdigits","Optdigits.csv"),
    ("MFeat","mfeat_full.csv"),("Segmentation","Segmentation.csv"),("ISOLET","isolet_full.csv"),
    ("Satimage","Satimage.csv"),("USPS","USPS.csv"),("PenDigits","PenDigits.csv")]
SEED, CAP, B, METRIC = 1001, 1000, 5, "cosine"
ENGS = ["Baseline","AMR-DE","EMR-CGC"]


def cluster_umap_louvain(X, seed, k, q, r, dr, rep="umap", nn=15, md=0.1):
    Z = make_representation(X, seed=seed, dr_dim=dr, rep_type=rep, n_neighbors=nn, min_dist=md)
    n, e = build_knn_graph(Z, k=int(min(k, len(X)-1)), metric=METRIC, graph_mode="symmetric",
                           prune_q=float(q), min_deg_guard=3)
    e = keep_graph_connected_by_bridging(Z, e, metric=METRIC)
    y, C, deg, m, Q, _ = cluster_graph(n=n, edges=e, seed=seed, resolution=float(r))
    return np.asarray(y)


def engine_labels(X, y, kind, amr=None):
    dr = max(2, min(40, X.shape[1]-1))
    if kind == "Baseline":
        return cluster_umap_louvain(X, SEED, 35, 0.1, 0.25, dr)
    if kind == "AMR-DE":
        return cluster_umap_louvain(X, SEED, amr.best_k, amr.best_q, amr.best_r,
                                    min(dr, amr.best_d), rep=amr.best_rep,
                                    nn=amr.best_n_neighbors, md=amr.best_min_dist)
    if kind == "EMR-CGC":
        return np.asarray(run_emr_cgc(X, y, seed=SEED, metric=METRIC, subsample=0).y_consensus)


def run_one(name, csv):
    X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
    rng = np.random.RandomState(SEED)
    if X.shape[0] > CAP:
        from sklearn.model_selection import train_test_split
        _, X, _, y = train_test_split(X, y, test_size=CAP, stratify=y, random_state=SEED)
    n = X.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        amr = run_fast_amr_de(X, y, seed=SEED, metric=METRIC, subsample=min(800,n),
                              de_maxiter=8, de_popsize=5, n_restarts=2,
                              max_dr_dim=X.shape[1]-1, max_k=n-1)
        ref = {e: engine_labels(X, y, e, amr) for e in ENGS}
        ari = {e: float(adjusted_rand_score(y, ref[e])) for e in ENGS}
        stab = {e: [] for e in ENGS}
        for b in range(B):
            idx = np.sort(rng.choice(n, size=int(0.8*n), replace=False))
            Xs, ys = X[idx], y[idx]
            amr_s = amr  # reuse config (don't re-optimize)
            for e in ENGS:
                try:
                    ls = engine_labels(Xs, ys, e, amr_s)
                    stab[e].append(float(adjusted_rand_score(ref[e][idx], ls)))
                except Exception:
                    stab[e].append(0.0)
        stability = {e: float(np.mean(stab[e])) for e in ENGS}
    res = {"name": name, "ari": ari, "stability": stability,
           "stab_pick": max(ENGS, key=lambda e: stability[e]),
           "oracle_pick": max(ENGS, key=lambda e: ari[e])}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT,f"{name}.json"),"w"), indent=2)
    print(f"{name:<13} stab->{res['stab_pick']:<9}(ARI={ari[res['stab_pick']]:.3f}) "
          f"oracle->{res['oracle_pick']:<9}(ARI={ari[res['oracle_pick']]:.3f}) "
          f"{'MATCH' if res['stab_pick']==res['oracle_pick'] else 'DIFFER'}", flush=True)
    return res


def analyze():
    rows = [json.load(open(os.path.join(OUT,f"{n}.json"))) for n,_ in DATASETS if os.path.exists(os.path.join(OUT,f"{n}.json"))]
    if len(rows) < 2: return
    base = np.mean([r["ari"]["Baseline"] for r in rows])
    orac = np.mean([max(r["ari"].values()) for r in rows])
    stab = np.mean([r["ari"][r["stab_pick"]] for r in rows])
    ag = sum(r["stab_pick"]==r["oracle_pick"] for r in rows)
    print("-"*60)
    print(f"STABILITY selector: agree={ag}/{len(rows)}  deployable ARI={stab:.3f}  "
          f"oracle={orac:.3f}  baseline={base:.3f}  "
          f"recover={100*(stab-base)/(orac-base) if orac>base else 0:.0f}%")


def main():
    for name, csv in DATASETS:
        if os.path.exists(os.path.join(OUT,f"{name}.json")):
            print(f"[cache] {name}"); continue
        try: run_one(name, csv)
        except Exception as ex:
            import traceback; print(f"[ERR] {name}: {ex}"); traceback.print_exc()
    analyze()


if __name__ == "__main__":
    main()
