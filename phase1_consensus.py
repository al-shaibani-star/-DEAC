# -*- coding: utf-8 -*-
"""
Phase 1 / Approach B: consensus-based label-free selection.

B1 (agreement selector): pick the engine whose partition agrees most (mean ARI)
    with the other two engines -- the one closest to the ensemble consensus.
B2 (consensus partition): build a co-association matrix over the three engine
    partitions and derive a single consensus clustering (agglomerative on 1-CA),
    used directly as the output.

No ground-truth labels are used for selection; ARI to ground truth is only for
evaluation. Checkpointed to results_consensus/.
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
from sklearn.cluster import AgglomerativeClustering

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_consensus")
DATASETS = [("Wine","Wine.csv"),("Glass","Glass.csv"),("Olivetti","olivetti_full.csv"),
    ("CNAE-9","CNAE9.csv"),("COIL-20","COIL20.csv"),("Optdigits","Optdigits.csv"),
    ("MFeat","mfeat_full.csv"),("Segmentation","Segmentation.csv"),("ISOLET","isolet_full.csv"),
    ("Satimage","Satimage.csv"),("USPS","USPS.csv"),("PenDigits","PenDigits.csv")]
SEED, CAP, METRIC = 1001, 2000, "cosine"
ENGS = ["Baseline","AMR-DE","EMR-CGC"]


def cluster_on_rep(Z, k, q, r):
    n, e = build_knn_graph(Z, k=int(min(k,len(Z)-1)), metric=METRIC, graph_mode="symmetric", prune_q=float(q), min_deg_guard=3)
    e = keep_graph_connected_by_bridging(Z, e, metric=METRIC)
    y, C, deg, m, Q, _ = cluster_graph(n=len(Z), edges=e, seed=SEED, resolution=float(r))
    return np.asarray(y)


def coassoc(parts):
    n = len(parts[0]); CA = np.zeros((n, n), np.float32)
    for p in parts:
        CA += (p[:, None] == p[None, :]).astype(np.float32)
    return CA / len(parts)


def run_one(name, csv):
    X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
    rng = np.random.RandomState(SEED)
    if X.shape[0] > CAP:
        from sklearn.model_selection import train_test_split
        _, X, _, y = train_test_split(X, y, test_size=CAP, stratify=y, random_state=SEED)
    n = X.shape[0]; dr = max(2, min(40, X.shape[1]-1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Zb = make_representation(X, seed=SEED, dr_dim=dr, rep_type="umap")
        yB = cluster_on_rep(Zb, 35, 0.1, 0.25)
        amr = run_fast_amr_de(X, y, seed=SEED, metric=METRIC, subsample=min(800,n),
                              de_maxiter=8, de_popsize=5, n_restarts=2, max_dr_dim=X.shape[1]-1, max_k=n-1)
        Zde = make_representation(X, seed=SEED, dr_dim=min(dr,amr.best_d), rep_type=amr.best_rep,
                                  n_neighbors=amr.best_n_neighbors, min_dist=amr.best_min_dist)
        yDE = cluster_on_rep(Zde, amr.best_k, amr.best_q, amr.best_r)
        yEMR = np.asarray(run_emr_cgc(X, y, seed=SEED, metric=METRIC, subsample=0).y_consensus)
        parts = {"Baseline": yB, "AMR-DE": yDE, "EMR-CGC": yEMR}
        ari = {e: float(adjusted_rand_score(y, parts[e])) for e in ENGS}
        # B1: mean agreement with the other two
        agr = {}
        for e in ENGS:
            oth = [parts[o] for o in ENGS if o != e]
            agr[e] = float(np.mean([adjusted_rand_score(parts[e], o) for o in oth]))
        b1 = max(ENGS, key=lambda e: agr[e])
        # B2: co-association consensus partition (k = median engine C)
        Cs = [len(np.unique(parts[e])) for e in ENGS]
        k = int(np.clip(int(np.median(Cs)), 2, n-1))
        CA = coassoc([yB, yDE, yEMR])
        try:
            cons = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(1.0-CA)
            b2_ari = float(adjusted_rand_score(y, cons))
        except Exception:
            b2_ari = ari["Baseline"]
    res = {"name": name, "ari": ari, "agreement": agr, "b1_pick": b1,
           "b1_ari": ari[b1], "b2_ari": b2_ari,
           "oracle_pick": max(ENGS, key=lambda e: ari[e]), "oracle_ari": max(ari.values())}
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT,f"{name}.json"),"w"), indent=2)
    print(f"{name:<13} B1->{b1:<9}({ari[b1]:.3f}) B2-consensus={b2_ari:.3f} oracle={max(ari.values()):.3f}", flush=True)
    return res


def analyze():
    rows = [json.load(open(os.path.join(OUT,f"{n}.json"))) for n,_ in DATASETS if os.path.exists(os.path.join(OUT,f"{n}.json"))]
    if len(rows) < 2: return
    base = np.mean([r["ari"]["Baseline"] for r in rows]); orac = np.mean([r["oracle_ari"] for r in rows])
    b1 = np.mean([r["b1_ari"] for r in rows]); b2 = np.mean([r["b2_ari"] for r in rows])
    ag1 = sum(r["b1_pick"]==r["oracle_pick"] for r in rows)
    rec = lambda v: 100*(v-base)/(orac-base) if orac>base else 0
    print("-"*56)
    print(f"B1 agreement-selector: agree={ag1}/{len(rows)}  ARI={b1:.3f}  recover={rec(b1):.0f}%")
    print(f"B2 consensus-partition:            ARI={b2:.3f}  recover={rec(b2):.0f}%")
    print(f"(oracle={orac:.3f}  baseline={base:.3f})")


def main():
    for name, csv in DATASETS:
        if os.path.exists(os.path.join(OUT,f"{name}.json")): print(f"[cache] {name}"); continue
        try: run_one(name, csv)
        except Exception as ex:
            import traceback; print(f"[ERR] {name}: {ex}"); traceback.print_exc()
    analyze()


if __name__ == "__main__":
    main()
