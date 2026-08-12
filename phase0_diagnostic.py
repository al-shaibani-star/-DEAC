# -*- coding: utf-8 -*-
"""
Phase 0 diagnostic: which INTERNAL signal predicts the best engine (by ARI)?

Re-runs the three engines (Baseline, AMR-DE, EMR-CGC) per dataset and computes,
for each engine's labels on a common reference representation, every candidate
internal metric (MOD, SIL, BAL, CH, DB, plus C) alongside the external ARI.
Saved to results_phase0/. Then reports, per metric: how often its argmax over
the three engines matches the ARI-argmax (oracle pick), and its mean Spearman
rank correlation with ARI across engines.

Subsampled to 2000 (exploratory; correlation structure is robust to this).
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
from src.io.utils import cluster_balance_score, modularity01_from_value, silhouette01_from_value
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                             calinski_harabasz_score, davies_bouldin_score)
import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_mod

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_phase0")
DATASETS = [("Wine","Wine.csv"),("Glass","Glass.csv"),("Olivetti","olivetti_full.csv"),
    ("CNAE-9","CNAE9.csv"),("COIL-20","COIL20.csv"),("Optdigits","Optdigits.csv"),
    ("MFeat","mfeat_full.csv"),("Segmentation","Segmentation.csv"),("ISOLET","isolet_full.csv"),
    ("Satimage","Satimage.csv"),("USPS","USPS.csv"),("PenDigits","PenDigits.csv")]
SEED, CAP, METRIC = 1001, 2000, "cosine"


def cluster_on_rep(Z, k, q, r):
    n, e = build_knn_graph(Z, k=int(k), metric=METRIC, graph_mode="symmetric", prune_q=float(q), min_deg_guard=3)
    e = keep_graph_connected_by_bridging(Z, e, metric=METRIC)
    y, C, deg, m, Q, _ = cluster_graph(n=n, edges=e, seed=SEED, resolution=float(r))
    return np.asarray(y), e


def metrics(labels, Z, G):
    u = np.unique(labels)
    if len(u) < 2:
        return dict(MOD=0, SIL=0, BAL=0, CH=0, DB=0, C=int(len(u)))
    comms = [set(np.where(labels==l)[0]) for l in u]
    try: mod = modularity01_from_value(float(nx_mod(G, comms, weight="weight")))
    except: mod = 0.0
    try: sil = silhouette01_from_value(float(silhouette_score(Z, labels, metric=METRIC, sample_size=min(1000,len(labels)), random_state=SEED)))
    except: sil = 0.0
    bal = cluster_balance_score(labels)[0]
    try: ch = float(np.clip(calinski_harabasz_score(Z, labels)/1000.0, 0, 1))
    except: ch = 0.0
    try: db = float(np.clip(1.0 - davies_bouldin_score(Z, labels)/5.0, 0, 1))
    except: db = 0.0
    return dict(MOD=mod, SIL=sil, BAL=float(bal), CH=ch, DB=db, C=int(len(u)))


def run_one(name, csv):
    X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
    if X.shape[0] > CAP:
        from sklearn.model_selection import train_test_split
        _, X, _, y = train_test_split(X, y, test_size=CAP, stratify=y, random_state=SEED)
    dr = max(2, min(40, X.shape[1]-1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Zb, _ = make_representation(X, seed=SEED, dr_dim=dr, rep_type="umap"), None
        Zb = make_representation(X, seed=SEED, dr_dim=dr, rep_type="umap")
        yB, eB = cluster_on_rep(Zb, 35, 0.1, 0.25)
        amr = run_fast_amr_de(X, y, seed=SEED, metric=METRIC, subsample=min(800,X.shape[0]),
                              de_maxiter=8, de_popsize=5, n_restarts=2, max_dr_dim=X.shape[1]-1, max_k=X.shape[0]-1)
        Zde = make_representation(X, seed=SEED, dr_dim=min(dr,amr.best_d), rep_type=amr.best_rep,
                                  n_neighbors=amr.best_n_neighbors, min_dist=amr.best_min_dist, perplexity=amr.best_perplexity)
        yDE, _ = cluster_on_rep(Zde, amr.best_k, amr.best_q, amr.best_r)
        emr = run_emr_cgc(X, y, seed=SEED, metric=METRIC, subsample=0)
        yEMR = np.asarray(emr.y_consensus)
        G = nx.Graph(); G.add_nodes_from(range(len(yB)))
        for a,b,w in eB:
            if a!=b: G.add_edge(int(a),int(b),weight=float(w))
        res = {"name": name, "engines": {}}
        for eng, yy in [("Baseline",yB),("AMR-DE",yDE),("EMR-CGC",yEMR)]:
            m = metrics(yy, Zb, G); m["ARI"] = float(adjusted_rand_score(y, yy))
            res["engines"][eng] = m
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"{name}.json"),"w"), indent=2)
    print(f"{name:<13} " + " ".join(f"{e}:ARI={res['engines'][e]['ARI']:.2f}" for e in res['engines']), flush=True)
    return res


def analyze():
    from scipy.stats import spearmanr
    rows = [json.load(open(os.path.join(OUT,f"{n}.json"))) for n,_ in DATASETS if os.path.exists(os.path.join(OUT,f"{n}.json"))]
    if len(rows) < 3: return
    engs = ["Baseline","AMR-DE","EMR-CGC"]
    sig_metrics = ["MOD","SIL","BAL","CH","DB"]
    print(f"\n{'metric':<8}{'picks-oracle':>14}{'mean-Spearman':>16}")
    print("-"*40)
    for m in sig_metrics + ["combined(.40MOD+.35SIL+.25BAL)"]:
        agree=0; sps=[]
        for r in rows:
            ari=[r["engines"][e]["ARI"] for e in engs]
            if m.startswith("combined"):
                val=[0.40*r["engines"][e]["MOD"]+0.35*r["engines"][e]["SIL"]+0.25*r["engines"][e]["BAL"] for e in engs]
            else:
                val=[r["engines"][e][m] for e in engs]
            if engs[int(np.argmax(val))]==engs[int(np.argmax(ari))]: agree+=1
            if len(set(val))>1 and len(set(ari))>1:
                sps.append(spearmanr(val,ari).correlation)
        print(f"{m:<8}{agree:>10}/{len(rows)}{np.nanmean(sps) if sps else 0:>16.2f}")


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
