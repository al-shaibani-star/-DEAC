# -*- coding: utf-8 -*-
"""
THE credibility experiment: does the deployable QualityGate (which selects an
engine using INTERNAL metrics only, NO labels) recover the oracle's ARI (which
picks the best-ARI engine per dataset using ground-truth labels)?

For each dataset we run the three engines (Baseline UMAP+Louvain, AMR-DE,
EMR-CGC), obtain their label vectors, then:
  * gate pick   = argmax internal quality score (0.40 MOD + 0.35 SIL + 0.25 BAL)
                  over a COMMON reference representation, no labels used,
                  with the over-fragmentation guard rho = max(4, sqrt(n/50)).
  * oracle pick = argmax ARI (uses ground-truth labels).
Report per-dataset gate-ARI vs oracle-ARI and the mean gap.

Datasets subsampled to 2000 for tractability (the gate-vs-oracle GAP is the
quantity of interest and is robust to subsampling). Checkpointed to
results_gate/.
"""
import json
import os
import sys
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
                          silhouette01_from_value)
from sklearn.metrics import adjusted_rand_score, silhouette_score
import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_mod

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_gate")
DATASETS = [
    ("Wine", "Wine.csv"), ("Glass", "Glass.csv"), ("Olivetti", "olivetti_full.csv"),
    ("CNAE-9", "CNAE9.csv"), ("COIL-20", "COIL20.csv"), ("Optdigits", "Optdigits.csv"),
    ("MFeat", "mfeat_full.csv"), ("Segmentation", "Segmentation.csv"),
    ("ISOLET", "isolet_full.csv"), ("Satimage", "Satimage.csv"),
    ("USPS", "USPS.csv"), ("PenDigits", "PenDigits.csv"),
]
SEED = 1001
CAP = 100000
METRIC = "cosine"


def cluster_on_rep(Z, k, q, r, seed):
    n, edges = build_knn_graph(Z, k=int(k), metric=METRIC, graph_mode="symmetric",
                               prune_q=float(q), min_deg_guard=3)
    edges = keep_graph_connected_by_bridging(Z, edges, metric=METRIC)
    y, C, degen, m, Q, _ = cluster_graph(n=n, edges=edges, seed=seed, resolution=float(r))
    return np.asarray(y), edges


def internal_quality(labels, Z_ref, G_ref):
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
                                     sample_size=min(1000, len(labels)), random_state=SEED))
    except Exception:
        sil = 0.0
    sil01 = silhouette01_from_value(sil)
    bal01, _, _ = cluster_balance_score(labels)
    return 0.40 * mod01 + 0.35 * sil01 + 0.25 * bal01


def run_one(name, csv):
    X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
    rng = np.random.RandomState(SEED)
    if X.shape[0] > CAP:
        from sklearn.model_selection import train_test_split
        _, X, _, y = train_test_split(X, y, test_size=CAP, stratify=y, random_state=SEED)
    n = X.shape[0]
    dr = max(2, min(40, X.shape[1] - 1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # --- Baseline: UMAP + Louvain (k=35,q=0.1,r=0.25) ---
        Zb = make_representation(X, seed=SEED, dr_dim=dr, rep_type="umap")
        yB, edgesB = cluster_on_rep(Zb, 35, 0.1, 0.25, SEED)

        # --- AMR-DE ---
        amr = run_fast_amr_de(X, y, seed=SEED, metric=METRIC, subsample=min(800, n),
                              de_maxiter=8, de_popsize=5, n_restarts=2,
                              max_dr_dim=X.shape[1] - 1, max_k=n - 1)
        Zde = make_representation(X, seed=SEED, dr_dim=min(dr, amr.best_d),
                                  rep_type=amr.best_rep,
                                  n_neighbors=amr.best_n_neighbors,
                                  min_dist=amr.best_min_dist, perplexity=amr.best_perplexity)
        yDE, _ = cluster_on_rep(Zde, amr.best_k, amr.best_q, amr.best_r, SEED)

        # --- EMR-CGC ---
        emr = run_emr_cgc(X, y, seed=SEED, metric=METRIC, subsample=0)
        yEMR = np.asarray(emr.y_consensus)

        # --- common reference graph on baseline rep for modularity ---
        G_ref = nx.Graph(); G_ref.add_nodes_from(range(n))
        for a, b, w in edgesB:
            if a != b:
                G_ref.add_edge(int(a), int(b), weight=float(w))

        cand = {"Baseline": yB, "AMR-DE": yDE, "EMR-CGC": yEMR}
        scores = {k: internal_quality(v, Zb, G_ref) for k, v in cand.items()}
        aris = {k: float(adjusted_rand_score(y, v)) for k, v in cand.items()}
        Cs = {k: int(len(np.unique(v))) for k, v in cand.items()}

    # over-fragmentation guard (baseline always eligible)
    rho = max(4.0, np.sqrt(n / 50.0))
    CB = Cs["Baseline"]
    eligible = [k for k in cand if k == "Baseline" or Cs[k] < rho * CB]
    gate_pick = max(eligible, key=lambda k: scores[k])
    oracle_pick = max(cand, key=lambda k: aris[k])

    res = {"name": name, "n": int(n), "rho": float(rho),
           "aris": aris, "scores": scores, "C": Cs,
           "gate_pick": gate_pick, "oracle_pick": oracle_pick,
           "gate_ari": aris[gate_pick], "oracle_ari": aris[oracle_pick],
           "baseline_ari": aris["Baseline"]}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"{name:<13} gate->{gate_pick:<9}({res['gate_ari']:.3f})  "
          f"oracle->{oracle_pick:<9}({res['oracle_ari']:.3f})  "
          f"gap={res['oracle_ari']-res['gate_ari']:+.3f}  "
          f"{'MATCH' if gate_pick==oracle_pick else 'DIFFER'}", flush=True)
    return res


def main():
    wanted = sys.argv[1:] if len(sys.argv) > 1 else [d[0] for d in DATASETS]
    rows = []
    for name, csv in DATASETS:
        if name not in wanted:
            continue
        ck = os.path.join(OUT_DIR, f"{name}.json")
        if os.path.exists(ck):
            rows.append(json.load(open(ck))); print(f"[cache] {name}"); continue
        try:
            rows.append(run_one(name, csv))
        except Exception as e:
            import traceback; print(f"[ERR] {name}: {e}"); traceback.print_exc()
    if len(rows) >= 2:
        g = np.mean([r["gate_ari"] for r in rows])
        o = np.mean([r["oracle_ari"] for r in rows])
        b = np.mean([r["baseline_ari"] for r in rows])
        match = sum(r["gate_pick"] == r["oracle_pick"] for r in rows)
        print("-" * 70)
        print(f"MEAN  gate={g:.3f}  oracle={o:.3f}  baseline={b:.3f}  "
              f"GAP={o-g:+.3f}  agree={match}/{len(rows)}")
        print(f"Gate recovers {100*(g-b)/(o-b) if o>b else 100:.0f}% of oracle's gain over baseline")


if __name__ == "__main__":
    main()
