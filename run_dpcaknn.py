# -*- coding: utf-8 -*-
"""
R2.4 comparator run: faithful DPC-AKNN (src/evaluation/dpc_aknn.py) on the 12
benchmarks, given the true cluster count (known-C protocol, matching the deep
baselines). DPC-AKNN is O(n^3) and the source paper uses small datasets, so each
dataset is a stratified subsample (cap=2000). Checkpointed/resumable per
(dataset, seed) to results_dpcaknn/. Single process (pure CPU/sklearn).
"""
import json, os, sys, time, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.io.data_loading import load_csv
from src.evaluation.dpc_aknn import dpc_aknn
from src.io.utils import best_map_acc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_dpcaknn")
DATASETS = [("Wine", "Wine.csv"), ("Glass", "Glass.csv"), ("Olivetti", "olivetti_full.csv"),
            ("CNAE-9", "CNAE9.csv"), ("COIL-20", "COIL20.csv"), ("Optdigits", "Optdigits.csv"),
            ("MFeat", "mfeat_full.csv"), ("Segmentation", "Segmentation.csv"),
            ("ISOLET", "isolet_full.csv"), ("Satimage", "Satimage.csv"),
            ("USPS", "USPS.csv"), ("PenDigits", "PenDigits.csv")]
SEEDS = [1001, 1002, 1003, 1004, 1005]
CAP = 2000


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, csv in DATASETS:
        for seed in SEEDS:
            ck = os.path.join(OUT, f"{name}_s{seed}.json")
            if os.path.exists(ck):
                print(f"[cache] {name} s{seed}", flush=True); continue
            try:
                X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
                if X.shape[0] > CAP:
                    from sklearn.model_selection import train_test_split
                    _, X, _, y = train_test_split(X, y, test_size=CAP, stratify=y, random_state=seed)
                C = int(len(np.unique(y)))
                ts = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lab = dpc_aknn(X, C=C, seed=seed)
                res = {"name": name, "seed": int(seed), "n": int(X.shape[0]), "C_true": C,
                       "ari": float(adjusted_rand_score(y, lab)),
                       "nmi": float(normalized_mutual_info_score(y, lab)),
                       "acc": float(best_map_acc(y, lab)),
                       "C_found": int(len(np.unique(lab)))}
                json.dump(res, open(ck, "w"), indent=2)
                print(f"[ok] {name:<13} s{seed} ARI={res['ari']:.3f} NMI={res['nmi']:.3f} "
                      f"({time.time()-ts:.0f}s)", flush=True)
            except Exception as e:
                import traceback; print(f"[ERR] {name} s{seed}: {e}"); traceback.print_exc()
    print("\nDONE. Checkpoints in results_dpcaknn/", flush=True)


if __name__ == "__main__":
    main()
