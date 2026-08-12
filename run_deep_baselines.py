# -*- coding: utf-8 -*-
"""
Standalone runner for modern deep-clustering baselines (N2D, DEC, IDEC, DCN).

Runs each method on the 12 paper datasets across 5 seeds (matching run_all.py:
seed = 1000 + s, s in 1..5), with the true cluster count as the oracle C.
Results are checkpointed per dataset to results_deep/ so the run is resumable.

Usage:
    python run_deep_baselines.py                 # all 12 datasets
    python run_deep_baselines.py Wine Glass       # subset (pilot)
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.io.data_loading import load_csv
from src.evaluation.deep_baselines import run_deep_baselines

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_deep")

# 12 paper datasets (name -> csv) in paper order
DATASETS = [
    ("Wine",         "Wine.csv"),
    ("Glass",        "Glass.csv"),
    ("Olivetti",     "olivetti_full.csv"),
    ("CNAE-9",       "CNAE9.csv"),
    ("COIL-20",      "COIL20.csv"),
    ("Optdigits",    "Optdigits.csv"),
    ("MFeat",        "mfeat_full.csv"),
    ("Segmentation", "Segmentation.csv"),
    ("ISOLET",       "isolet_full.csv"),
    ("Satimage",     "Satimage.csv"),
    ("USPS",         "USPS.csv"),
    ("PenDigits",    "PenDigits.csv"),
]

SEEDS = [1001, 1002, 1003, 1004, 1005]
METHODS = ["N2D", "DEC", "IDEC", "DCN"]
PRETRAIN_EPOCHS = 100


def _ckpt_path(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, f"{name}.json")


def run_one_dataset(name, csv):
    path = os.path.join(DATA_DIR, csv)
    X, y, _ = load_csv(path, label_col=-1)
    true_c = int(len(np.unique(y)))
    print(f"\n{'='*70}\n{name}: n={X.shape[0]} d={X.shape[1]} C_true={true_c}\n{'='*70}")

    # raw per-seed results: method -> list of dicts
    per_method = {m: [] for m in METHODS}
    for seed in SEEDS:
        print(f"  seed {seed}:")
        res = run_deep_baselines(X, y, n_clusters=true_c, seed=seed,
                                 methods=METHODS, pretrain_epochs=PRETRAIN_EPOCHS)
        for r in res:
            per_method[r.method].append(
                {"ARI": r.ARI, "NMI": r.NMI, "ACC": r.ACC, "C": r.C, "time": r.time_sec})

    # summarize
    summary = {"name": name, "n": int(X.shape[0]), "d": int(X.shape[1]),
               "C_true": true_c, "methods": {}}
    for m in METHODS:
        arr = per_method[m]
        summary["methods"][m] = {
            "ARI_mean": float(np.mean([a["ARI"] for a in arr])),
            "ARI_std":  float(np.std([a["ARI"] for a in arr])),
            "NMI_mean": float(np.mean([a["NMI"] for a in arr])),
            "NMI_std":  float(np.std([a["NMI"] for a in arr])),
            "ACC_mean": float(np.mean([a["ACC"] for a in arr])),
            "ACC_std":  float(np.std([a["ACC"] for a in arr])),
            "C_mean":   float(np.mean([a["C"] for a in arr])),
            "time_mean": float(np.mean([a["time"] for a in arr])),
            "raw": arr,
        }
    with open(_ckpt_path(name), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [SAVED] {_ckpt_path(name)}")
    return summary


def main():
    wanted = sys.argv[1:] if len(sys.argv) > 1 else [d[0] for d in DATASETS]
    t0 = time.time()
    all_summ = []
    for name, csv in DATASETS:
        if name not in wanted:
            continue
        ck = _ckpt_path(name)
        if os.path.exists(ck):
            print(f"[CACHE] {name} already done -- skipping")
            with open(ck, encoding="utf-8") as f:
                all_summ.append(json.load(f))
            continue
        try:
            all_summ.append(run_one_dataset(name, csv))
        except Exception as e:
            import traceback
            print(f"[ERROR] {name}: {e}")
            traceback.print_exc()

    # final table
    print(f"\n{'='*90}\nDEEP BASELINES SUMMARY (mean ARI over {len(SEEDS)} seeds)\n{'='*90}")
    print(f"{'Dataset':<14}" + "".join(f"{m:>10}" for m in METHODS))
    print("-" * 90)
    for s in all_summ:
        row = f"{s['name']:<14}"
        for m in METHODS:
            row += f"{s['methods'][m]['ARI_mean']:>10.3f}"
        print(row)
    print("-" * 90)
    # means across datasets
    row = f"{'MEAN':<14}"
    for m in METHODS:
        vals = [s["methods"][m]["ARI_mean"] for s in all_summ]
        row += f"{np.mean(vals):>10.3f}"
    print(row)

    with open(os.path.join(OUT_DIR, "_all_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summ, f, indent=2)
    print(f"\nTotal time: {time.time()-t0:.1f}s | Output: {OUT_DIR}/")


if __name__ == "__main__":
    main()
