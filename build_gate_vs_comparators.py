# -*- coding: utf-8 -*-
"""
Compare the DEPLOYABLE gate (canonical, 5-seed) against the classical auto-C
baselines (results_autoc/) and the deep baselines (results_deep/), dataset-level,
with two-tailed Wilcoxon signed-rank + Holm-Bonferroni over the 9 comparisons.
This is the honest competitive picture the oracle previously masked (Reviewer 2.1).
Prints a table and writes gate_vs_comparators.csv to the review folder.
"""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper", "generated_canonical")
ORDER = ["Wine", "Glass", "Olivetti", "CNAE-9", "COIL-20", "Optdigits", "MFeat",
         "Segmentation", "ISOLET", "Satimage", "USPS", "PenDigits"]
CLASS = ["KMeans", "Spectral", "Agglomerative", "GMM", "HDBSCAN"]
DEEP = ["N2D", "DEC", "IDEC", "DCN"]

# gate (deployable DEAC) mean ARI per dataset from canonical
gate = {}
cby = {}
for f in glob.glob(os.path.join(HERE, "results_canonical", "*_s*.json")):
    d = json.load(open(f))
    cby.setdefault(d["name"], []).append(d["ari"][d["gate_pick"]])
for n in ORDER:
    gate[n] = float(np.mean(cby[n]))

def load(dirn):
    out = {}
    for n in ORDER:
        p = os.path.join(HERE, dirn, f"{n}.json")
        d = json.load(open(p))
        out[n] = {m: d["methods"][m]["ARI_mean"] for m in d["methods"]}
    return out

cl = load("results_autoc")
dp = load("results_deep")
allm = {**{m: [cl[n][m] for n in ORDER] for m in CLASS},
        **{m: [dp[n][m] for n in ORDER] for m in DEEP}}
gvec = np.array([gate[n] for n in ORDER])

# Wilcoxon gate vs each, collect p-values then Holm
rows = []
for m in CLASS + DEEP:
    mv = np.array(allm[m])
    diff = gvec - mv
    wins = int(np.sum(diff > 1e-9)); losses = int(np.sum(diff < -1e-9)); ties = 12 - wins - losses
    try:
        p = float(wilcoxon(gvec, mv, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        p = 1.0
    rows.append([m, "classical" if m in CLASS else "deep", float(np.mean(mv)), wins, losses, ties, p])

# Holm-Bonferroni over the 9 p-values
order = sorted(range(len(rows)), key=lambda i: rows[i][6])
k = len(rows); prev = 0.0
holm = [0.0] * k
for rank, i in enumerate(order):
    val = min(1.0, (k - rank) * rows[i][6])
    prev = max(prev, val)
    holm[i] = prev

print(f"\nDEPLOYABLE gate mean ARI = {gvec.mean():.3f}  (over {len(ORDER)} datasets, 5 seeds)\n")
print(f"{'Method':<15}{'type':<10}{'meanARI':>8}  {'W/L/T vs gate':>14}  {'p':>7}  {'p_Holm':>7}  sig")
print("-" * 78)
for i, r in enumerate(rows):
    m, t, mean, w, l, ti, p = r
    sig = "gate>" if (holm[i] < 0.05 and gvec.mean() > mean) else ("<gate?" if holm[i] < 0.05 else "ns")
    print(f"{m:<15}{t:<10}{mean:>8.3f}  {str(w)+'/'+str(l)+'/'+str(ti):>14}  {p:>7.3f}  {holm[i]:>7.3f}  {sig}")

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "gate_vs_comparators.csv"), "w") as fh:
    fh.write("method,type,mean_ari,wins,losses,ties,p,p_holm\n")
    for i, r in enumerate(rows):
        fh.write(",".join(str(x) for x in r[:7]) + f",{holm[i]}\n")
print(f"\n[csv] {os.path.join(OUT,'gate_vs_comparators.csv')}")
print("\nNote: gate ARIs are canonical full-data 5-seed; comparator ARIs are from "
      "results_autoc/results_deep. Protocol match to be stated in the paper.")
