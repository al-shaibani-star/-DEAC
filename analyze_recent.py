# -*- coding: utf-8 -*-
"""
R2.4: build the recent-methods comparison (DPC-AKNN 2026, PRCut 2025) against our
deployable gate / EMR-CGC and N2D, all as mean ARI over the 12 benchmarks. States
each method's representation and whether it is given the true cluster count.
Writes paper/generated_canonical/recent_table.tex and prints a summary.
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper", "generated_canonical")
ORDER = ["Wine", "Glass", "Olivetti", "CNAE-9", "COIL-20", "Optdigits", "MFeat",
         "Segmentation", "ISOLET", "Satimage", "USPS", "PenDigits"]
SEEDS = [1001, 1002, 1003, 1004, 1005]


def mean_over(dirn, key="ari", seeds=None):
    by = {}
    for f in glob.glob(os.path.join(HERE, dirn, "*_s*.json")):
        d = json.load(open(f)); by.setdefault(d["name"], {})[d["seed"]] = d
    per = []
    for n in ORDER:
        ss = by.get(n, {})
        if ss:
            per.append(np.mean([ss[s][key] for s in ss]))
    return float(np.mean(per)) if per else float("nan")


def canonical_means():
    by = {}
    for f in glob.glob(os.path.join(HERE, "results_canonical", "*_s*.json")):
        d = json.load(open(f)); by.setdefault(d["name"], {})[d["seed"]] = d
    gate = np.mean([np.mean([by[n][s]["ari"][by[n][s]["gate_pick"]] for s in SEEDS]) for n in ORDER])
    emr = np.mean([np.mean([by[n][s]["ari"]["EMR-CGC"] for s in SEEDS]) for n in ORDER])
    orac = np.mean([np.mean([max(by[n][s]["ari"][e] for e in ["Baseline", "AMR-DE", "EMR-CGC"]) for s in SEEDS]) for n in ORDER])
    return float(gate), float(emr), float(orac)


def n2d_mean():
    v = []
    for n in ORDER:
        v.append(json.load(open(os.path.join(HERE, "results_deep", f"{n}.json")))["methods"]["N2D"]["ARI_mean"])
    return float(np.mean(v))


def main():
    gate, emr, orac = canonical_means()
    prcut = mean_over("results_prcut"); dpc = mean_over("results_dpcaknn"); n2d = n2d_mean()
    rows = [
        (r"\textbf{DEAC gate (ours)}", "UMAP+SNN ensemble", "no (infers $C$)", gate, True),
        (r"EMR-CGC (ours)", "UMAP+SNN ensemble", "no (infers $C$)", emr, False),
        (r"PRCut (2025)~\cite{ghriss2025prcut}", "common UMAP", "yes", prcut, False),
        (r"DPC-AKNN (2026)~\cite{wang2026density}", "native (raw+geodesic)", "yes", dpc, False),
        (r"N2D (deep)~\cite{mcconville2021n2d}", "autoencoder+UMAP", "yes", n2d, False),
        (r"\textit{Oracle (upper bound)}", "---", "uses labels", orac, False),
    ]
    print("Method                         repr                 givenC  meanARI")
    for name, rep, gc, v, _ in rows:
        print(f"  {name[:28]:<28} {rep:<20} {gc:<10} {v:.3f}")

    L = [r"% R2.4 recent-methods comparison (analyze_recent.py)", r"\begin{table}[htbp]", r"\centering",
         r"\caption{Comparison with recent methods (mean ARI over 12 datasets). DPC-AKNN and PRCut, "
         r"like the deep baselines, are given the true cluster count; our gate and EMR-CGC infer it. "
         r"PRCut is run on the same UMAP representation (common-representation comparison); DPC-AKNN uses "
         r"its native raw-feature geodesic pipeline (stratified cap 2000, its $O(n^3)$ cost). Our label-free "
         r"method is on par with PRCut and N2D and well above DPC-AKNN, without being told $C$.}",
         r"\label{tab:recent}", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{llcc}", r"\toprule",
         r"Method & Representation & Given $C$? & Mean ARI \\", r"\midrule"]
    for i, (name, rep, gc, v, bold) in enumerate(rows):
        cell = f"\\textbf{{{v:.3f}}}" if bold else (f"\\textit{{{v:.3f}}}" if "Oracle" in name else f"{v:.3f}")
        L.append(f"{name} & {rep} & {gc} & {cell} \\\\")
        if i == 3:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "recent_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"\n[tex] {os.path.join(OUT,'recent_table.tex')}")


if __name__ == "__main__":
    main()
