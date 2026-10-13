# -*- coding: utf-8 -*-
"""R2.3(a,b,c) controlled-ablation analysis from results_ablation/. Reports each
config's mean ARI and the three single-variable deltas, and writes
paper/generated_canonical/controlled_ablation_table.tex."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper", "generated_canonical")
ORDER = ["Wine", "Glass", "Olivetti", "CNAE-9", "COIL-20", "Optdigits", "MFeat",
         "Segmentation", "ISOLET", "Satimage", "USPS", "PenDigits"]
SEEDS = [1001, 1002, 1003, 1004, 1005]
CFG = ["PCA+KNN+Louvain", "UMAP+KNN+Louvain", "UMAP+KNN+Leiden", "UMAP+SNN+Leiden"]


def main():
    by = {}
    for f in glob.glob(os.path.join(HERE, "results_ablation", "*_s*.json")):
        d = json.load(open(f)); by.setdefault(d["name"], {})[d["seed"]] = d
    ready = [n for n in ORDER if all(s in by.get(n, {}) for s in SEEDS)]
    if len(ready) < 12:
        print("INCOMPLETE:", [n for n in ORDER if n not in ready]); return
    os.makedirs(OUT, exist_ok=True)
    mean = {}
    for c in CFG:
        vals = [np.mean([by[nm][s]["cfg"][c]["ari"] for s in SEEDS]) for nm in ORDER]
        mean[c] = float(np.mean(vals))
    print("Config mean ARI:")
    for c in CFG:
        print(f"  {c:<18} {mean[c]:.3f}")
    da = mean["UMAP+KNN+Louvain"] - mean["PCA+KNN+Louvain"]
    db = mean["UMAP+KNN+Leiden"] - mean["UMAP+KNN+Louvain"]
    dc = mean["UMAP+SNN+Leiden"] - mean["UMAP+KNN+Leiden"]
    print(f"\n  (a) representation  PCA->UMAP   : {da:+.3f}")
    print(f"  (b) detector        Louvain->Leiden: {db:+.3f}")
    print(f"  (c) weighting       KNN->SNN    : {dc:+.3f}")

    L = [r"% R2.3(a,b,c) controlled single-variable ablation", r"\begin{table}[htbp]", r"\centering",
         r"\caption{Controlled single-variable ablation (mean ARI, 12 datasets, 5 seeds, same budget "
         r"$k{=}35$, $r{=}0.25$). Each row changes exactly one component from the row above, so the "
         r"$\Delta$ isolates that factor.}",
         r"\label{tab:controlled}", r"\footnotesize", r"\begin{tabular}{llc}", r"\toprule",
         r"Configuration & Varied & Mean ARI ($\Delta$) \\", r"\midrule",
         f"PCA + KNN + Louvain & --- & {mean['PCA+KNN+Louvain']:.3f} \\\\",
         f"UMAP + KNN + Louvain & representation & {mean['UMAP+KNN+Louvain']:.3f} ({da:+.3f}) \\\\",
         f"UMAP + KNN + Leiden & detector & {mean['UMAP+KNN+Leiden']:.3f} ({db:+.3f}) \\\\",
         f"UMAP + SNN + Leiden & weighting & {mean['UMAP+SNN+Leiden']:.3f} ({dc:+.3f}) \\\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "controlled_ablation_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"\n[tex] {os.path.join(OUT,'controlled_ablation_table.tex')}")


if __name__ == "__main__":
    main()
