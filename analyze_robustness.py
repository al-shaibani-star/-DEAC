# -*- coding: utf-8 -*-
"""R2.5(c) robustness analysis from results_robustness/. Groups conditions by
stress axis and reports gate ARI (mean over seeds) vs the base, writing
paper/generated_canonical/robustness_table.tex."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper", "generated_canonical")
SEEDS = [1001, 1002, 1003]
AXES = [("Imbalance", ["base", "imbalance x3", "imbalance x10", "imbalance x30"]),
        ("Separation", ["separation 0.5", "separation 0.75", "base", "separation 1.5", "separation 2.0"]),
        ("Irrelevant dims", ["base", "irrelevant d=70", "irrelevant d=220", "irrelevant d=520"]),
        ("Sample size $n$", ["n=1000", "base", "n=5000", "n=10000"])]
SHORT = {"base": "base", "imbalance x3": "3$\\times$", "imbalance x10": "10$\\times$", "imbalance x30": "30$\\times$",
         "separation 0.5": "0.5", "separation 0.75": "0.75", "separation 1.5": "1.5", "separation 2.0": "2.0",
         "irrelevant d=70": "+50", "irrelevant d=220": "+200", "irrelevant d=520": "+500",
         "n=1000": "1000", "n=5000": "5000", "n=10000": "10000"}


def main():
    rows = {}
    for f in glob.glob(os.path.join(HERE, "results_robustness", "*_s*.json")):
        d = json.load(open(f)); rows.setdefault(d["label"], {})[d["seed"]] = d
    def gate(lbl):
        if lbl not in rows: return None
        return float(np.mean([rows[lbl][s]["gate_ari"] for s in rows[lbl]]))
    def base_ari(lbl):
        if lbl not in rows: return None
        return float(np.mean([rows[lbl][s]["ari"]["Baseline"] for s in rows[lbl]]))
    missing = [lbl for _, lvls in AXES for lbl in lvls if lbl not in rows]
    if missing:
        print("INCOMPLETE, missing:", sorted(set(missing))); return
    os.makedirs(OUT, exist_ok=True)
    print("axis: level  gate_ARI (baseline)")
    L = [r"% R2.5(c) robustness", r"\begin{table}[htbp]", r"\centering",
         r"\caption{Robustness of the deployable gate on controlled synthetic data (mean ARI over 3 "
         r"seeds; `base' is $n{=}2000$, $d{=}20$ all-informative, $C{=}10$, balanced, class-sep 1.0). "
         r"The gate is robust to class imbalance, improves with cluster separation and with larger "
         r"$n$, but degrades sharply under many irrelevant dimensions---a limitation noted in "
         r"\S\ref{sec:disc}.}",
         r"\label{tab:robustness}", r"\footnotesize", r"\begin{tabular}{llcc}", r"\toprule",
         r"Stress axis & Level & Gate ARI & Baseline ARI \\", r"\midrule"]
    for ax, lvls in AXES:
        for j, lbl in enumerate(lvls):
            g = gate(lbl); b = base_ari(lbl)
            print(f"  {ax:<16} {lbl:<16} {g:.3f} ({b:.3f})")
            name = ax if j == 0 else ""
            L.append(f"{name} & {SHORT[lbl]} & {g:.3f} & {b:.3f} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "robustness_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"\n[tex] {os.path.join(OUT,'robustness_table.tex')}")


if __name__ == "__main__":
    main()
