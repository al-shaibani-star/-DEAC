# -*- coding: utf-8 -*-
"""
Regenerate EVERY manuscript results table from ONE source: the canonical
per-(dataset,seed) files in results_canonical/, plus the comparator results in
results_autoc/ (classical, auto-C) and results_deep/ (deep). Satisfies IJIES
R1.9/R1.4 (one reproducible source) and the Option-2 reframe (the DEPLOYABLE gate
is "DEAC"; the label-selected oracle is a diagnostic upper bound only).

Writes LaTeX to paper/generated_canonical/ with the SAME \\label{} names the
manuscript already \\input, plus canonical_matrix.csv. Prints headline numbers.
"""
import csv, glob, json, os
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper", "generated_canonical")
ORDER = ["Wine", "Glass", "Olivetti", "CNAE-9", "COIL-20", "Optdigits", "MFeat",
         "Segmentation", "ISOLET", "Satimage", "USPS", "PenDigits"]
DISP = {"Segmentation": "Segment."}
ENG = ["Baseline", "AMR-DE", "EMR-CGC"]
CLASS = ["KMeans", "Spectral", "Agglomerative", "GMM", "HDBSCAN"]
DEEP = ["N2D", "DEC", "IDEC", "DCN"]
SEEDS = [1001, 1002, 1003, 1004, 1005]


def disp(n):
    return DISP.get(n, n)


def load_canon():
    by = {}
    for f in glob.glob(os.path.join(HERE, "results_canonical", "*_s*.json")):
        d = json.load(open(f))
        by.setdefault(d["name"], {})[d["seed"]] = d
    return by


def load_cmp(dirn, methods):
    out = {}
    for n in ORDER:
        d = json.load(open(os.path.join(HERE, dirn, f"{n}.json")))
        out[n] = {m: d["methods"][m] for m in methods}
    return out


def eng_mean(by, n, eng, metric):
    return float(np.mean([by[n][s][metric][eng] for s in SEEDS]))


def pick_mean(by, n, which, metric):
    return float(np.mean([by[n][s][metric][by[n][s][which]] for s in SEEDS]))


def pick_C_mean(by, n, which):
    return float(np.mean([by[n][s]["C"][by[n][s][which]] for s in SEEDS]))


def holm(ps):
    k = len(ps); order = sorted(range(k), key=lambda i: ps[i]); out = [0.0] * k; prev = 0.0
    for r, i in enumerate(order):
        prev = max(prev, min(1.0, (k - r) * ps[i])); out[i] = prev
    return out


# ---------- table writers ----------
def w(path, lines):
    open(os.path.join(OUT, path), "w").write("\n".join(lines) + "\n")
    print("  [tex]", path)


def t_main(by):
    L = [r"% canonical; DEAC = deployable gate; Oracle = diagnostic upper bound",
         r"\begin{table}[!t]", r"\centering",
         r"\caption{Main results: mean ARI over 5 seeds (1001--1005), full data. "
         r"\textbf{DEAC} is the \emph{deployable} label-free QualityGate; \emph{Oracle} "
         r"is the label-selected best engine per dataset (a diagnostic upper bound, not "
         r"a deployable result). Bold = best \emph{deployable} value per row.}",
         r"\label{tab:main_results}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Dataset & Baseline & AMR-DE & EMR-CGC & \textbf{DEAC} & \textit{Oracle} \\", r"\midrule"]
    acc = {k: [] for k in ENG + ["gate", "oracle"]}
    for n in ORDER:
        e = {x: eng_mean(by, n, x, "ari") for x in ENG}
        g = pick_mean(by, n, "gate_pick", "ari"); o = pick_mean(by, n, "oracle_pick", "ari")
        for x in ENG: acc[x].append(e[x])
        acc["gate"].append(g); acc["oracle"].append(o)
        depl = list(e.values()) + [g]; best = max(depl)
        def c(v): return f"$\\mathbf{{{v:.3f}}}$" if abs(v - best) < 1e-9 else f"{v:.3f}"
        L.append(f"{disp(n)} & {c(e['Baseline'])} & {c(e['AMR-DE'])} & {c(e['EMR-CGC'])} & {c(g)} & \\textit{{{o:.3f}}} \\\\")
    L.append(r"\midrule")
    m = {k: np.mean(v) for k, v in acc.items()}
    L.append(f"\\textbf{{Mean}} & {m['Baseline']:.3f} & {m['AMR-DE']:.3f} & {m['EMR-CGC']:.3f} & "
             f"\\textbf{{{m['gate']:.3f}}} & \\textit{{{m['oracle']:.3f}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("main_results_table.tex", L)
    return m


def t_multi(by, cl):
    L = [r"% canonical multi-metric means", r"\begin{table}[!t]", r"\centering",
         r"\caption{Mean ARI/NMI/ACC across 12 datasets (5 seeds). DEAC is the deployable "
         r"gate; the oracle row is a diagnostic upper bound.}",
         r"\label{tab:multi_metric}", r"\footnotesize", r"\begin{tabular}{lccc}", r"\toprule",
         r"Method & Mean ARI & Mean NMI & Mean ACC \\", r"\midrule"]
    def row(name, ari, nmi, acc, bold=False):
        f = (lambda v: f"\\textbf{{{v:.3f}}}") if bold else (lambda v: f"{v:.3f}")
        return f"{name} & {f(ari)} & {f(nmi)} & {f(acc)} \\\\"
    for x, lab in [("Baseline", "Baseline (UMAP)"), ("AMR-DE", "AMR-DE"), ("EMR-CGC", "EMR-CGC")]:
        L.append(row(lab, *[np.mean([eng_mean(by, n, x, met) for n in ORDER]) for met in ("ari", "nmi", "acc")]))
    L.append(row(r"\textbf{DEAC (gate)}",
                 *[np.mean([pick_mean(by, n, "gate_pick", met) for n in ORDER]) for met in ("ari", "nmi", "acc")], bold=True))
    L.append(row(r"\textit{Oracle (upper bnd)}",
                 *[np.mean([pick_mean(by, n, "oracle_pick", met) for n in ORDER]) for met in ("ari", "nmi", "acc")]))
    L.append(r"\midrule")
    for m in CLASS:
        L.append(row(m, np.mean([cl[n][m]["ARI_mean"] for n in ORDER]),
                     np.mean([cl[n][m]["NMI_mean"] for n in ORDER]),
                     np.mean([cl[n][m]["ACC_mean"] for n in ORDER])))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("multi_metric_table.tex", L)


def t_cmp(by, cmp_, methods, label, caption, fname):
    L = [f"% canonical: DEAC=deployable gate vs {label}", r"\begin{table}[!t]", r"\centering",
         caption, f"\\label{{{label}}}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "c" * (len(methods) + 1) + r"}", r"\toprule",
         "Dataset & \\textbf{DEAC} & " + " & ".join(methods) + r" \\", r"\midrule"]
    gate_mean = []; mmean = {m: [] for m in methods}; wins = 0
    for n in ORDER:
        g = pick_mean(by, n, "gate_pick", "ari"); gate_mean.append(g)
        vals = {m: cmp_[n][m]["ARI_mean"] for m in methods}
        for m in methods: mmean[m].append(vals[m])
        allv = [g] + list(vals.values()); best = max(allv)
        if g >= best - 1e-9: wins += 1
        def c(v): return f"$\\mathbf{{{v:.3f}}}$" if abs(v - best) < 1e-9 else f"{v:.3f}"
        L.append(f"{disp(n)} & {c(g)} & " + " & ".join(c(vals[m]) for m in methods) + r" \\")
    L.append(r"\midrule")
    gm = np.mean(gate_mean)
    means = {m: np.mean(mmean[m]) for m in methods}
    best = max([gm] + list(means.values()))
    def cm(v): return f"$\\mathbf{{{v:.3f}}}$" if abs(v - best) < 1e-9 else f"{v:.3f}"
    L.append(r"\textbf{Mean} & " + cm(gm) + " & " + " & ".join(cm(means[m]) for m in methods) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w(fname, L)


def t_stats(by, cl, dp):
    gate = np.array([pick_mean(by, n, "gate_pick", "ari") for n in ORDER])
    comps = [(m, "classical", np.array([cl[n][m]["ARI_mean"] for n in ORDER])) for m in CLASS] + \
            [(m, "deep", np.array([dp[n][m]["ARI_mean"] for n in ORDER])) for m in DEEP]
    ps = []
    for _, _, v in comps:
        try: ps.append(float(wilcoxon(gate, v, alternative="two-sided").pvalue))
        except Exception: ps.append(1.0)
    hp = holm(ps)
    L = [r"% canonical gate-vs-comparator Wilcoxon + Holm", r"\begin{table}[htbp]", r"\centering",
         r"\caption{Two-tailed Wilcoxon signed-rank tests of the \emph{deployable} DEAC gate "
         r"vs.\ each comparator over 12 datasets, Holm--Bonferroni corrected across the nine "
         r"tests. $\checkmark$ marks a Holm-significant DEAC advantage.}",
         r"\label{tab:stats}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llccc}", r"\toprule",
         r"Comparison & Type & $p$ & $p_{\text{Holm}}$ & Sig. \\", r"\midrule"]
    for (m, typ, v), p, h in zip(comps, ps, hp):
        sig = r"$\checkmark$" if (h < 0.05 and gate.mean() > v.mean()) else r"$\times$"
        L.append(f"DEAC vs {m} & {typ} & {p:.3f} & {h:.3f} & {sig} \\\\")
        if m == "HDBSCAN": L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("stats_table.tex", L)


def t_percol(by, metric, label, caption, fname):
    L = [f"% canonical per-dataset {metric}", r"\begin{table}[htbp]", r"\centering", caption,
         f"\\label{{{label}}}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Dataset & Baseline & AMR-DE & EMR-CGC & \textbf{DEAC} \\", r"\midrule"]
    for n in ORDER:
        e = {x: eng_mean(by, n, x, metric) for x in ENG}
        g = pick_mean(by, n, "gate_pick", metric)
        best = max(list(e.values()) + [g])
        def c(v): return f"$\\mathbf{{{v:.3f}}}$" if abs(v - best) < 1e-9 else f"{v:.3f}"
        L.append(f"{disp(n)} & {c(e['Baseline'])} & {c(e['AMR-DE'])} & {c(e['EMR-CGC'])} & {c(g)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w(fname, L)


def t_ccount(by):
    L = [r"% canonical cluster counts (deployable gate)", r"\begin{table}[!t]", r"\centering",
         r"\caption{Predicted cluster count (mean over 5 seeds) vs.\ ground truth. DEAC is the "
         r"deployable gate's selected engine; recovering the right granularity without labels is "
         r"itself hard (part of the misalignment). Best (closest to $C_{\text{true}}$) in bold.}",
         r"\label{tab:ccount}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Dataset & $C_{\text{true}}$ & Baseline & AMR-DE & EMR-CGC & \textbf{DEAC} \\", r"\midrule"]
    err = {k: [] for k in ENG + ["gate"]}; exact = {k: 0 for k in ENG + ["gate"]}
    for n in ORDER:
        ct = by[n][SEEDS[0]]["C_true"]
        cm = {x: np.mean([by[n][s]["C"][x] for s in SEEDS]) for x in ENG}
        gc = pick_C_mean(by, n, "gate_pick"); allc = {**cm, "gate": gc}
        for k, v in allc.items():
            err[k].append(abs(v - ct)); exact[k] += (round(v) == ct)
        best = min(abs(v - ct) for v in allc.values())
        def c(v): s = f"{v:.0f}"; return f"\\textbf{{{s}}}" if abs(abs(v - ct) - best) < 1e-9 else s
        L.append(f"{disp(n)} & {ct} & {c(cm['Baseline'])} & {c(cm['AMR-DE'])} & {c(cm['EMR-CGC'])} & {c(gc)} \\\\")
    L.append(r"\midrule")
    mae = {k: np.mean(v) for k, v in err.items()}; nn = len(ORDER)
    L.append(f"\\textbf{{MAE}} & --- & {mae['Baseline']:.2f} & {mae['AMR-DE']:.2f} & {mae['EMR-CGC']:.2f} & {mae['gate']:.2f} \\\\")
    L.append(f"\\textbf{{Exact}} & --- & {exact['Baseline']}/{nn} & {exact['AMR-DE']}/{nn} & {exact['EMR-CGC']}/{nn} & {exact['gate']}/{nn} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("ccount_table.tex", L)


def t_gate(by):
    L = [r"% canonical gate vs oracle", r"\begin{table}[!t]", r"\centering",
         r"\caption{Deployable QualityGate (internal metrics, no labels) vs.\ the oracle upper "
         r"bound, mean ARI over 5 seeds. $\Delta$ = oracle $-$ gate.}",
         r"\label{tab:gate}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lccc}", r"\toprule", r"Dataset & Gate ARI & Oracle ARI & $\Delta$ \\", r"\midrule"]
    G = []; O = []; B = []
    for n in ORDER:
        g = pick_mean(by, n, "gate_pick", "ari"); o = pick_mean(by, n, "oracle_pick", "ari")
        G.append(g); O.append(o); B.append(eng_mean(by, n, "Baseline", "ari"))
        L.append(f"{disp(n)} & {g:.3f} & {o:.3f} & {o-g:+.3f} \\\\")
    L.append(r"\midrule")
    gm, om, bm = np.mean(G), np.mean(O), np.mean(B)
    L.append(f"\\textbf{{Mean}} & \\textbf{{{gm:.3f}}} & {om:.3f} & {om-gm:+.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    w("gate_oracle_table.tex", L)
    return gm, om, bm


def matrix(by):
    with open(os.path.join(OUT, "canonical_matrix.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["dataset", "seed", "engine", "ARI", "NMI", "ACC", "C",
                                         "internal_score", "gate_pick", "oracle_pick"])
        for n in ORDER:
            for s in SEEDS:
                d = by[n][s]
                for e in ENG:
                    wr.writerow([n, s, e, d["ari"][e], d["nmi"][e], d["acc"][e], d["C"][e],
                                 d["scores"][e], d["gate_pick"], d["oracle_pick"]])
    print("  [csv] canonical_matrix.csv")


def main():
    by = load_canon()
    ready = [n for n in ORDER if all(s in by.get(n, {}) for s in SEEDS)]
    if len(ready) < 12:
        print("INCOMPLETE:", [n for n in ORDER if n not in ready], "-- aborting."); return
    cl = load_cmp("results_autoc", CLASS); dp = load_cmp("results_deep", DEEP)
    os.makedirs(OUT, exist_ok=True)
    matrix(by)
    m = t_main(by)
    t_multi(by, cl)
    t_ccount(by)
    gm, om, bm = t_gate(by)
    t_cmp(by, cl, CLASS,
          "tab:benchmarks",
          r"\caption{ARI vs.\ classical methods with automatic cluster-count selection "
          r"(no method is given $C$). Mean over 5 seeds; DEAC is the deployable gate. Bold = best per row.}",
          "autoc_comparison_table.tex")
    t_cmp(by, dp, DEEP,
          "tab:deep",
          r"\caption{ARI vs.\ deep-clustering methods (each \emph{given} the true $C$; DEAC infers it). "
          r"Mean over 5 seeds; DEAC is the deployable gate. Bold = best per row.}",
          "deep_comparison_table.tex")
    t_stats(by, cl, dp)
    t_percol(by, "nmi", "tab:nmi_per_dataset",
             r"\caption{Per-dataset NMI (mean over 5 seeds). DEAC is the deployable gate.}",
             "nmi_table.tex")
    t_percol(by, "acc", "tab:acc_per_dataset",
             r"\caption{Per-dataset ACC (mean over 5 seeds). DEAC is the deployable gate.}",
             "acc_table.tex")
    rec_g = 100 * (gm - bm) / (om - bm) if om > bm else 0
    emr = np.mean([eng_mean(by, n, "EMR-CGC", "ari") for n in ORDER])
    rec_e = 100 * (emr - bm) / (om - bm) if om > bm else 0
    print(f"\nHEADLINE: baseline={bm:.3f} EMR-CGC={emr:.3f} gate={gm:.3f} oracle={om:.3f}")
    print(f"  gate recovers {rec_g:.0f}%, EMR-CGC recovers {rec_e:.0f}% of oracle gain")
    print("All tables written to paper/generated_canonical/")


if __name__ == "__main__":
    main()
