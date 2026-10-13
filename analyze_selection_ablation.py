# -*- coding: utf-8 -*-
"""
R2.3 (d,e): selection-rule and gate-layer ablation, computed from the existing
per-engine data in results_sensitivity/ (components + C + ARI, 5 seeds). Compares,
as DEPLOYABLE label-free policies over the three engines:
  * Always-Baseline            (no selection)
  * Always-EMR-CGC             (trivial 'pick the workhorse')
  * Best-internal-score        (gate with the rho guard REMOVED)
  * QualityGate                (rho guard + internal score)   <- full gate
  * QualityGate w/o AMR-DE     (gate over {Baseline, EMR-CGC} only)
  * Oracle                     (label upper bound)
Answers: does the rho guard help? does adding AMR-DE help deployably?
Writes paper/generated_canonical/selection_ablation_table.tex and prints a summary.
"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper", "generated_canonical")
ORDER = ["Wine", "Glass", "Olivetti", "CNAE-9", "COIL-20", "Optdigits", "MFeat",
         "Segmentation", "ISOLET", "Satimage", "USPS", "PenDigits"]
SEEDS = [1001, 1002, 1003, 1004, 1005]
ENG = ["Baseline", "AMR-DE", "EMR-CGC"]
W = (0.40, 0.35, 0.25)


def load():
    by = {}
    for f in glob.glob(os.path.join(HERE, "results_sensitivity", "*_s*.json")):
        d = json.load(open(f)); by.setdefault(d["name"], {})[d["seed"]] = d
    return by


def score(rec, e):
    c = rec["components"][e]
    return W[0] * c["mod01"] + W[1] * c["sil01"] + W[2] * c["bal01"]


def gate_pick(rec, engines, use_rho):
    n = rec["n"]; CB = rec["C"]["Baseline"]; rho = max(4.0, np.sqrt(n / 50.0))
    elig = [e for e in engines if (e == "Baseline") or (not use_rho) or (rec["C"][e] < rho * CB)]
    return max(elig, key=lambda e: score(rec, e))


def policy_ari(by, fn):
    per = []
    for nm in ORDER:
        per.append(np.mean([by[nm][s]["ari"][fn(by[nm][s])] for s in SEEDS]))
    return float(np.mean(per))


def const_ari(by, eng):
    return float(np.mean([np.mean([by[nm][s]["ari"][eng] for s in SEEDS]) for nm in ORDER]))


def main():
    by = load()
    ready = [n for n in ORDER if all(s in by.get(n, {}) for s in SEEDS)]
    if len(ready) < 12:
        print("INCOMPLETE:", [n for n in ORDER if n not in ready]); return
    os.makedirs(OUT, exist_ok=True)
    base = const_ari(by, "Baseline"); emr = const_ari(by, "EMR-CGC")
    orac = float(np.mean([np.mean([max(by[nm][s]["ari"][e] for e in ENG) for s in SEEDS]) for nm in ORDER]))
    rows = [
        ("Always-Baseline (no selection)", base),
        ("Always-EMR-CGC", emr),
        ("Best-internal (no rho guard)", policy_ari(by, lambda r: gate_pick(r, ENG, False))),
        ("QualityGate w/o AMR-DE", policy_ari(by, lambda r: gate_pick(r, ["Baseline", "EMR-CGC"], True))),
        ("\\textbf{QualityGate (full)}", policy_ari(by, lambda r: gate_pick(r, ENG, True))),
        ("\\textit{Oracle (upper bound)}", orac),
    ]
    rec = lambda v: 100 * (v - base) / (orac - base) if orac > base else 0.0
    print(f"baseline={base:.3f} EMR-CGC={emr:.3f} oracle={orac:.3f}\n")
    for name, v in rows:
        print(f"  {name:<34} ARI={v:.3f}  recovery={rec(v):+.0f}%")

    L = [r"% R2.3(d,e) selection-rule / gate-layer ablation", r"\begin{table}[htbp]", r"\centering",
         r"\caption{Deployable selection-rule ablation (mean ARI, 12 datasets, 5 seeds). Removing the "
         r"rho guard (best-internal) and removing AMR-DE from the gate both leave performance essentially "
         r"unchanged; the trivial `always-EMR-CGC' policy slightly exceeds the gate. The gate's role is "
         r"protective (it avoids the optimizer's worst picks), not accuracy-maximizing.}",
         r"\label{tab:selection}", r"\footnotesize", r"\begin{tabular}{lcc}", r"\toprule",
         r"Deployable policy & Mean ARI & Oracle-gain recovery \\", r"\midrule"]
    for name, v in rows:
        L.append(f"{name} & {v:.3f} & {rec(v):+.0f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "selection_ablation_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"\n[tex] {os.path.join(OUT,'selection_ablation_table.tex')}")


if __name__ == "__main__":
    main()
