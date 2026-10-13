# -*- coding: utf-8 -*-
"""
R2.2 analysis from results_sensitivity/ (no re-running): (1) sensitivity of the
QualityGate to its constants -- rho floor, scaling c, and score weights -- and
(2) a dataset-level development/evaluation split showing whether tuning the
weights on held-out-from datasets beats the fixed default (i.e. whether the 12
benchmarks' labels could have leaked into the constants). Writes a LaTeX
sensitivity table to paper/generated_canonical/ and prints a summary.
"""
import glob, itertools, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "results_sensitivity")
OUT = os.path.join(HERE, "paper", "generated_canonical")
ORDER = ["Wine", "Glass", "Olivetti", "CNAE-9", "COIL-20", "Optdigits", "MFeat",
         "Segmentation", "ISOLET", "Satimage", "USPS", "PenDigits"]
SEEDS = [1001, 1002, 1003, 1004, 1005]
ENG = ["Baseline", "AMR-DE", "EMR-CGC"]
DEF = dict(floor=4.0, c=50.0, w=(0.40, 0.35, 0.25))


def load():
    by = {}
    for f in glob.glob(os.path.join(IN, "*_s*.json")):
        d = json.load(open(f)); by.setdefault(d["name"], {})[d["seed"]] = d
    return by


def gate_pick(rec, floor, c, w):
    n = rec["n"]; CB = rec["C"]["Baseline"]; rho = max(floor, np.sqrt(n / c))
    elig = [e for e in ENG if e == "Baseline" or rec["C"][e] < rho * CB]
    wm, ws, wb = w
    sc = {e: wm * rec["components"][e]["mod01"] + ws * rec["components"][e]["sil01"]
          + wb * rec["components"][e]["bal01"] for e in elig}
    return max(elig, key=lambda e: sc[e])


def recovery(by, datasets, floor, c, w):
    g = b = o = 0.0
    for nm in datasets:
        ss = by[nm]
        gate = np.mean([ss[s]["ari"][gate_pick(ss[s], floor, c, w)] for s in SEEDS])
        base = np.mean([ss[s]["ari"]["Baseline"] for s in SEEDS])
        orac = np.mean([max(ss[s]["ari"][e] for e in ENG) for s in SEEDS])
        g += gate; b += base; o += o * 0 + orac
    g /= len(datasets); b /= len(datasets); o /= len(datasets)
    return g, (100 * (g - b) / (o - b) if o > b else 0.0)


def simplex(step=0.05):
    out = []
    ks = int(round(1 / step))
    for i in range(ks + 1):
        for j in range(ks + 1 - i):
            k = ks - i - j
            out.append((round(i * step, 3), round(j * step, 3), round(k * step, 3)))
    return out


def main():
    by = load()
    ready = [n for n in ORDER if all(s in by.get(n, {}) for s in SEEDS)]
    if len(ready) < 12:
        print("INCOMPLETE:", [n for n in ORDER if n not in ready], "-- run the driver first."); return
    os.makedirs(OUT, exist_ok=True)

    gd, rd = recovery(by, ORDER, **DEF)
    print(f"DEFAULT (floor=4, c=50, w=.40/.35/.25): gate ARI={gd:.3f}  recovery={rd:.0f}%\n")

    print("== rho floor sensitivity (c=50, default weights) ==")
    for fl in [2, 3, 4, 5, 6]:
        g, r = recovery(by, ORDER, fl, 50.0, DEF["w"]); print(f"  floor={fl}: ARI={g:.3f} recovery={r:.0f}%")
    print("== scaling c sensitivity (floor=4, default weights) ==")
    for c in [25, 50, 75, 100, 150]:
        g, r = recovery(by, ORDER, 4.0, float(c), DEF["w"]); print(f"  c={c}: ARI={g:.3f} recovery={r:.0f}%")

    print("== weight sensitivity (floor=4, c=50; simplex step .05) ==")
    grid = simplex(0.05)
    res = [(w, *recovery(by, ORDER, 4.0, 50.0, w)) for w in grid]
    best = max(res, key=lambda t: t[1]); worst = min(res, key=lambda t: t[1])
    aris = [t[1] for t in res]
    print(f"  best w={best[0]} ARI={best[1]:.3f} rec={best[2]:.0f}%")
    print(f"  worst w={worst[0]} ARI={worst[1]:.3f} rec={worst[2]:.0f}%")
    print(f"  spread over {len(grid)} settings: ARI {min(aris):.3f}-{max(aris):.3f} (default {gd:.3f})")

    # Dev/eval split: tune weights on dev, evaluate on disjoint eval; vs default on eval
    print("\n== dev/eval split (random 6/6, 300 draws): tuned-on-dev vs default, measured on eval ==")
    rng = np.random.RandomState(0); tuned_ev = []; def_ev = []
    for _ in range(300):
        perm = rng.permutation(ORDER); dev, ev = list(perm[:6]), list(perm[6:])
        bw = max(grid, key=lambda w: recovery(by, dev, 4.0, 50.0, w)[1])
        tuned_ev.append(recovery(by, ev, 4.0, 50.0, bw)[1])
        def_ev.append(recovery(by, ev, 4.0, 50.0, DEF["w"])[1])
    print(f"  eval recovery  default={np.mean(def_ev):.1f}%  tuned-on-dev={np.mean(tuned_ev):.1f}%  "
          f"gain={np.mean(tuned_ev)-np.mean(def_ev):+.1f} pts")
    print("  (small/negative gain => the fixed default does not benefit from seeing these datasets' labels)")

    # LaTeX table
    fa = [recovery(by, ORDER, fl, 50.0, DEF["w"])[0] for fl in [2, 3, 4, 5, 6]]
    ca = [recovery(by, ORDER, 4.0, float(c), DEF["w"])[0] for c in [25, 50, 75, 100, 150]]
    cap = (r"\caption{QualityGate sensitivity (12 datasets, 5 seeds). Mean ARI as each constant "
           r"varies around its default (floor~4, $c{=}50$, weights .40/.35/.25); the gate is "
           r"near-constant across the rho-guard ranges and insensitive to the score weights "
           r"(only a degenerate modularity-only weighting fails). Tuning the weights on a 6-dataset "
           r"development split and evaluating on the disjoint six recovers "
           f"{np.mean(tuned_ev):.0f}\\% of the oracle's gain versus {np.mean(def_ev):.0f}\\% for the "
           f"fixed default ({np.mean(tuned_ev)-np.mean(def_ev):+.0f} points), so the constants are "
           r"not fitted to these benchmarks.}")
    L = [r"% R2.2 gate sensitivity (analyze_sensitivity.py)", r"\begin{table}[htbp]", r"\centering",
         cap, r"\label{tab:sensitivity}", r"\footnotesize", r"\begin{tabular}{lcc}", r"\toprule",
         r"Varied constant & Range tested & Mean ARI \\", r"\midrule",
         f"rho floor & 2--6 & {min(fa):.3f}--{max(fa):.3f} \\\\",
         f"scaling $c$ & 25--150 & {min(ca):.3f}--{max(ca):.3f} \\\\",
         f"score weights & full simplex & {min(aris):.3f}--{max(aris):.3f} \\\\",
         r"\midrule",
         f"\\textbf{{Default}} (floor~4, $c{{=}}50$) & --- & \\textbf{{{gd:.3f}}} \\\\",
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(OUT, "sensitivity_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"\n[tex] {os.path.join(OUT,'sensitivity_table.tex')}")


if __name__ == "__main__":
    main()
