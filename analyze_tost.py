# -*- coding: utf-8 -*-
"""
R2.5(a): proper inference for the DEAC-vs-N2D comparison. 'Failing to reject' is
not equivalence, so we report (i) the paired mean ARI difference with a 95% CI and
(ii) a two-one-sided-tests (TOST) equivalence test at margins Delta=0.02 and 0.05,
over the 12 datasets. Deployable gate and EMR-CGC are both compared to N2D.
Prints a summary and writes paper/generated_canonical/tost_note.tex.
"""
import glob, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper", "generated_canonical")
ORDER = ["Wine", "Glass", "Olivetti", "CNAE-9", "COIL-20", "Optdigits", "MFeat",
         "Segmentation", "ISOLET", "Satimage", "USPS", "PenDigits"]
SEEDS = [1001, 1002, 1003, 1004, 1005]


def canon():
    by = {}
    for f in glob.glob(os.path.join(HERE, "results_canonical", "*_s*.json")):
        d = json.load(open(f)); by.setdefault(d["name"], {})[d["seed"]] = d
    gate = {n: float(np.mean([by[n][s]["ari"][by[n][s]["gate_pick"]] for s in SEEDS])) for n in ORDER}
    emr = {n: float(np.mean([by[n][s]["ari"]["EMR-CGC"] for s in SEEDS])) for n in ORDER}
    return gate, emr


def n2d():
    return {n: json.load(open(os.path.join(HERE, "results_deep", f"{n}.json")))["methods"]["N2D"]["ARI_mean"]
            for n in ORDER}


def tost(diff, margin):
    # H0: |mean diff| >= margin ; equivalence if both one-sided t-tests reject at 0.05
    n = len(diff); m = np.mean(diff); sd = np.std(diff, ddof=1); se = sd / np.sqrt(n)
    t_lo = (m - (-margin)) / se; p_lo = 1 - stats.t.cdf(t_lo, n - 1)   # H0: mean <= -margin
    t_hi = (m - margin) / se; p_hi = stats.t.cdf(t_hi, n - 1)          # H0: mean >= +margin
    return max(p_lo, p_hi)


def analyze(name, series, nd):
    diff = np.array([series[n] - nd[n] for n in ORDER])
    m = float(np.mean(diff)); sd = float(np.std(diff, ddof=1)); se = sd / np.sqrt(len(diff))
    ci = (m - 1.96 * se, m + 1.96 * se)
    w = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
    dz = m / sd  # Cohen's dz (paired)
    out = {"name": name, "mean": m, "ci": ci, "wilcoxon_p": float(w), "dz": dz,
           "tost02": tost(diff, 0.02), "tost05": tost(diff, 0.05)}
    print(f"{name} vs N2D: mean diff={m:+.3f} 95%CI[{ci[0]:+.3f},{ci[1]:+.3f}] "
          f"Wilcoxon p={w:.3f} dz={dz:+.2f}")
    print(f"   TOST equivalence: Delta=0.02 p={out['tost02']:.3f} ({'EQUIV' if out['tost02']<0.05 else 'not equiv'}) | "
          f"Delta=0.05 p={out['tost05']:.3f} ({'EQUIV' if out['tost05']<0.05 else 'not equiv'})")
    return out


def main():
    gate, emr = canon(); nd = n2d()
    os.makedirs(OUT, exist_ok=True)
    g = analyze("Deployable gate", gate, nd)
    e = analyze("EMR-CGC", emr, nd)
    note = (r"% R2.5(a) TOST / effect size vs N2D" + "\n"
            r"\newcommand{\tostnote}{The deployable gate's mean ARI is "
            f"{g['mean']:+.3f} "
            r"relative to N2D (95\% CI $[" + f"{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}" + r"]$; "
            r"Wilcoxon $p=" + f"{g['wilcoxon_p']:.2f}" + r"$, Cohen's $d_z=" + f"{g['dz']:+.2f}" + r"$). "
            r"A two-one-sided-tests equivalence check does \emph{not} establish equivalence even at a "
            r"$\pm0.05$ ARI margin ($p=" + f"{g['tost05']:.2f}" + r"$), so we claim neither superiority "
            r"nor equivalence to N2D---only that the difference is small and not statistically resolved "
            r"at this sample size.}" + "\n")
    open(os.path.join(OUT, "tost_note.tex"), "w").write(note)
    print(f"\n[tex] {os.path.join(OUT,'tost_note.tex')}")


if __name__ == "__main__":
    main()
