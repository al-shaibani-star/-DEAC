# -*- coding: utf-8 -*-
"""
Regenerate the cross-dataset figures from the canonical run (results_canonical/),
replacing the stale ones (which showed impossible ARI>1, a GROUSE legend, and HAR/
Letter datasets -- IJIES R1.1/R1.5). DEAC = deployable gate. All ARI values are in
[0,1] by construction; only the 12 benchmark datasets appear.
Outputs to paper/figures/cross_dataset/ at >=300 DPI.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "paper", "figures", "cross_dataset")
ORDER = ["Wine", "Glass", "Olivetti", "CNAE-9", "COIL-20", "Optdigits", "MFeat",
         "Segmentation", "ISOLET", "Satimage", "USPS", "PenDigits"]
SEEDS = [1001, 1002, 1003, 1004, 1005]
ENG = ["Baseline", "AMR-DE", "EMR-CGC"]
COL = {"Baseline": "#6c757d", "AMR-DE": "#E69F00", "EMR-CGC": "#009E73", "DEAC": "#0065BD"}
DPI = 300


def load():
    by = {}
    for f in glob.glob(os.path.join(HERE, "results_canonical", "*_s*.json")):
        d = json.load(open(f)); by.setdefault(d["name"], {})[d["seed"]] = d
    agg = {}
    for n in ORDER:
        ss = by[n]
        rec = {"n": ss[SEEDS[0]]["n"], "d": ss[SEEDS[0]]["d"], "C_true": ss[SEEDS[0]]["C_true"]}
        for met in ("ari", "nmi", "acc", "C"):
            for e in ENG:
                rec[f"{e}_{met}"] = float(np.mean([ss[s][met][e] for s in SEEDS]))
            rec[f"DEAC_{met}"] = float(np.mean([ss[s][met][ss[s]["gate_pick"]] for s in SEEDS]))
        picks = [ss[s]["gate_pick"] for s in SEEDS]
        rec["gate_pick"] = max(set(picks), key=picks.count)
        agg[n] = rec
    return agg


def fig_characteristics(a):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for n in ORDER:
        ax.scatter(a[n]["n"], a[n]["d"], s=55, color="#0065BD", edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(n, (a[n]["n"], a[n]["d"]), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("samples $n$"); ax.set_ylabel("dimensions $d$")
    ax.set_title("Benchmark datasets ($n$ vs. $d$)", fontsize=10, fontweight="bold")
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)
    save(fig, "cross_dataset_characteristics.png")


def fig_ari_comparison(a):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(ORDER)); w = 0.38
    b = [a[n]["Baseline_ari"] for n in ORDER]; g = [a[n]["DEAC_ari"] for n in ORDER]
    ax.bar(x - w / 2, b, w, label="Baseline", color=COL["Baseline"], edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, g, w, label="DEAC (gate)", color=COL["DEAC"], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("ARI"); ax.set_ylim(0, 1.0)
    ax.set_title("Per-dataset ARI: Baseline vs.\\ DEAC", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    save(fig, "cross_ari_comparison.png")


def fig_qg_summary(a):
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(ORDER))
    vals = [a[n]["DEAC_ari"] for n in ORDER]; cols = [COL[a[n]["gate_pick"]] for n in ORDER]
    ax.bar(x, vals, 0.6, color=cols, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("DEAC (gate) ARI"); ax.set_ylim(0, 1.0)
    ax.set_title("QualityGate activation (bar colour = engine chosen)", fontsize=10, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COL[e], edgecolor="black", label=e) for e in ENG],
              fontsize=8, frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    save(fig, "cross_qualitygate_summary.png")


def fig_improvement(a):
    d = sorted(ORDER, key=lambda n: a[n]["DEAC_ari"] - a[n]["Baseline_ari"], reverse=True)
    delta = [a[n]["DEAC_ari"] - a[n]["Baseline_ari"] for n in d]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    cols = ["#0065BD" if v >= 0 else "#cc3333" for v in delta]
    ax.barh(np.arange(len(d)), delta, color=cols, edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d, fontsize=7.5); ax.invert_yaxis()
    ax.set_xlabel("$\\Delta$ARI  (DEAC $-$ Baseline)")
    ax.set_title("Gate improvement over baseline", fontsize=10, fontweight="bold")
    ax.axvline(0, color="black", lw=0.6); ax.grid(axis="x", ls="--", lw=0.4, alpha=0.4)
    save(fig, "cross_improvement_heatmap.png")


def fig_metric_bars(a):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)
    x = np.arange(len(ORDER)); methods = ENG + ["DEAC"]; w = 0.2
    for ax, met, lab in zip(axes, ("ari", "nmi", "acc"), ("ARI", "NMI", "ACC")):
        for i, m in enumerate(methods):
            vals = [a[n][f"{m}_{met}"] for n in ORDER]
            ax.bar(x + (i - 1.5) * w, vals, w, label=("DEAC (gate)" if m == "DEAC" else m),
                   color=COL[m], edgecolor="black", linewidth=0.3)
        ax.set_ylabel(lab); ax.set_ylim(0, 1.0); ax.grid(axis="y", ls="--", lw=0.4, alpha=0.4)
    axes[0].set_title("Per-dataset ARI / NMI / ACC across the four configurations", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8, frameon=False, ncol=4, loc="upper center")
    axes[-1].set_xticks(x); axes[-1].set_xticklabels(ORDER, rotation=40, ha="right", fontsize=7.5)
    save(fig, "metric_bars_4way.png")


def fig_radar(a):
    # axes: ARI, NMI, ACC, C-acc (=1/(1+|C-Ctrue|/Ctrue)), C-exact-ish (closeness)
    def cacc(m):
        vals = []
        for n in ORDER:
            e = max(1e-9, abs(a[n][f"{m}_C"] - a[n]["C_true"]) / a[n]["C_true"])
            vals.append(1.0 / (1.0 + e))
        return np.mean(vals)
    methods = ["Baseline", "EMR-CGC", "DEAC"]
    axes_lbl = ["ARI", "NMI", "ACC", "C-acc"]
    data = {m: [np.mean([a[n][f"{m}_{met}"] for n in ORDER]) for met in ("ari", "nmi", "acc")] + [cacc(m)]
            for m in methods}
    ang = np.linspace(0, 2 * np.pi, len(axes_lbl), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw=dict(polar=True))
    for m in methods:
        v = data[m] + data[m][:1]
        ax.plot(ang, v, label=("DEAC (gate)" if m == "DEAC" else m), color=COL[m], lw=1.6)
        ax.fill(ang, v, color=COL[m], alpha=0.08)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(axes_lbl, fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_title("Multi-axis comparison", fontsize=10, fontweight="bold", pad=14)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    save(fig, "radar_4way.png")


def save(fig, name):
    os.makedirs(OUTD, exist_ok=True)
    p = os.path.join(OUTD, name)
    fig.tight_layout(); fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print("  [fig]", name)


def main():
    a = load()
    fig_characteristics(a); fig_ari_comparison(a); fig_qg_summary(a)
    fig_improvement(a); fig_metric_bars(a); fig_radar(a)
    print("canonical cross-dataset figures regenerated in", OUTD)


if __name__ == "__main__":
    main()
