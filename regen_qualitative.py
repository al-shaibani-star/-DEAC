# -*- coding: utf-8 -*-
"""
Regenerate the qualitative figures (t-SNE + silhouette) using the REAL DEAC
(EMR-CGC engine) cluster labels. The previous images were recycled from an old
"Hybrid" AMR-DE run (title "Hybrid, C=6" on MFeat, contradicting DEAC's ARI
0.937 with ~10 clusters). For each of the four datasets shown in the paper we
run EMR-CGC, colour a t-SNE projection by its labels, and draw the silhouette.

Overwrites figures/<DS>/tsne_<DS>.png and silhouette_<DS>.png in place.
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.io.data_loading import load_csv
from src.core.emr_cgc import run_emr_cgc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figures")

# (paper name -> (csv, figdir))
DATASETS = [
    ("MFeat", "mfeat_full.csv", "MFeat"),
    ("COIL-20", "COIL20.csv", "COIL20"),
    ("Olivetti", "olivetti_full.csv", "Olivetti"),
    ("PenDigits", "PenDigits.csv", "PenDigits"),
]
SEED = 1001
MAXVIZ = 2000


def tsne_2d(X, seed):
    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] > 50:
        Xs = PCA(n_components=50, svd_solver="randomized", random_state=seed).fit_transform(Xs)
    ts = TSNE(n_components=2, perplexity=min(30, max(5, Xs.shape[0] // 4)),
              random_state=seed, init="pca", learning_rate="auto")
    return ts.fit_transform(Xs)


def plot_tsne(emb, labels, name, out):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    C = len(np.unique(labels))
    cmap = cm.get_cmap("tab20", max(C, 2))
    for ci, c in enumerate(np.unique(labels)):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=8, color=cmap(ci), alpha=0.75, linewidths=0)
    ax.set_title(f"{name} — DEAC clusters (C={C})", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def plot_silhouette(emb, labels, name, out):
    C = len(np.unique(labels))
    sv = silhouette_samples(emb, labels)
    avg = silhouette_score(emb, labels)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cmap = cm.get_cmap("tab20", max(C, 2))
    y_low = 0
    for ci, c in enumerate(sorted(np.unique(labels))):
        vals = np.sort(sv[labels == c])
        y_up = y_low + len(vals)
        ax.fill_betweenx(np.arange(y_low, y_up), 0, vals, color=cmap(ci), alpha=0.8)
        ax.text(-0.05, y_low + len(vals) / 2, f"C{ci}", fontsize=8, va="center", ha="right")
        y_low = y_up + 10
    ax.axvline(avg, color="red", ls="--", lw=1.5, label=f"Avg={avg:.3f}")
    ax.set_xlabel("Silhouette Coefficient"); ax.set_ylabel("Cluster")
    ax.set_title(f"{name} — Silhouette Analysis (DEAC, C={C})", fontweight="bold")
    ax.set_yticks([]); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def main():
    summary = []
    for name, csv, fdir in DATASETS:
        X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
        rng = np.random.RandomState(SEED)
        if X.shape[0] > MAXVIZ:
            from sklearn.model_selection import train_test_split
            _, Xv, _, yv = train_test_split(X, y, test_size=MAXVIZ, stratify=y,
                                            random_state=SEED)
        else:
            Xv, yv = X, y
        print(f"[{name}] n_viz={Xv.shape[0]} running EMR-CGC...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = run_emr_cgc(Xv, yv, seed=SEED, metric="cosine", subsample=0)
            y_deac = res.y_consensus
            ari = adjusted_rand_score(yv, y_deac)
            emb = tsne_2d(Xv, SEED)
        out_dir = os.path.join(FIG_DIR, fdir)
        os.makedirs(out_dir, exist_ok=True)
        plot_tsne(emb, y_deac, name, os.path.join(out_dir, f"tsne_{fdir}.png"))
        plot_silhouette(emb, y_deac, name, os.path.join(out_dir, f"silhouette_{fdir}.png"))
        avg = silhouette_score(emb, y_deac)
        summary.append((name, res.C, ari, avg))
        print(f"   DEAC C={res.C}  ARI={ari:.3f}  silhouette(tSNE)={avg:.3f}  saved.")
    print("\nSUMMARY (name, C, ARI, avg-silhouette):")
    for s in summary:
        print(f"  {s[0]:<11} C={s[1]:<3} ARI={s[2]:.3f} sil={s[3]:.3f}")


if __name__ == "__main__":
    main()
