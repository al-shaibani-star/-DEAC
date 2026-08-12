# DEAC — Defensive Ensemble Adaptive Clustering

A framework for automatic graph-based clustering of high-dimensional data that
**infers the number of clusters** rather than requiring it as input.

## Overview

DEAC is built around two ideas motivated by a negative result — *optimizing
internal clustering metrics (modularity, silhouette) can degrade external
accuracy (ARI)*:

- **EMR-CGC** — a multi-representation ensemble that runs Shared Nearest
  Neighbor (SNN) graphs with Leiden community detection across five
  representations × twelve configurations and selects the best-scoring
  partition.
- **QualityGate** — a three-layer, size-adaptive arbitrator that selects among
  candidate partitions using internal metrics only (no labels), defaulting to a
  fixed baseline unless an engine scores higher.
- **AMR-DE** — a complementary Differential-Evolution engine for the specific
  dataset regimes where parameter search helps.

The paper reports an **oracle upper bound** (per-dataset best engine) and
separately characterizes the **deployable, label-free gate**, which recovers
~75% of the oracle's gain over the baseline (`Table tab:gate`).

A follow-up investigation into **closing the oracle–deployable gap** — testing
learned, stability, and consensus selectors — is documented in
[`PHASE1_FINDINGS.md`](PHASE1_FINDINGS.md). The gap resists every standard
label-free approach (best is a modest RandomForest, 57%→64%), confirming the
internal–external misalignment is structural rather than a tuning artefact.

## Repository layout

```
src/                 core framework (engines, graph, evaluation, io, viz)
config/              hyperparameters
run_all.py           main pipeline (baseline + AMR-DE + benchmarks)
run_deep_baselines.py   N2D / DEC / IDEC / DCN comparison
run_autoc_baselines.py  fair classical baselines (auto-C selection)
run_snn_vs_knn.py       SNN+Leiden vs KNN+Louvain ablation
run_gate_vs_oracle.py   deployable-gate vs oracle-upper-bound experiment
phase0_diagnostic.py    internal-metric vs ARI diagnostic (oracle-gap study)
phase1_*.py             learned / stability / consensus selectors
PHASE1_FINDINGS.md      write-up of the oracle-gap investigation
build_*.py           result aggregators → paper/generated/*.tex
paper/               DEAC_paper.tex + figures + generated tables
results_*/           experiment outputs (JSON)
```

## Reproducing

1. `pip install -r requirements.txt`
2. Place the 12 benchmark CSVs (last column = label) in `datasets/`.
3. Run the experiment scripts above; each checkpoints to `results_*/`.
4. `cd paper && pdflatex DEAC_paper.tex` (twice).

## Datasets

Twelve public benchmarks (Wine, Glass, Olivetti, CNAE-9, COIL-20, Optdigits,
MFeat, Segmentation, ISOLET, Satimage, USPS, PenDigits), $n \in [178, 10992]$,
$d \in [9, 4096]$. Large CSVs are not tracked; download them from their public
sources (UCI / scikit-learn / OpenML).

## Requirements

Python 3.12, `numpy scipy scikit-learn umap-learn leidenalg igraph hdbscan
networkx matplotlib pandas`; `torch` for the deep baselines.
