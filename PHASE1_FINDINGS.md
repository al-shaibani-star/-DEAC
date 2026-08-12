# Closing the Oracle–Deployable QualityGate Gap — Investigation Findings

> Merged to `main` via [PR #1](https://github.com/al-shaibani-star/-DEAC/pull/1) in commit [`f43ec7f`](https://github.com/al-shaibani-star/-DEAC/commit/f43ec7f19b4e9e53e306903749cb5b9d76a579ce).

## Problem

The paper reports DEAC under two selection regimes:

- **Oracle** — pick the per-dataset best-ARI engine (uses ground-truth labels). Upper bound.
- **Deployable QualityGate** — pick by internal metrics only, no labels.

On the full data the deployable gate recovers **~75%** of the oracle's ARI gain
over the baseline and agrees with the oracle's engine choice on **6/12** datasets
(`paper/generated/gate_oracle_table.tex`). The residual gap is the paper's stated
open problem. This investigation asks: **can a smarter label-free selector close it?**

## Phase 0 — Which internal signal predicts the best engine?

`phase0_diagnostic.py` computes every internal metric per engine vs. ARI across
the three engines on 12 datasets.

| Metric | picks the oracle engine | mean Spearman with ARI |
|---|---|---|
| MOD | 5/12 | 0.21 |
| CH  | 5/12 | 0.25 |
| BAL | 5/12 | 0.17 |
| SIL | 4/12 | **0.08** |
| DB  | 4/12 | 0.08 |
| combined (current gate) | 5/12 | 0.25 |

**Finding.** No single internal metric reliably identifies the best engine.
Silhouette — weighted **0.35** in the current gate — barely correlates with ARI
(Spearman 0.08). A grid search over (MOD, SIL, BAL) weights reached 8/12 **on the
fitting set**, but this does not generalise (see Phase 1).

## Phase 1 — Learned and structural selectors

All selectors are label-free at deployment; learned ones use Leave-One-Dataset-Out
(no leakage). ARI is used only for evaluation.

| Selector | script | agree | recovery of oracle gain |
|---|---|---|---|
| Current gate (fixed weights) | — | 5/12 | 57% |
| Learned linear weights (D) | `phase1_selectors.py` | 5/12 | 57% |
| **Learned RandomForest (C)** | `phase1_selectors.py` | **6/12** | **64%** |
| Stability / bootstrap (A) | `phase1_stability.py` | 5/12 | 26% |
| Consensus — agreement selector (B1) | `phase1_consensus.py` | 5/12 | 46% |
| Consensus — co-association partition (B2) | `phase1_consensus.py` | — | 52% |
| Oracle (upper bound) | — | 12/12 | 100% |

*(Recovery percentages are computed per experiment; absolute ARIs vary slightly
with subsampling protocol — the gap, not the level, is the quantity of interest.)*

**Findings.**

1. **Re-weighting the same metrics does not close the gap.** Under honest LODO the
   learned linear selector equals the current gate (57%); RandomForest gives a
   modest bump (64%). The Phase-0 8/12 optimum was overfitting.
2. **Stability fails (26%).** The most *reproducible* clustering is not the most
   *accurate* — a coarse partition can be very stable yet low-ARI, so stability
   pulls toward stable-but-inaccurate engines.
3. **Consensus is weak (46–52%).** Cross-engine agreement and a co-association
   consensus partition both fall below the current gate; the ensemble sometimes
   agrees on the *wrong* structure.

## Conclusion

Across the standard families of label-free selection — metric re-weighting,
meta-learning, stability, and consensus — **the oracle–deployable gap resists all
of them**, with only a modest RandomForest gain (57% → 64%). This is a **negative
result**, and a scientifically useful one: it confirms empirically that the
internal–external misalignment at the heart of the paper is **deep and structural**,
not a tuning artefact. The deployable QualityGate is close to the best achievable
from internal signals alone; closing the remaining gap plausibly requires
**supervision or a learned representation of external quality**, which we leave as
future work.

## Reproduce

```bash
python phase0_diagnostic.py     # metric diagnostic  -> results_phase0/
python phase1_selectors.py      # linear + RF (LODO)
python phase1_stability.py      # stability          -> results_stability/
python phase1_consensus.py      # consensus B1/B2    -> results_consensus/
```
