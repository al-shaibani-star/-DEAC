# Algorithm: Fast AMR-DE (Adaptive Multi-Representation DE)

---

## Algorithm 1: Fast AMR-DE — Main

```
Input:  Data matrix X in R^(n x d), seeds S
Output: Cluster labels y_hat in {1,...,C}

 1  FOR each seed s in S DO
 2  |
 3  |   // Baseline with UMAP default params
 4  |   y_B = GraphCluster(X, UMAP(default), k=35, q=0.10, r=0.25)
 5  |
 6  |   // Stage 1: Quick representation scan
 7  |   best_rep, Z_best = RepScan(X, s)                          [Alg.2]
 8  |
 9  |   // Stage 2: Multi-restart DE on fixed Z_best
10  |   theta* = MultiRestartDE(Z_best, s)                         [Alg.3]
11  |
12  |   // Evaluate with optimized params on best representation
13  |   y_H = GraphCluster(X, best_rep, theta*)
14  |
15  |   // QualityGate: guarantee Hybrid >= Baseline
16  |   IF QualityGate(y_B, y_H, n) THEN
17  |       y_s = y_B                                // protected
18  |   ELSE
19  |       y_s = y_H                                // improved
20  |   END IF
21  |
22  END FOR
23
24  RETURN best y_s across seeds
```

---

## Algorithm 2: RepScan — Representation Selection

```
Input:  Data X, seed s
Output: Best representation type, pre-computed Z_best

 1  candidates = [
 2      PCA(d=40),
 3      UMAP(n_neighbors=15, min_dist=0.1),
 4      UMAP(n_neighbors=10, min_dist=0.0),
 5      UMAP(n_neighbors=30, min_dist=0.2),
 6      t-SNE(perplexity=30)
 7  ]
 8
 9  best_score = -inf
10  FOR each (rep_type, params) in candidates DO
11  |   Z = ComputeRepresentation(X, rep_type, params)
12  |   G = BuildGraph(Z, k=35, q=0.10)
13  |   y = Louvain(G, r=0.25)
14  |   score = 0.30*MOD(y) + 0.40*SIL(y) + 0.30*CH(y)
15  |   IF score > best_score THEN
16  |       best_score = score
17  |       best_rep = rep_type
18  |       Z_best = Z
19  |   END IF
20  END FOR
21
22  RETURN best_rep, Z_best
```

---

## Algorithm 3: MultiRestartDE — Graph Parameter Optimization

```
Input:  Fixed representation Z, seed s
Output: Optimized theta* = (k, d', q, r)

 1  R = 2                                      // restarts
 2  G = 8, P = 5 x 4 = 20                      // DE params
 3  bounds = [(5,80), (5,100), (0,0.40), (0.01,0.90)]
 4
 5  // Pre-compute KNN once (reused across all evaluations)
 6  KNN_all = ComputeKNN(Z, k_max=140)
 7
 8  global_best = null
 9  FOR restart = 1 TO R DO
10  |
11  |   // Latin Hypercube Sampling initialization
12  |   pop = LHS(P, bounds, seed + restart*1000)
13  |
14  |   // Differential Evolution
15  |   FOR g = 1 TO G DO
16  |   |   FOR i = 1 TO P DO
17  |   |       v = mutation(pop, F)
18  |   |       u = crossover(pop[i], v, CR)
19  |   |       IF fitness(u, Z, KNN_all) < fitness(pop[i]) THEN
20  |   |           pop[i] = u
21  |   |       END IF
22  |   |   END FOR
23  |   END FOR
24  |
25  |   // Nelder-Mead refinement
26  |   best_restart = NelderMead(best_of_pop, Z, KNN_all)
27  |
28  |   IF best_restart < global_best THEN
29  |       global_best = best_restart
30  |   END IF
31  |
32  END FOR
33
34  RETURN global_best.params
```

---

## Key Innovation: Two-Stage Separation

```
WRONG (slow, 800s/seed):
  DE evaluates fitness → each eval computes UMAP → 2-5s per eval
  Total: 200 evals x 3s = 600s

RIGHT (fast, 130s/seed):
  Stage 1: Scan 5 representations once → 30s
  Stage 2: DE on fixed Z with pre-computed KNN → 100s
  Total: 130s (6x faster)

WHY it works:
  - UMAP quality depends on n_neighbors, min_dist (scanned in Stage 1)
  - Graph quality depends on k, q, r (optimized in Stage 2)
  - These are SEPARABLE — optimizing them jointly wastes time
```
