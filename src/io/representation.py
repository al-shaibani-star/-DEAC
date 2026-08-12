# -*- coding: utf-8 -*-
"""Dimensionality reduction: PCA (linear) + UMAP (nonlinear).

Two-stage approach:
  1. PCA reduces to intermediate dimension (max 50) for speed
  2. UMAP further reduces to dr_dim, preserving nonlinear structure

When dr_dim >= 15, uses PCA-only (fast, sufficient for many datasets).
When dr_dim < 15 or use_umap=True, applies UMAP after PCA pre-reduction.
"""
from typing import Any

import numpy as np


def make_representation(X: Any, seed: int, dr_dim: int,
                        use_umap: bool = False,
                        rep_type: str = "pca",
                        n_neighbors: int = 15,
                        min_dist: float = 0.1,
                        perplexity: float = 30.0) -> np.ndarray:
    """Apply StandardScaler + dimensionality reduction.

    Parameters
    ----------
    X : array-like
        Input feature matrix.
    seed : int
        Random seed.
    dr_dim : int
        Target dimensionality.
    use_umap : bool
        If True, apply UMAP after PCA pre-reduction.
        Kept for backwards compatibility; overridden by rep_type.
    rep_type : str
        Representation method: "pca", "umap", or "tsne".
        When use_umap=True and rep_type="pca", UMAP is used (compat).
    n_neighbors : int
        UMAP n_neighbors parameter (default 15).
    min_dist : float
        UMAP min_dist parameter (default 0.1).
    perplexity : float
        t-SNE perplexity parameter (default 30.0).

    Returns
    -------
    np.ndarray
        Reduced feature matrix.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    rng_seed = int(seed)
    Xd = np.asarray(X, dtype=np.float32)
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(Xd)

    # Backwards compatibility: use_umap=True promotes rep_type to "umap"
    effective_rep = rep_type
    if use_umap and rep_type == "pca":
        effective_rep = "umap"

    if dr_dim >= Xs.shape[1]:
        if effective_rep == "pca":
            return Xs.astype(np.float32)
        dr_dim = min(dr_dim, Xs.shape[1] - 1)

    pca_dim = max(2, int(dr_dim))

    # ---- PCA ----
    if effective_rep == "pca":
        pca = PCA(n_components=pca_dim, whiten=True, svd_solver="randomized",
                  random_state=rng_seed)
        Z = pca.fit_transform(Xs)
        return Z.astype(np.float32)

    # ---- UMAP ----
    if effective_rep == "umap":
        try:
            import umap

            # PCA pre-reduction for speed
            pca_pre = min(50, Xs.shape[1] - 1)
            if Xs.shape[1] > pca_pre:
                pca = PCA(n_components=pca_pre, svd_solver="randomized",
                          random_state=rng_seed)
                Xs = pca.fit_transform(Xs)

            umap_dim = max(2, min(pca_dim, Xs.shape[1] - 1))
            n_nb = max(2, min(int(n_neighbors), Xs.shape[0] - 1))
            reducer = umap.UMAP(
                n_components=umap_dim,
                n_neighbors=n_nb,
                min_dist=float(min_dist),
                metric="euclidean",
                random_state=rng_seed,
            )
            Z = reducer.fit_transform(Xs)
            return Z.astype(np.float32)

        except ImportError:
            pca = PCA(n_components=pca_dim, whiten=True, svd_solver="randomized",
                      random_state=rng_seed)
            Z = pca.fit_transform(Xs)
            return Z.astype(np.float32)

    # ---- t-SNE ----
    if effective_rep == "tsne":
        try:
            from sklearn.manifold import TSNE

            # PCA pre-reduction for speed
            if Xs.shape[1] > 50:
                pca = PCA(n_components=50, svd_solver="randomized",
                          random_state=rng_seed)
                Xs = pca.fit_transform(Xs)

            tsne_dim = max(2, min(pca_dim, 3))  # t-SNE max 3D
            perp = max(5, min(int(perplexity), Xs.shape[0] // 4))
            reducer = TSNE(
                n_components=tsne_dim, perplexity=perp,
                random_state=rng_seed, n_iter=300, init="pca",
                learning_rate="auto",
            )
            Z = reducer.fit_transform(Xs)
            return Z.astype(np.float32)

        except Exception:
            pca = PCA(n_components=pca_dim, whiten=True, svd_solver="randomized",
                      random_state=rng_seed)
            Z = pca.fit_transform(Xs)
            return Z.astype(np.float32)

    # Fallback: PCA
    pca = PCA(n_components=pca_dim, whiten=True, svd_solver="randomized",
              random_state=rng_seed)
    Z = pca.fit_transform(Xs)
    return Z.astype(np.float32)
