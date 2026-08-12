# -*- coding: utf-8 -*-
"""
Modern deep-clustering baselines for the DEAC comparison.

Implements four canonical methods on a shared fully-connected autoencoder
backbone (DEC-standard d-500-500-2000-z architecture):

  * DEC   (Xie et al., ICML 2016)   -- KL self-training on soft assignments
  * IDEC  (Guo et al., IJCAI 2017)  -- DEC + reconstruction regularization
  * N2D   (McConville et al., 2021) -- AE latent -> UMAP -> clustering
  * DCN   (Yang et al., ICML 2017)  -- joint AE + k-means

All methods receive the true cluster count C (oracle), which is the standard
protocol for deep-clustering baselines. They operate on the standardized raw
features X (not the UMAP representation used by the classical baselines).

Returns BenchmarkResult objects compatible with src.evaluation.benchmarks.
"""
import time
import warnings

import numpy as np

from src.evaluation.benchmarks import BenchmarkResult, _evaluate_labels


# ============================================================
# Reproducibility
# ============================================================
def _seed_everything(seed: int):
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Shared Autoencoder (DEC architecture: d-500-500-2000-z)
# ============================================================
def _build_autoencoder(d_in: int, z_dim: int = 10):
    import torch.nn as nn

    dims = [d_in, 500, 500, 2000, z_dim]

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            enc = []
            for i in range(len(dims) - 1):
                enc.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    enc.append(nn.ReLU(True))
            self.encoder = nn.Sequential(*enc)
            dec = []
            rdims = dims[::-1]
            for i in range(len(rdims) - 1):
                dec.append(nn.Linear(rdims[i], rdims[i + 1]))
                if i < len(rdims) - 2:
                    dec.append(nn.ReLU(True))
            self.decoder = nn.Sequential(*dec)

        def forward(self, x):
            z = self.encoder(x)
            xr = self.decoder(z)
            return z, xr

    return AE()


def _pretrain_ae(model, X, seed, epochs=100, batch_size=256, lr=1e-3):
    """End-to-end MSE pretraining of the autoencoder."""
    import torch
    import torch.nn as nn

    dev = _device()
    model = model.to(dev)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    n = Xt.shape[0]
    bs = min(batch_size, n)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    g = torch.Generator(device="cpu").manual_seed(seed)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xt[idx]
            _, xr = model(xb)
            loss = mse(xr, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _encode(model, X):
    import torch
    dev = _device()
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        z, _ = model(Xt)
    return z.cpu().numpy()


# ============================================================
# DEC / IDEC
# ============================================================
def _target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def _dec_core(X, y_true, n_clusters, seed, metric,
              with_recon=False, gamma=0.1,
              pretrain_epochs=100, max_iter=4000, update_interval=140,
              batch_size=256, tol=1e-3):
    """Shared DEC / IDEC training loop. with_recon=True => IDEC."""
    import torch
    import torch.nn as nn
    from sklearn.cluster import KMeans

    _seed_everything(seed)
    dev = _device()
    d_in = X.shape[1]
    z_dim = min(10, max(2, d_in - 1))

    model = _build_autoencoder(d_in, z_dim)
    model = _pretrain_ae(model, X, seed, epochs=pretrain_epochs, batch_size=batch_size)

    # Initialize cluster centers with k-means on the latent space
    Z = _encode(model, X)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    y_pred_last = km.fit_predict(Z)
    centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=dev, requires_grad=True)

    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    n = Xt.shape[0]
    bs = min(batch_size, n)
    params = list(model.encoder.parameters()) + [centers]
    if with_recon:
        params = list(model.parameters()) + [centers]
    opt = torch.optim.Adam(params, lr=1e-3)
    mse = nn.MSELoss()

    def soft_assign(z):
        # student-t, alpha=1
        diff = z.unsqueeze(1) - centers.unsqueeze(0)
        dist2 = (diff ** 2).sum(2)
        q = 1.0 / (1.0 + dist2)
        q = q / q.sum(1, keepdim=True)
        return q

    p_all = None
    for it in range(max_iter):
        if it % update_interval == 0:
            model.eval()
            with torch.no_grad():
                z_all = model.encoder(Xt)
                q_all = soft_assign(z_all)
            p_all = _target_distribution(q_all).detach()
            y_pred = q_all.argmax(1).cpu().numpy()
            delta = np.mean(y_pred != y_pred_last)
            y_pred_last = y_pred
            if it > 0 and delta < tol:
                break
        model.train()
        idx = torch.randint(0, n, (bs,), device=dev)
        xb = Xt[idx]
        z = model.encoder(xb)
        q = soft_assign(z)
        kl = nn.functional.kl_div(q.log(), p_all[idx], reduction="batchmean")
        loss = kl
        if with_recon:
            xr = model.decoder(z)
            loss = kl + gamma * mse(xr, xb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        q_all = soft_assign(model.encoder(Xt))
        labels = q_all.argmax(1).cpu().numpy()
        Z_final = model.encoder(Xt).cpu().numpy()

    ARI, NMI, ACC, SIL, C = _evaluate_labels(Z_final, y_true, labels, metric)
    name = "IDEC" if with_recon else "DEC"
    return ARI, NMI, ACC, SIL, C, name


def run_dec(X, y_true, n_clusters, seed, metric="euclidean", pretrain_epochs=100):
    t0 = time.time()
    try:
        ARI, NMI, ACC, SIL, C, name = _dec_core(
            X, y_true, n_clusters, seed, metric,
            with_recon=False, pretrain_epochs=pretrain_epochs)
    except Exception as e:
        print(f"      [DEC ERROR] {e}")
        ARI = NMI = ACC = SIL = 0.0
        C = 0
    return BenchmarkResult("DEC", ARI, NMI, ACC, SIL, C, time.time() - t0)


def run_idec(X, y_true, n_clusters, seed, metric="euclidean", pretrain_epochs=100):
    t0 = time.time()
    try:
        ARI, NMI, ACC, SIL, C, name = _dec_core(
            X, y_true, n_clusters, seed, metric,
            with_recon=True, pretrain_epochs=pretrain_epochs)
    except Exception as e:
        print(f"      [IDEC ERROR] {e}")
        ARI = NMI = ACC = SIL = 0.0
        C = 0
    return BenchmarkResult("IDEC", ARI, NMI, ACC, SIL, C, time.time() - t0)


# ============================================================
# N2D  (AE latent -> UMAP -> clustering)
# ============================================================
def run_n2d(X, y_true, n_clusters, seed, metric="euclidean", pretrain_epochs=100):
    t0 = time.time()
    try:
        from sklearn.cluster import KMeans
        import umap

        _seed_everything(seed)
        d_in = X.shape[1]
        z_dim = min(10, max(2, d_in - 1))
        model = _build_autoencoder(d_in, z_dim)
        model = _pretrain_ae(model, X, seed, epochs=pretrain_epochs)
        Z = _encode(model, X)

        n_comp = min(n_clusters, Z.shape[1] - 1) if Z.shape[1] > 2 else 2
        n_comp = max(2, n_comp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer = umap.UMAP(n_neighbors=20, min_dist=0.0,
                                n_components=n_comp, random_state=seed)
            Z_umap = reducer.fit_transform(Z)
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        labels = km.fit_predict(Z_umap)
        ARI, NMI, ACC, SIL, C = _evaluate_labels(Z_umap, y_true, labels, metric)
    except Exception as e:
        print(f"      [N2D ERROR] {e}")
        ARI = NMI = ACC = SIL = 0.0
        C = 0
    return BenchmarkResult("N2D", ARI, NMI, ACC, SIL, C, time.time() - t0)


# ============================================================
# DCN  (joint AE + k-means)
# ============================================================
def run_dcn(X, y_true, n_clusters, seed, metric="euclidean",
            pretrain_epochs=100, joint_epochs=50, lam=0.5, batch_size=256):
    t0 = time.time()
    try:
        import torch
        import torch.nn as nn
        from sklearn.cluster import KMeans

        _seed_everything(seed)
        dev = _device()
        d_in = X.shape[1]
        z_dim = min(10, max(2, d_in - 1))
        model = _build_autoencoder(d_in, z_dim)
        model = _pretrain_ae(model, X, seed, epochs=pretrain_epochs, batch_size=batch_size)

        Z = _encode(model, X)
        km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        assign = km.fit_predict(Z)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=dev)

        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        n = Xt.shape[0]
        bs = min(batch_size, n)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        mse = nn.MSELoss()
        g = torch.Generator(device="cpu").manual_seed(seed)

        for _ in range(joint_epochs):
            model.train()
            perm = torch.randperm(n, generator=g).to(dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                xb = Xt[idx]
                z, xr = model(xb)
                target = centers[torch.tensor(assign, device=dev)[idx]]
                loss = mse(xr, xb) + lam * ((z - target) ** 2).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            # update assignments + centers (k-means step) on full latent
            Zf = _encode(model, X)
            km2 = KMeans(n_clusters=n_clusters, n_init=5, random_state=seed)
            assign = km2.fit_predict(Zf)
            centers = torch.tensor(km2.cluster_centers_, dtype=torch.float32, device=dev)

        Zf = _encode(model, X)
        labels = assign
        ARI, NMI, ACC, SIL, C = _evaluate_labels(Zf, y_true, labels, metric)
    except Exception as e:
        print(f"      [DCN ERROR] {e}")
        ARI = NMI = ACC = SIL = 0.0
        C = 0
    return BenchmarkResult("DCN", ARI, NMI, ACC, SIL, C, time.time() - t0)


# ============================================================
# Runner
# ============================================================
DEEP_METHODS = {
    "N2D": run_n2d,
    "DEC": run_dec,
    "IDEC": run_idec,
    "DCN": run_dcn,
}


def run_deep_baselines(X, y_true, n_clusters, seed, methods=None,
                       pretrain_epochs=100, metric="euclidean"):
    """Run selected deep baselines on standardized X with the true cluster count."""
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    k = max(2, int(n_clusters))
    methods = methods or list(DEEP_METHODS.keys())
    results = []
    for m in methods:
        fn = DEEP_METHODS[m]
        r = fn(Xs, y_true, k, seed, metric=metric, pretrain_epochs=pretrain_epochs)
        results.append(r)
        print(f"      {m:5s} ARI={r.ARI:.3f} NMI={r.NMI:.3f} ACC={r.ACC:.3f} "
              f"C={r.C} ({r.time_sec:.1f}s)")
    return results
