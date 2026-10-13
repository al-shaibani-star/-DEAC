# -*- coding: utf-8 -*-
"""
R2.4: run the authors' PRCut (Ghriss & Monteleoni, AISTATS 2025; repo ayghri/prcut
cloned under external/prcut) on our 12 benchmarks, using the SAME UMAP embedding our
other engines use (a fair common-representation comparison), with the true cluster
count (known-C). This imports the authors' exact model (get_prcut_encoder) and loss
(PRCutGradLoss / PRCutBatchLoss) and replicates run_prcut.py's training loop verbatim;
only data loading (our features), batch size (adapted to small n), and metric capture
are ours. Requires a CUDA GPU. Checkpointed/resumable to results_prcut/.
"""
import json, os, sys, time, warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "external", "prcut"))  # authors' package

from src.io.data_loading import load_csv
from src.io.representation import make_representation
from src.io.utils import best_map_acc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph
import torch
from torch import optim
from prcut.models.prcut import get_prcut_encoder          # authors' model
from prcut.losses.prcut import PRCutGradLoss, PRCutBatchLoss  # authors' loss

DATA_DIR = os.path.join(HERE, "datasets")
OUT = os.path.join(HERE, "results_prcut")
DATASETS = [("Wine", "Wine.csv"), ("Glass", "Glass.csv"), ("Olivetti", "olivetti_full.csv"),
            ("CNAE-9", "CNAE9.csv"), ("COIL-20", "COIL20.csv"), ("Optdigits", "Optdigits.csv"),
            ("MFeat", "mfeat_full.csv"), ("Segmentation", "Segmentation.csv"),
            ("ISOLET", "isolet_full.csv"), ("Satimage", "Satimage.csv"),
            ("USPS", "USPS.csv"), ("PenDigits", "PenDigits.csv")]
SEEDS = [1001, 1002, 1003]
CAP = 100000
# authors' defaults (configs/prcut.yaml): beta=100, lr=1e-4, wd=1e-7, gamma=0.8,
# num_layers=1, hidden=512, num_iter=500, eval_interval=50, k=100, batch=2048.
BETA, LR, WD, GAMMA, NLAYER, HID, NUMITER, EVAL = 100.0, 1e-4, 1e-7, 0.8, 1, 512, 300, 50


def run_prcut(Z, y, C, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    n, d = Z.shape
    b = int(min(2048, max(8, n // 2)))         # 2*b <= n for np.random.choice w/o replace
    k = int(min(100, max(5, n // 10)))
    W = kneighbors_graph(Z, n_neighbors=k)      # sparse connectivity graph (as authors do)
    lamb = 2.0 * b
    model = get_prcut_encoder(d, C, NLAYER, HID).cuda()
    op = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    grad_fn = PRCutGradLoss(); loss_fn = PRCutBatchLoss(num_clusters=C, gamma=GAMMA).cuda()
    Zt = torch.tensor(Z, dtype=torch.float32)
    for _ in range(NUMITER):
        model.train()
        for _ in range(EVAL):
            bi = np.random.choice(n, 2 * b, replace=False)
            li, ri = bi[:b], bi[b:]
            P = model(Zt[bi].cuda()); P_l, P_r = P[:b], P[b:]
            loss_fn.update_cluster_p(P); ov_P = loss_fn.clusters_p.data
            bW = torch.tensor(W[li][:, ri].toarray().astype(np.float32)).cuda()
            loss = grad_fn(bW, P_l, P_r, ov_P, lamb) / max(float(bW.sum()), 1.0)
            loss = loss - BETA * torch.special.entr(P.mean(0)).sum()
            op.zero_grad(); loss.backward(); op.step()
    with torch.no_grad():
        lab = torch.argmax(model(Zt.cuda()), dim=1).cpu().numpy()
    return lab


def main():
    os.makedirs(OUT, exist_ok=True)
    if not torch.cuda.is_available():
        print("[FATAL] CUDA GPU required for PRCut."); return
    for name, csv in DATASETS:
        for seed in SEEDS:
            ck = os.path.join(OUT, f"{name}_s{seed}.json")
            if os.path.exists(ck):
                print(f"[cache] {name} s{seed}", flush=True); continue
            try:
                X, y, _ = load_csv(os.path.join(DATA_DIR, csv), label_col=-1)
                if X.shape[0] > CAP:
                    from sklearn.model_selection import train_test_split
                    _, X, _, y = train_test_split(X, y, test_size=CAP, stratify=y, random_state=seed)
                C = int(len(np.unique(y)))
                dr = max(2, min(40, X.shape[1] - 1))
                ts = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Z = make_representation(X, seed=seed, dr_dim=dr, rep_type="umap").astype(np.float32)
                    lab = run_prcut(Z, y, C, seed)
                res = {"name": name, "seed": int(seed), "n": int(X.shape[0]), "C_true": C,
                       "ari": float(adjusted_rand_score(y, lab)),
                       "nmi": float(normalized_mutual_info_score(y, lab)),
                       "acc": float(best_map_acc(y, lab)),
                       "C_found": int(len(np.unique(lab)))}
                json.dump(res, open(ck, "w"), indent=2)
                print(f"[ok] {name:<13} s{seed} ARI={res['ari']:.3f} NMI={res['nmi']:.3f} "
                      f"({time.time()-ts:.0f}s)", flush=True)
            except Exception as e:
                import traceback; print(f"[ERR] {name} s{seed}: {e}"); traceback.print_exc()
            finally:
                import gc
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    print("\nDONE. Checkpoints in results_prcut/", flush=True)


if __name__ == "__main__":
    main()
