# -*- coding: utf-8 -*-
"""R1.7: measure end-to-end wall-clock and peak memory for the full 3-engine DEAC
pipeline on representative datasets (one seed). Writes results_timing/timing.json."""
import json, os, sys, time, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_canonical as rc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_timing")
PROBE = [("Wine", "Wine.csv"), ("MFeat", "mfeat_full.csv"), ("USPS", "USPS.csv")]

try:
    import psutil
    HAVE_PS = True
except Exception:
    HAVE_PS = False


def peak_mb():
    if not HAVE_PS:
        return None
    mi = psutil.Process().memory_info()
    pk = getattr(mi, "peak_wset", None) or mi.rss
    return pk / (1024 * 1024)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for name, csv in PROBE:
        ds = {"name": name, "path": os.path.join(rc.DATA_DIR, csv)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.time()
            res = rc.run_one(name, csv, 1001)
            dt = time.time() - t0
        rows.append({"name": name, "n": res["n"], "seconds": round(dt, 1), "peak_mb": round(peak_mb() or -1, 0)})
        print(f"{name:<10} n={res['n']:<6} {dt:6.1f}s  peak={rows[-1]['peak_mb']} MB", flush=True)
    json.dump(rows, open(os.path.join(OUT, "timing.json"), "w"), indent=2)
    print("saved results_timing/timing.json", flush=True)


if __name__ == "__main__":
    main()
