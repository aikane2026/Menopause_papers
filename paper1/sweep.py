"""
Extensive ABM parameter sweep -- designed to be run on a persistent machine
(e.g. the iMac), not the ephemeral sandbox.

Why local: a single run (2000 yr, K=2000) takes ~10-15 s. The grids below are
~hundreds-to-thousands of runs (hours of wall-clock), which exceed per-call
sandbox limits and need a machine that can run unattended and keep results.

Usage:
    python sweep.py --grid coarse   # ~minutes, sanity check
    python sweep.py --grid full     # hours; run on the iMac, ideally overnight
Results are written to sweep_results.csv (one row per run) for later analysis.

Parallelism: uses all CPU cores via multiprocessing. On an Apple-silicon iMac
this scales near-linearly with core count.
"""
import argparse, csv, itertools, os
from multiprocessing import Pool, cpu_count
import numpy as np
from model import Params
from abm import run_abm

# --- currency presets (population growth rate r) ---
CURRENCIES = {"R0": 0.0, "growing": 0.02, "declining": -0.01}
# --- grandmother efficacy-decline presets (a_zero, k) ---
EFFICACY = {"gradual": (90.0, 2.1), "chapman": (76.0, 1.6)}

GRIDS = {
    "coarse": dict(B=[0, 1, 3], rho=[0.06], currency=["R0"],
                   efficacy=["gradual"], seeds=[1, 2], years=1500, K=1500),
    "report": dict(B=[0, 1, 2, 3, 4, 6], rho=[0.045, 0.06, 0.075],
                   currency=["R0"], efficacy=["gradual", "chapman"],
                   seeds=[1, 2, 3], years=1600, K=1500),
    "full":   dict(B=[0, 0.5, 1, 1.5, 2, 3, 4, 6, 8],
                   rho=[0.045, 0.06, 0.075],
                   currency=["R0", "growing", "declining"],
                   efficacy=["gradual", "chapman"],
                   seeds=list(range(1, 11)), years=5000, K=3000),
}


def one_run(job):
    B, rho, currency, eff, seed, years, K = job
    a_zero, k = EFFICACY[eff]
    p = Params(rho=rho, rate=CURRENCIES[currency], gm_a_zero=a_zero, gm_decline_k=k)
    H, s = run_abm(B=B, years=years, K=K, seed=seed, burn_in=years // 2, p=p)
    return dict(B=B, rho=rho, currency=currency, efficacy=eff, seed=seed,
                Tr=round(s["meanTr"], 3), soma=round(s["meanSoma"], 3),
                ppost=round(s["meanPpost"], 3), Td=round(s["Td"], 2),
                l70=round(s["l70"], 3), modal=s["modal"], PRLS=round(s["PRLS"], 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=list(GRIDS), default="coarse")
    ap.add_argument("--out", default="sweep_results.csv")
    ap.add_argument("--procs", type=int, default=cpu_count())
    a = ap.parse_args()
    g = GRIDS[a.grid]
    jobs = [(B, rho, c, e, s, g["years"], g["K"])
            for B, rho, c, e, s in itertools.product(
                g["B"], g["rho"], g["currency"], g["efficacy"], g["seeds"])]
    print(f"{len(jobs)} runs on {a.procs} cores -> {a.out}")
    with Pool(a.procs) as pool:
        rows = pool.map(one_run, jobs)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"done: wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
