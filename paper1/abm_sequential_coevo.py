"""
abm_sequential_coevo.py  —  Long-run ABM for Section 4.3 (sequential co-evolution test).

QUESTION: If maintenance evolves first (δ_rep reduced), does the improved maintenance
level unlock earlier cessation in a second step?

OC prediction (already computed): NO — Q1 fitness gain barely changes with δ_rep
(1.061% at baseline → 1.011% at δ_rep × 0.1; T* stays at 40 yr throughout).

ABM test: does Tr advance when δ_rep is reduced?

CONDITIONS:
  1. Baseline              — δ_rep = 0.006, B = 1, B_pgm = 0
  2. Improved maint ×0.2   — δ_rep = 0.0012, B = 1, B_pgm = 0
  3. Improved maint ×0.1   — δ_rep = 0.0006, B = 1, B_pgm = 0
  4. Two-GMs + maint ×0.2  — δ_rep = 0.0012, B = 1, B_pgm = 1, p_u = 0.05

Run (≈ 4 h total on iMac):
    ABM_YEARS=100000 ABM_K=10000 ABM_SEEDS=5 python3 abm_sequential_coevo.py

Quick preview (≈ 20 min):
    ABM_YEARS=10000  ABM_K=3000  ABM_SEEDS=3 python3 abm_sequential_coevo.py

Output: results_abm_sequential_coevo.json  (same schema as results_abm_newrows.json)

Expected result: Td improves across conditions (confirming maintenance evolves), but
Tr stays near ceiling in all conditions (confirming the co-evolutionary constraint is
permanent and sequential evolution cannot bypass it).
"""

import os, json
import numpy as np
from abm import run_abm
from model import Params

YEARS = int(os.environ.get("ABM_YEARS", 10000))
K     = int(os.environ.get("ABM_K",      3000))
NSEED = int(os.environ.get("ABM_SEEDS",  3))
seeds = list(range(1, NSEED + 1))

print(f"Settings: years={YEARS:,}  K={K:,}  seeds={seeds}")
print()

BASE = dict(gm_a_zero=90, mm_scale=1, use_prod_curve=True)
BASELINE_DREP = 0.006

CONDITIONS = [
    dict(label="Baseline (δ_rep=0.006, B=1, B_pgm=0)",
         delta_rep=BASELINE_DREP,       B=1.0, B_pgm=0.0, p_u=0.05),
    dict(label="Improved maint ×0.2 (δ_rep=0.0012, B=1, B_pgm=0)",
         delta_rep=BASELINE_DREP * 0.2, B=1.0, B_pgm=0.0, p_u=0.05),
    dict(label="Improved maint ×0.1 (δ_rep=0.0006, B=1, B_pgm=0)",
         delta_rep=BASELINE_DREP * 0.1, B=1.0, B_pgm=0.0, p_u=0.05),
    dict(label="Two-GMs + maint ×0.2 (δ_rep=0.0012, B=1, B_pgm=1)",
         delta_rep=BASELINE_DREP * 0.2, B=1.0, B_pgm=1.0, p_u=0.05),
]

agg = {}
for cond in CONDITIONS:
    lbl   = cond["label"]
    drep  = cond["delta_rep"]
    B     = cond["B"]
    Bpgm  = cond["B_pgm"]
    pu    = cond["p_u"]

    params = Params(**BASE, delta_rep=drep)
    Trs, Tds, l70s = [], [], []

    for sd in seeds:
        print(f"  {lbl[:55]} seed={sd} ...", flush=True)
        H, s = run_abm(B=B, B_pgm=Bpgm, p_u=pu,
                       years=YEARS, K=K, seed=sd,
                       burn_in=YEARS // 2, p=params)
        Trs.append(s["meanTr"])
        Tds.append(s["Td"])
        l70s.append(s["l70"])

    key = f"drep{drep:.4f}_B{B}_pgm{Bpgm}"
    agg[key] = dict(
        label   = lbl,
        delta_rep = drep,
        B       = B, B_pgm = Bpgm, p_u = pu,
        Tr      = round(float(np.mean(Trs)), 2),
        Tr_sd   = round(float(np.std(Trs)),  2),
        Td      = round(float(np.mean(Tds)), 1),
        l70     = round(float(np.mean(l70s)), 2),
        PRLS    = round(float(np.mean(Tds)) - float(np.mean(Trs)), 1),
    )
    a = agg[key]
    print(f"  → Tr={a['Tr']:.2f}(±{a['Tr_sd']:.2f})  "
          f"Td={a['Td']:.1f}  PRLS={a['PRLS']:.1f}  l70={a['l70']:.2f}")
    print()

with open("results_abm_sequential_coevo.json", "w") as f:
    json.dump(agg, f, indent=2)
print("Saved results_abm_sequential_coevo.json")

print()
print(f"{'Condition':<55}  {'Tr':>10}  {'Td':>6}  {'PRLS':>6}")
print("-" * 80)
for v in agg.values():
    print(f"{v['label']:<55}  "
          f"{v['Tr']:>6.2f}±{v['Tr_sd']:.2f}  "
          f"{v['Td']:>6.1f}  "
          f"{v['PRLS']:>6.1f}")
