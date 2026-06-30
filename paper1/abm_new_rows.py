"""
abm_new_rows.py  —  Long-run ABM for Table 1 rows 8 and 9.

NEW CONDITIONS (both use full biology: age-graded productivity,
effort-dependent Kirkwood damage, declining grandmother efficacy):

  Row 8: B_eff = 0.5, B_PGM = 0  (grandchildren competition N_d = 2;
          empirical B = 1 per daughter diluted by half)

  Row 9: B_eff = 2.0, B_PGM = 0  (undiluted focused MGM; no PGM;
          empirical B = 1 reflects N_d ≈ 2 dilution)

For direct comparison, the existing two-grandmother result is also rerun:
  Row 7: B = 1, B_PGM = 1  (two grandmothers, as in results_abm.json)

Run on the iMac (slow machine OK — each seed takes ~30 min at K=10,000):

    ABM_YEARS=100000 ABM_K=10000 ABM_SEEDS=5 python3 abm_new_rows.py

Or for a quick preview (~5 min total):

    ABM_YEARS=10000 ABM_K=3000 ABM_SEEDS=3 python3 abm_new_rows.py

Output: results_abm_newrows.json  (same schema as results_abm.json)
"""

import os
import json
import numpy as np
from abm import run_abm
from model import Params

# ── Run settings (override via environment variables) ───────────────────────
YEARS = int(os.environ.get("ABM_YEARS", 10000))
K     = int(os.environ.get("ABM_K",      3000))
NSEED = int(os.environ.get("ABM_SEEDS",     3))
seeds = list(range(1, NSEED + 1))

print(f"Settings: years={YEARS:,}  K={K:,}  seeds={seeds}")
print()

# ── Full biology (matching the existing long-run conditions) ────────────────
FULL = dict(gm_a_zero=90, mm_scale=1, use_prod_curve=True, delta_rep=0.006)
params = Params(**FULL)

# ── Conditions ───────────────────────────────────────────────────────────────
CONDITIONS = [
    dict(B=0.5, B_pgm=0.0, p_u=0.05,
         label="B=0.5 MGM only (competition N_d=2, row 8)"),
    dict(B=2.0, B_pgm=0.0, p_u=0.05,
         label="B=2.0 MGM only undiluted (row 9)"),
    # Rerun row 7 for direct comparison in same output file
    dict(B=1.0, B_pgm=1.0, p_u=0.05,
         label="B=1.0 MGM + B=1.0 PGM p_u=0.05 (row 7, comparison)"),
]

# ── Run ──────────────────────────────────────────────────────────────────────
agg = {}
for cond in CONDITIONS:
    lbl  = cond["label"]
    Bv   = cond["B"]
    Bpgm = cond["B_pgm"]
    pu   = cond["p_u"]

    Trs=[]; somas=[]; pposts=[]; Tds=[]; l70s=[]; modals=[]; trajTr=[]; yrs=None
    for sd in seeds:
        print(f"  {lbl[:52]} seed={sd} ...", flush=True)
        H, s = run_abm(B=Bv, B_pgm=Bpgm, p_u=pu,
                       years=YEARS, K=K, seed=sd,
                       burn_in=YEARS // 2, p=params)
        Trs.append(s["meanTr"])
        somas.append(s["meanSoma"])
        pposts.append(s["meanPpost"])
        Tds.append(s["Td"])
        l70s.append(s["l70"])
        modals.append(s["modal"])
        trajTr.append(H["meanTr"])
        yrs = H["year"]

    n = min(len(t) for t in trajTr)
    key = f"B{Bv}_pgm{Bpgm}_pu{pu}"
    agg[key] = dict(
        B=Bv, B_pgm=Bpgm, p_u=pu, label=lbl,
        Tr      = float(np.mean(Trs)),
        Tr_sd   = float(np.std(Trs)),
        soma    = float(np.mean(somas)),
        ppost   = float(np.mean(pposts)),
        help    = float(1.0 - np.mean(pposts)),
        Td      = float(np.mean(Tds)),
        PRLS    = float(np.mean(Tds)) - float(np.mean(Trs)),
        l70     = float(np.mean(l70s)),
        modal   = float(np.mean(modals)),
        year    = yrs[:n].tolist() if hasattr(yrs, "tolist") else list(yrs[:n]),
        trajTr  = np.mean([np.asarray(t[:n]) for t in trajTr], axis=0).tolist(),
    )
    a = agg[key]
    print(f"  → Tr={a['Tr']:.2f}(±{a['Tr_sd']:.2f})  Td={a['Td']:.1f}  "
          f"l70={a['l70']:.2f}  PRLS={a['PRLS']:.1f}")
    print()

with open("results_abm_newrows.json", "w") as f:
    json.dump(agg, f, indent=2)
print("Saved results_abm_newrows.json")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
print(f"{'Condition':<52}  {'Tr':>8}  {'Td':>6}  {'PRLS':>6}  {'l70':>6}")
print("-" * 80)
for v in agg.values():
    print(f"{v['label']:<52}  "
          f"{v['Tr']:>6.2f}±{v['Tr_sd']:.2f}  "
          f"{v['Td']:>6.1f}  "
          f"{v['PRLS']:>6.1f}  "
          f"{v['l70']:>6.2f}")
