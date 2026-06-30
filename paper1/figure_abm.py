"""figure_abm.py  —  Figure 4: OptMeno vs ABM evolved allele + trajectory.

Two panels:
  A  Optimal-control OptMeno (blue) vs long-run ABM evolved allele (red dots).
  B  Representative evolutionary trajectories at B=0 and B=6.

Data: reads results_abm.json (K=10,000, 100,000 yr, 5 seeds).
Trajectories for Panel B: K=1,000, 100,000 yr, 1 seed per condition.

Run:
    python3 figure_abm.py
Writes: fig8_abm.png
"""

import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil
from matplotlib.lines import Line2D
from model import Params, run_cell
from abm   import run_abm

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})
RED  = "#c0392b"; BLUE = "#1b6ca8"; GREY = "#6b6b6b"

# ── 1. Load ABM equilibrium data ───────────────────────────────────────────────
d = json.load(open("results_abm.json"))
KEYS = ["B0.0_pgm0.0_pu0.05","B1.0_pgm0.0_pu0.05",
        "B3.0_pgm0.0_pu0.05","B6.0_pgm0.0_pu0.05"]
Bs_abm   = [0.0, 1.0, 3.0, 6.0]
abm_Tr   = [d[k]["Tr"]    for k in KEYS]
abm_TrSD = [d[k]["Tr_sd"] for k in KEYS]

# ── 2. Compute OptMeno curve ────────────────────────────────────────────────────
FULL   = dict(gm_a_zero=90, mm_scale=1, use_prod_curve=True)
Tstops = np.arange(30, 51)
Bs_oc  = np.linspace(0, 8, 41)
opt_meno = []
for B in Bs_oc:
    W = np.array([run_cell(Params(**FULL, B=float(B), a_ceiling=int(T), rate=0.0),
                           free_u=False)[0]["fitness"] for T in Tstops])
    opt_meno.append(int(Tstops[np.argmax(W)]))
opt_meno = np.array(opt_meno, dtype=float)

# ── 3. Trajectories for Panel B (100,000 yr) ──────────────────────────────────
p = Params(**FULL)
TRAJ_YRS = 100_000
TRAJ_K   = 1_000
print(f"Running trajectory B=0  ({TRAJ_YRS:,} yr, K={TRAJ_K:,}) …")
H0, _ = run_abm(B=0.0, B_pgm=0.0, p_u=0.05,
                years=TRAJ_YRS, K=TRAJ_K, seed=42, burn_in=0, p=p)
print(f"Running trajectory B=6  ({TRAJ_YRS:,} yr, K={TRAJ_K:,}) …")
H6, _ = run_abm(B=6.0, B_pgm=0.0, p_u=0.05,
                years=TRAJ_YRS, K=TRAJ_K, seed=42, burn_in=0, p=p)

# Thin to ~500 display points so lines aren't over-dense
def thin(H, n=500):
    step = max(1, len(H["year"]) // n)
    return H["year"][::step], H["meanTr"][::step]

yr0, tr0 = thin(H0)
yr6, tr6 = thin(H6)

# ── 4. Figure ──────────────────────────────────────────────────────────────────
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))
CEILING = 50.0

# ─── Panel A ──────────────────────────────────────────────────────────────────
ax = axA
ax.plot(Bs_oc, opt_meno, color=BLUE, lw=2.2)
ax.axhline(CEILING, color="0.4", ls="--", lw=1.1, zorder=1)
ax.errorbar(Bs_abm, abm_Tr, yerr=abm_TrSD,
            fmt="o", color=RED, ms=7, lw=1.8, capsize=4, zorder=5)

# Annotation 1: "atresia ceiling" — lower border just above dashed line
ax.text(0.18, CEILING + 0.25, "atresia ceiling (50 yr)",
        fontsize=9, color="0.4", va="bottom", ha="left")

# Annotation 2: "ABM allele stays..." — top edge just below dashed line (no arrow)
ax.text(2.2, CEILING - 0.25,
        "ABM allele stays near ceiling:\nearlier cessation inaccessible\nto single-allele change",
        fontsize=9, color=RED, va="top", ha="left")

# Annotation 3: OptMeno description — lower-left corner
ax.text(0.3, 33.8,
        "OptMeno declines from 50 to ~30\n(real selection); fitness gain\n"
        "≈1% at B=1,  ≈4.5% at B=8",
        fontsize=9, color=BLUE, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

ax.set(xlabel="Grandmothering strength B",
       ylabel="Age at menopause (yr)",
       xlim=(-0.2, 8.2), ylim=(30, 51.8))
ax.set_title("A  OptMeno vs ABM evolved age at menopause (Tr)", loc="left",
             fontsize=11, pad=6)
ax.legend(handles=[
    Line2D([0],[0], color=BLUE, lw=2.2, label="OptMeno (optimal-control prediction)"),
    Line2D([0],[0], marker="o", color=RED, lw=1.8, ms=7,
           label="Tr (ABM evolved allele; mean ± SD, 5 seeds,\nK=10,000, 100,000 yr)"),
    Line2D([0],[0], color="0.4", ls="--", lw=1.1, label="atresia ceiling (50 yr)"),
], frameon=False, fontsize=9, loc="center right", bbox_to_anchor=(1.0, 0.58))

# ─── Panel B ──────────────────────────────────────────────────────────────────
ax = axB
ax.plot(yr0, tr0, color=GREY, lw=1.2, alpha=0.9, label="B = 0 (no grandmothering)")
ax.plot(yr6, tr6, color=RED,  lw=1.2, alpha=0.9, label="B = 6  (6× empirical)")
ax.axhline(CEILING, color="0.4", ls="--", lw=1.1, zorder=1,
           label="atresia ceiling (50 yr)")
ax.set(xlabel="Year", ylabel="Population-mean Tr (yr)",
       xlim=(0, TRAJ_YRS), ylim=(40, 51.8))
ax.set_title("B  Evolutionary trajectory of Tr", loc="left", fontsize=11, pad=6)
ax.legend(frameon=False, fontsize=9, loc="upper right")

# ── Shared title ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Individual-based model: the evolved age-at-menopause allele stays near the "
    "atresia ceiling;\nearlier cessation is inaccessible to single-allele change",
    fontsize=11, fontweight="bold", y=1.01)

fig.tight_layout()
fig.savefig("fig8_abm.png", bbox_inches="tight")
shutil.copy("fig8_abm.png","figure_04.png")
print("wrote fig8_abm.png / figure_04.png")
