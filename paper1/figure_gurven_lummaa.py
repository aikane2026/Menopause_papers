"""
figure_gurven_lummaa.py  --  new figures addressing Gurven and Lummaa reviews.

Fig S5: Mortality sensitivity — back-end and menopause results across the
        forager life-table envelope (Gurven concern 1).
Fig S6: Two-grandmother extension — B_mgm + B_pgm, varying paternity
        uncertainty p_u; comparison against single-grandmother (Lummaa).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil
from model import Params, run_cell, productivity_curve

plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})
R0, GROW, DECL = "#1b6ca8", "#d1495b", "#2a9d4a"
GREY = "#6b6b6b"
Bgrid = np.linspace(0, 6, 22)

# ============================================================
# Helper: get Td, PRLS, l70 over Bgrid
# ============================================================
def sweep_back(extra_params, Bgrid):
    Td, PR, L70 = [], [], []
    for B in Bgrid:
        p = Params(B=B, rate=0.0, **extra_params)
        _, sim = run_cell(p, free_u=False)
        lx = sim['l']
        ages = np.arange(p.a_mat, p.a_max + 1)
        i70 = np.searchsorted(ages, 70)
        Td.append(sim['Td']); PR.append(sim['PRLS'])
        L70.append(float(lx[i70]) if i70 < len(lx) else 0.0)
    return np.array(Td), np.array(PR), np.array(L70)

def sweep_front(extra_params, Bgrid, Tstops=np.arange(30,51)):
    gain = []
    for B in Bgrid:
        W = np.array([run_cell(Params(B=B, rate=0.0, a_ceiling=int(T), **extra_params),
                               free_u=False)[0]['fitness'] for T in Tstops])
        gain.append(100*(W.max()-W[-1])/abs(W[-1]))
    return np.array(gain)


# ============================================================
# Figure S5: Mortality sensitivity
# ============================================================
print("Computing mortality sensitivity...")
sens_cases = {
    "low extrinsic mortality\n(μ₀ × 0.7)":   dict(mu0=0.006*0.7, color="#3a86ff", ls="-", lw=2),
    "baseline":                                 dict(mu0=0.006,     color=GREY,      ls="--", lw=2),
    "high extrinsic mortality\n(μ₀ × 1.4)":   dict(mu0=0.006*1.4, color="#e63946", ls="-", lw=2),
}

fig, ax = plt.subplots(1, 3, figsize=(14, 4.8))

for label, cfg in sens_cases.items():
    kw = {k: v for k, v in cfg.items() if k not in ("color", "ls", "lw")}
    Td, PR, L70 = sweep_back(kw, Bgrid)
    plot_kw = dict(color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"])
    ax[0].plot(Bgrid, PR,  label=label, **plot_kw)
    ax[1].plot(Bgrid, L70, label=label, **plot_kw)
    # front end
    gain = sweep_front(kw, Bgrid)
    ax[2].plot(Bgrid, gain, label=label, **plot_kw)

ax[0].axvline(1, color="0.5", lw=0.8, ls=":")
ax[0].text(1.1, 1.5, "B=1 empirical", fontsize=8, color="0.5")
ax[0].set(xlabel="grandmothering strength B", ylabel="PRLS (yr)",
          title="A  Back end: PRLS robust\nacross forager mortality envelope")
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left",
             bbox_to_anchor=(3, 8), bbox_transform=ax[0].transData)

ax[1].axhline(0.40, color="b", lw=0.8, ls=":", alpha=0.5)
ax[1].axhline(0.47, color="b", lw=0.8, ls=":", alpha=0.5)
ax[1].text(5.5, 0.405, "Gurven–Kaplan target band", fontsize=7.5, color="0.4", ha="right", va="bottom")
ax[1].axvline(1, color="0.5", lw=0.8, ls=":")
ax[1].set(xlabel="grandmothering strength B", ylabel="fraction surviving to 70  (l₇₀)",
          title="B  Calibration: l₇₀ vs mortality\nscenario and grandmothering strength")

ax[2].axhline(1.0, color="0.5", lw=0.8, ls=":")
ax[2].text(0.1, 1.05, "1%: appreciable threshold", fontsize=8, color="0.4")
ax[2].axvline(1, color="0.5", lw=0.8, ls=":")
ax[2].set(xlabel="grandmothering strength B",
          ylabel="fitness gain from earlier menopause (%)",
          title="C  Front end: menopause fitness gain\nacross mortality scenarios")

fig.suptitle("Mortality sensitivity: the back-end PRLS\n"
             "result and the menopause fitness gain are robust across the forager life-table envelope",
             fontweight="bold")
fig.tight_layout()
fig.savefig("fig_S5_mortality_sensitivity.png", bbox_inches="tight")
shutil.copy("fig_S5_mortality_sensitivity.png","figure_S1.png")
print("wrote fig_S5_mortality_sensitivity.png / figure_S1.png")


# ============================================================
# Figure S6: Two-grandmother extension
# ============================================================
print("Computing two-grandmother results...")
two_gm_cases = {
    "MGM only (B=1)":              dict(B_pgm=0.0, p_u=0.0, color=GREY, ls="--", lw=2),
    "MGM + PGM, pᵤ=0 (certain)":  dict(B_pgm=1.0, p_u=0.0, color=R0, ls="-", lw=2),
    "MGM + PGM, pᵤ=0.05":         dict(B_pgm=1.0, p_u=0.05, color=GROW, ls="-", lw=2),
    "MGM + PGM, pᵤ=0.10":         dict(B_pgm=1.0, p_u=0.10, color=DECL, ls="-", lw=2),
}

fig, ax = plt.subplots(1, 3, figsize=(14, 4.8))

for label, cfg in two_gm_cases.items():
    kw = {k: v for k, v in cfg.items() if k not in ("color", "ls", "lw")}
    Td, PR, L70 = sweep_back(kw, Bgrid)
    plot_kw = dict(color=cfg["color"], ls=cfg["ls"], lw=cfg["lw"])
    ax[0].plot(Bgrid, PR,  label=label, **plot_kw)
    ax[1].plot(Bgrid, L70, label=label, **plot_kw)
    gain = sweep_front(kw, Bgrid)
    ax[2].plot(Bgrid, gain, label=label, **plot_kw)

# Gurven-Kaplan target band
for a in ax[:2]:
    a.axvline(1, color="0.5", lw=0.8, ls=":")
ax[1].axhspan(0.40, 0.47, alpha=0.12, color="blue")
ax[1].text(5.8, 0.435, "G-K target", fontsize=8, color="0.4", ha="right")

ax[0].set(xlabel="maternal grandmothering strength B_mgm",
          ylabel="PRLS (yr)", title="A  PRLS: two grandmothers\nextend well beyond one")
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left",
             bbox_to_anchor=(2, 10), bbox_transform=ax[0].transData)

ax[1].set(xlabel="B_mgm", ylabel="fraction surviving to 70  (l₇₀)",
          title="B  l₇₀: two grandmothers (pᵤ<0.10)\nmatch Gurven–Kaplan target at B=1")

ax[2].axhline(1.0, color="0.5", lw=0.8, ls=":")
ax[2].axvline(1, color="0.5", lw=0.8, ls=":")
ax[2].set(xlabel="B_mgm",
          ylabel="fitness gain from earlier menopause (%)",
          title="C  Front end: two grandmothers\nstrengthen selection for earlier menopause")

fig.suptitle("Two-grandmother extension: combined maternal and paternal\n"
             "grandmothering reproduces Gurven–Kaplan demography at the empirical grandmothering strength B=1",
             fontweight="bold")
fig.tight_layout()
fig.savefig("fig_S6_two_grandmothers.png", bbox_inches="tight")
shutil.copy("fig_S6_two_grandmothers.png","figure_S6.png")
print("wrote fig_S6_two_grandmothers.png / figure_S6.png")


# ============================================================
# Figure: Productivity curve (for supplement)
# ============================================================
ages = np.arange(18, 96)
g = productivity_curve(ages)
fig, ax = plt.subplots(figsize=(7, 4))
ax.fill_between(ages, g, alpha=0.15, color="#2a9d4a")
ax.plot(ages, g, color="#2a9d4a", lw=2.5, label="age-graded productivity g(a)")
ax.axvline(38, color="0.5", lw=0.8, ls=":")
ax.text(38.5, 1.01, "peak ~38 yr", fontsize=9, color="0.4", va="bottom")
ax.axvline(50, color="#cc4444", lw=0.8, ls="--")
ax.text(50.5, 0.40, "menopause", fontsize=9, color="#cc4444", va="bottom", ha="left")
ax.set(xlabel="age (yr)", ylabel="net production relative to peak",
       title="Age-graded productivity g(a): calibrated to Kaplan et al. (2000)\nforager net-production curves",
       xlim=(18, 96), ylim=(0, 1.05))
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("fig_productivity_curve.png", bbox_inches="tight")
shutil.copy("fig_productivity_curve.png","figure_S5.png")
print("wrote fig_productivity_curve.png / figure_S5.png")
