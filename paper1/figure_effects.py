"""
figure_effects.py  --  stepwise model-construction figures for the paper.

Four model conditions, progressively adding each effect from the baseline:
  C0  baseline          B=0, standard q, mm_scale=0
  C1  +grandmothering   B>0, standard q, mm_scale=0, FLAT efficacy (a_zero=500)
  C2  +efficacy decline B>0, standard q, mm_scale=0, GRADUAL declining efficacy
  C3  +late-repro costs B>0, standard q, mm_scale=1 (q-discounted), declining efficacy

Produces three figures:
  fig_A_profiles.png   -- age-specific curves that define each effect
  fig_B_backend.png    -- back-end: PRLS and Td vs B (the four conditions)
  fig_C_frontend.png   -- front-end: menopause fitness landscape (four conditions)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil
from model import (Params, run_cell, quality_curve, maternal_mortality,
                   grandmother_efficacy, grandmother_benefit)

# ---- palette ---------------------------------------------------------------
C0_col = "#555555"      # baseline  -- dark grey
C1_col = "#1b6ca8"      # +GM flat  -- blue
C2_col = "#d1495b"      # +decl E   -- red
C3_col = "#2a9d4a"      # +costs    -- green
B1_dash = dict(ls=":", color="0.5", lw=1.0)   # empirical B=1 marker

plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})

Bgrid = np.linspace(0, 6, 25)
CONDS = {
    "C0_base":  dict(gm_a_zero=90,  gm_decline_k=2.1, mm_scale=0.0, label="Baseline (B=0)", B_fixed=0, color=C0_col, lw=2.0, ls="--"),
    "C1_flat":  dict(gm_a_zero=500, gm_decline_k=2.1, mm_scale=0.0, use_prod_curve=False, delta_rep=0.0, label="+grandmothering,\nconstant efficacy", color=C1_col, lw=2.0, ls="-"),
    "C2_decl":  dict(gm_a_zero=90,  gm_decline_k=2.1, mm_scale=0.0, use_prod_curve=False, delta_rep=0.0, label="+efficacy declines\nwith age", color=C2_col, lw=2.0, ls="-"),
    "C3_full":  dict(gm_a_zero=90,  gm_decline_k=2.1, mm_scale=1.0, use_prod_curve=False, delta_rep=0.0, label="+rising late-repro\ncosts (q, p_mat)", color=C3_col, lw=2.0, ls="-"),
}
TSTOPS = np.arange(30, 51)

# ============================================================
# 0. Pre-compute back-end and front-end for each condition
# ============================================================
results = {}
for cname, cfg in CONDS.items():
    Td_arr = np.zeros_like(Bgrid); PR_arr = np.zeros_like(Bgrid)
    gain_arr = np.zeros_like(Bgrid); argT_arr = np.zeros_like(Bgrid)
    Brun = cfg.get("B_fixed", None)   # if fixed, use same B for all grid pts
    for i, B in enumerate(Bgrid):
        B_use = Brun if Brun is not None else B
        p = Params(B=B_use, rate=0.0,
                   gm_a_zero=cfg["gm_a_zero"], gm_decline_k=cfg["gm_decline_k"],
                   mm_scale=cfg["mm_scale"])
        _, sim = run_cell(p, free_u=False)
        Td_arr[i] = sim["Td"]; PR_arr[i] = sim["PRLS"]
        # menopause fitness landscape
        W = np.array([
            run_cell(Params(B=B_use, rate=0.0, a_ceiling=int(T),
                            gm_a_zero=cfg["gm_a_zero"], gm_decline_k=cfg["gm_decline_k"],
                            use_prod_curve=cfg.get("use_prod_curve", True),
                            delta_rep=cfg.get("delta_rep", 0.006),
                            mm_scale=cfg["mm_scale"]), free_u=False)[0]["fitness"]
            for T in TSTOPS])
        W50 = W[-1]
        gain_arr[i] = 100.0 * (W.max() - W50) / abs(W50)
        argT_arr[i] = TSTOPS[np.argmax(W)]
    results[cname] = dict(Td=Td_arr, PRLS=PR_arr, gain=gain_arr, argT=argT_arr)
    print(f"  {cname} done: B=1 PRLS={PR_arr[int(np.argmin(abs(Bgrid-1)))]:.1f}")

# ============================================================
# Fig A -- age-specific profiles of the modelled effects
# ============================================================
ages = np.arange(18, 96)
p_ref = Params()
q_vals   = quality_curve(ages, p_ref.q0, p_ref.qk, p_ref.qa50)
pmat_vals = maternal_mortality(ages, p_ref.mm_base, p_ref.mm_k, p_ref.mm_a0, p_ref.mm_floor)
E_flat   = grandmother_efficacy(ages, a_full=50, a_zero=500, k=2.1)
E_decl   = grandmother_efficacy(ages, a_full=50, a_zero=p_ref.gm_a_zero, k=p_ref.gm_decline_k)
dG_flat  = grandmother_benefit(ages, p_ref.a_gm, p_ref.gm_ramp, p_ref.gm_plateau, 50, 500, 2.1)
dG_decl  = grandmother_benefit(ages, p_ref.a_gm, p_ref.gm_ramp, p_ref.gm_plateau,
                                p_ref.gm_a_full, p_ref.gm_a_zero, p_ref.gm_decline_k)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))

# A1: quality / maternal-mortality curves
ax[0].fill_between(ages, q_vals, alpha=0.15, color=C3_col)
ax[0].plot(ages, q_vals, color=C3_col, lw=2.0, label="oocyte viability  q(a)")
ax2 = ax[0].twinx()
ax2.plot(ages, pmat_vals, color="#a05020", lw=2.0, ls="--", label="maternal mortality  p(a)")
# maternal-mortality label: right of the age-40 line, lower edge at 0.5 of axis height;
# arrow head anchored on the brown dashed p(a) curve (data coords), tail just below the text
ax2.annotate("maternal\nmortality p(a)",
             xy=(45, pmat_vals[45 - 18]),
             xytext=(0.62, 0.50), textcoords="axes fraction",
             fontsize=8, color="#a05020", va="bottom", ha="left",
             arrowprops=dict(arrowstyle="->", color="#a05020", lw=0.7))
ax2.set_ylabel("per-birth maternal mortality", color="#a05020", fontsize=9.5)
ax2.tick_params(axis="y", colors="#a05020")
ax2.spines["top"].set_visible(False)
ax[0].axvline(40, color="0.6", lw=0.8, ls=":")
ax[0].text(40.5, 0.75, "age 40", fontsize=8, color="0.5")
ax[0].set(xlabel="maternal age (yr)", ylabel="oocyte viability  q(a)",
          title="A  Late-reproduction costs", xlim=(18, 55))
# oocyte-viability label: arrow head anchored on the green q(a) curve (data coords),
# text moved to the lower-left so it does not overlap the curve or the arrow
ax[0].annotate("oocyte viability q(a)",
               xy=(28, q_vals[28 - 18]), xytext=(0.04, 0.28), textcoords="axes fraction",
               fontsize=8, color=C3_col, va="bottom", ha="left",
               arrowprops=dict(arrowstyle="->", color=C3_col, lw=0.7))

# A2: grandmother efficacy curves
ax[1].plot(ages, E_flat, color=C1_col, lw=2.0, ls="-", label="constant efficacy")
ax[1].plot(ages, E_decl, color=C2_col, lw=2.0, ls="-", label="declining efficacy (calibrated)")
ax[1].axvline(50, color="0.6", lw=0.8, ls=":")
# move "menopause" label to LEFT of the vertical line to avoid overlap
ax[1].text(49.4, 0.90, "menopause", fontsize=8, color="0.5", ha="right")
ax[1].set(xlabel="grandmother age (yr)", ylabel="relative helping efficacy  E(a)",
          title="B  Grandmother efficacy", xlim=(18, 96), ylim=(-0.05, 1.05))
ax[1].legend(frameon=False, fontsize=9)

# A3: grandmother benefit curves (ΔG × B, B=1)
ax[2].plot(ages, dG_flat, color=C1_col, lw=2.0, ls="-", label="constant efficacy")
ax[2].plot(ages, dG_decl, color=C2_col, lw=2.0, ls="-", label="declining efficacy")
ax[2].axvline(50, color="0.6", lw=0.8, ls=":")
ax[2].fill_between(ages, dG_decl, alpha=0.12, color=C2_col)
ax[2].set(xlabel="grandmother age (yr)", ylabel="per-year grandoffspring benefit  ΔG(a)",
          title="C  Per-year grandmother benefit (B=1)", xlim=(18, 96))
ax[2].legend(frameon=False, fontsize=9)

fig.suptitle("Age-specific curves defining each modelled effect", fontweight="bold", fontsize=11.5)
fig.tight_layout()
fig.savefig("fig_A_profiles.png", bbox_inches="tight")
shutil.copy("fig_A_profiles.png", "figure_S4.png")
print("wrote fig_A_profiles.png / figure_S4.png")

# ============================================================
# Fig B -- back-end: PRLS and Td vs B
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

for cname, cfg in CONDS.items():
    r = results[cname]; col = cfg["color"]; ls = cfg["ls"]
    lbl = cfg["label"]; lw = cfg["lw"]
    B_fixed = cfg.get("B_fixed", None)
    if B_fixed is not None:
        # horizontal reference line (baseline at B=0)
        ax[0].axhline(r["PRLS"][0], color=col, lw=lw, ls=ls, label=lbl)
        ax[1].axhline(r["Td"][0], color=col, lw=lw, ls=ls, label=lbl)
    else:
        ax[0].plot(Bgrid, r["PRLS"], color=col, lw=lw, ls=ls, label=lbl)
        ax[1].plot(Bgrid, r["Td"], color=col, lw=lw, ls=ls, label=lbl)

for a in ax:
    a.axvline(1, **B1_dash)

ax[0].set(xlabel="grandmothering strength  B", ylabel="post-reproductive lifespan  PRLS (yr)",
          title="A  Back end: grandmothering extends PRLS")
ax[1].set(xlabel="grandmothering strength  B", ylabel="life expectancy at maturity  T_d (yr)",
          title="B  Back end: life expectancy at maturity")
# legend in lower-right, lowest part just above the horizontal baseline (~3.5 yr)
ax[0].legend(frameon=False, fontsize=8.5, loc="lower right",
             bbox_to_anchor=(0.98, 0.15))
# B=1 empirical label: bottom edge at y=4, bold, on left panel only
ax[0].annotate("B = 1\n(empirical)", xy=(1, 4), xytext=(1.35, 4.3),
               fontsize=8.5, fontweight="bold", color="0.4",
               arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))

fig.suptitle("Back end: the dominant effect is grandmothering; efficacy decline and\n"
             "late-reproduction costs each trim it modestly", fontweight="bold")
fig.tight_layout()
fig.savefig("fig_B_backend.png", bbox_inches="tight")
shutil.copy("fig_B_backend.png", "figure_S10.png")
print("wrote fig_B_backend.png / figure_S10.png")

# ============================================================
# Fig C -- front-end: menopause fitness landscape
# ============================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

for cname, cfg in CONDS.items():
    r = results[cname]; col = cfg["color"]; ls = cfg["ls"]
    lbl = cfg["label"]; lw = cfg["lw"]
    B_fixed = cfg.get("B_fixed", None)
    if B_fixed is not None:
        ax[0].axhline(0.0, color=col, lw=lw, ls=ls, label=lbl)
        ax[1].axhline(50.0, color=col, lw=lw, ls=ls, label=lbl)
    else:
        ax[0].plot(Bgrid, r["gain"], color=col, lw=lw, ls=ls, label=lbl)
        ax[1].plot(Bgrid, r["argT"], color=col, lw=lw, ls=ls, label=lbl)

ax[0].axhline(1.0, color="0.6", ls=":", lw=1.0)
ax[0].text(0.04, 1.06, "1%: appreciable effect threshold", fontsize=8.5, color="0.4",
           transform=ax[0].get_yaxis_transform(), va="bottom", ha="left")
# Effective-selection thresholds s=1/(2Ne) expressed as fitness gain (~100*s %)
# for Ne=25 (band), 100 (local group), 500 (dialect tribe):
for Ne, col_t, lbl in [(25, "#993333", "Ne=25 (band, s≈2%)"),
                       (100, "#cc7700", "Ne=100 (s≈0.5%)")]:
    pct = 100.0 / (2 * Ne)
    ax[0].axhline(pct, color=col_t, lw=0.9, ls=(0,(4,3)), alpha=0.75)
    ax[0].text(5.85, pct + 0.04, lbl, fontsize=7.5, color=col_t, va="bottom", ha="right")
ax[1].axhline(50, color="0.5", ls="--", lw=1.2, label="atresia ceiling (50)")
for a in ax:
    a.axvline(1, **B1_dash)
# annotate B=1 gain values in panel A with clear callouts so they can be read
idx1 = int(np.argmin(abs(Bgrid-1)))
y_offsets = {"C1_flat": +0.25, "C2_decl": 0.0, "C3_full": -0.25}
for cname, cfg in CONDS.items():
    if cfg.get("B_fixed", None) is None:
        g = results[cname]["gain"][idx1]
        yo = y_offsets.get(cname, 0)
        ax[0].annotate(f'  {g:.1f}%',
                       xy=(1, g), xytext=(1.55, g + yo),
                       fontsize=9, fontweight="bold", color=cfg["color"],
                       arrowprops=dict(arrowstyle="-", color=cfg["color"], lw=0.8),
                       va="center")
# Panel B: move all labels well below the dashed ceiling line; no B=1 label in figure
ax[1].legend(frameon=False, fontsize=8.5, loc="upper left",
             bbox_to_anchor=(3.5, 47.5), bbox_transform=ax[1].transData)
ax[0].set(xlabel="grandmothering strength  B",
          ylabel="fitness gain from optimal menopause (%)",
          title="A  Fitness gain from earlier menopause")
ax[1].set(xlabel="grandmothering strength  B", ylabel="fitness-optimal age at menopause (yr)",
          title="B  Fitness-optimal age at menopause",
          ylim=(28, 51))
ax[0].set_xlim(0, 6)
ax[1].set_xlim(0, 6)
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")

fig.suptitle("Front end: grandmothering creates real but quantitatively modest selection for\n"
             "earlier menopause; effects comparable to those on PRLS need implausible B",
             fontweight="bold")
fig.tight_layout()
fig.savefig("fig_C_frontend.png", bbox_inches="tight")
shutil.copy("fig_C_frontend.png", "figure_S3.png")
print("wrote fig_C_frontend.png / figure_S3.png")
