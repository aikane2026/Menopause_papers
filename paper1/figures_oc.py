"""Generate optimal-control figures as separate PNG files."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "font.family": "DejaVu Sans", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6,
})
C = {"R0 (stationary)": "#1b6ca8", "r = +0.02 (growing)": "#d1495b",
     "r = -0.01 (declining)": "#2a9d4a"}
ATRESIA = 50

d = np.load("results_oc.npz", allow_pickle=True)
Bgrid = d["Bgrid"]; backend = d["backend"].item()
Tstops = d["Tstops"]; B_profiles = d["B_profiles"]; front = d["front"].item()
Bfine = d["Bfine"]; grad = d["grad"].item(); thresholds = d["thresholds"].item()
ggrid = d["ggrid"]; reaction = d["reaction"].item()
mu0grid = d["mu0grid"]; extrinsic = d["extrinsic"].item()
sched = d["sched"].item(); fos = d["fos"].item()
currencies = list(d["currencies"])


# ----- FIG 1: back-end response -----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
for lab in currencies:
    ax[0].plot(Bgrid, backend[lab]["PRLS"], color=C[lab], lw=2.4, label=lab)
    ax[1].plot(Bgrid, backend[lab]["Td"], color=C[lab], lw=2.4, label=lab)
ax[0].axvline(1.0, color="0.5", ls=":", lw=1.2)
ax[0].text(1.05, 0.5, "empirical\nstrength (B=1)", color="0.4", fontsize=8.5, va="bottom")
ax[0].set(xlabel="grandmothering strength  B", ylabel="post-reproductive lifespan (yr)",
          title="A  Back end: grandmothering extends PRLS")
ax[1].set(xlabel="grandmothering strength  B", ylabel="age at death  $T_d$ (yr)",
          title="B  Age at death vs grandmothering")
ax[1].axhline(ATRESIA, color="0.5", ls="--", lw=1, label="menopause (fixed, 50)")
ax[0].legend(frameon=False, fontsize=9); ax[1].legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig("fig1_backend.png"); plt.close(fig)

# ----- FIG 2: menopause fitness landscape (flatness) -----
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=False)
for j, lab in enumerate(currencies):
    ax = axs[j]
    for B in B_profiles:
        W = front[lab][B]
        Wn = W / W[-1]   # normalise to fitness at menopause=50
        ax.plot(Tstops, Wn, lw=2.2, marker="o", ms=3,
                label=f"B = {B:g}")
    ax.axhline(1.0, color="0.6", ls=":", lw=1)
    ax.axvline(ATRESIA, color="0.5", ls="--", lw=1)
    ax.set(xlabel="age at menopause  $T_r$ (yr)",
           title=lab)
    if j == 0:
        ax.set_ylabel("relative fitness  W($T_r$) / W(50)")
    ax.legend(frameon=False, fontsize=8.5, title="grandmothering")
fig.suptitle("Without grandmothering selection strongly favours reproducing to the ceiling; "
             "grandmothering flattens the\nlandscape but the optimum never sits more than ~2% "
             "above the ceiling value", fontweight="bold", y=1.04, fontsize=11.5)
fig.tight_layout(); fig.savefig("fig2_menopause_landscape.png", bbox_inches="tight"); plt.close(fig)

# ----- FIG 3: menopause selection gradient + optimum + thresholds -----
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
for lab in currencies:
    ax[0].plot(Bfine, 100 * grad[lab]["s"], color=C[lab], lw=2.4, label=lab)
    ax[1].plot(Bfine, grad[lab]["argT"], color=C[lab], lw=2.4, label=lab)
ax[0].axhline(1.0, color="0.5", ls=":", lw=1)
ax[0].text(0.1, 1.05, "1% (appreciable)", color="0.4", fontsize=8)
ax[0].axvline(1.0, color="0.6", ls="--", lw=1)
ax[0].text(1.05, 0.05, "B=1\n(empirical)", color="0.4", fontsize=8, va="bottom")
ax[0].set(xlabel="grandmothering strength  B",
          ylabel="fitness gain from reducing $T_r$ (%)",
          title="A  Strength of selection to reduce menopause age")
ax[1].axhline(ATRESIA, color="0.5", ls="--", lw=1, label="atresia ceiling (50)")
ax[1].axvline(1.0, color="0.6", ls="--", lw=1)
ax[1].set(xlabel="grandmothering strength  B",
          ylabel="fitness-optimal menopause age (yr)",
          title="B  Optimal menopause age")
ax[0].legend(frameon=False, fontsize=8.5); ax[1].legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig("fig3_menopause_selection.png"); plt.close(fig)

# ----- FIG 4: reaction norm for age at death over environment g -----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
lab = currencies[0]
ax[0].plot(ggrid, reaction[lab]["Td_B0"], color="0.45", lw=2.4, ls="--",
           label="B = 0 (somatic byproduct)")
ax[0].plot(ggrid, reaction[lab]["Td_B1"], color=C[lab], lw=2.6,
           label="B = 1 (with grandmothering)")
ax[0].fill_between(ggrid, reaction[lab]["Td_B0"], reaction[lab]["Td_B1"],
                   color=C[lab], alpha=0.12)
ax[0].set(xlabel="environment  g  (production multiplier)",
          ylabel="age at death  $T_d$ (yr)",
          title="A  Reaction norm for age at death ($R_0$)")
ax[0].legend(frameon=False, fontsize=9)
for lab in currencies:
    incr = reaction[lab]["Td_B1"] - reaction[lab]["Td_B0"]
    ax[1].plot(ggrid, incr, color=C[lab], lw=2.4, label=lab)
ax[1].set(xlabel="environment  g  (production multiplier)",
          ylabel="grandmothering increment  $\\Delta T_d$ (yr)",
          title="B  Increment depends on environment & currency")
ax[1].legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig("fig4_reaction_norm.png"); plt.close(fig)

# ----- FIG 5: extrinsic mortality (disposable-soma test) -----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
lab = currencies[0]
ax[0].plot(mu0grid, extrinsic[lab]["Td_B0"], color="0.45", lw=2.4, ls="--", label="B = 0")
ax[0].plot(mu0grid, extrinsic[lab]["Td_B1"], color=C[lab], lw=2.6, label="B = 1")
ax[0].set(xlabel="extrinsic mortality  $\\mu_0$ (/yr)", ylabel="age at death  $T_d$ (yr)",
          title="A  Higher extrinsic mortality shortens life")
ax[0].legend(frameon=False, fontsize=9)
ax[1].plot(mu0grid, extrinsic[lab]["repair_B0"], color="#7a4ea3", lw=2.4)
ax[1].set(xlabel="extrinsic mortality  $\\mu_0$ (/yr)",
          ylabel="mean repair effort during reproduction",
          title="B  ...and selects for less maintenance")
fig.tight_layout(); fig.savefig("fig5_extrinsic_mortality.png"); plt.close(fig)

# ----- FIG 6: mechanism - force of selection + allocation schedule -----
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
for B, col, ls in [("B0.0", "0.5", "--"), ("B1.0", C[currencies[0]], "-")]:
    a = fos[B]["ages"]; s = fos[B]["s"]
    ax[0].plot(a, s / s[0], color=col, ls=ls, lw=2.4,
               label=("no grandmothering" if B == "B0.0" else "with grandmothering (B=1)"))
ax[0].axvline(ATRESIA, color="0.5", ls=":", lw=1)
ax[0].text(ATRESIA + 0.5, 0.6, "menopause", rotation=90, color="0.4", fontsize=8, va="center")
ax[0].set(xlabel="age (yr)", ylabel="force of selection on survival (rel.)",
          title="A  Grandmothering sustains selection past menopause")
ax[0].legend(frameon=False, fontsize=9)
s = sched["B1.0"]
ax[1].stackplot(s["ages"], s["u"], s["w"], s["z"],
                labels=["reproduction $u$", "help $w$", "repair $z$"],
                colors=["#1b6ca8", "#2a9d4a", "#d1495b"], alpha=0.85)
ax[1].plot(s["ages"], s["l"], color="k", lw=2, label="survivorship $\\ell$")
ax[1].axvline(ATRESIA, color="0.4", ls=":", lw=1)
ax[1].set(xlabel="age (yr)", ylabel="surplus allocation / survivorship",
          title="B  Optimal life-history schedule (B=1, $R_0$)", ylim=(0, 1.05))
ax[1].legend(frameon=False, fontsize=8, loc="upper right",
             bbox_to_anchor=(92, 0.98), bbox_transform=ax[1].transData)
fig.tight_layout(); fig.savefig("fig6_mechanism.png"); shutil.copy("fig6_mechanism.png","figure_S9.png"); plt.close(fig)

print("OC figures written: fig1..fig6")

# ----- FIG 7: CALIBRATION - grandmother efficacy E(a) and survival vs Gurven -----
survival = d["survival"].item(); efficacy = d["efficacy"].item(); gurven = d["gurven"].item()
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
ea = efficacy["ages"]
ax[0].plot(ea, efficacy["E"], color="#7a4ea3", lw=2.6, label="modelled efficacy $E(a)$")
ax[0].scatter(efficacy["anchors_age"], efficacy["anchors_E"], color="#d1495b", zorder=5, s=45,
              label="empirical anchors (HG / traditional)")
ax[0].axhline(0, color="0.7", lw=0.8)
ax[0].annotate("Hadza foraging\ndecline (early 70s)", (70, 0.77), (62, 0.42),
               fontsize=8, color="0.4", arrowprops=dict(arrowstyle="->", color="0.6"))
ax[0].annotate("Chapman 2019:\nbenefit $\\to$0 by 75", (75, 0.63), (76, 0.8),
               fontsize=8, color="0.4", arrowprops=dict(arrowstyle="->", color="0.6"))
ax[0].set(xlabel="grandmother age (yr)", ylabel="helping efficacy  $E(a)$",
          title="A  Age-declining grandmother efficacy", ylim=(-0.05, 1.08))
ax[0].legend(frameon=False, fontsize=8.5, loc="lower left")
for B, col, lab in [("B0.0", "0.5", "B=0 (no grandmothering)"),
                    ("B1.0", C["R0 (stationary)"], "B=1 (empirical strength)"),
                    ("B3.0", "#e8a33d", "B=3")]:
    sv = survival[B]
    ax[1].plot(sv["ages"], sv["l"], color=col, lw=2.4,
               label=f"{lab}: modal {sv['modal']}, $l_{{70}}$={sv['l70']:.2f}")
ax[1].axhline(gurven["l70"], color="#1b6ca8", ls=":", lw=1.3)
ax[1].axvspan(gurven["modal_lo"], gurven["modal_hi"], color="#1b6ca8", alpha=0.08)
ax[1].text(73, gurven["l70"] + 0.015,
           "Gurven & Kaplan: 47% reach 70, modal death 68-78",
           fontsize=8, color="#1b6ca8", va="bottom")
ax[1].set(xlabel="age (yr)", ylabel="survivorship from maturity  $\\ell(a)$",
          title="B  Emergent survival vs hunter-gatherer data", xlim=(18, 92))
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("fig7_calibration.png"); shutil.copy("fig7_calibration.png","figure_S2.png"); plt.close(fig)
print("OC figure written: fig7_calibration")
