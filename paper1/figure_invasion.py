"""
Invasion-gradient evidence for both take-home messages, measured as the
direction of SELECTION (drift-separated) rather than the finite-population
equilibrium. Reads results_invasion.json written by invasion.py.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

R0, GROW, DECL = "#1b6ca8", "#d1495b", "#2a9d4a"
GREY = "#6b6b6b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})

d = json.load(open("results_invasion.json"))
B = np.array(d["B"], float)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# Effective-selection threshold colours (consistent across both left panels)
NE_COLS = {30: "#9b2226", 500: "#ae7b11"}
NE_LABS = {30: "$N_e$=30  (band)", 500: "$N_e$=500 (dialect tribe)"}

# ---- Panel A (LEFT): Message 1 - selection profile on menopause age --------
a = ax[0]
tr = np.array(d["Tr_grid"], float)
a.plot(tr, d["s_meno_B1"], "o-", color=R0, lw=2, ms=6, label="$B=1$")
a.plot(tr, d["s_meno_B4"], "s-", color=GROW, lw=2, ms=6, label="$B=4$")
a.axhline(0, color="k", lw=0.8, ls=":")
a.axvspan(46, 49, color="0.85", alpha=0.5, zorder=0)
a.text(47.5, -0.0037, "shallow gradient\nnear ceiling",
       ha="center", va="center", fontsize=8.5, color=GREY)
# Effective-selection thresholds — selection for earlier cessation is negative,
# so thresholds are shown on the negative side: if |s| < 1/(2Ne), drift dominates.
a_ymin = -0.0070
a.set_ylim(a_ymin, 0.0008)
# Ne=500: threshold |s|=0.001 → within panel at s=-0.001
a.axhline(-1/(2*500), color=NE_COLS[500], lw=1.1, ls=(0,(5,3)), alpha=0.85,
          label=f"effective-selection threshold,\n{NE_LABS[500]}")
a.text(41.0, -1/(2*500)+0.00015, NE_LABS[500], fontsize=7.5, color=NE_COLS[500], ha="right")
# (Ne=30 band threshold annotation removed — overlapped the line labels; extending the y-axis to show it would weaken the visual)
a.set_xlabel("menopause age of mutant  $T_r^{*}$  (resident at ceiling 50)")
a.set_ylabel("selection for earlier menopause  $s$")
a.set_title("Message 1: at current maintenance, earlier\ncessation is not favoured (single-allele)", fontsize=11)
a.invert_xaxis()
a.legend(frameon=False, fontsize=9, loc="lower right")

# ---- Panel B (CENTRE): Message 2 - selection on post-reproductive survival --
b = ax[1]
colours = {"0.010": R0, "0.020": GROW, "0.030": DECL}
labels  = {"0.010": "cheap (1% fec. cost)", "0.020": "moderate (2%)",
           "0.030": "costly (3%)"}
for c_key, col in colours.items():
    b.plot(B, d["s_surv"][c_key], "o-", color=col, lw=2, ms=6, label=labels[c_key])
b.axhline(0, color="k", lw=0.8, ls=":")
b_ymax = 0.0025
b.set_ylim(-0.0018, b_ymax)
# Ne=500: threshold s=+0.001 → within panel
b.axhline(1/(2*500), color=NE_COLS[500], lw=1.1, ls=(0,(5,3)), alpha=0.85)
b.text(5.85, 1/(2*500)+0.00003, NE_LABS[500], fontsize=7.5, color=NE_COLS[500], ha="right", va="bottom")
# Ne=30: threshold s=+0.017 → above panel; annotate at top edge
b.annotate(f"{NE_LABS[30]}\n(threshold s=+0.017, above)",
           xy=(5.0, b_ymax-0.00008), xytext=(3.2, b_ymax-0.0005),
           fontsize=7.5, color=NE_COLS[30],
           arrowprops=dict(arrowstyle="->", color=NE_COLS[30], lw=0.7))
b.set_xlabel("grandmothering strength  $B$")
b.set_ylabel("selection on post-repro. survival  $s$")
b.set_title("Message 2: grandmothering selects\nfor longer post-reproductive life", fontsize=11)
b.legend(frameon=False, fontsize=9, loc="upper left")
b.annotate("at $B{=}0$ an extra post-repro.\nyear is worth nothing (Hamilton)",
           xy=(0, d["s_surv"]["0.030"][0]), xytext=(1.4, -0.0013),
           fontsize=8.5, color=GREY,
           arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))

# ---- Panel C (RIGHT): maternal-cost interaction ----------------------------
c = ax[2]
c.plot(B, d["s_meno40_mmoff"], "o-", color=GREY, lw=2, ms=6,
       label="maternal cost OFF")
c.plot(B, d["s_meno40_mmon"], "s-", color=DECL, lw=2, ms=6,
       label="maternal cost ON")
c.axhline(0, color="k", lw=0.8, ls=":")
c.set_xlabel("grandmothering strength  $B$")
c.set_ylabel("selection for earlier cessation  $s$\n($T_r:50\\to40$)")
c.set_title("Cost interaction: rising late-birth risk\nstrengthens selection for earlier stopping",
            fontsize=11)
c.legend(frameon=False, fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig("fig10_invasion.png", bbox_inches="tight")
shutil.copy("fig10_invasion.png","figure_06.png")
print("wrote fig10_invasion.png / figure_06.png")
