"""
chek2_figure.py  --  Paper 2, Figure 2 (CHEK2 variants go from neutral to
selected as grandmothering strengthens).  Reconstructed generator.

NOTE: the within-figure number label (upper-left, gray italic) reads "Figure 3",
matching its position in the current manuscript after the new conceptual diagram
was inserted as Figure 2.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})

GREEN = "#2a9d4a"; RED = "#d1495b"
GOLD  = "#c8a21a"; DKRED = "#9b2226"
GRAD  = 0.37          # single-MGM selection per unit B for a +/-3.5 yr shift
B = np.linspace(0, 6, 200)
mk = np.array([0, 0.5, 1, 1.5, 2, 3])

fig = plt.figure(figsize=(14.6, 7.81))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.28], wspace=0.18,
                       left=0.06, right=0.98, top=0.88, bottom=0.17)

# ---------------- Panel A : selection vs B ----------------
axA = fig.add_subplot(gs[0, 0])
axA.axhline(0, color="k", lw=0.8)
for y, c in [(1.67, DKRED), (-1.67, DKRED)]:
    axA.axhline(y, color=c, ls="--", lw=1.1)
for y in (0.10, -0.10):
    axA.axhline(y, color=GOLD, ls="--", lw=1.1)
axA.axvline(1, color="0.55", ls=":", lw=1.2)
axA.plot(B,  GRAD*B, color=GREEN, lw=2.4)
axA.plot(B, -GRAD*B, color=RED,  lw=2.2, ls="--")
axA.plot(mk,  GRAD*mk, "o", color=GREEN, ms=7)
axA.plot(mk, -GRAD*mk, "s", color=RED,  ms=6)
axA.plot(1,  0.60, "*", color=GREEN, ms=20, zorder=5)
axA.plot(1, -0.60, "*", color=RED,  ms=20, zorder=5)
axA.text(4.05, 1.74, "Threshold Ne=30",  color=DKRED, fontsize=9)
axA.text(3.05, -1.60, "Threshold Ne=30", color=DKRED, fontsize=9)
axA.text(4.05, 0.20, "Threshold Ne=500", color=GOLD, fontsize=9)
axA.text(4.05, -0.40, "Threshold Ne=500", color=GOLD, fontsize=9)
axA.text(1.12, -0.92, "Empirical B=1", color="0.5", fontsize=9)
axA.set(xlim=(-0.1, 6.2), ylim=(-2.1, 2.1),
        xlabel="Grandmothering strength B",
        ylabel="Selection coefficient  s (%)")
axA.set_title("CHEK2 variants go from neutral\nto selected as grandmothering strengthens",
              fontsize=11.5)
axA.text(-0.16, 1.06, "A", transform=axA.transAxes, fontsize=15, fontweight="bold")

# ---------------- Panel B : schematic table ----------------
axB = fig.add_subplot(gs[0, 1]); axB.axis("off")
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
axB.text(-0.04, 1.06, "B", transform=axB.transAxes, fontsize=15, fontweight="bold")
axB.set_title("How grandmothering creates selection on CHEK2\n"
              "variants (each variant shifts menopause age by \u00b13.5 yr)",
              fontsize=11.5)
c1, c2, c3 = 0.02, 0.40, 0.72
# header
axB.text(c1, 0.93, "Grandmothering", fontweight="bold", va="center")
axB.text(c2, 0.93, "CHEK2 GOF\n(earlier menopause)", fontweight="bold",
         color=GREEN, va="center", fontsize=10.5)
axB.text(c3, 0.93, "CHEK2 LOF\n(later menopause)", fontweight="bold",
         color=RED, va="center", fontsize=10.5)
axB.plot([0, 1], [0.85, 0.85], color="0.3", lw=0.8)
rows = [
    (0.70, "B = 0\n(no grandmothering)",
     "s \u2248 0%\nneutral", "s \u2248 0%\nneutral"),
    (0.44, "B = 1, single MGM\n(empirical)",
     "s \u2248 +0.37%\nbeneficial \u2713\n+3.5 yr\ngrandmothering",
     "s \u2248 \u22120.37%\ndeleterious \u2717\n\u22123.5 yr\ngrandmothering"),
    (0.16, "B = 1, MGM + PGM\n(two grandmothers)",
     "s \u2248 +0.60%\nbeneficial \u2713", "s \u2248 \u22120.60%\ndeleterious \u2717"),
]
for yc, lab, gof, lof in rows:
    axB.add_patch(Rectangle((0, yc-0.115), 1, 0.23, facecolor="#e9f0fb",
                            edgecolor="none", zorder=0))
    axB.text(c1, yc, lab, va="center", fontsize=10)
    axB.text(c2, yc, gof, va="center", color=GREEN, fontsize=10)
    axB.text(c3, yc, lof, va="center", color=RED, fontsize=10)

# ---------------- bottom legend ----------------
handles = [
    Line2D([0],[0], color=GREEN, lw=2.4, marker="o", ms=7,
           label="CHEK2 GOF \u2014 earlier menopause, more grandmothering"),
    Line2D([0],[0], color=RED, lw=2.2, ls="--", marker="s", ms=6,
           label="CHEK2 LOF \u2014 later menopause, less grandmothering"),
    Line2D([0],[0], color=GREEN, lw=0, marker="*", ms=16,
           label="MGM+PGM at B=1, GOF  (s = +0.60%)"),
    Line2D([0],[0], color=RED, lw=0, marker="*", ms=16,
           label="MGM+PGM at B=1, LOF  (s = \u22120.60%)"),
]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.005),
           ncol=2, frameon=False, fontsize=9, handletextpad=0.5, columnspacing=1.6)
fig.text(0.62, 0.085,
         "GOF = gain-of-function: more aggressive CHK2/p63 checkpoint, "
         "accelerated atresia, earlier menopause", fontsize=8.6, color=GREEN, va="center")
fig.text(0.62, 0.045,
         "LOF = loss-of-function: less aggressive checkpoint, slower atresia, "
         "later menopause", fontsize=8.6, color=RED, va="center")

# ---------------- figure-number label (THE FIX: was "Figure 3") ----------------
fig.text(0.012, 0.97, "Figure 3", style="italic", color="0.5",
         fontsize=12, ha="left", va="top")

fig.savefig("Figure_3_CHEK2.png", dpi=135,
            bbox_inches="tight", facecolor="white")
from PIL import Image
print("saved:", Image.open("Figure_3_CHEK2.png").size)
