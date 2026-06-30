import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})

BLUE="#1f5fa5"; RED="#d1495b"; GRAY="#7a7a74"; PUR="#5b4fb5"; GRN="#8fb83f"
N0=25000.0; FLOOR=1000.0
a1=np.linspace(33,38,40); N1=N0*np.exp(-0.097*(a1-38))
lamB=np.log(N0/FLOOR)/14.0; aB=np.linspace(38,52,120); NB=N0*np.exp(-lamB*(aB-38))
lamG=np.log(N0/FLOOR)/11.0; aG=np.linspace(38,49,120); NG=N0*np.exp(-lamG*(aG-38))

fig,ax=plt.subplots(figsize=(9.2,5.4))
ax.set_yscale("log"); ax.set_xlim(32.5,55); ax.set_ylim(500,3e6)

# forfeited -> grandmothering wedge (between the two Phase 2 curves)
px=list(aB)+[49]+list(aG[::-1]); py=list(NB)+[FLOOR]+list(NG[::-1])
ax.fill(px,py,color=GRN,alpha=0.22,lw=0)

ax.plot(a1,N1,color=GRAY,lw=1.6)
ax.plot(aB,NB,color=BLUE,lw=2.6)
ax.plot(aG,NG,color=RED,lw=2.6)
ax.axhline(FLOOR,color=GRAY,ls="--",lw=1.0)
ax.text(32.8,FLOOR*1.12,"menopause threshold (~1,000 follicles)",fontsize=8.5,color=GRAY,va="bottom")

ax.scatter([38],[N0],s=34,color="#444",zorder=5)
ax.scatter([52],[FLOOR],s=34,color=BLUE,zorder=5)
ax.scatter([49],[FLOOR],s=34,color=RED,zorder=5)

# force-of-selection arc (endpoint -> age of expression)
arc=FancyArrowPatch((50.6,1.0e5),(38.3,1.0e5),connectionstyle="arc3,rad=0.32",
                    arrowstyle="-|>",mutation_scale=15,lw=1.7,color=PUR,zorder=6)
ax.add_patch(arc)
ax.plot([38.3,38.3],[1.0e5,2.9e4],ls=":",lw=0.9,color=PUR)
ax.plot([50.6,50.6],[1.0e5,1.6e3],ls=":",lw=0.9,color=PUR)
ax.text(44.3,9.5e5,"age-specific force of selection (Hamilton 1966):",
        ha="center",fontsize=9.2,color=PUR)
ax.text(44.3,5.6e5,"the gradient at the endpoint is carried back to the age of expression",
        ha="center",fontsize=9.2,color=PUR)

# annotations
ax.annotate("Phase 2 begins\n(~38 yr, 25,000 follicles)",xy=(38,N0),xytext=(33.0,2.4e3),
            fontsize=8.8,arrowprops=dict(arrowstyle="-",color=GRAY,lw=0.8))
ax.annotate("GOF allele steepens the\nPhase 2 rate (acts from 38)",xy=(43.5,N0*np.exp(-lamG*5.5)),
            xytext=(38.6,1.5e3),fontsize=8.8,color=RED,
            arrowprops=dict(arrowstyle="-",color=RED,lw=0.8))
ax.annotate("forfeited low-quality years become\ngrandmothering years   (s \u2248 0.66%)",
            xy=(47.4,2.3e3),xytext=(49.2,4.0e4),fontsize=8.8,color="#3b6d11",
            arrowprops=dict(arrowstyle="-",color="#3b6d11",lw=0.8))
ax.annotate("",xy=(49,FLOOR*0.78),xytext=(52,FLOOR*0.78),
            arrowprops=dict(arrowstyle="<->",color="#444",lw=1.0))
ax.text(50.5,FLOOR*0.60,"endpoint 52 \u2192 49 (\u22483 yr)",ha="center",fontsize=8.5,color="#444")

ax.set_xlabel("age (years)"); ax.set_ylabel("follicle count (log scale)")
ax.legend(handles=[Line2D([0],[0],color=BLUE,lw=2.6,label="current human (endpoint \u224852)"),
                   Line2D([0],[0],color=RED,lw=2.6,label="with Phase 2 GOF allele (endpoint \u224849)")],
          loc="upper right",fontsize=9,frameon=False)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout()
fig.savefig("Figure_2_phase2_force.png",dpi=200,bbox_inches="tight",facecolor="white")
from PIL import Image; print("size",Image.open("Figure_2_phase2_force.png").size)
