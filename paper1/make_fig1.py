import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family']='DejaVu Sans'

# palette (light fill, dark text, mid stroke)
TEAL=('#E1F5EE','#0F6E56','#0F6E56'); CORAL=('#FAECE7','#993C1D','#D85A30')
GRAY=('#F1EFE8','#444441','#888780'); AMBER=('#FAEEDA','#854F0B','#BA7517')

boxes=[
 (TEAL ,"Grandmothering does select for earlier menopause","a fitness optimum exists near age 40", False),
 (GRAY ,"But the realistic starting point is the follicular ceiling","reproduction ceases near age 50, with low somatic maintenance", False),
 (CORAL,"\u2460  Forfeited reproduction","stopping earlier gives up certain late births, immediately", False),
 (CORAL,"\u2461  Discounted, delayed benefit","help to grandkin is worth \u00bc, paid later, and only if she survives", False),
 (CORAL,"\u2462  Maintenance lag","at low maintenance she dies before helping \u2014 survival must co-evolve first", False),
 (CORAL,"\u2463  Damage feedback","if it spreads, daughters reproduce more \u2192 faster aging \u2192 benefit erodes", False),
 (CORAL,"Net selection on the first step \u2264 0","at the observed grandmother benefit (B \u2248 1)", True),
 (GRAY ,"The optimum is real but evolutionarily unreachable","the population stays pinned at the follicular ceiling", False),
]
footer=(AMBER,"Accessible only if grandmothering were ~3\u00d7 stronger than observed")

fig,ax=plt.subplots(figsize=(7.2,9.3))
ax.set_xlim(0,10); ax.set_ylim(0,100); ax.axis('off')
W=9.2; X=0.4; H=8.6; GAP=2.2
y=98
for fill,title,sub,emph in boxes:
    bb=FancyBboxPatch((X,y-H),W,H,boxstyle="round,pad=0.02,rounding_size=0.25",
                      fc=fill[0],ec=fill[2],lw=1.6 if emph else 1.0)
    ax.add_patch(bb)
    cx=X+W/2
    ax.text(cx,y-H*0.36,title,ha='center',va='center',fontsize=12.5,
            fontweight='bold',color=fill[1])
    ax.text(cx,y-H*0.70,sub,ha='center',va='center',fontsize=10.3,color=fill[1])
    ystart=y-H
    if (fill,title,sub,emph)!=boxes[-1]:
        ax.add_patch(FancyArrowPatch((cx,ystart-0.15),(cx,ystart-GAP+0.25),
            arrowstyle='-|>',mutation_scale=14,lw=1.3,color='#73726c'))
    y=ystart-GAP
# footer (after a slightly larger gap)
y-=0.6
bb=FancyBboxPatch((X,y-H*0.62),W,H*0.62,boxstyle="round,pad=0.02,rounding_size=0.25",
                  fc=footer[0][0],ec=footer[0][2],lw=1.0)
ax.add_patch(bb)
ax.text(X+W/2,y-H*0.31,footer[1],ha='center',va='center',fontsize=11,
        fontweight='bold',color=footer[0][1])
plt.tight_layout(pad=0.4)
fig.savefig('Figure_1_conceptual.png',dpi=300,bbox_inches='tight',facecolor='white')
print('saved Figure_1_conceptual.png')
