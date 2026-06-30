# Code for two companion papers on grandmothering and the evolution of the human female life history

**Author:** [Author information withheld for double-anonymous review]  
**Both papers submitted to:** American Naturalist 
**Contact:** [Contact information withheld for double-anonymous review]

---

## Papers in this repository

### Paper 1 — `paper1/`
**"Grandmothering selects on both age at menopause and post-reproductive lifespan, but only the lifespan response is evolutionarily accessible"**

Code for the optimal-control (OC) and individual-based (ABM) models that separate grandmothering's effects on age at menopause and post-reproductive lifespan into distinct evolutionary questions and answer each quantitatively.

→ See `paper1/README.md` for full details, file inventory, and run instructions.

---

### Paper 2 — `paper2/`
**"The human-specific acceleration of ovarian follicle depletion after age 35 as a predicted signature of grandmothering selection"**

Code for the biexponential follicle-depletion model and selection-coefficient analysis that formalise the Phase 2 signature hypothesis and generate all figures.  This script is self-contained and does not depend on Paper 1's code.

→ See `paper2/README.md` for full details and run instructions.

---

## Relationship between the papers

The two papers address complementary questions and share a common life-history framework:

- Paper 1 asks *how strongly grandmothering selects for each of the two key female life-history traits* (age at menopause and post-reproductive lifespan) and why the evolutionary response is asymmetric.
- Paper 2 asks *what molecular signature grandmothering selection should have left in the comparative biology of ovarian atresia*, and identifies the human-specific Phase 2 steepening as that signature.

The fitness-gain parameters used in Paper 2 (Section 3.1) come from the quantitative model developed in Paper 1.  The papers can be read and reviewed independently; Paper 2 cites the companion model.

---

## Quick-start

```bash
# Paper 1 figures (requires numpy, matplotlib; ~5 min)
cd paper1
python3 make_fig1.py
python3 figure_effects.py
python3 figures_oc.py
python3 figure_gurven_lummaa.py
python3 analyze_sweep.py

# Paper 2 figures (requires numpy, matplotlib; < 1 min)
cd ../paper2
python3 atresia_model.py
python3 make_fig2_phase2.py
python3 chek2_figure.py
```
