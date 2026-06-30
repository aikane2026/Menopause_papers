"""
atresia_model.py  –  Oocyte atresia rate, the Phase 2 signature hypothesis,
                      selection on atresia-rate alleles, and quality-control
                      coevolution.

THE PHASE 2 SIGNATURE HYPOTHESIS (this paper):
  The human-specific steepening of follicle depletion after age ~35 (Cloutier
  et al. 2015, GeroScience) is a signature of grandmothering selection acting
  on the Phase 2 atresia rate in the human lineage.

  The reasoning:
    (1) The ~50-yr follicle-depletion endpoint is ANCESTRALLY CONSERVED across
        great apes: chimps (Wood et al. 2023 Ngogo: last birth by ~50, hormonal
        menopause signs after 50), gorillas (captive; Atsalis & Videan 2009), and
        humans all share this endpoint.  Neither grandmothering selection nor
        any other recently derived selection has moved the endpoint from some
        much later ancestral state.  The endpoint reflects the intrinsic biology
        of oocyte storage: cohesins and the spindle assembly checkpoint degrade
        in meiotic arrest over ~50 years, setting a quality window regardless of
        pool dynamics.

    (2) BUT Cloutier et al. (2015) showed, with a sample 3× larger than Jones
        et al. (2007), that human and chimp depletion rates are similar to age 35
        and DIVERGE afterwards: humans continue falling steeply while the change
        is much less steep in chimpanzees.  This post-35 acceleration — Phase 2 —
        is a human-derived character, absent or less intense in chimps.

    (3) Grandmothering selection predicts exactly this change.  The selection
        coefficient for advancing menopause 3–5 yr relative to chimp-equivalent
        dynamics is s ≈ 0.4–0.7% (two grandmothers, full model), above the drift
        threshold for Ne ≥ 100–250.  This is achievable at dialect-tribe or
        regional-group scale.  Phase 2 changes are preferred over Phase 1 changes
        because Phase 2 affects only the last ~10–14 years before pool exhaustion,
        when oocyte quality is already declining — minimal reproductive cost.

    (4) The ABM evolves Tr to approximately 47 yr (just below the atresia ceiling
        of 50 yr) under grandmothering.  The biexponential model shows that
        Tr = 47 yr corresponds to a Phase 2 rate λ₂ ≈ 0.36/yr (+52% above the
        Faddy 1992 baseline of 0.237/yr).  The ABM result is therefore consistent
        with the Phase 2 hypothesis: grandmothering selection has advanced the
        human Phase 2 trigger, but not far enough to reach the OC co-evolutionary
        optimum of Tr ≈ 37–40 yr, for reasons discussed in Section 4 (main ms).

    (5) The quality-control coevolution: the CHK2/p63 checkpoint eliminates
        DNA-damaged oocytes selectively.  More aggressive checkpoint activity
        simultaneously accelerates atresia AND improves quality per ovulation
        (fewer aneuploid eggs survive to be ovulated).  A single allele affecting
        CHK2 activity can therefore advance menopause AND improve reproductive
        quality — both beneficial under grandmothering.  This sign reversal
        (CHEK2 loss-of-function delayed menopause is costly without grandmothering,
        beneficial with it) is a specific empirical prediction.

Calibration sources:
  Faddy MJ et al. 1992 Hum Reprod 7:1342–1346.
  Faddy MJ & Gosden RG. 1995 Hum Reprod 10:770–775.
  Jones KP et al. 2007 Biol Reprod 77:247–251.
  Cloutier CT et al. 2015 GeroScience (Age) 37:9746.
  Wood BM et al. 2023 Science 382:eadd5473.
  Bolcun-Filas E et al. 2014 Science 343:533–536.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "font.size": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False})

# ── Parameters ─────────────────────────────────────────────────────────────
N0_H    = 1_000_000   # Human follicle count at birth
N0_C    =   177_000   # Chimp N0 giving T_m≈52 with human Phase 2 dynamics
                      # (chimp endowment ~18% of human; Cloutier et al. 2015
                      #  show measurement on sections vs whole ovaries may
                      #  account for most of this difference, but a smaller
                      #  chimp initial endowment is also plausible)
N_SWITCH =  25_000    # Phase 2 trigger (Faddy 1992: ~age 37–38 in humans)
N_CRIT   =   1_000    # Menopause threshold (Faddy & Gosden 1995)

LAM1_H  = 0.097       # Human Phase 1 rate (Faddy 1992)
LAM2_H  = 0.237       # Human Phase 2 rate (Faddy 1992)
LAM1_C  = 0.051       # Chimp Phase 1 rate (Cloutier 2015; Jones et al. 2007)

GAIN_PER_YR = 2.23 / 10   # % fitness gain per yr of earlier menopause (two GMs)


# ── Core functions ─────────────────────────────────────────────────────────

def N_at_age(age, lam1, lam2, N0=N0_H):
    age  = np.atleast_1d(np.asarray(age, float))
    t_sw = np.log(N0 / N_SWITCH) / lam1 if N0 > N_SWITCH else 0.0
    N    = np.where(age <= t_sw,
                    N0 * np.exp(-lam1 * age),
                    N_SWITCH * np.exp(-lam2 * (age - t_sw)))
    return np.maximum(N, 0.0)


def Tm(lam1, lam2, N0=N0_H):
    """Age at menopause given biexponential atresia parameters."""
    ts = np.log(N0 / N_SWITCH) / lam1
    return ts + np.log(N_SWITCH / N_CRIT) / lam2


def quality_curve(ages, q0=0.85, k=0.20, a50=36.0):
    return q0 / (1.0 + np.exp(k * (ages - a50)))


def quality_with_checkpoint_aging(ages, alpha_qc=0.0, **kw):
    q_base = quality_curve(ages, **kw)
    eff    = np.clip(1.0 - alpha_qc * np.maximum(ages - 35.0, 0), 0, 1)
    return q_base * eff + (1.0 - eff) * 0.05


# ── Computed quantities ─────────────────────────────────────────────────────
Tm_H  = Tm(LAM1_H, LAM2_H)
Tm_C  = Tm(LAM1_C, LAM2_H, N0=N0_C)
ts_H  = np.log(N0_H / N_SWITCH) / LAM1_H   # Phase 2 trigger age in humans = 38 yr
# λ₂ implied by ABM evolved Tr≈47:
lam2_abm47 = np.log(N_SWITCH / N_CRIT) / (47 - ts_H)


# ── Figure ─────────────────────────────────────────────────────────────────
ages  = np.linspace(0, 65, 400)
ages_r = np.linspace(18, 55, 200)

fig, axs = plt.subplots(2, 3, figsize=(16, 10))

# ─── A: Follicle depletion — human vs chimp ───────────────────────────────
ax = axs[0, 0]
ax.semilogy(ages, N_at_age(ages, LAM1_H, LAM2_H),
            color="#1b6ca8", lw=2.2,
            label=f"Human (Faddy 1992); T_m = {Tm_H:.1f} yr")
ax.semilogy(ages, N_at_age(ages, LAM1_C, LAM2_H, N0=N0_C),
            color="#888888", lw=2.0, ls="--",
            label=f"Chimp model; T_m ≈ {Tm_C:.1f} yr")
ax.axvspan(35, 52, alpha=0.08, color="#d1495b",
           label="Human Phase 2 steeper after 35\n(Cloutier 2015 — Phase 2 signature)")
ax.axhline(N_CRIT,   color="0.5", ls=":", lw=0.8)
ax.axhline(N_SWITCH, color="0.5", ls=":", lw=0.8)
ax.text(0.5, N_CRIT*1.6,    "Menopause threshold", fontsize=8, color="0.4")
ax.text(0.5, N_SWITCH*1.6,  "Phase 2 trigger",     fontsize=8, color="0.4")
ax.text(38.5, 4e4, "Human Phase 2\nstarts ~38 yr", fontsize=8.5, color="#1b6ca8")
ax.set(xlabel="Age (yr from birth)", ylabel="Follicle count (log scale)",
       title="A  Human and chimp follicle depletion\n(~50-yr endpoint ancestrally conserved)",
       xlim=(0, 65), ylim=(500, 1.5e6))
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

# ─── B: Sensitivity T_m vs Δλ₂ ────────────────────────────────────────────
ax = axs[0, 1]
pct_range = np.linspace(-20, 130, 300)
tms_lam2  = [Tm(LAM1_H, LAM2_H*(1+p/100)) for p in pct_range]
ax.plot(pct_range, tms_lam2, color="#d1495b", lw=2.2, label="T_m(λ₂), Phase 1 fixed")
ax.axhline(Tm_H, color="0.5", ls=":", lw=0.8)
ax.axhline(40, color="0.4", ls="--", lw=0.8, alpha=0.7)
ax.text(118, 40.4, "OC optimum\n(two GMs)", fontsize=8, color="0.4", ha="right")
ax.text(118, Tm_H + 0.5, f"Baseline {Tm_H:.1f} yr", fontsize=8, color="0.4", ha="right")
# "Signature zone": λ₂ increases consistent with human Phase 2 advance
pct_signature = (lam2_abm47 / LAM2_H - 1)*100
ax.axvspan(10, pct_signature, alpha=0.12, color="green")
ax.text((10+pct_signature)/2, 53.5,
        f"Signature zone\n(+10–{pct_signature:.0f}%)\nT_m ≈ 47–50 yr",
        ha="center", fontsize=8.5, color="darkgreen")
ax.axvspan(pct_signature, 130, alpha=0.07, color="orange")
ax.text(pct_signature + 15, 53.5, "Co-evo\nrequired",
        ha="center", fontsize=8.5, color="#994400")
ax.axvline(0, color="k", lw=0.5)
ax.set(xlabel="Change in Phase 2 rate λ₂ (%)",
       ylabel="Age at menopause (yr from birth)",
       title="B  T_m sensitivity to Phase 2 rate\n"
             "Signature zone consistent with human-chimp difference",
       xlim=(-20, 130), ylim=(36, 56))
ax.legend(frameon=False, fontsize=9)

# ─── C: Selection coefficients on Phase 2 alleles ─────────────────────────
ax = axs[0, 2]
Ne_range  = np.logspace(1.3, 4.5, 300)
threshold = 1.0 / (2 * Ne_range)
for pct, col, lbl in [
    (15,  "#1b6ca8", f"λ₂ +15%  (ΔT_m ≈ {Tm(LAM1_H,LAM2_H*1.15)-Tm_H:+.1f} yr)"),
    (30,  "#d1495b", f"λ₂ +30%  (ΔT_m ≈ {Tm(LAM1_H,LAM2_H*1.30)-Tm_H:+.1f} yr)"),
    (52,  "#2a9d4a", f"λ₂ +52%  (ABM Tr≈47; ΔT_m ≈ {Tm(LAM1_H,lam2_abm47)-Tm_H:+.1f} yr)"),
    (100, "#e7852a", f"λ₂ ×2    (ΔT_m ≈ {Tm(LAM1_H,LAM2_H*2.0)-Tm_H:+.1f} yr)"),
]:
    l2  = LAM2_H * (1 + pct/100)
    dTm = Tm(LAM1_H, l2) - Tm_H
    s   = GAIN_PER_YR * abs(dTm) / 100
    ax.semilogx(Ne_range, np.full_like(Ne_range, s*100), lw=2.0, label=lbl)
ax.semilogx(Ne_range, threshold*100, "k:", lw=1.2, label="Drift threshold 1/(2Ne)")
ax.fill_between(Ne_range, threshold*100, 8, alpha=0.07, color="green")
for label, Ne, colt in [("Band Ne=30",   30, "#9b2226"),
                         ("Tribe Ne=500", 500, "#ae7b11"),
                         ("Genomic Ne=10k", 10000, "#1b6ca8")]:
    ax.axvline(Ne, color=colt, lw=0.9, ls=":", alpha=0.7)
    ax.text(Ne*1.1, 7.5, label, fontsize=7.5, color=colt, va="top")
ax.set(xlabel="Effective population size Ne",
       ylabel="Selection coefficient  s (%)",
       title="C  Selection on Phase 2 rate alleles vs Ne\n"
             "(two grandmothers, full model)",
       ylim=(0, 8.0))
ax.legend(frameon=False, fontsize=7.5, loc="upper right")

# ─── D: Quality window and checkpoint aging ───────────────────────────────
ax = axs[1, 0]
ax.plot(ages_r, quality_curve(ages_r)*100, "k-", lw=2.2,
        label="Intrinsic oocyte quality (time in meiotic arrest)")
for alph, col, lbl in [(0.015, "#d1495b", "Checkpoint aging α=0.015"),
                        (0.030, "#9b2226", "Checkpoint aging α=0.030 (severe)")]:
    ax.plot(ages_r, quality_with_checkpoint_aging(ages_r, alph)*100,
            color=col, lw=1.8, label=lbl)
ax.axvline(50, color="0.5", ls="--", lw=0.9)
ax.text(49.5, 72, "~50-yr\nquality\nwindow", ha="right", fontsize=8.5, color="0.4")
ax.axvspan(47, 52, alpha=0.12, color="#2a9d4a")
ax.text(49.5, 45, "Phase 2\nadvanced\nregion\n(38→47 yr)", ha="center",
        fontsize=8, color="darkgreen")
ax.set(xlabel="Maternal age (yr)", ylabel="Oocyte viability  q(a)  (%)",
       title="D  Quality window constraint\n"
             "Advancing Phase 2 removes worst-quality years",
       xlim=(18, 55))
ax.legend(frameon=False, fontsize=8.5, loc="lower left")

# ─── E: Phase 2 signature hypothesis — the human-derived window ───────────
ax = axs[1, 1]
# Ancestral: chimp-like (no Phase 2 advance)
ax.semilogy(ages, N_at_age(ages, LAM1_C, LAM2_H, N0=N0_C),
            color="#888888", ls="--", lw=2.0,
            label=f"Chimp-equivalent (T_m≈{Tm_C:.1f} yr)")
# Current human
ax.semilogy(ages, N_at_age(ages, LAM1_H, LAM2_H),
            color="#1b6ca8", lw=2.2,
            label=f"Current human (T_m≈{Tm_H:.1f} yr)\n→ Phase 2 advanced ~{Tm_C-Tm_H:.1f} yr")
# ABM prediction (λ₂ +52%)
ax.semilogy(ages, N_at_age(ages, LAM1_H, lam2_abm47),
            color="#d1495b", lw=1.8, ls="-.",
            label=f"ABM-implied λ₂=0.36 (T_m={Tm(LAM1_H,lam2_abm47):.1f} yr)\n"
                  f"ABM evolves Tr≈47 = further Phase 2 advance")
# Arrow showing direction of selection
ax.annotate("", xy=(48, 5000), xytext=(52.5, 5000),
            arrowprops=dict(arrowstyle="->", color="green", lw=2))
ax.text(50.2, 6500, "Grandmothering\nselects for\nfaster Phase 2",
        ha="center", fontsize=8, color="darkgreen")
ax.axhline(N_CRIT,   color="0.5", ls=":", lw=0.8)
ax.axhline(N_SWITCH, color="0.5", ls=":", lw=0.8)
ax.set(xlabel="Age (yr)", ylabel="Follicle count (log scale)",
       title="E  Phase 2 signature hypothesis:\n"
             "Selection has already advanced Phase 2; further advance expected",
       xlim=(30, 60), ylim=(500, 5e5))
ax.legend(frameon=False, fontsize=8.5)

# ─── F: ABM evolved Tr → implied λ₂ ──────────────────────────────────────
ax = axs[1, 2]
Tr_range = np.linspace(39, 52, 200)
lam2_imp = np.where(Tr_range > ts_H + 0.5,
                    np.log(N_SWITCH/N_CRIT) / (Tr_range - ts_H),
                    np.nan)
ax.plot(Tr_range, lam2_imp, color="#1b6ca8", lw=2.2,
        label="λ₂ implied by evolved Tr")
ax.axhline(LAM2_H, color="0.5", ls=":", lw=0.9)
ax.text(40.0, LAM2_H + 0.01, f"Faddy 1992 baseline\nλ₂ = {LAM2_H}", fontsize=8.5, color="0.4")
ax.axvline(47, color="#d1495b", ls="--", lw=1.2)
ax.text(47.2, 0.38,
        f"ABM Tr≈47 yr\n→ λ₂≈{lam2_abm47:.3f} (+{(lam2_abm47/LAM2_H-1)*100:.0f}%)",
        fontsize=8.5, color="#d1495b")
ax.axvline(52, color="#888888", ls="--", lw=1.0)
ax.text(51.7, 0.22, f"Chimp T_m≈52\n→ λ₂≈{Tm_C:.1f}", fontsize=8, color="#888888", ha="right")
ax.axhline(lam2_abm47, color="#d1495b", ls=":", lw=0.8, alpha=0.6)
ax.set(xlabel="Evolved / predicted age at menopause (yr)",
       ylabel="Implied Phase 2 rate λ₂  (yr⁻¹)",
       title="F  Correspondence between evolved Tr (ABM) and\nimplied Phase 2 rate λ₂ (atresia model)",
       xlim=(39, 53), ylim=(0.15, 0.70))
ax.legend(frameon=False, fontsize=9)

fig.suptitle(
    "The Phase 2 atresia acceleration as a signature of grandmothering selection\n"
    "Hypothesis: the human-specific steepening of follicle depletion after age 35 "
    "reflects grandmothering selection on the Phase 2 rate",
    fontweight="bold", fontsize=11)
fig.tight_layout()
fig.savefig("fig_atresia_model.png", bbox_inches="tight")
print("wrote fig_atresia_model.png")

# ── Numerical summary for manuscript ───────────────────────────────────────
print()
print(f"Tm human = {Tm_H:.1f} yr  |  Tm chimp model = {Tm_C:.1f} yr  "
      f"→  human earlier by {Tm_C-Tm_H:.1f} yr")
print(f"ABM evolved Tr≈47 → implied λ₂ = {lam2_abm47:.3f}/yr "
      f"(+{(lam2_abm47/LAM2_H-1)*100:.0f}% above baseline)")
print()
print("Selection on Phase 2 alleles (two GMs, GAIN_PER_YR={:.3f}%/yr):".format(GAIN_PER_YR))
for pct in [15, 30, 52, 100]:
    l2  = LAM2_H*(1+pct/100)
    dT  = Tm(LAM1_H, l2) - Tm_H
    s   = GAIN_PER_YR * abs(dT) / 100
    print(f"  λ₂+{pct:3d}%: ΔTm={dT:+.1f}yr  s={s*100:.3f}%  Ne_threshold={int(1/(2*s))}")
