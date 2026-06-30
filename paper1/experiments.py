"""Run all optimal-control experiments and save results to results_oc.npz."""
import numpy as np
from model import Params, run_cell, quality_curve, grandmother_benefit, make_grids

CURRENCIES = [(0.0, "R0 (stationary)"), (0.02, "r = +0.02 (growing)"),
              (-0.01, "r = -0.01 (declining)")]


def fitness_at_menopause(rate, B, T_stop, **kw):
    """Lifetime inclusive fitness of a genotype whose menopause is fixed at
    T_stop (reproduce to T_stop, then optimise repair/help)."""
    p = Params(B=B, rate=rate, a_ceiling=int(T_stop), **kw)
    sol, sim = run_cell(p, free_u=False)
    return sol['fitness'], sim['Td'], sim['PRLS']


# ----------------------------------------------------------------------
# 1. BACK END: T_d and PRLS vs grandmothering strength (exogenous T_r=50)
# ----------------------------------------------------------------------
Bgrid = np.linspace(0, 6, 25)
backend = {}
for rate, lab in CURRENCIES:
    Td = np.zeros_like(Bgrid); PR = np.zeros_like(Bgrid)
    for i, B in enumerate(Bgrid):
        _, sim = run_cell(Params(B=B, rate=rate), free_u=False)
        Td[i], PR[i] = sim['Td'], sim['PRLS']
    backend[lab] = dict(Td=Td, PRLS=PR)

# ----------------------------------------------------------------------
# 2. THE MENOPAUSE QUESTION: fitness landscape over menopause age + gradient
# ----------------------------------------------------------------------
Tstops = np.arange(30, 51)          # candidate menopause ages
B_profiles = [0.0, 1.0, 3.0, 6.0]   # plausible (~1) to implausible
front = {}
for rate, lab in CURRENCIES:
    prof = {}
    for B in B_profiles:
        W = np.array([fitness_at_menopause(rate, B, T)[0] for T in Tstops])
        prof[B] = W
    front[lab] = prof

# selection differential on menopause age vs B (fine grid):
# s = (W at optimal T_stop) - (W at ceiling 50), normalised by W(50);
# and the argmax menopause age. Positive s => selection to move OFF the ceiling
# (i.e. to REDUCE menopause age).
Bfine = np.linspace(0, 8, 33)
grad = {}
for rate, lab in CURRENCIES:
    s = np.zeros_like(Bfine); argT = np.zeros_like(Bfine)
    for i, B in enumerate(Bfine):
        W = np.array([fitness_at_menopause(rate, B, T)[0] for T in Tstops])
        W50 = W[-1]
        s[i] = (W.max() - W50) / abs(W50)        # fractional fitness gain from reducing T_r
        argT[i] = Tstops[np.argmax(W)]
    grad[lab] = dict(s=s, argT=argT)

# threshold B at which selection to reduce menopause becomes 'appreciable'
# (define appreciable as a >1% fitness gain AND optimum >=2 yr below ceiling)
THRESH = 0.01
thresholds = {}
for rate, lab in CURRENCIES:
    s = grad[lab]['s']; argT = grad[lab]['argT']
    appreciable = (s > THRESH) & (argT <= 48)
    thresholds[lab] = float(Bfine[appreciable][0]) if appreciable.any() else np.inf

# ----------------------------------------------------------------------
# 3. REACTION NORM for age at death over environment g (g scales production)
# ----------------------------------------------------------------------
ggrid = np.linspace(0.6, 1.7, 23)
reaction = {}
for rate, lab in CURRENCIES:
    Td0 = np.zeros_like(ggrid); Td1 = np.zeros_like(ggrid)
    for i, g in enumerate(ggrid):
        _, s0 = run_cell(Params(B=0.0, rate=rate, g=g), free_u=False)
        _, s1 = run_cell(Params(B=1.0, rate=rate, g=g), free_u=False)
        Td0[i], Td1[i] = s0['Td'], s1['Td']
    reaction[lab] = dict(Td_B0=Td0, Td_B1=Td1)

# ----------------------------------------------------------------------
# 4. EXTRINSIC MORTALITY axis (disposable-soma test): T_d & repair vs mu0
# ----------------------------------------------------------------------
mu0grid = np.linspace(0.006, 0.045, 18)
extrinsic = {}
for rate, lab in [CURRENCIES[0]]:  # R0 is enough to make the point
    Td0 = np.zeros_like(mu0grid); Td1 = np.zeros_like(mu0grid)
    rep0 = np.zeros_like(mu0grid)
    for i, m in enumerate(mu0grid):
        _, s0 = run_cell(Params(B=0.0, rate=rate, mu0=m), free_u=False)
        _, s1 = run_cell(Params(B=1.0, rate=rate, mu0=m), free_u=False)
        Td0[i], Td1[i] = s0['Td'], s1['Td']
        rep0[i] = np.mean(s0['z'][:32])  # mean repair effort over reproductive years
    extrinsic[lab] = dict(Td_B0=Td0, Td_B1=Td1, repair_B0=rep0)

# ----------------------------------------------------------------------
# 5. Representative life-history schedules (for an illustrative figure)
# ----------------------------------------------------------------------
sched = {}
for B in [0.0, 1.0]:
    sol, sim = run_cell(Params(B=B, rate=0.0), free_u=False)
    sched[f"B{B}"] = dict(ages=sol['ages'], u=sim['u'], w=sim['w'], z=sim['z'],
                          l=sim['l'], mu=sim['mu'], q=sol['q'])

# force-of-selection illustration: residual reproductive value past T_r
def force_of_selection(rate, B):
    p = Params(B=B, rate=rate)
    sol, sim = run_cell(p, free_u=False)
    ages = sol['ages']; l = sim['l']
    q = sol['q']; dG = grandmother_benefit(ages, p.a_gm, p.gm_ramp, p.gm_plateau)
    Mmax = p.Mmax
    m = np.where(ages <= p.a_ceiling, Mmax * q, 0.0)
    help_eff = p.B * (p.r_g / p.r_o) * dG  # in own-offspring-equivalents
    disc = np.exp(-rate * (ages - ages[0]))
    integrand = disc * l * (p.r_o * m + p.r_o * help_eff)
    s = np.array([integrand[i:].sum() for i in range(len(ages))])
    return ages, s

fos = {}
for B in [0.0, 1.0]:
    a, s = force_of_selection(0.0, B)
    fos[f"B{B}"] = dict(ages=a, s=s)

# ----------------------------------------------------------------------
# 6. SURVIVAL CURVES vs Gurven & Kaplan (2007), and grandmother efficacy E(a)
# ----------------------------------------------------------------------
from model import grandmother_efficacy, grandmother_benefit

def survstats(l, ages):
    l70 = l[70 - ages[0]]
    deaths = -np.diff(np.concatenate([l, [0]]))
    modal = int(ages[np.argmax(deaths)])
    e45 = l[45 - ages[0]:].sum() / l[45 - ages[0]]
    return float(l70), modal, float(e45)

survival = {}
for B in [0.0, 1.0, 3.0]:
    sol, sim = run_cell(Params(B=B, rate=0.0), free_u=False)
    l70, modal, e45 = survstats(sim['l'], sol['ages'])
    survival[f"B{B}"] = dict(ages=sol['ages'], l=sim['l'], l70=l70, modal=modal, e45=e45)

p0 = Params()
eff_ages = np.arange(40, 96)
efficacy = dict(ages=eff_ages,
                E=grandmother_efficacy(eff_ages, p0.gm_a_full, p0.gm_a_zero, p0.gm_decline_k),
                dG=grandmother_benefit(eff_ages, p0.a_gm, p0.gm_ramp, p0.gm_plateau,
                                       p0.gm_a_full, p0.gm_a_zero, p0.gm_decline_k),
                anchors_age=np.array([50, 60, 70, 80, 90]),
                anchors_E=np.array([1.00, 0.95, 0.75, 0.50, 0.0]))  # user/empirical targets
gurven = dict(l70=0.47, modal_lo=68, modal_hi=78, modal=72, e45_lo=20, e45_hi=22)

np.savez("results_oc.npz",
         Bgrid=Bgrid, backend=backend,
         Tstops=Tstops, B_profiles=np.array(B_profiles), front=front,
         Bfine=Bfine, grad=grad, thresholds=thresholds, THRESH=THRESH,
         ggrid=ggrid, reaction=reaction,
         mu0grid=mu0grid, extrinsic=extrinsic,
         sched=sched, fos=fos,
         survival=survival, efficacy=efficacy, gurven=gurven,
         currencies=[c[1] for c in CURRENCIES],
         allow_pickle=True)

print("=== SUMMARY ===")
print("\nBack-end PRLS (exogenous T_r=50):")
for lab in [c[1] for c in CURRENCIES]:
    print(f"  {lab:24s} B=0:{backend[lab]['PRLS'][0]:.1f}  B=1:{backend[lab]['PRLS'][np.argmin(abs(Bgrid-1))]:.1f}  B=6:{backend[lab]['PRLS'][-1]:.1f}")

print("\nMenopause selection: optimum age at menopause (argmax) and fractional")
print("fitness gain from reducing it below the atresia ceiling:")
for lab in [c[1] for c in CURRENCIES]:
    for B in B_profiles:
        W = front[lab][B]
        argT = Tstops[np.argmax(W)]
        s = (W.max() - W[-1]) / abs(W[-1])
        print(f"  {lab:24s} B={B}: opt T_r={argT}  gain={s*100:.2f}%")
    print()

print("Threshold B for appreciable (>1% gain, opt<=48) downward selection on T_r:")
for lab in [c[1] for c in CURRENCIES]:
    t = thresholds[lab]
    print(f"  {lab:24s}: B* = {t if np.isfinite(t) else 'never in [0,8]'}")
print("\nSaved results_oc.npz")
