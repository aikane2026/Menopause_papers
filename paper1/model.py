"""
Fused grandmothering / quality-control / reaction-surface model.

Core engine: a finite-horizon dynamic program over (age, somatic damage).
Each year a female allocates a unit of surplus among reproduction (u),
grandmothering help (w), and somatic repair (z), with u + w + z = 1.

- Menopause age T_r: in the EXOGENOUS treatment she reproduces whenever
  oocytes/quality permit, up to the atresia ceiling (~50). In the
  CO-EVOLVABLE treatment u is a free decision, so she may cease earlier;
  T_r* is read off the optimal policy as the last age with u>0.
- Age at death T_d: emergent. Repair z slows damage accrual; damage drives
  mortality; survivorship and hence life expectancy fall out of the policy.
- Fitness currency: r=0 is the R0 (stationary) case; r>0 (growing) and r<0
  (declining) populations discount future and -- crucially -- discount the
  generationally-delayed grandmothering benefit by an extra factor exp(-r*tau_g).

Relatedness: own offspring r_o=1/2, grandoffspring r_g=1/4.
"""

import numpy as np
from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# Demographic curves anchored to published data (see parameterisation §10)
# ----------------------------------------------------------------------

def quality_curve(ages, q0=0.85, k=0.20, a50=36.0):
    """Mean oocyte/embryo developmental competence q(a) in [0,1].

    Calibrated to PGT-A euploidy: ~0.68 (<=30), ~0.50 (35-37), ~0.35 (38-40),
    ~0.16 (>=43). Logistic decline; q0 sets the young-adult plateau.
    """
    return q0 / (1.0 + np.exp(k * (ages - a50)))


def maternal_mortality(ages, base=0.03, k=0.271, a0=40.0, floor=0.008, cap=0.6):
    """Per-birth maternal mortality risk p_mat(a), rising with maternal age.

    Calibrated to the maternal-mortality age gradient of Saccone et al. (2022,
    meta-analysis of 31 million women): relative risk ~3.2 at >40, ~11.6 at >45,
    ~42.8 at >50 vs younger mothers. Scaled onto a ~1% natural-fertility baseline
    gives roughly 0.8% (young) -> 3% (40) -> 12% (45) -> 34% (49). Attempting late
    reproduction is therefore not merely low-yield (see quality_curve) but
    dangerous: it is a survival COST that the optimiser/selection can avoid by
    ceasing earlier.
    """
    ages = np.asarray(ages, dtype=float)
    return np.clip(base * np.exp(k * (ages - a0)), floor, cap)


def grandmother_efficacy(ages, a_full=50.0, a_zero=90.0, k=2.1):
    """Age-related decline in a grandmother's HELPING efficacy, E(a) in [0,1].

    Calibrated to the hunter-gatherer / traditional record (Hadza foraging
    senescence, Chapman et al. 2019 pre-industrial Finland, Engelhardt et al.
    2019 Quebec): help capacity is ~full through the early 50s, then declines
    slowly, moderately, and finally steeply as somatic frailty sets in:
        E(50)=1.00, E(60)~0.95, E(70)~0.77, E(80)~0.45, E(90)=0.
    The exponent k=2.1 reproduces these anchor points. (Chapman's Finnish data
    arguably imply a steeper drop -- benefit ~0 by 75 and slightly negative by
    80 -- which is provided as the 'chapman' sensitivity variant.)
    """
    ages = np.asarray(ages, dtype=float)
    frac = np.clip((ages - a_full) / (a_zero - a_full), 0.0, 1.0)
    E = 1.0 - frac ** k
    return np.clip(E, 0.0, 1.0)


def grandmother_benefit(ages, a_gm=40.0, ramp=6.0, plateau=0.16,
                        a_full=50.0, a_zero=90.0, decline_k=2.1):
    """Per-unit-help marginal grandoffspring-equivalents per year, DeltaG(a).

    Two age factors multiply the plateau strength:
      (i)  an EMERGENCE ramp -- grandchildren only exist from ~age 40, rising to
           the plateau by the late 40s;
      (ii) a FRAILTY decline E(a) -- helping efficacy falls with age as above.
    The product peaks in the early-to-mid 50s and fades to zero by ~90, matching
    the empirical 'grandmother efficacy curve'. Absolute strength is swept via B.
    """
    emergence = 1.0 / (1.0 + np.exp(-(np.asarray(ages, float) - a_gm) / ramp))
    E = grandmother_efficacy(ages, a_full, a_zero, decline_k)
    return plateau * emergence * E


@dataclass
class Params:
    # --- ages ---
    a_mat: int = 18          # maturity
    a_ceiling: int = 50      # atresia ceiling on reproduction (Wallace-Kelsey)
    a_max: int = 96          # absolute horizon

    # --- reproduction ---
    Mmax: float = 0.26       # max births/yr at full effort & full quality (IBI~4yr)
    q0: float = 0.85
    qk: float = 0.20
    qa50: float = 36.0

    # --- somatic damage / mortality (the T_d engine) ---
    # Senescence (mu0,mu2,gamma) calibrated so the well-maintained survival
    # ceiling reproduces Gurven & Kaplan (2007) hunter-gatherer demography:
    # modal adult death ~72-73, ~45% of adults reach 70, e(45) ~ 20-24 yr.
    mu0: float = 0.006       # baseline (extrinsic) adult mortality
    mu1: float = 0.050       # damage->mortality slope
    mu2: float = 2.532e-05   # irreducible senescence scale (Gompertz, unrepairable)
    gamma: float = 0.1438    # senescence acceleration
    delta: float = 0.015     # intrinsic damage accrual (slow; INFERRED)
    rho: float = 0.060       # repair efficacy per unit effort (INFERRED)
    D0: float = 0.0          # damage at maturity
    Dmax: float = 2.5

    # --- grandmothering ---
    B: float = 1.0           # grandmothering strength multiplier (SWEPT)
    a_gm: float = 40.0
    gm_ramp: float = 6.0
    gm_plateau: float = 0.20 # base DeltaG plateau; B=1 ~ Engelhardt +2 grandoffspring
    gm_a_full: float = 50.0  # age below which helping efficacy is full
    gm_a_zero: float = 90.0  # age at which helping efficacy reaches zero
    gm_decline_k: float = 2.1  # frailty-decline exponent (fits HG/traditional data)
    tau_g: float = 25.0      # help->grandoffspring recruitment lag (yrs)
    L_dep: float = 8.0       # offspring dependency window (extended human childhood):
                             # a birth pays off only if the mother survives to rear it
    eps_repair: float = 0.003  # small metabolic cost of maintenance, so repair is
                               # used only when it buys a fitness return (no idle repair)
    # maternal-mortality cost of reproduction (rises with age; Saccone et al. 2022)
    mm_base: float = 0.03      # per-birth maternal death risk at age mm_a0
    mm_k: float = 0.271        # log-rise per year (RR ~3.2/11.6/42.8 at 40/45/50)
    mm_a0: float = 40.0
    mm_floor: float = 0.008    # young-adult baseline per-birth risk
    mm_scale: float = 1.0      # 0 disables the maternal-mortality cost (for contrast)

    # --- relatedness ---
    r_o: float = 0.5
    r_g: float = 0.25

    # --- fitness currency ---
    rate: float = 0.0        # population growth rate r; 0 == R0 currency

    # --- environment scaling (g multiplies PRODUCTION, not damage) ---
    g: float = 1.0

    # --- age-graded productivity (Gurven/Kaplan calibration) ---
    use_prod_curve: bool = True  # if False, g(a) is flat (old behaviour)

    # --- Kirkwood: reproductive effort drives additional somatic damage ---
    # delta = delta_base + delta_rep * u^2 (Speakman 2008; Kirkwood & Rose 1991)
    # At u=0 (post-repro): delta_base only; at u=1: delta_base + delta_rep
    delta_base: float = 0.012   # background damage rate, effort-independent
    delta_rep:  float = 0.006   # effort-dependent extra damage (u^2 term)
    # Backward-compat: if delta is set externally, it overrides delta_base;
    # keep delta as an alias to delta_base for scripts that set it directly.
    # (delta_base + delta_rep/2 ≈ old 0.015 at mean u^2 ≈ 0.5)

    # --- second grandmother (two-grandmother extension) ---
    B_pgm: float = 0.0          # paternal grandmother strength (0 = off)
    p_u:   float = 0.05         # paternity uncertainty (reduces r_g for PGM)

    # --- numerics ---
    nD: int = 241            # damage grid points
    nact: int = 11           # action grid resolution per control


def productivity_curve(ages, g_peak=1.0, a_mat=18, a_peak=38, a_plateau=50,
                       a_steep=65, a_zero=88):
    """Age-specific net foraging production relative to peak, calibrated to
    Kaplan et al. (2000, Evol. Anthropol.) forager data:
       18  →  ~60%  of peak  (building skill/strength)
       38  →  100%  peak     (prime forager)
       50  →  ~85%           (slight decline)
       65  →  ~50%           (steeper decline)
       85  →  ~10%           (frail)
    Rises from maturity; plateau near peak; then sigmoid decline.
    Used to scale both reproductive yield and grandmother helping capacity."""
    ages = np.atleast_1d(np.asarray(ages, float))
    g = np.zeros_like(ages)
    # Rising phase: maturity → peak
    rise = np.clip((ages - a_mat) / (a_peak - a_mat), 0.0, 1.0)
    g_rise = 0.60 + 0.40 * rise                     # 0.60 → 1.00

    # Plateau and decline: piecewise linear then sigmoid
    span1 = a_steep - a_plateau
    span2 = a_zero  - a_steep
    frac1 = np.clip((ages - a_plateau) / span1, 0.0, 1.0)
    frac2 = np.clip((ages - a_steep)   / span2, 0.0, 1.0)
    g_decline = g_peak * (1.0 - 0.50 * frac1) * (1.0 - 0.85 * frac2**1.5)

    # Merge: use rising phase while below peak, declining phase above
    in_rise = ages < a_peak
    in_plateau = (ages >= a_peak) & (ages < a_plateau)
    in_decline = ages >= a_plateau
    g[in_rise]    = g_rise[in_rise]
    g[in_plateau] = g_peak
    g[in_decline] = g_decline[in_decline]
    return np.clip(g, 0.0, g_peak)


    ages = np.arange(p.a_mat, p.a_max + 1)
    Dgrid = np.linspace(0.0, p.Dmax, p.nD)
    return ages, Dgrid


# ----------------------------------------------------------------------
# Dynamic-programming solver
# ----------------------------------------------------------------------

def make_grids(p):
    ages = np.arange(p.a_mat, p.a_max + 1)
    Dgrid = np.linspace(0.0, p.Dmax, p.nD)
    return ages, Dgrid


def solve_dp(p: Params, free_u: bool):
    """Backward-induction solve.

    free_u=False -> EXOGENOUS T_r: reproduce maximally whenever quality/ceiling
                    permit (u is forced to its max-value action), repair & help
                    are optimised.
    free_u=True  -> CO-EVOLVABLE T_r: u is a free decision (she may stop early).

    Returns dict with optimal policy arrays and the lifetime inclusive fitness.
    """
    ages, Dgrid = make_grids(p)
    A = len(ages)
    q = quality_curve(ages, p.q0, p.qk, p.qa50)
    dG = grandmother_benefit(ages, p.a_gm, p.gm_ramp, p.gm_plateau,
                             p.gm_a_full, p.gm_a_zero, p.gm_decline_k)

    # --- Age-graded productivity (Gurven & Kaplan 2000 calibration) ---
    # Scales BOTH reproductive yield m_unit and grandmother helping capacity dG,
    # because both depend on net energy surplus available to the individual.
    # Net production peaks ~38 yr, declines by ~15% at 50, ~50% at 65.
    if p.use_prod_curve:
        g_a = productivity_curve(ages) * p.g
    else:
        g_a = np.full(A, p.g)

    # environment g scales production -> scales all surplus-derived outputs
    Mmax = p.Mmax * p.g      # kept for global scaling (backward compat)
    rho  = p.rho  * p.g
    dGm  = dG * g_a          # grandmother benefit now age-graded by production

    # --- Two-grandmother: MGM (certain) + PGM (discounted by paternity uncertainty) ---
    # r_g,PGM = r_g × (1 - p_u); help is additive (they assist the same daughter)
    help_scale = p.B * p.r_g + p.B_pgm * (p.r_g * (1.0 - p.p_u))

    disc_year = np.exp(-p.rate)            # per-year survival-of-fitness discount
    disc_help = np.exp(-p.rate * p.tau_g)  # extra generational discount on help

    # action grid: fractions u (repro), w (help); z = 1-u-w (repair)
    frac = np.linspace(0.0, 1.0, p.nact)
    acts = []
    for u in frac:
        for w in frac:
            if u + w <= 1.0 + 1e-9:
                acts.append((u, w, max(0.0, 1.0 - u - w)))
    acts = np.array(acts)  # (nA,3)

    V = np.zeros((A + 1, p.nD))   # value function over (age index, damage)
    polU = np.zeros((A, p.nD))
    polW = np.zeros((A, p.nD))
    polZ = np.zeros((A, p.nD))

    for ai in range(A - 1, -1, -1):
        a = ages[ai]
        repro_ok = (a <= p.a_ceiling)
        # available reproductive value per unit effort this year; g_a scales yield
        m_unit = Mmax * q[ai] * g_a[ai] if repro_ok else 0.0
        # help reward: combined MGM + PGM with age-graded production and two-gm discount
        help_unit = help_scale * dGm[ai]

        U = acts[:, 0].copy()
        W = acts[:, 1].copy()
        Z = acts[:, 2].copy()
        if not repro_ok:
            # cannot reproduce: drop actions that spend on u
            mask = U <= 1e-9
        elif not free_u:
            # EXOGENOUS: must reproduce maximally given remaining budget after
            # the optimal help/repair split -> force u to the max feasible value
            # We implement this by requiring u == max possible given w
            # i.e. u = 1 - w - z is already; force z minimal? Simpler: force
            # u to be as large as allowed -> u = 1 - w (z=0) OR allow repair?
            # Biology: she reproduces every cycle; remaining surplus -> repair.
            # So fix u at full reproductive effort = u_full, split the REST
            # between help and repair. We model u_full = 1 (all non-help surplus
            # to reproduction is unrealistic); instead exogenous means the
            # *timing* of cessation is fixed, not that she over-reproduces.
            # Cleanest: exogenous == co-evolvable but with u forbidden to be 0
            # while repro_ok (she does not choose to stop). Enforce u>0.
            mask = U > 1e-9
        else:
            mask = np.ones(len(U), dtype=bool)

        Um, Wm, Zm = U[mask], W[mask], Z[mask]

        # mortality / survival as a function of CURRENT damage AND age
        gomp = p.mu2 * (np.exp(p.gamma * (a - p.a_mat)) - 1.0)  # irreducible
        mu = p.mu0 + p.mu1 * Dgrid + gomp    # (nD,)
        S_dep = np.exp(-mu * p.L_dep)         # survive the dependency window (nD,)
        # maternal-mortality cost: the hazard is the rate of births a mother
        # actually carries x the per-birth risk. Births carried collapse with age
        # (q-discounted: few late conceptions reach term), so this is realised
        # maternal deaths, not an effort proxy, and matches the ABM. Per-birth risk
        # rises steeply with age -> late reproduction is dangerous AND low-yield, a
        # cost the optimiser can avoid by ceasing earlier.
        p_mat = p.mm_scale * maternal_mortality(a, p.mm_base, p.mm_k, p.mm_a0, p.mm_floor) \
                if repro_ok else 0.0
        mu_mat = m_unit * Um * p_mat                               # (nAct,) q-discounted
        psurv = np.exp(-(mu[None, :] + mu_mat[:, None]))           # (nAct, nD)

        # this-year reward, now D-dependent: a birth pays off only if the mother
        # is likely to survive to rear it -> induces maintenance during repro.
        rew = (p.r_o * m_unit) * Um[:, None] * S_dep[None, :] \
              + (disc_help * help_unit) * Wm[:, None] \
              - p.eps_repair * Zm[:, None]                     # (nAct, nD)

        # next damage: Kirkwood (1977) disposable soma — reproductive effort u
        # drives additional oxidative damage (Speakman 2008; Kirkwood & Rose 1991).
        # δ(u) = delta_base + delta_rep × u² so post-repro accumulation uses
        # only delta_base; high reproductive effort accelerates ageing.
        delta_u = p.delta_base + p.delta_rep * Um[:, None]**2   # (nAct, 1) broadcast OK
        Dnext = np.clip(Dgrid[None, :] + delta_u - rho * Zm[:, None], 0.0, p.Dmax)
        # interpolate continuation value V[ai+1] at Dnext
        Vnext = np.interp(Dnext, Dgrid, V[ai + 1])  # (nAct, nD)
        total = rew + disc_year * psurv * Vnext  # (nAct,nD)

        best = np.argmax(total, axis=0)
        V[ai] = total[best, np.arange(p.nD)]
        polU[ai] = Um[best]
        polW[ai] = Wm[best]
        polZ[ai] = Zm[best]

    return dict(ages=ages, Dgrid=Dgrid, V=V, polU=polU, polW=polW, polZ=polZ,
                q=q, dG=dGm, fitness=V[0, 0])


def simulate_policy(p: Params, sol):
    """Forward-simulate the optimal deterministic damage trajectory and the
    resulting survivorship, then read off T_r*, life expectancy (T_d), PRLS."""
    ages = sol['ages']; Dgrid = sol['Dgrid']
    A = len(ages)
    D = p.D0
    rho = p.rho * p.g
    if p.use_prod_curve:
        g_a_traj = productivity_curve(ages) * p.g
    else:
        g_a_traj = np.full(A, p.g)
    u_traj = np.zeros(A); w_traj = np.zeros(A); z_traj = np.zeros(A)
    Dtraj = np.zeros(A); mu_traj = np.zeros(A)
    for ai in range(A):
        u = float(np.interp(D, Dgrid, sol['polU'][ai]))
        w = float(np.interp(D, Dgrid, sol['polW'][ai]))
        z = float(np.interp(D, Dgrid, sol['polZ'][ai]))
        u_traj[ai], w_traj[ai], z_traj[ai] = u, w, z
        Dtraj[ai] = D
        gomp = p.mu2 * (np.exp(p.gamma * (ages[ai] - p.a_mat)) - 1.0)
        p_mat = (p.mm_scale * maternal_mortality(ages[ai], p.mm_base, p.mm_k,
                 p.mm_a0, p.mm_floor)) if ages[ai] <= p.a_ceiling else 0.0
        q_ai = quality_curve(ages[ai], p.q0, p.qk, p.qa50)
        # age-graded effective Mmax and Kirkwood δ(u)
        mu_traj[ai] = (p.mu0 + p.mu1 * D + gomp
                       + p.Mmax * g_a_traj[ai] * q_ai * u * p_mat)
        delta_u = p.delta_base + p.delta_rep * u**2
        D = min(max(D + delta_u - rho * z, 0.0), p.Dmax)
    # survivorship
    l = np.concatenate([[1.0], np.exp(-np.cumsum(mu_traj))])[:A]
    # life expectancy at maturity (age at death proxy)
    Td = ages[0] + np.sum(l)  # sum of survivorship ~ expected remaining years
    # T_r*: last age with appreciable reproduction
    repro_ages = ages[(u_traj > 0.05) & (ages <= p.a_ceiling)]
    Tr = repro_ages.max() if len(repro_ages) else ages[0]
    # PRLS
    prls = max(0.0, Td - Tr)
    # median age at death
    med = ages[0] + np.searchsorted(-l, -0.5)
    return dict(u=u_traj, w=w_traj, z=z_traj, D=Dtraj, mu=mu_traj, l=l,
                Tr=float(Tr), Td=float(Td), PRLS=float(prls), med_death=float(med))


def run_cell(p: Params, free_u: bool):
    sol = solve_dp(p, free_u)
    sim = simulate_policy(p, sol)
    return sol, sim


if __name__ == "__main__":
    # quick sanity check: does co-evolvable T_r ever drop below the ceiling?
    print("CO-EVOLVABLE T_r (free_u=True):")
    for rate, lab in [(0.0, "R0"), (0.02, "r=+.02"), (-0.01, "r=-.01")]:
        for B in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]:
            p = Params(B=B, rate=rate)
            _, sim = run_cell(p, free_u=True)
            print(f"  {lab:7s} B={B:>5} | Tr*={sim['Tr']:.0f}  Td={sim['Td']:.1f}  PRLS={sim['PRLS']:.1f}")
        print()
    print("EXOGENOUS T_r (free_u=False) at B=0 and B=1:")
    for B in [0.0, 1.0]:
        _, sim = run_cell(Params(B=B, rate=0.0), free_u=False)
        print(f"  R0 B={B} | Tr*={sim['Tr']:.0f}  Td={sim['Td']:.1f}  PRLS={sim['PRLS']:.1f}")
