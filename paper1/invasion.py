"""
Mutant-invasion experiment for the individual-based model.

The full eco-evolutionary ABM sits near a SHORT-life equilibrium: post-
reproductive selection is weak (Hamilton's declining force of selection), so in
a finite population genetic drift holds maintenance well below the deterministic
optimum and equilibrium PRLS is small. That is a statement about drift, not
about the DIRECTION of selection. The two take-home messages are claims about
selection, so we measure selection directly: hold a resident genotype, seed a
competing variant differing in ONE axis, switch mutation off, start near 50:50
to suppress drift, and read the selection coefficient s = slope of logit(mutant
frequency) over time. We sweep grandmothering strength B and average many seeds.

  Message 2 (back end):  a LONG-LIFE allele (more somatic maintenance during and
      after reproduction -> higher T_d and PRLS, at a direct fecundity cost)
      competes against a short-life allele. s>0 and rising in B  =>  grand-
      mothering selects for longer post-reproductive life.

  Message 1 (front end): an EARLIER-cessation allele (lower T_r) competes against
      cessation at the atresia ceiling, in a population that survives to help.
      s ~ 0 (and weakly B-dependent)  =>  the menopause age is ~selectively
      neutral; running it with / without the maternal-mortality cost shows how
      far the late-reproduction cost shifts that near-neutrality.

Mechanism matches abm.py: grandmother help is a CONTINUOUS boost to a helped
daughter's fecundity (Hawkes provisioning), and the maternal-mortality hazard
tracks the realised (q-discounted) birth rate x per-birth risk (Saccone 2022).
"""
import numpy as np
from model import (quality_curve, grandmother_benefit, maternal_mortality,
                   productivity_curve, Params)


def run_invasion(focal, resident, mutant, B, B_pgm=0.0, p_u=0.05,
                 years=1200, K=3000, seed=0,
                 fixed=None, p: Params = None, init_mut=0.5):
    """Two fixed genotypes competing under clonal inheritance, no mutation.

    focal = 'long' : mutant scales somatic maintenance (soma & ppost) by a factor
                     -> lives longer; resident/mutant are the scale factors.
    focal = 'Tr'   : mutant has an earlier menopause; resident/mutant are T_r.
    Returns (s, freq_trace).
    """
    rng = np.random.default_rng(seed)
    if p is None:
        p = Params()
    a_mat, ceiling, a_max = p.a_mat, p.a_ceiling, p.a_max
    qc  = np.zeros(a_max + 1)
    qc[a_mat:] = quality_curve(np.arange(a_mat, a_max + 1), p.q0, p.qk, p.qa50)
    # Age-graded productivity (Kaplan et al. 2000): scales birth yield and GM help
    gc  = productivity_curve(np.arange(0, a_max + 1)) * p.g if p.use_prod_curve \
          else np.full(a_max + 1, p.g)
    dG  = grandmother_benefit(np.arange(0, a_max + 1), p.a_gm, p.gm_ramp, p.gm_plateau,
                              p.gm_a_full, p.gm_a_zero, p.gm_decline_k)
    pmat = maternal_mortality(np.arange(0, a_max + 1), p.mm_base, p.mm_k, p.mm_a0, p.mm_floor)

    # baseline genotype; the focal axis is overwritten per individual below
    base = dict(soma=0.12, ppost=0.20, Tr=float(ceiling), surv_cost=0.03)
    base.update(fixed or {})

    def build(is_m):
        Tr = np.full(len(is_m), base["Tr"], float)
        soma = np.full(len(is_m), base["soma"], float)
        ppost = np.full(len(is_m), base["ppost"], float)
        survmult = np.ones(len(is_m))   # post-repro mortality multiplier (<1 = longer life)
        fcost = np.ones(len(is_m))      # fecundity multiplier (cost of the longevity phenotype)
        if focal in ("long", "soma", "ppost"):
            f = np.where(is_m, mutant, resident)             # maintenance scale
            if focal in ("long", "soma"):
                soma = np.clip(base["soma"] * f, 0.02, 0.9)
            if focal in ("long", "ppost"):
                ppost = np.clip(base["ppost"] * f, 0.0, 0.9)
        elif focal == "surv":
            # CLEAN Message-2 test: the mutant simply survives the post-reproductive
            # phase better (lower mortality), helps at the SAME rate as everyone, and
            # pays a fixed fecundity cost during reproduction for the phenotype.
            # resident/mutant are the post-repro mortality multipliers (e.g. 1.0 vs 0.5).
            survmult = np.where(is_m, mutant, resident)
            fcost = np.where(is_m, 1.0 - base["surv_cost"], 1.0)
        elif focal == "Tr":
            Tr = np.where(is_m, mutant, resident)
        return Tr, soma, ppost, survmult, fcost

    N0 = K
    age = rng.integers(a_mat, ceiling, N0).astype(float)
    D = np.maximum(0.0, (age - a_mat) * p.delta_base * 0.4)
    is_mut = rng.random(N0) < init_mut
    Tr, soma, ppost, survmult, fcost = build(is_mut)
    ids = np.arange(N0); nid = N0
    mother = np.full(N0, -1, dtype=np.int64)
    gmother = np.full(N0, -1, dtype=np.int64)
    pgmother = np.full(N0, -1, dtype=np.int64)   # paternal grandmother
    juv_scale = 1.0
    freq = []

    for yr in range(years):
        Nn = len(age)
        if Nn == 0 or is_mut.all() or (~is_mut).all():
            break
        aint = np.clip(age.astype(int), 0, a_max)
        Te = np.minimum(Tr, ceiling)
        repro = (age >= a_mat) & (age <= Te)
        post = (age > Te) & (age >= a_mat)
        repair = np.where(repro, soma, np.where(post, ppost, 0.0))
        # Kirkwood δ(u): effort ≈ (1-soma) during reproduction, 0 post-repro
        effort  = np.where(repro, 1.0 - soma, 0.0)
        delta_u = p.delta_base + p.delta_rep * effort**2
        D = np.clip(D + delta_u - p.rho * repair, 0.0, p.Dmax)
        gomp = p.mu2 * (np.exp(p.gamma * np.maximum(age - a_mat, 0)) - 1.0)
        mu = p.mu0 + p.mu1 * D + gomp
        # age-graded productivity scales the effective birth hazard
        birth_haz = p.Mmax * gc[aint] * qc[aint] * (1.0 - soma)
        mu = mu + np.where(repro, p.mm_scale * birth_haz * pmat[aint], 0.0)
        mu = np.where(post, mu * survmult, mu)                 # 'surv' longevity allele
        mu = np.where(age >= a_mat, mu, p.mu0 + 0.02)
        mu = np.where(age >= a_max, 50.0, mu)
        psurv = np.exp(-mu)
        juv = age < a_mat
        psurv[juv] = np.clip(psurv[juv] * juv_scale, 0, 1)
        alive = rng.random(Nn) < psurv

        elig = repro & alive
        fert = np.ones(Nn)
        if B > 0:
            posm  = np.clip(np.searchsorted(ids, mother), 0, Nn - 1)
            gm_ok = (mother >= 0) & (ids[posm] == mother) & post[posm]
            gma   = np.clip(age[posm].astype(int), 0, a_max)
            # grandmother help scaled by her age-graded productivity gc[gma]
            fert = np.where(gm_ok,
                            1.0 + B * gc[gma] * dG[gma] * (1.0 - ppost[posm]),
                            1.0)
        # birth rate: age-graded productivity scales the effective Mmax
        if B_pgm > 0:
            r_g_pgm = p.r_g * (1.0 - p_u)
            posp   = np.clip(np.searchsorted(ids, pgmother), 0, Nn - 1)
            pgm_ok = ((pgmother >= 0) & (ids[posp] == pgmother)
                      & post[posp] & alive[posp])
            pgma   = np.clip(age[posp].astype(int), 0, a_max)
            fert   = np.where(pgm_ok,
                               fert + B_pgm * r_g_pgm / p.r_g
                                 * gc[pgma] * dG[pgma] * (1.0 - ppost[posp]),
                               fert)
        brate = np.where(elig,
                         p.Mmax * gc[aint] * qc[aint] * (1.0 - soma) * fert * fcost,
                         0.0)
        births = rng.random(Nn) < brate
        bi = np.where(births)[0]
        if len(bi) > 0:
            S_dep = np.exp(-mu[bi] * p.L_dep)
            bi = bi[rng.random(len(bi)) < np.clip(S_dep, 0, 1)]
        # Assign PGM: sample from post-repro females at birth
        n_pgm_ids = np.full(len(bi), -1, dtype=np.int64)
        if B_pgm > 0 and len(bi) > 0:
            cand = np.where(post & alive)[0]
            if len(cand) > 0:
                chosen    = cand[rng.integers(0, len(cand), len(bi))]
                pat_known = rng.random(len(bi)) >= p_u
                n_pgm_ids = np.where(pat_known, ids[chosen], -1)
        
        if len(bi) > 0:
            cm = is_mut[bi]; cid = np.arange(nid, nid + len(bi)); nid += len(bi)
            n_mother = ids[bi]; n_gmother = mother[bi]

        keep = alive
        pgmother = pgmother[keep]
        age = age[keep] + 1.0; D = D[keep]; Tr = Tr[keep]; soma = soma[keep]
        ppost = ppost[keep]; ids = ids[keep]; mother = mother[keep]
        gmother = gmother[keep]; is_mut = is_mut[keep]
        survmult = survmult[keep]; fcost = fcost[keep]
        if len(bi) > 0:
            cTr, cSoma, cPpost, cSurv, cFcost = build(cm)
            pgmother = np.concatenate([pgmother, n_pgm_ids]).astype(np.int64)
            age = np.concatenate([age, np.zeros(len(cid))])
            D = np.concatenate([D, np.zeros(len(cid))])
            Tr = np.concatenate([Tr, cTr]); soma = np.concatenate([soma, cSoma])
            ppost = np.concatenate([ppost, cPpost]); ids = np.concatenate([ids, cid]).astype(int)
            mother = np.concatenate([mother, n_mother]).astype(int)
            gmother = np.concatenate([gmother, n_gmother]).astype(int)
            is_mut = np.concatenate([is_mut, cm])
            survmult = np.concatenate([survmult, cSurv])
            fcost = np.concatenate([fcost, cFcost])
        N = len(age); juv_scale *= (K / max(N, 1)) ** 0.10
        juv_scale = float(np.clip(juv_scale, 0.2, 3.0))
        ad = age >= a_mat
        if ad.sum() > 0:
            freq.append(float(is_mut[ad].mean()))

    freq = np.array(freq)
    f = np.clip(freq, 1e-4, 1 - 1e-4)
    t0 = max(20, len(f) // 6)
    if len(f) - t0 < 30:
        return np.nan, freq
    logit = np.log(f / (1 - f))
    t = np.arange(len(f))
    s = np.polyfit(t[t0:], logit[t0:], 1)[0]
    return float(s), freq


def gradient(focal, resident, mutant, B, fixed, seeds=(1, 2, 3, 4, 5, 6),
             p: Params = None, **kw):
    vals = [run_invasion(focal, resident, mutant, B, seed=s, fixed=fixed, p=p, **kw)[0]
            for s in seeds]
    vals = np.array(vals, float)
    return float(np.nanmean(vals)), float(np.nanstd(vals) / max(1, np.sqrt(np.isfinite(vals).sum())))


if __name__ == "__main__":
    import json, time
    from model import Params
    t = time.time()
    Bs = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
    SEEDS = tuple(range(1, 9))
    YEARS, K = 1000, 2600
    out = {"B": Bs}

    # ---- MESSAGE 2: selection on post-reproductive survival itself ----------
    # A mutant that simply survives the post-reproductive phase better (mortality
    # x0.5), helps at the SAME per-year rate as the resident, and pays a fixed
    # fecundity cost for the longevity phenotype. s(B=0) = -cost (Hamilton: an
    # extra post-reproductive year is worth nothing); s rises with grandmothering
    # and crosses zero once help-years outweigh the cost.
    print("MESSAGE 2 - selection on post-reproductive survival (mu x0.5, helps at same rate)")
    print(" cost    " + "   ".join(f"B={B:.0f}" for B in Bs))
    out["s_surv"] = {}
    for c in (0.010, 0.020, 0.030):
        row = [gradient("surv", 1.0, 0.5, B, fixed=dict(soma=0.14, ppost=0.30, surv_cost=c),
                        seeds=SEEDS, years=YEARS, K=K)[0] for B in Bs]
        out["s_surv"][f"{c:.3f}"] = row
        print(f" {c:.3f}  " + "  ".join(f"{x:+.4f}" for x in row))

    # ---- MESSAGE 1: selection on age at menopause ---------------------------
    # Resident ceases at the atresia ceiling (50); mutant ceases earlier. s>0
    # would favour earlier menopause. Run with the maternal-mortality cost ON.
    print("\nMESSAGE 1 - selection on menopause age (resident Tr=50, maternal cost ON)")
    print("  Tr*   " + "   ".join(f"B={B:.0f}" for B in (1.0, 4.0)))
    p_on = Params(); p_on.mm_scale = 1.0
    out["Tr_grid"] = [48, 47, 46, 44, 42, 40]
    out["s_meno_B1"], out["s_meno_B4"] = [], []
    for trm in out["Tr_grid"]:
        s1 = gradient("Tr", 50.0, float(trm), 1.0, fixed=dict(soma=0.14, ppost=0.30),
                      p=p_on, seeds=SEEDS, years=YEARS, K=K)[0]
        s4 = gradient("Tr", 50.0, float(trm), 4.0, fixed=dict(soma=0.14, ppost=0.30),
                      p=p_on, seeds=SEEDS, years=YEARS, K=K)[0]
        out["s_meno_B1"].append(s1); out["s_meno_B4"].append(s4)
        print(f"  {trm:2d}    {s1:+.4f}     {s4:+.4f}")

    # ---- cost interaction: maternal mortality ON vs OFF at Tr 49->40 --------
    print("\nMaternal-cost interaction (earlier cessation Tr 50->40)")
    out["s_meno40_mmoff"], out["s_meno40_mmon"] = [], []
    for mm, key in ((0.0, "s_meno40_mmoff"), (1.0, "s_meno40_mmon")):
        p = Params(); p.mm_scale = mm
        row = [gradient("Tr", 50.0, 40.0, B, fixed=dict(soma=0.14, ppost=0.30),
                        p=p, seeds=SEEDS, years=YEARS, K=K)[0] for B in Bs]
        out[key] = row
        print(f"  mm={'ON ' if mm else 'OFF'}  " + "  ".join(f"{x:+.4f}" for x in row))

    json.dump(out, open("results_invasion.json", "w"))
    print(f"\nsaved results_invasion.json  [{time.time()-t:.0f}s]")
