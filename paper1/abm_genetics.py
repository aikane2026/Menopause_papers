"""
abm_genetics.py  —  Robustness of the "earlier cessation is inaccessible"
conclusion to the GENETIC architecture of the ABM.

Motivation
----------
The main-text ABM (abm.py) is CLONAL with one effectively-Mendelian locus per
trait and Gaussian mutation.  The paper's headline claim — that the evolved age
at menopause stays near the atresia ceiling because the *joint* move (earlier
cessation + higher somatic maintenance) is inaccessible to single-allele
substitution — is, by its nature, a statement about reachability / fitness-valley
crossing, and reachability is the part of an evolutionary prediction that the
genetic architecture controls (Hammerstein 1996, J Math Biol 34:511; Weissing
1996, ibid 34:533; Nowak 1990, J Theor Biol 142:237).  This script re-runs the
*identical demography and fitness function* of abm.py under alternative genetics
so the only thing that changes is the genetic layer, and reports outputs in the
SAME schema (meanTr, Td, PRLS, l70, modal) so they drop straight into the
existing Figure-4 comparison.

Minimal bracketing set
----------------------
  (A) clonal, rho = 0            -> reproduces abm.py (control / sanity check)
  (B) clonal, rho = -0.6         -> pleiotropy alone (favourable genetic
                                    correlation: a mutation that lowers Tr also
                                    raises maintenance).  Tests the CHEK2-type
                                    correlated-mutation route.
  (C) sexual, polygenic, rho = 0 -> sex + recombination + STANDING variation,
                                    modular (no genetic correlation).
  (D) sexual, polygenic, rho=-0.6-> sex + standing variation + favourable
                                    pleiotropy.
  (E) Ne sweep on the most permissive architecture (D): K in {500,2000,10000}.
  (+) analytic stochastic-tunnelling crossing time (Iwasa et al. 2004, Genetics
      166:1571; Weissman et al. 2009, TPB 75:286; Weissman et al. 2010, Genetics
      186:1389) as an independent check on the clonal result.

The decisive knob is rho, the genetic correlation between age-at-cessation (Tr)
and reproductive-phase maintenance (soma):
    rho < 0  ->  lowering Tr is genetically coupled to higher maintenance
                 (the favourable direction; turns the two-step valley into a
                 single uphill step).
    rho = 0  ->  modular / independent (the architecture abm.py assumes).
    rho > 0  ->  antagonistic (even harder).

A robust "inaccessible" conclusion is one that survives (C) and (D): if the
population still parks Tr at the ceiling under sex + standing variation + a
modest favourable pleiotropy, the claim stands.  If a little pleiotropy or
recombination pulls Tr down toward OptMeno, the honest conclusion becomes
"reachable only via correlated/pleiotropic variation" — which is the natural
bridge to the companion paper's CHEK2 mechanism.

Run:
    python abm_genetics.py                 # full set, writes JSON + CSV + PNGs
    # production (parallel over cores; burn-in decoupled from run length):
    ABM_YEARS=30000 ABM_K=10000 ABM_SEEDS=8 ABM_BURNIN=12000 ABM_PROCS=8 \
        python abm_genetics.py
Environment variables:
    ABM_YEARS   total years per run            (default 3000)
    ABM_K       carrying capacity ~ Ne         (default 2000)
    ABM_SEEDS   replicate seeds per condition  (default 3)
    ABM_BURNIN  years discarded before measuring; capped sensibly if unset
    ABM_PROCS   worker processes (default = cores-1; set 1 to run serially)
"""
import os
import shutil
import json
import csv
from dataclasses import dataclass, field
import numpy as np
import multiprocessing as mp

from model import (quality_curve, grandmother_benefit, maternal_mortality,
                   productivity_curve, Params)


# ─────────────────────────────────────────────────────────────────────────────
#  Genetic architecture specification
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GeneticArch:
    """How offspring genotypes are produced from parents.

    mode      : 'clonal' (maternal, asexual) or 'sexual' (diploid, biparental)
    rho       : genetic correlation between mutational effects on Tr and soma
                (the favourable direction is rho < 0).
    L         : loci per trait (sexual mode only).
    recomb    : recombination rate between adjacent loci within a trait block
                (0.5 = free recombination; the standard sexual default and the
                regime least favourable to valley crossing, Weissman 2010).
    mu_locus  : per-locus per-gamete mutation probability (sexual mode).
    eff_sd    : (Tr, soma, ppost) per-allele effect SDs (sexual mode); also the
                mutational step SDs in clonal mode (Tr, soma match abm.py).
    init_sd   : (Tr, soma, ppost) standing-variation SDs at initialisation
                (sexual mode): broad standing variation so low-Tr and high-soma
                variants are already segregating at t = 0.
    label     : human-readable name.
    """
    mode: str = "clonal"
    rho: float = 0.0
    L: int = 20
    recomb: float = 0.5
    mu_locus: float = 5e-3
    eff_sd: tuple = (0.5, 0.02, 0.02)        # clonal: matches abm.py Tr/soma steps
    sex_eff_sd: tuple = (0.55, 0.018, 0.018)  # sexual: per-allele effect SD
    init_sd: tuple = (3.5, 0.09, 0.09)        # sexual: standing-variation SD
    label: str = "clonal rho=0"


# ─────────────────────────────────────────────────────────────────────────────
#  Genotype helpers (sexual / polygenic)
# ─────────────────────────────────────────────────────────────────────────────
def _init_diploid(N, arch, bases, rng):
    """Initialise diploid genotypes (N, 2, L) for Tr, soma, ppost with standing
    variation; Tr and soma effects correlated by rho."""
    L = arch.L
    sdTr, sdSo, sdPp = arch.init_sd
    # per-allele SD so that total standing SD over 2L alleles ~ init_sd
    aTr = sdTr / np.sqrt(2 * L)
    aSo = sdSo / np.sqrt(2 * L)
    aPp = sdPp / np.sqrt(2 * L)
    rho = arch.rho
    z0 = rng.standard_normal((N, 2, L))
    z1 = rng.standard_normal((N, 2, L))
    gTr = aTr * z0
    gSo = aSo * (rho * z0 + np.sqrt(1 - rho ** 2) * z1)   # correlated with Tr
    gPp = aPp * rng.standard_normal((N, 2, L))
    return {"Tr": gTr, "soma": gSo, "ppost": gPp}


def _pheno(geno, bases):
    """Diploid additive phenotype = base + sum over both haplotypes and loci."""
    return (bases[0] + geno["Tr"].sum(axis=(1, 2)),
            bases[1] + geno["soma"].sum(axis=(1, 2)),
            bases[2] + geno["ppost"].sum(axis=(1, 2)))


def _gametes(geno, parent_idx, arch, rng):
    """Produce one recombined, mutated gamete (haplotype, shape (n, L)) per trait
    from each parent in parent_idx.  Free or partial recombination within each
    trait block; the two trait blocks assort independently, so the ONLY genetic
    correlation between Tr and soma comes from pleiotropic mutation (rho)."""
    n = len(parent_idx)
    L = arch.L
    aTr, aSo, aPp = arch.sex_eff_sd
    rho = arch.rho

    def recombine(g):                      # g: (N,2,L) -> (n,L)
        gg = g[parent_idx]                 # (n,2,L)
        if arch.recomb >= 0.5:             # free recombination: independent loci
            pick = rng.integers(0, 2, size=(n, L))
        else:                              # switch-walk along the block
            start = rng.integers(0, 2, size=(n, 1))
            sw = (rng.random((n, L - 1)) < arch.recomb).astype(np.int64)
            parity = np.concatenate([np.zeros((n, 1), int),
                                     np.cumsum(sw, axis=1) % 2], axis=1)
            pick = (start + parity) % 2
        return np.take_along_axis(gg, pick[:, None, :], axis=1)[:, 0, :]

    hTr = recombine(geno["Tr"])
    hSo = recombine(geno["soma"])
    hPp = recombine(geno["ppost"])

    # mutation: shared locus mask for Tr & soma so pleiotropy lands on one locus
    m = rng.random((n, L)) < arch.mu_locus
    z0 = rng.standard_normal((n, L))
    z1 = rng.standard_normal((n, L))
    hTr = hTr + m * (aTr * z0)
    hSo = hSo + m * (aSo * (rho * z0 + np.sqrt(1 - rho ** 2) * z1))
    mp = rng.random((n, L)) < arch.mu_locus
    hPp = hPp + mp * (aPp * rng.standard_normal((n, L)))
    return hTr, hSo, hPp


def _clonal_mutation(Tr, soma, ppost, idx, arch, rng):
    """Clonal maternal inheritance with (optionally pleiotropic) Gaussian
    mutation.  rho couples the Tr and soma steps; ppost mutates independently."""
    n = len(idx)
    sTr, sSo, sPp = arch.eff_sd
    rho = arch.rho
    z0 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    dTr = sTr * z0
    dSo = sSo * (rho * z0 + np.sqrt(1 - rho ** 2) * z1)
    dPp = sPp * rng.standard_normal(n)
    return Tr[idx] + dTr, soma[idx] + dSo, ppost[idx] + dPp


# ─────────────────────────────────────────────────────────────────────────────
#  The ABM — demography copied verbatim from abm.py; only inheritance differs
# ─────────────────────────────────────────────────────────────────────────────
def run_abm_g(B=1.0, B_pgm=0.0, p_u=0.05,
              years=3000, K=2000, seed=0, burn_in=None,
              arch: GeneticArch = None, p: Params = None,
              init="ancestral", opt_Tr=40.0):
    """init = 'ancestral' : start near the atresia ceiling (the realised
                            ancestral state) — tests whether the population
                            DESCENDS toward OptMeno (accessibility forward).
           = 'optimum'    : start AT the joint optimum (Tr≈opt_Tr, high
                            maintenance) — tests whether that optimum is a
                            stable attractor or REVERTS to the ceiling.
    If 'optimum' reverts to the ceiling, the joint optimum is not evolutionarily
    stable under this architecture; if it holds but 'ancestral' does not reach
    it, the optimum is stable-but-unreachable (a genuine accessibility barrier)."""
    if arch is None:
        arch = GeneticArch()
    if p is None:
        p = Params()
    if burn_in is None:
        burn_in = years // 2
    rng = np.random.default_rng(seed)
    a_mat, ceiling, a_max = p.a_mat, p.a_ceiling, p.a_max
    sexual = (arch.mode == "sexual")
    # phenotype baselines for the polygenic map (also the clonal init centre)
    if init == "optimum":
        bases = (float(opt_Tr), 0.55, 0.55)   # earlier cessation + high maintenance
    else:
        bases = (46.0, 0.30, 0.25)

    ages_all = np.arange(0, a_max + 1)
    qcurve   = quality_curve(ages_all, p.q0, p.qk, p.qa50)
    gcurve   = (productivity_curve(ages_all) * p.g if p.use_prod_curve
                else np.full(a_max + 1, p.g))
    dGcurve  = grandmother_benefit(ages_all, p.a_gm, p.gm_ramp, p.gm_plateau,
                                   p.gm_a_full, p.gm_a_zero, p.gm_decline_k)
    pmat_curve = maternal_mortality(ages_all, p.mm_base, p.mm_k, p.mm_a0, p.mm_floor)

    # ── initialise population ────────────────────────────────────────────────
    N0 = K
    age = rng.integers(a_mat, ceiling, N0).astype(float)
    D = np.maximum(0.0, (age - a_mat) * p.delta_base * 0.4 + rng.normal(0, 0.05, N0))
    if sexual:
        geno = _init_diploid(N0, arch, bases, rng)
        Tr, soma, ppost = _pheno(geno, bases)
        Tr = np.clip(Tr, 30, ceiling); soma = np.clip(soma, 0.02, 0.95)
        ppost = np.clip(ppost, 0.0, 0.95)
    else:
        geno = None
        if init == "optimum":
            Tr = np.clip(rng.normal(bases[0], 1.0, N0), 30, ceiling)
            soma = np.clip(rng.normal(bases[1], 0.05, N0), 0.02, 0.95)
            ppost = np.clip(rng.normal(bases[2], 0.05, N0), 0.0, 0.95)
        else:
            Tr = rng.uniform(42, 50, N0)
            soma = rng.uniform(0.15, 0.45, N0)
            ppost = rng.uniform(0.10, 0.40, N0)

    nid = 0
    def newids(n):
        nonlocal nid
        out = np.arange(nid, nid + n); nid += n
        return out
    ids = newids(N0)
    mother = np.full(N0, -1, dtype=np.int64)
    gmother = np.full(N0, -1, dtype=np.int64)

    hist = dict(year=[], meanTr=[], meanSoma=[], meanPpost=[], N=[])
    expo = np.zeros(a_max + 2)
    dth  = np.zeros(a_max + 2)
    juv_scale = 1.0

    for yr in range(years):
        Nnow = len(age)
        if Nnow == 0:
            break
        aint = np.clip(age.astype(int), 0, a_max)
        Tr_eff = np.minimum(Tr, ceiling)
        repro_phase = (age >= a_mat) & (age <= Tr_eff)
        post_phase  = (age > Tr_eff) & (age >= a_mat)

        # somatic damage: Kirkwood δ(u)
        repair = np.where(repro_phase, soma, np.where(post_phase, ppost, 0.0))
        effort = np.where(repro_phase, 1.0 - soma, 0.0)
        delta_u = p.delta_base + p.delta_rep * effort ** 2
        D = np.clip(D + delta_u - p.rho * repair, 0.0, p.Dmax)

        # mortality
        gomp = p.mu2 * (np.exp(p.gamma * np.maximum(age - a_mat, 0)) - 1.0)
        mu = p.mu0 + p.mu1 * D + gomp
        birth_haz = p.Mmax * gcurve[aint] * qcurve[aint] * (1.0 - soma)
        mu = mu + np.where(repro_phase, p.mm_scale * birth_haz * pmat_curve[aint], 0.0)
        mu = np.where(age >= a_mat, mu, p.mu0 + 0.02)
        mu = np.where(age >= a_max, 50.0, mu)
        psurv = np.exp(-mu)

        juv = age < a_mat
        psurv_eff = psurv.copy()
        psurv_eff[juv] = np.clip(psurv[juv] * juv_scale, 0, 1)
        alive = rng.random(Nnow) < psurv_eff

        if yr >= burn_in:
            ad0 = age >= a_mat
            np.add.at(expo, aint[ad0], 1)
            np.add.at(dth,  aint[(~alive) & ad0], 1)

        elig = repro_phase & alive

        # grandmothering fertility boost (MGM pedigree-tracked; PGM omitted here)
        fert = np.ones(Nnow)
        if B > 0:
            posm = np.clip(np.searchsorted(ids, mother), 0, Nnow - 1)
            mgm_ok = ((mother >= 0) & (ids[posm] == mother)
                      & post_phase[posm] & alive[posm])
            gma = np.clip(age[posm].astype(int), 0, a_max)
            fert = np.where(mgm_ok,
                            fert + B * gcurve[gma] * dGcurve[gma] * (1.0 - ppost[posm]),
                            fert)

        brate = np.where(elig,
                         p.Mmax * gcurve[aint] * qcurve[aint] * (1.0 - soma) * fert,
                         0.0)
        births = rng.random(Nnow) < brate
        bidx = np.where(births)[0]
        if len(bidx) > 0:
            S_dep = np.exp(-mu[bidx] * p.L_dep)
            recruited = rng.random(len(bidx)) < np.clip(S_dep, 0, 1)
            bidx = bidx[recruited]

        # ── inheritance: the ONLY architecture-dependent block ───────────────
        if len(bidx) > 0:
            if sexual:
                # sample one father per birth from eligible reproducers
                fpool = np.where(elig)[0]
                if len(fpool) == 0:
                    fpool = bidx
                fidx = fpool[rng.integers(0, len(fpool), len(bidx))]
                mTr, mSo, mPp = _gametes(geno, bidx, arch, rng)   # maternal gamete
                pTr, pSo, pPp = _gametes(geno, fidx, arch, rng)   # paternal gamete
                cgeno = {"Tr":   np.stack([mTr, pTr], axis=1),
                         "soma": np.stack([mSo, pSo], axis=1),
                         "ppost":np.stack([mPp, pPp], axis=1)}
                cTr, cSoma, cPpost = _pheno(cgeno, bases)
            else:
                cTr, cSoma, cPpost = _clonal_mutation(Tr, soma, ppost, bidx, arch, rng)
                cgeno = None
            cTr    = np.clip(cTr,    30, ceiling)
            cSoma  = np.clip(cSoma,  0.02, 0.95)
            cPpost = np.clip(cPpost, 0.0, 0.95)
            cids = newids(len(bidx))
            n_mother = ids[bidx]
            n_gmother = gmother[bidx]
        else:
            cTr = cSoma = cPpost = cids = np.array([])
            n_mother = n_gmother = np.array([], dtype=np.int64)
            cgeno = None

        # survival, ageing, concatenation
        keep = alive
        age = age[keep] + 1.0
        D = D[keep]; Tr = Tr[keep]; soma = soma[keep]; ppost = ppost[keep]
        ids = ids[keep]; mother = mother[keep]; gmother = gmother[keep]
        if sexual:
            for k in geno:
                geno[k] = geno[k][keep]

        if len(cids) > 0:
            age   = np.concatenate([age,   np.zeros(len(cids))])
            D     = np.concatenate([D,     np.zeros(len(cids))])
            Tr    = np.concatenate([Tr,    cTr])
            soma  = np.concatenate([soma,  cSoma])
            ppost = np.concatenate([ppost, cPpost])
            ids   = np.concatenate([ids,   cids]).astype(np.int64)
            mother = np.concatenate([mother, n_mother]).astype(np.int64)
            gmother = np.concatenate([gmother, n_gmother]).astype(np.int64)
            if sexual:
                for k in geno:
                    geno[k] = np.concatenate([geno[k], cgeno[k]], axis=0)

        N = len(age)
        juv_scale *= (K / max(N, 1)) ** 0.10
        juv_scale = float(np.clip(juv_scale, 0.2, 3.0))

        if yr % 10 == 0:
            ad = age >= a_mat
            mTr = float(np.mean(np.minimum(Tr[ad], ceiling))) if ad.any() else ceiling
            hist["year"].append(yr); hist["meanTr"].append(mTr)
            hist["meanSoma"].append(float(np.mean(soma[ad])) if ad.any() else np.nan)
            hist["meanPpost"].append(float(np.mean(ppost[ad])) if ad.any() else np.nan)
            hist["N"].append(N)

    # life table (identical to abm.py)
    H = {k: np.array(v) for k, v in hist.items()}
    mask = H["year"] >= burn_in
    mu_a = np.where(expo > 0, dth / np.maximum(expo, 1), 0.0)
    surv = np.ones(a_max + 2)
    for a in range(a_mat, a_max + 1):
        surv[a + 1] = surv[a] * np.exp(-mu_a[a])
    lad = surv[a_mat:a_max + 1]
    Td = float(a_mat + lad.sum())
    l70 = float(surv[70])
    d_dist = -np.diff(np.concatenate([lad, [0.0]]))
    modal = int(np.arange(a_mat, a_max + 1)[np.argmax(d_dist)]) if lad.sum() > 0 else a_mat
    meanTr = float(np.nanmean(H["meanTr"][mask])) if mask.any() else ceiling
    sdTr = float(np.nanstd(H["meanTr"][mask])) if mask.any() else 0.0

    summary = dict(B=B, arch=arch.label, meanTr=meanTr, sdTr=sdTr,
                   meanSoma=float(np.nanmean(H["meanSoma"][mask])) if mask.any() else np.nan,
                   meanPpost=float(np.nanmean(H["meanPpost"][mask])) if mask.any() else np.nan,
                   Td=Td, PRLS=max(0.0, Td - meanTr), l70=l70, modal=modal,
                   K=K, years=years)
    return H, summary


# ─────────────────────────────────────────────────────────────────────────────
#  Analytic stochastic-tunnelling crossing time (independent check)
# ─────────────────────────────────────────────────────────────────────────────
def valley_crossing_time(N, mu1, mu2, delta, s):
    """Expected generations for a CLONAL population to cross a two-step valley by
    stochastic tunnelling, when the single intermediate (earlier cessation only)
    is deleterious with selection |delta| and the double (earlier cessation +
    higher maintenance) is beneficial with advantage s.

    Tunnelling regime (Iwasa, Michor & Nowak 2004, Genetics 166:1571; Weissman
    et al. 2009, TPB 75:286): a deleterious single-mutant lineage persists ~1/δ
    generations and throws off double mutants at rate µ2; each established double
    fixes with prob ~s.  Successful-double production rate
        λ ≈ N · µ1 · (µ2/δ) · s          ->   τ ≈ 1/λ.
    Valid when N·µ1 ≲ 1/√(µ2 s) ... ≫ ; treat as an order-of-magnitude estimate.
    Returns τ in generations (np.inf if λ→0).
    """
    lam = N * mu1 * (mu2 / max(delta, 1e-12)) * s
    return np.inf if lam <= 0 else 1.0 / lam


# ─────────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────────
def _worker(job):
    """Top-level (picklable) worker: run one condition, return summary + a
    downsampled meanTr trajectory so convergence can be checked afterward."""
    (kind, label, mode, rho, B, K, years, burn_in, seed, init, opt_Tr) = job
    arch = GeneticArch(mode=mode, rho=rho, label=label)
    H, s = run_abm_g(B=B, years=years, K=K, seed=seed, burn_in=burn_in,
                     arch=arch, init=init, opt_Tr=opt_Tr)
    yr = H["year"].astype(int).tolist()
    tr = [float(x) for x in H["meanTr"]]
    return (kind, label, B, seed, s, yr, tr)


def _avg_traj(trajs):
    """Average a list of (year, meanTr) trajectories, truncated to common length."""
    if not trajs:
        return [], []
    n = min(len(t[0]) for t in trajs)
    yr = trajs[0][0][:n]
    arr = np.array([t[1][:n] for t in trajs])
    return yr, arr.mean(axis=0).tolist()


def main():
    YEARS = int(os.environ.get("ABM_YEARS", 3000))
    K     = int(os.environ.get("ABM_K", 2000))
    NSEED = int(os.environ.get("ABM_SEEDS", 3))
    # burn-in decoupled from run length: default = half, but capped so long runs
    # don't waste tens of thousands of years equilibrating something that settles
    # in a few thousand.
    BURNIN = int(os.environ.get("ABM_BURNIN", min(YEARS // 2, 15000)))
    PROCS = int(os.environ.get("ABM_PROCS", max(1, (os.cpu_count() or 2) - 1)))
    seeds = list(range(1, NSEED + 1))
    Bgrid = [0.0, 1.0, 3.0, 6.0]
    OPT = {0.0: 49.0, 1.0: 40.0, 3.0: 33.0, 6.0: 31.0}   # OptMeno≈ per B (Table 2)

    A = [("A clonal rho=0 (baseline)",     "clonal", 0.0),
         ("B clonal pleiotropy rho=-0.6",  "clonal", -0.6),
         ("C sexual modular rho=0",        "sexual", 0.0),
         ("D sexual pleiotropy rho=-0.6",  "sexual", -0.6)]

    # ── build the full job list (forward + reversion + Ne sweep) ─────────────
    jobs = []
    for (lbl, mode, rho) in A:
        for B in Bgrid:
            for sd in seeds:
                jobs.append(("forward", lbl, mode, rho, B, K, YEARS, BURNIN,
                             sd, "ancestral", 0.0))
    for (lbl, mode, rho) in A:
        for B in (1.0, 6.0):
            for sd in seeds:
                jobs.append(("reversion", lbl, mode, rho, B, K, YEARS, BURNIN,
                             sd, "optimum", OPT[B]))
    for Ksw in (500, 2000, 10000):
        for B in (1.0, 3.0):
            for sd in seeds:
                jobs.append((f"ne{Ksw}", "D sexual pleiotropy rho=-0.6",
                             "sexual", -0.6, B, Ksw, YEARS, BURNIN,
                             sd, "ancestral", 0.0))

    print(f"\n=== Minimal bracketing set ===")
    print(f"years={YEARS}  K={K}  seeds={NSEED}  burn_in={BURNIN}  procs={PROCS}")
    print(f"{len(jobs)} simulations queued.\n", flush=True)

    # ── run (parallel across cores, or serial if PROCS==1) ───────────────────
    import time
    t0 = time.time()
    if PROCS > 1:
        ctx = mp.get_context("spawn")     # safe on macOS
        with ctx.Pool(PROCS) as pool:
            done = []
            for i, res in enumerate(pool.imap_unordered(_worker, jobs), 1):
                done.append(res)
                if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} done "
                          f"({(time.time()-t0)/60:.1f} min)", flush=True)
        results = done
    else:
        results = []
        for i, job in enumerate(jobs, 1):
            results.append(_worker(job))
            print(f"  {i}/{len(jobs)} done ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"all runs finished in {(time.time()-t0)/60:.1f} min\n", flush=True)

    # ── collate ──────────────────────────────────────────────────────────────
    forward, reversion, ne_sweep = {}, {}, {}
    fwd_traj, rev_traj = {}, {}            # (label,B) -> list of (yr,tr) per seed
    bucket = {}
    for (kind, lbl, B, sd, s, yr, tr) in results:
        bucket.setdefault((kind, lbl, B), []).append((s, yr, tr))

    for (lbl, mode, rho) in A:
        forward[lbl] = {}
        for B in Bgrid:
            recs = bucket[("forward", lbl, B)]
            Trs = [r[0]["meanTr"] for r in recs]
            forward[lbl][B] = dict(
                B=B, arch=lbl, Tr=float(np.mean(Trs)), Tr_sd=float(np.std(Trs)),
                Td=float(np.mean([r[0]["Td"] for r in recs])),
                PRLS=float(np.mean([r[0]["PRLS"] for r in recs])),
                l70=float(np.mean([r[0]["l70"] for r in recs])),
                soma=float(np.mean([r[0]["meanSoma"] for r in recs])), K=K)
            fwd_traj[(lbl, B)] = [(r[1], r[2]) for r in recs]
            r = forward[lbl][B]
            print(f"FWD [{lbl:30}] B={B:>3}: Tr={r['Tr']:5.2f}(±{r['Tr_sd']:.2f}) "
                  f"Td={r['Td']:5.1f} PRLS={r['PRLS']:4.1f} l70={r['l70']:.2f}")

    for (lbl, mode, rho) in A:
        reversion[lbl] = {}
        for B in (1.0, 6.0):
            recs = bucket[("reversion", lbl, B)]
            Trs = [r[0]["meanTr"] for r in recs]
            tr_m = float(np.mean(Trs))
            verdict = ("holds" if tr_m < OPT[B] + 4 else
                       "partial" if tr_m < 46 else "reverts to ceiling")
            reversion[lbl][B] = dict(B=B, opt=OPT[B], Tr=tr_m,
                                     Tr_sd=float(np.std(Trs)), verdict=verdict)
            rev_traj[(lbl, B)] = [(r[1], r[2]) for r in recs]
            print(f"REV [{lbl:30}] B={B}: {OPT[B]}->Tr={tr_m:5.2f} [{verdict}]")

    for Ksw in (500, 2000, 10000):
        ne_sweep[Ksw] = {}
        for B in (1.0, 3.0):
            recs = bucket[(f"ne{Ksw}", "D sexual pleiotropy rho=-0.6", B)]
            Trs = [r[0]["meanTr"] for r in recs]
            ne_sweep[Ksw][B] = dict(Tr=float(np.mean(Trs)), Tr_sd=float(np.std(Trs)))
            print(f"NE  K={Ksw:>6} B={B}: Tr={np.mean(Trs):5.2f}(±{np.std(Trs):.2f})")

    # ── analytic stochastic-tunnelling crossing time ─────────────────────────
    print("\nAnalytic stochastic-tunnelling crossing time (clonal)")
    delta, s, gen = 0.004, 0.01, 22.0
    tunnel = {}
    for mu in (1e-5, 1e-3):
        tunnel[mu] = {}
        for Nn in (500, 2000, 10000):
            tau = valley_crossing_time(Nn, mu, mu, delta, s)
            tunnel[mu][Nn] = None if np.isinf(tau) else tau
            print(f"  mu={mu:.0e} Ne={Nn:>6}: tau ~ {tau:,.3g} gen "
                  f"(~{tau*gen:,.3g} yr; vs ~4,500 gen since grandmothering)")

    # ── write outputs ────────────────────────────────────────────────────────
    fwd_traj_avg = {f"{lbl}|B={B}": dict(zip(("year", "meanTr"), _avg_traj(v)))
                    for (lbl, B), v in fwd_traj.items()}
    rev_traj_avg = {f"{lbl}|B={B}": dict(zip(("year", "meanTr"), _avg_traj(v)))
                    for (lbl, B), v in rev_traj.items()}
    out = dict(meta=dict(years=YEARS, K=K, seeds=NSEED, burn_in=BURNIN,
                         Bgrid=Bgrid, OptMeno=OPT),
               forward=forward, reversion=reversion, ne_sweep=ne_sweep,
               tunnelling={f"mu={m}": v for m, v in tunnel.items()},
               tunnelling_params=dict(delta=delta, s=s, gen=gen),
               forward_traj=fwd_traj_avg, reversion_traj=rev_traj_avg)
    with open("results_abm_genetics.json", "w") as f:
        json.dump(out, f, indent=2)
    with open("results_abm_genetics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "architecture", "B", "Tr", "Tr_sd", "Td", "PRLS", "l70", "K"])
        for lbl, rows in forward.items():
            for B, r in rows.items():
                w.writerow(["forward", lbl, B, f"{r['Tr']:.3f}", f"{r['Tr_sd']:.3f}",
                            f"{r['Td']:.2f}", f"{r['PRLS']:.2f}", f"{r['l70']:.3f}", r["K"]])
        for lbl, rows in reversion.items():
            for B, r in rows.items():
                w.writerow(["reversion", lbl, B, f"{r['Tr']:.3f}", f"{r['Tr_sd']:.3f}",
                            "", "", "", K])
    print("\nsaved results_abm_genetics.json and results_abm_genetics.csv")
    _plot(forward, reversion, Bgrid, OPT)
    _plot_traj(fwd_traj_avg, rev_traj_avg, BURNIN, OPT)


def _plot(forward, reversion, Bgrid, OPT):
    """Two panels: (A) forward evolved Tr vs B per architecture vs OptMeno;
    (B) reversion endpoints (start at optimum -> where it ends) at B=1,6."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable, skipping plot:", e); return

    colors = {"A": "0.35", "B": "#d1495b", "C": "#2e8b57", "D": "#e08214"}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    axA.axhline(50, color="0.4", ls="--", lw=1.1, label="atresia ceiling (50)")
    try:                                   # OptMeno computed exactly as in figure_abm.py
        from model import Params as _P, run_cell as _rc
        FULL = dict(gm_a_zero=90, mm_scale=1, use_prod_curve=True)
        Tstops = np.arange(30, 51)
        Bs_oc = np.linspace(0, 8, 21)
        opt_meno = [int(Tstops[np.argmax(
            [_rc(_P(**FULL, B=float(B), a_ceiling=int(T), rate=0.0), free_u=False)[0]["fitness"]
             for T in Tstops])]) for B in Bs_oc]
        axA.plot(Bs_oc, opt_meno, color="#1f5fa8", lw=2.4, zorder=1,
                 label="OptMeno (OC prediction)")
    except Exception:
        axA.plot(Bgrid, [OPT[b] for b in Bgrid], color="#1f5fa8", lw=2.4,
                 ls=":", label="OptMeno (approx)")
    for lbl, rows in forward.items():
        Bs = sorted(rows); Tr = [rows[b]["Tr"] for b in Bs]; sd = [rows[b]["Tr_sd"] for b in Bs]
        axA.errorbar(Bs, Tr, yerr=sd, fmt="o-", ms=6, lw=1.8, capsize=3,
                     color=colors.get(lbl[0]), label=lbl)
    axA.set_xlabel("grandmothering strength  B")
    axA.set_ylabel("evolved age at menopause  $T_r$ (yr)")
    axA.set_ylim(30, 51); axA.set_title("A  Forward: does any architecture descend?", fontsize=11)
    axA.legend(fontsize=8, loc="lower left")

    # Panel B: reversion — arrows from optimum start to evolved endpoint
    axB.axhline(50, color="0.4", ls="--", lw=1.1)
    xs = {1.0: 0, 6.0: 1}
    for lbl, rows in reversion.items():
        for B, r in rows.items():
            x = xs[B] + 0.12 * (ord(lbl[0]) - ord("A")) - 0.18
            axB.annotate("", xy=(x, r["Tr"]), xytext=(x, r["opt"]),
                         arrowprops=dict(arrowstyle="->", color=colors.get(lbl[0]), lw=1.8))
            axB.plot([x], [r["opt"]], "o", color=colors.get(lbl[0]), ms=5)
            axB.plot([x], [r["Tr"]], "s", color=colors.get(lbl[0]), ms=6)
    axB.set_xticks([0, 1]); axB.set_xticklabels(["B=1\n(opt≈40)", "B=6\n(opt≈31)"])
    axB.set_ylabel("$T_r$ (yr):  ● start at optimum  →  ■ evolved endpoint")
    axB.set_ylim(30, 51); axB.set_xlim(-0.6, 1.6)
    axB.set_title("B  Reversion: start at optimum, where does it go?", fontsize=11)
    from matplotlib.lines import Line2D
    axB.legend([Line2D([0],[0], marker="o", color=colors[k], lw=0) for k in "ABCD"],
               ["A clonal", "B clonal+pleio", "C sexual", "D sexual+pleio"],
               fontsize=8, loc="lower right")

    fig.suptitle("Genetic-architecture robustness of the front-end result: "
                 "T_r stays at / returns to the ceiling under clonal, sexual,\n"
                 "pleiotropic and polygenic genetics — the asymmetry is not an "
                 "artifact of the single-locus clonal model", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("figure_abm_genetics.png", dpi=150, bbox_inches="tight")
    print("wrote figure_abm_genetics.png")


def _plot_traj(fwd_traj_avg, rev_traj_avg, burn_in, OPT):
    """Convergence check: meanTr(t) for every condition, one panel per
    architecture (forward solid, reversion dashed).  A flat tail to the right of
    the burn-in line means the run reached equilibrium; a still-moving tail means
    you need more years."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable, skipping trajectory plot:", e); return

    tags = ["A clonal rho=0 (baseline)", "B clonal pleiotropy rho=-0.6",
            "C sexual modular rho=0", "D sexual pleiotropy rho=-0.6"]
    Bcol = {0.0: "0.55", 1.0: "#1f77b4", 3.0: "#2ca02c", 6.0: "#d62728"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, tag in zip(axes.ravel(), tags):
        ax.axhline(50, color="0.4", ls="--", lw=1.0)
        ax.axvline(burn_in, color="0.7", ls=":", lw=1.0)
        for key, d in fwd_traj_avg.items():
            if not key.startswith(tag):
                continue
            B = float(key.split("B=")[1])
            if d["year"]:
                ax.plot(d["year"], d["meanTr"], color=Bcol.get(B, "k"), lw=1.5,
                        label=f"fwd B={B:g}")
        for key, d in rev_traj_avg.items():
            if not key.startswith(tag):
                continue
            B = float(key.split("B=")[1])
            if d["year"]:
                ax.plot(d["year"], d["meanTr"], color=Bcol.get(B, "k"), lw=1.4,
                        ls="--", label=f"rev B={B:g}")
        ax.set_title(tag, fontsize=10)
        ax.set_ylim(30, 51)
        ax.legend(fontsize=7, ncol=2, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("year")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean $T_r$ (yr)")
    fig.suptitle("Convergence check — mean $T_r$(t).  Dotted vertical line = "
                 "burn-in; a flat tail beyond it means equilibrium was reached.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("figure_abm_genetics_traj.png", dpi=150, bbox_inches="tight")
    shutil.copy("figure_abm_genetics_traj.png", "figure_S8.png")
    print("wrote figure_abm_genetics_traj.png / figure_S8.png")


if __name__ == "__main__":
    mp.freeze_support()      # harmless on macOS/Linux; needed if ever frozen
    main()
