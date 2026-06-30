"""
Individual-based model (ABM) for the fused grandmothering model.

Evolvable genome (clonal maternal inheritance + Gaussian mutation):
  - Tr_gene   : age at reproductive cessation (≤ atresia ceiling)
  - soma_gene : maintenance fraction during reproduction
  - ppost_gene: maintenance fraction post-reproduction (help = 1-ppost)

TWO-GRANDMOTHER TRACKING (new):
  Each individual carries TWO grandmother IDs:
    gmother  : maternal grandmother (mother's mother)  — tracked through pedigree
    pgmother : paternal grandmother (father's mother)  — sampled at birth from
               post-reproductive females in the population, discounted by p_u
               (paternity uncertainty) and r_g_pgm = r_g × (1 - p_u).
  Both grandmothers contribute to a reproducing daughter's fecundity:
    fert = 1  +  B_mgm × g(mgm_age) × ΔG(mgm_age) × (1-ppost_mgm)     [MGM]
              +  B_pgm × (1-p_u) × g(pgm_age) × ΔG(pgm_age) × (1-ppost_pgm)  [PGM]

NEW BIOLOGY:
  - Age-graded productivity g(a): scales birth yield and GM help capacity
    (Kaplan et al. 2000, calibrated to forager net-production data).
  - Kirkwood effort-dependent damage: δ(u) = delta_base + delta_rep × u²
    (Disposable Soma Theory; Kirkwood 1977; Kirkwood & Rose 1991).

Density-dependent juvenile survival keeps population near K → stationary (R0).
"""
import os
import json
import numpy as np
from model import (quality_curve, grandmother_benefit, maternal_mortality,
                   productivity_curve, Params)


def run_abm(B=1.0, B_pgm=0.0, p_u=0.05,
            years=4000, K=2500, seed=0,
            mut_sd_Tr=0.5, mut_sd_soma=0.02, burn_in=1500,
            p: Params = None):
    """Run the ABM and return (history_dict, summary_dict).

    Parameters
    ----------
    B      : maternal grandmothering strength
    B_pgm  : paternal grandmothering strength (0 = off)
    p_u    : paternity uncertainty (0–1); reduces effective PGM relatedness
    """
    rng = np.random.default_rng(seed)
    if p is None:
        p = Params()
    a_mat, ceiling, a_max = p.a_mat, p.a_ceiling, p.a_max
    r_g_pgm = p.r_g * (1.0 - p_u)   # PGM relatedness discount

    # ── precompute age-indexed curves ──────────────────────────────────────
    ages_all = np.arange(0, a_max + 1)
    qcurve   = quality_curve(ages_all, p.q0, p.qk, p.qa50)
    gcurve   = (productivity_curve(ages_all) * p.g if p.use_prod_curve
                else np.full(a_max + 1, p.g))
    dGcurve  = grandmother_benefit(ages_all, p.a_gm, p.gm_ramp, p.gm_plateau,
                                   p.gm_a_full, p.gm_a_zero, p.gm_decline_k)
    pmat_curve = maternal_mortality(ages_all, p.mm_base, p.mm_k,
                                    p.mm_a0, p.mm_floor)

    # ── initialise population ──────────────────────────────────────────────
    nid = 0
    def newids(n):
        nonlocal nid
        out = np.arange(nid, nid + n); nid += n
        return out

    N0   = K
    age  = rng.integers(a_mat, ceiling, N0).astype(float)
    D    = np.maximum(0.0, (age - a_mat) * p.delta_base * 0.4
                      + rng.normal(0, 0.05, N0))
    Tr   = rng.uniform(42, 50, N0)
    soma = rng.uniform(0.15, 0.45, N0)
    ppost = rng.uniform(0.10, 0.40, N0)
    ids     = newids(N0)
    mother  = np.full(N0, -1, dtype=np.int64)
    gmother = np.full(N0, -1, dtype=np.int64)   # maternal grandmother
    pgmother= np.full(N0, -1, dtype=np.int64)   # paternal grandmother (new)

    hist = dict(year=[], meanTr=[], meanSoma=[], meanPpost=[], N=[])
    expo = np.zeros(a_max + 2)
    dth  = np.zeros(a_max + 2)
    juv_scale = 1.0

    # ── main simulation loop ───────────────────────────────────────────────
    for yr in range(years):
        Nnow = len(age)
        if Nnow == 0:
            break

        aint   = np.clip(age.astype(int), 0, a_max)
        Tr_eff = np.minimum(Tr, ceiling)
        repro_phase = (age >= a_mat) & (age <= Tr_eff)
        post_phase  = (age > Tr_eff)  & (age >= a_mat)

        # ── somatic damage: Kirkwood δ(u) ──────────────────────────────
        repair  = np.where(repro_phase, soma, np.where(post_phase, ppost, 0.0))
        effort  = np.where(repro_phase, 1.0 - soma, 0.0)
        delta_u = p.delta_base + p.delta_rep * effort**2
        D = np.clip(D + delta_u - p.rho * repair, 0.0, p.Dmax)

        # ── mortality ─────────────────────────────────────────────────
        gomp = p.mu2 * (np.exp(p.gamma * np.maximum(age - a_mat, 0)) - 1.0)
        mu   = p.mu0 + p.mu1 * D + gomp
        birth_haz = p.Mmax * gcurve[aint] * qcurve[aint] * (1.0 - soma)
        mu = mu + np.where(repro_phase,
                           p.mm_scale * birth_haz * pmat_curve[aint], 0.0)
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

        # ── grandmothering: MGM + PGM fertility boost ─────────────────
        fert = np.ones(Nnow)

        if B > 0:
            # Maternal grandmother (MGM) — pedigree-tracked
            posm  = np.clip(np.searchsorted(ids, mother), 0, Nnow - 1)
            mgm_ok = ((mother >= 0) & (ids[posm] == mother)
                      & post_phase[posm] & alive[posm])
            gma   = np.clip(age[posm].astype(int), 0, a_max)
            fert  = np.where(mgm_ok,
                             fert + B * gcurve[gma] * dGcurve[gma]
                               * (1.0 - ppost[posm]),
                             fert)

        if B_pgm > 0:
            # Paternal grandmother (PGM) — sampled at birth, tracked by id
            posp   = np.clip(np.searchsorted(ids, pgmother), 0, Nnow - 1)
            pgm_ok = ((pgmother >= 0) & (ids[posp] == pgmother)
                      & post_phase[posp] & alive[posp])
            pgma   = np.clip(age[posp].astype(int), 0, a_max)
            # relatedness discounted by (1 - p_u) relative to MGM
            fert   = np.where(pgm_ok,
                               fert + B_pgm * r_g_pgm / p.r_g
                                 * gcurve[pgma] * dGcurve[pgma]
                                 * (1.0 - ppost[posp]),
                               fert)

        # ── births ────────────────────────────────────────────────────
        brate  = np.where(elig,
                          p.Mmax * gcurve[aint] * qcurve[aint]
                            * (1.0 - soma) * fert,
                          0.0)
        births = rng.random(Nnow) < brate
        bidx   = np.where(births)[0]
        if len(bidx) > 0:
            S_dep     = np.exp(-mu[bidx] * p.L_dep)
            recruited = rng.random(len(bidx)) < np.clip(S_dep, 0, 1)
            bidx      = bidx[recruited]

        if len(bidx) > 0:
            # ── assign paternal grandmother (PGM) at birth ────────────
            # PGM = random post-reproductive female in population (not
            # necessarily the same as MGM).  Discounted by p_u:
            # with prob p_u the paternity is uncertain → no PGM assigned.
            n_pgm_ids = np.full(len(bidx), -1, dtype=np.int64)
            if B_pgm > 0:
                cand = np.where(post_phase & alive)[0]
                if len(cand) > 0:
                    chosen    = cand[rng.integers(0, len(cand), len(bidx))]
                    pat_known = rng.random(len(bidx)) >= p_u
                    n_pgm_ids = np.where(pat_known, ids[chosen], -1)

            cTr    = np.clip(Tr[bidx]    + rng.normal(0, mut_sd_Tr,   len(bidx)),
                             30, ceiling)
            cSoma  = np.clip(soma[bidx]  + rng.normal(0, mut_sd_soma, len(bidx)),
                             0.02, 0.95)
            cPpost = np.clip(ppost[bidx] + rng.normal(0, mut_sd_soma, len(bidx)),
                             0.0, 0.95)
            cids      = newids(len(bidx))
            n_mother  = ids[bidx]
            n_gmother = gmother[bidx]   # MGM of offspring = gmother of birth mother
        else:
            cTr = cSoma = cPpost = cids = n_pgm_ids = np.array([])
            n_mother = n_gmother = np.array([], dtype=np.int64)

        # ── survival, ageing, concatenation ───────────────────────────
        keep    = alive
        age     = age[keep] + 1.0
        D       = D[keep];    Tr  = Tr[keep];    soma  = soma[keep]
        ppost   = ppost[keep]
        ids     = ids[keep];  mother = mother[keep]
        gmother = gmother[keep]; pgmother = pgmother[keep]

        if len(cids) > 0:
            age     = np.concatenate([age,     np.zeros(len(cids))])
            D       = np.concatenate([D,       np.zeros(len(cids))])
            Tr      = np.concatenate([Tr,      cTr])
            soma    = np.concatenate([soma,    cSoma])
            ppost   = np.concatenate([ppost,   cPpost])
            ids     = np.concatenate([ids,     cids]).astype(np.int64)
            mother  = np.concatenate([mother,  n_mother]).astype(np.int64)
            gmother = np.concatenate([gmother, n_gmother]).astype(np.int64)
            pgmother= np.concatenate([pgmother, n_pgm_ids]).astype(np.int64)

        N = len(age)
        juv_scale *= (K / max(N, 1)) ** 0.10
        juv_scale  = float(np.clip(juv_scale, 0.2, 3.0))

        if yr % 10 == 0:
            ad  = age >= a_mat
            mTr = float(np.mean(np.minimum(Tr[ad], ceiling))) if ad.any() else ceiling
            hist["year"].append(yr);      hist["meanTr"].append(mTr)
            hist["meanSoma"].append(float(np.mean(soma[ad]))  if ad.any() else np.nan)
            hist["meanPpost"].append(float(np.mean(ppost[ad])) if ad.any() else np.nan)
            hist["N"].append(N)

    # ── life table from accumulated exposure / deaths ──────────────────────
    H    = {k: np.array(v) for k, v in hist.items()}
    mask = H["year"] >= burn_in

    mu_a = np.where(expo > 0, dth / np.maximum(expo, 1), 0.0)
    surv = np.ones(a_max + 2)
    for a in range(a_mat, a_max + 1):
        surv[a + 1] = surv[a] * np.exp(-mu_a[a])
    lad   = surv[a_mat:a_max + 1]
    Td    = float(a_mat + lad.sum())
    l70   = float(surv[70])
    d_dist = -np.diff(np.concatenate([lad, [0.0]]))
    modal  = int(np.arange(a_mat, a_max + 1)[np.argmax(d_dist)]) if lad.sum() > 0 else a_mat
    e45    = float(surv[45:a_max+1].sum() / surv[45]) if surv[45] > 0 else 0.0
    meanTr = float(np.nanmean(H["meanTr"][mask]))

    summary = dict(B=B, B_pgm=B_pgm, p_u=p_u,
                   meanTr=meanTr, sdTr=float(np.nanstd(H["meanTr"][mask])),
                   meanSoma=float(np.nanmean(H["meanSoma"][mask])),
                   meanPpost=float(np.nanmean(H["meanPpost"][mask])),
                   Td=Td, PRLS=max(0.0, Td - meanTr),
                   l70=l70, modal=modal, e45=e45)
    H["surv_ages"] = np.arange(a_mat, a_max + 1)
    H["surv_l"]    = lad
    return H, summary


if __name__ == "__main__":
    # Multi-seed production run; writes results_abm.json.
    # Override via environment variables:
    #   ABM_YEARS=4000 ABM_K=3000 ABM_SEEDS=5 python3 abm.py
    YEARS = int(os.environ.get("ABM_YEARS", 2000))
    K     = int(os.environ.get("ABM_K",     2000))
    NSEED = int(os.environ.get("ABM_SEEDS", 3))
    seeds = list(range(1, NSEED + 1))

    # Run conditions: single MGM only, then MGM + PGM
    conditions = [
        dict(B=0.0, B_pgm=0.0, p_u=0.05, label="B=0 (no GM)"),
        dict(B=1.0, B_pgm=0.0, p_u=0.05, label="B=1 MGM only"),
        dict(B=1.0, B_pgm=1.0, p_u=0.05, label="B=1 MGM + B=1 PGM (p_u=0.05)"),
        dict(B=1.0, B_pgm=1.0, p_u=0.10, label="B=1 MGM + B=1 PGM (p_u=0.10)"),
        dict(B=3.0, B_pgm=0.0, p_u=0.05, label="B=3 MGM only"),
        dict(B=6.0, B_pgm=0.0, p_u=0.05, label="B=6 MGM only"),
    ]

    agg = {}
    for cond in conditions:
        lbl = cond["label"]
        Bv, Bpgm, pu = cond["B"], cond["B_pgm"], cond["p_u"]
        key = f"B{Bv}_pgm{Bpgm}_pu{pu}"
        Trs=[]; somas=[]; pposts=[]; Tds=[]; l70s=[]; modals=[]; trajTr=[]; yrs=None
        for sd in seeds:
            H, s = run_abm(B=Bv, B_pgm=Bpgm, p_u=pu,
                           years=YEARS, K=K, seed=sd, burn_in=YEARS//2)
            Trs.append(s["meanTr"]); somas.append(s["meanSoma"])
            pposts.append(s["meanPpost"]); Tds.append(s["Td"])
            l70s.append(s["l70"]); modals.append(s["modal"])
            trajTr.append(H["meanTr"]); yrs = H["year"]
        n = min(len(t) for t in trajTr)
        agg[key] = dict(
            B=Bv, B_pgm=Bpgm, p_u=pu, label=lbl,
            Tr=float(np.mean(Trs)),   Tr_sd=float(np.std(Trs)),
            soma=float(np.mean(somas)), ppost=float(np.mean(pposts)),
            help=float(1.0 - np.mean(pposts)),
            Td=float(np.mean(Tds)),   PRLS=float(np.mean(Tds))-float(np.mean(Trs)), l70=float(np.mean(l70s)),
            modal=float(np.mean(modals)),
            year=np.asarray(yrs[:n]).tolist(),
            trajTr=np.mean([np.asarray(t[:n]) for t in trajTr], axis=0).tolist())
        a = agg[key]
        print(f"{lbl}: Tr={a['Tr']:.1f}(±{a['Tr_sd']:.2f})  "
              f"Td={a['Td']:.1f}  l70={a['l70']:.2f}  PRLS={a['PRLS']:.1f}",
              flush=True)

    with open("results_abm.json", "w") as f:
        json.dump(agg, f)
    print("saved results_abm.json")
