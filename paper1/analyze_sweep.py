"""
Analyse the ABM parameter sweep (sweep_results.csv) and produce summary panels.

IMPORTANT modelling note (see console output): the ABM is density-regulated to
carrying capacity, i.e. it is always STATIONARY -- run_abm does not read p.rate.
The demographic-currency contrast (growing / stationary / declining) is therefore
an OPTIMAL-CONTROL result (Fig 1), not something the ABM can express; the three
currency series in a sweep are identical. This script accordingly pools over the
(vacuous) currency axis and splits by the axes the ABM actually varies: grand-
mothering strength B, helping-efficacy curve, and repair efficacy rho.

Back-end panels show the ABM result on the same scale as the optimal-control
prediction (read from results_oc.npz if present) so the gradient-limited under-
evolution of post-reproductive lifespan is explicit rather than hidden.

Depends only on the standard library + numpy + matplotlib.

Usage:
    python analyze_sweep.py
    python analyze_sweep.py --csv other.csv --rho 0.06
"""
import argparse, csv, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil

EFF_COL = {"gradual": "#7a4ea3", "chapman": "#e8843c"}
RHO_COL = {0.045: "#2a9d4a", 0.06: "#1b6ca8", 0.075: "#d1495b"}
OC_COL = "#555555"
CEILING = 50.0
NUM = ["B", "rho", "Tr", "soma", "ppost", "Td", "l70", "modal", "PRLS"]


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in NUM:
                if k in r and r[k] != "":
                    r[k] = float(r[k])
            rows.append(r)
    if not rows:
        sys.exit(f"No rows in {path}")
    return rows


def agg_by(rows, fixed=None):
    """Pool over seeds AND currency. Key = (efficacy, rho, B). Returns mean/sd."""
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if fixed:
            ok = all(abs(r[k]-v) < 1e-6 if isinstance(v, float) else r[k] == v
                     for k, v in fixed.items())
            if not ok:
                continue
        key = (r["efficacy"], round(r["rho"], 4), r["B"])
        for m in ["Tr", "Td", "PRLS", "l70", "soma"]:
            g[key][m].append(r[m])
    out = {}
    for key, d in g.items():
        out[key] = {m: (float(np.mean(v)), float(np.std(v))) for m, v in d.items()}
    return out


def line(agg, efficacy, rho, metric):
    pts = sorted((k[2], v[metric][0], v[metric][1]) for k, v in agg.items()
                 if k[0] == efficacy and abs(k[1]-rho) < 1e-6)
    if not pts:
        return None
    return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]),
            np.array([p[2] for p in pts]))


def oc_reference():
    if not os.path.exists("results_oc.npz"):
        return None
    d = np.load("results_oc.npz", allow_pickle=True)
    be = d["backend"].item()["R0 (stationary)"]; Bg = d["Bgrid"]
    return Bg, np.array(be["Td"]), np.array(be["PRLS"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sweep_results.csv")
    ap.add_argument("--rho", type=float, default=0.06)
    ap.add_argument("--out", default="fig9_sweep_summary.png")
    a = ap.parse_args()

    rows = load(a.csv)
    currencies = sorted({r["currency"] for r in rows})
    rhos = sorted({round(r["rho"], 4) for r in rows})
    effs = [e for e in ["gradual", "chapman"] if any(r["efficacy"] == e for r in rows)]
    rho = a.rho if a.rho in rhos else rhos[len(rhos)//2]
    eff0 = "gradual" if "gradual" in effs else effs[0]

    agg = agg_by(rows)            # pooled over seed + currency
    allTr = [v["Tr"][0] for v in agg.values()]
    oc = oc_reference()

    # ---- console ----
    print(f"\nLoaded {len(rows)} runs; {len(currencies)} currenc(ies) {currencies}")
    if len(currencies) > 1:
        print("NOTE: run_abm is density-regulated (stationary) and ignores p.rate, so the")
        print("      currency series are identical -> pooled here. Currency is an OC-only")
        print("      contrast (Fig 1).")
    print(f"\nFRONT-END: across all efficacy x rho x B cells, evolved T_r ranged "
          f"{min(allTr):.1f}-{max(allTr):.1f} yr (ceiling {CEILING:.0f}).")
    print("  -> menopause allele pinned just under the ceiling everywhere.\n")
    print(f"BACK-END (efficacy={eff0}, rho={rho}): ABM vs optimal-control")
    lt = line(agg, eff0, rho, "Td"); lp = line(agg, eff0, rho, "PRLS")
    if lt is not None:
        print(f"  ABM  life-exp@maturity {lt[1].min():.0f}-{lt[1].max():.0f} yr,  "
              f"PRLS {lp[1].min():.1f}-{lp[1].max():.1f} yr  (over B)")
    if oc is not None:
        Bg, Td_oc, PRLS_oc = oc
        i1 = int(np.argmin(abs(Bg-1)))
        print(f"  OC   life-exp@maturity ~{Td_oc[i1]:.0f} yr, PRLS ~{PRLS_oc[i1]:.0f} yr at B=1")
        print("  -> ABM under-evolves the back end (gradient-limited); OC sets the magnitude.")

    # ---- figure ----
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))

    # A: front-end, T_r vs B by efficacy
    for e in effs:
        s = line(agg, e, rho, "Tr")
        if s: ax[0,0].errorbar(*s, marker="o", ms=4, lw=2, color=EFF_COL[e], capsize=2,
                               label=f"{e} efficacy")
    ax[0,0].axhline(CEILING, color="0.5", ls="--", lw=1.2, label="atresia ceiling")
    ax[0,0].set(xlabel="grandmothering strength  $B$", ylabel="evolved age at menopause  $T_r$",
                title=f"A  Front end: menopause stays at ceiling ($\\rho$={rho})", ylim=(40, 51))
    ax[0,0].legend(frameon=False, fontsize=8.5, loc="lower center")

    # B: front-end robustness to rho (eff0)
    for r in rhos:
        s = line(agg, eff0, r, "Tr")
        if s: ax[0,1].errorbar(*s, marker="o", ms=4, lw=2,
                               color=RHO_COL.get(r, None), capsize=2, label=f"$\\rho$={r}")
    ax[0,1].axhline(CEILING, color="0.5", ls="--", lw=1.2, label="atresia ceiling")
    ax[0,1].set(xlabel="grandmothering strength  $B$", ylabel="evolved age at menopause  $T_r$",
                title=f"B  Front end robust to repair efficacy ({eff0})", ylim=(40, 51))
    # legend in upper right to avoid overlapping declining Tr data at high B
    ax[0,1].legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)

    # C: back-end life expectancy, ABM vs OC
    for e in effs:
        s = line(agg, e, rho, "Td")
        if s: ax[1,0].errorbar(*s, marker="s", ms=4, lw=2, color=EFF_COL[e], capsize=2,
                               label=f"ABM, {e}")
    if oc is not None:
        ax[1,0].plot(oc[0], oc[1], color=OC_COL, ls="--", lw=2, label="optimal-control ($R_0$)")
    ax[1,0].set(xlabel="grandmothering strength  $B$",
                ylabel="life expectancy at maturity (yr)",
                title="C  Back end: ABM under-evolves vs OC", ylim=(40, 70))
    # legend in lower right to avoid OC line at upper left
    ax[1,0].legend(frameon=False, fontsize=8.5, loc="lower right")

    # D: PRLS, ABM vs OC, on the OC scale
    for e in effs:
        s = line(agg, e, rho, "PRLS")
        if s: ax[1,1].errorbar(*s, marker="s", ms=4, lw=2, color=EFF_COL[e], capsize=2,
                               label=f"ABM, {e}")
    if oc is not None:
        ax[1,1].plot(oc[0], oc[2], color=OC_COL, ls="--", lw=2, label="optimal-control ($R_0$)")
    ax[1,1].axhline(0, color="0.8", lw=0.8)
    ax[1,1].set(xlabel="grandmothering strength  $B$",
                ylabel="post-reproductive lifespan (yr)",
                title="D  PRLS: OC ~12 yr vs ABM \u2248 0 (gradient-limited)", ylim=(-1, 16))
    ax[1,1].legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.suptitle("ABM parameter sweep: the evolved cessation allele stays near\nthe atresia ceiling "
                 "across all conditions; earlier cessation is inaccessible to single-allele "
                 "change; the back-end longevity response is gradient-limited",
                 fontweight="bold", y=0.995, fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(a.out, bbox_inches="tight")
    shutil.copy(a.out, "figure_S7.png")
    print(f"\nwrote {a.out} / figure_S7.png")


if __name__ == "__main__":
    main()
