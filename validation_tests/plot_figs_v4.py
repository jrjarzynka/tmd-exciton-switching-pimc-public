#!/usr/bin/env python3
"""
Regenerate Fig. 5 (harmonic benchmark) and Fig. 8 (occupation scan) with
seed-to-seed error bars.

Run from validation_tests/, where the campaign CSVs live. Writes into ../figures/
by default so the Overleaf filenames stay unchanged:

    fig5_harmonic_benchmark.png
    fig8_occupation_scurve.png

Both are drawn from the 8-chain campaigns; nothing is hand-entered.

Usage:
    python3 plot_figs_v4.py
    python3 plot_figs_v4.py --outdir .
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RED = "#C1272D"
BLUE = "#1F4E79"


def load_grouped(path, keycol, valcol, extra=None):
    """Group valcol by keycol; return sorted keys, list of arrays, extra lookup."""
    rows = list(csv.DictReader(open(path)))
    keys = sorted({float(r[keycol]) for r in rows})
    vals = [np.array([float(r[valcol]) for r in rows if float(r[keycol]) == k])
            for k in keys]
    ref = ({k: float(next(r[extra] for r in rows if float(r[keycol]) == k))
            for k in keys} if extra else None)
    return np.array(keys), vals, ref


def fig5(outdir):
    T, r2, ref80 = load_grouped("harmonic_seed_check.csv", "T_K",
                                "R2_pimc_nm2", "R2_exact_finiteP_nm2")
    mean = np.array([v.mean() for v in r2])
    sem = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in r2])
    exact80 = np.array([ref80[t] for t in T])
    n = len(r2[0])

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(6.0, 5.6), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.08})

    ax.plot(T, exact80, "k-", lw=1.3, zorder=2,
            label=f"exact, $P=80$ (same $P$ as PIMC)")
    ax.errorbar(T, mean, yerr=sem, fmt="o", ms=5, mfc="none", mec=RED,
                ecolor=RED, capsize=3, lw=1.2, zorder=3,
                label=f"PIMC, mean $\\pm$ s.e.m. ($n={n}$ chains)")
    for t, v in zip(T, r2):
        ax.plot(np.full_like(v, t), v, ".", ms=2.5, color=RED, alpha=0.35,
                zorder=1)
    ax.plot([], [], ".", ms=4, color=RED, alpha=0.5, label="individual chains")

    ax.set_xscale("log")
    ax.set_ylabel(r"$\langle R^2 \rangle$ (nm$^2$)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("COM sampler vs. exact harmonic-oscillator benchmark",
                 fontsize=10)

    res = 100 * (mean - exact80) / exact80
    res_e = 100 * sem / exact80
    axr.axhline(0, color="k", lw=0.8)
    axr.axhspan(-1, 1, color="0.85", zorder=0)
    axr.errorbar(T, res, yerr=res_e, fmt="o", ms=4, color=BLUE, ecolor=BLUE,
                 capsize=3, lw=1.2)
    axr.set_xscale("log")
    axr.set_xlabel(r"$T$ (K)")
    axr.set_ylabel("residual (%)", fontsize=9)
    axr.set_ylim(-1.6, 1.6)
    axr.grid(alpha=0.25, lw=0.5)
    axr.text(0.015, 0.09, "shaded: $\\pm1\\%$", transform=axr.transAxes,
             fontsize=7.5, color="0.35")

    out = os.path.join(outdir, "fig5_harmonic_benchmark.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    for t, m, s, r in zip(T, mean, sem, res):
        print(f"    T={t:5.0f} K  <R2>={m:.4f} +/- {s:.4f}   residual {r:+.3f}%")


def fig8(outdir):
    rows = list(csv.DictReader(open("occ_scan_v4_summary.csv")))
    Ex = np.array([float(r["Ex_eV_per_nm"]) for r in rows]) * 1000.0
    m = np.array([float(r["fraction_right_mean"]) for r in rows])
    s = np.array([float(r["fraction_right_sem"]) for r in rows])
    n = int(rows[0]["n_seeds"])

    per = list(csv.DictReader(open("occ_scan_v4_multiseed.csv")))

    fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=200)
    ax.axhline(0.5, ls="--", lw=0.9, color="0.55", zorder=1)
    ax.axvline(0.0, ls="--", lw=0.9, color="0.55", zorder=1)

    for i in sorted({int(r["i_field"]) for r in per}):
        v = np.array([float(r["fraction_right"]) for r in per
                      if int(r["i_field"]) == i])
        ax.plot(np.full_like(v, Ex[i]), v, ".", ms=3, color=RED, alpha=0.3,
                zorder=2)

    ax.errorbar(Ex, m, yerr=s, fmt="o-", ms=5.5, color=RED, ecolor=RED,
                capsize=3.5, lw=1.4, zorder=3,
                label=f"mean $\\pm$ s.e.m. ($n={n}$ chains per point)")
    ax.plot([], [], ".", ms=4, color=RED, alpha=0.45,
            label="individual chains")

    ax.set_xlabel(r"$E_x$ (meV/nm)")
    ax.set_ylabel("fraction of centroids in right well")
    ax.set_title("In-plane-field-driven COM relocation, symmetric double well",
                 fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(-0.02, 1.02)

    out = os.path.join(outdir, "fig8_occupation_scurve.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")
    i0 = int(np.argmin(np.abs(Ex)))
    print(f"    at Ex=0: {m[i0]:.4f} +/- {s[i0]:.4f}  "
          f"({abs(0.5-m[i0])/s[i0]:.2f} sigma from 0.5)")
    d = np.diff(m)
    print(f"    monotonic: {bool((d > 0).all())}, "
          f"smallest increment {d.min():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="../figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    missing = [f for f in ("harmonic_seed_check.csv", "occ_scan_v4_summary.csv",
                           "occ_scan_v4_multiseed.csv") if not os.path.exists(f)]
    if missing:
        raise SystemExit("missing campaign output: " + ", ".join(missing)
                         + "\nRun this from validation_tests/.")

    print("Fig. 5:")
    fig5(a.outdir)
    print("Fig. 8:")
    fig8(a.outdir)


if __name__ == "__main__":
    main()
