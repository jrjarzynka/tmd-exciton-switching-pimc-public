#!/usr/bin/env python3
"""
Graphical abstract for the Computational Materials Science submission.

Three panels, left to right: the sampler is validated, the field relocates the
exciton, and the relocation carries a quantum signature. Every number is read
from the campaign CSVs -- nothing is typed in by hand.

Canvas meets the CMS requirement: >= 1328 x 531 px (w x h), legible at 13 x 5 cm.
Fonts are set for that printed size rather than shrunk after the fact.

Run from validation_tests/. Requires:
    harmonic_seed_check.csv        (6 T x 8 chains)
    occ_scan_v4_summary.csv        (9 fields x 8 chains)
    samples_symmetric_Ex0.npy      (ring polymers at Ex = 0)
    straddling_multiseed.csv       (8 chains)

Output: graphical_abstract.pdf, graphical_abstract.tif
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

RED = "#C1272D"
BLUE = "#1F4E79"
GREY = "0.45"

CM = 1 / 2.54
W_CM, H_CM = 13.0, 5.0
DPI = 300

# Double-well parameters, identical to test3b_tunneling_snapshot.py
SEPARATION_NM = 10.0
SIGMA_NM = 3.0
V0_EV = 0.05

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 7.2,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "lines.linewidth": 1.1,
    "legend.frameon": False,
    "legend.fontsize": 5.6,
    "savefig.facecolor": "white",
})


def panel_benchmark(ax):
    rows = list(csv.DictReader(open("harmonic_seed_check.csv")))
    Ts = sorted({float(r["T_K"]) for r in rows})
    mean, sem, ref = [], [], []
    for T in Ts:
        v = np.array([float(r["R2_pimc_nm2"]) for r in rows if float(r["T_K"]) == T])
        mean.append(v.mean())
        sem.append(v.std(ddof=1) / np.sqrt(len(v)))
        ref.append(float(next(r["R2_exact_finiteP_nm2"] for r in rows
                              if float(r["T_K"]) == T)))
    mean, sem, ref = map(np.array, (mean, sem, ref))
    res = 100 * (mean - ref) / ref
    res_e = 100 * sem / ref

    ax.axhspan(-1, 1, color="0.88", zorder=0)
    ax.axhline(0, color="k", lw=0.7, zorder=1)
    ax.errorbar(Ts, res, yerr=res_e, fmt="o", ms=3.4, color=BLUE, ecolor=BLUE,
                capsize=2, lw=1.0, zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r"$T$ (K)", labelpad=1.5)
    ax.set_ylabel("residual vs exact (%)", labelpad=2)
    ax.set_title("Validated against exact result", pad=3)
    ax.text(0.04, 0.06, "shaded $\\pm1\\%$", transform=ax.transAxes,
            fontsize=5.2, color=GREY)
    ax.grid(alpha=0.2, lw=0.4)
    return np.abs(res).max()


def panel_scurve(ax):
    rows = list(csv.DictReader(open("occ_scan_v4_summary.csv")))
    Ex = np.array([float(r["Ex_eV_per_nm"]) for r in rows]) * 1000
    m = np.array([float(r["fraction_right_mean"]) for r in rows])
    s = np.array([float(r["fraction_right_sem"]) for r in rows])

    ax.axhline(0.5, ls="--", lw=0.7, color=GREY, zorder=1)
    ax.axvline(0.0, ls="--", lw=0.7, color=GREY, zorder=1)
    ax.errorbar(Ex, m, yerr=s, fmt="o-", ms=3.4, color=RED, ecolor=RED,
                capsize=2, lw=1.2, zorder=3)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"$E_x$ (meV/nm)", labelpad=1.5)
    ax.set_ylabel("right-well occupation", labelpad=2)
    ax.set_title("Field drives COM relocation", pad=3)
    ax.grid(alpha=0.2, lw=0.4)
    i0 = int(np.argmin(np.abs(Ex)))
    return m[i0], s[i0]


def panel_delocalization(ax):
    s = np.load("samples_symmetric_Ex0.npy")          # (n_samples, P, 2)
    left_frac = np.mean(s[:, :, 0] < 0, axis=1)
    straddling = (left_frac > 0.05) & (left_frac < 0.95)

    v = np.array([float(r["straddling_fraction"])
                  for r in csv.DictReader(open("straddling_multiseed.csv"))])
    frac, frac_sem = v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    # Potential profile along y = 0, drawn in the lower third only so the
    # sampled chains above it stay readable.
    x = np.linspace(-12, 12, 400)
    d = SEPARATION_NM / 2
    V = -V0_EV * (np.exp(-((x - d) ** 2) / (2 * SIGMA_NM ** 2))
                  + np.exp(-((x + d) ** 2) / (2 * SIGMA_NM ** 2)))
    V = 0.30 * (V - V.min()) / (V.max() - V.min())   # shape only, arb. units

    ax.fill_between(x, 0, V, color="0.90", zorder=0)
    ax.plot(x, V, color=GREY, lw=0.9, zorder=1)

    rng = np.random.default_rng(0)
    loc_idx = rng.choice(np.flatnonzero(~straddling), 4, replace=False)
    str_idx = rng.choice(np.flatnonzero(straddling), 4, replace=False)

    for j, i in enumerate(loc_idx):
        bx = s[i, :, 0]
        ax.plot(bx, np.full_like(bx, 0.34 + 0.065 * j), ".", ms=1.6,
                color=GREY, alpha=0.8, zorder=2)
    for j, i in enumerate(str_idx):
        bx = s[i, :, 0]
        ax.plot(bx, np.full_like(bx, 0.66 + 0.065 * j), ".", ms=1.9,
                color=RED, alpha=0.9, zorder=3)

    # The headline number belongs in the panel, not only in the caption: the
    # chains show that spanning happens, this says how often.
    ax.text(-11.4, 0.955,
            f"{100*frac:.1f}$\\pm${100*frac_sem:.1f}% span both wells",
            fontsize=5.6, color=RED, va="center")
    ax.text(-11.4, 0.545, "localized", fontsize=5.2, color=GREY, va="center")
    # The vertical axis carries no physical meaning: the potential is drawn
    # shape-only and the chain rows are stacked for legibility. Say so, or a
    # reader will take height for energy.
    ax.text(-11.4, 0.13, r"$V(X)$, arb.", fontsize=5.0, color=GREY,
            ha="left", va="center")
    ax.text(0.5, 0.30, "rows offset for clarity",
            transform=ax.transAxes, ha="center", va="center", fontsize=4.8,
            color="0.55")
    ax.set_xlim(-12, 12)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([])
    ax.set_xlabel(r"$X$ (nm)", labelpad=1.5)
    ax.set_title("Quantum delocalization", pad=3)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    return frac, frac_sem


def main():
    need = ["harmonic_seed_check.csv", "occ_scan_v4_summary.csv",
            "samples_symmetric_Ex0.npy", "straddling_multiseed.csv"]
    missing = [f for f in need if not os.path.exists(f)]
    if missing:
        raise SystemExit("missing input: " + ", ".join(missing)
                         + "\nRun this from validation_tests/.")

    fig = plt.figure(figsize=(W_CM * CM, H_CM * CM), dpi=DPI)
    gs = fig.add_gridspec(1, 3, left=0.082, right=0.988, bottom=0.155,
                          top=0.700, wspace=0.36)

    # Header is the article title verbatim, wrapped rather than abbreviated:
    # a graphical abstract that paraphrases its own title reads as a different
    # paper in a search-results list.
    fig.suptitle("Finite-temperature centre-of-mass path-integral Monte Carlo\n"
                 "for field-driven exciton relocation in moir\u00e9 landscapes",
                 y=0.985, fontsize=7.4, fontweight="bold",
                 linespacing=1.35, va="top")

    worst = panel_benchmark(fig.add_subplot(gs[0, 0]))
    occ, occ_e = panel_scurve(fig.add_subplot(gs[0, 1]))
    fr, fr_e = panel_delocalization(fig.add_subplot(gs[0, 2]))

    fig.savefig("graphical_abstract.pdf")

    # Render to RGBA in memory, then composite onto opaque white and store as
    # RGB. Matplotlib's TIFF writer keeps an alpha channel even with a white
    # facecolor; some production pipelines drop it silently and others read it
    # as a mask, so the safe deliverable carries no alpha at all.
    from PIL import Image
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    buf.seek(0)
    rgba = Image.open(buf).convert("RGBA")
    flat = Image.alpha_composite(
        Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    flat.save("graphical_abstract.tif", compression="tiff_lzw",
              dpi=(DPI, DPI))

    w, h = flat.size
    assert flat.mode == "RGB", flat.mode
    ok = "OK" if (w >= 1328 and h >= 531) else "TOO SMALL"
    print(f"written: graphical_abstract.pdf / .tif  ({w} x {h} px) -> {ok}")
    print(f"  panel 1: largest mean residual {worst:.3f}%")
    print(f"  panel 2: occupation at Ex=0 = {occ:.4f} +/- {occ_e:.4f}")
    print(f"  panel 3: straddling = {100*fr:.1f} +/- {100*fr_e:.1f}%")


if __name__ == "__main__":
    main()
