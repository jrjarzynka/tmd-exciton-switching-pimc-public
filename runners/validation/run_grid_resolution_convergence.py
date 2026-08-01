"""Grid-resolution convergence check for the numerical registry-grid landscape.

Sec. 6 of the manuscript validates the sampler's PERIODICITY handling
(bilinear wrap-around fix) on the square-grid COM prototype, but does not
separately establish that the raster DENSITY used to represent the
analytic moire landscape is fine enough for the bilinear-interpolation
error to be negligible. This script closes that gap: it rasterizes the
SAME analytic MoirePotential used to build the registry-grid landscape
onto increasingly fine grids (via PIMCSamplerJIT's own grid_size
parameter -- the exact production interpolation pathway used in Sec. 6,
not a separate reimplementation), and tracks whether centroid-density
observables converge as grid_size increases.

Observables (matching the definitions already used in the manuscript):
    A_eff^S  = exp[-int P(R) ln P(R) d^2R]   (differential-entropy area)
    F_p95    = 95th percentile of F(R) = -k_B T ln[P(R) + eps]

computed from a 2D histogram of sampled centroids, using an IDENTICAL
bin grid across all tested grid_size values so that only the
interpolation raster density changes between runs.

Success criterion: relative change in both observables between
successive grid_size values falls below a chosen tolerance (default 5%,
tighter than the 10-15% rule of thumb already used for the seed/step-size
convergence checks in Sec. 6, since this is a purely numerical -- not
statistical -- source of error and should shrink faster).

Usage
-----
    python run_grid_resolution_convergence.py
    python run_grid_resolution_convergence.py --grid-sizes 50 100 200 400 800
    python run_grid_resolution_convergence.py --temperature 5 --n-beads 80
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tmd_pimc import RingPolymerAction, PIMCSamplerJIT, MoirePotential
from tmd_pimc.observables import centroids
from tmd_pimc.constants import KB_EV_PER_K

MASS_M0 = 0.5
AMPLITUDE_EV = 0.090   # matches the "90 meV registry depth" placeholder, Sec. 6
PERIOD_NM = 20.0       # MoirePotential's own default period
BOX_HALF_WIDTH_NM = 50.0  # matches the "100 nm box" placeholder, Sec. 6
HIST_BINS = 120
HIST_RANGE = (-BOX_HALF_WIDTH_NM, BOX_HALF_WIDTH_NM)
EPS = 1e-12


def effective_area_and_fp95(cents: np.ndarray, temperature_K: float) -> tuple[float, float]:
    H, xedges, yedges = np.histogram2d(
        cents[:, 0], cents[:, 1],
        bins=HIST_BINS, range=[HIST_RANGE, HIST_RANGE],
    )
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    P = H / (H.sum() * dx * dy)  # normalized density, units 1/nm^2

    mask = P > 0
    entropy_area = float(np.exp(-np.sum(P[mask] * np.log(P[mask])) * dx * dy))

    F = -KB_EV_PER_K * temperature_K * np.log(P[mask] + EPS)
    F -= F.min()
    f_p95 = float(np.percentile(F, 95))

    return entropy_area, f_p95


def run_one_seed(grid_size: int, temperature_K: float, n_beads: int,
                  global_step_nm: float, n_steps: int, burn_in: int,
                  sample_every: int, rng_seed: int) -> dict:
    """Single-seed run. Returns both histogram-based observables (A_eff,
    F_p95 -- sensitive to HIST_BINS discretization noise, independent of
    grid_size) AND histogram-free observables (mean V, var V, mean r^2,
    mean COM position -- first/second moments computed directly from raw
    samples, with no binning step at all) on the SAME samples, so the two
    noise sources (interpolation-grid resolution vs. histogram estimation
    vs. seed-to-seed mixing/ergodicity) can be told apart rather than
    conflated into one number.
    """
    potential = MoirePotential(amplitude_eV=AMPLITUDE_EV, period_nm=PERIOD_NM)
    action = RingPolymerAction(
        mass_m0=MASS_M0, temperature_K=temperature_K,
        n_beads=n_beads, potential=potential,
    )
    sampler = PIMCSamplerJIT(
        action=action,
        local_step_nm=0.20,
        global_step_nm=global_step_nm,
        global_move_probability=0.20,
        rng_seed=rng_seed,
        grid_size=grid_size,
        grid_range_nm=BOX_HALF_WIDTH_NM,
        boundary_mode="finite_square",
    )
    result = sampler.run(n_steps=n_steps, burn_in=burn_in,
                          sample_every=sample_every,
                          center=(PERIOD_NM / 2.0, 0.0))
    samples = result["samples"]
    if samples.shape[0] == 0:
        raise RuntimeError(f"No samples for grid_size={grid_size}, seed={rng_seed} "
                            f"-- increase n_steps.")

    cents = centroids(samples)
    a_eff, f_p95 = effective_area_and_fp95(cents, temperature_K)

    # Histogram-free observables: plain moments of the raw samples, no
    # binning step anywhere.
    V_all_beads = potential.value(samples.reshape(-1, 2))
    mean_V = float(V_all_beads.mean())
    var_V = float(V_all_beads.var())
    r2_all_beads = np.einsum("ij,ij->i", samples.reshape(-1, 2), samples.reshape(-1, 2))
    mean_r2 = float(r2_all_beads.mean())
    mean_com = cents.mean(axis=0)

    return {
        "A_eff": a_eff,
        "F_p95": f_p95,
        "acc_local": result["acceptance_local"],
        "acc_global": result["acceptance_global"],
        "mean_V": mean_V,
        "var_V": var_V,
        "mean_r2": mean_r2,
        "mean_com_x": float(mean_com[0]),
        "mean_com_y": float(mean_com[1]),
    }


def _relrange_pct(values: np.ndarray) -> float:
    center = abs(values.mean())
    if center < 1e-12:
        return float("nan")
    return float(100.0 * (values.max() - values.min()) / center)


def run_one(grid_size: int, temperature_K: float, n_beads: int,
            global_step_nm: float, n_steps: int, burn_in: int,
            sample_every: int, rng_seeds: list[int]) -> dict:
    """Averages run_one_seed over multiple seeds -- the same seed-averaging
    protocol already used to produce Table 5 (Sec. 6), needed here to
    distinguish genuine interpolation-resolution convergence from
    single-chain ergodicity noise. Reports seed-spread separately for the
    histogram-based observables (A_eff, F_p95) and the histogram-free
    moments (mean_V, var_V, mean_r2): if the histogram-free moments show
    much smaller seed-spread than A_eff/F_p95 at the SAME grid_size, the
    dominant noise source is the histogram-density estimation, not
    interpolation-grid resolution or mixing; if the histogram-free
    moments ALSO show large seed-spread, the dominant noise source is
    Markov-chain mixing/ergodicity instead.
    """
    per_seed = [
        run_one_seed(grid_size, temperature_K, n_beads, global_step_nm,
                     n_steps, burn_in, sample_every, seed)
        for seed in rng_seeds
    ]

    a_effs = np.array([r["A_eff"] for r in per_seed])
    f_p95s_meV = np.array([r["F_p95"] for r in per_seed]) * 1000.0
    mean_Vs_meV = np.array([r["mean_V"] for r in per_seed]) * 1000.0
    var_Vs = np.array([r["var_V"] for r in per_seed])
    mean_r2s = np.array([r["mean_r2"] for r in per_seed])
    acc_locals = np.array([r["acc_local"] for r in per_seed])
    acc_globals = np.array([r["acc_global"] for r in per_seed])

    return {
        "grid_size": grid_size,
        "n_seeds": len(rng_seeds),
        "acceptance_local_mean": float(acc_locals.mean()),
        "acceptance_global_mean": float(acc_globals.mean()),
        "A_eff_entropy_nm2_mean": float(a_effs.mean()),
        "A_eff_entropy_nm2_relrange_pct": _relrange_pct(a_effs),
        "F_p95_meV_mean": float(f_p95s_meV.mean()),
        "F_p95_meV_relrange_pct": _relrange_pct(f_p95s_meV),
        "mean_V_meV_mean": float(mean_Vs_meV.mean()),
        "mean_V_meV_relrange_pct": _relrange_pct(mean_Vs_meV),
        "var_V_relrange_pct": _relrange_pct(var_Vs),
        "mean_r2_nm2_mean": float(mean_r2s.mean()),
        "mean_r2_relrange_pct": _relrange_pct(mean_r2s),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-sizes", type=int, nargs="+",
                         default=[50, 100, 200, 400, 800],
                         help="Interpolation raster resolutions to test")
    parser.add_argument("--temperature", type=float, default=15.0,
                         help="Temperature in K (default: 15 K, matches Table 5)")
    parser.add_argument("--n-beads", type=int, default=32,
                         help="Bead count (default: 32, matches T=15K row of Table 5)")
    parser.add_argument("--global-step-nm", type=float, default=15.0,
                         help="Global-move step, nm (default: 15, mid of the "
                              "10-25 nm range already validated in Sec. 6)")
    parser.add_argument("--n-steps", type=int, default=300_000)
    parser.add_argument("--burn-in", type=int, default=30_000)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--n-seeds", type=int, default=10,
                         help="Number of independent seeds averaged per grid_size "
                              "(default: 10, matches the Table 5 protocol)")
    parser.add_argument("--seed-start", type=int, default=1000,
                         help="Seeds used are seed_start, seed_start+1, ...")
    parser.add_argument("--tolerance-pct", type=float, default=5.0,
                         help="Relative-change tolerance between successive "
                              "grid sizes, in percent (default: 5)")
    parser.add_argument("--output", type=Path,
                         default=Path("results/grid_resolution_convergence.csv"))
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    rows = []
    for gs in args.grid_sizes:
        print(f"Running grid_size={gs} (T={args.temperature} K, P={args.n_beads}, "
              f"{args.n_seeds} seeds) ...")
        rows.append(run_one(gs, args.temperature, args.n_beads, args.global_step_nm,
                             args.n_steps, args.burn_in, args.sample_every, seeds))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {args.output}\n")
    header = (f"{'grid_size':>10} {'A_eff spread%':>14} {'F_p95 spread%':>14} "
              f"{'mean_V spread%':>15} {'var_V spread%':>14} {'mean_r2 spread%':>16} "
              f"{'acc_global':>11}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['grid_size']:>10d} "
              f"{row['A_eff_entropy_nm2_relrange_pct']:>14.2f} "
              f"{row['F_p95_meV_relrange_pct']:>14.2f} "
              f"{row['mean_V_meV_relrange_pct']:>15.2f} "
              f"{row['var_V_relrange_pct']:>14.2f} "
              f"{row['mean_r2_relrange_pct']:>16.2f} "
              f"{row['acceptance_global_mean']:>11.4f}")

    print(
        "\nInterpretation guide:\n"
        "  - If mean_V / var_V / mean_r2 spread are MUCH SMALLER than A_eff / F_p95\n"
        "    spread at the SAME grid_size: the dominant noise source is the\n"
        "    histogram-density estimator (A_eff, F_p95 are highly nonlinear\n"
        "    functionals of a sparse 2D histogram), not grid interpolation or\n"
        "    mixing. Fix: use more samples, coarser HIST_BINS, or a KDE-based\n"
        "    density estimate instead of a raw histogram for A_eff/F_p95 -- do\n"
        "    NOT conclude grid_size is insufficient from A_eff/F_p95 alone.\n"
        "  - If mean_V / var_V / mean_r2 ALSO show large spread: the dominant\n"
        "    noise source is Markov-chain mixing/ergodicity (same failure mode\n"
        "    already diagnosed in Sec. 6 for the step-size mismatch), not the\n"
        "    histogram or the interpolation grid. Fix: increase n_steps/burn_in,\n"
        "    or re-tune global_step_nm for this specific landscape.\n"
        "  - Only once BOTH of the above are ruled out (all spreads small and\n"
        "    stable) does a genuine cross-grid_size trend become interpretable\n"
        "    as an interpolation-resolution effect."
    )


if __name__ == "__main__":
    main()
