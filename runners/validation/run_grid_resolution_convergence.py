"""Grid-resolution convergence check for the numerical registry-grid landscape.

Sec. 6 of the manuscript validates the sampler's PERIODICITY handling
(bilinear wrap-around fix) on the square-grid COM prototype, but does not
separately establish that the raster DENSITY used to represent the
analytic moire landscape is fine enough for the bilinear-interpolation
error to be negligible. This script closes that gap: it rasterizes an
analytic MoirePotential onto increasingly fine grids (via PIMCSamplerJIT's
own grid_size parameter -- the exact production interpolation pathway used
in Sec. 6, not a separate reimplementation), and tracks whether
centroid-density observables converge as grid_size increases.

Corrugation-depth conventions -- READ BEFORE CHANGING AMPLITUDE_EV
------------------------------------------------------------------
Three quantities in this codebase are all informally called the "registry
depth", and they differ by a factor of 4.5:

  amplitude_eV          MoirePotential.value() returns
  (this script,          V0 * [cos(G1.r) + cos(G2.r) + cos(G3.r)],
   numerics/)            so amplitude_eV IS V0, the first-shell Fourier
                         amplitude. The landscape spans 4.5 * V0, from
                         -1.5*V0 at the minima to +3.0*V0 at AA.

  moire_amplitude_eV    Same convention (feeds MoirePotential directly).
  (configs/two_body/)   Production value for the two-body scans: 0.045 eV.

  registry_depth_meV    generate_potential_map.registry_energy() normalises
  (future_work/          the raw three-cosine sum onto [0, 1] and multiplies
   atomistic_bridge/)    by depth_eV, so registry_depth_meV is the
                         PEAK-TO-PEAK span, i.e. 4.5 * V0 -- NOT V0.

Consequently registry_depth_meV = 90 corresponds to V0 = 20 meV, and the
AMPLITUDE_EV = 0.090 used here is 4.5x deeper than that placeholder rather
than equal to it. The value is kept for backward compatibility with the
already-published convergence campaign; do not "fix" it by editing the
constant, since that would silently change results that are already
reported. Use --amplitude-eV to run at a different depth.

Relaxed-DFT GSFE calibration (49-point grid + direct AA/AB/BA runs) gives
V0 = 121.60 meV for MoSe2/WSe2, i.e. --amplitude-eV 0.12160 here, or
registry_depth_meV = 537.99 in the atomistic-bridge convention.

One caveat when comparing against that calibration: the real GSFE surface
carries a second reciprocal-lattice shell at 5.7% of the first. That shifts
the peak-to-peak span by under 2% (537.99 vs 547.20 meV) but raises the
SADDLE by 47% (89.30 vs 60.80 meV), because higher harmonics barely move the
well depths while sharpening the barrier between them. Since it is the
barrier, not the well depth, that sets escape and hopping rates, a
three-cosine landscape fitted to reproduce the correct V0 will still
underestimate activated transport on the real surface.

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

from tmd_pimc import (
    RingPolymerAction,
    PIMCSamplerJIT,
    MoirePotential,
    moire_hop_vectors_nm,
)
from tmd_pimc.observables import centroids
from tmd_pimc.constants import KB_EV_PER_K

MASS_M0 = 0.5

# V0, the first-shell Fourier amplitude -- NOT the peak-to-peak span, and NOT
# the same convention as registry_depth_meV in the atomistic bridge (see the
# module docstring). Legacy default, retained so the published convergence
# campaign stays reproducible; override with --amplitude-eV.
AMPLITUDE_EV = 0.090
PERIOD_NM = 20.0       # MoirePotential's own default period
BOX_HALF_WIDTH_NM = 50.0  # matches the "100 nm box" placeholder, Sec. 6
HIST_BINS = 120
HIST_RANGE = (-BOX_HALF_WIDTH_NM, BOX_HALF_WIDTH_NM)
EPS = 1e-12

# Exact ratios for the pure three-cosine form V = V0 * sum_i cos(G_i . r):
#   AA (r = 0)        -> +3.0 * V0
#   minima            -> -1.5 * V0
#   saddle (M point)  -> -1.0 * V0
PEAK_TO_PEAK_OVER_V0 = 4.5
SADDLE_ABOVE_MIN_OVER_V0 = 0.5


def v0_from_peak_to_peak(peak_to_peak_eV: float) -> float:
    """Convert an atomistic-bridge registry_depth to this script's amplitude_eV."""
    return peak_to_peak_eV / PEAK_TO_PEAK_OVER_V0


def peak_to_peak_from_v0(v0_eV: float) -> float:
    """Convert this script's amplitude_eV to the atomistic-bridge convention."""
    return v0_eV * PEAK_TO_PEAK_OVER_V0


def landscape_diagnostics(amplitude_eV: float, period_nm: float) -> dict:
    """Measure the rasterised landscape and check it against the analytic form.

    Printed at startup so the depth convention in force is visible in the log
    rather than inferred from a constant's name. The assertion catches a
    MoirePotential whose normalisation has changed underneath this script --
    the failure mode that let amplitude_eV and registry_depth_meV drift apart
    in the first place.
    """
    V = MoirePotential(amplitude_eV=amplitude_eV, period_nm=period_nm)
    g = np.linspace(0.0, period_nm, 401)
    X, Y = np.meshgrid(g, g, indexing="ij")
    Z = V.value(np.column_stack([X.ravel(), Y.ravel()]))

    measured_ptp = float(Z.max() - Z.min())
    expected_ptp = peak_to_peak_from_v0(amplitude_eV)
    if abs(measured_ptp - expected_ptp) / expected_ptp > 1e-3:
        raise RuntimeError(
            f"MoirePotential no longer follows V0 * sum_i cos(G_i . r): "
            f"measured peak-to-peak {measured_ptp * 1000:.3f} meV, expected "
            f"{expected_ptp * 1000:.3f} meV for amplitude_eV={amplitude_eV}. "
            f"The depth conventions documented in this module are stale."
        )

    return {
        "V0_meV": amplitude_eV * 1000.0,
        "peak_to_peak_meV": measured_ptp * 1000.0,
        "saddle_above_min_meV": SADDLE_ABOVE_MIN_OVER_V0 * amplitude_eV * 1000.0,
        "equivalent_registry_depth_meV": expected_ptp * 1000.0,
    }


def effective_area_and_fp95(cents: np.ndarray, temperature_K: float,
                             hist_bins: int = HIST_BINS) -> tuple[float, float]:
    """A_eff and F_p95 from a 2D histogram of centroids at the given bin count.

    Both are histogram-BASED estimators, unlike the raw moments reported
    alongside them, and their variance is controlled by the number of
    samples PER BIN rather than the total sample count. At the production
    settings (13500 samples over 120x120 = 14400 bins, i.e. 0.94 samples
    per bin) the histogram is extremely sparse, which is the regime of
    maximal variance for a Shannon-entropy estimator: A_eff integrates
    P ln P over every bin, so bins flickering between 0, 1 and 2 counts
    move it directly. F_p95 is a percentile and ignores most of the
    distribution, so it is far less exposed. Sweeping hist_bins on the
    SAME samples separates this estimator variance from any sampling
    effect, at no extra Monte Carlo cost.
    """
    H, xedges, yedges = np.histogram2d(
        cents[:, 0], cents[:, 1],
        bins=hist_bins, range=[HIST_RANGE, HIST_RANGE],
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
                  sample_every: int, rng_seed: int,
                  directed_move_frac: float = 0.0,
                  directed_jitter_nm: float = 0.5,
                  hist_bins_list: tuple[int, ...] = (HIST_BINS,),
                  amplitude_eV: float = AMPLITUDE_EV,
                  period_nm: float = PERIOD_NM) -> dict:
    """Single-seed run. Returns both histogram-based observables (A_eff,
    F_p95 -- sensitive to HIST_BINS discretization noise, independent of
    grid_size) AND histogram-free observables (mean V, var V, mean r^2,
    mean COM position -- first/second moments computed directly from raw
    samples, with no binning step at all) on the SAME samples, so the two
    noise sources (interpolation-grid resolution vs. histogram estimation
    vs. seed-to-seed mixing/ergodicity) can be told apart rather than
    conflated into one number.
    """
    potential = MoirePotential(amplitude_eV=amplitude_eV, period_nm=period_nm)
    action = RingPolymerAction(
        mass_m0=MASS_M0, temperature_K=temperature_K,
        n_beads=n_beads, potential=potential,
    )
    # Lattice-directed global proposals (opt-in). An isotropic global step
    # lands on an equivalent moire minimum only by chance: the bottleneck is
    # ANGULAR, not radial, so tuning global_step_nm alone does not help
    # (measured: 15.0 -> 11.5 nm, i.e. exactly the honeycomb hop distance,
    # changes global acceptance by <6% and leaves the seed-to-seed spread of
    # the spatial observables unchanged). Quantising the proposal direction
    # to the lattice raises acceptance by more than an order of magnitude.
    hop_vectors = (
        moire_hop_vectors_nm(period_nm) if directed_move_frac > 0.0 else None
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
        global_disp_vectors_nm=hop_vectors,
        directed_move_frac=directed_move_frac,
        directed_jitter_nm=directed_jitter_nm,
    )
    result = sampler.run(n_steps=n_steps, burn_in=burn_in,
                          sample_every=sample_every,
                          center=(period_nm / 2.0, 0.0))
    samples = result["samples"]
    if samples.shape[0] == 0:
        raise RuntimeError(f"No samples for grid_size={grid_size}, seed={rng_seed} "
                            f"-- increase n_steps.")

    cents = centroids(samples)
    a_eff, f_p95 = effective_area_and_fp95(cents, temperature_K, HIST_BINS)
    # Same samples, several histogram resolutions -- a direct test of whether
    # the residual A_eff variance is an estimator artefact (see the docstring
    # of effective_area_and_fp95). Costs no extra sampling.
    a_eff_by_bins, f_p95_by_bins = {}, {}
    for nb in hist_bins_list:
        ae, fp = effective_area_and_fp95(cents, temperature_K, nb)
        a_eff_by_bins[nb] = ae
        f_p95_by_bins[nb] = fp

    # Histogram-free observables: plain moments of the raw samples, no
    # binning step anywhere.
    V_all_beads = potential.value(samples.reshape(-1, 2))
    mean_V = float(V_all_beads.mean())
    var_V = float(V_all_beads.var())
    r2_all_beads = np.einsum("ij,ij->i", samples.reshape(-1, 2), samples.reshape(-1, 2))
    mean_r2 = float(r2_all_beads.mean())
    # Second moment of the CENTROID trajectory specifically. mean_r2 above
    # averages over every bead, so it is <r_bead^2> = <R_COM^2> + <intra-ring
    # spread>, a different random variable from the one A_eff and F_p95 are
    # built from (both use centroids). Comparing spreads across observables
    # is only clean if they refer to the same variable, so report the COM
    # moment explicitly. In practice the two give the same seed-to-seed
    # relative range (verified: 17.0% vs 17.0% isotropic, 5.0% vs 5.0%
    # directed) because the intra-ring term relaxes fast and is essentially
    # seed-independent -- but the argument should not depend on that
    # coincidence holding.
    mean_r2_com = float(np.einsum("ij,ij->i", cents, cents).mean())
    mean_com = cents.mean(axis=0)

    return {
        "A_eff": a_eff,
        "F_p95": f_p95,
        "acc_local": result["acceptance_local"],
        "acc_global": result["acceptance_global"],
        "mean_V": mean_V,
        "var_V": var_V,
        "mean_r2": mean_r2,
        "mean_r2_com": mean_r2_com,
        "n_samples": int(cents.shape[0]),
        "A_eff_by_bins": a_eff_by_bins,
        "F_p95_by_bins": f_p95_by_bins,
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
            sample_every: int, rng_seeds: list[int],
            directed_move_frac: float = 0.0,
            directed_jitter_nm: float = 0.5,
            hist_bins_list: tuple[int, ...] = (HIST_BINS,),
            amplitude_eV: float = AMPLITUDE_EV,
            period_nm: float = PERIOD_NM) -> dict:
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
                     n_steps, burn_in, sample_every, seed,
                     directed_move_frac=directed_move_frac,
                     directed_jitter_nm=directed_jitter_nm,
                     hist_bins_list=hist_bins_list,
                     amplitude_eV=amplitude_eV, period_nm=period_nm)
        for seed in rng_seeds
    ]

    a_effs = np.array([r["A_eff"] for r in per_seed])
    f_p95s_meV = np.array([r["F_p95"] for r in per_seed]) * 1000.0
    mean_Vs_meV = np.array([r["mean_V"] for r in per_seed]) * 1000.0
    var_Vs = np.array([r["var_V"] for r in per_seed])
    mean_r2s = np.array([r["mean_r2"] for r in per_seed])
    mean_r2_coms = np.array([r["mean_r2_com"] for r in per_seed])
    acc_locals = np.array([r["acc_local"] for r in per_seed])
    acc_globals = np.array([r["acc_global"] for r in per_seed])

    return {
        "grid_size": grid_size,
        "n_seeds": len(rng_seeds),
        "amplitude_V0_meV": amplitude_eV * 1000.0,
        "peak_to_peak_meV": peak_to_peak_from_v0(amplitude_eV) * 1000.0,
        "period_nm": period_nm,
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
        "mean_r2_com_nm2_mean": float(mean_r2_coms.mean()),
        "mean_r2_com_relrange_pct": _relrange_pct(mean_r2_coms),
        **{
            f"A_eff_bins{nb}_relrange_pct": _relrange_pct(
                np.array([r["A_eff_by_bins"][nb] for r in per_seed]))
            for nb in hist_bins_list
        },
        **{
            f"samples_per_bin_bins{nb}": float(
                np.mean([r["n_samples"] for r in per_seed]) / (nb * nb))
            for nb in hist_bins_list
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    parser.add_argument("--directed-move-frac", type=float, default=0.0,
                         help="Fraction of global moves proposed along moire "
                              "lattice hop vectors instead of isotropically "
                              "(0 = current isotropic behaviour, the default; "
                              "0.8 measured to raise global acceptance from "
                              "~0.8%% to ~19%% and cut the seed-to-seed spread "
                              "of <r^2> from ~25%% to ~6%%). NOTE: tuning "
                              "--global-step-nm alone does NOT help -- the "
                              "bottleneck is angular, not radial.")
    parser.add_argument("--hist-bins", type=int, nargs="+", default=[HIST_BINS],
                         help="Histogram bin counts (per side) at which to "
                              "additionally evaluate A_eff, all from the SAME "
                              "samples. A_eff is a histogram-based estimator "
                              "whose variance is set by samples PER BIN: at the "
                              "production 120x120 with 13500 samples that is "
                              "0.94 samples/bin, the maximal-variance regime "
                              "for a Shannon-entropy estimator. Sweeping this "
                              "(e.g. --hist-bins 30 40 60 120) tests that "
                              "directly at no extra sampling cost.")
    parser.add_argument("--directed-jitter-nm", type=float, default=0.5,
                         help="Gaussian jitter added to a directed proposal, nm")
    parser.add_argument("--amplitude-eV", type=float, default=AMPLITUDE_EV,
                         dest="amplitude_eV",
                         help="Moire corrugation V0 -- the FIRST-SHELL FOURIER "
                              "AMPLITUDE, not the peak-to-peak span. The "
                              "landscape spans 4.5*V0. Default %(default)s "
                              "(legacy, keeps the published campaign "
                              "reproducible). Relaxed-DFT calibration for "
                              "MoSe2/WSe2 gives 0.12160. To reproduce an "
                              "atomistic-bridge registry_depth_meV of D, pass "
                              "D/4500.")
    parser.add_argument("--period-nm", type=float, default=PERIOD_NM,
                         dest="period_nm",
                         help="Moire period in nm (default: %(default)s)")
    parser.add_argument("--output", type=Path,
                         default=Path("results/grid_resolution_convergence.csv"))
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    diag = landscape_diagnostics(args.amplitude_eV, args.period_nm)
    print("Landscape in force (see module docstring for the depth conventions):")
    print(f"  V0, first-shell amplitude   : {diag['V0_meV']:8.3f} meV")
    print(f"  peak-to-peak span           : {diag['peak_to_peak_meV']:8.3f} meV")
    print(f"  saddle above minimum        : {diag['saddle_above_min_meV']:8.3f} meV")
    print(f"  equivalent registry_depth   : "
          f"{diag['equivalent_registry_depth_meV']:8.3f} meV "
          f"(atomistic-bridge convention)")
    print(f"  period                      : {args.period_nm:8.3f} nm\n")

    rows = []
    for gs in args.grid_sizes:
        print(f"Running grid_size={gs} (T={args.temperature} K, P={args.n_beads}, "
              f"{args.n_seeds} seeds) ...")
        rows.append(run_one(gs, args.temperature, args.n_beads, args.global_step_nm,
                             args.n_steps, args.burn_in, args.sample_every, seeds,
                             directed_move_frac=args.directed_move_frac,
                             directed_jitter_nm=args.directed_jitter_nm,
                             hist_bins_list=tuple(args.hist_bins),
                             amplitude_eV=args.amplitude_eV,
                             period_nm=args.period_nm))

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
