"""Validation IV (harmonic): primitive vs. virial energy estimator.

Re-runs the same harmonic-oscillator benchmark landscape used in Sec. 4 of
the manuscript (M_X = 0.5 m0, k = 0.010 eV/nm^2), computes BOTH the
primitive and virial total-energy estimators on the returned raw path
samples, and reports autocorrelation-corrected error bars via the
existing statistics.analyze_timeseries machinery.

Success criterion (see energy_estimators.py docstring for the formulas
and their independent numerical verification): E_prim and E_vir should
agree within combined statistical error at every (T, P) point, and the
ratio std(E_prim)/std(E_vir) should grow with P (virial has bounded
variance; primitive does not).

Usage
-----
    python run_energy_validation_harmonic.py
    python run_energy_validation_harmonic.py --temperatures 5 10 15 20 50 100 --n-beads 80
    python run_energy_validation_harmonic.py --p-scan --temperature 5

Requires the tmd_pimc package to be importable (same convention as the
other runner scripts in this project).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tmd_pimc import RingPolymerAction, PIMCSampler, HarmonicPotential
from tmd_pimc.energy_estimators import primitive_energy_series, virial_energy_series
from tmd_pimc.statistics import analyze_timeseries

MASS_M0 = 0.5
K_EV_PER_NM2 = 0.010


def run_one(temperature_K: float, n_beads: int, n_steps: int, burn_in: int,
            sample_every: int, rng_seed: int) -> dict:
    potential = HarmonicPotential(k_eV_per_nm2=K_EV_PER_NM2)
    action = RingPolymerAction(
        mass_m0=MASS_M0,
        temperature_K=temperature_K,
        n_beads=n_beads,
        potential=potential,
    )
    sampler = PIMCSampler(action=action, rng_seed=rng_seed)
    result = sampler.run(n_steps=n_steps, burn_in=burn_in, sample_every=sample_every)
    samples = result["samples"]

    if samples.shape[0] == 0:
        raise RuntimeError(
            f"No samples returned for T={temperature_K} K, P={n_beads} "
            f"(n_steps={n_steps}, burn_in={burn_in}) -- increase n_steps."
        )

    e_prim_series = primitive_energy_series(samples, potential, MASS_M0, temperature_K)
    e_vir_series = virial_energy_series(samples, potential)

    prim_stats = analyze_timeseries(e_prim_series)
    vir_stats = analyze_timeseries(e_vir_series)

    n = len(e_prim_series)
    e_prim_mean = float(e_prim_series.mean())
    e_vir_mean = float(e_vir_series.mean())
    e_prim_err = float(e_prim_series.std() / np.sqrt(prim_stats["ESS"]))
    e_vir_err = float(e_vir_series.std() / np.sqrt(vir_stats["ESS"]))

    return {
        "temperature_K": temperature_K,
        "n_beads": n_beads,
        "n_samples": n,
        "acceptance_local": result["acceptance_local"],
        "acceptance_global": result["acceptance_global"],
        "E_prim_mean": e_prim_mean,
        "E_prim_err": e_prim_err,
        "E_prim_tau_int": prim_stats["tau_int"],
        "E_prim_ESS": prim_stats["ESS"],
        "E_vir_mean": e_vir_mean,
        "E_vir_err": e_vir_err,
        "E_vir_tau_int": vir_stats["tau_int"],
        "E_vir_ESS": vir_stats["ESS"],
        "agreement_sigma": abs(e_prim_mean - e_vir_mean) / np.sqrt(e_prim_err**2 + e_vir_err**2),
        "std_ratio_prim_over_vir": float(e_prim_series.std() / e_vir_series.std()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperatures", type=float, nargs="+",
                         default=[5, 10, 15, 20, 50, 100],
                         help="Temperatures in K (default: matches Table 3, Sec. 4)")
    parser.add_argument("--n-beads", type=int, default=80,
                         help="Fixed P for the temperature scan (default: 80, matches Sec. 4)")
    parser.add_argument("--p-scan", action="store_true",
                         help="Instead of a temperature scan, scan P at a fixed temperature "
                              "(matches the complementary bead-convergence check in Sec. 4)")
    parser.add_argument("--p-values", type=int, nargs="+",
                         default=[20, 40, 60, 80, 100, 140, 200],
                         help="Bead counts for --p-scan (default: matches Table 4, Sec. 4)")
    parser.add_argument("--temperature", type=float, default=5.0,
                         help="Fixed temperature for --p-scan (default: 5 K, matches Sec. 4)")
    parser.add_argument("--n-steps", type=int, default=200_000)
    parser.add_argument("--burn-in", type=int, default=20_000)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--rng-seed", type=int, default=1234)
    parser.add_argument("--output", type=Path,
                         default=Path("results/energy_validation_harmonic.csv"))
    args = parser.parse_args()

    rows = []
    if args.p_scan:
        for P in args.p_values:
            print(f"Running T={args.temperature} K, P={P} ...")
            rows.append(run_one(args.temperature, P, args.n_steps, args.burn_in,
                                 args.sample_every, args.rng_seed))
    else:
        for T in args.temperatures:
            print(f"Running T={T} K, P={args.n_beads} ...")
            rows.append(run_one(T, args.n_beads, args.n_steps, args.burn_in,
                                 args.sample_every, args.rng_seed))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {args.output}\n")
    header = (f"{'T (K)':>8} {'P':>5} {'E_prim':>12} {'E_vir':>12} "
              f"{'agree (sigma)':>14} {'std_prim/std_vir':>18}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['temperature_K']:>8.1f} {row['n_beads']:>5d} "
              f"{row['E_prim_mean']:>7.5f}+-{row['E_prim_err']:.5f} "
              f"{row['E_vir_mean']:>7.5f}+-{row['E_vir_err']:.5f} "
              f"{row['agreement_sigma']:>14.2f} "
              f"{row['std_ratio_prim_over_vir']:>18.2f}")


if __name__ == "__main__":
    main()
