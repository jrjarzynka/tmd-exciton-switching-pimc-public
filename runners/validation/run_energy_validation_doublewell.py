"""Validation IV (double well): primitive vs. virial energy estimator.

Re-runs the same symmetric double-well landscape used in Sec. 5 of the
manuscript (V0 = 50 meV, sigma = 3 nm, separation = 10 nm, Delta_0 = 0,
T = 20 K, P = 32), at zero field (Ex = 0) -- i.e. the same landscape
already used for the tunnelling-paths and free-energy-map figures --
computes both energy estimators, and reports autocorrelation-corrected
error bars.

This is a non-harmonic potential with no simple closed-form exact energy
reference, so the validation criterion here is agreement between the two
INDEPENDENT estimators (primitive and virial) on the same samples, not
agreement with an external exact value. See energy_estimators.py for the
formulas and their independent verification against the harmonic case
(where an exact reference does exist).

Usage
-----
    python run_energy_validation_doublewell.py
    python run_energy_validation_doublewell.py --n-beads 32 80 --temperature 20

Requires the tmd_pimc package to be importable (same convention as the
other runner scripts in this project).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tmd_pimc import RingPolymerAction, PIMCSampler, DoubleGaussianWellPotential
from tmd_pimc.energy_estimators import primitive_energy_series, virial_energy_series
from tmd_pimc.statistics import analyze_timeseries

MASS_M0 = 0.5
V0_EV = 0.05
SIGMA_NM = 3.0
SEPARATION_NM = 10.0
ASYMMETRY_EV = 0.0  # Delta_0 = 0, matches the symmetric well used in Sec. 5


def run_one(temperature_K: float, n_beads: int, n_steps: int, burn_in: int,
            sample_every: int, rng_seed: int) -> dict:
    potential = DoubleGaussianWellPotential(
        V0_eV=V0_EV, sigma_nm=SIGMA_NM,
        separation_nm=SEPARATION_NM, asymmetry_eV=ASYMMETRY_EV,
    )
    action = RingPolymerAction(
        mass_m0=MASS_M0,
        temperature_K=temperature_K,
        n_beads=n_beads,
        potential=potential,
    )
    sampler = PIMCSampler(action=action, rng_seed=rng_seed)
    # Start near one well; global moves + burn-in handle equilibration
    # across both basins, matching the protocol already used in Sec. 5/6.
    result = sampler.run(n_steps=n_steps, burn_in=burn_in,
                          sample_every=sample_every,
                          center=(SEPARATION_NM / 2.0, 0.0))
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
    parser.add_argument("--temperature", type=float, default=20.0,
                         help="Temperature in K (default: 20 K, matches Sec. 5)")
    parser.add_argument("--n-beads", type=int, nargs="+", default=[32],
                         help="Bead count(s) to run (default: 32, matches Sec. 5). "
                              "Pass multiple values e.g. '32 80' for a P cross-check.")
    parser.add_argument("--n-steps", type=int, default=200_000)
    parser.add_argument("--burn-in", type=int, default=20_000)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--rng-seed", type=int, default=1234)
    parser.add_argument("--output", type=Path,
                         default=Path("results/energy_validation_doublewell.csv"))
    args = parser.parse_args()

    rows = []
    for P in args.n_beads:
        print(f"Running T={args.temperature} K, P={P} (symmetric double well, Ex=0) ...")
        rows.append(run_one(args.temperature, P, args.n_steps, args.burn_in,
                             args.sample_every, args.rng_seed))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {args.output}\n")
    header = (f"{'T (K)':>8} {'P':>5} {'E_prim':>14} {'E_vir':>14} "
              f"{'agree (sigma)':>14} {'std_prim/std_vir':>18}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['temperature_K']:>8.1f} {row['n_beads']:>5d} "
              f"{row['E_prim_mean']:>8.5f}+-{row['E_prim_err']:.5f} "
              f"{row['E_vir_mean']:>8.5f}+-{row['E_vir_err']:.5f} "
              f"{row['agreement_sigma']:>14.2f} "
              f"{row['std_ratio_prim_over_vir']:>18.2f}")


if __name__ == "__main__":
    main()
