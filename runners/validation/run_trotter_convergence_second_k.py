"""Trotter/bead-count convergence at a second spring constant.

Sec. 4 validates the adaptive bead-count rule P(T) = max(32, floor(400/T))
and the sampler's accuracy against the exact finite-P harmonic reference
at a single spring constant, k = 0.010 eV/nm^2. This script repeats the
SAME already-validated comparison (sampler vs.
analytic.harmonic_r2_primitive_finite_P -- no new formula, no new
estimator) at a second, distinct k, to check that the production P(T)
rule is not implicitly tuned to one specific curvature.

Choose the second k to bracket the production value from the other side:
a shallower well (smaller k, slower quantum decoherence, more demanding
for a fixed P) and/or a stiffer well (larger k) are both informative;
the default below tests one shallower and one stiffer value in the same
run.

Usage
-----
    python run_trotter_convergence_second_k.py
    python run_trotter_convergence_second_k.py --k-values 0.005 0.010 0.020
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tmd_pimc import RingPolymerAction, PIMCSampler, HarmonicPotential
from tmd_pimc.observables import r2_mean_pimc
from tmd_pimc.analytic import harmonic_r2_primitive_finite_P

MASS_M0 = 0.5


def run_one(k_eV_per_nm2: float, temperature_K: float, n_beads: int,
            n_steps: int, burn_in: int, sample_every: int, rng_seed: int) -> dict:
    potential = HarmonicPotential(k_eV_per_nm2=k_eV_per_nm2)
    action = RingPolymerAction(
        mass_m0=MASS_M0, temperature_K=temperature_K,
        n_beads=n_beads, potential=potential,
    )
    sampler = PIMCSampler(action=action, rng_seed=rng_seed)
    result = sampler.run(n_steps=n_steps, burn_in=burn_in, sample_every=sample_every)
    samples = result["samples"]
    if samples.shape[0] == 0:
        raise RuntimeError(f"No samples for k={k_eV_per_nm2}, P={n_beads} -- increase n_steps.")

    r2_pimc = r2_mean_pimc(samples)
    r2_exact = harmonic_r2_primitive_finite_P(MASS_M0, k_eV_per_nm2, temperature_K, n_beads)
    rel_err_pct = 100.0 * (r2_pimc - r2_exact) / r2_exact

    return {
        "k_eV_per_nm2": k_eV_per_nm2,
        "temperature_K": temperature_K,
        "n_beads": n_beads,
        "n_samples": samples.shape[0],
        "r2_pimc_nm2": r2_pimc,
        "r2_exact_finite_P_nm2": r2_exact,
        "rel_err_pct": rel_err_pct,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=float, nargs="+", default=[0.005, 0.020],
                         help="Spring constants to test, eV/nm^2 (default: brackets "
                              "the production k=0.010 from Sec. 4 on both sides)")
    parser.add_argument("--temperature", type=float, default=5.0,
                         help="Temperature in K (default: 5 K, the most demanding "
                              "point in the cryogenic protocol)")
    parser.add_argument("--p-values", type=int, nargs="+", default=[20, 40, 80, 140, 200],
                         help="Bead counts to scan (default: matches Table 4, Sec. 4)")
    parser.add_argument("--n-steps", type=int, default=200_000)
    parser.add_argument("--burn-in", type=int, default=20_000)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--rng-seed", type=int, default=1234)
    parser.add_argument("--output", type=Path,
                         default=Path("results/trotter_convergence_second_k.csv"))
    args = parser.parse_args()

    rows = []
    for k in args.k_values:
        for P in args.p_values:
            print(f"Running k={k} eV/nm^2, T={args.temperature} K, P={P} ...")
            rows.append(run_one(k, args.temperature, P, args.n_steps,
                                 args.burn_in, args.sample_every, args.rng_seed))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {args.output}\n")
    header = f"{'k (eV/nm2)':>12} {'P':>5} {'r2_pimc':>10} {'r2_exact':>10} {'rel_err (%)':>12}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['k_eV_per_nm2']:>12.4f} {row['n_beads']:>5d} "
              f"{row['r2_pimc_nm2']:>10.4f} {row['r2_exact_finite_P_nm2']:>10.4f} "
              f"{row['rel_err_pct']:>12.2f}")

    max_err = max(abs(row["rel_err_pct"]) for row in rows)
    print(f"\nMax |relative error| across all (k, P) tested: {max_err:.2f}%")


if __name__ == "__main__":
    main()
