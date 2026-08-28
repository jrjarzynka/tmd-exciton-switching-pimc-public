#!/usr/bin/env python3
"""
Seed-uncertainty campaign for the two remaining single-chain numbers.

Part A -- straddling fraction ("quantum delocalization signature")
    The 25.6% quoted three times in the manuscript comes from
    test3b_tunneling_snapshot.py, a single chain at rng_seed=2024, with no
    uncertainty attached. This repeats it over independent chains.

Part B -- harmonic benchmark seed check
    Fig. 5 / Table 3 report <R^2> from a single chain (rng_seed=42) against an
    analytic reference. <R^2> is a fast observable, so the seed scatter is
    expected to be far smaller than for basin occupation -- but "expected" is
    not "measured". This checks it at the two most demanding temperatures.
    If the scatter is well under the quoted sub-1% agreement, Fig. 5 and
    Table 3 stand unchanged and you can say so.

Neither part changes any published file: results go to new CSVs.

Usage
-----
    python3 run_seed_uncertainty_campaign.py                # both parts
    python3 run_seed_uncertainty_campaign.py --part A --seeds 8
    python3 run_seed_uncertainty_campaign.py --workers 8

Output: straddling_multiseed.csv
        harmonic_seed_check.csv
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tmd_pimc import (RingPolymerAction, DoubleGaussianWellPotential,
                      HarmonicPotential, PIMCSamplerStaging)
from tmd_pimc.analytic import (harmonic_r2_analytic,
                               harmonic_r2_primitive_finite_P)
from tmd_pimc.observables import r2_mean_pimc

# --- shared parameters: identical to the published scripts -------------------
MASS_M0 = 0.5
T_K = 20.0
N_BEADS = 32
SEPARATION_NM = 10.0
SIGMA_NM = 3.0
V0_EV = 0.05

N_STEPS = 60000
BURN_IN = 15000
SAMPLE_EVERY = 20

# Part B parameters, from test1_single_well.py
K_EV_PER_NM2 = 0.010
N_BEADS_HARMONIC = 80
T_CHECK_K = [5.0, 10.0, 15.0, 20.0, 50.0, 100.0]     # coldest points: worst case for sampling

# Offsets chosen to avoid every published seed (42, 200-208, 300-308, 888, 2024)
SEED_BASE_A = 7000
SEED_BASE_B = 8000


def straddling_chain(i_seed):
    """Fraction of ring polymers supported across both wells, one chain."""
    seed = SEED_BASE_A + i_seed
    dw = DoubleGaussianWellPotential(V0_eV=V0_EV, sigma_nm=SIGMA_NM,
                                     separation_nm=SEPARATION_NM,
                                     asymmetry_eV=0.0)
    action = RingPolymerAction(mass_m0=MASS_M0, temperature_K=T_K,
                               n_beads=N_BEADS, potential=dw)
    out = PIMCSamplerStaging(action=action, rng_seed=seed).run(
        n_steps=N_STEPS, burn_in=BURN_IN, sample_every=SAMPLE_EVERY)
    s = out["samples"]
    # Identical criterion to test3b: bead fraction in the left well, counted
    # as straddling when strictly between 5% and 95%.
    left_frac = np.mean(s[:, :, 0] < 0, axis=1)
    straddling = (left_frac > 0.05) & (left_frac < 0.95)
    return seed, float(np.mean(straddling)), int(s.shape[0])


def harmonic_chain(job):
    """<R^2> for one temperature and one seed, plus the analytic references."""
    i_T, i_seed = job
    T = T_CHECK_K[i_T]
    seed = SEED_BASE_B + 100 * i_T + i_seed
    pot = HarmonicPotential(k_eV_per_nm2=K_EV_PER_NM2)
    action = RingPolymerAction(mass_m0=MASS_M0, temperature_K=T,
                               n_beads=N_BEADS_HARMONIC, potential=pot)
    out = PIMCSamplerStaging(action=action, rng_seed=seed).run(
        n_steps=N_STEPS, burn_in=BURN_IN, sample_every=SAMPLE_EVERY)
    r2 = float(r2_mean_pimc(out["samples"]))
    ref_P = float(harmonic_r2_primitive_finite_P(MASS_M0, K_EV_PER_NM2, T,
                                                 N_BEADS_HARMONIC))
    return i_T, T, seed, r2, ref_P


def run_jobs(fn, jobs, workers):
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, jobs))
    return [fn(j) for j in jobs]


def part_a(seeds, workers):
    print(f"\n=== Part A: straddling fraction, {seeds} chains ===")
    t0 = time.time()
    res = run_jobs(straddling_chain, list(range(seeds)), workers)
    vals = np.array([r[1] for r in res])

    with open("straddling_multiseed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "straddling_fraction", "n_samples"])
        for seed, v, n in res:
            w.writerow([seed, f"{v:.6f}", n])
            print(f"  seed={seed}  straddling={100*v:.1f}%")

    m = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else float("nan")
    print(f"  -> {100*m:.1f}% +/- {100*sem:.1f}% (sem, n={len(vals)}); "
          f"per-chain sd {100*vals.std(ddof=1):.1f}%")
    print(f"  published single-chain value: 25.6% (seed 2024)")
    print(f"  elapsed {time.time()-t0:.0f} s")
    return m, sem


def part_b(seeds, workers):
    print(f"\n=== Part B: harmonic benchmark seed check, "
          f"{len(T_CHECK_K)} T x {seeds} chains ===")
    t0 = time.time()
    jobs = [(i, s) for i in range(len(T_CHECK_K)) for s in range(seeds)]
    res = run_jobs(harmonic_chain, jobs, workers)

    with open("harmonic_seed_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_K", "seed", "R2_pimc_nm2", "R2_exact_finiteP_nm2",
                    "err_pct"])
        for i_T, T, seed, r2, ref in sorted(res):
            w.writerow([T, seed, f"{r2:.6f}", f"{ref:.6f}",
                        f"{100*(r2-ref)/ref:+.4f}"])

    for i_T, T in enumerate(T_CHECK_K):
        v = np.array([r[3] for r in res if r[0] == i_T])
        ref = [r[4] for r in res if r[0] == i_T][0]
        rel_sd = 100 * v.std(ddof=1) / v.mean()
        print(f"  T={T:5.1f} K  <R^2> = {v.mean():.4f} nm^2, "
              f"per-chain sd = {rel_sd:.3f}% of mean; "
              f"mean error vs exact finite-P = {100*(v.mean()-ref)/ref:+.3f}%")
    print(f"  If the sd stays well below the sub-1% agreement quoted in the")
    print(f"  text, Fig. 5 and Table 3 need no change.")
    print(f"  elapsed {time.time()-t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()

    if a.part in ("A", "both"):
        part_a(a.seeds, a.workers)
    if a.part in ("B", "both"):
        part_b(a.seeds, a.workers)

    print("\nwritten: straddling_multiseed.csv, harmonic_seed_check.csv")


if __name__ == "__main__":
    sys.exit(main())
