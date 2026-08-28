#!/usr/bin/env python3
"""
Test 3 v4: symmetric double well, occupation scan through Ex = 0, WITH
seed-to-seed error bars.

Why this supersedes v3
----------------------
v3 ran one seed per field point (seed = 200 + i) and quoted a binomial
standard error computed as if the 2250 stored centroids were independent
draws. They are not: they are thinned samples from a correlated Markov chain,
and the basin-occupation indicator is sensitive to the slowest mode in the
chain (barrier crossing), not the fastest. Comparing the v3 scan against the
independently seeded filmstrip run (seed = 300 + i) showed disagreements up to
0.079 in fraction_right -- roughly three times the quoted binomial error,
implying an effective sample size of order 250, not 2250.

This script measures that scatter directly instead of assuming it away: it
repeats every field point over N_SEEDS independent chains and reports the mean
and the standard error of the mean across seeds. That is the error bar that
belongs on the figure and in the text.

Cost: N_FIELDS x N_SEEDS chains. On one core, ~82 s per chain, so the default
9 x 8 = 72 chains takes about 100 minutes. Reduce N_SEEDS to 4 for a quick
look; use --workers to spread across cores.

Usage
-----
    python3 test3_double_well_v4_multiseed.py                  # full campaign
    python3 test3_double_well_v4_multiseed.py --seeds 4        # quicker
    python3 test3_double_well_v4_multiseed.py --workers 8      # parallel
    python3 test3_double_well_v4_multiseed.py --only-index 5   # one field point

Output: occ_scan_v4_multiseed.csv   (per-seed values, long format)
        occ_scan_v4_summary.csv     (mean, sem, n_seeds per field point)
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from tmd_pimc import (RingPolymerAction, DoubleGaussianWellPotential,
                      ExternalFieldPotential, CompositePotential,
                      PIMCSamplerStaging)
from tmd_pimc.observables import centroids

# --- physical parameters: identical to v3, do not change ---------------------
MASS_M0 = 0.5
T_K = 20.0
N_BEADS = 32
SEPARATION_NM = 10.0
SIGMA_NM = 3.0
V0_EV = 0.05
Q_EFF = 1.0

N_STEPS = 60000
BURN_IN = 15000
SAMPLE_EVERY = 20

KB = 8.617333262e-5
EX_CHAR = KB * T_K / (Q_EFF * SEPARATION_NM)
EX_SCAN = np.linspace(-3 * EX_CHAR, 3 * EX_CHAR, 9)

# Seed offset chosen to avoid overlapping either published run (200-208 for the
# v3 scan, 300-308 for the filmstrip), so this campaign is independent of both.
SEED_BASE = 5000


def one_chain(args):
    """Run a single chain and return its basin occupation."""
    i_field, i_seed = args
    Ex = float(EX_SCAN[i_field])
    seed = SEED_BASE + 100 * i_field + i_seed

    dw = DoubleGaussianWellPotential(V0_eV=V0_EV, sigma_nm=SIGMA_NM,
                                     separation_nm=SEPARATION_NM,
                                     asymmetry_eV=0.0)
    field = ExternalFieldPotential(E=(Ex, 0.0), q_eff=Q_EFF)
    potential = CompositePotential(terms=[dw, field])
    action = RingPolymerAction(mass_m0=MASS_M0, temperature_K=T_K,
                               n_beads=N_BEADS, potential=potential)
    sampler = PIMCSamplerStaging(action=action, rng_seed=seed)
    out = sampler.run(n_steps=N_STEPS, burn_in=BURN_IN,
                      sample_every=SAMPLE_EVERY)
    cents = centroids(out["samples"])
    return i_field, i_seed, Ex, seed, float(np.mean(cents[:, 0] > 0)), cents.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8,
                    help="independent chains per field point (default 8)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes (default 1)")
    ap.add_argument("--only-index", type=int, default=None,
                    help="run a single field point by index 0-8")
    args = ap.parse_args()

    fields = ([args.only_index] if args.only_index is not None
              else list(range(len(EX_SCAN))))
    jobs = [(i, s) for i in fields for s in range(args.seeds)]

    print(f"Ex_char = {EX_CHAR:.6f} eV/nm")
    print(f"{len(jobs)} chains "
          f"({len(fields)} field points x {args.seeds} seeds), "
          f"{args.workers} worker(s)")
    t0 = time.time()

    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(one_chain, jobs):
                results.append(r)
                print(f"  Ex={r[2]:+.6f} seed={r[3]} f_R={r[4]:.4f}", flush=True)
    else:
        for j in jobs:
            r = one_chain(j)
            results.append(r)
            print(f"  Ex={r[2]:+.6f} seed={r[3]} f_R={r[4]:.4f}", flush=True)

    print(f"elapsed: {time.time() - t0:.0f} s")

    with open("occ_scan_v4_multiseed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i_field", "Ex_eV_per_nm", "seed", "fraction_right", "n_samples"])
        for i_f, i_s, Ex, seed, fr, n in sorted(results):
            w.writerow([i_f, f"{Ex:.14f}", seed, f"{fr:.6f}", n])

    with open("occ_scan_v4_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Ex_eV_per_nm", "fraction_right_mean",
                    "fraction_right_sem", "n_seeds", "binomial_se_if_independent"])
        for i_f in sorted(set(r[0] for r in results)):
            vals = np.array([r[4] for r in results if r[0] == i_f])
            n_s = len(vals)
            Ex = EX_SCAN[i_f]
            sem = vals.std(ddof=1) / np.sqrt(n_s) if n_s > 1 else float("nan")
            p = vals.mean()
            n_samp = [r[5] for r in results if r[0] == i_f][0]
            binom = np.sqrt(p * (1 - p) / n_samp)
            w.writerow([f"{Ex:.14f}", f"{p:.6f}", f"{sem:.6f}", n_s, f"{binom:.6f}"])
            print(f"Ex={Ex:+.6f}  f_R = {p:.4f} +/- {sem:.4f} (n={n_s})   "
                  f"binomial-if-independent would claim +/- {binom:.4f}")

    print("\nwritten: occ_scan_v4_multiseed.csv, occ_scan_v4_summary.csv")


if __name__ == "__main__":
    sys.exit(main())
