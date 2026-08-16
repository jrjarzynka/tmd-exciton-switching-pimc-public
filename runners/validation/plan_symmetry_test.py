#!/usr/bin/env python3
"""
plan_symmetry_test.py

How many seeds, and on what field grid, does the landscape-symmetry test need?

The question this answers
-------------------------
compare_landscape_symmetry.py can only DISPROVE the V0-independence of F_z,50
unless the confidence interval on the shift is narrow enough to call the two
scans equal. That precision has to be bought before the run, not discovered
after it. This script simulates the whole measurement -- binomial seed
outcomes, threshold classification, threshold extraction, bootstrap -- and
reports the confidence interval a given design would deliver.

Two estimators are compared, because the choice matters more than the seed count
----------------------------------------------------------------------------
INTERPOLATION uses only the two field points bracketing the 50% crossing. Every
other point in the scan, each carrying a full binomial sample, is discarded.

LOGISTIC MLE fits p(F) = 1 / (1 + exp(-(F - F50)/s)) to all points at once by
maximum likelihood on the binomial counts. It uses the entire curve, so it
extracts the shape information that interpolation throws away and it degrades
gracefully when no single point sits near 50%.

The efficiency ratio between them is reported in seeds: how many seeds
interpolation needs to match the MLE's precision. Buying precision with a better
estimator is free; buying it with seeds costs core-hours.

What must be supplied honestly
------------------------------
The transition width `s` is the one input that cannot be guessed from first
principles -- it is a property of the system. Take it from an existing scan:
fit the fraction curve, or read the field interval over which the fraction runs
from 0.1 to 0.9 and divide by 4.4 (the logistic 10-90 width in units of s).
A scan planned with too sharp an assumed transition will be under-powered.

Usage
-----
    python plan_symmetry_test.py --f50 3.0 --width-s 0.15 \\
        --field-min 2.0 --field-max 4.0 \\
        --seconds-per-seed 90 --target-ci-rel 10
"""

from __future__ import annotations

import argparse
import math
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import minimize


# ----------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------
def interp_threshold(fields: np.ndarray, fracs: np.ndarray,
                     level: float = 0.5) -> Optional[float]:
    """Linear interpolation between the two points bracketing `level`."""
    order = np.argsort(fields)
    f, p = fields[order], fracs[order]
    if p.min() > level or p.max() < level:
        return None
    for i in range(len(f) - 1):
        if (p[i] - level) * (p[i + 1] - level) <= 0:
            if p[i + 1] == p[i]:
                return float(f[i])
            t = (level - p[i]) / (p[i + 1] - p[i])
            return float(f[i] + t * (f[i + 1] - f[i]))
    return None


def logistic_mle(fields: np.ndarray, counts: np.ndarray, n_seeds: int
                 ) -> Optional[tuple]:
    """Maximum-likelihood fit of p(F) = 1/(1+exp(-(F-F50)/s)) to binomial counts.

    Returns (F50, s), or None if the optimiser fails. The likelihood is the
    binomial one, not a least-squares fit to the fractions: a fraction of 0 or 1
    carries different information from a fraction of 0.5 at the same seed count,
    and least squares cannot see that difference.
    """
    k = np.asarray(counts, dtype=float)
    n = float(n_seeds)

    def nll(theta):
        f50, log_s = theta
        s = math.exp(log_s)
        z = (fields - f50) / s
        # log(1+exp(-z)) computed stably
        log_p = -np.logaddexp(0.0, -z)
        log_q = -np.logaddexp(0.0, z)
        return -float(np.sum(k * log_p + (n - k) * log_q))

    # Initialise from the empirical crossing where one exists, else the midpoint.
    p_emp = k / n
    f0 = interp_threshold(fields, p_emp)
    if f0 is None:
        f0 = float(np.mean(fields))
    s0 = max((fields.max() - fields.min()) / 8.0, 1e-3)

    try:
        res = minimize(nll, x0=[f0, math.log(s0)], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    except Exception:
        return None
    if not res.success:
        return None
    f50, log_s = res.x
    if not (fields.min() - 2.0 <= f50 <= fields.max() + 2.0):
        # An F50 far outside the scanned window is an extrapolation, not a
        # measurement; report failure rather than a confident wrong number.
        return None
    return float(f50), float(math.exp(log_s))


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------
def simulate_scan(fields: np.ndarray, f50: float, s: float, n_seeds: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Binomial dissociation counts at each field, for one realisation."""
    p = 1.0 / (1.0 + np.exp(-(fields - f50) / s))
    return rng.binomial(n_seeds, p)


def precision_of_design(fields: np.ndarray, f50: float, s: float, n_seeds: int,
                        n_trials: int, rng: np.random.Generator) -> dict:
    """Spread of each estimator over repeated realisations of the same design.

    The spread across independent realisations IS the standard error, measured
    rather than assumed -- no normality, no delta method, and it accounts for
    the realisations in which an estimator fails outright.
    """
    interp_vals, mle_vals = [], []
    interp_fail = mle_fail = 0
    for _ in range(n_trials):
        counts = simulate_scan(fields, f50, s, n_seeds, rng)
        fr = counts / n_seeds
        t = interp_threshold(fields, fr)
        if t is None:
            interp_fail += 1
        else:
            interp_vals.append(t)
        m = logistic_mle(fields, counts, n_seeds)
        if m is None:
            mle_fail += 1
        else:
            mle_vals.append(m[0])

    def summarise(vals, fails):
        a = np.asarray(vals)
        if a.size < 10:
            return {"sd": float("nan"), "bias": float("nan"),
                    "ci_rel": float("nan"), "fail_rate": fails / n_trials}
        sd = float(np.std(a, ddof=1))
        return {"sd": sd, "bias": float(np.mean(a) - f50),
                "ci_rel": 100.0 * 2.0 * 1.96 * sd / f50,
                "fail_rate": fails / n_trials}

    return {"interp": summarise(interp_vals, interp_fail),
            "mle": summarise(mle_vals, mle_fail)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f50", type=float, required=True,
                    help="assumed true threshold, eV/nm (from an existing scan)")
    ap.add_argument("--width-s", type=float, required=True,
                    help="logistic width parameter s, eV/nm; the 10-90 width "
                         "of the transition divided by 4.4")
    ap.add_argument("--field-min", type=float, required=True)
    ap.add_argument("--field-max", type=float, required=True)
    ap.add_argument("--field-steps", type=float, nargs="+",
                    default=[0.5, 0.25, 0.1, 0.05],
                    help="field grid spacings to evaluate, eV/nm")
    ap.add_argument("--seed-counts", type=int, nargs="+",
                    default=[8, 16, 24, 48, 96],
                    help="seeds per field point to evaluate")
    ap.add_argument("--target-ci-rel", type=float, default=10.0,
                    help="target 95%% CI width on the SHIFT, as a percentage "
                         "of F_z,50")
    ap.add_argument("--seconds-per-seed", type=float, default=None,
                    help="wall-clock per seed-run, for a core-hour estimate")
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.rng_seed)

    print(f"Assumed transition : F_z,50 = {args.f50} eV/nm, "
          f"s = {args.width_s} eV/nm "
          f"(10-90 width {4.4 * args.width_s:.3f} eV/nm)")
    print(f"Field window       : {args.field_min} to {args.field_max} eV/nm")
    print(f"Target             : 95% CI on the shift < {args.target_ci_rel:.0f}% "
          f"of F_z,50")
    print()
    print("The CI on the SHIFT between two scans is sqrt(2) times that on a")
    print("single threshold, since both scans carry independent error.")
    print()

    target_single = args.target_ci_rel / math.sqrt(2.0)

    header = (f"{'step':>7} {'pts':>4} {'seeds':>6} | "
              f"{'interp CI%':>11} {'fail%':>6} | {'MLE CI%':>9} {'fail%':>6} | "
              f"{'runs':>6}")
    print(header)
    print("-" * len(header))

    viable = []
    for step in args.field_steps:
        fields = np.arange(args.field_min, args.field_max + 1e-9, step)
        for n_seeds in args.seed_counts:
            r = precision_of_design(fields, args.f50, args.width_s,
                                    n_seeds, args.n_trials, rng)
            runs = len(fields) * n_seeds
            i, m = r["interp"], r["mle"]
            print(f"{step:7.3f} {len(fields):4d} {n_seeds:6d} | "
                  f"{i['ci_rel']:11.1f} {100 * i['fail_rate']:6.1f} | "
                  f"{m['ci_rel']:9.1f} {100 * m['fail_rate']:6.1f} | "
                  f"{runs:6d}")
            for name, res in (("interp", i), ("mle", m)):
                if (math.isfinite(res["ci_rel"]) and res["ci_rel"] < target_single
                        and res["fail_rate"] < 0.05):
                    viable.append((runs, name, step, n_seeds, res["ci_rel"]))

    print()
    if not viable:
        print("No design in the grid searched reaches the target. Widen the")
        print("seed counts, or accept a looser target -- and note that a scan")
        print("which cannot reach it can still DISPROVE independence, which may")
        print("be all that is needed.")
        return 0

    viable.sort()
    print("Cheapest designs reaching the target (fewest seed-runs first):")
    for runs, name, step, n_seeds, ci in viable[:6]:
        line = (f"  {runs:6d} runs  |  step {step:.3f} eV/nm, {n_seeds} seeds/pt"
                f"  |  {name} estimator, CI {ci:.1f}%")
        if args.seconds_per_seed:
            hours = runs * args.seconds_per_seed / 3600.0
            line += f"  |  {hours:.1f} core-h per scan, {2 * hours:.1f} for both"
        print(line)

    best_by = {}
    for runs, name, step, n_seeds, ci in viable:
        best_by.setdefault(name, (runs, step, n_seeds, ci))
    if "mle" in best_by and "interp" in best_by:
        ratio = best_by["interp"][0] / best_by["mle"][0]
        print()
        print(f"Estimator efficiency: interpolation needs {ratio:.1f}x the "
              f"seed-runs of the logistic MLE for the same precision.")
        print("Fitting the whole curve is free; the extra seed-runs are not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
