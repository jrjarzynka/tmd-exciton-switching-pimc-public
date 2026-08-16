#!/usr/bin/env python3
"""
analyse_dissociation_scan.py

Post-process the CSV written by run_dissociation_time_convergence.py.

Reports, for each sampling length:
  * the transition curve and F_z,50 by interpolation
  * W_10-90, the field range over which the fraction rises from 10% to 90%
  * dF_20-40, the sensitivity of F_z,50 to the classification threshold
  * the per-seed distribution, and whether the transition is a genuine
    two-state switch or a lottery over initial conditions
  * the contact density across the transition, as a relative recombination rate

What the per-seed reading is for
--------------------------------
A sharp 0 -> 1 transition has two incompatible explanations that the aggregate
fraction cannot separate:

  SWITCHING   at a given field every seed ends up in the same state, and the
              per-seed values cluster tightly. The field genuinely drives the
              transition.

  LOTTERY     seeds split into two clusters, some bound and some separated,
              with nothing in between. Each trajectory stayed where it started
              and the "fraction" counts initial conditions, not physics.

  IN TRANSIT  seeds spread continuously between the two states, meaning they
              were caught part-way through a slow escape. The fraction then
              depends on run length and F_z,50 will drift.

The bimodality coefficient below separates these. It is Sarle's statistic,
(skew^2 + 1) / kurtosis, which exceeds 5/9 for a two-cluster distribution and
falls below it for a unimodal one -- a threshold, not a fit, so it needs no
tuning and behaves sensibly on the small seed counts these scans use.

Note on what is NOT available here
----------------------------------
The CSV stores one <rho^2> per seed, so the variance BETWEEN seeds is
recoverable but the variance WITHIN a seed is not. A tight per-seed spread is
therefore consistent both with every trajectory being genuinely locked and with
every trajectory rattling between states fast enough to average out. Settling
that needs per-seed histograms, which the runner does not currently write.

Usage
-----
    python analyse_dissociation_scan.py results/converge_s0.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from typing import Optional, Sequence

import numpy as np


def _interp_threshold(fields, fractions, level=0.5) -> Optional[float]:
    f = np.asarray(fields, float)
    p = np.asarray(fractions, float)
    o = np.argsort(f)
    f, p = f[o], p[o]
    if p.min() > level or p.max() < level:
        return None
    for i in range(len(f) - 1):
        if (p[i] - level) * (p[i + 1] - level) <= 0:
            if p[i + 1] == p[i]:
                return float(f[i])
            t = (level - p[i]) / (p[i + 1] - p[i])
            return float(f[i] + t * (f[i + 1] - f[i]))
    return None


def transition_width(fields, fractions) -> Optional[float]:
    """W_10-90 as a percentage of F_z,50, or None if the scan does not bracket it."""
    lo = _interp_threshold(fields, fractions, 0.10)
    hi = _interp_threshold(fields, fractions, 0.90)
    mid = _interp_threshold(fields, fractions, 0.50)
    if lo is None or hi is None or mid is None or mid == 0:
        return None
    return float((hi - lo) / mid * 100.0)


def threshold_sensitivity(fields, per_seed, lo=20.0, hi=40.0) -> Optional[float]:
    """Shift in F_z,50 when the classification threshold moves from lo to hi nm^2.

    Recomputed from the per-seed values rather than from the recorded fraction,
    so it costs nothing and needs no extra runs. Reported as a percentage of
    F_z,50 at the baseline threshold.
    """
    def frac_at(th):
        return [float(np.mean([v > th for v in seeds])) for seeds in per_seed]

    f_lo = _interp_threshold(fields, frac_at(lo))
    f_hi = _interp_threshold(fields, frac_at(hi))
    f_mid = _interp_threshold(fields, frac_at(30.0))
    if None in (f_lo, f_hi, f_mid) or f_mid == 0:
        return None
    return float(abs(f_hi - f_lo) / f_mid * 100.0)


def bimodality_coefficient(x: Sequence[float]) -> float:
    """Sarle's bimodality coefficient, (skew^2 + 1) / kurtosis.

    Above 5/9 = 0.555 indicates two clusters; below indicates unimodal. Uses the
    sample-size corrected form, which matters at the 16-40 seeds these scans use.
    """
    a = np.asarray(x, float)
    n = a.size
    if n < 4:
        return float("nan")
    m = a.mean()
    s = a.std(ddof=1)
    if s == 0:
        return float("nan")
    g1 = np.mean(((a - m) / s) ** 3) * math.sqrt(n * (n - 1)) / (n - 2)
    g2 = (np.mean(((a - m) / s) ** 4) - 3.0)
    g2 = ((n - 1) * ((n + 1) * g2 + 6.0)) / ((n - 2) * (n - 3))
    denom = g2 + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    if denom == 0:
        return float("nan")
    return float((g1 ** 2 + 1.0) / denom)


def classify_seed_distribution(seeds: Sequence[float], threshold: float = 30.0) -> str:
    """Describe the per-seed distribution at one field point."""
    a = np.asarray(seeds, float)
    if a.size < 4:
        return "too few seeds"
    frac = float(np.mean(a > threshold))
    if frac == 0.0 or frac == 1.0:
        rel = float(a.std(ddof=1) / a.mean()) if a.mean() else float("inf")
        return "uniform (tight)" if rel < 0.25 else "uniform (broad)"

    bc = bimodality_coefficient(a)
    bound = a[a <= threshold]
    free = a[a > threshold]
    gap = (free.min() - bound.max()) / np.ptp(a) if np.ptp(a) > 0 else 0.0

    if bc > 5.0 / 9.0 and gap > 0.3:
        return "two clusters (lottery or true switching)"
    return "continuous (in transit)"


def load(path: str):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            r["n_steps"] = int(r["n_steps"])
            r["Fz"] = float(r["Fz_eV_per_nm"])
            r["frac"] = float(r["dissociated_fraction"])
            r["rho2_median"] = float(r["rho2_median"])
            r["psi0_sq"] = float(r["psi0_sq"]) if r.get("psi0_sq") else None
            r["seeds"] = ([float(v) for v in r["rho2_per_seed"].split(";")]
                          if r.get("rho2_per_seed") else [])
            r["acc"] = float(r["acceptance_global"])
            r["cutoff_ok"] = r["cutoff_ok"] == "1"
            rows.append(r)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--threshold", type=float, default=30.0)
    args = ap.parse_args(argv)

    rows = load(args.csv)
    if not rows:
        print("empty CSV")
        return 1

    if not all(r["cutoff_ok"] for r in rows):
        bad = [r for r in rows if not r["cutoff_ok"]]
        print(f"!! {len(bad)} point(s) exceeded the interaction table cutoff; "
              f"those separations are unphysical and the results below are void "
              f"for them.\n")

    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_steps"], []).append(r)

    thresholds = {}
    for n in sorted(by_n):
        pts = sorted(by_n[n], key=lambda r: r["Fz"])
        fields = [p["Fz"] for p in pts]
        fracs = [p["frac"] for p in pts]

        f50 = _interp_threshold(fields, fracs)
        thresholds[n] = f50
        w = transition_width(fields, fracs)
        per_seed = [p["seeds"] for p in pts]
        dsens = (threshold_sensitivity(fields, per_seed)
                 if all(len(s) > 0 for s in per_seed) else None)

        print("=" * 78)
        print(f"n_steps = {n}")
        print("=" * 78)
        print(f"{'Fz':>7}{'frac':>7}{'<rho2> med':>12}{'seed range':>22}"
              f"{'|psi(0)|^2':>12}  distribution")
        for p in pts:
            s = p["seeds"]
            rng = f"[{min(s):7.1f},{max(s):8.1f}]" if s else "n/a"
            cls = classify_seed_distribution(s, args.threshold) if s else "n/a"
            psi = f"{p['psi0_sq']:12.5f}" if p["psi0_sq"] is not None else " " * 12
            print(f"{p['Fz']:7.3f}{p['frac']:7.2f}{p['rho2_median']:12.2f}{rng:>22}"
                  f"{psi}  {cls}")
        print()
        print(f"  F_z,50    = {'not bracketed' if f50 is None else f'{f50:.4f} eV/nm'}")
        print(f"  W_10-90   = {'n/a' if w is None else f'{w:.1f}% of F_z,50'}")
        print(f"  dF_20-40  = {'n/a' if dsens is None else f'{dsens:.1f}% of F_z,50'}")
        acc = [p["acc"] for p in pts]
        # NOTE: the warning is built before the f-string rather than inside it.
        # A conditional expression spanning several lines within f-string braces
        # is only valid from Python 3.12 (PEP 701); this project declares
        # requires-python = ">=3.11", so the earlier form raised SyntaxError for
        # anyone on 3.11 -- including a reviewer following the README.
        acc_warning = ("   [warning] below 1%: basin-to-basin mixing is suppressed"
                       if min(acc) < 0.01 else "")
        print(f"  global acceptance {min(acc) * 100:.2f}-{max(acc) * 100:.2f}%"
              f"{acc_warning}")
        print()

    vals = [(n, t) for n, t in sorted(thresholds.items()) if t is not None]
    if len(vals) >= 2:
        print("=" * 78)
        print("CONVERGENCE IN SAMPLING LENGTH")
        print("=" * 78)
        for n, t in vals:
            print(f"  {n:>8} steps : F_z,50 = {t:.4f} eV/nm")
        drift = (vals[-1][1] - vals[0][1]) / vals[0][1] * 100.0
        print(f"\n  drift across the sweep: {drift:+.1f}%")
        if abs(drift) > 5.0:
            print("  NOT converged. F_z,50 is a threshold at a stated observation "
                  "time, not a material constant, and must be quoted that way -- "
                  "as a coercive field or blocking temperature is quoted with its "
                  "measurement rate.")
        else:
            print("  Stable across the sweep.")

    # recombination contrast, the optical observable
    psis = [(r["Fz"], r["psi0_sq"]) for r in rows
            if r["psi0_sq"] is not None and r["n_steps"] == max(by_n)]
    if len(psis) >= 2:
        psis.sort()
        lo, hi = psis[0][1], psis[-1][1]
        if hi > 0:
            print(f"\n  |psi(0)|^2 falls {lo / hi:.0f}x across the scan "
                  f"({lo:.5f} -> {hi:.5f} nm^-2): the relative radiative rate, "
                  f"and the optical signature of the transition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
