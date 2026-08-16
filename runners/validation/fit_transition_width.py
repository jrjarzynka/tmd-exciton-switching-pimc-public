#!/usr/bin/env python3
"""
fit_transition_width.py

Extract F_z,50 and the transition width `s` from a scan you already have.

`s` is the one input plan_symmetry_test.py cannot guess: it is a property of
the system, not of the design. Reading it off a plot by eye (the field interval
over which the fraction runs 0.1 to 0.9, divided by 4.4) works, but a maximum-
likelihood fit to the binomial counts uses every point and needs no judgement
about where the curve "really" crosses.

Accepts three CSV layouts:
  * run_dissociation_time_convergence.py  (rho2_per_seed, grouped by n_steps)
  * adaptive_dense_field_scan.py          (adaptive_per_seed.csv)
  * run_dissociation_hysteresis.py        (rho2_per_seed + branch column)

For the hysteresis layout each BRANCH is a separate fraction-vs-field curve and
is fitted separately. The `bound` branch is the forward transition and is
normally the one to plan against; `separated` is the reverse branch and will fit
a different F50 whenever the two branches have not converged onto each other --
that gap is the physics the hysteresis run exists to measure, not an error.

Usage
-----
    python fit_transition_width.py results/scan.csv
    python fit_transition_width.py results/adaptive_per_seed.csv --threshold 100
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from tmd_pimc.threshold_estimators import logistic_mle, interp_threshold


def load(path: str, threshold: float, shift_nm=None):
    """-> {group: {field: [rho2 per seed]}} for either layout."""
    groups = defaultdict(lambda: defaultdict(list))
    with open(path) as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        if "rho2_per_seed" in cols and "branch" in cols:
            layout, gname = "hysteresis", "branch"
            for r in reader:
                if not r.get("rho2_per_seed"):
                    continue
                g = r["branch"]
                f = float(r["Fz_eV_per_nm"])
                groups[g][f].extend(float(v) for v in r["rho2_per_seed"].split(";"))
        elif "rho2_per_seed" in cols:
            layout, gname = "time_convergence", "n_steps"
            for r in reader:
                if not r.get("rho2_per_seed"):
                    continue
                g = int(r["n_steps"])
                f = float(r["Fz_eV_per_nm"])
                groups[g][f].extend(float(v) for v in r["rho2_per_seed"].split(";"))
        elif {"rho2_nm2", "field_eV_per_nm"} <= cols:
            layout, gname = "adaptive", "shift_nm"
            for r in reader:
                if not r.get("rho2_nm2"):
                    continue
                sh = float(r.get("shift_magnitude_nm", 0.0))
                if shift_nm is not None and abs(sh - shift_nm) > 1e-9:
                    continue
                groups[sh][float(r["field_eV_per_nm"])].append(float(r["rho2_nm2"]))
        else:
            raise SystemExit(f"{path}: unrecognised CSV layout")
    return groups, layout, gname


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--threshold", type=float, default=30.0,
                    help="rho^2 above which a seed counts as dissociated, nm^2")
    ap.add_argument("--shift-nm", type=float, default=None)
    ap.add_argument("--branch", default=None,
                    help="hysteresis layout: fit only this branch "
                         "(e.g. 'bound'). Default: fit each branch separately.")
    args = ap.parse_args()

    groups, layout, gname = load(args.csv, args.threshold, args.shift_nm)
    if not groups:
        raise SystemExit("no usable rows found")

    print(f"{args.csv}  [{layout}]   threshold {args.threshold} nm^2\n")

    keys = sorted(groups)
    if args.branch is not None:
        keys = [k for k in keys if str(k) == args.branch]
        if not keys:
            raise SystemExit(f"branch {args.branch!r} not found; "
                             f"available: {sorted(groups)}")

    for g in keys:
        by_field = groups[g]
        fields = np.array(sorted(by_field))
        counts = np.array([sum(v > args.threshold for v in by_field[f]) for f in fields],
                          dtype=float)
        nseeds = np.array([len(by_field[f]) for f in fields], dtype=float)
        fracs = counts / nseeds

        print(f"{gname} = {g}   ({len(fields)} field points, "
              f"{int(nseeds.min())}-{int(nseeds.max())} seeds each)")
        print("   Fz      frac   seeds")
        for f, p, n in zip(fields, fracs, nseeds):
            print(f"  {f:6.3f}  {p:6.2f}  {int(n):5d}")

        fit = logistic_mle(fields, counts, nseeds)
        f50_i = interp_threshold(fields, fracs)

        if fit is None:
            print("\n  logistic fit FAILED -- usually the scan does not span the")
            print("  transition. Widen the field range; s cannot be read off")
            print("  a curve that is flat across the whole window.\n")
            continue

        f50, s = fit
        print(f"\n  F_z,50 (MLE)        = {f50:.4f} eV/nm")
        if f50_i is not None:
            print(f"  F_z,50 (interp)     = {f50_i:.4f} eV/nm   [control]")
        print(f"  width s             = {s:.4f} eV/nm")
        print(f"  10-90 width (4.4 s) = {4.4 * s:.4f} eV/nm")

        span = float(fields.max() - fields.min())
        if 4.4 * s > 0.8 * span:
            print("\n  [warning] the fitted transition is nearly as wide as the")
            print("  scanned window, so s is poorly constrained -- the fit cannot")
            print("  see where the curve flattens. Widen the field range before")
            print("  using this value to plan.")
        elif 4.4 * s < 2.0 * float(np.min(np.diff(fields))) if fields.size > 1 else False:
            print("\n  [warning] the transition is narrower than about two grid")
            print("  steps, so it is barely resolved. A denser field grid would")
            print("  pin s down considerably better.")

        print(f"\n  -> plan_symmetry_test.py --f50 {f50:.3f} --width-s {s:.3f} "
              f"--field-min {fields.min():.2f} --field-max {fields.max():.2f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
