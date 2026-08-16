#!/usr/bin/env python3
"""
compare_landscape_symmetry.py

Does F_z,50 depend on the electron-hole contrast of the moire landscape?

The question
------------
Until v1.1 the two-body model was built from one registry amplitude and one
dipole length, so V_e and V_h differed only by a registry offset and by the sign
of the Stark term. Under that symmetry the relative-coordinate problem may lose
its dependence on V0 by construction, in which case the reported independence of
F_z,50 from V0 is a property of the model rather than of the material.

This script compares two dissociation scans that differ ONLY in that symmetry --
one with V0_e = V0_h, one with them decoupled -- and reports whether the
threshold moved by more than the sampling noise.

Why bootstrap rather than error propagation
-------------------------------------------
F_z,50 is obtained by interpolating a step-like curve of binomial fractions, so
its uncertainty is neither Gaussian nor analytic. Resampling the per-seed rho^2
values (which is why run_dissociation_time_convergence.py records them
individually) and re-deriving the whole curve each time propagates the
uncertainty through the interpolation without assuming anything about its shape.

Verdict is three-valued, matching classify_time_dependence(): reporting
"unchanged" when the data merely lack the power to see a shift would convert a
sampling limitation into a physical claim.

Usage
-----
    python compare_landscape_symmetry.py \\
        --symmetric  results/scan_sym.csv \\
        --asymmetric results/scan_asym.csv \\
        --n-steps 240000
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Optional, Sequence

import numpy as np


def _interp_threshold(fields: Sequence[float], fractions: Sequence[float],
                      level: float = 0.5) -> Optional[float]:
    """Field at which the fraction first crosses `level`, by linear interpolation.

    Returns None when the scan does not bracket the crossing. Identical
    convention to interpolate_threshold_field() in
    run_dissociation_time_convergence.py -- do not let the two drift apart.
    """
    f = np.asarray(fields, dtype=float)
    p = np.asarray(fractions, dtype=float)
    order = np.argsort(f)
    f, p = f[order], p[order]
    if p.min() > level or p.max() < level:
        return None
    for i in range(len(f) - 1):
        if (p[i] - level) * (p[i + 1] - level) <= 0:
            if p[i + 1] == p[i]:
                return float(f[i])
            t = (level - p[i]) / (p[i + 1] - p[i])
            return float(f[i] + t * (f[i + 1] - f[i]))
    return None


def _provenance_from_row(r: dict, into: dict) -> None:
    for key in ("moire_amplitude_e_eV", "moire_amplitude_h_eV",
                "dipole_length_e_nm", "dipole_length_h_nm",
                "landscape_symmetric", "amplitude_ratio_h_over_e"):
        if key in r and r[key] not in ("", None):
            into[key] = r[key]


def load_scan(path: str, shift_nm: Optional[float] = None) -> dict:
    """Read either supported CSV layout, auto-detected from its columns.

    Two runners produce dissociation data in different shapes:

      run_dissociation_time_convergence.py -- one row per (n_steps, Fz), with
      the per-seed rho^2 values packed into a semicolon-separated
      `rho2_per_seed` field. Groups are sampling lengths.

      adaptive_dense_field_scan.py -- one row per (shift, field, seed), with
      `rho2_nm2` already granular. Groups are registry offsets, since that
      runner sweeps the field within a single run length.

    Both reduce to the same structure: a mapping from group label to a list of
    field points, each carrying its own list of per-seed rho^2 values. Rows
    without a rho^2 value are skipped -- the adaptive runner writes a row with
    an `error` column when a worker fails, and counting those as bound seeds
    would silently bias the fraction downward.
    """
    groups: dict = {}
    provenance: dict = {}
    skipped = 0

    with open(path) as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])

        if "rho2_per_seed" in cols:
            layout = "time_convergence"
            group_name = "n_steps"
            for r in reader:
                seeds = ([float(v) for v in r["rho2_per_seed"].split(";")]
                         if r.get("rho2_per_seed") else [])
                groups.setdefault(int(r["n_steps"]), []).append({
                    "Fz": float(r["Fz_eV_per_nm"]),
                    "seeds": seeds,
                    "cutoff_ok": r.get("cutoff_ok", "1") == "1",
                })
                _provenance_from_row(r, provenance)

        elif {"rho2_nm2", "field_eV_per_nm", "seed"} <= cols:
            layout = "adaptive"
            group_name = "shift_nm"
            bucket: dict = {}
            for r in reader:
                if not r.get("rho2_nm2"):
                    skipped += 1
                    continue
                shift = float(r.get("shift_magnitude_nm", 0.0))
                if shift_nm is not None and abs(shift - shift_nm) > 1e-9:
                    continue
                key = (shift, float(r["field_eV_per_nm"]))
                bucket.setdefault(key, []).append(float(r["rho2_nm2"]))
                _provenance_from_row(r, provenance)
            for (shift, Fz), vals in bucket.items():
                groups.setdefault(shift, []).append({
                    "Fz": Fz, "seeds": vals, "cutoff_ok": True,
                })

        else:
            raise ValueError(
                f"{path}: unrecognised layout. Expected either a "
                f"`rho2_per_seed` column (time-convergence runner) or "
                f"`rho2_nm2`/`field_eV_per_nm`/`seed` columns (adaptive runner)."
            )

    return {"by_n": groups, "provenance": provenance, "path": path,
            "layout": layout, "group_name": group_name, "skipped": skipped}


def bootstrap_threshold(points: Sequence[dict], threshold_nm2: float,
                        n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Distribution of F_z,50 under resampling of the seeds at every field.

    Seeds are resampled independently at each field point, which is how they
    were generated: distinct RNG streams, no shared state between fields.
    """
    pts = sorted(points, key=lambda p: p["Fz"])
    fields = [p["Fz"] for p in pts]
    seed_arrays = [np.asarray(p["seeds"], dtype=float) for p in pts]

    if any(a.size == 0 for a in seed_arrays):
        raise ValueError(
            "per-seed rho^2 values missing from at least one field point; "
            "bootstrap needs the rho2_per_seed column"
        )

    out = []
    for _ in range(n_boot):
        fracs = []
        for a in seed_arrays:
            draw = rng.choice(a, size=a.size, replace=True)
            fracs.append(float(np.mean(draw > threshold_nm2)))
        t = _interp_threshold(fields, fracs)
        if t is not None:
            out.append(t)
    return np.asarray(out, dtype=float)


def describe(scan: dict, label: str) -> None:
    p = scan["provenance"]
    print(f"  {label:11s} : {scan['path']}  [{scan['layout']}]")
    if scan.get("skipped"):
        print(f"                {scan['skipped']} row(s) without a rho^2 value "
              f"skipped (failed workers)")
    if not p:
        print("                (no landscape provenance -- pre-v1.1 file; "
              "cannot confirm which model produced it)")
        return
    amp_e = p.get("moire_amplitude_e_eV", "?")
    amp_h = p.get("moire_amplitude_h_eV", "?")
    sym = p.get("landscape_symmetric", "?")
    print(f"                V0_e={amp_e} eV  V0_h={amp_h} eV  "
          f"d_e={p.get('dipole_length_e_nm','?')} d_h={p.get('dipole_length_h_nm','?')}  "
          f"symmetric={sym}")


def _is_symmetric(scan: dict) -> Optional[bool]:
    v = scan["provenance"].get("landscape_symmetric")
    if v is None:
        return None
    return str(v).strip().lower() in ("1", "true", "yes")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symmetric", required=True,
                    help="CSV from the scan with V0_e = V0_h")
    ap.add_argument("--asymmetric", required=True,
                    help="CSV from the scan with V0_e != V0_h")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="time-convergence layout: which sampling length to "
                         "compare; default: the longest present in both files")
    ap.add_argument("--shift-nm", type=float, default=None,
                    help="adaptive layout: which registry offset to compare; "
                         "default: the only one present, or 0.0")
    ap.add_argument("--threshold", type=float, default=30.0,
                    help="rho^2 above which a seed counts as dissociated, nm^2")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args(argv)

    sym = load_scan(args.symmetric, shift_nm=args.shift_nm)
    asym = load_scan(args.asymmetric, shift_nm=args.shift_nm)

    if sym["layout"] != asym["layout"]:
        print(f"!! the two files use different layouts "
              f"({sym['layout']} vs {asym['layout']}). Compare like with like: "
              f"the two runners differ in run length and sampling protocol, so "
              f"a threshold from one is not commensurable with the other.")
        return 1

    print("Scans")
    print("-" * 72)
    describe(sym, "symmetric")
    describe(asym, "asymmetric")

    # Guard: the two files must actually differ in the way the test assumes.
    s_flag, a_flag = _is_symmetric(sym), _is_symmetric(asym)
    if s_flag is False:
        print("\n!! the file passed as --symmetric reports landscape_symmetric=False")
    if a_flag is True:
        print("\n!! the file passed as --asymmetric reports landscape_symmetric=True")
    if s_flag is not None and a_flag is not None and s_flag == a_flag:
        print("!! both scans report the SAME landscape mode; this comparison "
              "cannot answer the question it is for.")
        return 1

    group_name = sym["group_name"]
    common = sorted(set(sym["by_n"]) & set(asym["by_n"]))
    if not common:
        print(f"\nNo {group_name} value is present in both files.")
        return 1

    requested = args.n_steps if sym["layout"] == "time_convergence" else args.shift_nm
    if requested is None:
        n_steps = common[-1] if sym["layout"] == "time_convergence" else common[0]
        if len(common) > 1:
            print(f"\n  ({len(common)} {group_name} values in common; using "
                  f"{n_steps}. Select another with "
                  f"{'--n-steps' if sym['layout'] == 'time_convergence' else '--shift-nm'}.)")
    else:
        matches = [g for g in common if abs(float(g) - float(requested)) < 1e-9]
        if not matches:
            print(f"\n{group_name}={requested} is not present in both files "
                  f"(available: {common})")
            return 1
        n_steps = matches[0]

    for scan, label in ((sym, "symmetric"), (asym, "asymmetric")):
        bad = [p for p in scan["by_n"][n_steps] if not p["cutoff_ok"]]
        if bad:
            print(f"\n!! {label}: {len(bad)} field point(s) exceeded the "
                  f"interaction table cutoff; those separations are unphysical.")

    rng = np.random.default_rng(args.rng_seed)
    try:
        b_sym = bootstrap_threshold(sym["by_n"][n_steps], args.threshold,
                                    args.n_boot, rng)
        b_asym = bootstrap_threshold(asym["by_n"][n_steps], args.threshold,
                                     args.n_boot, rng)
    except ValueError as exc:
        print(f"\n{exc}")
        return 1

    n_ok = min(b_sym.size, b_asym.size)
    if n_ok < 0.5 * args.n_boot:
        print(f"\n!! only {b_sym.size}/{args.n_boot} (symmetric) and "
              f"{b_asym.size}/{args.n_boot} (asymmetric) bootstrap replicas "
              f"bracketed the 50% crossing. The scan barely spans the "
              f"transition; widen the field range before reading anything below.")
    if b_sym.size == 0 or b_asym.size == 0:
        print("\nOne of the scans never brackets the 50% crossing. No threshold "
              "to compare.")
        return 1

    print()
    print("=" * 72)
    print(f"F_z,50 AT {group_name} = {n_steps}   (threshold {args.threshold} nm^2, "
          f"{args.n_boot} bootstrap replicas)")
    print("=" * 72)

    for b, label in ((b_sym, "symmetric"), (b_asym, "asymmetric")):
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"  {label:11s} : {np.median(b):8.4f} eV/nm   "
              f"95% CI [{lo:.4f}, {hi:.4f}]")

    # Paired difference: resample both distributions against each other rather
    # than differencing the point estimates, so the CI on the shift carries both
    # scans' uncertainty.
    k = min(b_sym.size, b_asym.size)
    diff = rng.choice(b_asym, size=k, replace=True) - rng.choice(b_sym, size=k, replace=True)
    d_med = float(np.median(diff))
    d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
    rel = 100.0 * d_med / float(np.median(b_sym))

    print()
    print(f"  shift (asym - sym) : {d_med:+8.4f} eV/nm  ({rel:+.1f}%)   "
          f"95% CI [{d_lo:+.4f}, {d_hi:+.4f}]")

    print()
    print("-" * 72)
    excludes_zero = (d_lo > 0.0) or (d_hi < 0.0)
    ci_width_rel = 100.0 * (d_hi - d_lo) / float(np.median(b_sym))

    if excludes_zero:
        print("VERDICT: F_z,50 DEPENDS on the electron-hole landscape contrast.")
        print()
        print("  The threshold's independence of V0 is therefore not established")
        print("  as physical: under the symmetric model it may follow from the")
        print("  residual electron-hole symmetry. This is not a negative result --")
        print("  a threshold set by the CONTRAST between V_e and V_h connects")
        print("  directly to the Stage 0 finding that the hole landscape depends")
        print("  critically on spin-orbit coupling. Reframe rather than discard.")
    elif ci_width_rel < 10.0:
        print("VERDICT: F_z,50 is UNCHANGED within a tight bound.")
        print()
        print(f"  The 95% CI on the shift spans {ci_width_rel:.1f}% of the")
        print("  threshold and contains zero, so the independence of V0 survives")
        print("  removal of the electron-hole symmetry. The scaling law can be")
        print("  quoted as physical, with this test cited as its control.")
    else:
        print("VERDICT: INCONCLUSIVE.")
        print()
        print(f"  The CI on the shift contains zero but spans {ci_width_rel:.1f}%")
        print("  of the threshold, which is too wide to call the two scans equal.")
        print("  More seeds per field point, or a denser field grid through the")
        print("  transition, are needed before either reading is defensible.")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
