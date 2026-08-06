#!/usr/bin/env python3
"""
generate_highsym_points.py

Generate Quantum ESPRESSO inputs at the exact high-symmetry stackings
AA (0, 0), AB (1/3, 1/3) and BA (2/3, 2/3), by shifting the mobile (top) layer
of an existing GSFE input.

Why this is needed
------------------
A 7x7 registry grid samples fractional shifts k/7, so it never lands on 1/3 or
2/3.  Every high-symmetry value therefore comes from extrapolating the Fourier
fit.  For the large first-shell amplitude that is harmless, but the AB/BA
splitting Delta_0 (~1.6 meV) and the corresponding interlayer-distance
difference (~0.005 A) are both smaller than the fit's own leave-one-out
prediction error, so they cannot be taken from the fit.  These three points
measure them directly.

The generator does text surgery on the template: everything outside the
ATOMIC_POSITIONS block is copied verbatim, so pseudopotentials, cutoffs,
k-points and vdW settings are guaranteed identical to the production grid.
The mobile layer is identified by a non-zero if_pos flag, not by atom order.

Usage
-----
    python generate_highsym_points.py tier1_relaxed_full/point_000.in \\
        --lattice-a 3.285 --outdir highsym
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

STACKINGS = {
    "AA": (0.0, 0.0),
    "AB": (1.0 / 3.0, 1.0 / 3.0),
    "BA": (2.0 / 3.0, 2.0 / 3.0),
}

_POS_HDR = re.compile(r"^\s*ATOMIC_POSITIONS\s*[({]?\s*(\w+)\s*[)}]?\s*$", re.I)
_ATOM = re.compile(
    r"^\s*([A-Z][a-z]?\d*)\s+"
    r"(-?\d+\.?\d*(?:[eEdD][-+]?\d+)?)\s+"
    r"(-?\d+\.?\d*(?:[eEdD][-+]?\d+)?)\s+"
    r"(-?\d+\.?\d*(?:[eEdD][-+]?\d+)?)"
    r"((?:\s+[01])*)\s*$"
)


def parse_template(path: str):
    """Split the file into (head, unit, atoms, tail)."""
    with open(path) as fh:
        lines = fh.read().splitlines()

    start = None
    for i, line in enumerate(lines):
        m = _POS_HDR.match(line)
        if m:
            start = i
            unit = m.group(1).lower()
            break
    if start is None:
        raise SystemExit(f"{path}: no ATOMIC_POSITIONS block found")

    atoms = []
    end = start + 1
    for i in range(start + 1, len(lines)):
        m = _ATOM.match(lines[i])
        if not m:
            break
        sym, x, y, z, flags = m.groups()
        atoms.append(
            {
                "sym": sym,
                "xyz": np.array([float(v.replace("D", "E")) for v in (x, y, z)]),
                "flags": flags.split(),
                "raw": lines[i],
            }
        )
        end = i + 1

    if not atoms:
        raise SystemExit(f"{path}: ATOMIC_POSITIONS block is empty")

    return lines[:start], unit, atoms, lines[end:]


def mobile_mask(atoms):
    """
    Mobile atoms are those with a non-zero if_pos flag.

    Falls back to a z-split if no flags are present, and says so, rather than
    silently guessing.
    """
    flagged = [a for a in atoms if a["flags"]]
    if flagged:
        return [bool(a["flags"]) and any(f != "0" for f in a["flags"]) for a in atoms]

    print(
        "  [warning] no if_pos flags in template; falling back to a z-split",
        file=sys.stderr,
    )
    zs = np.array([a["xyz"][2] for a in atoms])
    return list(zs > zs.mean())


def write_input(path, head, unit, atoms, tail, mask, shift_xy):
    out = list(head)
    out.append(f"ATOMIC_POSITIONS {unit}")
    for a, mob in zip(atoms, mask):
        xyz = a["xyz"] + (np.array([shift_xy[0], shift_xy[1], 0.0]) if mob else 0.0)
        flags = ("  " + " ".join(a["flags"])) if a["flags"] else ""
        out.append(
            f"  {a['sym']} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}{flags}"
        )
    out += tail
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("template", help="an existing point_*.in from the grid")
    ap.add_argument("--lattice-a", type=float, required=True,
                    help="in-plane lattice constant in Angstrom")
    ap.add_argument("--outdir", default="highsym")
    args = ap.parse_args(argv)

    head, unit, atoms, tail = parse_template(args.template)
    if unit != "angstrom":
        raise SystemExit(
            f"template uses {unit!r} positions; this script assumes angstrom"
        )

    mask = mobile_mask(atoms)
    a = args.lattice_a
    c1 = np.array([a, 0.0])
    c2 = np.array([a / 2.0, a * np.sqrt(3.0) / 2.0])

    print(f"template : {args.template}")
    print(f"atoms    : {len(atoms)}  ({sum(mask)} mobile, {len(mask) - sum(mask)} fixed)")
    for at, mob in zip(atoms, mask):
        print(f"    {'MOBILE' if mob else 'fixed '}  {at['sym']:3s} "
              f"{at['xyz'][0]:9.4f} {at['xyz'][1]:9.4f} {at['xyz'][2]:9.4f}")

    if sum(mask) == 0 or sum(mask) == len(mask):
        raise SystemExit(
            "Expected a fixed bottom layer and a mobile top layer; got all one "
            "or the other.  Check the template's if_pos flags."
        )

    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nwriting to {args.outdir}/")
    for name, (u, v) in STACKINGS.items():
        shift = u * c1 + v * c2
        path = os.path.join(args.outdir, f"{name}.in")
        write_input(path, head, unit, atoms, tail, mask, shift)
        print(f"  {name}.in   shift = ({u:.6f}, {v:.6f}) frac "
              f"= ({shift[0]:.6f}, {shift[1]:.6f}) A")

    print(
        "\nNote: AA duplicates grid point (0,0).  Keep it -- it is a free "
        "consistency check that this generator reproduces the production setup."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
