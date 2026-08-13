#!/usr/bin/env python3
"""
stage0_collect.py

Parse the Stage 0 runs and decide whether the existing 49-point scalar landscape
can still be used for Paper 2.

What is compared, and why it is a difference of differences
-----------------------------------------------------------
An absolute band edge can shift by hundreds of meV between pseudopotential
families without any consequence for the moire landscape, because the landscape
is built from how the edges vary WITH REGISTRY. The quantity that matters is
therefore

    Delta_VBM = E_VBM(AB) - E_VBM(BA)        (and likewise for the CBM)

and the question is whether SOC or the dipole correction changes THAT:

    delta_SOC   = Delta(C) - Delta(B)
    delta_dip   = Delta(D) - Delta(C)

A 400 meV common shift is irrelevant; a 20 meV change in Delta is not.

Three quantities need no alignment at all
-----------------------------------------
The gap CBM - VBM at fixed k in a single calculation is reference-free: any
rigid shift of the eigenvalues cancels. So is the layer character. Both are
reported first, because if they are stable the remaining concerns are narrower.

Layer character is a veto
-------------------------
The two-body model represents the electron by the MoSe2 conduction edge and the
hole by the WSe2 valence edge. If SOC moves the VBM onto the wrong layer at any
registry, or splits it so that neither layer dominates, then no correction to
V_h rescues the picture -- the hole is no longer a single-layer state and the
model needs a local band Hamiltonian instead. That is a RED verdict regardless
of how small the energy shifts are.

Usage
-----
    python stage0_collect.py stage0/
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from typing import Optional

import numpy as np

# atoms 1-3 are the lower MoSe2 layer, 4-6 the upper WSe2 layer
LOWER = {1, 2, 3}
UPPER = {4, 5, 6}

NUM = re.compile(r"-?\d+\.\d+")
GREEN, YELLOW = 5.0, 15.0            # meV thresholds on delta


def parse_scf(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fh:
        text = fh.read()
    if "JOB DONE" not in text:
        return None

    seg = text[text.rfind("End of self-consistent calculation"):]
    fermi = re.findall(r"the Fermi energy is\s+([-\d.]+) ev", seg)
    if not fermi:
        return None
    ef = float(fermi[-1])

    blocks = re.findall(
        r"k =([ \d.\-]+)\(\s*\d+ PWs\)\s+bands \(ev\):\s*\n\n((?:\s+[-\d.]+.*\n)+)", seg)
    bands = {tuple(round(float(x), 4) for x in NUM.findall(k)):
             np.array([float(x) for x in NUM.findall(b)]) for k, b in blocks}

    K = next((k for k in bands
              if abs(abs(k[0]) - 0.3333) < 2e-3 and abs(abs(k[1]) - 0.5774) < 2e-3), None)
    if K is None:
        return None

    e = bands[K]
    occ, emp = e[e < ef], e[e > ef]
    if not len(occ) or not len(emp):
        return None
    vbm, cbm = float(occ.max()), float(emp.min())

    # index of the band edges, needed to pull their layer projection
    i_vbm = int(np.argmin(np.abs(e - vbm))) + 1
    i_cbm = int(np.argmin(np.abs(e - cbm))) + 1

    # semicore alignment: the deepest states split into two groups, one per
    # layer. Their MEAN tracks the common reference drift, their DIFFERENCE is
    # the physical interlayer potential step. Costs nothing and needs no .save.
    deep = np.sort(e)[:16]
    semicore = float(deep.mean())

    return dict(path=path, fermi=ef, vbm=vbm, cbm=cbm, gap=cbm - vbm,
                i_vbm=i_vbm, i_cbm=i_cbm, semicore=semicore, k=K, n_bands=len(e))


def parse_projwfc(path: str) -> Optional[dict]:
    """Return {band_index: (P_lower, P_upper)} at the K point."""
    if not os.path.exists(path):
        return None
    with open(path, errors="replace") as fh:
        text = fh.read()

    state_atom = {}
    for m in re.finditer(r"state #\s*(\d+):\s*atom\s+(\d+)", text):
        state_atom[int(m.group(1))] = int(m.group(2))
    if not state_atom:
        return None

    # find the K-point section, then the per-band decompositions inside it
    kblocks = re.split(r"\n\s*k =\s*", text)
    target = None
    for blk in kblocks[1:]:
        head = blk.split("\n", 1)[0]
        v = NUM.findall(head)
        if len(v) >= 2 and abs(abs(float(v[0])) - 0.3333) < 2e-3 \
                and abs(abs(float(v[1])) - 0.5774) < 2e-3:
            target = blk
            break
    if target is None:
        return None

    out = {}
    for m in re.finditer(r"====\s*e\(\s*(\d+)\)\s*=\s*[-\d.]+\s*eV\s*====\s*\n(.*?)(?=\n\s*={4}|\Z)",
                         target, re.S):
        band = int(m.group(1))
        lower = upper = 0.0
        for w, st in re.findall(r"([\d.]+)\*\[#\s*(\d+)\]", m.group(2)):
            a = state_atom.get(int(st))
            if a in LOWER:
                lower += float(w)
            elif a in UPPER:
                upper += float(w)
        tot = lower + upper
        if tot > 0:
            out[band] = (lower / tot, upper / tot)
    return out or None


def classify(delta_meV: float) -> str:
    a = abs(delta_meV)
    return "GREEN" if a < GREEN else ("YELLOW" if a < YELLOW else "RED")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default="stage0")
    args = ap.parse_args(argv)

    runs, proj = {}, {}
    for s in ("AA", "AB", "BA"):
        for v in ("B", "C", "D"):
            n = f"{s}_{v}"
            r = parse_scf(os.path.join(args.dir, n + ".out"))
            if r:
                runs[n] = r
                proj[n] = parse_projwfc(os.path.join(args.dir, n + ".projwfc.out"))

    if not runs:
        print(f"No completed runs found in {args.dir}/")
        return 1

    print("=" * 84)
    print("BAND EDGES AT K")
    print("=" * 84)
    print(f"{'run':>8}{'E_F':>10}{'VBM':>10}{'CBM':>10}{'gap':>9}"
          f"{'P_W(VBM)':>11}{'P_Mo(CBM)':>11}  layer")
    veto = []
    for n in sorted(runs):
        r = runs[n]
        p = proj.get(n)
        pv = p.get(r["i_vbm"]) if p else None
        pc = p.get(r["i_cbm"]) if p else None
        pw = pv[1] if pv else float("nan")     # upper layer = WSe2
        pm = pc[0] if pc else float("nan")     # lower layer = MoSe2
        ok = "PASS" if (pw > 0.7 and pm > 0.7) else (
            "n/a" if math.isnan(pw) or math.isnan(pm) else "** FAIL **")
        if ok.startswith("**"):
            veto.append(n)
        print(f"{n:>8}{r['fermi']:10.4f}{r['vbm']:10.4f}{r['cbm']:10.4f}"
              f"{r['gap']:9.4f}{pw:11.3f}{pm:11.3f}  {ok}")

    print()
    print("=" * 84)
    print("REGISTRY SPLITTING  Delta = AB - BA   (reference-free for the gap)")
    print("=" * 84)
    have = [v for v in ("B", "C", "D") if f"AB_{v}" in runs and f"BA_{v}" in runs]
    deltas = {}
    print(f"{'variant':>9}{'d_VBM':>12}{'d_CBM':>12}{'d_gap':>12}   (meV)")
    for v in have:
        a, b = runs[f"AB_{v}"], runs[f"BA_{v}"]
        # remove the common reference drift using the semicore mean
        sh = a["semicore"] - b["semicore"]
        d = dict(vbm=(a["vbm"] - b["vbm"] - sh) * 1000,
                 cbm=(a["cbm"] - b["cbm"] - sh) * 1000,
                 gap=(a["gap"] - b["gap"]) * 1000)
        deltas[v] = d
        print(f"{v:>9}{d['vbm']:12.2f}{d['cbm']:12.2f}{d['gap']:12.2f}")

    print()
    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    worst = "GREEN"
    order = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    for lab, x, y in (("SOC        ", "B", "C"), ("dipole corr", "C", "D")):
        if x in deltas and y in deltas:
            for q, nm in (("vbm", "V_h"), ("cbm", "V_e")):
                dd = deltas[y][q] - deltas[x][q]
                c = classify(dd)
                worst = max(worst, c, key=lambda s: order[s])
                print(f"  {lab} effect on {nm} : {dd:+8.2f} meV   {c}")

    if veto:
        worst = "RED"
        print(f"\n  layer character FAILED for: {', '.join(veto)}")
        print("  The band edge is no longer dominated by the expected layer, so the "
              "hole cannot be represented by a single WSe2 landscape (or the "
              "electron by a single MoSe2 one). This overrides the energy "
              "thresholds: a local band Hamiltonian is needed, not a corrected V_h.")

    # convergence probes, each against the AB_C reference
    ref = runs.get("AB_C")
    probes = [(os.path.basename(p)[:-4], parse_scf(p))
              for p in sorted(glob.glob(os.path.join(args.dir, "conv_*.out")))]
    probes = [(n, r) for n, r in probes if r]
    if ref and probes:
        print()
        print("  convergence probes (vs AB_C, on the gap -- alignment-free):")
        for n, r in probes:
            dg = (r["gap"] - ref["gap"]) * 1000
            print(f"    {n:14s} d_gap = {dg:+7.2f} meV   {classify(dg)}")

    print()
    print(f"  FINAL: {worst}")
    if worst == "GREEN":
        print("  The 49-point scalar landscape stands. Proceed with V0 = 121.60 meV,")
        print("  d_p = 0.00798 nm, Delta_0 = 1.6119 meV.")
    elif worst == "YELLOW":
        print("  Marginal. Recompute 3-6 additional registry points with SOC before")
        print("  committing the landscape to the manuscript.")
    else:
        print("  The 49-point scan must be repeated with the corrected treatment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
