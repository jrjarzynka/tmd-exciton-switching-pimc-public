#!/usr/bin/env python3
"""
stage0_generate.py

Generate the complete Stage 0 validation package: does spin-orbit coupling or
the slab dipole correction change the moire landscape that Paper 2 is built on?

The comparison must be three-stage, not two
-------------------------------------------
The existing 49-point scan uses scalar-relativistic pseudopotentials, which
cannot represent spin-orbit coupling at all: switching on noncolin/lspinorb with
them runs without error and produces something that is not SOC. Fully
relativistic pseudopotentials are therefore mandatory -- but they also come from
a different family (PAW rather than ONCV+USPP, and Se carries its 3d shell in
valence, 16 electrons against 6). Comparing the old scalar run directly against
a new SOC run would mix the two changes with no way to separate them.

    A   old pseudos, scalar              reference for the 49-point landscape
    B   FR pseudos,  scalar              isolates the PSEUDOPOTENTIAL change
    C   FR pseudos,  noncolin + SOC      isolates SOC                (B -> C)
    D   C + dipole correction            isolates the dipole field   (C -> D)

Variant A already exists from the highsym runs, so only B, C, D are generated.

Geometry is frozen
------------------
Stage 0 asks what the electronic treatment does at FIXED structure, so nothing
relaxes and no coordinate, cell vector, cutoff, smearing or k-grid changes
between variants. The relaxed positions are taken verbatim from the completed
AA/AB/BA runs.

Convergence is tested separately
--------------------------------
Cutoff, k-grid and vacuum thickness are each varied on ONE stacking in variant
C or D, never mixed into the four physical variants. "Does dipfield move the
VBM" and "is 15 A of vacuum enough" are different questions and a single sweep
that changes both answers neither.

Vacuum is deliberately left at the production value of 24.5 A for B/C/D, so
that the comparison against the existing 49 points stays clean; a separate
sweep at 24.5 / 30 / 35 A on one stacking asks whether that value suffices.
"""

from __future__ import annotations

import argparse
import os

A_LAT = 3.2850000000
CELL_C_DEFAULT = 24.5

# Relaxed geometries, verbatim from the completed highsym runs.
# Atoms 1-3 are the lower MoSe2 layer, 4-6 the upper WSe2 layer.
GEOMETRIES = {
    "AA": [
        ("Mo", 0.0000000000, 0.0000000000, -3.2500000000),
        ("Se", 1.6425000000, 0.9482978171, -1.5800000000),
        ("Se", 1.6425000000, 0.9482978171, -4.9200000000),
        ("W",  0.0000000000, 0.0000000000,  3.0165592181),
        ("Se", 1.6425000000, 0.9482978171,  4.6502734382),
        ("Se", 1.6425000000, 0.9482978171,  1.3800071365),
    ],
    "AB": [
        ("Mo", 0.0000000000, 0.0000000000, -3.2500000000),
        ("Se", 1.6425000000, 0.9482978171, -1.5800000000),
        ("Se", 1.6425000000, 0.9482978171, -4.9200000000),
        ("W",  1.6425000000, 0.9482978171,  2.6693045762),
        ("Se", 3.2850000000, 1.8965956342,  4.3295805825),
        ("Se", 3.2850000000, 1.8965956342,  1.0060303660),
    ],
    "BA": [
        ("Mo", 0.0000000000, 0.0000000000, -3.2500000000),
        ("Se", 1.6425000000, 0.9482978171, -1.5800000000),
        ("Se", 1.6425000000, 0.9482978171, -4.9200000000),
        ("W",  3.2850000000, 1.8965956343,  2.6634203798),
        ("Se", 4.9275000000, 2.8448934514,  4.3262302251),
        ("Se", 4.9275000000, 2.8448934514,  1.0055160059),
    ],
}

# Two sets from the SAME PSLibrary family, same generator, same z_valence
# (14/16/14) and the same suggested cutoffs. They differ only in whether the
# j-resolved channels are present.
#
# Variant B must use the scalar set: QE refuses a fully relativistic
# pseudopotential unless lspinorb is on ("Fully relativistic PPs, need
# spin-orbit calc."), so a noncollinear-without-SOC control is impossible with
# these files. B -> C therefore compares scalar against j-resolved channels --
# which is what spin-orbit IS, at the pseudopotential level -- rather than
# toggling one flag. Nothing else changes: same family, same valence
# configuration, same cutoffs.
PSEUDO_FR = {
    "Mo": ("95.9500",  "Mo.rel-pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "Se": ("78.9710",  "Se.rel-pbe-dn-kjpaw_psl.1.0.0.UPF"),
    "W":  ("183.8400", "W.rel-pbe-spn-kjpaw_psl.1.0.0.UPF"),
}
PSEUDO_SR = {
    "Mo": ("95.9500",  "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF"),
    "Se": ("78.9710",  "Se.pbe-dn-kjpaw_psl.1.0.0.UPF"),
    "W":  ("183.8400", "W.pbe-spn-kjpaw_psl.1.0.0.UPF"),
}

# 92 valence electrons -> 92 occupied spinor bands under SOC, 46 under scalar.
# Eight empty bands put the highest computed state ~1.9 eV above E_F, which is
# ample: only the band immediately above the gap is needed, but band ordering
# shifts between stackings and a CBM falling outside nbnd would be wrong
# silently rather than loudly.
NBND_SOC = 100
NBND_SCALAR = 54


def vacuum_window(atoms, cell_c):
    """Fractional-z interval occupied by vacuum, and a safe sawtooth placement.

    The slab straddles z = 0, so wrapping puts it at both ends of the cell and
    the vacuum in the middle. The dipole correction's discontinuity region
    [emaxpos, emaxpos + eopreg] must lie entirely inside that vacuum, or the
    sawtooth cuts through the slab and the correction is meaningless.
    """
    fz = sorted((z / cell_c) % 1.0 for _, _, _, z in atoms)
    lo = max(v for v in fz if v < 0.5)
    hi = min(v for v in fz if v > 0.5)
    eopreg = 0.10
    emaxpos = 0.5 * (lo + hi) - 0.5 * eopreg
    return lo, hi, emaxpos, eopreg


def build_input(stacking, variant, cell_c=CELL_C_DEFAULT, ecutwfc=70.0,
                ecutrho=700.0, kgrid=9, prefix=None, outdir="./tmp_stage0"):
    atoms = GEOMETRIES[stacking]
    # Variant B is noncollinear WITHOUT spin-orbit, not scalar. A scalar run
    # with these fully relativistic pseudopotentials fails in average_pp: QE
    # cannot j-average these particular files. Running B noncollinear is in any
    # case the tighter control -- identical spinor machinery, only lspinorb
    # toggled -- so B -> C isolates spin-orbit and nothing else. It costs the
    # same as C rather than a quarter, which is the price of the cleaner test.
    soc = variant in ("C", "D")
    noncolin = soc
    dip = variant == "D"
    prefix = prefix or f"{stacking}_{variant}"

    sys_lines = [
        "  ibrav = 0",
        "  nat = 6",
        "  ntyp = 3",
        f"  ecutwfc = {ecutwfc:.2f}",
        f"  ecutrho = {ecutrho:.2f}",
        "  occupations = 'smearing'",
        "  smearing = 'gaussian'",
        "  degauss = 0.01",
        "  vdw_corr = 'grimme-d3'",
        f"  nbnd = {NBND_SOC if noncolin else NBND_SCALAR}",
    ]
    if noncolin:
        sys_lines += ["  noncolin = .true.",
                      f"  lspinorb = .{'true' if soc else 'false'}."]
    if dip:
        lo, hi, emaxpos, eopreg = vacuum_window(atoms, cell_c)
        # No comment lines inside the namelist: Fortran namelist parsing
        # rejects "!" here, and QE reports it as a bad line on the NEXT entry,
        # which points at tefield rather than at the actual offender.
        sys_lines += [
            "  tefield = .true.",
            "  dipfield = .true.",
            "  edir = 3",
            f"  emaxpos = {emaxpos:.4f}",
            f"  eopreg = {eopreg:.4f}",
            "  eamp = 0.0",
        ]

    out = [
        "&CONTROL",
        "  calculation = 'scf'",
        f"  prefix = '{prefix}'",
        "  pseudo_dir = './pseudo_rel'",
        f"  outdir = '{outdir}'",
        "  disk_io = 'low'",
        "  verbosity = 'high'",
        "/",
        "&SYSTEM",
        *sys_lines,
        "/",
        "&ELECTRONS",
        "  conv_thr = 1.0d-7",
        "  mixing_beta = 0.5",
        "  mixing_mode = 'local-TF'",
        "/",
        "ATOMIC_SPECIES",
    ]
    for sym in ("Mo", "Se", "W"):
        mass, upf = (PSEUDO_FR if soc else PSEUDO_SR)[sym]
        out.append(f"  {sym} {mass} {upf}")

    out += [
        "CELL_PARAMETERS angstrom",
        f"  {A_LAT:.10f} 0.0000000000 0.0000000000",
        f"  {A_LAT / 2:.10f} {A_LAT * 3 ** 0.5 / 2:.10f} 0.0000000000",
        f"  0.0000000000 0.0000000000 {cell_c:.10f}",
        "ATOMIC_POSITIONS angstrom",
    ]
    for sym, x, y, z in atoms:
        out.append(f"  {sym} {x:.10f} {y:.10f} {z:.10f}")

    out += ["K_POINTS automatic", f"  {kgrid} {kgrid} 1 0 0 0", ""]
    return "\n".join(out)


def build_projwfc(prefix, outdir="./tmp_stage0"):
    """projwfc.x input. lsym=.false. keeps per-atom projections unsymmetrised,
    which is what the layer decomposition needs."""
    return "\n".join([
        "&PROJWFC",
        f"  prefix = '{prefix}'",
        f"  outdir = '{outdir}'",
        "  lsym = .false.",
        "  ngauss = 0",
        "  degauss = 0.01",
        f"  filpdos = 'pdos/{prefix}'",
        "/",
        "",
    ])


def build_pp(prefix, outdir="./tmp_stage0"):
    """Planar-averaged electrostatic potential along z.

    plot_num = 11 is the bare + Hartree potential, which is what defines the
    vacuum level. Comparing its plateau with and without the dipole correction
    shows directly whether the image field is contaminating the band alignment
    -- and lets the cheap semicore alignment, which needs no post-processing at
    all, be validated against it.
    """
    return "\n".join([
        "&INPUTPP",
        f"  prefix = '{prefix}'",
        f"  outdir = '{outdir}'",
        "  plot_num = 11",
        f"  filplot = 'pot/{prefix}.pot'",
        "/",
        "&PLOT",
        "  iflag = 1",
        "  output_format = 0",
        "  e1(1) = 0.0, e1(2) = 0.0, e1(3) = 1.0",
        "  nx = 400",
        f"  fileout = 'pot/{prefix}_vz.dat'",
        "/",
        "",
    ])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="stage0")
    args = ap.parse_args(argv)

    d = args.outdir
    for sub in ("", "/pdos", "/pot"):
        os.makedirs(d + sub, exist_ok=True)

    written = []

    # main grid: three variants on three stackings
    for stacking in ("AA", "AB", "BA"):
        for variant in ("B", "C", "D"):
            name = f"{stacking}_{variant}"
            with open(f"{d}/{name}.in", "w") as fh:
                fh.write(build_input(stacking, variant))
            with open(f"{d}/{name}.projwfc.in", "w") as fh:
                fh.write(build_projwfc(name))
            written.append(name)

    # electrostatic profile: only where it answers something, i.e. C vs D
    for name in ("AB_C", "AB_D"):
        with open(f"{d}/{name}.pp.in", "w") as fh:
            fh.write(build_pp(name))

    # convergence probes, one stacking each, one knob at a time
    probes = {
        "conv_ecut85": dict(stacking="AB", variant="C", ecutwfc=85.0, ecutrho=850.0),
        "conv_k12":    dict(stacking="AB", variant="C", kgrid=12),
        "conv_k15":    dict(stacking="AB", variant="C", kgrid=15),
        "conv_vac30":  dict(stacking="AB", variant="D", cell_c=30.0),
        "conv_vac35":  dict(stacking="AB", variant="D", cell_c=35.0),
    }
    for name, kw in probes.items():
        with open(f"{d}/{name}.in", "w") as fh:
            fh.write(build_input(prefix=name, **kw))
        written.append(name)

    print(f"wrote {len(written)} SCF inputs to {d}/")
    for stacking in ("AA", "AB", "BA"):
        atoms = GEOMETRIES[stacking]
        lo, hi, em, eo = vacuum_window(atoms, CELL_C_DEFAULT)
        d_mw = atoms[3][3] - atoms[0][3]
        print(f"  {stacking}: d(Mo-W) = {d_mw:.4f} A   vacuum fractional "
              f"[{hi:.4f}, {lo + 1:.4f}]   emaxpos {em:.4f}")
    print()
    print("estimated wall time (8 cores, from the AB variant-C benchmark):")
    print("   9 x SCF   : 3 x B ~6 min + 3 x C ~22 min + 3 x D ~23 min  = 2.6 h")
    print("   5 x probe : ~35 + 40 + 60 + 30 + 35 min                   = 3.3 h")
    print("   projwfc   : 9 x ~15 s                                     = 0.1 h")
    print("   TOTAL                                                     ~ 6.0 h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
