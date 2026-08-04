"""gsfe_qe_input.py

Writes Quantum ESPRESSO pw.x input files for a GSFE (gamma-surface) scan
of a small TMD bilayer cell, at a grid (or high-symmetry path) of rigid
in-plane registry shifts, with the interlayer separation allowed to relax
(z-only) at each shift while all in-plane coordinates stay fixed (the
standard "relaxed GSFE" recipe).

Two vdW treatments are supported, matching the two-tier strategy discussed
with the group (Magda/Jaro): a cheap Grimme-D3 pass for the full grid, and
a nonlocal vdW-DF pass for a handful of high-symmetry points as a
cross-check.

IMPORTANT: pseudopotential filenames below are PLACEHOLDERS
--------------------------------------------------------------
`PSEUDO_FILENAMES` must be edited to match whatever pseudopotential set is
actually installed (e.g. SSSP precision/efficiency, PSlibrary, or a custom
set) -- this script does not download or verify pseudopotentials. Wrong
filenames will fail at pw.x startup, not silently produce wrong numbers,
so this is a fail-safe (not fail-silent) placeholder, but must be filled
in before actually submitting jobs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .gsfe_geometry import BilayerCell, avec

# EDIT ME: point these at your actual installed pseudopotential files.
PSEUDO_FILENAMES = {
    "Mo": "Mo_ONCV_PBE-1.0.oncvpsp.upf",
    "W": "W_pbe_v1.2.uspp.F.UPF",
    "Se": "Se_pbe_v1.uspp.F.UPF",
    "S": "S.upf",  # not used for MoSe2/WSe2; fill in only if you later add MoS2/WS2
}
PSEUDO_MASS_AMU = {"Mo": 95.95, "W": 183.84, "Se": 78.971, "S": 32.06}

VACUUM_A = 18.0  # extra vacuum above/below the bilayer to avoid image interaction


def _cell_parameters_block(a_common_A: float, c_A: float) -> str:
    a1, a2 = avec(a_common_A)
    lines = [
        "CELL_PARAMETERS angstrom",
        f"  {a1[0]:.10f} {a1[1]:.10f} 0.00000000",
        f"  {a2[0]:.10f} {a2[1]:.10f} 0.00000000",
        f"  0.00000000 0.00000000 {c_A:.10f}",
    ]
    return "\n".join(lines)


def _atomic_positions_block(cell: BilayerCell, shift_frac: tuple[float, float],
                             relax_interlayer_z: bool) -> str:
    """Cartesian-angstrom ATOMIC_POSITIONS block with optional selective-
    dynamics constraint flags (fix in-plane always; free z on the top
    layer only, when relax_interlayer_z=True -- the bottom layer stays
    fully fixed as the reference frame)."""
    a1, a2 = avec(cell.a_common_A)
    lines = ["ATOMIC_POSITIONS angstrom"]
    for i, (sp, is_top, (u, v), z) in enumerate(
        zip(cell.species, cell.is_top, cell.xy_frac, cell.z_A)
    ):
        if is_top:
            u = u + shift_frac[0]
            v = v + shift_frac[1]
        xy = u * a1 + v * a2
        if relax_interlayer_z and is_top:
            flags = "1"  # free z (uniform-ish interlayer relaxation)
        else:
            flags = "0"
        lines.append(f"  {sp} {xy[0]:.10f} {xy[1]:.10f} {z:.10f}  0 0 {flags}")
    return "\n".join(lines)


def write_pw_input(
    out_path: str | Path,
    cell: BilayerCell,
    shift_frac: tuple[float, float],
    vdw_mode: str = "grimme-d3",   # 'grimme-d3' | 'ts' | 'mbd' | 'vdw-df2-b86r' | 'rvv10' | 'none'
    relax_interlayer_z: bool = True,
    ecutwfc_Ry: float = 60.0,
    ecutrho_Ry: float | None = None,
    kpoints: tuple[int, int, int] = (9, 9, 1),
    prefix: str | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    c_A = cell.interlayer_gap_A + VACUUM_A
    species_unique = sorted(set(cell.species))
    ntyp = len(species_unique)
    nat = len(cell.species)

    calculation = "relax" if relax_interlayer_z else "scf"
    ecutrho = ecutrho_Ry if ecutrho_Ry is not None else 8.0 * ecutwfc_Ry

    if vdw_mode.lower() in ("grimme-d3", "grimme-d2", "ts", "mbd", "xdm"):
        vdw_lines = f"    vdw_corr = '{vdw_mode.lower()}'\n"
        input_dft_line = ""
    elif vdw_mode.lower() == "none":
        vdw_lines = ""
        input_dft_line = ""
    else:
        # nonlocal vdW-DF family, set via input_dft, e.g. 'vdw-df2-b86r', 'rvv10'
        vdw_lines = ""
        input_dft_line = f"    input_dft = '{vdw_mode}'\n"

    pfx = prefix or f"gsfe_u{shift_frac[0]:.4f}_v{shift_frac[1]:.4f}_{vdw_mode}"

    control = (
        "&CONTROL\n"
        f"    calculation = '{calculation}'\n"
        f"    prefix = '{pfx}'\n"
        "    pseudo_dir = './pseudo'\n"
        "    outdir = './out'\n"
        "    tprnfor = .true.\n"
        "    tstress = .false.\n"
        "    etot_conv_thr = 1.0d-6\n"
        "    forc_conv_thr = 1.0d-4\n"
        "/\n"
    )
    system = (
        "&SYSTEM\n"
        "    ibrav = 0\n"
        f"    nat = {nat}\n"
        f"    ntyp = {ntyp}\n"
        f"    ecutwfc = {ecutwfc_Ry:.2f}\n"
        f"    ecutrho = {ecutrho:.2f}\n"
        "    occupations = 'smearing'\n"
        "    smearing = 'gaussian'\n"
        "    degauss = 0.01\n"
        "    assume_isolated = '2D'\n"
        f"{vdw_lines}"
        f"{input_dft_line}"
        "/\n"
    )
    electrons = (
        "&ELECTRONS\n"
        "    conv_thr = 1.0d-9\n"
        "    mixing_beta = 0.3\n"
        "/\n"
    )
    ions = "&IONS\n    ion_dynamics = 'bfgs'\n/\n" if calculation == "relax" else ""

    atomic_species = ["ATOMIC_SPECIES"]
    for sp in species_unique:
        atomic_species.append(f"  {sp} {PSEUDO_MASS_AMU[sp]:.4f} {PSEUDO_FILENAMES[sp]}")
    atomic_species_block = "\n".join(atomic_species)

    cell_block = _cell_parameters_block(cell.a_common_A, c_A)
    positions_block = _atomic_positions_block(cell, shift_frac, relax_interlayer_z)
    kpoints_block = f"K_POINTS automatic\n  {kpoints[0]} {kpoints[1]} {kpoints[2]}  0 0 0"

    text = "\n".join([
        control, system, electrons, ions,
        atomic_species_block, "", cell_block, "", positions_block, "", kpoints_block, "",
    ])
    out_path.write_text(text)
    return out_path
