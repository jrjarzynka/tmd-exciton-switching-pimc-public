"""gsfe_geometry.py

Small-cell (single primitive hexagonal cell, ~6 atoms) bilayer builder for
a generalized stacking-fault-energy (GSFE / "gamma surface") DFT scan.

WHY A SMALL CELL, NOT THE FULL MOIRE SUPERCELL
-------------------------------------------------
registry_energy() in moire_pipeline/generate_potential_map.py is a local,
intrinsic property of the stacking (how the energy varies as you rigidly
slide the top layer over the bottom one) -- it does NOT depend on the
long-range moire periodicity itself. The standard way to compute this
(the "gamma surface"/GSFE approach used throughout the 2D-materials
literature) is: build ONE small hexagonal bilayer cell at a common
(strain-averaged) in-plane lattice constant, rigidly translate the top
layer over a grid of in-plane offsets covering the primitive cell, and
compute the DFT total energy (optionally relaxing only the interlayer
separation) at each offset. The large (~4902-atom) commensurate moire
supercell is a SEPARATE, later step needed only for the strain/deformation
part of the potential map (Vdef, Vinterlayer terms) -- it is not required
to calibrate registry_energy() itself.

A single common lattice constant (here: the simple average of the two
monolayers' lattice constants) is used because the layer mismatch is tiny
(~0.18% for WSe2/MoSe2) -- this is the same "leading-order moire theory"
approximation already used (and noted as physically correct) for
registry_energy()'s three-G-vector cosine-sum form.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

# Same source-of-truth lattice constants as structure_relaxation.py's TMD dict.
LATTICE_A = {
    "MoSe2": 3.288,
    "WSe2": 3.282,
    "MoS2": 3.160,
    "WS2": 3.153,
}
CHALCOGEN_Z_A = {"MoSe2": 1.67, "WSe2": 1.67, "MoS2": 1.56, "WS2": 1.56}
METAL = {"MoSe2": "Mo", "WSe2": "W", "MoS2": "Mo", "WS2": "W"}
CHALCOGEN = {"MoSe2": "Se", "WSe2": "Se", "MoS2": "S", "WS2": "S"}


def avec(a: float):
    """Primitive in-plane lattice vectors (Angstrom), 2D hexagonal lattice."""
    return np.array([a, 0.0]), np.array([0.5 * a, 0.5 * math.sqrt(3) * a])


@dataclass
class BilayerCell:
    a_common_A: float
    interlayer_gap_A: float
    species: list       # length-6 list of element symbols
    is_top: np.ndarray   # bool mask, True for top-layer atoms
    xy_frac: np.ndarray  # (6,2) fractional in-plane coords (bottom-layer frame)
    z_A: np.ndarray      # (6,) absolute z coordinates, Angstrom


def build_gsfe_cell(bottom_material: str, top_material: str,
                     interlayer_gap_A: float = 6.5,
                     a_common_A: float | None = None) -> BilayerCell:
    """One primitive hexagonal cell per layer (1 metal + 2 chalcogen each,
    2H stacking basis matching structure_relaxation.py's convention),
    stacked with AA registry (top directly above bottom) at zero shift.
    A registry shift is applied later, in fractional coordinates, by
    generate_qe_input -- so this builder always returns the AA reference.
    """
    a_bot = LATTICE_A[bottom_material]
    a_top = LATTICE_A[top_material]
    a_common = a_common_A if a_common_A is not None else 0.5 * (a_bot + a_top)

    # BUGFIX (2026-08-02): true 2H trigonal-prismatic coordination requires
    # the top and bottom chalcogens to be ECLIPSED (same in-plane
    # position, differing only in z) -- verified numerically that a
    # shared (1/3,1/3) offset gives exactly three degenerate M-X bonds at
    # 2.528 A (matching experiment: Mo-Se ~2.53 A), while the previously-
    # used (1/3,2/3)/(2/3,1/3) staggered pair does not (unphysical,
    # non-degenerate nearest-neighbor distances -- confirmed by direct
    # calculation, not just a symmetry argument). Matching fix applied in
    # structure_relaxation.py.
    basis_frac = np.array([
        [0.0, 0.0],               # metal
        [1.0 / 3.0, 1.0 / 3.0],   # chalcogen top and bottom share this xy
        [1.0 / 3.0, 1.0 / 3.0],   # (differ only in z, set below)
    ])

    species = (
        [METAL[bottom_material], CHALCOGEN[bottom_material], CHALCOGEN[bottom_material]]
        + [METAL[top_material], CHALCOGEN[top_material], CHALCOGEN[top_material]]
    )
    is_top = np.array([False, False, False, True, True, True])

    z_intra_bottom = CHALCOGEN_Z_A[bottom_material]
    z_intra_top = CHALCOGEN_Z_A[top_material]
    z0_bottom = -0.5 * interlayer_gap_A
    z0_top = +0.5 * interlayer_gap_A
    z_A = np.array([
        z0_bottom, z0_bottom + z_intra_bottom, z0_bottom - z_intra_bottom,
        z0_top, z0_top + z_intra_top, z0_top - z_intra_top,
    ])

    xy_frac = np.vstack([basis_frac, basis_frac])  # AA reference: same in-plane frac coords

    return BilayerCell(a_common, interlayer_gap_A, species, is_top, xy_frac, z_A)


def registry_shift_grid(n: int = 7) -> np.ndarray:
    """Fractional-coordinate (du, dv) grid over the primitive cell,
    n x n points including both endpoints 0 and (n-1)/n (periodic, so the
    point at u=1 is redundant with u=0 and is NOT included -- standard
    GSFE grid convention)."""
    u = np.arange(n) / n
    v = np.arange(n) / n
    U, V = np.meshgrid(u, v, indexing="ij")
    return np.column_stack([U.ravel(), V.ravel()])


def high_symmetry_path(n_per_segment: int = 6) -> tuple[np.ndarray, list[str]]:
    """AA -> bridge -> AB -> AA path in fractional coords, the standard
    reduced set of high-symmetry stacking points for a hexagonal bilayer
    GSFE, useful as a cheap validation subset (e.g. for the nonlocal vdW-DF
    cross-check) instead of the full 2D grid."""
    pts_frac = {
        "AA": np.array([0.0, 0.0]),
        "bridge": np.array([1.0 / 3.0, 1.0 / 6.0]),
        "AB": np.array([1.0 / 3.0, 2.0 / 3.0]),
    }
    order = ["AA", "bridge", "AB", "AA"]
    coords = []
    labels = []
    for i in range(len(order) - 1):
        p0, p1 = pts_frac[order[i]], pts_frac[order[i + 1]]
        for t in np.linspace(0.0, 1.0, n_per_segment, endpoint=False):
            coords.append(p0 + t * (p1 - p0))
            labels.append(order[i] if t == 0.0 else f"{order[i]}-{order[i+1]}_{t:.2f}")
    coords.append(pts_frac[order[-1]])
    labels.append(order[-1])
    return np.array(coords), labels
