"""
build_commensurate_moire_v2.py

Reference implementation for constructing commensurate twisted
transition-metal dichalcogenide (TMD) bilayers.

Version: 2.1 (Fixed rotation & boundary overlap)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase import Atoms


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(slots=True)
class MoireMetadata:

    m: int
    n: int

    theta_rad: float
    theta_deg: float

    primitive_vectors: np.ndarray
    supercell_vectors: np.ndarray

    moire_period: float

    n_cells: int
    n_atoms: int


@dataclass(slots=True)
class MoireStructure:

    atoms: Atoms

    metadata: MoireMetadata


# =============================================================================
# BASIC GEOMETRY
# =============================================================================


def primitive_vectors(a: float):

    c1 = np.array(
        [
            a,
            0.0,
        ],
        dtype=float,
    )

    c2 = np.array(
        [
            a / 2.0,
            a * np.sqrt(3.0) / 2.0,
        ],
        dtype=float,
    )

    return c1, c2


def supercell_vectors(
    c1,
    c2,
    m,
    n,
):

    s1 = m * c1 + n * c2

    s2 = -n * c1 + (m + n) * c2

    return s1, s2


def supercell_matrix(
    s1,
    s2,
):

    return np.column_stack((s1, s2))


def commensurate_angle(
    m,
    n,
):

    num = m * m + n * n + 4 * m * n

    den = 2 * (m * m + n * n + m * n)

    value = np.clip(num / den, -1.0, 1.0)

    theta = np.arccos(value)

    return float(theta)


def rotation_matrix(theta):

    c = np.cos(theta)

    s = np.sin(theta)

    return np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=float,
    )


# =============================================================================
# BASIS
# =============================================================================


def basis_atoms(
    symbol_M,
    symbol_X,
    h_X,
):

    basis = []

    basis.append(
        (
            symbol_M,
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )
    )

    basis.append(
        (
            symbol_X,
            np.array(
                [
                    1.0 / 3.0,
                    1.0 / 3.0,
                    h_X,
                ]
            ),
        )
    )

    basis.append(
        (
            symbol_X,
            np.array(
                [
                    1.0 / 3.0,
                    1.0 / 3.0,
                    -h_X,
                ]
            ),
        )
    )

    return basis


# =============================================================================
# EXPECTED COUNTS
# =============================================================================


def expected_number_of_cells(
    m,
    n,
):

    return m * m + m * n + n * n


def expected_number_of_atoms(
    m,
    n,
):

    return 6 * expected_number_of_cells(
        m,
        n,
    )


# =============================================================================
# ENUMERATION
# =============================================================================


def generate_translations(
    c1,
    c2,
    s1,
    s2,
    m,
    n,
    eps=1e-10,
):

    S = supercell_matrix(
        s1,
        s2,
    )

    Sinv = np.linalg.inv(S)

    grid = 2 * (m + n)

    translations = []

    for i in range(
        -grid,
        grid + 1,
    ):

        for j in range(
            -grid,
            grid + 1,
        ):

            R = i * c1 + j * c2

            frac = Sinv @ R

            if (
                -eps <= frac[0] < 1.0 - eps
                and
                -eps <= frac[1] < 1.0 - eps
            ):

                translations.append(R)

    expected = expected_number_of_cells(
        m,
        n,
    )

    if len(translations) != expected:

        raise RuntimeError(
            f"Expected {expected} primitive cells "
            f"but generated {len(translations)}."
        )

    return translations


# =============================================================================
# BASIS -> CARTESIAN
# =============================================================================


def fractional_to_cartesian(
    frac_xy,
    c1,
    c2,
):

    return frac_xy[0] * c1 + frac_xy[1] * c2
    
# =============================================================================
# LAYER CONSTRUCTION
# =============================================================================


def build_layer(
    translations,
    c1,
    c2,
    basis,
    z_shift=0.0,
):
    """
    Build one MX2 layer using native primitive vectors and translations.
    """

    positions = []
    symbols = []

    for T in translations:

        origin_xy = T.copy()

        for symbol, frac in basis:

            xy = (
                origin_xy
                +
                fractional_to_cartesian(
                    frac[:2],
                    c1,
                    c2,
                )
            )

            pos = np.array(
                [
                    xy[0],
                    xy[1],
                    frac[2] + z_shift,
                ],
                dtype=float,
            )

            positions.append(pos)

            symbols.append(symbol)

    return symbols, positions



# =============================================================================
# DUPLICATE CHECKING
# =============================================================================


def remove_duplicate_atoms(
    atoms,
    tolerance=1e-5,
):
    """
    Remove numerically duplicated atoms.
    """

    positions = atoms.get_positions()

    symbols = atoms.get_chemical_symbols()

    keep = []

    for i, pos in enumerate(positions):

        duplicate = False

        for j in keep:

            if (
                symbols[i] == symbols[j]
                and
                np.linalg.norm(
                    pos - positions[j]
                ) < tolerance
            ):
                duplicate = True
                break

        if not duplicate:
            keep.append(i)


    if len(keep) == len(atoms):

        return atoms


    return Atoms(
        symbols=[
            symbols[i]
            for i in keep
        ],
        positions=[
            positions[i]
            for i in keep
        ],
        cell=atoms.cell,
        pbc=atoms.pbc,
    )



# =============================================================================
# VALIDATION
# =============================================================================


def validate_structure(
    atoms,
    m,
    n,
):
    """
    Validate generated structure.
    """

    expected = expected_number_of_atoms(
        m,
        n,
    )

    actual = len(atoms)

    if actual != expected:

        raise RuntimeError(
            f"Wrong atom count: "
            f"expected {expected}, got {actual}"
        )


    cell = atoms.cell.array


    det = np.linalg.det(
        cell[:2,:2]
    )


    if det <= 0:

        raise RuntimeError(
            "Invalid cell orientation."
        )


    positions = atoms.get_positions()


    min_distance = np.inf


    for i in range(len(positions)):

        for j in range(i+1,len(positions)):

            d = np.linalg.norm(
                positions[i]
                -
                positions[j]
            )

            if d < min_distance:

                min_distance = d


    if min_distance < 0.5:

        raise RuntimeError(
            f"Suspicious atom overlap: "
            f"{min_distance:.3f} Angstrom"
        )


    return True



# =============================================================================
# MAIN BUILDER
# =============================================================================


def build_commensurate_moire(
    symbol_M="Mo",
    symbol_X="Se",
    a=3.28,
    h_X=1.66,
    interlayer_d=6.5,
    m=17,
    n=16,
    vacuum_z=20.0,
):
    """
    Build commensurate twisted MX2 bilayer.

    Returns
    -------
    MoireStructure
    """


    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    c1, c2 = primitive_vectors(a)

    s1, s2 = supercell_vectors(
        c1,
        c2,
        m,
        n,
    )

    theta = commensurate_angle(
        m,
        n,
    )

    R = rotation_matrix(theta)

    # Rotated primitive vectors for the top layer
    c1_top = R @ c1
    c2_top = R @ c2


    # -------------------------------------------------------------------------
    # Translation vectors for both layers (shared supercell s1, s2)
    # -------------------------------------------------------------------------

    bot_translations = generate_translations(
        c1,
        c2,
        s1,
        s2,
        m,
        n,
    )

    top_translations = generate_translations(
        c1_top,
        c2_top,
        s1,
        s2,
        m,
        n,
    )


    # -------------------------------------------------------------------------
    # Basis
    # -------------------------------------------------------------------------

    basis = basis_atoms(
        symbol_M,
        symbol_X,
        h_X,
    )


    # -------------------------------------------------------------------------
    # Bottom layer
    # -------------------------------------------------------------------------

    bot_symbols, bot_positions = build_layer(
        bot_translations,
        c1,
        c2,
        basis,
        z_shift=0.0,
    )


    # -------------------------------------------------------------------------
    # Top layer (natively rotated via c1_top, c2_top)
    # -------------------------------------------------------------------------

    top_symbols, top_positions = build_layer(
        top_translations,
        c1_top,
        c2_top,
        basis,
        z_shift=interlayer_d,
    )


    symbols = (
        bot_symbols
        +
        top_symbols
    )


    positions = (
        bot_positions
        +
        top_positions
    )


    # -------------------------------------------------------------------------
    # ASE object
    # -------------------------------------------------------------------------

    cell = np.array(
        [
            [
                s1[0],
                s1[1],
                0.0,
            ],
            [
                s2[0],
                s2[1],
                0.0,
            ],
            [
                0.0,
                0.0,
                vacuum_z,
            ],
        ]
    )


    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=cell,
        pbc=[
            True,
            True,
            False,
        ],
    )


    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    atoms.wrap()

    atoms = remove_duplicate_atoms(
        atoms
    )


    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    validate_structure(
        atoms,
        m,
        n,
    )


    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    metadata = MoireMetadata(

        m=m,

        n=n,

        theta_rad=theta,

        theta_deg=np.degrees(theta),

        primitive_vectors=np.array(
            [
                c1,
                c2,
            ]
        ),

        supercell_vectors=np.array(
            [
                s1,
                s2,
            ]
        ),

        moire_period=np.linalg.norm(s1),

        n_cells=expected_number_of_cells(
            m,
            n,
        ),

        n_atoms=len(atoms),
    )


    return MoireStructure(
        atoms=atoms,
        metadata=metadata,
    )

# =============================================================================
# EXPORT HELPERS
# =============================================================================


def write_xyz(
    structure: MoireStructure,
    filename: str,
):
    """
    Write structure to XYZ format.
    """

    structure.atoms.write(
        filename,
        format="xyz",
    )



def write_lammps(
    structure: MoireStructure,
    filename: str,
):
    """
    Write structure in LAMMPS data format.
    """

    from ase.io import write

    write(
        filename,
        structure.atoms,
        format="lammps-data",
    )



# =============================================================================
# REPORT
# =============================================================================


def print_report(
    structure: MoireStructure,
):

    meta = structure.metadata

    print("=" * 70)
    print(" COMMENSURATE MOIRE STRUCTURE REPORT")
    print("=" * 70)

    print(
        f"m,n                : "
        f"{meta.m}, {meta.n}"
    )

    print(
        f"Twist angle        : "
        f"{meta.theta_deg:.8f} deg"
    )

    print(
        f"Moire period       : "
        f"{meta.moire_period:.6f} Angstrom"
    )

    print(
        f"Primitive cells    : "
        f"{meta.n_cells}"
    )

    print(
        f"Atoms              : "
        f"{meta.n_atoms}"
    )

    print(
        "Supercell vectors:"
    )

    print(
        meta.supercell_vectors
    )

    print("=" * 70)



# =============================================================================
# SELF TEST
# =============================================================================


if __name__ == "__main__":


    structure = build_commensurate_moire(
        symbol_M="Mo",
        symbol_X="Se",
        a=3.28,
        h_X=1.66,
        interlayer_d=6.5,
        m=17,
        n=16,
        vacuum_z=20.0,
    )


    print_report(
        structure
    )


    write_xyz(
        structure,
        "MoSe2_moire_17_16.xyz",
    )


    print(
        "XYZ written successfully."
    )
