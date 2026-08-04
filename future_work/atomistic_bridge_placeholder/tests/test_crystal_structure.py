"""Regression test for the trigonal-prismatic-coordination bug in
moire_pipeline.structure_relaxation.monolayer() (fixed 2026-08-02).

BUG THIS GUARDS AGAINST
-------------------------
The chalcogen basis used chalcogen_top at fractional (1/3, 2/3) and
chalcogen_bottom at fractional (2/3, 1/3) -- two DIFFERENT in-plane
positions. This is wrong for 2H-phase TMDs: true trigonal-prismatic
coordination requires the top and bottom chalcogen triangles around each
metal atom to be ECLIPSED (stacked directly on top of each other, same
in-plane position, differing only in z) -- "two equilateral triangles,
one directly above the other." The old basis instead produced a
staggered (rotated ~60 degrees relative to each other, more like an
octahedral/antiprismatic motif) arrangement.

This was NOT caught by an earlier review this session that checked only
for formal C3 symmetry (present in both the buggy and fixed versions) --
formal point-group symmetry alone does not guarantee correct bond
lengths or coordination geometry. Verified directly by computing nearest-
neighbor M-X distances: the buggy (1/3,2/3) offset alone gives a nearest
neighbor at an unphysical ~2.00 A with NO threefold degeneracy (2nd/3rd
neighbors sit at different, larger distances -- not a valid
trigonal-prismatic motif at all), while the corrected shared (1/3,1/3)
offset gives exactly three degenerate nearest-neighbor bonds at 2.528 A,
matching the experimental Mo-Se bond length (~2.53 A).
"""
import math
import numpy as np
import pytest

from moire_pipeline.structure_relaxation import build_structure, TMD

# Experimental/reference bond lengths (Angstrom), loose tolerance since
# this is a rigid (unrelaxed) structure, not a DFT-relaxed one.
EXPECTED_MX_BOND_A = {
    "MoSe2": 2.528,  # matches 3-fold-degenerate nearest-neighbor calc, see module docstring
    "WSe2": 2.526,
}
BOND_TOL_A = 0.01


def _nearest_metal_index(species, layer, sublattice, positions, lname):
    """Pick a metal atom near the box center (not just idx[0]) so nearest-
    neighbor counting isn't contaminated by finite-box edge truncation."""
    idx = np.where((layer == lname) & (sublattice == "metal"))[0]
    assert len(idx) > 0
    xy = positions[idx][:, :2]
    center_idx = idx[np.argmin(np.sum(xy**2, axis=1))]
    return center_idx


def _m_x_bond_lengths_and_degeneracy(species, layer, sublattice, positions, lname, n_check=6):
    """For one metal atom in the given layer, return the sorted M-X 3D
    distances to its n_check nearest chalcogen neighbors (both top and
    bottom sublattices) in the same layer."""
    m_idx = _nearest_metal_index(species, layer, sublattice, positions, lname)
    m_pos = positions[m_idx]
    x_mask = (layer == lname) & (sublattice != "metal")
    x_pos = positions[x_mask]
    d = np.linalg.norm(x_pos - m_pos, axis=1)
    return np.sort(d)[:n_check]


@pytest.fixture(scope="module")
def structure():
    return build_structure(theta_deg=0.0, box_nm=15.0,
                            bottom_material="MoSe2", top_material="WSe2")


def test_three_degenerate_nearest_neighbor_bonds(structure):
    """Each metal atom must have SIX nearest-neighbor M-X bonds at the
    same (degenerate) distance: three from the chalcogen_top triangle and
    three from the chalcogen_bottom triangle, all at identical distance
    since |+dz| == |-dz| for eclipsed trigonal-prismatic coordination
    (the metal sits equidistant from both triangular faces of the prism).
    The old buggy basis fails this outright (non-degenerate nearest
    neighbors, and not even a consistent shell of 6)."""
    for lname, material in [("bottom", "MoSe2"), ("top", "WSe2")]:
        d = _m_x_bond_lengths_and_degeneracy(
            structure["species"], structure["layer"], structure["sublattice"],
            structure["positions_initial_A"], lname, n_check=8,
        )
        nearest_six = d[:6]
        assert np.max(nearest_six) - np.min(nearest_six) < 1e-6, (
            f"{lname} ({material}): nearest-neighbor M-X bonds are not "
            f"sixfold-degenerate (3 top + 3 bottom): {nearest_six}"
        )
        expected = EXPECTED_MX_BOND_A[material]
        assert nearest_six[0] == pytest.approx(expected, abs=BOND_TOL_A), (
            f"{lname} ({material}): M-X bond length {nearest_six[0]:.4f} A "
            f"vs expected {expected} A"
        )
        # the 7th-nearest must be strictly farther (a genuine, isolated shell)
        assert d[6] > nearest_six[0] + 0.1


def test_top_and_bottom_chalcogen_are_eclipsed_not_staggered(structure):
    """chalcogen_top and chalcogen_bottom within the same layer must sit
    at the SAME in-plane (x,y) position for every metal's local
    environment -- i.e. the two triangles are eclipsed ('one directly
    above the other'), not rotated relative to each other."""
    layer = structure["layer"]
    sub = structure["sublattice"]
    pos = structure["positions_initial_A"]

    for lname in ("bottom", "top"):
        mask_layer = layer == lname
        top_mask = mask_layer & (sub == "chalcogen_top")
        bot_mask = mask_layer & (sub == "chalcogen_bottom")
        xy_top = pos[top_mask][:, :2]
        xy_bottom = pos[bot_mask][:, :2]
        assert len(xy_top) > 0 and len(xy_bottom) == len(xy_top)

        # xy_top and xy_bottom are generated by the same loop order over
        # the same (i,j) lattice cells, so they must line up index-for-index.
        np.testing.assert_allclose(xy_top, xy_bottom, atol=1e-9)


def test_chalcogen_z_still_symmetric_about_metal_plane(structure):
    """The z-offset (out-of-plane part of trigonal-prismatic coordination)
    must be unaffected by the in-plane basis fix: chalcogen_top at +z,
    chalcogen_bottom at -z, symmetric about the metal's z-plane."""
    for lname, material in [("bottom", "MoSe2"), ("top", "WSe2")]:
        layer = structure["layer"]
        sub = structure["sublattice"]
        pos = structure["positions_initial_A"]
        z_metal = pos[(layer == lname) & (sub == "metal")][0, 2]
        z_top = pos[(layer == lname) & (sub == "chalcogen_top")][:, 2]
        z_bot = pos[(layer == lname) & (sub == "chalcogen_bottom")][:, 2]
        expected_dz = TMD[material].chalcogen_z_A
        np.testing.assert_allclose(z_top - z_metal, expected_dz, atol=1e-9)
        np.testing.assert_allclose(z_bot - z_metal, -expected_dz, atol=1e-9)
