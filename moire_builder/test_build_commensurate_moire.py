"""
Regression tests for build_commensurate_moire.

Each test in the first group corresponds to a specific defect in v2.0 that
passed that version's own validation.  The point of the group is that atom
counting and naive nearest-neighbour checks detect none of them.
"""

import numpy as np
import pytest
from ase import Atoms

import build_commensurate_moire as B


M, N_IDX = 17, 16
EXPECTED_CELLS = 817
EXPECTED_ATOMS = 4902
EXPECTED_THETA_DEG = 2.004628


# =============================================================================
# GROUP 1 -- the three v2.0 defects
# =============================================================================


def test_v2_supercell_is_rejected():
    """
    v2.0 bug 1: supercell built from (m, n) while the top layer is rotated by
    +theta.  R(-theta) T1 is then not a lattice vector, so the rotated layer is
    not periodic under the cell.
    """
    c1, c2 = B.primitive_vectors(3.288)
    theta = B.commensurate_angle(M, N_IDX)

    bad_T1 = M * c1 + N_IDX * c2
    bad_T2 = -N_IDX * c1 + (M + N_IDX) * c2

    with pytest.raises(RuntimeError, match="Non-commensurate"):
        B.assert_commensurate(c1, c2, bad_T1, bad_T2, theta)


def test_correct_supercell_is_accepted():
    c1, c2 = B.primitive_vectors(3.288)
    theta = B.commensurate_angle(M, N_IDX)
    T1, T2 = B.coincidence_vectors(c1, c2, M, N_IDX)
    B.assert_commensurate(c1, c2, T1, T2, theta)


def test_csl_not_invariant_under_positive_rotation():
    """
    v2.0 bug 2, root cause: the CSL is invariant under R(-theta) but not under
    R(+theta).  Rotating the bottom layer's in-cell sites therefore does not
    produce a valid set of coset representatives for the top layer.
    """
    c1, c2 = B.primitive_vectors(3.288)
    theta = B.commensurate_angle(M, N_IDX)
    T1, T2 = B.coincidence_vectors(c1, c2, M, N_IDX)
    A = np.column_stack((c1, c2))

    for T in (T1, T2):
        f_minus = np.linalg.solve(A, B.rotation_matrix(-theta) @ T)
        assert np.allclose(f_minus, np.round(f_minus), atol=1e-8)

        f_plus = np.linalg.solve(A, B.rotation_matrix(theta) @ T)
        assert not np.allclose(f_plus, np.round(f_plus), atol=1e-8)


def test_rotating_in_cell_sites_collapses_atoms():
    """
    v2.0 bug 2, observable consequence: building the top layer by rotating the
    bottom layer's in-cell sites and wrapping collapses distinct atoms onto
    identical positions.  Atom count still passes, because the collapse happens
    after enumeration.
    """
    c1, c2 = B.primitive_vectors(3.288)
    theta = B.commensurate_angle(M, N_IDX)
    T1, T2 = B.coincidence_vectors(c1, c2, M, N_IDX)

    sites = B.enumerate_layer_sites(c1, c2, T1, T2, EXPECTED_CELLS)
    rotated = sites @ B.rotation_matrix(theta).T

    cell = np.array([[T1[0], T1[1], 0], [T2[0], T2[1], 0], [0, 0, 30.0]])
    atoms = Atoms(
        symbols=["Mo"] * len(rotated),
        positions=np.column_stack((rotated, np.full(len(rotated), 15.0))),
        cell=cell,
        pbc=[True, True, False],
    )
    atoms.wrap()

    unique = np.unique(np.round(atoms.get_positions(), 5), axis=0)
    assert len(unique) < len(atoms), "expected collapsed atoms from the v2.0 route"


def test_ab_initio_route_produces_no_collapse():
    s = B.build_commensurate_moire(m=M, n=N_IDX)
    unique = np.unique(np.round(s.atoms.get_positions(), 5), axis=0)
    assert len(unique) == EXPECTED_ATOMS


def test_rotation_origin_is_a_lattice_site():
    """
    v2.0 bug 3: rotation about 0.5*(T1 + T2), which is not a lattice site.  Both
    layers must share the origin, so that AA (metal over metal) is well defined.
    """
    s = B.build_commensurate_moire(m=M, n=N_IDX)
    pos = s.atoms.get_positions()
    sym = np.array(s.atoms.get_chemical_symbols())
    cell = s.atoms.cell.array

    frac = np.linalg.solve(cell[:2, :2].T, pos[:, :2].T).T
    frac -= np.round(frac)
    at_corner = np.all(np.abs(frac) < 1e-6, axis=1)

    z = pos[at_corner, 2]
    assert (sym[at_corner] == "Mo").all()
    assert len(z) == 2, "expected exactly one metal per layer at the cell corner"
    assert np.isclose(abs(z[1] - z[0]), s.metadata.interlayer_d, atol=1e-6)


# =============================================================================
# GROUP 2 -- lattice periodicity (the decisive structural test)
# =============================================================================


def _layer_is_periodic(atoms: Atoms, mask: np.ndarray, v: np.ndarray) -> bool:
    """Shift one layer by its own primitive vector, wrap, compare the sets."""
    cell = atoms.cell.array
    pos = atoms.get_positions()[mask]

    def key(p):
        frac = np.linalg.solve(cell[:2, :2].T, p[:, :2].T).T % 1.0
        frac = np.round(frac, 6) % 1.0
        return np.array(sorted(map(tuple, np.column_stack((frac, np.round(p[:, 2], 6))))))

    shifted = pos.copy()
    shifted[:, :2] += v
    return np.allclose(key(pos), key(shifted), atol=1e-5)


def test_each_layer_is_periodic_under_the_supercell():
    """
    The decisive test.  Each layer, shifted by its OWN primitive vectors and
    wrapped, must map onto itself.  This is what fails for a non-coincidence
    cell, and what no atom-count check can see.
    """
    s = B.build_commensurate_moire(m=M, n=N_IDX)
    md = s.metadata
    c1, c2 = md.primitive_vectors
    R = B.rotation_matrix(md.theta_rad)
    d1, d2 = R @ c1, R @ c2

    z = s.atoms.get_positions()[:, 2]
    bottom = z < z.mean()
    top = ~bottom

    for v in (c1, c2):
        assert _layer_is_periodic(s.atoms, bottom, v)
    for v in (d1, d2):
        assert _layer_is_periodic(s.atoms, top, v)


# =============================================================================
# GROUP 3 -- geometry and crystallography
# =============================================================================


def test_counts_and_angle():
    s = B.build_commensurate_moire(m=M, n=N_IDX)
    assert s.metadata.n_cells == EXPECTED_CELLS
    assert s.metadata.n_atoms == EXPECTED_ATOMS
    assert np.isclose(s.metadata.theta_deg, EXPECTED_THETA_DEG, atol=1e-6)


def test_theta_atan2_beats_arccos_at_small_angle():
    """arccos loses roughly half the mantissa as theta -> 0."""
    m, n = 400, 399
    exact = B.commensurate_angle(m, n)
    N = B.n_cells(m, n)
    viacos = np.arccos((m * m + n * n + 4 * m * n) / (2 * N))
    rel = abs(exact - viacos) / exact
    # arccos is ~2e-11 off here, four orders of magnitude worse than the
    # ~1e-15 that atan2 delivers.
    assert rel > 1e-12


def test_supercell_is_rhombic():
    s = B.build_commensurate_moire(m=M, n=N_IDX)
    T1, T2 = s.metadata.supercell_vectors
    assert np.isclose(np.linalg.norm(T1), np.linalg.norm(T2))
    cos = T1 @ T2 / (np.linalg.norm(T1) * np.linalg.norm(T2))
    assert np.isclose(np.degrees(np.arccos(cos)), 60.0, atol=1e-9)


def test_six_equal_MX_bonds():
    """
    Trigonal-prismatic coordination: every metal has exactly six equal M-X
    bonds.  Chalcogens at (1/3, 2/3) and (2/3, 1/3) instead of a shared
    (1/3, 1/3) would break this.
    """
    from ase.neighborlist import neighbor_list

    s = B.build_commensurate_moire(m=M, n=N_IDX)
    sym = np.array(s.atoms.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", s.atoms, 2.9)

    is_mx = (sym[i] == "Mo") & (sym[j] == "Se")
    # 2 layers x 817 metals x 6 bonds, counted as directed Mo -> Se pairs
    assert is_mx.sum() == 6 * 2 * EXPECTED_CELLS
    assert np.allclose(d[is_mx], d[is_mx][0], atol=1e-9)
    assert 2.45 < d[is_mx][0] < 2.60


def test_in_plane_MX_distance_matches_analytic():
    a, h = 3.288, 1.664
    s = B.build_commensurate_moire(m=M, n=N_IDX, a=a, h_X=h)
    expected = np.hypot(a / np.sqrt(3.0), h)
    assert np.isclose(s.metadata.min_MX_bond, expected, atol=1e-9)


def test_max_deviation_is_machine_precision():
    s = B.build_commensurate_moire(m=M, n=N_IDX)
    assert s.metadata.max_deviation_from_ideal < 1e-9


def test_moire_period_matches_small_angle_formula():
    s = B.build_commensurate_moire(m=M, n=N_IDX)
    md = s.metadata
    assert np.isclose(md.moire_period, md.a_bottom * np.sqrt(md.n_cells), rtol=1e-12)
    assert np.isclose(md.moire_period, md.a_bottom / md.theta_rad, rtol=2e-3)


# =============================================================================
# GROUP 4 -- registry and heterostructure handling
# =============================================================================


def test_registry_shift_preserves_commensurability_and_counts():
    for shift in [(0.0, 0.0), (1 / 3, 1 / 3), (2 / 3, 2 / 3), (0.17, 0.41)]:
        s = B.build_commensurate_moire(m=M, n=N_IDX, registry_shift_frac=shift)
        assert s.metadata.n_atoms == EXPECTED_ATOMS
        assert s.metadata.max_deviation_from_ideal < 1e-9


def test_registry_shift_actually_moves_the_top_layer():
    s0 = B.build_commensurate_moire(m=M, n=N_IDX)
    s1 = B.build_commensurate_moire(m=M, n=N_IDX, registry_shift_frac=(1 / 3, 1 / 3))
    z = s0.atoms.get_positions()[:, 2]
    top = z > z.mean()
    d = s1.atoms.get_positions()[top] - s0.atoms.get_positions()[top]
    assert np.abs(d[:, :2]).max() > 0.5


def test_heterobilayer_reports_strain_and_true_period():
    s = B.build_commensurate_moire(
        symbol_M_bottom="Mo", symbol_X_bottom="Se",
        symbol_M_top="W", symbol_X_top="Se",
        a=3.288, a_top=3.282, m=M, n=N_IDX,
    )
    md = s.metadata
    assert abs(md.top_layer_strain) > 1e-4
    assert md.true_hetero_period is not None
    assert any("strained" in w for w in md.warnings)
    assert set(s.atoms.get_chemical_symbols()) == {"Mo", "W", "Se"}


def test_vacuum_is_centred():
    s = B.build_commensurate_moire(m=M, n=N_IDX, vacuum=18.0)
    z = s.atoms.get_positions()[:, 2]
    cz = s.atoms.cell.array[2, 2]
    assert np.isclose(z.min() + z.max(), cz, atol=1e-6)
    assert z.min() > 8.0


# =============================================================================
# GROUP 5 -- commensurate pair search
# =============================================================================


def test_find_commensurate_pair_recovers_the_known_cell():
    hits = B.find_commensurate_pair(2.004628, max_index=40, top=1)
    assert (hits[0]["m"], hits[0]["n"]) == (M, N_IDX)
    assert hits[0]["n_atoms"] == EXPECTED_ATOMS


def test_find_commensurate_pair_respects_atom_budget():
    hits = B.find_commensurate_pair(1.0, max_index=80, max_atoms=6000)
    assert all(h["n_atoms"] <= 6000 for h in hits)


def test_invalid_indices_rejected():
    with pytest.raises(ValueError):
        B.build_commensurate_moire(m=16, n=17)
