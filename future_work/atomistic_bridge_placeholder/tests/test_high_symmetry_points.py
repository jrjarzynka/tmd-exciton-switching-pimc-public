"""Regression test for the high-symmetry stacking points (fixed 2026-08-06).

BUG THIS GUARDS AGAINST
-------------------------
`HIGH_SYM_POINTS` and `high_symmetry_path` declared "AB" at fractional
coordinates (1/3, 2/3). That is not a high-symmetry stacking: folded into
the primitive cell it sits a/3 from AA, whereas every genuine non-AA
high-symmetry stacking of a hexagonal bilayer sits exactly a/sqrt(3) away.
The two correct ones are AB = (1/3, 1/3) and BA = (2/3, 2/3), related to
each other by inversion.

This is the *same wrong fractional coordinate* that had already been fixed
in the chalcogen basis of structure_relaxation.py and gsfe_geometry.py --
it survived here because the atomic basis and the stacking-point
definitions live in different places and nothing tied them together. The
test below states the invariant directly (distance from AA must be
a/sqrt(3), modulo the lattice), so any future hand-edited coordinate that
is not a high-symmetry site fails immediately rather than silently
producing DFT runs at a mislabelled stacking.

Consequence of the original bug, for the record: three relaxed-geometry
DFT points (AA / "bridge" / "AB") were computed at coordinates of which
only AA was genuinely high-symmetry, so any AB-derived quantity taken from
that set has to be recomputed.
"""
import numpy as np
import pytest

from dft_gsfe_scan.gsfe_geometry import avec, high_symmetry_path
from dft_gsfe_scan.run_gsfe_scan import HIGH_SYM_POINTS

A_NM = 3.285  # in-plane lattice constant used throughout (Angstrom here)


def _fold_distance_from_AA(u, v, a=A_NM, search=2):
    """Shortest distance from AA to the fractional point (u, v), minimised
    over lattice translations -- i.e. the distance within the primitive
    cell rather than the raw |u*a1 + v*a2|."""
    a1, a2 = avec(a)
    best = np.inf
    for i in range(-search, search + 1):
        for j in range(-search, search + 1):
            r = (u + i) * a1 + (v + j) * a2
            best = min(best, float(np.linalg.norm(r)))
    return best


def test_declared_high_symmetry_points_really_are_high_symmetry():
    """Every non-AA entry must sit exactly a/sqrt(3) from AA; AA itself at 0."""
    expected = A_NM / np.sqrt(3.0)
    for label, (u, v) in HIGH_SYM_POINTS.items():
        d = _fold_distance_from_AA(u, v)
        if label == "AA":
            assert d == pytest.approx(0.0, abs=1e-12), f"AA is not at the origin: {d}"
        else:
            assert d == pytest.approx(expected, rel=1e-9), (
                f"'{label}' at fractional ({u:.4f}, {v:.4f}) sits {d:.4f} A from AA, "
                f"but a genuine high-symmetry stacking must sit {expected:.4f} A "
                f"(= a/sqrt(3)) away. This is exactly the failure mode of the old "
                f"(1/3, 2/3) coordinate, which sits a/3 = {A_NM/3:.4f} A away."
            )


def test_the_old_wrong_coordinate_would_be_rejected():
    """Explicit negative control: (1/3, 2/3) must NOT pass the invariant
    above, so the test is actually capable of catching the original bug."""
    d = _fold_distance_from_AA(1.0 / 3.0, 2.0 / 3.0)
    assert d != pytest.approx(A_NM / np.sqrt(3.0), rel=1e-6)
    assert d == pytest.approx(A_NM / 3.0, rel=1e-9)


def test_AB_and_BA_are_distinct_and_inversion_related():
    """AB and BA are the two distinct non-AA stackings; each is the other's
    inverse modulo the lattice."""
    assert "AB" in HIGH_SYM_POINTS and "BA" in HIGH_SYM_POINTS
    ab = np.array(HIGH_SYM_POINTS["AB"])
    ba = np.array(HIGH_SYM_POINTS["BA"])
    assert not np.allclose(ab, ba)
    # AB + BA = (1, 1) = a lattice vector, i.e. BA == -AB modulo the lattice
    np.testing.assert_allclose(ab + ba, np.array([1.0, 1.0]), atol=1e-12)


def test_high_symmetry_path_endpoints_are_high_symmetry():
    """The path helper must be consistent with HIGH_SYM_POINTS: every named
    vertex it visits has to satisfy the same invariant."""
    coords, labels = high_symmetry_path(n_per_segment=3)
    expected = A_NM / np.sqrt(3.0)
    named = [(c, l) for c, l in zip(coords, labels) if "-" not in l]
    assert len(named) >= 3, "path should visit at least three named vertices"
    for (u, v), label in named:
        d = _fold_distance_from_AA(u, v)
        target = 0.0 if label == "AA" else expected
        assert d == pytest.approx(target, abs=1e-9), (
            f"path vertex '{label}' at ({u:.4f}, {v:.4f}) is {d:.4f} A from AA, "
            f"expected {target:.4f} A"
        )


def test_no_hand_picked_saddle_point_reintroduced():
    """The old 'bridge' entry at (1/3, 1/6) was a hand-picked guess and is
    not a high-symmetry point (0.441a from AA). Guard against it coming
    back: a saddle point should be located numerically from the fitted
    surface, not hard-coded."""
    assert "bridge" not in HIGH_SYM_POINTS
    _, labels = high_symmetry_path(n_per_segment=3)
    assert not any("bridge" in l for l in labels)
