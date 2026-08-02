"""Regression test for the registry-shift bug in
TwoBodyPIMCSamplerStagingPeriodicJIT (fixed 2026-08-02).

BUG THAT THIS GUARDS AGAINST
-----------------------------
origin_e_nm / origin_h_nm used to be forwarded directly into
build_periodic_cell_grid's `origin_nm` kwarg, which only relocates the
rasterization *window*. Because MoirePotential is exactly periodic under
the same lattice vectors used for the periodic wrap-and-lookup, moving the
rasterization window and then wrapping the query point through that same
window cancels out completely -- the electron and hole ended up sampling
the IDENTICAL, unshifted moire registry regardless of origin_h_nm. Only
the OutOfPlaneStarkPotential anchor (which bakes its shift into value(r)
directly, not via a rasterization window) was ever actually shifted.

This silently invalidated any "registry offset" scan run through this
sampler (both the zero-field equilibrium scan and the Fz-driven
dissociation scan in the two-body manuscript).

The fix bakes the shift into the potential itself via ShiftedPotential
(inner.value(r - shift), a genuine query-point shift) before rasterizing.
These tests check:
  1. A nonzero origin_h_nm now produces a landscape that actually differs
     from the unshifted one, and matches the analytic ShiftedPotential
     reference to interpolation-grid accuracy.
  2. origin_e_nm=origin_h_nm=(0,0) (the common/default case) is unaffected
     and matches the bare, unshifted MoirePotential.
  3. With Fz != 0, the Stark phase is not double-shifted: the hole grid
     equals [registry-shifted moire] + [Stark term anchored at the shift],
     evaluated independently.
"""
import numpy as np
import pytest

from tmd_pimc.two_body_action import TwoBodyRingPolymerAction
from tmd_pimc.two_body_sampler_periodic_jit import TwoBodyPIMCSamplerStagingPeriodicJIT
from tmd_pimc.potentials import CompositePotential, MoirePotential, OutOfPlaneStarkPotential
from tmd_pimc.potential_helpers import ShiftedPotential
from tmd_pimc.kernels_jit import bilinear_interpolate_periodic_cell

PERIOD_NM = 20.0
AMPLITUDE_EV = 0.02
SHIFT_NM = (5.0, 0.0)
GRID_SIZE = 200
# Bilinear interpolation on a 200x200 rasterized cell of a ~0.02 eV-scale
# potential; this is comfortably above interpolation noise (~1e-5 eV) but
# far below the ~0.02-0.04 eV scale of a genuine shift/no-shift difference.
ATOL_EV = 5.0e-4


class _ZeroInteraction:
    """Interaction-potential stand-in; only .value() is required by the action."""

    def value(self, r):
        r = np.asarray(r)
        return np.zeros(r.shape[0])


def _make_action(n_beads=16):
    return TwoBodyRingPolymerAction(
        mass_e_m0=0.5,
        mass_h_m0=0.5,
        temperature_K=10.0,
        n_beads=n_beads,
        potential_e=CompositePotential(terms=[]),
        potential_h=CompositePotential(terms=[]),
        potential_interaction=_ZeroInteraction(),
    )


def _sample_grid(grid, pts):
    return np.array([
        bilinear_interpolate_periodic_cell(
            x, y, grid["v_grid"], grid["origin_x"], grid["origin_y"],
            grid["ainv00"], grid["ainv01"], grid["ainv10"], grid["ainv11"],
        )
        for x, y in pts
    ])


@pytest.fixture
def test_points():
    rng = np.random.default_rng(0)
    return rng.uniform(-30.0, 30.0, size=(24, 2))


def test_registry_shift_actually_changes_hole_landscape(test_points):
    """A nonzero origin_h_nm must produce a genuinely different landscape
    from the unshifted case -- this is the core regression check."""
    action = _make_action()
    s_shift = TwoBodyPIMCSamplerStagingPeriodicJIT(
        action=action, moire_period_nm=PERIOD_NM, moire_amplitude_eV=AMPLITUDE_EV,
        origin_e_nm=(0.0, 0.0), origin_h_nm=SHIFT_NM, periodic_cell_grid_size=GRID_SIZE,
    )
    s_noshift = TwoBodyPIMCSamplerStagingPeriodicJIT(
        action=action, moire_period_nm=PERIOD_NM, moire_amplitude_eV=AMPLITUDE_EV,
        origin_e_nm=(0.0, 0.0), origin_h_nm=(0.0, 0.0), periodic_cell_grid_size=GRID_SIZE,
    )
    v_shift = _sample_grid(s_shift._grid_h, test_points)
    v_noshift = _sample_grid(s_noshift._grid_h, test_points)

    max_diff = np.max(np.abs(v_shift - v_noshift))
    assert max_diff > 0.5 * AMPLITUDE_EV, (
        f"origin_h_nm={SHIFT_NM} produced almost no change in the hole "
        f"landscape (max diff {max_diff:.2e} eV) -- registry shift is not "
        f"being applied (this is the bug this test guards against)."
    )


def test_shifted_grid_matches_analytic_reference(test_points):
    """grid_h under origin_h_nm=SHIFT_NM must reproduce ShiftedPotential(
    MoirePotential, shift_nm=SHIFT_NM) to interpolation accuracy."""
    action = _make_action()
    sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(
        action=action, moire_period_nm=PERIOD_NM, moire_amplitude_eV=AMPLITUDE_EV,
        origin_e_nm=(0.0, 0.0), origin_h_nm=SHIFT_NM, periodic_cell_grid_size=GRID_SIZE,
    )
    bare = MoirePotential(amplitude_eV=AMPLITUDE_EV, period_nm=PERIOD_NM)
    expected = ShiftedPotential(inner=bare, shift_nm=SHIFT_NM).value(test_points)
    got = _sample_grid(sampler._grid_h, test_points)
    np.testing.assert_allclose(got, expected, atol=ATOL_EV)


def test_zero_shift_matches_bare_moire(test_points):
    """origin_e_nm=origin_h_nm=(0,0) (the common default) must reproduce
    the bare, unshifted MoirePotential -- no accidental behaviour change
    for the already-published shift=0 reference points."""
    action = _make_action()
    sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(
        action=action, moire_period_nm=PERIOD_NM, moire_amplitude_eV=AMPLITUDE_EV,
        origin_e_nm=(0.0, 0.0), origin_h_nm=(0.0, 0.0), periodic_cell_grid_size=GRID_SIZE,
    )
    bare = MoirePotential(amplitude_eV=AMPLITUDE_EV, period_nm=PERIOD_NM)
    expected = bare.value(test_points)
    got_e = _sample_grid(sampler._grid_e, test_points)
    got_h = _sample_grid(sampler._grid_h, test_points)
    np.testing.assert_allclose(got_e, expected, atol=ATOL_EV)
    np.testing.assert_allclose(got_h, expected, atol=ATOL_EV)


def test_stark_phase_not_double_shifted(test_points):
    """With Fz != 0, grid_h must equal [registry-shifted moire] +
    [Stark term anchored at the same shift] -- not a double-shifted
    version of either term."""
    Fz = 0.5
    dipole_length_nm = 0.05
    action = _make_action()
    sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(
        action=action, moire_period_nm=PERIOD_NM, moire_amplitude_eV=AMPLITUDE_EV,
        origin_e_nm=(0.0, 0.0), origin_h_nm=SHIFT_NM,
        Fz_eV_per_nm=Fz, dipole_length_nm=dipole_length_nm,
        periodic_cell_grid_size=GRID_SIZE,
    )
    bare = MoirePotential(amplitude_eV=AMPLITUDE_EV, period_nm=PERIOD_NM)
    stark_h = OutOfPlaneStarkPotential(
        Fz_eV_per_nm=+Fz, dipole_length_nm=dipole_length_nm,
        period_nm=PERIOD_NM, anchor_nm=SHIFT_NM,
    )
    expected = (
        ShiftedPotential(inner=bare, shift_nm=SHIFT_NM).value(test_points)
        + stark_h.value(test_points)
    )
    got = _sample_grid(sampler._grid_h, test_points)
    np.testing.assert_allclose(got, expected, atol=ATOL_EV)
