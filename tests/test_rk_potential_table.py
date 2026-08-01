"""Independent unit tests for tmd_pimc.rk_potential.

These tests do not touch the PIMC sampler, action, or radial solver. They
validate rk_potential.py against itself and against hand-derived closed-form
checks, so that a future change to thresholds, asymptotics, or the table
builder is caught immediately rather than only showing up as a mysterious
PIMC-vs-reference discrepancy downstream.

What is checked
----------------
1. The three evaluation branches (small-x analytic asymptote, middle
   Struve/Bessel, large-x Coulomb tail) agree with each other across the
   switch points x=small_x_threshold and x=large_x_threshold. A future
   change to either threshold that breaks continuity will fail this test.
2. The small-r asymptotic coefficient A = C/(kappa*r0) matches the
   definition used in the module docstring and README:
   V(r) ~ A*[ln(r/2r0) + gamma - r/r0].
3. The large-r branch converges to the Coulomb tail -C/(kappa*r).
4. RytovaKeldyshTablePotential.radial_value (the fast interpolation path
   actually used during PIMC) agrees with the direct special-function
   evaluator rk_energy_direct_eV, both inside the tabulated range and in
   the below-r_min / above-r_max analytic-extrapolation regions.
5. RytovaKeldyshTablePotential.value / RytovaKeldyshWallPotential.wall_value
   handle both a single (2,) coordinate and an (N,2) batch consistently.
6. Parameter validation actually rejects non-physical inputs.

Tolerances below were set with margin above values observed when this file
was written (kappa=4.5, r0=5.0 nm, default table settings): branch-boundary
continuity ~2e-6 relative, table interpolation ~5.6e-6 (inside) / ~6e-5
(above r_max) relative, well under the 1e-3 production gate used in
run_relative_rk_validation.py's prepare_table().
"""

from __future__ import annotations

import numpy as np
import pytest

from tmd_pimc.rk_potential import (
    COULOMB_CONSTANT_EV_NM,
    EULER_GAMMA,
    RytovaKeldyshWallPotential,
    build_rk_table,
    rk_energy_direct_eV,
    rk_short_distance_coefficient_eV,
    rk_table_diagnostics,
)

KAPPA = 4.5
R0_NM = 5.0


def _build_table():
    return build_rk_table(
        kappa=KAPPA,
        screening_length_nm=R0_NM,
        r_max_nm=80.0,
        n_log=4000,
        n_linear=4000,
    )


# ---------------------------------------------------------------------------
# Branch continuity and asymptotics
# ---------------------------------------------------------------------------

def test_small_x_threshold_is_continuous():
    """Small-branch analytic asymptote must match the middle Struve/Bessel
    branch just below/above x = small_x_threshold (default 1e-4)."""
    r_threshold = 1.0e-4 * R0_NM
    r_below = r_threshold * (1.0 - 1.0e-6)
    r_above = r_threshold * (1.0 + 1.0e-6)
    v_below = rk_energy_direct_eV(r_below, kappa=KAPPA, screening_length_nm=R0_NM)
    v_above = rk_energy_direct_eV(r_above, kappa=KAPPA, screening_length_nm=R0_NM)
    rel_err = abs(v_below - v_above) / abs(v_below)
    assert rel_err < 1.0e-4, f"small-x branch discontinuity: {rel_err:.3e}"


def test_large_x_threshold_is_continuous():
    """Middle Struve/Bessel branch must match the Coulomb-tail asymptote
    just below/above x = large_x_threshold (default 500), despite the
    cancellation between H0 and Y0 that occurs in that regime."""
    r_threshold = 500.0 * R0_NM
    r_below = r_threshold * (1.0 - 1.0e-6)
    r_above = r_threshold * (1.0 + 1.0e-6)
    v_below = rk_energy_direct_eV(r_below, kappa=KAPPA, screening_length_nm=R0_NM)
    v_above = rk_energy_direct_eV(r_above, kappa=KAPPA, screening_length_nm=R0_NM)
    rel_err = abs(v_below - v_above) / abs(v_below)
    assert rel_err < 1.0e-4, f"large-x branch discontinuity: {rel_err:.3e}"


def test_short_distance_coefficient_matches_definition():
    """A = C/(kappa*r0), as stated in the module docstring."""
    A = rk_short_distance_coefficient_eV(
        kappa=KAPPA, screening_length_nm=R0_NM,
        coulomb_constant_eV_nm=COULOMB_CONSTANT_EV_NM,
    )
    expected = COULOMB_CONSTANT_EV_NM / (KAPPA * R0_NM)
    assert A == pytest.approx(expected, rel=1.0e-12)


def test_small_r_matches_hand_derived_log_asymptote():
    """V(r) ~ A*[ln(r/2r0) + gamma - r/r0] deep in the small-x regime,
    independently re-derived from the known small-x expansions
    H0(x)~(2/pi)x, Y0(x)~(2/pi)[ln(x/2)+gamma]."""
    A = rk_short_distance_coefficient_eV(kappa=KAPPA, screening_length_nm=R0_NM)
    r_test = 1.0e-8  # x = 2e-9, deep in the small branch
    x = r_test / R0_NM
    expected = A * (np.log(x / 2.0) + EULER_GAMMA - x)
    value = rk_energy_direct_eV(r_test, kappa=KAPPA, screening_length_nm=R0_NM)
    assert value == pytest.approx(expected, rel=1.0e-10)


def test_large_r_converges_to_coulomb_tail():
    """V(r) -> -C/(kappa*r) far outside the screening length."""
    r_test = 1.0e5  # x = 2e4, deep in the large branch
    value = rk_energy_direct_eV(r_test, kappa=KAPPA, screening_length_nm=R0_NM)
    coulomb = -COULOMB_CONSTANT_EV_NM / (KAPPA * r_test)
    assert value == pytest.approx(coulomb, rel=1.0e-8)


def test_potential_is_attractive_and_monotonic_with_kappa():
    """Physical sanity: V < 0 everywhere, and screening a larger environment
    dielectric constant should weaken (raise towards zero) the potential."""
    r_test = np.geomspace(1.0e-3, 50.0, 200)
    v_weak = rk_energy_direct_eV(r_test, kappa=2.0, screening_length_nm=R0_NM)
    v_strong = rk_energy_direct_eV(r_test, kappa=8.0, screening_length_nm=R0_NM)
    assert np.all(v_weak < 0.0)
    assert np.all(v_strong < 0.0)
    # Larger kappa screens more -> potential closer to zero (less negative).
    assert np.all(v_strong > v_weak)


# ---------------------------------------------------------------------------
# Table construction and interpolation accuracy
# ---------------------------------------------------------------------------

def test_table_diagnostics_below_production_gate():
    """Matches the 1e-3 relative-error gate enforced in
    run_relative_rk_validation.py's prepare_table()."""
    table = _build_table()
    diagnostics = rk_table_diagnostics(table, n_test=20000)
    assert diagnostics["max_relative_interpolation_error"] < 1.0e-3


def test_table_matches_direct_evaluator_inside_range():
    table = _build_table()
    rng = np.random.default_rng(0)
    r_inside = rng.uniform(table.r_min_nm * 1.1, table.r_max_nm * 0.9, 200)
    direct = rk_energy_direct_eV(r_inside, kappa=KAPPA, screening_length_nm=R0_NM)
    interpolated = table.radial_value(r_inside)
    rel_err = np.abs(interpolated - direct) / np.maximum(np.abs(direct), 1.0e-14)
    assert np.max(rel_err) < 1.0e-4


def test_table_matches_direct_evaluator_below_r_min():
    """Below r_min the table falls back to the analytic small-r branch;
    this should match rk_energy_direct_eV exactly (same formula)."""
    table = _build_table()
    rng = np.random.default_rng(1)
    r_below = rng.uniform(1.0e-9, table.r_min_nm * 0.9, 50)
    direct = rk_energy_direct_eV(r_below, kappa=KAPPA, screening_length_nm=R0_NM)
    interpolated = table.radial_value(r_below)
    rel_err = np.abs(interpolated - direct) / np.maximum(np.abs(direct), 1.0e-14)
    assert np.max(rel_err) < 1.0e-8


def test_table_matches_direct_evaluator_above_r_max():
    """Above r_max the table falls back to the analytic Coulomb tail."""
    table = _build_table()
    rng = np.random.default_rng(2)
    r_above = rng.uniform(table.r_max_nm * 1.1, table.r_max_nm * 100.0, 50)
    direct = rk_energy_direct_eV(r_above, kappa=KAPPA, screening_length_nm=R0_NM)
    interpolated = table.radial_value(r_above)
    rel_err = np.abs(interpolated - direct) / np.maximum(np.abs(direct), 1.0e-14)
    assert np.max(rel_err) < 1.0e-3


# ---------------------------------------------------------------------------
# Coordinate-shape handling (value() as used by the sampler/action)
# ---------------------------------------------------------------------------

def test_value_single_point_matches_radial_value():
    table = _build_table()
    point = np.array([3.0, 4.0])  # r = 5.0
    r = np.hypot(3.0, 4.0)
    expected = rk_energy_direct_eV(r, kappa=KAPPA, screening_length_nm=R0_NM)
    assert table.value(point) == pytest.approx(expected, rel=1.0e-4)


def test_value_batch_matches_radial_value():
    table = _build_table()
    points = np.array([[3.0, 4.0], [0.0, 1.0], [10.0, 0.0]])
    radii = np.hypot(points[:, 0], points[:, 1])
    expected = rk_energy_direct_eV(radii, kappa=KAPPA, screening_length_nm=R0_NM)
    values = table.value(points)
    np.testing.assert_allclose(values, expected, rtol=1.0e-4)


def test_value_rejects_malformed_shapes():
    table = _build_table()
    with pytest.raises(ValueError):
        table.value(np.array([1.0, 2.0, 3.0]))  # wrong (3,) vector
    with pytest.raises(ValueError):
        table.value(np.zeros((5, 3)))  # wrong second dimension


# ---------------------------------------------------------------------------
# Wall potential
# ---------------------------------------------------------------------------

def test_wall_potential_shape_and_values():
    table = _build_table()
    wall = RytovaKeldyshWallPotential(
        rk=table, wall_radius_nm=25.0, wall_height_eV=1.0, wall_power=8
    )
    assert wall.wall_value(np.array([0.0, 0.0])) == pytest.approx(0.0)
    assert wall.wall_value(np.array([25.0, 0.0])) == pytest.approx(1.0)
    # (r/R)^8 at r=2R should be 2^8 = 256 times the wall height.
    assert wall.wall_value(np.array([50.0, 0.0])) == pytest.approx(256.0)


def test_wall_negligible_inside_bound_state_region():
    """Sanity check on the default validation wall_radius_nm=25 relative to
    the default r0=5 nm screening length: at r=r0 the wall must be utterly
    negligible compared to a typical RK interaction energy, so that it does
    not distort the bound-state physics it is meant to merely regularise."""
    table = _build_table()
    wall = RytovaKeldyshWallPotential(
        rk=table, wall_radius_nm=25.0, wall_height_eV=1.0, wall_power=8
    )
    point = np.array([R0_NM, 0.0])
    v_wall = wall.wall_value(point)
    v_rk = abs(wall.central_value(point))
    # Observed ratio for the default validation parameters is ~5.3e-5;
    # 1e-3 leaves comfortable margin while still catching a badly chosen
    # wall_radius_nm/wall_power that would contaminate the bound state.
    assert v_wall < 1.0e-3 * v_rk


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kappa=-1.0, screening_length_nm=R0_NM),
        dict(kappa=0.0, screening_length_nm=R0_NM),
        dict(kappa=KAPPA, screening_length_nm=0.0),
        dict(kappa=KAPPA, screening_length_nm=-2.0),
        dict(kappa=float("nan"), screening_length_nm=R0_NM),
    ],
)
def test_build_rk_table_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        build_rk_table(r_max_nm=80.0, **kwargs)


def test_build_hybrid_grid_requires_ordered_bounds():
    from tmd_pimc.rk_potential import build_hybrid_radial_grid_nm

    with pytest.raises(ValueError):
        build_hybrid_radial_grid_nm(
            r_min_nm=1.0, log_switch_nm=0.5, r_max_nm=10.0, n_log=100, n_linear=100
        )
