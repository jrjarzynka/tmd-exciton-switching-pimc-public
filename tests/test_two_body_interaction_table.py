"""Validation of the interaction lookup table shared by both JIT two-body
kernels (two_body_kernels_jit.py: build_uniform_interaction_table,
radial_table_interpolate -- imported unchanged into
two_body_kernels_periodic_jit.py too, so this covers both backends).

WHY THIS TEST EXISTS
---------------------
Every JIT two-body run resamples the (already-validated, see
test_bilayer_keldysh_potential.py) BilayerKeldyshWallPotential onto a
dense UNIFORM 1D radial grid once at construction time, then does O(1)
linear interpolation on that table inside the hot MC loop instead of
calling potential.value() directly. This table + interpolation step is
itself untested code, run millions of times per production scan, and
sits exactly where reviewer-flagged concerns about the interaction's
short-range "soft wall" (BilayerKeldyshWallPotential's repulsive
wall_power=8 core) would show up as accuracy loss if the grid were too
coarse to resolve the steep curvature there.
"""
import numpy as np
import pytest

from tmd_pimc.bilayer_keldysh_potential import (
    build_bilayer_keldysh_table,
    BilayerKeldyshWallPotential,
)
from tmd_pimc.two_body_kernels_jit import (
    build_uniform_interaction_table,
    radial_table_interpolate,
)

# Production-like parameters (same shape as adaptive_dense_field_scan.py /
# run_two_body_validation.py configs).
SEPARATION_NM = 0.65
SCREEN_1_NM = 4.0
SCREEN_2_NM = 4.5
KAPPA_ENV = 4.0
WALL_RADIUS_NM = 15.0
WALL_HEIGHT_EV = 0.08
WALL_POWER = 8
R_MAX_NM = 80.0
N_POINTS = 20_000

# kT at a typical cryogenic run temperature (~20 K) -- the scale table
# error must be negligible against for MC results to be trustworthy.
KT_SCALE_EV = 8.617e-5 * 20.0  # ~1.7e-3 eV


@pytest.fixture(scope="module")
def interaction():
    table = build_bilayer_keldysh_table(
        separation_nm=SEPARATION_NM,
        screening_length_layer1_nm=SCREEN_1_NM,
        screening_length_layer2_nm=SCREEN_2_NM,
        kappa_environment=KAPPA_ENV,
        r_max_nm=R_MAX_NM, n_log=1000, n_linear=2000,
    )
    return BilayerKeldyshWallPotential(
        bilayer=table, wall_radius_nm=WALL_RADIUS_NM,
        wall_height_eV=WALL_HEIGHT_EV, wall_power=WALL_POWER,
    )


@pytest.fixture(scope="module")
def uniform_table(interaction):
    return build_uniform_interaction_table(interaction, r_max_nm=R_MAX_NM, n_points=N_POINTS)


def _direct(interaction, r):
    return float(interaction.value(np.array([[r, 0.0]]))[0])


def test_table_matches_direct_evaluation_across_full_range(interaction, uniform_table):
    """Interpolated table values must match direct potential.value() calls
    to well below kT in the physically relevant range (origin through the
    repulsive wall and a bit beyond -- r up to ~25 nm here), and to a tight
    RELATIVE tolerance further out where the wall value itself is huge
    (r ~ 70-80 nm, values ~1e4 eV) -- an absolute-only criterion is the
    wrong yardstick there, since a relatively tiny (and harmless) fractional
    error translates into a large-looking absolute number."""
    v_table, r_min, r_max = uniform_table
    rng = np.random.default_rng(0)

    near_r = np.concatenate([
        [0.0, 1e-4, 1e-3, 1e-2],                                    # origin region
        rng.uniform(0.05, 5.0, 25),                                  # bound-state bulk
        rng.uniform(WALL_RADIUS_NM - 3.0, WALL_RADIUS_NM + 10.0, 25),  # wall + just beyond
    ])
    for r in near_r:
        direct = _direct(interaction, float(r))
        table_val = radial_table_interpolate(float(r), v_table, r_min, r_max)
        assert table_val == pytest.approx(direct, abs=1.0e-3 * KT_SCALE_EV), (
            f"r={r:.5f} nm: table={table_val:.6f} vs direct={direct:.6f}"
        )

    far_r = rng.uniform(30.0, R_MAX_NM - 1.0, 15)  # deep in the saturated wall
    for r in far_r:
        direct = _direct(interaction, float(r))
        table_val = radial_table_interpolate(float(r), v_table, r_min, r_max)
        assert table_val == pytest.approx(direct, rel=1e-4), (
            f"r={r:.4f} nm (deep wall): table={table_val:.6f} vs direct={direct:.6f}"
        )


def test_table_resolves_the_repulsive_wall_specifically(interaction, uniform_table):
    """Focused check right at and inside the wall radius, where curvature
    (wall_power=8) is steepest and a too-coarse grid would show up first."""
    v_table, r_min, r_max = uniform_table
    test_r = np.linspace(WALL_RADIUS_NM - 1.0, WALL_RADIUS_NM + 1.0, 41)
    for r in test_r:
        direct = _direct(interaction, float(r))
        table_val = radial_table_interpolate(float(r), v_table, r_min, r_max)
        # Absolute tolerance scaled to the local value: near the wall values
        # themselves are O(0.01-1 eV), so a tight relative check is more
        # informative there than a flat absolute one.
        assert table_val == pytest.approx(direct, abs=1e-4, rel=1e-4), (
            f"r={r:.4f} nm: table={table_val:.6f} vs direct={direct:.6f}"
        )


def test_convergence_with_grid_density(interaction):
    """Error must shrink as n_points increases -- confirms the discretization
    error is well-behaved (not e.g. dominated by some fixed bug) and that
    the production default (20000) sits well past the point of diminishing
    returns."""
    rng = np.random.default_rng(1)
    probe_r = rng.uniform(0.0, R_MAX_NM, 15)

    def max_err(n_points):
        v_table, r_min, r_max = build_uniform_interaction_table(
            interaction, r_max_nm=R_MAX_NM, n_points=n_points
        )
        errs = [
            abs(_direct(interaction, r) - radial_table_interpolate(float(r), v_table, r_min, r_max))
            for r in probe_r
        ]
        return max(errs)

    err_coarse = max_err(2_000)
    err_fine = max_err(20_000)
    assert err_fine < err_coarse, (
        f"error did not shrink with finer grid: coarse={err_coarse:.3e}, "
        f"fine={err_fine:.3e}"
    )


def test_edge_clamping_behaviour(interaction, uniform_table):
    """Below r_min clamps to table[0]; at/above r_max clamps to table[-1]
    (the already very large, wall-dominated edge value) -- per the
    documented, intentional design (see radial_table_interpolate
    docstring), not an artificial hard cutoff."""
    v_table, r_min, r_max = uniform_table
    assert radial_table_interpolate(-1.0, v_table, r_min, r_max) == pytest.approx(v_table[0])
    assert radial_table_interpolate(r_min, v_table, r_min, r_max) == pytest.approx(v_table[0])
    assert radial_table_interpolate(r_max, v_table, r_min, r_max) == pytest.approx(v_table[-1])
    assert radial_table_interpolate(r_max + 50.0, v_table, r_min, r_max) == pytest.approx(v_table[-1])
    # the r_max edge value must itself be large/repulsive (wall-dominated),
    # confirming clamping never silently hides a walker that escaped the
    # confined region rather than genuinely dissociating.
    assert v_table[-1] > 100.0 * WALL_HEIGHT_EV
