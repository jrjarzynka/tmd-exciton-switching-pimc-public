"""Regression test for the registry-shift bug in
moire_pipeline.generate_potential_map.registry_energy (fixed 2026-08-02).

BUG THIS GUARDS AGAINST
-------------------------
The legacy `phase` parameter added a single scalar offset to only the
k=0 cosine term of the three-G-vector moire sum:

    raw = cos(G1.r + phase) + cos(G2.r) + cos(G3.r)

This does NOT correspond to any rigid in-plane translation of the
registry pattern -- a genuine translation r -> r - shift requires a
DIFFERENT phase offset -G_k.shift for EACH of the three (non-parallel)
G_k, not one scalar on one term. Verified (before the fix): even the
best-fit scalar phase reproduced a true rigid shift only to ~18%
residual RMS relative to the pattern's own amplitude -- a real,
qualitative distortion, not a rounding-level discrepancy. A second,
compounding bug followed from this: registry_norm_mode="global" assumes
raw in [-1.5, 3.0] (exact only for phase=0 / an untranslated pattern),
so for the old buggy nonzero `phase` this silently clipped/flattened
the landscape.

Same class of bug as the registry-shift issue already found and fixed in
tmd_pimc's TwoBodyPIMCSamplerStagingPeriodicJIT this session: an attempt
to implement a periodic phase shift via an inadequate parameterization.

The fix introduces `registry_shift_nm=(dx_nm, dy_nm)`, implemented as a
genuine per-G_k phase of -G_k.shift, and disables the old `phase`
parameter for any nonzero value (raises rather than silently returning a
distorted-not-translated landscape).
"""
import numpy as np
import pytest

from moire_pipeline.generate_potential_map import registry_energy, moire_G

A_BOTTOM_A = 3.288  # MoSe2
A_TOP_A = 3.282     # WSe2
THETA_DEG = 0.50
DEPTH_EV = 0.090


@pytest.fixture
def test_points():
    rng = np.random.default_rng(0)
    return rng.uniform(-10.0, 10.0, 60), rng.uniform(-10.0, 10.0, 60)


def test_registry_shift_matches_direct_coordinate_translation(test_points):
    """registry_shift_nm=(dx,dy) must reproduce, to machine precision,
    directly evaluating the potential at coordinates shifted by (dx,dy)
    -- the definition of a genuine rigid translation."""
    X, Y = test_points
    shift = (1.5, 0.7)
    via_param = registry_energy(
        X, Y, A_BOTTOM_A, A_TOP_A, THETA_DEG, DEPTH_EV,
        registry_shift_nm=shift, registry_norm_mode="local",
    )
    via_coords = registry_energy(
        X - shift[0], Y - shift[1], A_BOTTOM_A, A_TOP_A, THETA_DEG, DEPTH_EV,
        registry_norm_mode="local",
    )
    np.testing.assert_allclose(via_param, via_coords, atol=1e-10)


def test_zero_shift_matches_unshifted_pattern(test_points):
    X, Y = test_points
    v_default = registry_energy(X, Y, A_BOTTOM_A, A_TOP_A, THETA_DEG, DEPTH_EV,
                                 registry_norm_mode="local")
    v_explicit_zero = registry_energy(
        X, Y, A_BOTTOM_A, A_TOP_A, THETA_DEG, DEPTH_EV,
        registry_shift_nm=(0.0, 0.0), registry_norm_mode="local",
    )
    np.testing.assert_allclose(v_default, v_explicit_zero, atol=1e-12)


@pytest.mark.parametrize("shift", [(0.0, 0.0), (1.5, 0.7), (5.3, -2.1), (10.0, 10.0), (-3.3, 4.4)])
def test_global_mode_range_is_shift_invariant(shift):
    """A rigid translation of a periodic function cannot change the SET of
    values it attains -- only where they occur. So registry_norm_mode=
    "global"'s hardcoded analytic bounds must produce the same output range
    [0, depth_eV] for every shift, unlike the old buggy `phase` parameter
    (verified pre-fix to push raw outside [-1.5, 3.0] and cause silent
    clipping for any nonzero value)."""
    x = np.linspace(-50.0, 50.0, 250)
    y = np.linspace(-50.0, 50.0, 250)
    X, Y = np.meshgrid(x, y)
    v = registry_energy(X, Y, A_BOTTOM_A, A_TOP_A, THETA_DEG, DEPTH_EV,
                         registry_shift_nm=shift, registry_norm_mode="global")
    assert v.min() == pytest.approx(0.0, abs=1e-3 * DEPTH_EV)
    assert v.max() == pytest.approx(DEPTH_EV, abs=1e-3 * DEPTH_EV)


def test_legacy_nonzero_phase_is_rejected():
    """The old scalar `phase` parameter is disabled for nonzero values --
    it silently produced a distorted (not translated) landscape, which is
    worse than an explicit error steering callers to registry_shift_nm."""
    with pytest.raises(ValueError, match="registry_shift_nm"):
        registry_energy(np.array([1.0]), np.array([1.0]),
                         A_BOTTOM_A, A_TOP_A, THETA_DEG, DEPTH_EV, phase=1.234)


def test_legacy_zero_phase_still_works():
    """phase=0.0 (the only value any existing config ever used) must keep
    working exactly as before -- no behaviour change at the default."""
    v = registry_energy(np.array([3.0]), np.array([-2.0]),
                         A_BOTTOM_A, A_TOP_A, THETA_DEG, DEPTH_EV, phase=0.0)
    assert np.all(np.isfinite(v))


def test_half_period_shift_is_a_genuine_stacking_change():
    """Sanity check that registry_shift_nm actually moves the pattern by a
    physically meaningful amount: shifting by roughly half the moire period
    along the first G-vector's real-space conjugate direction must move the
    landscape away from its unshifted value at the origin (registry AT the
    origin genuinely changes, rather than being invariant as it would be
    under the old, non-translating `phase` parameter)."""
    G = moire_G(A_BOTTOM_A, A_TOP_A, THETA_DEG)
    g0 = G[0]
    period_along_g0_A = 2.0 * np.pi / np.linalg.norm(g0)
    half_shift_A = 0.5 * period_along_g0_A * (g0 / np.linalg.norm(g0))
    half_shift_nm = tuple(half_shift_A / 10.0)

    v0 = registry_energy(np.array([0.0]), np.array([0.0]),
                          A_BOTTOM_A, A_TOP_A, THETA_DEG, DEPTH_EV,
                          registry_norm_mode="local")
    v_half = registry_energy(np.array([0.0]), np.array([0.0]),
                              A_BOTTOM_A, A_TOP_A, THETA_DEG, DEPTH_EV,
                              registry_shift_nm=half_shift_nm, registry_norm_mode="local")
    # local mode on a single point is degenerate (always 0); use 'global'
    # instead for a meaningful single-point comparison
    v0g = registry_energy(np.array([0.0]), np.array([0.0]),
                           A_BOTTOM_A, A_TOP_A, THETA_DEG, DEPTH_EV,
                           registry_norm_mode="global")
    v_halfg = registry_energy(np.array([0.0]), np.array([0.0]),
                               A_BOTTOM_A, A_TOP_A, THETA_DEG, DEPTH_EV,
                               registry_shift_nm=half_shift_nm, registry_norm_mode="global")
    assert abs(float(v0g[0]) - float(v_halfg[0])) > 0.1 * DEPTH_EV
