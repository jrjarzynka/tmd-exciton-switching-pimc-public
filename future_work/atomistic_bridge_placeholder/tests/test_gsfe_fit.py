"""Validation of dft_gsfe_scan.gsfe_fit's fitting pipeline against known
synthetic data.

No pw.x is available to generate real DFT reference data for this
package's automated tests, so the fitting machinery itself is validated
here by fitting its own exact analytic three-cosine model back to itself
(must recover the injected amplitude to numerical precision) and by
checking that the "is a single harmonic enough" R^2/residual diagnostic
correctly degrades when higher-harmonic content is injected into the
synthetic data. This does NOT validate any real DFT physics -- it only
validates that the fit code is not silently wrong. A real GSFE DFT scan
is still required before trusting any fitted registry_depth_meV value for
production use in generate_potential_map().
"""
import numpy as np
import pytest

from dft_gsfe_scan.gsfe_geometry import registry_shift_grid, avec
from dft_gsfe_scan.gsfe_fit import fit_registry_depth, _raw_model, _recip_common

A_COMMON_A = 3.285


@pytest.fixture
def grid_uv():
    grid = registry_shift_grid(9)
    return grid[:, 0], grid[:, 1]


def test_pure_single_harmonic_is_recovered_exactly(grid_uv):
    u, v = grid_uv
    true_depth = 0.045
    true_E0 = -123.456
    raw = _raw_model(u, v, A_COMMON_A)
    E_synthetic = true_E0 + true_depth * raw

    E0_fit, depth_fit, rms, r2 = fit_registry_depth(u, v, E_synthetic, A_COMMON_A)

    assert depth_fit == pytest.approx(true_depth, abs=1e-10)
    assert E0_fit == pytest.approx(true_E0, abs=1e-8)
    assert rms < 1e-10
    assert r2 == pytest.approx(1.0, abs=1e-10)


def test_higher_harmonic_content_degrades_fit_but_leading_term_still_correct(grid_uv):
    """If the real DFT landscape needs more than one harmonic,
    registry_energy()'s current single-harmonic model would be an
    incomplete description -- the fit's R^2/residual must visibly signal
    this, while still correctly isolating the leading-harmonic amplitude
    (since the injected second harmonic is orthogonal to the first over
    this grid)."""
    u, v = grid_uv
    true_depth = 0.045
    true_E0 = -123.456
    raw = _raw_model(u, v, A_COMMON_A)
    E_pure = true_E0 + true_depth * raw

    B = _recip_common(A_COMMON_A)
    a1, a2 = avec(A_COMMON_A)
    delta = np.outer(u, a1) + np.outer(v, a2)
    higher_harmonic = 0.010 * np.cos(2.0 * (delta @ B[0]))
    E_with_overtone = E_pure + higher_harmonic

    E0_fit, depth_fit, rms, r2 = fit_registry_depth(u, v, E_with_overtone, A_COMMON_A)

    assert depth_fit == pytest.approx(true_depth, abs=1e-6)
    assert rms > 1e-3, "residual should visibly increase once overtone content is present"
    assert r2 < 0.999, "R^2 should visibly degrade once overtone content is present"


def test_pure_noise_gives_near_zero_depth_and_poor_r2(grid_uv):
    """Sanity check at the other extreme: fitting pure noise must not
    spuriously report a confident nonzero depth or a good R^2."""
    rng = np.random.default_rng(0)
    u, v = grid_uv
    E_noise = rng.normal(0.0, 0.02, size=len(u))
    _, depth_fit, _, r2 = fit_registry_depth(u, v, E_noise, A_COMMON_A)
    assert abs(depth_fit) < 0.05
    assert r2 < 0.5
