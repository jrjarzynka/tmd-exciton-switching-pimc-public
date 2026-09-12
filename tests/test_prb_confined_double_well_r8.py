from pathlib import Path
import sys

import numpy as np
import pytest

from tmd_pimc.energy_estimators import potential_gradient
from tmd_pimc.potentials import GridPotential2D

CAMPAIGN = Path(__file__).resolve().parents[1] / "runners" / "validation" / "prb_confined_double_well_r8"
sys.path.insert(0, str(CAMPAIGN))

import model
import fast_staging


def test_specialized_potential_matches_generic_model():
    rng = np.random.default_rng(20260912)
    points = rng.uniform(-12.0, 12.0, size=(64, 2))
    for ex in (-0.00051703999572, 0.0, 0.00025851999786):
        for asym in (0.0, 0.004, 0.006):
            for wall_r in (20.0, 25.0, 30.0):
                generic = model.build_potential(ex, asym, wall_R_nm=wall_r)
                expected = generic.value(points)
                observed = np.array([
                    fast_staging.v_scalar(x, y, ex, asym, wall_r, model.WALL_V0_EV)
                    for x, y in points
                ])
                np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-14)


def test_confined_composite_gradient_matches_finite_difference():
    rng = np.random.default_rng(17)
    points = rng.uniform(-9.0, 9.0, size=(24, 2))
    potential = model.build_potential(0.00025851999786, 0.004, wall_R_nm=25.0)
    analytic = potential_gradient(potential, points)
    h = 1e-6
    numeric = np.empty_like(points)
    for axis in range(2):
        shift = np.zeros(2)
        shift[axis] = h
        numeric[:, axis] = (
            potential.value(points + shift) - potential.value(points - shift)
        ) / (2.0 * h)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-9)


def test_grid_gradient_fails_loudly():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    grid = GridPotential2D(x, y, np.zeros((2, 2)), periodic=True)
    with pytest.raises(NotImplementedError):
        potential_gradient(grid, np.array([[0.25, 0.25]]))


def test_authoritative_generic_sampler_quick_smoke():
    potential, _, sampler = model.make_sampler(16, 1234, Ex_eV_per_nm=0.0, staging_length=8)
    out = sampler.run(n_steps=120, burn_in=20, sample_every=10)
    samples = np.asarray(out["samples"])
    assert samples.shape == (10, 16, 2)
    assert np.isfinite(samples).all()
    summary = model.summarize_samples(samples, potential)
    assert 0.0 <= summary["p_bead"] <= 1.0
    assert np.isfinite(summary["Rg2_nm2"])
