"""
Tests for the contact-density estimator.

The estimator is checked against two exact references chosen to probe opposite
failure modes:

  Gaussian     smooth at the origin, zero slope. Any reasonable extrapolation
               passes. Also gives a reference-free identity,
               |psi(0)|^2 = 1/(pi <rho^2>), that holds at finite bead count.

  Exponential  cusped at the origin, finite slope in rho and infinite slope in
               u = rho^2. Fails if the extrapolation is done in u, which is the
               obvious implementation and the reason this file exists.

A short primitive-action PI-QMC of a harmonic relative coordinate provides an
end-to-end check on correlated samples rather than independent draws.
"""

import numpy as np
import pytest

from tmd_pimc import contact as cd


# =============================================================================
# EXACT REFERENCES, INDEPENDENT SAMPLES
# =============================================================================


def sample_gaussian_2d(mean_rho2, n, rng):
    """2D isotropic Gaussian with prescribed <rho^2>."""
    sigma = np.sqrt(mean_rho2 / 2.0)          # per Cartesian component
    xy = rng.normal(0.0, sigma, size=(n, 2))
    return np.hypot(xy[:, 0], xy[:, 1])


def sample_exponential_2d(a, n, rng):
    """|psi|^2 ~ exp(-2 rho / a): radial density P(rho) ~ rho exp(-2 rho / a).

    That is a Gamma(shape=2, scale=a/2) distribution in rho.
    """
    return rng.gamma(shape=2.0, scale=a / 2.0, size=n)


def test_gaussian_contact_density():
    """Systematic-bias check: high statistics, tight tolerance."""
    rng = np.random.default_rng(1)
    mean_rho2 = 4.5
    rho = sample_gaussian_2d(mean_rho2, 2_000_000, rng)

    res = cd.contact_density(rho, rng=rng)
    exact = cd.contact_density_gaussian_exact(mean_rho2)

    assert abs(res.psi0_sq - exact) / exact < 0.02
    assert res.stderr / res.psi0_sq < 0.05
    assert abs(res.slope_rho) < 0.35 * res.psi0_sq   # smooth: shallow slope
    assert not res.curvature_flag


def test_exponential_contact_density_cusped():
    """Systematic-bias check on the cusped reference."""
    rng = np.random.default_rng(2)
    a = 1.7
    rho = sample_exponential_2d(a, 2_000_000, rng)

    res = cd.contact_density(rho, rng=rng)
    exact = cd.contact_density_exponential_exact(a)

    assert abs(res.psi0_sq - exact) / exact < 0.03
    # the cusp must be visible: d|psi|^2/drho = -(2/a)|psi(0)|^2
    assert res.slope_rho < 0
    assert abs(res.slope_rho / res.psi0_sq + 2.0 / a) < 0.5


def test_extrapolating_in_u_would_bias_the_cusped_case():
    """Guard on the design choice: fitting in u = rho^2 fails for a cusp.

    This reproduces the naive implementation and asserts that it is wrong, so
    that anyone 'simplifying' the estimator back to a fit in u trips this test
    rather than silently biasing every cusped state.
    """
    rng = np.random.default_rng(3)
    a = 1.7
    rho = sample_exponential_2d(a, 2_000_000, rng)
    exact = cd.contact_density_exponential_exact(a)

    window = 0.35 * np.sqrt(np.mean(rho ** 2))
    r_eff, dens_u, counts = cd._histogram_in_u(rho, window ** 2, 16)
    psi2 = dens_u / np.pi
    w = np.sqrt(counts)

    in_rho = np.polyfit(r_eff, psi2, 2, w=w)[-1]
    in_u = np.polyfit(r_eff ** 2, psi2, 2, w=w)[-1]

    assert abs(in_rho - exact) / exact < 0.03
    assert abs(in_u - exact) / exact > 0.10


# =============================================================================
# ROBUSTNESS
# =============================================================================


@pytest.mark.parametrize("n_bins", [10, 16, 24, 32])
def test_insensitive_to_bin_count(n_bins):
    """Stability check: the answer must sit within its own quoted error bar.

    Compared against the bootstrap error rather than a fixed percentage, so the
    test measures whether the estimator KNOWS how uncertain it is -- a fixed
    tolerance would either pass a badly underestimated error or fail on noise.
    """
    rng = np.random.default_rng(4)
    mean_rho2 = 3.0
    rho = sample_gaussian_2d(mean_rho2, 300_000, rng)
    res = cd.contact_density(rho, n_bins=n_bins, rng=rng)
    exact = cd.contact_density_gaussian_exact(mean_rho2)
    assert abs(res.psi0_sq - exact) < 3.0 * res.stderr


@pytest.mark.parametrize("frac", [0.25, 0.35, 0.45])
def test_insensitive_to_window(frac):
    rng = np.random.default_rng(5)
    mean_rho2 = 3.0
    rho = sample_gaussian_2d(mean_rho2, 300_000, rng)
    res = cd.contact_density(rho, rho_window=frac * np.sqrt(mean_rho2), rng=rng)
    exact = cd.contact_density_gaussian_exact(mean_rho2)
    assert abs(res.psi0_sq - exact) < 3.0 * res.stderr


def test_scaling_with_units():
    """|psi(0)|^2 has dimensions 1/length^2: rescaling rho by s scales it by 1/s^2."""
    rng = np.random.default_rng(6)
    rho = sample_gaussian_2d(2.0, 200_000, rng)
    a = cd.contact_density(rho, rng=np.random.default_rng(7)).psi0_sq
    b = cd.contact_density(3.0 * rho, rng=np.random.default_rng(7)).psi0_sq
    assert abs(b * 9.0 - a) / a < 1e-9   # exact: pure rescaling, same bins


def test_rejects_bad_input():
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError):
        cd.contact_density(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        cd.contact_density(-sample_gaussian_2d(1.0, 1000, rng))


# =============================================================================
# END-TO-END: PRIMITIVE-ACTION PI-QMC ON A HARMONIC RELATIVE COORDINATE
# =============================================================================


def pimc_harmonic_relative(mu, k, T, P, n_sweeps, rng, step=0.25, burn=2000):
    """Minimal 2D primitive-action ring-polymer sampler for V = k rho^2 / 2.

    Units: energies in eV, lengths in nm, mu in eV ps^2 / nm^2 equivalents is
    avoided by absorbing hbar and mass into the spring constant directly, which
    is all the test needs -- the reference identity holds for ANY Gaussian.
    """
    beta = 1.0 / T
    tau = beta / P
    spring = mu / (2.0 * tau ** 2)          # per unit tau in the action
    path = rng.normal(0.0, 0.5, size=(P, 2))
    samples = []

    def local_action(j, r):
        jm, jp = (j - 1) % P, (j + 1) % P
        kin = spring * (np.sum((r - path[jm]) ** 2) + np.sum((path[jp] - r) ** 2))
        pot = 0.5 * k * np.sum(r ** 2)
        return tau * (kin + pot)

    for sweep in range(n_sweeps + burn):
        for j in range(P):
            old = path[j].copy()
            new = old + rng.normal(0.0, step, size=2)
            if rng.random() < np.exp(-(local_action(j, new) - local_action(j, old))):
                path[j] = new
        if sweep >= burn and sweep % 5 == 0:
            samples.append(np.hypot(path[:, 0], path[:, 1]).copy())

    return np.concatenate(samples)


def test_pimc_harmonic_matches_the_gaussian_identity():
    """End-to-end on correlated samples at finite P.

    The harmonic thermal density matrix is Gaussian at any P with the primitive
    action, so |psi(0)|^2 = 1/(pi <rho^2>) holds exactly and needs no continuum
    reference. Any inconsistency is the estimator's, not the sampler's.
    """
    rng = np.random.default_rng(11)
    rho = pimc_harmonic_relative(mu=1.0, k=2.0, T=0.5, P=16,
                                 n_sweeps=12_000, rng=rng)

    res = cd.contact_density(rho, rng=rng)
    identity = cd.contact_density_gaussian_exact(res.mean_rho2)

    assert abs(res.psi0_sq - identity) / identity < 0.05


def test_relative_rate_ratio_and_error():
    rng = np.random.default_rng(12)
    tight = cd.contact_density(sample_gaussian_2d(2.0, 300_000, rng), rng=rng)
    loose = cd.contact_density(sample_gaussian_2d(8.0, 300_000, rng), rng=rng)

    ratio, err = cd.relative_radiative_rate(tight, loose)
    # rate ~ 1/<rho^2> for a Gaussian, so the ratio is 8.0/2.0 = 4.0.
    # The method bias largely cancels between two same-shape distributions,
    # which is exactly why ratios are the intended use.
    assert abs(ratio - 4.0) < 3.0 * err
    assert 0.0 < err < 0.5
