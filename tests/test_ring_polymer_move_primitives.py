"""Standalone mathematical tests for reusable staging/Fourier primitives."""
import numpy as np
import pytest

from tmd_pimc.staging import (
    bridge_conditional_moments,
    free_bridge_proposal,
    free_link_variance_nm2,
)
from tmd_pimc.normal_modes import (
    centroid_free_mode_powers,
    path_from_real_normal_modes,
    path_from_rfft_modes,
    propose_real_normal_mode_shift,
    propose_rfft_mode_shift,
    real_normal_mode_coordinates,
    rfft_mode_powers,
    rfft_modes,
    rfft_parseval_weights,
    spring_bond_sum_from_rfft,
    spring_mode_eigenvalue,
)


def test_bridge_supports_distinct_particle_masses_via_lambda_with_nonzero_variates():
    tau = 29.011295304363966
    lambda_e = 0.0380998 / 0.40
    lambda_h = 0.0380998 / 0.60
    var_e = free_link_variance_nm2(lambda_e, tau)
    var_h = free_link_variance_nm2(lambda_h, tau)
    assert var_e > var_h > 0.0

    path = np.zeros((80, 2))
    path[0] = [1.0, -2.0]
    path[32] = [8.0, 5.0]
    z = np.random.default_rng(123).normal(size=(31, 2))
    pe = free_bridge_proposal(path, 0, 32, z, var_e)
    ph = free_bridge_proposal(path, 0, 32, z, var_h)

    mean, _ = bridge_conditional_moments(path[0], path[32], 32, var_e)
    dev_e = pe[1:32] - mean
    dev_h = ph[1:32] - mean
    expected_ratio = np.sqrt(var_e / var_h)
    np.testing.assert_allclose(dev_e, expected_ratio * dev_h, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("p_beads", [79, 80])
def test_rfft_round_trip_and_full_fft_agreement(p_beads):
    path = np.random.default_rng(7 + p_beads).normal(size=(p_beads, 2))
    q = rfft_modes(path)
    back = path_from_rfft_modes(q, p_beads)
    np.testing.assert_allclose(back, path, rtol=0.0, atol=4e-15)

    full = np.fft.fft(path, axis=0, norm="ortho")
    np.testing.assert_allclose(q, full[: p_beads // 2 + 1], rtol=0.0, atol=2e-15)


@pytest.mark.parametrize("p_beads", [79, 80])
def test_rfft_parseval_weights_reconstruct_real_space_norm(p_beads):
    path = np.random.default_rng(100 + p_beads).normal(size=(p_beads, 3))
    q = rfft_modes(path)
    w = rfft_parseval_weights(p_beads)
    raw = np.sum(np.abs(q) ** 2, axis=1)
    real_norm = float(np.sum(path * path))
    weighted_norm = float(np.dot(w, raw))
    np.testing.assert_allclose(weighted_norm, real_norm, rtol=2e-15, atol=2e-13)
    np.testing.assert_allclose(rfft_mode_powers(path).sum(), real_norm, rtol=2e-15, atol=2e-13)

    assert w[0] == 1.0
    if p_beads % 2 == 0:
        assert w[-1] == 1.0
        assert np.all(w[1:-1] == 2.0)
    else:
        assert np.all(w[1:] == 2.0)


@pytest.mark.parametrize("p_beads", [79, 80])
def test_spring_action_diagonalization_in_weighted_rfft(p_beads):
    path = np.random.default_rng(200 + p_beads).normal(size=(p_beads, 2))
    bonds = np.roll(path, -1, axis=0) - path
    real_bond_sum = float(np.sum(bonds * bonds))
    modal_bond_sum = spring_bond_sum_from_rfft(path)
    np.testing.assert_allclose(modal_bond_sum, real_bond_sum, rtol=3e-15, atol=4e-13)


@pytest.mark.parametrize("p_beads", [79, 80])
def test_real_normal_mode_basis_is_orthonormal_and_round_trips(p_beads):
    path = np.random.default_rng(300 + p_beads).normal(size=(p_beads, 2))
    c, s = real_normal_mode_coordinates(path)
    back = path_from_real_normal_modes(c, s, p_beads)
    np.testing.assert_allclose(back, path, rtol=0.0, atol=4e-15)
    modal_norm = float(np.sum(c * c) + np.sum(s * s))
    np.testing.assert_allclose(modal_norm, np.sum(path * path), rtol=2e-15, atol=2e-13)
    assert np.all(s[0] == 0.0)
    if p_beads % 2 == 0:
        assert np.all(s[-1] == 0.0)


def test_literal_raw_rfft_mode_shift_is_reversible():
    path = np.random.default_rng(9).normal(size=(80, 2))
    dr = np.array([0.21, -0.13])
    di = np.array([0.07, 0.03])
    moved = propose_rfft_mode_shift(path, 1, dr, di)
    restored = propose_rfft_mode_shift(moved, 1, -dr, -di)
    np.testing.assert_allclose(restored, path, rtol=0.0, atol=4e-15)


def test_literal_real_normal_mode_shift_is_reversible_and_preserves_centroid_for_k_gt_zero():
    path = np.random.default_rng(10).normal(size=(80, 2))
    centroid_before = path.mean(axis=0)
    dc = np.array([0.19, -0.11])
    ds = np.array([0.04, 0.06])
    moved = propose_real_normal_mode_shift(path, 1, dc, ds)
    np.testing.assert_allclose(moved.mean(axis=0), centroid_before, rtol=0.0, atol=3e-16)
    restored = propose_real_normal_mode_shift(moved, 1, -dc, -ds)
    np.testing.assert_allclose(restored, path, rtol=0.0, atol=4e-15)


def test_centroid_shift_has_expected_orthonormal_scaling():
    path = np.random.default_rng(11).normal(size=(80, 2))
    delta = np.array([0.8, -0.4])
    moved = propose_real_normal_mode_shift(path, 0, delta)
    expected = path.mean(axis=0) + delta / np.sqrt(80.0)
    np.testing.assert_allclose(moved.mean(axis=0), expected, rtol=0.0, atol=4e-16)


def test_even_nyquist_is_real_but_odd_highest_rfft_bin_is_not_self_conjugate():
    path_even = np.random.default_rng(12).normal(size=(80, 2))
    with pytest.raises(ValueError):
        propose_real_normal_mode_shift(path_even, 40, np.zeros(2), np.ones(2))

    path_odd = np.random.default_rng(13).normal(size=(79, 2))
    moved = propose_real_normal_mode_shift(path_odd, 39, np.zeros(2), np.array([0.1, -0.2]))
    np.testing.assert_allclose(moved.mean(axis=0), path_odd.mean(axis=0), atol=4e-16)


def test_centroid_free_powers_use_parseval_weighting():
    path = np.random.default_rng(14).normal(size=(80, 2))
    power = centroid_free_mode_powers(path)
    assert power[0] == 0.0
    centered = path - path.mean(axis=0)
    np.testing.assert_allclose(power.sum(), np.sum(centered * centered), rtol=2e-15, atol=2e-13)


def test_harmonic_primitive_action_matches_modal_formula():
    p_beads = 80
    tau = 29.011295304363966
    lam = 0.0761996
    kpf = 1.0 / (4.0 * lam * tau)
    k_harm = 0.0002
    path = np.random.default_rng(15).normal(size=(p_beads, 2))

    bonds = np.roll(path, -1, axis=0) - path
    real_action = kpf * np.sum(bonds * bonds) + tau * 0.5 * k_harm * np.sum(path * path)

    powers = rfft_mode_powers(path, parseval_weighted=True)
    eigenvalues = np.array([spring_mode_eigenvalue(k, p_beads) for k in range(p_beads // 2 + 1)])
    modal_action = np.sum((kpf * eigenvalues + tau * 0.5 * k_harm) * powers)
    np.testing.assert_allclose(modal_action, real_action, rtol=3e-15, atol=3e-13)


def test_ring_spring_eigenvalues():
    assert spring_mode_eigenvalue(0, 80) == 0.0
    assert 0.0 < spring_mode_eigenvalue(1, 80) < spring_mode_eigenvalue(2, 80)
    np.testing.assert_allclose(spring_mode_eigenvalue(40, 80), 4.0, atol=1e-15)
