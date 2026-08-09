"""Tests for the observable layer, including the periodicity question."""
import numpy as np
import pytest
from tmd_pimc import contact as cd
from tmd_pimc import pair_observables as po


def fake_run(n_cfg=800, P=64, mean_rho2=4.5, drift_cells=6.0, L=20.0, seed=0):
    """Synthetic sampler output: a bound pair whose COM diffuses across cells.

    This is the configuration that motivates the minimum-image question -- the
    absolute coordinates wander over many periods while the pair stays bound.
    """
    rng = np.random.default_rng(seed)
    com = rng.normal(0.0, drift_cells * L, size=(n_cfg, 1, 2)) \
        + rng.normal(0.0, 1.0, size=(n_cfg, P, 2))
    sig = np.sqrt(mean_rho2 / 2.0)
    d = rng.normal(0.0, sig, size=(n_cfg, P, 2))
    return com + 0.5 * d, com - 0.5 * d


def test_separations_are_invariant_to_com_drift():
    """rho must not depend on how far the pair has wandered from the origin."""
    e, h = fake_run(drift_cells=0.0, seed=1)
    e2, h2 = fake_run(drift_cells=50.0, seed=1)
    r1 = po.pair_separations(e, h)
    r2 = po.pair_separations(e2, h2)
    assert np.allclose(np.sort(r1.ravel()), np.sort(r2.ravel()), rtol=1e-12)


def test_minimum_image_would_hide_dissociation():
    """Guard on the design choice, aimed at the failure mode that matters.

    Minimum-imaging rho barely moves the contact density -- |psi(0)|^2 probes
    only small separations, which wrapping leaves alone -- so a check on that
    quantity would pass and give false confidence.

    The damage is to <rho^2>, and it is worse the more it matters. At bound
    separations wrapping is nearly harmless (a few percent), so it would look
    fine in every equilibrium test. As the pair dissociates, wrapping folds it
    back and <rho^2> SATURATES at the cell geometry instead of growing: a
    thirteen-fold increase in the true value produces barely a doubling of the
    wrapped one. A dissociation threshold measured that way reports the cell
    size, not the physics.
    """
    L = 20.0
    truth, wrapped = [], []
    for mean_rho2 in (30.0, 100.0, 400.0):
        e, h = fake_run(mean_rho2=mean_rho2, drift_cells=3.0, seed=2)
        raw = po.pair_separations(e, h)
        d = e - h
        d_mic = d - L * np.round(d / L)
        mic = np.hypot(d_mic[..., 0], d_mic[..., 1])
        assert mic.max() <= L * np.sqrt(2.0) / 2.0 + 1e-9
        truth.append(np.mean(raw ** 2))
        wrapped.append(np.mean(mic ** 2))

    # bound case: wrapping looks innocuous, which is why the bug would survive
    assert wrapped[0] / truth[0] > 0.95
    # dissociated case: the true value grows >13x, the wrapped one saturates
    assert truth[-1] / truth[0] > 13.0
    assert wrapped[-1] / wrapped[0] < 3.0
    assert wrapped[-1] < 0.2 * truth[-1]


def test_assert_interaction_is_aperiodic_detects_a_periodic_interaction():
    L = np.array([20.0, 0.0])
    r_e, r_h = np.array([0.3, 0.1]), np.array([1.7, -0.4])

    def good(re, rh):                      # aperiodic interaction
        return -1.0 / (1.0 + np.hypot(*(re - rh)))

    def bad(re, rh):                       # minimum-imaged: periodic
        d = re - rh
        d = d - 20.0 * np.round(d / 20.0)
        return -1.0 / (1.0 + np.hypot(*d))

    po.assert_interaction_is_aperiodic(good, r_e, r_h, L)
    with pytest.raises(AssertionError, match="minimum-imaged"):
        po.assert_interaction_is_aperiodic(bad, r_e, r_h, L)


def test_cutoff_diagnostics():
    e, h = fake_run(mean_rho2=4.5, seed=3)
    rho = po.pair_separations(e, h)
    ok = po.separation_diagnostics(rho, r_int_max=80.0)
    assert ok.cutoff_ok and not ok.notes

    bad = po.separation_diagnostics(rho * 40.0, r_int_max=80.0)
    assert not bad.cutoff_ok
    assert any("EXCEEDS" in n for n in bad.notes)


def test_contact_density_from_samples_matches_exact():
    e, h = fake_run(n_cfg=3000, P=64, mean_rho2=4.5, seed=4)
    res, diag = po.contact_density_from_samples(e, h, r_int_max=80.0)
    exact = cd.contact_density_gaussian_exact(diag.mean_rho2)
    assert abs(res.psi0_sq - exact) / exact < 0.03
    assert diag.cutoff_ok


def test_registry_resolved_recovers_a_planted_variation():
    """Plant a tighter pair in one half of the cell; the binning must see it."""
    rng = np.random.default_rng(5)
    L, n_cfg, P = 20.0, 4000, 32
    com = rng.uniform(0.0, L, size=(n_cfg, 1, 2)) + np.zeros((1, P, 2))
    tight = com[:, 0, 0] < L / 2
    s = np.where(tight, np.sqrt(2.0 / 2), np.sqrt(8.0 / 2))[:, None, None]
    d = rng.normal(0.0, 1.0, size=(n_cfg, P, 2)) * s
    e, h = com + 0.5 * d, com - 0.5 * d

    out = po.contact_density_by_registry(
        e, h, lattice_a1_nm=(L, 0.0), lattice_a2_nm=(0.0, L), n_bins_uv=2)
    left = out[(0, 0)].psi0_sq
    right = out[(1, 0)].psi0_sq
    assert left / right > 3.0     # 1/<rho^2> ratio is 4
