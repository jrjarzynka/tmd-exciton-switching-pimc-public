import numpy as np, pytest
from tmd_pimc import pair_observables as po
from tmd_pimc import contact as cd


def gauss(mean_rho2, n, rng):
    s = np.sqrt(mean_rho2 / 2.0)
    xy = rng.normal(0.0, s, size=(n, 2))
    return np.hypot(xy[:, 0], xy[:, 1])


def shell(r0, width, n, rng):
    """Pair localised at separation r0: two carriers in adjacent minima."""
    ang = rng.uniform(0, 2*np.pi, n)
    c = np.column_stack([r0*np.cos(ang), r0*np.sin(ang)]) + rng.normal(0, width, (n,2))
    return np.hypot(c[:,0], c[:,1])


def test_g_of_rho_matches_the_gaussian_reference():
    rng = np.random.default_rng(1); s = 4.0
    pc = po.pair_correlation(gauss(s, 400_000, rng), n_bins=40)
    exact = np.exp(-pc.rho_nm**2 / s) / (np.pi * s)
    m = pc.counts > 200
    assert np.allclose(pc.g_of_rho[m], exact[m], rtol=0.08)


def test_first_bin_underestimates_the_contact_density():
    """Documents why g(0) must come from contact_density, not from bin 0.

    The first bin averages g over [0, sqrt(du)] with du set by the whole
    plotted range, so for a decaying g it is biased low. contact_density fits
    a narrow window instead.
    """
    rng = np.random.default_rng(2); s = 4.0
    r = gauss(s, 800_000, rng)
    pc = po.pair_correlation(r, n_bins=40)
    ce = cd.contact_density(r, rng=rng)
    assert pc.g_of_rho[0] < ce.psi0_sq
    assert pc.g_of_rho[0] > 0.5 * ce.psi0_sq


def test_radial_density_is_normalised():
    rng = np.random.default_rng(3)
    pc = po.pair_correlation(gauss(4.0, 200_000, rng), n_bins=60)
    w = np.diff(pc.bin_edges_rho)
    assert abs(np.sum(pc.radial_density * w) - 1.0) < 0.02


def test_bimodality_detected_when_present_and_not_otherwise():
    rng = np.random.default_rng(4)
    uni = po.pair_correlation(gauss(4.0, 300_000, rng), n_bins=60)
    assert po.detect_bimodality(uni)["n_modes"] == 1

    mix = np.concatenate([gauss(4.0, 150_000, rng), shell(11.55, 1.0, 150_000, rng)])
    bi = po.detect_bimodality(po.pair_correlation(mix, n_bins=60))
    assert bi["n_modes"] == 2
    peaks = sorted(bi["peak_rho_nm"])
    assert peaks[0] < 4.0 and 10.0 < peaks[1] < 13.0
    assert bi["dip_depth"] > 0.3


def test_variance_decomposition_separates_the_two_bimodalities():
    """The discriminating test: same aggregate g(rho), opposite meaning."""
    rng = np.random.default_rng(5)
    n, seeds = 20_000, 12

    # (a) every seed visits both states -> bimodality is within-seed
    within = [np.concatenate([gauss(4.0, n//2, rng), shell(11.55, 1.0, n//2, rng)])
              for _ in range(seeds)]
    # (b) each seed is stuck in one state -> bimodality is between-seed
    between = [gauss(4.0, n, rng) if i % 2 else shell(11.55, 1.0, n, rng)
               for i in range(seeds)]

    a = po.variance_decomposition(within)
    b = po.variance_decomposition(between)

    # aggregates look alike
    ga = po.detect_bimodality(po.pair_correlation(np.concatenate(within), n_bins=60))
    gb = po.detect_bimodality(po.pair_correlation(np.concatenate(between), n_bins=60))
    assert ga["n_modes"] == gb["n_modes"] == 2

    # the decomposition tells them apart
    assert a["between_seed_fraction"] < 0.05
    assert b["between_seed_fraction"] > 0.90


def test_rejects_single_seed():
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        po.variance_decomposition([gauss(4.0, 1000, rng)])
