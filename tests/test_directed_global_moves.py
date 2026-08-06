"""Regression tests for lattice-directed global Monte Carlo moves
(added 2026-08-05).

MOTIVATION
----------
In a moire landscape the potential minima form a honeycomb lattice, with
two shells of hop vectors (nearest-neighbour, |d| = L/sqrt(3), connecting
the two sublattices; and Bravais, |d| = L). A rigid global translation
therefore lands on an equivalent minimum only if the displacement matches
a lattice vector in BOTH magnitude and direction. An isotropic Gaussian
proposal of the right magnitude still lands on a random point of a circle
of that radius, almost all of which sits at high potential -- the
bottleneck is angular, not radial.

This was verified empirically before implementing the fix: changing
global_step_nm from 15.0 to 11.5 nm (i.e. matching it exactly to the
nearest-neighbour hop distance for a 20 nm period) changed the global
acceptance by less than 6% relative (0.772% -> 0.815%) and did not reduce
the seed-to-seed spread of spatial observables at all. Tuning the step
magnitude alone is therefore NOT sufficient; the proposal direction must
be quantised to the lattice.

With directed proposals enabled, measured on the same landscape
(T = 15 K, P = 32, 10 seeds, 300k steps): global acceptance rises from
0.77% to 18.6% (directed_move_frac = 0.8), and the seed-to-seed spread of
<r^2> falls from 25.2% to 5.9%. Crucially, <V> is unchanged within error:
a directed run at 300k steps agrees to 0.47 sigma with an isotropic run at
2.7M steps, i.e. the directed sampler reaches in 300k steps what the
isotropic one needs roughly an order of magnitude longer to reach, with no
detectable bias.
"""
import numpy as np
import pytest

from tmd_pimc import (
    PIMCSamplerJIT,
    RingPolymerAction,
    MoirePotential,
    moire_hop_vectors_nm,
)

PERIOD_NM = 20.0
AMPLITUDE_EV = 0.09
TEMPERATURE_K = 15.0
N_BEADS = 32

SAMPLER_KWARGS = dict(
    local_step_nm=0.20,
    global_step_nm=15.0,
    global_move_probability=0.20,
    grid_size=200,
    grid_range_nm=50.0,
    boundary_mode="finite_square",
)


def _action():
    return RingPolymerAction(
        mass_m0=0.5, temperature_K=TEMPERATURE_K, n_beads=N_BEADS,
        potential=MoirePotential(amplitude_eV=AMPLITUDE_EV, period_nm=PERIOD_NM),
    )


# --- hop-vector geometry -------------------------------------------------

def test_hop_vectors_have_the_two_expected_shells():
    """Honeycomb minima give 6 nearest-neighbour vectors at L/sqrt(3) and
    6 Bravais vectors at L -- not a single shell."""
    v = moire_hop_vectors_nm(PERIOD_NM)
    lengths = np.sort(np.linalg.norm(v, axis=1))
    assert len(v) == 12
    np.testing.assert_allclose(lengths[:6], PERIOD_NM / np.sqrt(3.0), rtol=1e-9)
    np.testing.assert_allclose(lengths[6:], PERIOD_NM, rtol=1e-9)


def test_hop_vectors_are_closed_under_negation():
    """Required for proposal symmetry: without this, plain Metropolis
    acceptance (no Hastings factor) samples the wrong distribution."""
    v = moire_hop_vectors_nm(PERIOD_NM)
    for a in v:
        assert any(np.allclose(-a, b, atol=1e-9) for b in v)


def test_hop_vectors_map_minima_onto_minima():
    """All 6 Bravais vectors, and 3 of the 6 honeycomb vectors (those
    pointing to the other sublattice from a given site), must map a
    potential minimum onto an equivalent minimum."""
    from scipy.optimize import minimize
    pot = MoirePotential(amplitude_eV=AMPLITUDE_EV, period_nm=PERIOD_NM)
    m = minimize(lambda r: pot.value(np.array([r]))[0], [5.0, 5.0]).x
    v_min = pot.value(np.array([m]))[0]
    n_ok = sum(
        1 for d in moire_hop_vectors_nm(PERIOD_NM)
        if abs(pot.value(np.array([m + d]))[0] - v_min) < 1e-9
    )
    assert n_ok == 9


# --- sampler wiring ------------------------------------------------------

def test_directed_moves_disabled_by_default_reproduce_previous_behaviour():
    """Backward compatibility: with no directed-move arguments, or with
    directed_move_frac=0, the sampler must be bit-for-bit identical to the
    previous isotropic-only implementation."""
    act = _action()
    r_default = PIMCSamplerJIT(action=act, rng_seed=7, **SAMPLER_KWARGS).run(
        n_steps=8000, burn_in=2000, sample_every=20, center=(0.0, 0.0))
    r_explicit_off = PIMCSamplerJIT(
        action=act, rng_seed=7, global_disp_vectors_nm=None,
        directed_move_frac=0.0, **SAMPLER_KWARGS).run(
        n_steps=8000, burn_in=2000, sample_every=20, center=(0.0, 0.0))
    assert np.array_equal(r_default["samples"], r_explicit_off["samples"])


def test_asymmetric_vector_set_is_rejected():
    """A proposal set not closed under negation breaks detailed balance and
    must be refused at construction time rather than silently producing a
    wrong distribution."""
    with pytest.raises(ValueError, match="closed under negation"):
        PIMCSamplerJIT(action=_action(), rng_seed=7,
                        global_disp_vectors_nm=[[11.5, 0.0]],
                        directed_move_frac=0.5, **SAMPLER_KWARGS)


def test_directed_move_frac_out_of_range_is_rejected():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="directed_move_frac"):
            PIMCSamplerJIT(action=_action(), rng_seed=7,
                            global_disp_vectors_nm=moire_hop_vectors_nm(PERIOD_NM),
                            directed_move_frac=bad, **SAMPLER_KWARGS)


# --- the actual effect ---------------------------------------------------

def test_directed_moves_raise_global_acceptance_substantially():
    """The point of the whole exercise: acceptance should rise by more than
    an order of magnitude. (Measured ~0.77% -> ~18.6% at frac=0.8; the
    threshold below is deliberately loose so the test is robust to the
    shorter run length used here.)"""
    act = _action()
    hops = moire_hop_vectors_nm(PERIOD_NM)
    common = dict(n_steps=60_000, burn_in=10_000, sample_every=20, center=(0.0, 0.0))

    acc_iso = PIMCSamplerJIT(action=act, rng_seed=11, **SAMPLER_KWARGS).run(**common)["acceptance_global"]
    acc_dir = PIMCSamplerJIT(
        action=act, rng_seed=11, global_disp_vectors_nm=hops,
        directed_move_frac=0.8, **SAMPLER_KWARGS).run(**common)["acceptance_global"]

    assert acc_dir > 5.0 * acc_iso, (
        f"directed acceptance {acc_dir:.4f} vs isotropic {acc_iso:.4f} -- "
        f"expected a large improvement"
    )
