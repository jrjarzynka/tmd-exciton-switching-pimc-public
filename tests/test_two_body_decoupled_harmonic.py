"""Test #1 in the two-body validation programme: single independent wells.

With the interaction switched off (V_int = 0), the coupled electron-hole
model MUST reduce exactly to two independent single-body ring polymers,
each in its own harmonic well, with its own mass and spring constant.
Two independent masses/spring constants are used deliberately (not a
symmetric V_e = V_h setup): if electron/hole beads, masses, or spring
factors were ever swapped or cross-wired in action.py/sampler.py, a
symmetric test would not catch it, since e and h would be
indistinguishable.

Two kinds of check:
  1. DETERMINISTIC (no MC noise): with V_int = 0, delta_action_bead_move_e
     must not depend on path_h at all, and total_action must be exactly
     additive: total_action(path_e, path_h) = single_body_action(path_e)
     + single_body_action(path_h), verified against the already-validated
     single-body RingPolymerAction. This isolates a wiring bug from a
     sampling/statistics problem.
  2. STATISTICAL: running the full sampler, the thermal <r^2> per chain
     must match the exact finite-P discretized harmonic-oscillator result
     (harmonic_r2_primitive_finite_P), which separates true Trotter
     discretization error from Monte Carlo noise -- the same benchmark
     already used to validate the single-body sampler.
"""
import numpy as np
import pytest

from tmd_pimc.action import RingPolymerAction
from tmd_pimc.two_body_action import TwoBodyRingPolymerAction
from tmd_pimc.two_body_sampler import TwoBodyPIMCSamplerStaging
from tmd_pimc.potentials import HarmonicPotential, CompositePotential
from tmd_pimc.analytic import harmonic_r2_primitive_finite_P

# Deliberately asymmetric electron/hole parameters.
MASS_E_M0 = 0.40
MASS_H_M0 = 0.70
K_E = 0.0030   # eV/nm^2
K_H = 0.0012   # eV/nm^2
TEMPERATURE_K = 20.0
N_BEADS = 24


class _ZeroInteraction:
    def value(self, r):
        r = np.asarray(r)
        return np.zeros(r.shape[0])


def _make_action():
    return TwoBodyRingPolymerAction(
        mass_e_m0=MASS_E_M0,
        mass_h_m0=MASS_H_M0,
        temperature_K=TEMPERATURE_K,
        n_beads=N_BEADS,
        potential_e=HarmonicPotential(k_eV_per_nm2=K_E),
        potential_h=HarmonicPotential(k_eV_per_nm2=K_H),
        potential_interaction=_ZeroInteraction(),
    )


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# --- 1. Deterministic wiring checks --------------------------------------

def test_delta_action_e_independent_of_hole_path(rng):
    """With V_int = 0, moving an electron bead must give the identical
    delta-action no matter what the (fixed) hole path is."""
    action = _make_action()
    path_e = 0.3 * rng.standard_normal((N_BEADS, 2))
    path_h_a = 0.3 * rng.standard_normal((N_BEADS, 2))
    path_h_b = 5.0 * rng.standard_normal((N_BEADS, 2))  # wildly different

    j = 5
    r_new = path_e[j] + 0.1 * rng.standard_normal(2)

    dS_a = action.delta_action_bead_move_e(path_e, path_h_a, j, r_new)
    dS_b = action.delta_action_bead_move_e(path_e, path_h_b, j, r_new)
    assert dS_a == pytest.approx(dS_b, abs=1e-13)


def test_delta_action_h_independent_of_electron_path(rng):
    action = _make_action()
    path_h = 0.3 * rng.standard_normal((N_BEADS, 2))
    path_e_a = 0.3 * rng.standard_normal((N_BEADS, 2))
    path_e_b = 5.0 * rng.standard_normal((N_BEADS, 2))

    j = 11
    r_new = path_h[j] + 0.1 * rng.standard_normal(2)

    dS_a = action.delta_action_bead_move_h(path_e_a, path_h, j, r_new)
    dS_b = action.delta_action_bead_move_h(path_e_b, path_h, j, r_new)
    assert dS_a == pytest.approx(dS_b, abs=1e-13)


def test_total_action_is_exactly_additive(rng):
    """total_action(path_e, path_h) must equal the sum of two independent
    single-body RingPolymerAction evaluations, to machine precision."""
    action = _make_action()
    path_e = 0.3 * rng.standard_normal((N_BEADS, 2))
    path_h = 0.4 * rng.standard_normal((N_BEADS, 2))

    combined = action.total_action(path_e, path_h)

    ref_e = RingPolymerAction(
        mass_m0=MASS_E_M0, temperature_K=TEMPERATURE_K, n_beads=N_BEADS,
        potential=HarmonicPotential(k_eV_per_nm2=K_E),
    )
    ref_h = RingPolymerAction(
        mass_m0=MASS_H_M0, temperature_K=TEMPERATURE_K, n_beads=N_BEADS,
        potential=HarmonicPotential(k_eV_per_nm2=K_H),
    )
    expected = ref_e.total_action(path_e) + ref_h.total_action(path_h)

    assert combined == pytest.approx(expected, rel=1e-10)


# --- 2. Statistical check against the exact finite-P benchmark -----------

def test_thermal_r2_matches_exact_finite_P_benchmark():
    """Full sampler run: <r_e^2> and <r_h^2> must independently match the
    exact discretized (finite-P) harmonic-oscillator result, within MC
    error. This is the same style of benchmark already used to validate
    the single-body sampler (radial_solver / analytic.py)."""
    action = _make_action()
    sampler = TwoBodyPIMCSamplerStaging(
        action=action,
        local_step_nm=0.25,
        global_step_nm=1.0,          # unused: global_move_probability=0.0
        global_move_probability=0.0,
        rng_seed=42,
        staging_segment_lengths=(4, 8, 12),
        staging_moves_per_step=2,
    )
    result = sampler.run(
        n_steps=40_000, burn_in=8_000, sample_every=10,
        center_e=(0.0, 0.0), center_h=(0.0, 0.0),
    )
    samples_e, samples_h = result["samples_e"], result["samples_h"]

    r2_e_per_sample = np.mean(np.sum(samples_e**2, axis=-1), axis=1)
    r2_h_per_sample = np.mean(np.sum(samples_h**2, axis=-1), axis=1)

    mean_r2_e = float(np.mean(r2_e_per_sample))
    mean_r2_h = float(np.mean(r2_h_per_sample))

    # Bootstrap SEM (samples are correlated, so this is an approximate but
    # workable error bar; tolerance below is generous -- 6 sigma -- to
    # keep the test robust against modest autocorrelation underestimation
    # while still failing hard on an actual wiring bug, which produces
    # O(1) (not O(1 sigma)) discrepancies).
    def bootstrap_sem(x, nboot=1000, seed=0):
        rng_b = np.random.default_rng(seed)
        n = len(x)
        means = [np.mean(x[rng_b.integers(0, n, n)]) for _ in range(nboot)]
        return float(np.std(means, ddof=1))

    sem_e = bootstrap_sem(r2_e_per_sample, seed=1)
    sem_h = bootstrap_sem(r2_h_per_sample, seed=2)

    expected_e = harmonic_r2_primitive_finite_P(MASS_E_M0, K_E, TEMPERATURE_K, N_BEADS)
    expected_h = harmonic_r2_primitive_finite_P(MASS_H_M0, K_H, TEMPERATURE_K, N_BEADS)

    assert mean_r2_e == pytest.approx(expected_e, abs=6 * sem_e), (
        f"<r_e^2>={mean_r2_e:.5f} nm^2 vs exact finite-P {expected_e:.5f} nm^2 "
        f"(sem={sem_e:.5f})"
    )
    assert mean_r2_h == pytest.approx(expected_h, abs=6 * sem_h), (
        f"<r_h^2>={mean_r2_h:.5f} nm^2 vs exact finite-P {expected_h:.5f} nm^2 "
        f"(sem={sem_h:.5f})"
    )
