"""Cross-validation of TwoBodyPIMCSamplerStagingPeriodicJIT -- the sampler
actually used in production (run_two_body_landscape_scan.py,
adaptive_dense_field_scan.py) -- against the pure-Python
TwoBodyPIMCSamplerStaging reference, using the real MoirePotential physics
(registry offset included) rather than a harmonic toy problem.

WHY A HARMONIC/DOUBLE-WELL GRID EIGENSOLVER ISN'T NEEDED HERE
----------------------------------------------------------------
MoirePotential.value() is an exact analytic function (sum of three
cosines), evaluable anywhere with no rasterization. That means the
pure-Python, already-validated TwoBodyPIMCSamplerStaging can run the
EXACT SAME physics as the periodic JIT sampler -- moire wells for both
electron and hole, hole registry-shifted via ShiftedPotential -- with no
grid/interpolation machinery at all. This gives a clean, independent
reference for the periodic JIT kernel specifically (rather than a
harmonic surrogate), and as a side effect re-exercises the registry-shift
fix (two_body_sampler_periodic_jit.py, fixed 2026-08-02) from a
completely different angle: if ShiftedPotential's registry shift and the
periodic JIT sampler's origin_h_nm ever silently diverged again, thermal
averages of the potential energy would disagree well outside MC error.

Observable used: thermal <V_e(r_e)> and <V_h(r_h)>, i.e. the actual
(un-wrapped) instantaneous potential value at each particle's real
position. Because MoirePotential is itself exactly periodic, this is a
clean, wrap-free observable -- no need to fold sampled positions back
into a unit cell by hand.
"""
import numpy as np
import pytest

from tmd_pimc.two_body_action import TwoBodyRingPolymerAction
from tmd_pimc.two_body_sampler import TwoBodyPIMCSamplerStaging
from tmd_pimc.two_body_sampler_periodic_jit import TwoBodyPIMCSamplerStagingPeriodicJIT
from tmd_pimc.potentials import MoirePotential, CompositePotential
from tmd_pimc.potential_helpers import ShiftedPotential

PERIOD_NM = 20.0
AMPLITUDE_EV = 0.02
SHIFT_NM = (5.0, 0.0)
MASS_E_M0 = 0.40
MASS_H_M0 = 0.70
TEMPERATURE_K = 20.0
N_BEADS = 24

V_E_REF = MoirePotential(amplitude_eV=AMPLITUDE_EV, period_nm=PERIOD_NM)
V_H_REF = ShiftedPotential(inner=MoirePotential(amplitude_eV=AMPLITUDE_EV, period_nm=PERIOD_NM),
                            shift_nm=SHIFT_NM)


class _ZeroInteraction:
    def value(self, r):
        r = np.asarray(r)
        return np.zeros(r.shape[0])


def _block_bootstrap_sem(x, block_size=100, nboot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = max(1, n // block_size)
    blocks = [x[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    block_means = np.array([b.mean() for b in blocks if len(b) > 0])
    means = [np.mean(block_means[rng.integers(0, len(block_means), len(block_means))])
             for _ in range(nboot)]
    return float(np.std(means, ddof=1))


def _thermal_V(potential, samples):
    """<V(r)> averaged over beads then over samples, plus block-bootstrap SEM."""
    per_sample = np.mean(
        potential.value(samples.reshape(-1, 2)).reshape(samples.shape[:2]), axis=1
    )
    return float(np.mean(per_sample)), _block_bootstrap_sem(per_sample)


def test_periodic_jit_matches_python_reference_moire_registry_physics():
    action_py = TwoBodyRingPolymerAction(
        mass_e_m0=MASS_E_M0, mass_h_m0=MASS_H_M0, temperature_K=TEMPERATURE_K, n_beads=N_BEADS,
        potential_e=V_E_REF, potential_h=V_H_REF, potential_interaction=_ZeroInteraction(),
    )
    sampler_py = TwoBodyPIMCSamplerStaging(
        action=action_py, local_step_nm=0.3, global_step_nm=PERIOD_NM,
        global_move_probability=0.3, rng_seed=21,
        staging_segment_lengths=(4, 8, 12), staging_moves_per_step=2,
    )
    res_py = sampler_py.run(n_steps=100_000, burn_in=15_000, sample_every=10,
                             center_e=(0.0, 0.0), center_h=(0.0, 0.0))
    mean_Ve_py, sem_Ve_py = _thermal_V(V_E_REF, res_py["samples_e"])
    mean_Vh_py, sem_Vh_py = _thermal_V(V_H_REF, res_py["samples_h"])

    # potential_e/h on this action are unused placeholders for the periodic
    # sampler (it builds its own moire+Stark landscape internally from
    # moire_period_nm/moire_amplitude_eV/origin_e_nm/origin_h_nm) -- only
    # potential_interaction is actually read from it.
    action_jit = TwoBodyRingPolymerAction(
        mass_e_m0=MASS_E_M0, mass_h_m0=MASS_H_M0, temperature_K=TEMPERATURE_K, n_beads=N_BEADS,
        potential_e=CompositePotential(terms=[]), potential_h=CompositePotential(terms=[]),
        potential_interaction=_ZeroInteraction(),
    )
    sampler_jit = TwoBodyPIMCSamplerStagingPeriodicJIT(
        action=action_jit, moire_period_nm=PERIOD_NM, moire_amplitude_eV=AMPLITUDE_EV,
        origin_e_nm=(0.0, 0.0), origin_h_nm=SHIFT_NM, Fz_eV_per_nm=0.0,
        local_step_nm=0.3, global_step_nm=PERIOD_NM, global_move_probability=0.3,
        rng_seed=99, staging_segment_lengths=(4, 8, 12), staging_moves_per_step=2,
        periodic_cell_grid_size=200,
    )
    res_jit = sampler_jit.run(n_steps=100_000, burn_in=15_000, sample_every=10,
                               center_e=(0.0, 0.0), center_h=(0.0, 0.0))
    mean_Ve_jit, sem_Ve_jit = _thermal_V(V_E_REF, res_jit["samples_e"])
    mean_Vh_jit, sem_Vh_jit = _thermal_V(V_H_REF, res_jit["samples_h"])

    # Combine both runs' MC errors in quadrature; 8-sigma margin as used
    # elsewhere in this programme for autocorrelated multi-well observables.
    tol_e = 8.0 * np.hypot(sem_Ve_py, sem_Ve_jit)
    tol_h = 8.0 * np.hypot(sem_Vh_py, sem_Vh_jit)

    assert mean_Ve_jit == pytest.approx(mean_Ve_py, abs=tol_e), (
        f"<V_e>: python={mean_Ve_py:.6f}+/-{sem_Ve_py:.6f}, "
        f"periodic-JIT={mean_Ve_jit:.6f}+/-{sem_Ve_jit:.6f}"
    )
    assert mean_Vh_jit == pytest.approx(mean_Vh_py, abs=tol_h), (
        f"<V_h>: python={mean_Vh_py:.6f}+/-{sem_Vh_py:.6f}, "
        f"periodic-JIT={mean_Vh_jit:.6f}+/-{sem_Vh_jit:.6f}"
    )
