"""Cross-validation of TwoBodyPIMCSamplerStagingJIT (Numba, finite-box)
against the already-validated pure-Python TwoBodyPIMCSamplerStaging.

WHY THIS TEST EXISTS
---------------------
Numba cannot JIT-compile the dataclass-based TwoBodyRingPolymerAction
directly, so the physics (spring terms, potential lookups, interaction
table interpolation, staging bridge, global move) is hand-duplicated
inside run_pimc_core_two_body_staging_jit (two_body_kernels_jit.py) as a
SEPARATE implementation from action.py/two_body_action.py. Tests #1-#3 in
this validation programme only ever exercised the pure-Python reference
sampler; the JIT kernel -- the one actually used by
runners/validation/run_two_body_validation.py in production -- had never
been checked against anything. This file closes that gap by re-running
the same two exactly-solvable scenarios (independent wells, coupled
harmonic pair) through the JIT sampler.

TwoBodyPIMCSamplerStagingPeriodicJIT (the periodic sampler used by the
Fz-scan and landscape-scan production scripts) is NOT covered here: its
landscape is hardcoded to a MoirePotential + optional Stark term rather
than an arbitrary Potential2D, so it cannot run these harmonic scenarios
directly. It shares the interaction-table code path validated here
(build_uniform_interaction_table, radial_table_interpolate, imported
unchanged from two_body_kernels_jit.py), but its own landscape/staging
loop still needs a dedicated periodic-landscape cross-validation test.
"""
import numpy as np
import pytest

from tmd_pimc.two_body_action import TwoBodyRingPolymerAction
from tmd_pimc.two_body_sampler_jit import TwoBodyPIMCSamplerStagingJIT
from tmd_pimc.potentials import HarmonicPotential
from tmd_pimc.analytic import harmonic_r2_primitive_finite_P
from tmd_pimc.constants import HBAR2_OVER_2M0, KB_EV_PER_K

MASS_E_M0 = 0.40
MASS_H_M0 = 0.70
TEMPERATURE_K = 20.0
N_BEADS = 24
LANDSCAPE_GRID_SIZE = 400
LANDSCAPE_GRID_RANGE_NM = 40.0


class _ZeroInteraction:
    def value(self, r):
        r = np.asarray(r)
        return np.zeros(r.shape[0])


def _block_bootstrap_sem(x, block_size=100, nboot=1000, seed=0):
    """Block bootstrap SEM: the coupled harmonic case has no global move
    (single shared basin), so local+staging moves alone give a noticeably
    longer integrated autocorrelation time (~16 samples at sample_every=10
    here, verified directly) than the decoupled case. A naive i.i.d.
    bootstrap underestimates the true error by roughly sqrt(2*tau_int) in
    this regime; block bootstrap with a block width well above tau_int
    fixes this."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = max(1, n // block_size)
    blocks = [x[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    block_means = np.array([b.mean() for b in blocks if len(b) > 0])
    means = [np.mean(block_means[rng.integers(0, len(block_means), len(block_means))])
             for _ in range(nboot)]
    return float(np.std(means, ddof=1))


def _exact_coupled_covariance(mass_e_m0, mass_h_m0, k_conf, k_int, T, P):
    """Same exact finite-P reference as test_two_body_coupled_harmonic.py."""
    lam_e = HBAR2_OVER_2M0 / mass_e_m0
    lam_h = HBAR2_OVER_2M0 / mass_h_m0
    beta = 1.0 / (KB_EV_PER_K * T)
    tau = beta / P
    kpf_e = 1.0 / (4.0 * lam_e * tau)
    kpf_h = 1.0 / (4.0 * lam_h * tau)
    C = 2 * np.eye(P) - np.eye(P, k=1) - np.eye(P, k=-1)
    C[0, -1] -= 1.0
    C[-1, 0] -= 1.0
    diag_e = kpf_e * C + (tau * k_conf / 2.0 + tau * k_int / 2.0) * np.eye(P)
    diag_h = kpf_h * C + (tau * k_conf / 2.0 + tau * k_int / 2.0) * np.eye(P)
    off = -(tau * k_int / 2.0) * np.eye(P)
    M = np.block([[diag_e, off], [off, diag_h]])
    cov = np.linalg.inv(2.0 * M)
    var_xe = float(np.mean(np.diag(cov)[:P]))
    var_xh = float(np.mean(np.diag(cov)[P:]))
    cross = float(np.mean(np.diag(cov[:P, P:])))
    r2_e = 2.0 * var_xe
    r2_h = 2.0 * var_xh
    r2_rel = r2_e + r2_h - 4.0 * cross
    return r2_e, r2_h, r2_rel


# --- JIT analogue of test #1: independent wells (V_int = 0) --------------

def test_jit_finite_box_decoupled_harmonic_matches_exact_reference():
    K_E, K_H = 0.0030, 0.0012
    action = TwoBodyRingPolymerAction(
        mass_e_m0=MASS_E_M0, mass_h_m0=MASS_H_M0, temperature_K=TEMPERATURE_K,
        n_beads=N_BEADS,
        potential_e=HarmonicPotential(k_eV_per_nm2=K_E),
        potential_h=HarmonicPotential(k_eV_per_nm2=K_H),
        potential_interaction=_ZeroInteraction(),
    )
    sampler = TwoBodyPIMCSamplerStagingJIT(
        action, local_step_nm=0.25, global_step_nm=1.0,
        global_move_probability=0.0, rng_seed=42,
        staging_segment_lengths=(4, 8, 12), staging_moves_per_step=2,
        landscape_grid_size=LANDSCAPE_GRID_SIZE,
        landscape_grid_range_nm=LANDSCAPE_GRID_RANGE_NM,
    )
    result = sampler.run(n_steps=40_000, burn_in=8_000, sample_every=10,
                          center_e=(0.0, 0.0), center_h=(0.0, 0.0))
    r2_e_t = np.mean(np.sum(result["samples_e"]**2, axis=-1), axis=1)
    r2_h_t = np.mean(np.sum(result["samples_h"]**2, axis=-1), axis=1)

    mean_r2_e, sem_e = float(np.mean(r2_e_t)), _block_bootstrap_sem(r2_e_t, block_size=50, seed=1)
    mean_r2_h, sem_h = float(np.mean(r2_h_t)), _block_bootstrap_sem(r2_h_t, block_size=50, seed=2)

    exp_e = harmonic_r2_primitive_finite_P(MASS_E_M0, K_E, TEMPERATURE_K, N_BEADS)
    exp_h = harmonic_r2_primitive_finite_P(MASS_H_M0, K_H, TEMPERATURE_K, N_BEADS)

    assert mean_r2_e == pytest.approx(exp_e, abs=8 * sem_e), (
        f"JIT <r_e^2>={mean_r2_e:.4f} vs exact {exp_e:.4f} (sem={sem_e:.4f})"
    )
    assert mean_r2_h == pytest.approx(exp_h, abs=8 * sem_h), (
        f"JIT <r_h^2>={mean_r2_h:.4f} vs exact {exp_h:.4f} (sem={sem_h:.4f})"
    )


# --- JIT analogue of test #2: coupled harmonic pair -----------------------

def test_jit_finite_box_coupled_harmonic_matches_exact_reference():
    K_CONF, K_INT = 0.0015, 0.0200
    action = TwoBodyRingPolymerAction(
        mass_e_m0=MASS_E_M0, mass_h_m0=MASS_H_M0, temperature_K=TEMPERATURE_K,
        n_beads=N_BEADS,
        potential_e=HarmonicPotential(k_eV_per_nm2=K_CONF),
        potential_h=HarmonicPotential(k_eV_per_nm2=K_CONF),
        potential_interaction=HarmonicPotential(k_eV_per_nm2=K_INT),
    )
    sampler = TwoBodyPIMCSamplerStagingJIT(
        action, local_step_nm=0.25, global_step_nm=1.0,
        global_move_probability=0.0, rng_seed=7,
        staging_segment_lengths=(4, 8, 12), staging_moves_per_step=2,
        landscape_grid_size=LANDSCAPE_GRID_SIZE,
        landscape_grid_range_nm=LANDSCAPE_GRID_RANGE_NM,
    )
    result = sampler.run(n_steps=100_000, burn_in=15_000, sample_every=10,
                          center_e=(0.0, 0.0), center_h=(0.0, 0.0))
    samples_e, samples_h = result["samples_e"], result["samples_h"]

    r2_e_t = np.mean(np.sum(samples_e**2, axis=-1), axis=1)
    r2_h_t = np.mean(np.sum(samples_h**2, axis=-1), axis=1)
    rel_t = samples_e - samples_h
    r2_rel_t = np.mean(np.sum(rel_t**2, axis=-1), axis=1)

    mean_r2_e, sem_e = float(np.mean(r2_e_t)), _block_bootstrap_sem(r2_e_t, seed=1)
    mean_r2_h, sem_h = float(np.mean(r2_h_t)), _block_bootstrap_sem(r2_h_t, seed=2)
    mean_r2_rel, sem_rel = float(np.mean(r2_rel_t)), _block_bootstrap_sem(r2_rel_t, seed=3)

    exp_r2_e, exp_r2_h, exp_r2_rel = _exact_coupled_covariance(
        MASS_E_M0, MASS_H_M0, K_CONF, K_INT, TEMPERATURE_K, N_BEADS
    )

    assert mean_r2_e == pytest.approx(exp_r2_e, abs=8 * sem_e), (
        f"JIT <r_e^2>={mean_r2_e:.4f} vs exact {exp_r2_e:.4f} (sem={sem_e:.4f})"
    )
    assert mean_r2_h == pytest.approx(exp_r2_h, abs=8 * sem_h), (
        f"JIT <r_h^2>={mean_r2_h:.4f} vs exact {exp_r2_h:.4f} (sem={sem_h:.4f})"
    )
    assert mean_r2_rel == pytest.approx(exp_r2_rel, abs=8 * sem_rel), (
        f"JIT <(r_e-r_h)^2>={mean_r2_rel:.4f} vs exact {exp_r2_rel:.4f} "
        f"(sem={sem_rel:.4f})"
    )
