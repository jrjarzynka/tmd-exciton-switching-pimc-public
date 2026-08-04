"""Test #2 in the two-body validation programme: coupled harmonic e-h pair.

Both electron and hole sit in their own confining harmonic well (same
spring constant, same center -- so the combined system has a single basin
and no need for a global move), AND interact through a harmonic spring
V_int(r_e - r_h) = (1/2) k_int |r_e - r_h|^2.

This is the first test in the programme where the interaction actually
couples the two ring polymers (test #1 checked V_int = 0). Because every
term in the action is quadratic, the full discretized (finite-P) path
integral is an exactly solvable Gaussian -- no continuum/Trotter
approximation is needed for the reference, unlike a generic interaction.

Reference method
-----------------
For a single Cartesian component (x say; y is independent and identical
by isotropy), the discretized primitive action is

    S = kpf_e * x_e^T C x_e + kpf_h * x_h^T C x_h
        + (tau*k_conf/2) * (x_e^T x_e + x_h^T x_h)
        + (tau*k_int/2)  * (x_e - x_h)^T (x_e - x_h)

where C is the standard cyclic ring matrix (2 on the diagonal, -1 on the
cyclic off-diagonals) satisfying x^T C x = sum_j (x_j - x_{j+1})^2 -- the
same discretization already used (and validated) for the single-body
harmonic benchmark in analytic.harmonic_r2_primitive_finite_P.

Writing Z = (x_e, x_h) (a 2P vector), S = Z^T M Z for a symmetric 2P x 2P
block matrix M, exp(-S) = exp(-(1/2) Z^T Lambda Z) with Lambda = 2M, so
Cov(Z) = Lambda^{-1} = (2M)^{-1} exactly (for this finite-P discretization,
to floating-point precision, no Monte Carlo involved). This exact
reference is checked to reduce to the already-validated
harmonic_r2_primitive_finite_P in the k_int -> 0 limit before being
trusted (see test_exact_reference_reduces_to_independent_case_at_kint_zero
below).
"""
import numpy as np
import pytest

from tmd_pimc.constants import HBAR2_OVER_2M0, KB_EV_PER_K
from tmd_pimc.analytic import harmonic_r2_primitive_finite_P
from tmd_pimc.two_body_action import TwoBodyRingPolymerAction
from tmd_pimc.two_body_sampler import TwoBodyPIMCSamplerStaging
from tmd_pimc.potentials import HarmonicPotential

MASS_E_M0 = 0.40
MASS_H_M0 = 0.70
K_CONF = 0.0015   # eV/nm^2, same confinement for e and h (shared center)
K_INT = 0.0200    # eV/nm^2, e-h harmonic coupling
TEMPERATURE_K = 20.0
N_BEADS = 24


def _exact_coupled_covariance(mass_e_m0, mass_h_m0, k_conf, k_int, T, P):
    """Exact single-Cartesian-component covariance matrix Cov(Z), Z=(x_e,x_h),
    for the discretized (finite-P) primitive action of the coupled model.
    Returns (r2_e, r2_h, r2_rel, cross_cov_xexh)."""
    lam_e = HBAR2_OVER_2M0 / mass_e_m0
    lam_h = HBAR2_OVER_2M0 / mass_h_m0
    beta = 1.0 / (KB_EV_PER_K * T)
    tau = beta / P
    kpf_e = 1.0 / (4.0 * lam_e * tau)
    kpf_h = 1.0 / (4.0 * lam_h * tau)

    C = 2 * np.eye(P) - np.eye(P, k=1) - np.eye(P, k=-1)
    C[0, -1] -= 1.0
    C[-1, 0] -= 1.0  # close the cyclic wraparound bond

    diag_e = kpf_e * C + (tau * k_conf / 2.0 + tau * k_int / 2.0) * np.eye(P)
    diag_h = kpf_h * C + (tau * k_conf / 2.0 + tau * k_int / 2.0) * np.eye(P)
    off = -(tau * k_int / 2.0) * np.eye(P)

    M = np.block([[diag_e, off], [off, diag_h]])
    cov = np.linalg.inv(2.0 * M)

    var_xe = float(np.mean(np.diag(cov)[:P]))
    var_xh = float(np.mean(np.diag(cov)[P:]))
    cross = float(np.mean(np.diag(cov[:P, P:])))

    r2_e = 2.0 * var_xe    # isotropic 2D: <r^2> = <x^2> + <y^2> = 2<x^2>
    r2_h = 2.0 * var_xh
    r2_rel = r2_e + r2_h - 4.0 * cross   # <(x_e-x_h)^2+(y_e-y_h)^2>
    return r2_e, r2_h, r2_rel, cross


# --- 0. Trust the reference method itself before using it ----------------

def test_exact_reference_reduces_to_independent_case_at_kint_zero():
    """Sanity check on the exact-covariance machinery, not on the PIMC code:
    with k_int=0, it must reduce to the already-validated single-body
    finite-P harmonic result to machine precision."""
    r2_e, r2_h, r2_rel, cross = _exact_coupled_covariance(
        MASS_E_M0, MASS_H_M0, K_CONF, 0.0, TEMPERATURE_K, N_BEADS
    )
    ref_e = harmonic_r2_primitive_finite_P(MASS_E_M0, K_CONF, TEMPERATURE_K, N_BEADS)
    ref_h = harmonic_r2_primitive_finite_P(MASS_H_M0, K_CONF, TEMPERATURE_K, N_BEADS)
    assert r2_e == pytest.approx(ref_e, rel=1e-10)
    assert r2_h == pytest.approx(ref_h, rel=1e-10)
    assert cross == pytest.approx(0.0, abs=1e-12)


# --- 1. Deterministic wiring check: interaction must now couple the chains

def test_delta_action_e_now_depends_on_hole_path():
    """Complement of the test-#1 independence check: with k_int != 0,
    moving an electron bead MUST give a different delta-action for
    different (fixed) hole configurations -- if it didn't, the interaction
    term would not actually be wired into delta_action_bead_move_e."""
    action = TwoBodyRingPolymerAction(
        mass_e_m0=MASS_E_M0, mass_h_m0=MASS_H_M0, temperature_K=TEMPERATURE_K,
        n_beads=N_BEADS,
        potential_e=HarmonicPotential(k_eV_per_nm2=K_CONF),
        potential_h=HarmonicPotential(k_eV_per_nm2=K_CONF),
        potential_interaction=HarmonicPotential(k_eV_per_nm2=K_INT),
    )
    rng = np.random.default_rng(1)
    path_e = 0.3 * rng.standard_normal((N_BEADS, 2))
    path_h_a = 0.3 * rng.standard_normal((N_BEADS, 2))
    path_h_b = 2.0 * rng.standard_normal((N_BEADS, 2))
    j = 3
    r_new = path_e[j] + 0.1 * rng.standard_normal(2)

    dS_a = action.delta_action_bead_move_e(path_e, path_h_a, j, r_new)
    dS_b = action.delta_action_bead_move_e(path_e, path_h_b, j, r_new)
    assert abs(dS_a - dS_b) > 1e-6, (
        "delta-action for an electron move did not change when the hole "
        "path changed -- the interaction term looks disconnected."
    )


def test_attractive_coupling_gives_positive_cross_correlation():
    """Sign/wiring check via the exact reference: an attractive e-h spring
    must produce POSITIVE <x_e x_h> correlation (they are pulled to move
    together). A sign error in how r_e - r_h enters the interaction
    (e.g. accidentally using r_e + r_h somewhere) would flip this."""
    _, _, _, cross = _exact_coupled_covariance(
        MASS_E_M0, MASS_H_M0, K_CONF, K_INT, TEMPERATURE_K, N_BEADS
    )
    assert cross > 0.0


# --- 2. Statistical check: full sampler vs exact finite-P reference ------

def test_thermal_r2_matches_exact_coupled_finite_P_reference():
    action = TwoBodyRingPolymerAction(
        mass_e_m0=MASS_E_M0, mass_h_m0=MASS_H_M0, temperature_K=TEMPERATURE_K,
        n_beads=N_BEADS,
        potential_e=HarmonicPotential(k_eV_per_nm2=K_CONF),
        potential_h=HarmonicPotential(k_eV_per_nm2=K_CONF),
        potential_interaction=HarmonicPotential(k_eV_per_nm2=K_INT),
    )
    sampler = TwoBodyPIMCSamplerStaging(
        action=action,
        local_step_nm=0.25,
        global_step_nm=1.0,
        global_move_probability=0.0,   # single shared basin, no global move needed
        rng_seed=7,
        staging_segment_lengths=(4, 8, 12),
        staging_moves_per_step=2,
    )
    result = sampler.run(
        n_steps=60_000, burn_in=10_000, sample_every=10,
        center_e=(0.0, 0.0), center_h=(0.0, 0.0),
    )
    samples_e, samples_h = result["samples_e"], result["samples_h"]

    r2_e_t = np.mean(np.sum(samples_e**2, axis=-1), axis=1)
    r2_h_t = np.mean(np.sum(samples_h**2, axis=-1), axis=1)
    rel_t = samples_e - samples_h
    r2_rel_t = np.mean(np.sum(rel_t**2, axis=-1), axis=1)

    def bootstrap_sem(x, nboot=1000, seed=0):
        rng_b = np.random.default_rng(seed)
        n = len(x)
        means = [np.mean(x[rng_b.integers(0, n, n)]) for _ in range(nboot)]
        return float(np.std(means, ddof=1))

    mean_r2_e, sem_r2_e = float(np.mean(r2_e_t)), bootstrap_sem(r2_e_t, seed=1)
    mean_r2_h, sem_r2_h = float(np.mean(r2_h_t)), bootstrap_sem(r2_h_t, seed=2)
    mean_r2_rel, sem_r2_rel = float(np.mean(r2_rel_t)), bootstrap_sem(r2_rel_t, seed=3)

    exp_r2_e, exp_r2_h, exp_r2_rel, _ = _exact_coupled_covariance(
        MASS_E_M0, MASS_H_M0, K_CONF, K_INT, TEMPERATURE_K, N_BEADS
    )

    # 6-sigma tolerance: generous against autocorrelation-underestimated
    # bootstrap error bars, still tight enough to fail hard on a real bug
    # (wiring/sign errors here produce O(1) discrepancies, not O(1 sigma)).
    assert mean_r2_e == pytest.approx(exp_r2_e, abs=6 * sem_r2_e), (
        f"<r_e^2>={mean_r2_e:.4f} vs exact {exp_r2_e:.4f} (sem={sem_r2_e:.4f})"
    )
    assert mean_r2_h == pytest.approx(exp_r2_h, abs=6 * sem_r2_h), (
        f"<r_h^2>={mean_r2_h:.4f} vs exact {exp_r2_h:.4f} (sem={sem_r2_h:.4f})"
    )
    assert mean_r2_rel == pytest.approx(exp_r2_rel, abs=6 * sem_r2_rel), (
        f"<(r_e-r_h)^2>={mean_r2_rel:.4f} vs exact {exp_r2_rel:.4f} "
        f"(sem={sem_r2_rel:.4f})"
    )
