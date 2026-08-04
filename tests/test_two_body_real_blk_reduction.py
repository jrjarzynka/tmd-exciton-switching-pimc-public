"""End-to-end reduction test: the full two-body engine, running the real
(already-validated) BilayerKeldyshWallPotential as the ONLY potential
(V_e = V_h = 0, unconfined), must reduce to the independent, exact
single-body reduced-mass radial-Schroedinger reference
(radial_solver.thermal_radial_reference_2d) for the relative-coordinate
observables <(r_e-r_h)^2> and <V_int>.

This is the physical justification, not just a numerical coincidence: for
a two-body Hamiltonian where the potential depends only on r_e - r_h, the
COM (mass M = m_e + m_h) and relative (mass mu = m_e*m_h/(m_e+m_h))
coordinates separate EXACTLY -- including at the level of the discretized
ring-polymer action, since the COM/relative change of variables is an
exact linear (Jacobian = 1) transformation of the purely-quadratic
kinetic term, for any potential. So the relative-coordinate marginal of
the full two-body PIMC simulation must equal, in the P -> infinity
(continuum) limit, the single-body reduced-mass thermal density -- which
radial_solver computes essentially exactly (independent numerical method:
finite-volume Schroedinger diagonalization, not a path integral at all).

CRITICAL FINDING FROM DEVELOPING THIS TEST: Trotter (finite-P) error for
this potential is much larger than for the soft harmonic/double-well
potentials used earlier in this validation programme, because the real
BLK interaction is a much more tightly bound, higher-curvature problem
(binding energy ~0.12 eV, first excitation gap ~0.06 eV -- both large
compared to the thermal scale ~kT ~ 1.7 meV at 20 K, unlike the loosely
confined harmonic test wells used earlier). A P-convergence scan showed:

    P=24:  <r_rel^2> = 2.24 nm^2   (49.7% below the exact 4.457 nm^2 !)
    P=48:  <r_rel^2> = 3.69 nm^2
    P=96:  <r_rel^2> = 4.21 nm^2
    P=192: <r_rel^2> = 4.41 nm^2
    P=384: <r_rel^2> = 4.43 nm^2
    exact: <r_rel^2> = 4.457 nm^2  (continuum, radial_solver)

i.e. smooth, monotonic, textbook Trotter convergence -- not a bug. P=24
(used throughout the earlier harmonic/double-well tests in this
programme, purely as a fast convenient default) would be badly wrong for
this potential. Reassuringly, the actual production configs
(configs/two_body/*.json) use n_beads=256 or 512, not 24 -- so this test
uses P=256 (the smaller of the two production values, i.e. the more
conservative/stricter check) as the physically representative case.
"""
import numpy as np
import pytest

from tmd_pimc.bilayer_keldysh_potential import (
    build_bilayer_keldysh_table,
    BilayerKeldyshWallPotential,
)
from tmd_pimc.two_body_action import TwoBodyRingPolymerAction
from tmd_pimc.two_body_sampler_jit import TwoBodyPIMCSamplerStagingJIT
from tmd_pimc.potentials import CompositePotential
from tmd_pimc.radial_solver import thermal_radial_reference_2d

MASS_E_M0 = 0.40
MASS_H_M0 = 0.70
TEMPERATURE_K = 20.0
N_BEADS_PRODUCTION = 256  # matches configs/two_body/*.json (256 or 512)


def _make_blk_interaction():
    """Production-like BLK parameters (same as
    test_two_body_interaction_table.py / test_bilayer_keldysh_potential.py)."""
    table = build_bilayer_keldysh_table(
        separation_nm=0.65, screening_length_layer1_nm=4.0,
        screening_length_layer2_nm=4.5, kappa_environment=4.0,
        r_max_nm=80.0, n_log=1000, n_linear=2000,
    )
    return BilayerKeldyshWallPotential(
        bilayer=table, wall_radius_nm=15.0, wall_height_eV=0.08, wall_power=8,
    )


def _block_bootstrap_sem(x, block_size=100, nboot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = max(1, n // block_size)
    blocks = [x[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    block_means = np.array([b.mean() for b in blocks if len(b) > 0])
    means = [np.mean(block_means[rng.integers(0, len(block_means), len(block_means))])
             for _ in range(nboot)]
    return float(np.std(means, ddof=1))


def _run_unconfined_pimc(interaction, n_beads, n_steps, burn_in, seed, start_sep_nm=1.0):
    action = TwoBodyRingPolymerAction(
        mass_e_m0=MASS_E_M0, mass_h_m0=MASS_H_M0, temperature_K=TEMPERATURE_K,
        n_beads=n_beads,
        potential_e=CompositePotential(terms=[]), potential_h=CompositePotential(terms=[]),
        potential_interaction=interaction,
    )
    seglens = tuple(L for L in (4, 8, 16, 32, 64, 128) if L < n_beads) or (n_beads - 1,)
    sampler = TwoBodyPIMCSamplerStagingJIT(
        action, local_step_nm=0.15, global_step_nm=1.0,
        global_move_probability=0.0,  # COM translation only; irrelevant to r_rel, kept off
        rng_seed=seed, staging_segment_lengths=seglens, staging_moves_per_step=2,
        rasterize_e=False, rasterize_h=False,  # true V_e=V_h=0, no artificial box
    )
    result = sampler.run(n_steps=n_steps, burn_in=burn_in, sample_every=20,
                          center_e=(0.0, 0.0), center_h=(start_sep_nm, 0.0))
    rel = result["samples_e"] - result["samples_h"]
    return rel, result["acceptance_staging"]


def test_production_beads_reduces_to_exact_reduced_mass_reference():
    """Main test: at the actual production Trotter number (P=256), the full
    two-body engine's relative-coordinate observables must match the exact
    reduced-mass radial-Schroedinger reference to within a few percent
    (small residual Trotter bias expected and budgeted for; see module
    docstring for the measured P-convergence trend)."""
    interaction = _make_blk_interaction()
    mu = MASS_E_M0 * MASS_H_M0 / (MASS_E_M0 + MASS_H_M0)

    ref = thermal_radial_reference_2d(
        mass_m0=mu, potential=interaction, temperature_K=TEMPERATURE_K,
        r_max_nm=40.0, n_grid=2000, m_max=6, states_per_m=15,
    )
    assert ref.edge_weight_fraction < 1e-6, "radial solver reference not converged (edge leakage)"

    rel, staging_acc = _run_unconfined_pimc(
        interaction, N_BEADS_PRODUCTION, n_steps=200_000, burn_in=40_000, seed=321,
    )
    r2_rel_t = np.mean(np.sum(rel**2, axis=-1), axis=1)
    Vint_t = np.mean(interaction.value(rel.reshape(-1, 2)).reshape(rel.shape[0], rel.shape[1]), axis=1)

    mean_r2, sem_r2 = float(np.mean(r2_rel_t)), _block_bootstrap_sem(r2_rel_t)
    mean_V, sem_V = float(np.mean(Vint_t)), _block_bootstrap_sem(Vint_t)

    # Tolerance = max(a few-percent budget for residual Trotter bias at
    # P=256, generous multiple of the MC error bar). Not a pure-noise
    # (sigma-only) test: there IS a small, real, expected systematic
    # offset here (radial_solver is the P->infinity continuum answer),
    # unlike the exactly-solvable harmonic tests earlier in this programme.
    tol_r2 = max(0.03 * ref.mean_r2_nm2, 8 * sem_r2)
    tol_V = max(0.03 * abs(ref.mean_potential_eV), 8 * sem_V)

    assert mean_r2 == pytest.approx(ref.mean_r2_nm2, abs=tol_r2), (
        f"P={N_BEADS_PRODUCTION}: PIMC <r_rel^2>={mean_r2:.4f}+/-{sem_r2:.4f} nm^2 vs "
        f"exact {ref.mean_r2_nm2:.4f} nm^2 (staging_acc={staging_acc:.3f})"
    )
    assert mean_V == pytest.approx(ref.mean_potential_eV, abs=tol_V), (
        f"P={N_BEADS_PRODUCTION}: PIMC <V_int>={mean_V:.5f}+/-{sem_V:.5f} eV vs "
        f"exact {ref.mean_potential_eV:.5f} eV"
    )


def test_result_is_independent_of_starting_separation():
    """Guards against the discrepancy being an equilibration/initial-condition
    artifact rather than a genuine (converged) thermal average: two very
    different starting separations must agree with each other at P=256."""
    interaction = _make_blk_interaction()
    rel_close, _ = _run_unconfined_pimc(
        interaction, N_BEADS_PRODUCTION, n_steps=120_000, burn_in=25_000,
        seed=11, start_sep_nm=0.05,
    )
    rel_far, _ = _run_unconfined_pimc(
        interaction, N_BEADS_PRODUCTION, n_steps=120_000, burn_in=25_000,
        seed=13, start_sep_nm=8.0,
    )
    r2_close_t = np.mean(np.sum(rel_close**2, axis=-1), axis=1)
    r2_far_t = np.mean(np.sum(rel_far**2, axis=-1), axis=1)
    mean_close, sem_close = float(np.mean(r2_close_t)), _block_bootstrap_sem(r2_close_t)
    mean_far, sem_far = float(np.mean(r2_far_t)), _block_bootstrap_sem(r2_far_t)

    assert mean_close == pytest.approx(mean_far, abs=8 * np.hypot(sem_close, sem_far)), (
        f"start_sep=0.05nm gave <r_rel^2>={mean_close:.4f}+/-{sem_close:.4f}, "
        f"start_sep=8.0nm gave <r_rel^2>={mean_far:.4f}+/-{sem_far:.4f} -- "
        f"result depends on initial condition, run is not converged"
    )


def test_p_convergence_trend_is_monotonic_toward_exact_value():
    """Cheap regression guard on the Trotter-convergence direction itself
    (documented in the module docstring): coarser P must NOT overshoot or
    diverge away from the exact answer -- a fast, P=24-vs-96 check that
    would catch a future regression in tau/kpf wiring without needing the
    expensive P=256 production-scale run above."""
    interaction = _make_blk_interaction()
    mu = MASS_E_M0 * MASS_H_M0 / (MASS_E_M0 + MASS_H_M0)
    ref = thermal_radial_reference_2d(
        mass_m0=mu, potential=interaction, temperature_K=TEMPERATURE_K,
        r_max_nm=40.0, n_grid=2000, m_max=6, states_per_m=15,
    )

    rel_p24, _ = _run_unconfined_pimc(interaction, 24, n_steps=80_000, burn_in=15_000, seed=7)
    rel_p96, _ = _run_unconfined_pimc(interaction, 96, n_steps=80_000, burn_in=15_000, seed=7)
    r2_p24 = float(np.mean(np.sum(rel_p24**2, axis=-1)))
    r2_p96 = float(np.mean(np.sum(rel_p96**2, axis=-1)))

    assert r2_p24 < r2_p96 < ref.mean_r2_nm2 * 1.05, (
        f"expected monotonic increase toward the exact value: "
        f"P=24 -> {r2_p24:.3f}, P=96 -> {r2_p96:.3f}, exact -> {ref.mean_r2_nm2:.3f}"
    )
