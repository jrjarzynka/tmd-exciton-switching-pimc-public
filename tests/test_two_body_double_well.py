"""Test #3 in the two-body validation programme: double-well landscape.

Electron and hole each sit in their own DoubleGaussianWellPotential
(V_int = 0, i.e. still decoupled -- see test #1 -- but now the single-body
landscape is genuinely non-harmonic and multi-well). This is the first
test that exercises:

  - correct sampling of a non-convex, multi-modal potential (harmonic
    tests can't catch e.g. a sampler that gets stuck in one well without
    working global moves -- the exact "basin lottery" pathology already
    documented for the moire landscape in this project's own history,
    where the global move step must be tuned to the physical inter-well
    spacing for ergodic sampling);
  - an asymmetric double well (hole), giving a non-trivial (not 50/50)
    thermal population split between wells, checked against an exact
    reference.

Reference method
-----------------
Unlike the harmonic tests, there is no closed-form solution here. Instead
the reference is obtained by an entirely different numerical method to
the ring-polymer PIMC being validated: direct finite-difference
diagonalization of the single-particle 2D Hamiltonian on a real-space
grid (scipy.sparse.linalg.eigsh for the lowest eigenstates), followed by
an exact thermal (Boltzmann) average over those eigenstates. Convergence
in the number of retained eigenstates is checked explicitly. This is a
genuine cross-check: two independent numerical methods (path integral
Monte Carlo vs. direct Schroedinger-equation diagonalization) must agree
on the same physical observables.
"""
import numpy as np
import pytest
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import eigsh

from tmd_pimc.constants import HBAR2_OVER_2M0, KB_EV_PER_K
from tmd_pimc.potentials import DoubleGaussianWellPotential
from tmd_pimc.two_body_action import TwoBodyRingPolymerAction
from tmd_pimc.two_body_sampler import TwoBodyPIMCSamplerStaging

MASS_E_M0 = 0.40
MASS_H_M0 = 0.70
TEMPERATURE_K = 20.0
N_BEADS = 24

V_E = DoubleGaussianWellPotential(V0_eV=0.05, sigma_nm=3.0, separation_nm=10.0)
# Hole: deeper wells + asymmetry -> non-trivial (not 50/50) thermal
# population split, a stronger test than a symmetric well would be.
V_H = DoubleGaussianWellPotential(V0_eV=0.06, sigma_nm=3.0, separation_nm=10.0,
                                   asymmetry_eV=0.015)

GRID_L_NM = 40.0
GRID_N = 128
N_EIGENSTATES = 12  # verified converged well beyond this, see module test


class _ZeroInteraction:
    def value(self, r):
        r = np.asarray(r)
        return np.zeros(r.shape[0])


def _grid_eigenstates(mass_m0, potential, L_nm=GRID_L_NM, N=GRID_N, n_states=N_EIGENSTATES):
    lam = HBAR2_OVER_2M0 / mass_m0
    x = np.linspace(-L_nm / 2.0, L_nm / 2.0, N)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    V = potential.value(pts).reshape(N, N)

    main = -2.0 * np.ones(N)
    off = np.ones(N - 1)
    lap_1d = diags([off, main, off], [-1, 0, 1]) / dx**2
    identity_1d = identity(N)
    lap_2d = kron(lap_1d, identity_1d) + kron(identity_1d, lap_1d)

    H = -lam * lap_2d + diags(V.ravel())
    evals, evecs = eigsh(H, k=n_states, which="SA")
    return evals, evecs, X, Y, dx


def _thermal_r2_and_frac_right(evals, evecs, X, Y, dx, temperature_K, x_split=0.0):
    beta = 1.0 / (KB_EV_PER_K * temperature_K)
    weights = np.exp(-beta * (evals - evals[0]))
    Z = np.sum(weights)
    r2 = np.empty(len(evals))
    frac_right = np.empty(len(evals))
    for n in range(len(evals)):
        density = evecs[:, n] ** 2 / dx**2  # |psi(x,y)|^2 on the grid
        r2[n] = np.sum((X.ravel() ** 2 + Y.ravel() ** 2) * density) * dx**2
        frac_right[n] = np.sum(density[X.ravel() > x_split]) * dx**2
    return float(np.sum(weights * r2) / Z), float(np.sum(weights * frac_right) / Z)


def _exact_reference(mass_m0, potential, x_split=0.0):
    evals, evecs, X, Y, dx = _grid_eigenstates(mass_m0, potential)
    return _thermal_r2_and_frac_right(evals, evecs, X, Y, dx, TEMPERATURE_K, x_split)


# --- 0. Trust the reference method: check convergence with n_states ------

def test_grid_eigensolver_reference_is_converged():
    """Population/​<r^2> must not change when more eigenstates are kept --
    otherwise the 'exact' reference below would itself be unreliable."""
    r2_12, fr_12 = _exact_reference(MASS_E_M0, V_E)
    evals, evecs, X, Y, dx = _grid_eigenstates(MASS_E_M0, V_E, n_states=20)
    r2_20, fr_20 = _thermal_r2_and_frac_right(evals, evecs, X, Y, dx, TEMPERATURE_K)
    assert r2_20 == pytest.approx(r2_12, abs=1e-4)
    assert fr_20 == pytest.approx(fr_12, abs=1e-6)


def test_symmetric_well_reference_gives_50_50_population():
    """Sanity check on the reference pipeline itself: a perfectly symmetric
    double well must give exactly a 50/50 population split."""
    _, frac_right = _exact_reference(MASS_E_M0, V_E)
    assert frac_right == pytest.approx(0.5, abs=1e-6)


# --- 1. Full PIMC run vs exact grid-diagonalization reference ------------

def _block_bootstrap_sem(x, block_size=50, nboot=1000, seed=0):
    """Block bootstrap SEM: resamples contiguous blocks rather than single
    samples, since well-population observables are autocorrelated over
    many samples near a tunnelling-limited transition rate."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = max(1, n // block_size)
    blocks = [x[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    block_means = np.array([b.mean() for b in blocks if len(b) > 0])
    means = [np.mean(block_means[rng.integers(0, len(block_means), len(block_means))])
             for _ in range(nboot)]
    return float(np.std(means, ddof=1))


def test_double_well_populations_and_r2_match_exact_diagonalization():
    action = TwoBodyRingPolymerAction(
        mass_e_m0=MASS_E_M0, mass_h_m0=MASS_H_M0, temperature_K=TEMPERATURE_K,
        n_beads=N_BEADS, potential_e=V_E, potential_h=V_H,
        potential_interaction=_ZeroInteraction(),
    )
    sampler = TwoBodyPIMCSamplerStaging(
        action=action,
        local_step_nm=0.3,
        global_step_nm=10.0,          # matched to separation_nm -- see
                                       # module docstring on basin lottery
        global_move_probability=0.3,
        rng_seed=11,
        staging_segment_lengths=(4, 8, 12),
        staging_moves_per_step=2,
    )
    result = sampler.run(
        n_steps=80_000, burn_in=15_000, sample_every=10,
        center_e=(-5.0, 0.0), center_h=(-5.0, 0.0),
    )
    samples_e, samples_h = result["samples_e"], result["samples_h"]

    r2_e_t = np.mean(np.sum(samples_e**2, axis=-1), axis=1)
    r2_h_t = np.mean(np.sum(samples_h**2, axis=-1), axis=1)
    # per-sample well assignment via centroid x-position (bead-averaged)
    right_e_t = (np.mean(samples_e[:, :, 0], axis=1) > 0.0).astype(float)
    right_h_t = (np.mean(samples_h[:, :, 0], axis=1) > 0.0).astype(float)

    mean_r2_e, sem_r2_e = float(np.mean(r2_e_t)), _block_bootstrap_sem(r2_e_t, seed=1)
    mean_r2_h, sem_r2_h = float(np.mean(r2_h_t)), _block_bootstrap_sem(r2_h_t, seed=2)
    mean_fr_e, sem_fr_e = float(np.mean(right_e_t)), _block_bootstrap_sem(right_e_t, seed=3)
    mean_fr_h, sem_fr_h = float(np.mean(right_h_t)), _block_bootstrap_sem(right_h_t, seed=4)

    exact_r2_e, exact_fr_e = _exact_reference(MASS_E_M0, V_E)
    exact_r2_h, exact_fr_h = _exact_reference(MASS_H_M0, V_H)

    # 8-sigma + absolute floor: population-fraction observables near a
    # tunnelling-limited transition rate have long, hard-to-estimate
    # autocorrelation times, so the tolerance is deliberately generous.
    # A real bug here (e.g. broken global move, mixed-up e/h chains,
    # wrong asymmetry sign) produces O(0.1-1) discrepancies, far above
    # this floor.
    assert mean_r2_e == pytest.approx(exact_r2_e, abs=max(8 * sem_r2_e, 1.5))
    assert mean_r2_h == pytest.approx(exact_r2_h, abs=max(8 * sem_r2_h, 1.5))
    assert mean_fr_e == pytest.approx(exact_fr_e, abs=max(8 * sem_fr_e, 0.05))
    assert mean_fr_h == pytest.approx(exact_fr_h, abs=max(8 * sem_fr_h, 0.05))
