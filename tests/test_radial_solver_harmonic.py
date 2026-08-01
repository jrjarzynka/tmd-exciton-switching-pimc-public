"""Independent analytic cross-check for tmd_pimc.radial_solver.

This test does NOT touch rk_potential.py, PIMC, or any staging/sampler code.
It validates the finite-volume radial Schroedinger solver on its own, using
the one central potential for which the 2D problem has a closed-form
solution: the isotropic harmonic oscillator.

Physics used
------------
For  -lam*[psi'' + psi'/r - m^2 psi/r^2] + 0.5*k*r^2*psi = E*psi
(equivalently -hbar^2/(2*mass)*Laplacian + 0.5*k*r^2, with
lam = HBAR2_OVER_2M0 / mass_m0), the 2D isotropic QHO separates into two
independent 1D oscillators of the same frequency omega, giving:

    hbar*omega = sqrt(2 * lam * k)
    E_{n,m}    = hbar*omega * (2*n + |m| + 1),   n = 0, 1, 2, ...

Degeneracy is 1 for m=0 and 2 for m != 0 (+m, -m).

Thermal averages (from Z_2D = Z_1D^2, Z_1D = 1/(2*sinh(beta*hbar*omega/2))):

    n_bar         = 1 / (exp(beta*hbar*omega) - 1)
    <E>           = hbar*omega * (1 + 2*n_bar)
    <r^2>         = sqrt(2*lam/k) / tanh(beta*hbar*omega/2)

These give an end-to-end check of both solve_radial_spectrum_2d (spectrum)
and thermal_radial_reference_2d (Boltzmann-weighted radial density), which
is the function used as ground truth in run_relative_rk_validation.py.

Numerically verified before writing this file (n_grid=4000, r_max=60 nm,
mass_m0=0.5, k=0.02 eV/nm^2): spectrum matches to ~1e-5 relative, thermal
<E> and <r^2> match to ~2-4e-5 relative, and the discretization error scales
as dr^2 (4x reduction per doubling of n_grid), consistent with the
finite-volume scheme's expected second-order accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from tmd_pimc.constants import HBAR2_OVER_2M0, KB_EV_PER_K
from tmd_pimc.radial_solver import (
    solve_radial_spectrum_2d,
    thermal_radial_reference_2d,
)


class HarmonicPotential2D:
    """V(r) = 0.5 * k * r^2. Satisfies the Potential2DLike protocol:
    value() takes (N, 2) Cartesian points and returns (N,) energies."""

    def __init__(self, k_eV_per_nm2: float):
        self.k = float(k_eV_per_nm2)

    def value(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        r2 = np.einsum("ij,ij->i", points, points)
        return 0.5 * self.k * r2


MASS_M0 = 0.5
K_SPRING_EV_NM2 = 0.02
LAM = HBAR2_OVER_2M0 / MASS_M0
HBAR_OMEGA = np.sqrt(2.0 * LAM * K_SPRING_EV_NM2)


def analytic_energy(n: int, m_abs: int) -> float:
    return HBAR_OMEGA * (2 * n + m_abs + 1)


@pytest.mark.parametrize("m_abs", [0, 1, 2, 3])
def test_spectrum_matches_analytic_2d_qho(m_abs: int) -> None:
    potential = HarmonicPotential2D(K_SPRING_EV_NM2)
    spectrum = solve_radial_spectrum_2d(
        mass_m0=MASS_M0,
        potential=potential,
        r_max_nm=60.0,
        n_grid=4000,
        m_abs=m_abs,
        n_states=6,
    )
    for n, energy in enumerate(spectrum.energies_eV):
        expected = analytic_energy(n, m_abs)
        rel_err = abs(energy - expected) / expected
        assert rel_err < 5.0e-4, (
            f"m={m_abs}, n={n}: numeric={energy:.8f} eV, "
            f"analytic={expected:.8f} eV, rel_err={rel_err:.3e}"
        )


@pytest.mark.parametrize("m_abs", [0, 2])
def test_ground_state_wavefunction_is_normalised_and_positive(m_abs: int) -> None:
    potential = HarmonicPotential2D(K_SPRING_EV_NM2)
    spectrum = solve_radial_spectrum_2d(
        mass_m0=MASS_M0,
        potential=potential,
        r_max_nm=60.0,
        n_grid=4000,
        m_abs=m_abs,
        n_states=1,
    )
    r = spectrum.r_nm
    psi0 = spectrum.wavefunctions_per_nm[:, 0]
    norm = 2.0 * np.pi * np.sum(r * psi0 * psi0) * spectrum.dr_nm
    assert abs(norm - 1.0) < 1.0e-6
    assert psi0[0] > 0.0
    pdf = spectrum.radial_pdfs_per_nm[:, 0]
    assert abs(np.sum(pdf) * spectrum.dr_nm - 1.0) < 1.0e-6


def test_thermal_reference_matches_analytic_2d_qho() -> None:
    """Checks thermal_radial_reference_2d end-to-end: this is the exact
    function used as the PIMC ground-truth reference in
    run_relative_rk_validation.py, so it is worth validating on its own
    against a case with a known closed-form thermal state."""
    temperature_K = 150.0  # chosen so several excited states are populated
    beta = 1.0 / (KB_EV_PER_K * temperature_K)
    n_bar = 1.0 / (np.exp(beta * HBAR_OMEGA) - 1.0)
    expected_energy = HBAR_OMEGA * (1.0 + 2.0 * n_bar)
    expected_r2 = np.sqrt(2.0 * LAM / K_SPRING_EV_NM2) / np.tanh(
        beta * HBAR_OMEGA / 2.0
    )

    potential = HarmonicPotential2D(K_SPRING_EV_NM2)
    reference = thermal_radial_reference_2d(
        mass_m0=MASS_M0,
        potential=potential,
        temperature_K=temperature_K,
        r_max_nm=80.0,
        n_grid=4000,
        m_max=14,
        states_per_m=25,
    )

    energy_rel_err = abs(reference.mean_energy_eV - expected_energy) / expected_energy
    r2_rel_err = abs(reference.mean_r2_nm2 - expected_r2) / expected_r2

    assert energy_rel_err < 1.0e-3, (
        f"<E>: numeric={reference.mean_energy_eV:.8f}, "
        f"analytic={expected_energy:.8f}, rel_err={energy_rel_err:.3e}"
    )
    assert r2_rel_err < 1.0e-3, (
        f"<r^2>: numeric={reference.mean_r2_nm2:.8f}, "
        f"analytic={expected_r2:.8f}, rel_err={r2_rel_err:.3e}"
    )
    # m_max/states_per_m were chosen generously for this T; truncation
    # should be negligible.
    assert reference.edge_weight_fraction < 1.0e-8


def test_discretization_error_scales_quadratically_with_grid_spacing() -> None:
    """Finite-volume schemes of this type are second order in dr. Confirms
    the observed 4x error reduction per doubling of n_grid (checked here at
    the m=0 ground state, r_max fixed so dr halves with n_grid)."""
    potential = HarmonicPotential2D(K_SPRING_EV_NM2)
    r_max = 60.0
    grid_sizes = [1000, 2000, 4000]
    errors = []
    for n_grid in grid_sizes:
        spectrum = solve_radial_spectrum_2d(
            mass_m0=MASS_M0,
            potential=potential,
            r_max_nm=r_max,
            n_grid=n_grid,
            m_abs=0,
            n_states=1,
        )
        errors.append(abs(spectrum.energies_eV[0] - analytic_energy(0, 0)))

    ratio_1 = errors[0] / errors[1]
    ratio_2 = errors[1] / errors[2]
    # Expect ~4x per doubling (2nd order); allow generous slack since this
    # is a regression guard, not a precision requirement.
    assert 3.0 < ratio_1 < 5.0, f"grid 1000->2000 error ratio={ratio_1:.3f}"
    assert 3.0 < ratio_2 < 5.0, f"grid 2000->4000 error ratio={ratio_2:.3f}"
