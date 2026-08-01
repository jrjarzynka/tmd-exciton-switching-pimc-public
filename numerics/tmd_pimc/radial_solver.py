"""Independent finite-volume radial Schroedinger solver for 2D central potentials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.linalg import eigh_tridiagonal

from .constants import HBAR2_OVER_2M0, KB_EV_PER_K


class Potential2DLike(Protocol):
    def value(self, r: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RadialSpectrum2D:
    m_abs: int
    r_nm: np.ndarray
    dr_nm: float
    potential_eV: np.ndarray
    energies_eV: np.ndarray
    wavefunctions_per_nm: np.ndarray
    radial_pdfs_per_nm: np.ndarray


@dataclass(frozen=True)
class ThermalRadialReference2D:
    temperature_K: float
    r_nm: np.ndarray
    dr_nm: float
    radial_pdf_per_nm: np.ndarray
    ground_energy_eV: float
    first_excited_energy_eV: float
    excitation_gap_eV: float
    ground_state_weight: float
    mean_energy_eV: float
    mean_potential_eV: float
    mean_r_nm: float
    mean_r2_nm2: float
    edge_weight_fraction: float
    included_state_count: int


def evaluate_radial_potential(potential: Potential2DLike, r_nm: np.ndarray) -> np.ndarray:
    """Evaluate a Potential2D object along the positive x axis."""
    r = np.asarray(r_nm, dtype=float)
    points = np.column_stack([r, np.zeros_like(r)])
    values = np.asarray(potential.value(points), dtype=float)
    if values.shape != r.shape:
        raise ValueError(f"Potential returned shape {values.shape}; expected {r.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Potential returned non-finite values")
    return values


def solve_radial_spectrum_2d(
    *,
    mass_m0: float,
    potential: Potential2DLike,
    r_max_nm: float,
    n_grid: int,
    m_abs: int = 0,
    n_states: int = 12,
) -> RadialSpectrum2D:
    """
    Solve the 2D radial Schroedinger equation for fixed |m|.

    A cell-centred finite-volume discretisation is used for

        -lambda [psi'' + psi'/r - m^2 psi/r^2] + V psi = E psi.

    Regularity at r=0 is imposed by zero radial flux.  The outer boundary is
    Dirichlet at r=r_max.  The resulting generalised eigenproblem is
    symmetrised and solved as a real tridiagonal problem.
    """
    if mass_m0 <= 0.0:
        raise ValueError("mass_m0 must be positive")
    if r_max_nm <= 0.0:
        raise ValueError("r_max_nm must be positive")
    if n_grid < 100:
        raise ValueError("n_grid must be >= 100")
    if m_abs < 0:
        raise ValueError("m_abs must be non-negative")
    if n_states < 1 or n_states >= n_grid:
        raise ValueError("n_states must satisfy 1 <= n_states < n_grid")

    lam = HBAR2_OVER_2M0 / float(mass_m0)
    dr = float(r_max_nm) / int(n_grid)
    r = (np.arange(int(n_grid), dtype=float) + 0.5) * dr
    r_minus = np.arange(int(n_grid), dtype=float) * dr
    r_plus = (np.arange(int(n_grid), dtype=float) + 1.0) * dr
    V = evaluate_radial_potential(potential, r)

    diagonal_A = (
        lam * (r_plus + r_minus) / (dr * dr)
        + r * V
        + lam * float(m_abs * m_abs) / r
    )
    # Dirichlet condition at the outer cell face: ghost value = -psi_last.
    diagonal_A[-1] = (
        lam * (2.0 * r_plus[-1] + r_minus[-1]) / (dr * dr)
        + r[-1] * V[-1]
        + lam * float(m_abs * m_abs) / r[-1]
    )
    off_diagonal_A = -lam * r_plus[:-1] / (dr * dr)

    diagonal = diagonal_A / r
    off_diagonal = off_diagonal_A / np.sqrt(r[:-1] * r[1:])

    energies, transformed = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, int(n_states) - 1),
        check_finite=False,
        tol=1.0e-12,
    )

    wavefunctions = np.empty_like(transformed)
    radial_pdfs = np.empty_like(transformed)
    for state in range(int(n_states)):
        psi = transformed[:, state] / np.sqrt(r)
        if psi[0] < 0.0:
            psi = -psi
        norm = np.sqrt(2.0 * np.pi * np.sum(r * psi * psi) * dr)
        psi = psi / norm
        radial_pdf = 2.0 * np.pi * r * psi * psi
        # Remove tiny accumulated quadrature error.
        radial_pdf = radial_pdf / (np.sum(radial_pdf) * dr)
        wavefunctions[:, state] = psi
        radial_pdfs[:, state] = radial_pdf

    return RadialSpectrum2D(
        m_abs=int(m_abs),
        r_nm=r,
        dr_nm=dr,
        potential_eV=V,
        energies_eV=np.asarray(energies, dtype=float),
        wavefunctions_per_nm=wavefunctions,
        radial_pdfs_per_nm=radial_pdfs,
    )


def thermal_radial_reference_2d(
    *,
    mass_m0: float,
    potential: Potential2DLike,
    temperature_K: float,
    r_max_nm: float,
    n_grid: int,
    m_max: int = 8,
    states_per_m: int = 20,
) -> ThermalRadialReference2D:
    """Construct the finite-temperature radial density from a spectral sum."""
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if m_max < 0:
        raise ValueError("m_max must be non-negative")

    spectra: list[RadialSpectrum2D] = []
    records: list[tuple[float, int, int, int, np.ndarray]] = []

    for m_abs in range(int(m_max) + 1):
        spectrum = solve_radial_spectrum_2d(
            mass_m0=mass_m0,
            potential=potential,
            r_max_nm=r_max_nm,
            n_grid=n_grid,
            m_abs=m_abs,
            n_states=states_per_m,
        )
        spectra.append(spectrum)
        degeneracy = 1 if m_abs == 0 else 2
        for radial_index, energy in enumerate(spectrum.energies_eV):
            records.append(
                (
                    float(energy),
                    degeneracy,
                    m_abs,
                    radial_index,
                    spectrum.radial_pdfs_per_nm[:, radial_index],
                )
            )

    records.sort(key=lambda item: item[0])
    ground_energy = records[0][0]
    first_excited_energy = records[1][0]
    beta = 1.0 / (KB_EV_PER_K * float(temperature_K))
    raw_weights = np.array(
        [deg * np.exp(-beta * (energy - ground_energy)) for energy, deg, *_ in records],
        dtype=float,
    )
    partition = float(np.sum(raw_weights))
    weights = raw_weights / partition

    radial_pdf = np.zeros_like(spectra[0].r_nm)
    mean_energy = 0.0
    edge_weight = 0.0
    for weight, (energy, _deg, m_abs, radial_index, state_pdf) in zip(weights, records):
        radial_pdf += weight * state_pdf
        mean_energy += weight * energy
        if m_abs == int(m_max) or radial_index == int(states_per_m) - 1:
            edge_weight += weight

    r = spectra[0].r_nm
    dr = spectra[0].dr_nm
    radial_pdf /= np.sum(radial_pdf) * dr
    V = spectra[0].potential_eV

    return ThermalRadialReference2D(
        temperature_K=float(temperature_K),
        r_nm=r,
        dr_nm=dr,
        radial_pdf_per_nm=radial_pdf,
        ground_energy_eV=float(ground_energy),
        first_excited_energy_eV=float(first_excited_energy),
        excitation_gap_eV=float(first_excited_energy - ground_energy),
        ground_state_weight=float(weights[0]),
        mean_energy_eV=float(mean_energy),
        mean_potential_eV=float(np.sum(radial_pdf * V) * dr),
        mean_r_nm=float(np.sum(radial_pdf * r) * dr),
        mean_r2_nm2=float(np.sum(radial_pdf * r * r) * dr),
        edge_weight_fraction=float(edge_weight),
        included_state_count=len(records),
    )
