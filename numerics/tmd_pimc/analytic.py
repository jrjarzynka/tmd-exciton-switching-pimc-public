"""
Exact analytical solutions for benchmark potentials.

Used in v0.7 to validate the PIMC sampler against known results.
"""

import numpy as np
from .constants import HBAR2_OVER_2M0, KB_EV_PER_K


def harmonic_hbar_omega(mass_m0: float, k_eV_per_nm2: float) -> float:
    """
    Characteristic energy ħω for a 2D harmonic oscillator.

    V(r) = (1/2) k r²
    ω    = sqrt(k / m)
    ħω   = sqrt(ħ²k / m) = sqrt(2 λ k)

    where λ = ħ²/(2m) = HBAR2_OVER_2M0 / mass_m0.
    """
    lam = HBAR2_OVER_2M0 / mass_m0          # eV·nm²
    return float(np.sqrt(2.0 * lam * k_eV_per_nm2))  # eV


def harmonic_r2_analytic(mass_m0: float,
                          k_eV_per_nm2: float,
                          temperature_K: float) -> float:
    """
    Exact thermal expectation value <r²> for the 2D isotropic harmonic
    oscillator at temperature T.

    Derivation
    ----------
    The partition function for a 2D harmonic oscillator factorises into
    two independent 1D oscillators, each contributing:

        <x²> = (ħ / 2mω) coth(βħω / 2)

    so:
        <r²> = <x²> + <y²> = (ħ / mω) coth(βħω / 2)

    In terms of λ = ħ²/(2m):
        ħ/(mω) = 2λ/(ħω)

    Limits
    ------
    T → 0  :  <r²> → ħ/(mω)           (zero-point motion)
    T → ∞  :  <r²> → 2 k_B T / (mω²)  = 2 k_B T / k  (classical equipartition)

    Parameters
    ----------
    mass_m0       : exciton effective mass in units of electron mass m₀
    k_eV_per_nm2  : harmonic spring constant  V = ½k r²  [eV/nm²]
    temperature_K : temperature [K]

    Returns
    -------
    <r²>  in nm²
    """
    lam        = HBAR2_OVER_2M0 / mass_m0          # eV·nm²
    hbar_omega = np.sqrt(2.0 * lam * k_eV_per_nm2) # eV
    beta       = 1.0 / (KB_EV_PER_K * temperature_K)
    x          = 0.5 * beta * hbar_omega            # dimensionless βħω/2
    r2         = (2.0 * lam / hbar_omega) / np.tanh(x)
    return float(r2)


def harmonic_r2_classical(k_eV_per_nm2: float,
                           temperature_K: float) -> float:
    """
    Classical (high-T) limit: <r²> = 2 k_B T / k.
    Useful as a sanity check at high temperatures.
    """
    return float(2.0 * KB_EV_PER_K * temperature_K / k_eV_per_nm2)


def harmonic_r2_zeropoint(mass_m0: float, k_eV_per_nm2: float) -> float:
    """
    Zero-temperature (quantum) limit: <r²> = ħ/(mω) = 2λ/(ħω).
    This is the zero-point spread of the ground state.
    """
    lam        = HBAR2_OVER_2M0 / mass_m0
    hbar_omega = np.sqrt(2.0 * lam * k_eV_per_nm2)
    return float(2.0 * lam / hbar_omega)


def harmonic_r2_primitive_finite_P(
    mass_m0: float,
    k_eV_per_nm2: float,
    temperature_K: float,
    n_beads: int,
) -> float:
    """Exact ``<r²>`` of the discretized primitive-action 2D oscillator.

    This is the exact Gaussian result for the same finite-``P`` ring-polymer
    action sampled by :class:`RingPolymerAction`.  It separates Monte Carlo
    sampling error from the physical Trotter discretization error.
    """
    if mass_m0 <= 0.0:
        raise ValueError("mass_m0 must be positive")
    if k_eV_per_nm2 <= 0.0:
        raise ValueError("k_eV_per_nm2 must be positive")
    if temperature_K <= 0.0:
        raise ValueError("temperature_K must be positive")
    if n_beads < 1:
        raise ValueError("n_beads must be >= 1")

    lam = HBAR2_OVER_2M0 / float(mass_m0)
    beta = 1.0 / (KB_EV_PER_K * float(temperature_K))
    tau = beta / int(n_beads)
    modes = np.arange(int(n_beads), dtype=float)
    denominator = (
        np.sin(np.pi * modes / int(n_beads)) ** 2 / (lam * tau)
        + 0.5 * float(k_eV_per_nm2) * tau
    )
    return float(np.mean(1.0 / denominator))
