"""Analytic gradients and primitive/virial total-energy estimators.

Additive module, kept separate from the already-validated potentials.py /
observables.py / action.py (same pattern as potential_helpers.py) rather
than editing them in place.

Scope
-----
Gradients are provided ONLY for the two potential classes used in the
analytic validation sections of this work (HarmonicPotential,
DoubleGaussianWellPotential) -- deliberately NOT for GridPotential2D,
whose bilinear interpolation has a discontinuous gradient at cell
boundaries (the same boundary that motivated the periodicity fix). Adding
a numerical/finite-difference fallback there would risk masking real bugs
rather than catching them, so unsupported potentials fail loudly instead
of silently falling back to finite differences.

Formula
-------
Both estimators are for the SAME finite-P, finite-T total energy of the
discretized ring-polymer model already sampled by RingPolymerAction /
PIMCSampler -- they are two independent estimators of one quantity, not
two different physical quantities. Their agreement (and the shrinking
variance ratio primitive/virial with increasing P) is the validation
criterion; see Sec. "Validation IV" of the manuscript.

Primitive (thermodynamic) estimator, D=2:
    E_prim = D*P/(2*beta)
             - <sum_k |r_{k+1}-r_k|^2> / (4*lambda*P*tau^2)
             + <mean_k V(r_k)>

Virial estimator (derived here from the generalized virial identity
<x_i dS/dx_i> = 1, summed over all P*D coordinates; NOT the
centroid-subtracted form found in some references -- that form requires
separately handling the translational zero mode and was not needed to
get an estimator that already agrees with the primitive one to within
statistical error):
    E_vir = <mean_k V(r_k)> + <sum_k r_k . grad_V(r_k)> / (2*P)

Both formulas were re-derived and cross-checked numerically (against each
other, against the exact P->infinity continuum harmonic-oscillator energy,
and against direct-sampling of the exact finite-P harmonic ring-polymer
distribution) before being added here -- see the verification script in
the accompanying development notes if you want to reproduce that check.
"""

from __future__ import annotations
import numpy as np

from .constants import HBAR2_OVER_2M0, KB_EV_PER_K
from .potentials import HarmonicPotential, DoubleGaussianWellPotential


def potential_gradient(potential, r: np.ndarray) -> np.ndarray:
    """Analytic gradient dV/dr, evaluated at each row of r: shape (N, 2).

    Dispatches on potential type. Raises NotImplementedError for any
    potential without an explicit analytic implementation here -- see
    module docstring for why this does not silently fall back to finite
    differences.
    """
    r = np.asarray(r, dtype=float)

    if isinstance(potential, HarmonicPotential):
        return potential.k_eV_per_nm2 * r

    if isinstance(potential, DoubleGaussianWellPotential):
        left = np.array([-potential.separation_nm / 2.0, 0.0])
        right = np.array([potential.separation_nm / 2.0, 0.0])
        inv2s2 = 1.0 / (2.0 * potential.sigma_nm ** 2)
        right_depth = potential.V0_eV + potential.asymmetry_eV

        dl = r - left
        dr = r - right
        f_left = np.exp(-np.einsum("ij,ij->i", dl, dl) * inv2s2)
        f_right = np.exp(-np.einsum("ij,ij->i", dr, dr) * inv2s2)

        c_left = potential.V0_eV / potential.sigma_nm ** 2
        c_right = right_depth / potential.sigma_nm ** 2
        return c_left * dl * f_left[:, None] + c_right * dr * f_right[:, None]

    raise NotImplementedError(
        f"potential_gradient has no analytic implementation for "
        f"{type(potential).__name__}. Add one explicitly (do not fall "
        f"back to finite differences for grid-interpolated potentials -- "
        f"see module docstring)."
    )


def primitive_energy_series(samples: np.ndarray, potential,
                             mass_m0: float, temperature_K: float) -> np.ndarray:
    """Per-sample primitive (thermodynamic) total-energy estimator.

    samples: shape (N, P, 2), one closed ring-polymer path per sample
    (same convention as observables.r2_time_series). Returns shape (N,);
    feed this directly into statistics.analyze_timeseries for
    autocorrelation-corrected error bars, exactly as done for the r^2
    diagnostics elsewhere in this codebase.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3 or samples.shape[2] != 2:
        raise ValueError("samples must have shape (N_samples, P, 2)")
    N, P, D = samples.shape

    beta = 1.0 / (KB_EV_PER_K * temperature_K)
    tau = beta / P
    lam = HBAR2_OVER_2M0 / mass_m0

    diffs = samples - np.roll(samples, -1, axis=1)
    spring_sq = np.einsum("npd,npd->n", diffs, diffs)
    kinetic = D * P / (2.0 * beta) - spring_sq / (4.0 * lam * P * tau ** 2)

    V = potential.value(samples.reshape(-1, 2)).reshape(N, P)
    return kinetic + V.mean(axis=1)


def virial_energy_series(samples: np.ndarray, potential) -> np.ndarray:
    """Per-sample virial total-energy estimator (see module docstring for
    the derivation and why it has no centroid subtraction or leftover
    D/(2*beta) term). Returns shape (N,), same usage pattern as
    primitive_energy_series above.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3 or samples.shape[2] != 2:
        raise ValueError("samples must have shape (N_samples, P, 2)")
    N, P, D = samples.shape

    flat = samples.reshape(-1, 2)
    V = potential.value(flat).reshape(N, P)
    grad = potential_gradient(potential, flat).reshape(N, P, 2)
    r_dot_grad = np.einsum("npd,npd->np", samples, grad)

    return V.mean(axis=1) + r_dot_grad.sum(axis=1) / (2.0 * P)
