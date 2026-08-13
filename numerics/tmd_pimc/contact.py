"""
contact_density.py

Estimator for the electron-hole contact density |psi(0)|^2 of a two-dimensional
pair, from PI-QMC samples of the relative coordinate.

Why this quantity
-----------------
The radiative recombination rate of an exciton factorises as

    Gamma_rad  ~  |mu_cv|^2 * |psi(0)|^2

where mu_cv is an interband dipole matrix element (a band-structure quantity,
outside an effective-mass model) and |psi(0)|^2 is the probability of finding
the electron and hole at the same in-plane position. Only the second factor
depends on the moire landscape and the applied field, so RATIOS of recombination
rates -- between registries, fields, or temperatures -- are accessible here even
though absolute lifetimes are not.

The estimator
-------------
Naively one histograms rho = |r_e - r_h| and divides by the 2D Jacobian 2*pi*rho.
That division blows up exactly where the signal is wanted, because the number of
samples in a bin at small rho vanishes linearly with rho.

Changing variable to u = rho^2 removes the Jacobian entirely:

    P(rho) d(rho) = 2*pi*rho |psi(rho)|^2 d(rho) = pi |psi(sqrt(u))|^2 du

so a histogram that is UNIFORM IN u has a flat weight, and its density is
pi|psi|^2 directly -- no division, no divergence, and equal statistics per bin
all the way to the origin.

The remaining step is extrapolating to u = 0. The extrapolation is polynomial in
rho, NOT in u: a wavefunction with a cusp at the origin behaves as
|psi(rho)|^2 ~ |psi(0)|^2 (1 - 2*rho/a + ...), which is linear in rho and has
infinite slope in u. Fitting in u would bias any cusped state -- including the
Coulomb-like limit of the screened interaction. Fitting in rho handles both the
cusped and the smooth (Gaussian) case, since a Gaussian simply returns a zero
linear coefficient.

Slice matching
--------------
Samples must be the slice-matched relative coordinates rho_j = |r_e,j - r_h,j|
at equal imaginary-time index j, pooled over j and over configurations. This is
the diagonal of the thermal density matrix. Centroid separations are a different
random variable and must not be substituted -- the same distinction that applies
to mean_r2 elsewhere in this codebase.

What this is and is not
-----------------------
The result is the THERMAL contact density, Sum_n |psi_n(0)|^2 exp(-beta E_n) / Z,
not the ground-state value. At temperatures comparable to the level spacing
excited states contribute. For a ground-state number, extrapolate in beta.

"Contact" here means zero IN-PLANE separation, not zero separation. For an
interlayer exciton the electron and hole sit in different layers, and the
bilayer Keldysh interaction enters through the three-dimensional distance
sqrt(rho^2 + D^2) with D the interlayer separation (0.6 nm for MoSe2/WSe2).
Two consequences follow.

First, the interaction is never singular: at rho = 0 it is finite, set by D.
The pair wavefunction therefore has NO cusp and is smooth at the origin, the
milder of the two cases the estimator is calibrated against.

Second, and more importantly for interpretation: the interlayer overlap that
suppresses interlayer-exciton recombination by orders of magnitude is NOT in
|psi(0)|^2. It sits in the prefactor, together with mu_cv, and neither is
available in an effective-mass model. Ratios between registries, fields and
temperatures are meaningful because that prefactor cancels; absolute rates are
not, and no amount of sampling will make them so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "ContactDensityResult",
    "contact_density",
    "contact_density_gaussian_exact",
    "contact_density_exponential_exact",
    "relative_radiative_rate",
]


@dataclass(slots=True)
class ContactDensityResult:
    """Contact density and the diagnostics needed to judge whether to trust it."""

    psi0_sq: float           # |psi(0)|^2, units of 1/length^2
    stderr: float            # bootstrap standard error
    n_samples: int
    fit_order: int
    rho_window: float        # largest rho included in the fit
    n_bins_used: int
    slope_rho: float         # linear coefficient; large |slope| means a cusp
    curvature_flag: bool     # True if the quadratic term dominates the window
    mean_rho2: float         # <rho^2> over the same samples, for cross-checks


def _histogram_in_u(rho: np.ndarray, u_max: float, n_bins: int):
    """Uniform histogram in u = rho^2; returns (rho_eff, density_in_u, counts).

    Uniform bins in u are equal-AREA annuli in the plane, so each bin carries
    comparable statistics regardless of its radius. The bin representative is
    taken at the area-weighted midpoint, sqrt((u_lo + u_hi)/2), which is exact
    for a density linear in u and a good approximation otherwise.
    """
    u = rho * rho
    counts, edges = np.histogram(u, bins=n_bins, range=(0.0, u_max))
    du = edges[1] - edges[0]
    density_u = counts / (len(rho) * du)          # normalised density in u
    u_mid = 0.5 * (edges[:-1] + edges[1:])
    return np.sqrt(u_mid), density_u, counts


def contact_density(
    rho: np.ndarray,
    rho_window: Optional[float] = None,
    n_bins: int = 16,
    fit_order: int = 2,
    n_boot: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> ContactDensityResult:
    """Estimate |psi(0)|^2 from slice-matched relative-coordinate samples.

    Parameters
    ----------
    rho
        Flat array of |r_e - r_h| at matched imaginary-time slices.
    rho_window
        Fit only samples with rho below this. Defaults to 0.35*sqrt(<rho^2>).
    n_bins
        Number of equal-area bins inside the window.
    fit_order
        Polynomial order in rho. Order 2 is required: order 1 misses the
        curvature and biases the smooth (Gaussian) case by 3-9%.
    n_boot
        Bootstrap resamples for the error bar.

    Accuracy
    --------
    Defaults were calibrated against both exact references over a grid of
    window widths, bin counts and fit orders. The window sets a bias/variance
    trade-off with opposite signs for the two references -- widening it biases
    a smooth state high and a cusped state low -- and 0.35*sqrt(<rho^2>) with
    16 bins and a quadratic fit keeps BOTH within 0.6%. Wider windows reach 15%
    error on a cusped state at first order.

    That residual ~0.5% is a method bias, not noise, and it does not shrink with
    more samples. It largely cancels in ratios between conditions with similar
    pair-distribution shapes, which is the intended use.

    The returned `curvature_flag` is set when the quadratic term contributes
    more than half the extrapolated value across the window, meaning the fit is
    doing structural work rather than a short extrapolation. Shrink
    `rho_window` when that happens.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    rho = rho[np.isfinite(rho)]
    if rho.size < 100:
        raise ValueError(f"Need at least 100 samples, got {rho.size}.")
    if np.any(rho < 0):
        raise ValueError("Negative separations: rho must be a magnitude.")

    mean_rho2 = float(np.mean(rho * rho))
    if rho_window is None:
        rho_window = 0.35 * np.sqrt(mean_rho2)
    u_max = rho_window ** 2

    if np.count_nonzero(rho <= rho_window) < 50:
        raise ValueError(
            f"Only {np.count_nonzero(rho <= rho_window)} samples inside "
            f"rho_window={rho_window:.4g}; widen the window or sample longer."
        )

    def _estimate(sample: np.ndarray) -> tuple[float, float, float]:
        r_eff, dens_u, counts = _histogram_in_u(sample, u_max, n_bins)
        # density in u equals pi*|psi|^2, so |psi|^2 = density_u / pi
        psi2 = dens_u / np.pi
        m = counts > 0
        if m.sum() <= fit_order:
            raise ValueError("Too few populated bins for the requested fit order.")
        # Poisson weights: var(counts) = counts, so weight residuals by sqrt(counts)
        coeffs = np.polyfit(r_eff[m], psi2[m], fit_order, w=np.sqrt(counts[m]))
        value = coeffs[-1]                       # evaluated at rho = 0
        slope = coeffs[-2] if fit_order >= 1 else 0.0
        quad = coeffs[-3] if fit_order >= 2 else 0.0
        return float(value), float(slope), float(quad)

    value, slope, quad = _estimate(rho)

    rng = rng or np.random.default_rng(0)
    boots = np.empty(n_boot)
    n = rho.size
    for b in range(n_boot):
        boots[b] = _estimate(rho[rng.integers(0, n, n)])[0]
    stderr = float(np.std(boots, ddof=1))

    curvature = abs(quad) * rho_window ** 2
    flag = bool(value != 0.0 and curvature > 0.5 * abs(value))

    return ContactDensityResult(
        psi0_sq=value,
        stderr=stderr,
        n_samples=int(n),
        fit_order=fit_order,
        rho_window=float(rho_window),
        n_bins_used=n_bins,
        slope_rho=slope,
        curvature_flag=flag,
        mean_rho2=mean_rho2,
    )


def contact_density_gaussian_exact(mean_rho2: float) -> float:
    """Exact contact density of a 2D Gaussian pair distribution.

    Any harmonic relative-coordinate problem -- at any temperature and, with the
    primitive action, at any finite bead count P -- has a Gaussian thermal
    density matrix, for which

        |psi(0)|^2 = 1 / (pi * <rho^2>).

    This makes the identity itself a self-contained test: run a harmonic pair,
    estimate the contact density, and compare against <rho^2> from the same
    samples. No external reference value is required, and the test is valid at
    finite P where the continuum answer is not.
    """
    if mean_rho2 <= 0:
        raise ValueError("mean_rho2 must be positive.")
    return 1.0 / (np.pi * mean_rho2)


def contact_density_exponential_exact(a: float) -> float:
    """Exact contact density for psi ~ exp(-rho/a), the cusped 2D reference.

    Normalising |psi|^2 = A exp(-2 rho / a) over the plane gives A = 2/(pi a^2).
    This is the case that breaks an extrapolation performed in u = rho^2, since
    |psi(rho)|^2 has finite slope in rho but infinite slope in u.
    """
    if a <= 0:
        raise ValueError("a must be positive.")
    return 2.0 / (np.pi * a * a)


def relative_radiative_rate(
    result: ContactDensityResult, reference: ContactDensityResult
) -> tuple[float, float]:
    """Ratio of radiative rates between two conditions, with propagated error.

    The interband dipole matrix element cancels in the ratio provided both
    conditions describe the same pair of bands -- the same valley and the same
    spin state. Comparing across valleys or spin-split branches does NOT cancel
    it, and this function must not be used for that.
    """
    if reference.psi0_sq == 0:
        raise ValueError("Reference contact density is zero.")
    ratio = result.psi0_sq / reference.psi0_sq
    rel = np.hypot(
        result.stderr / result.psi0_sq if result.psi0_sq else np.inf,
        reference.stderr / reference.psi0_sq,
    )
    return float(ratio), float(abs(ratio) * rel)
