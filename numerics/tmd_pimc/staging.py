"""Exact free-particle Brownian-bridge primitives for ring-polymer paths.

This module is intentionally *particle agnostic*.  It knows only a path,
segment geometry and the free-link variance.  COM, electron, hole and any
future distinguishable-particle sampler should reuse these primitives.

Coordinates remain on the infinite Cartesian cover.  Periodicity, external
one-body potentials and inter-particle interactions belong to the sampler's
potential/action layer, not to the bridge proposal itself.
"""
from __future__ import annotations

import math
import numpy as np


def free_link_variance_nm2(lambda_nm2_eV: float, tau_eV_inv: float) -> float:
    """Return one-link Cartesian variance ``2 * lambda * tau``.

    ``lambda = hbar^2/(2m)`` is particle specific, so the same bridge code can
    be reused for COM, electron and hole polymers with different masses.
    """
    lam = float(lambda_nm2_eV)
    tau = float(tau_eV_inv)
    variance = 2.0 * lam * tau
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("2*lambda*tau must be positive and finite")
    return variance


def segment_indices(start: int, length: int, n_beads: int) -> np.ndarray:
    """Return both endpoints and all interior indices of a periodic segment."""
    if not 2 <= int(length) < int(n_beads):
        raise ValueError("segment length must satisfy 2 <= length < n_beads")
    return (int(start) + np.arange(int(length) + 1, dtype=np.int64)) % int(n_beads)


def free_bridge_proposal(
    path: np.ndarray,
    start: int,
    length: int,
    gaussian_variates: np.ndarray,
    free_link_variance_nm2: float,
) -> np.ndarray:
    """Redraw a segment interior from its exact free conditional density.

    Parameters
    ----------
    path
        Array of shape ``(P, D)``.  Paper-1 uses ``D=2``; the mathematics is
        independent of dimension.
    gaussian_variates
        Standard-normal array of shape ``(length-1, D)``.  Supplying literal
        variates rather than an RNG enables common-random Python/JIT tests.
    free_link_variance_nm2
        One-link variance ``2*lambda_particle*tau`` for the polymer being
        updated.
    """
    old = np.asarray(path, dtype=np.float64)
    z = np.asarray(gaussian_variates, dtype=np.float64)
    if old.ndim != 2 or old.shape[1] < 1:
        raise ValueError("path must have shape (P, D) with D >= 1")
    if z.shape != (int(length) - 1, old.shape[1]):
        raise ValueError("gaussian_variates must have shape (length-1, D)")
    if not np.isfinite(free_link_variance_nm2) or free_link_variance_nm2 <= 0:
        raise ValueError("free_link_variance_nm2 must be positive and finite")
    indices = segment_indices(start, length, old.shape[0])
    proposed = old.copy()
    previous = old[indices[0]].copy()
    endpoint = old[indices[-1]].copy()
    for offset in range(1, int(length)):
        remaining = int(length) - offset
        denominator = remaining + 1.0
        mean = (remaining * previous + endpoint) / denominator
        variance = free_link_variance_nm2 * remaining / denominator
        previous = mean + math.sqrt(variance) * z[offset - 1]
        proposed[indices[offset]] = previous
    return proposed


def bridge_conditional_moments(
    r0: np.ndarray, rL: np.ndarray, length: int, free_link_variance_nm2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Independent Brownian-bridge mean and one-coordinate covariance.

    ``r0`` and ``rL`` may have any common Cartesian dimension ``D``.  The
    returned covariance is for one Cartesian coordinate because coordinates
    are independent and identically distributed in the free action.
    """
    r0 = np.asarray(r0, dtype=np.float64)
    rL = np.asarray(rL, dtype=np.float64)
    if r0.ndim != 1 or rL.shape != r0.shape:
        raise ValueError("r0 and rL must be equal-shape 1D coordinate vectors")
    interior = np.arange(1, int(length), dtype=np.float64)
    mean = r0 + (interior[:, None] / float(length)) * (rL - r0)
    covariance = free_link_variance_nm2 * (
        np.minimum.outer(interior, interior)
        - np.outer(interior, interior) / float(length)
    )
    return mean, covariance


def bridge_log_density(
    path: np.ndarray, start: int, length: int, free_link_variance_nm2: float
) -> float:
    """Normalized ``D``-dimensional bridge log density for validation."""
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 1:
        raise ValueError("path must have shape (P, D) with D >= 1")
    indices = segment_indices(start, length, len(p))
    links = p[indices[1:]] - p[indices[:-1]]
    endpoint = p[indices[-1]] - p[indices[0]]
    dim = p.shape[1]
    # Product of D independent conditioned Gaussian bridges.
    return float(
        -0.5 * dim * (length - 1) * math.log(2.0 * math.pi * free_link_variance_nm2)
        + 0.5 * dim * math.log(float(length))
        - np.sum(links * links) / (2.0 * free_link_variance_nm2)
        + np.dot(endpoint, endpoint) / (2.0 * length * free_link_variance_nm2)
    )
