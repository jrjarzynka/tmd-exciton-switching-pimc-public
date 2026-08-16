"""Estimators for the field at which a dissociation curve crosses 50%.

Two estimators, deliberately kept together so that they cannot drift apart:
the analysis tools and the planning tool must extract F_z,50 the same way, or
a planned precision will not be the precision delivered.

INTERPOLATION uses only the two field points that bracket the crossing.
Every other point in the scan, each carrying a full binomial sample, is
discarded. It is simple, assumption-free about the curve's shape, and
statistically wasteful.

LOGISTIC MLE fits

    p(F) = 1 / (1 + exp(-(F - F50) / s))

to the binomial counts at every field by maximum likelihood. It uses the whole
curve, so it recovers the shape information interpolation throws away, and it
still returns a value when no single point happens to sit near 50%. Simulation
(see plan_symmetry_test.py) puts it at roughly 3.6x the seed efficiency of
interpolation for a realistic transition width.

The likelihood is binomial rather than least-squares on the fractions: a
fraction of 0 or 1 carries different information from a fraction of 0.5 at the
same seed count, and least squares is blind to that difference.

Neither estimator extrapolates. Interpolation returns None when the scan does
not bracket the crossing; the MLE returns None when the fitted F50 lands far
outside the scanned window, since a threshold outside the data is a guess
about the curve's tails rather than a measurement.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

__all__ = ["interp_threshold", "logistic_mle", "estimate_threshold"]

# How far beyond the scanned field window a fitted F50 may fall before the fit
# is treated as an extrapolation and refused, in units of the window width.
_EXTRAPOLATION_TOLERANCE = 0.25


def interp_threshold(fields: Sequence[float], fractions: Sequence[float],
                     level: float = 0.5) -> Optional[float]:
    """Linear interpolation between the two points bracketing `level`."""
    f = np.asarray(fields, dtype=float)
    p = np.asarray(fractions, dtype=float)
    order = np.argsort(f)
    f, p = f[order], p[order]
    if p.size < 2 or p.min() > level or p.max() < level:
        return None
    for i in range(len(f) - 1):
        if (p[i] - level) * (p[i + 1] - level) <= 0:
            if p[i + 1] == p[i]:
                return float(f[i])
            t = (level - p[i]) / (p[i + 1] - p[i])
            return float(f[i] + t * (f[i + 1] - f[i]))
    return None


def logistic_mle(fields: Sequence[float], counts: Sequence[float],
                 n_seeds: Sequence[float] | float) -> Optional[tuple]:
    """Maximum-likelihood logistic fit. Returns (F50, s), or None on failure.

    `counts` are dissociated-seed counts per field; `n_seeds` may be a scalar
    or one value per field, so that scans with unequal seed counts per point
    (a resumed or partially failed run) are handled without silently averaging.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        return None

    f = np.asarray(fields, dtype=float)
    k = np.asarray(counts, dtype=float)
    n = (np.full_like(f, float(n_seeds)) if np.isscalar(n_seeds)
         else np.asarray(n_seeds, dtype=float))

    if f.size < 3:
        # Three free-standing points is the minimum for a two-parameter fit to
        # mean anything; below that, fall back rather than pretend.
        return None

    def nll(theta):
        f50, log_s = theta
        s = math.exp(log_s)
        if not math.isfinite(s) or s <= 0.0:
            return 1e30
        z = (f - f50) / s
        log_p = -np.logaddexp(0.0, -z)
        log_q = -np.logaddexp(0.0, z)
        return -float(np.sum(k * log_p + (n - k) * log_q))

    span = float(f.max() - f.min())
    p_emp = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    f0 = interp_threshold(f, p_emp)
    if f0 is None:
        f0 = float(np.mean(f))
    s0 = max(span / 8.0, 1e-3)

    try:
        res = minimize(nll, x0=[f0, math.log(s0)], method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    except Exception:
        return None
    if not res.success:
        return None

    f50, log_s = res.x
    pad = _EXTRAPOLATION_TOLERANCE * span
    if not (f.min() - pad <= f50 <= f.max() + pad):
        return None
    return float(f50), float(math.exp(log_s))


def estimate_threshold(fields, counts, n_seeds, estimator: str = "mle"
                       ) -> Optional[float]:
    """Dispatch to one estimator by name; returns F50 only.

    'mle' does NOT silently fall back to interpolation on failure. A fit that
    fails is information -- usually that the scan does not span the transition
    -- and hiding it behind a different estimator would make a mixed-estimator
    bootstrap distribution that means nothing.
    """
    f = np.asarray(fields, dtype=float)
    k = np.asarray(counts, dtype=float)
    n = (np.full_like(f, float(n_seeds)) if np.isscalar(n_seeds)
         else np.asarray(n_seeds, dtype=float))

    if estimator == "interp":
        p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
        return interp_threshold(f, p)
    if estimator == "mle":
        r = logistic_mle(f, k, n)
        return None if r is None else r[0]
    raise ValueError(f"unknown estimator {estimator!r}; use 'mle' or 'interp'")
