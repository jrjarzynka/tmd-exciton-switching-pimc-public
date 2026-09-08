"""Particle-agnostic normal-mode development primitives for real ring polymers.

The module deliberately separates three related but *not identical* notions:

1. ``np.fft.rfft(..., norm='ortho')`` coefficients, convenient for storage and
   fast transforms;
2. Parseval-weighted one-sided mode powers, where interior rFFT bins carry
   weight two because they represent the conjugate ``+k/-k`` pair;
3. real orthonormal cosine/sine coordinates, the preferred convention for
   proposal amplitudes and analytic mode variances.

No production Monte Carlo policy is defined here.  A sampler must still
compute the appropriate full target-action change (including e-h interaction
when present) and apply Metropolis acceptance.
"""
from __future__ import annotations

import numpy as np


def _validate_path(path: np.ndarray) -> np.ndarray:
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2 or p.shape[1] < 1:
        raise ValueError("path must have shape (P, D)")
    return p


def rfft_modes(path: np.ndarray) -> np.ndarray:
    """Return one-sided orthogonally-scaled FFT coefficients.

    Input shape is ``(P,D)`` and output shape is ``(P//2+1,D)``.  The
    ``norm='ortho'`` convention is the orthonormal normalization of the *full*
    complex DFT.  For a real path, the stored one-sided rFFT array is not by
    itself an ordinary Euclidean-orthonormal coordinate vector: interior bins
    represent conjugate ``+k/-k`` pairs and therefore carry Parseval weight 2.

    Use :func:`rfft_parseval_weights`, :func:`rfft_mode_powers`, or
    :func:`real_normal_mode_coordinates` when interpreting norms, spring
    action, analytic variances, or proposal amplitudes.
    """
    p = _validate_path(path)
    return np.fft.rfft(p, axis=0, norm="ortho")


def path_from_rfft_modes(modes: np.ndarray, n_beads: int) -> np.ndarray:
    """Invert :func:`rfft_modes` to a real ``(P,D)`` path."""
    q = np.asarray(modes, dtype=np.complex128)
    p = int(n_beads)
    expected = p // 2 + 1
    if q.ndim != 2 or q.shape[0] != expected or q.shape[1] < 1:
        raise ValueError("modes must have shape (P//2+1, D)")
    return np.fft.irfft(q, n=p, axis=0, norm="ortho")


def rfft_parseval_weights(n_beads: int) -> np.ndarray:
    """Return one-sided rFFT multiplicities needed for Parseval sums.

    ``w[0]=1``.  Interior positive-frequency bins have ``w=2`` because the
    omitted negative-frequency coefficient has equal magnitude.  For even
    ``P``, the Nyquist bin ``k=P/2`` is self-conjugate and has ``w=1``.
    For odd ``P`` there is no Nyquist self-conjugate bin, so every stored
    ``k>0`` bin has weight 2.
    """
    p = int(n_beads)
    if p < 2:
        raise ValueError("n_beads must be >= 2")
    w = np.full(p // 2 + 1, 2.0, dtype=np.float64)
    w[0] = 1.0
    if p % 2 == 0:
        w[-1] = 1.0
    return w


def rfft_mode_powers(path: np.ndarray, *, parseval_weighted: bool = True) -> np.ndarray:
    """Return summed-Cartesian one-sided mode powers.

    With ``parseval_weighted=True`` (recommended), the returned contributions
    satisfy exactly, up to float64 roundoff,

    ``sum_j |r_j|^2 = sum_k power[k]``.

    With ``False`` the function returns raw stored-half-spectrum ``|q_k|^2``;
    those raw powers must *not* be interpreted as orthonormal real-mode powers
    for interior bins without the factor of two.
    """
    p = _validate_path(path)
    q = rfft_modes(p)
    raw = np.sum(np.abs(q) ** 2, axis=1)
    if parseval_weighted:
        raw = raw * rfft_parseval_weights(p.shape[0])
    return raw


def spring_mode_eigenvalue(mode: int, n_beads: int) -> float:
    """Nearest-neighbour ring-polymer eigenvalue ``4 sin^2(pi*k/P)``."""
    k = int(mode)
    p = int(n_beads)
    if not 0 <= k <= p // 2:
        raise ValueError("mode must satisfy 0 <= mode <= P//2")
    return float(4.0 * np.sin(np.pi * k / p) ** 2)


def spring_bond_sum_from_rfft(path: np.ndarray) -> float:
    """Reconstruct ``sum_j |r_(j+1)-r_j|^2`` from one-sided rFFT modes."""
    p = _validate_path(path)
    powers = rfft_mode_powers(p, parseval_weighted=True)
    eig = np.array(
        [spring_mode_eigenvalue(k, p.shape[0]) for k in range(p.shape[0] // 2 + 1)],
        dtype=np.float64,
    )
    return float(np.dot(eig, powers))


def real_normal_mode_coordinates(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return real orthonormal cosine/sine coordinates.

    Returns ``(cosine, sine)``, each shape ``(P//2+1,D)``.  For interior modes
    ``k`` the mapping is

    ``c_k = sqrt(2) Re(q_k)``, ``s_k = -sqrt(2) Im(q_k)``.

    The sign of ``s_k`` follows NumPy's ``exp(-i 2pi jk/P)`` FFT convention.
    The centroid and (for even P) Nyquist bins are self-conjugate, so their sine
    coordinates are exactly zero and cosine coordinates equal the real rFFT
    coefficient.  These coordinates obey ordinary Euclidean Parseval:

    ``sum_j |r_j|^2 = sum_k (|c_k|^2 + |s_k|^2)``.

    This is the preferred representation for symmetric proposal amplitudes and
    analytic real-mode variances.
    """
    p = _validate_path(path)
    q = rfft_modes(p)
    c = np.sqrt(2.0) * q.real
    s = -np.sqrt(2.0) * q.imag
    c[0] = q[0].real
    s[0] = 0.0
    if p.shape[0] % 2 == 0:
        c[-1] = q[-1].real
        s[-1] = 0.0
    return c, s


def path_from_real_normal_modes(
    cosine: np.ndarray, sine: np.ndarray, n_beads: int
) -> np.ndarray:
    """Invert :func:`real_normal_mode_coordinates` to a real path."""
    c = np.asarray(cosine, dtype=np.float64)
    s = np.asarray(sine, dtype=np.float64)
    p = int(n_beads)
    expected = p // 2 + 1
    if c.ndim != 2 or c.shape != s.shape or c.shape[0] != expected or c.shape[1] < 1:
        raise ValueError("cosine and sine must have shape (P//2+1, D)")
    if np.any(s[0] != 0.0):
        raise ValueError("centroid sine coordinate must be zero")
    if p % 2 == 0 and np.any(s[-1] != 0.0):
        raise ValueError("Nyquist sine coordinate must be zero for even P")

    q = (c - 1j * s) / np.sqrt(2.0)
    q[0] = c[0].astype(np.complex128)
    if p % 2 == 0:
        q[-1] = c[-1].astype(np.complex128)
    return path_from_rfft_modes(q, p)


def propose_rfft_mode_shift(
    path: np.ndarray,
    mode: int,
    delta_real: np.ndarray,
    delta_imag: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a symmetric literal shift to one stored rFFT coefficient.

    ``delta_real`` and ``delta_imag`` are in *raw rFFT coefficient units*.
    For an interior mode, the corresponding orthonormal real cosine/sine
    shifts are ``sqrt(2)*delta_real`` and ``-sqrt(2)*delta_imag``.  Prefer
    :func:`propose_real_normal_mode_shift` when proposal amplitudes should have
    a direct orthonormal real-coordinate meaning.

    For ``k=0`` and the even-P Nyquist mode, only a real shift is allowed.  If
    coefficient shifts are drawn from zero-mean symmetric distributions, the
    proposal is symmetric and Metropolis acceptance uses the full target-action
    change.
    """
    p = _validate_path(path)
    q = rfft_modes(p)
    k = int(mode)
    if not 0 <= k < q.shape[0]:
        raise ValueError("invalid rFFT mode index")
    dr = np.asarray(delta_real, dtype=np.float64)
    if dr.shape != (p.shape[1],):
        raise ValueError("delta_real must have shape (D,)")
    self_conjugate = (k == 0) or (p.shape[0] % 2 == 0 and k == p.shape[0] // 2)
    if delta_imag is None:
        di = np.zeros_like(dr)
    else:
        di = np.asarray(delta_imag, dtype=np.float64)
        if di.shape != dr.shape:
            raise ValueError("delta_imag must have shape (D,)")
    if self_conjugate and np.any(di != 0.0):
        raise ValueError("centroid/Nyquist rFFT modes are purely real")
    q[k] += dr + 1j * di
    return path_from_rfft_modes(q, p.shape[0])


def propose_real_normal_mode_shift(
    path: np.ndarray,
    mode: int,
    delta_cosine: np.ndarray,
    delta_sine: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a literal symmetric shift in the real orthonormal cos/sin basis.

    This is the recommended development primitive for physically interpretable
    mode amplitudes.  For centroid and even-P Nyquist modes the sine shift must
    be zero.  The routine is RNG-free for deterministic replay tests.
    """
    p = _validate_path(path)
    c, s = real_normal_mode_coordinates(p)
    k = int(mode)
    if not 0 <= k < c.shape[0]:
        raise ValueError("invalid normal-mode index")
    dc = np.asarray(delta_cosine, dtype=np.float64)
    if dc.shape != (p.shape[1],):
        raise ValueError("delta_cosine must have shape (D,)")
    if delta_sine is None:
        ds = np.zeros_like(dc)
    else:
        ds = np.asarray(delta_sine, dtype=np.float64)
        if ds.shape != dc.shape:
            raise ValueError("delta_sine must have shape (D,)")
    self_conjugate = (k == 0) or (p.shape[0] % 2 == 0 and k == p.shape[0] // 2)
    if self_conjugate and np.any(ds != 0.0):
        raise ValueError("centroid/Nyquist modes have no sine coordinate")
    c[k] += dc
    s[k] += ds
    return path_from_real_normal_modes(c, s, p.shape[0])


def centroid_free_mode_powers(path: np.ndarray) -> np.ndarray:
    """Return Parseval-weighted one-sided mode powers with centroid zeroed.

    The returned values are contributions to ``sum_j |r_j|^2``.  Interior
    rFFT bins therefore include the required factor of two.  This convention
    is suitable for reporting mode powers and comparing mode-by-mode ESS/IAT.
    """
    power = rfft_mode_powers(path, parseval_weighted=True)
    power[0] = 0.0
    return power
