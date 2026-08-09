"""
pair_observables.py

Observable layer over the two-body periodic sampler. Consumes the raw
`samples_e`, `samples_h` arrays of shape (n_configs, P, 2) and produces
relative-coordinate quantities. The JIT kernel is not touched.

Periodicity: why the relative coordinate is NOT minimum-imaged
--------------------------------------------------------------
The periodic sampler wraps ONLY the landscape lookup. Inside
`bilinear_interpolate_periodic_cell` the query point is converted to
fractional lattice coordinates and wrapped with u % 1, v % 1; the stored path
coordinates stay unwrapped Cartesian, and the electron-hole interaction is
evaluated from a 1D table in the raw separation.

That split is physically correct and must be preserved:

  * The moire landscape IS periodic. It is a property of the crystal, and a
    particle arbitrarily far from the origin sees the same potential as its
    image in the home cell. Wrapping the lookup is exact.

  * The electron-hole interaction is NOT periodic. There is one electron and
    one hole in an infinite plane. Applying the minimum-image convention to
    rho would introduce periodic images of the hole, replacing a single
    exciton in a moire potential with a LATTICE of excitons at one per moire
    cell -- a different physical system, and a dense one.

So the raw difference is the physical relative coordinate. Minimum-imaging it
would be a bug, not a missing correction. `assert_interaction_is_aperiodic`
turns that reasoning into an executable check against the actual kernel.

The joint global-translation move shifts both chains by the same vector and so
leaves rho invariant; only the interaction confines it. A run in which rho
approaches the interaction table's cutoff is therefore not merely inaccurate,
it is unbound -- see `separation_diagnostics`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .contact import ContactDensityResult, contact_density

__all__ = [
    "SeparationDiagnostics",
    "pair_separations",
    "separation_diagnostics",
    "assert_interaction_is_aperiodic",
    "contact_density_from_samples",
    "contact_density_by_registry",
]


@dataclass(slots=True)
class SeparationDiagnostics:
    n_configs: int
    n_beads: int
    mean_rho2: float
    max_rho: float
    r_int_max: Optional[float]
    frac_beyond_half_cutoff: float
    cutoff_ok: bool
    notes: list


def pair_separations(samples_e: np.ndarray, samples_h: np.ndarray) -> np.ndarray:
    """Slice-matched separations rho[c, j] = |r_e[c,j] - r_h[c,j]|.

    Matching is at equal imaginary-time index j, which is what the interaction
    couples and what the thermal density matrix is diagonal in. Centroid
    separation is a different random variable and must not be substituted: the
    same distinction that applies to mean_r2 elsewhere in this codebase.
    """
    e = np.asarray(samples_e, dtype=float)
    h = np.asarray(samples_h, dtype=float)
    if e.shape != h.shape:
        raise ValueError(f"Shape mismatch: samples_e {e.shape} vs samples_h {h.shape}")
    if e.ndim != 3 or e.shape[2] != 2:
        raise ValueError(f"Expected (n_configs, P, 2); got {e.shape}")
    d = e - h                      # raw, NOT minimum-imaged -- see module docstring
    return np.hypot(d[..., 0], d[..., 1])


def separation_diagnostics(
    rho: np.ndarray, r_int_max: Optional[float] = None
) -> SeparationDiagnostics:
    """Check the sampled separations against the interaction table's range.

    Beyond `r_int_max` a tabulated interaction has no data. Whatever the kernel
    does there -- clamp, extrapolate, or wrap the index -- the force is wrong,
    and since the interaction is the ONLY thing binding the pair, a run that
    reaches the cutoff is unbound rather than merely inaccurate. The failure is
    silent: the run completes, the acceptance rates look normal, and <rho^2>
    simply grows.

    This matters more after a d_p recalibration than before it. Dissociation
    fields scale as 1/d_p, so a smaller dipole length pushes the whole scan to
    stronger fields and larger separations at fixed physics.
    """
    rho = np.asarray(rho, dtype=float)
    notes: list[str] = []
    max_rho = float(rho.max())
    mean_rho2 = float(np.mean(rho ** 2))

    frac = float("nan")
    ok = True
    if r_int_max is not None:
        frac = float(np.mean(rho > 0.5 * r_int_max))
        if max_rho > r_int_max:
            ok = False
            notes.append(
                f"max separation {max_rho:.2f} nm EXCEEDS the interaction table "
                f"cutoff {r_int_max:.2f} nm: the pair left the tabulated region "
                f"and these samples are unphysical."
            )
        elif frac > 1e-3:
            notes.append(
                f"{frac * 100:.2f}% of beads are beyond half the cutoff "
                f"({0.5 * r_int_max:.1f} nm); the table's outer range is being "
                f"exercised. Widen interaction_table_r_max_nm."
            )

    if mean_rho2 <= 0:
        notes.append("Zero mean square separation: electron and hole coincide.")

    return SeparationDiagnostics(
        n_configs=int(rho.shape[0]),
        n_beads=int(rho.shape[1]) if rho.ndim > 1 else 1,
        mean_rho2=mean_rho2,
        max_rho=max_rho,
        r_int_max=r_int_max,
        frac_beyond_half_cutoff=frac,
        cutoff_ok=ok,
        notes=notes,
    )


def assert_interaction_is_aperiodic(
    energy_fn, r_e: np.ndarray, r_h: np.ndarray, lattice_vector_nm: Sequence[float],
    tol: float = 1e-9,
) -> None:
    """Assert that translating ONLY the hole by a lattice vector changes the energy.

    If the interaction were minimum-imaged in the periodic cell, moving the hole
    by exactly one lattice vector would map it onto its own image and leave the
    total energy unchanged. It must not: the landscape term is invariant under
    that translation, so any surviving change comes from the interaction, which
    is the aperiodic part.

    Passing `energy_fn` the sampler's own potential evaluation makes this a test
    of the kernel actually in use rather than of an assumption about it.
    """
    r_e = np.asarray(r_e, dtype=float)
    r_h = np.asarray(r_h, dtype=float)
    L = np.asarray(lattice_vector_nm, dtype=float)

    e0 = float(energy_fn(r_e, r_h))
    e1 = float(energy_fn(r_e, r_h + L))

    if abs(e1 - e0) <= tol:
        raise AssertionError(
            f"Translating the hole by a lattice vector left the energy unchanged "
            f"({e0:.6e} -> {e1:.6e}). The electron-hole interaction appears to be "
            f"minimum-imaged, which models a lattice of excitons rather than a "
            f"single pair in a moire potential."
        )


def contact_density_from_samples(
    samples_e: np.ndarray,
    samples_h: np.ndarray,
    r_int_max: Optional[float] = None,
    **kwargs,
) -> tuple[ContactDensityResult, SeparationDiagnostics]:
    """Contact density from raw sampler output, with the range check attached.

    The diagnostics are returned alongside rather than logged, so a caller
    cannot use the contact density without also having the evidence that the
    samples were bound.
    """
    rho = pair_separations(samples_e, samples_h)
    diag = separation_diagnostics(rho, r_int_max=r_int_max)
    return contact_density(rho.ravel(), **kwargs), diag


def contact_density_by_registry(
    samples_e: np.ndarray,
    samples_h: np.ndarray,
    lattice_a1_nm: Sequence[float],
    lattice_a2_nm: Sequence[float],
    n_bins_uv: int = 4,
    origin_nm: Sequence[float] = (0.0, 0.0),
    min_samples: int = 5000,
    **kwargs,
) -> dict:
    """Contact density resolved by position within the moire cell.

    Pooling every sample gives one number averaged over the whole moire cell,
    which discards the physics of interest: the recombination rate varies with
    local stacking, so an AB domain and a BA domain emit differently even at the
    same field. Binning by the wrapped centre-of-mass position in fractional
    lattice coordinates recovers that variation.

    The centre of mass is the right binning variable -- it is where the pair
    sits in the landscape -- while the contact density itself is built from the
    slice-matched separations of the samples in that bin. Bins holding fewer
    than `min_samples` beads are returned as None rather than as a noisy number.

    Returns a dict keyed by (i, j) fractional-cell index, plus 'u_edges',
    'v_edges' and 'counts'.
    """
    e = np.asarray(samples_e, dtype=float)
    h = np.asarray(samples_h, dtype=float)
    rho = pair_separations(e, h)

    com = 0.5 * (e + h)                                  # (n_cfg, P, 2)
    A = np.column_stack((np.asarray(lattice_a1_nm, float),
                         np.asarray(lattice_a2_nm, float)))
    Ainv = np.linalg.inv(A)
    rel = com - np.asarray(origin_nm, dtype=float)
    uv = rel @ Ainv.T
    uv -= np.floor(uv)                                   # wrap into the home cell

    iu = np.clip((uv[..., 0] * n_bins_uv).astype(int), 0, n_bins_uv - 1)
    iv = np.clip((uv[..., 1] * n_bins_uv).astype(int), 0, n_bins_uv - 1)

    out: dict = {
        "u_edges": np.linspace(0.0, 1.0, n_bins_uv + 1),
        "v_edges": np.linspace(0.0, 1.0, n_bins_uv + 1),
        "counts": np.zeros((n_bins_uv, n_bins_uv), dtype=int),
    }
    for i in range(n_bins_uv):
        for j in range(n_bins_uv):
            m = (iu == i) & (iv == j)
            n = int(m.sum())
            out["counts"][i, j] = n
            if n < min_samples:
                out[(i, j)] = None
                continue
            try:
                out[(i, j)] = contact_density(rho[m], **kwargs)
            except ValueError:
                out[(i, j)] = None
    return out
