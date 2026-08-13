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
    "PairCorrelation",
    "pair_separations",
    "separation_diagnostics",
    "assert_interaction_is_aperiodic",
    "contact_density_from_samples",
    "contact_density_by_registry",
    "pair_correlation",
    "variance_decomposition",
    "detect_bimodality",
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


# =============================================================================
# PAIR CORRELATION FUNCTION
# =============================================================================


@dataclass(slots=True)
class PairCorrelation:
    """Radial density and pair correlation function of the relative coordinate."""

    rho_nm: np.ndarray            # bin representatives
    radial_density: np.ndarray    # P(rho), normalised so that int P drho = 1
    g_of_rho: np.ndarray          # |psi(rho)|^2 = P(rho) / (2 pi rho)
    counts: np.ndarray
    bin_edges_rho: np.ndarray
    n_samples: int


def pair_correlation(
    rho: np.ndarray,
    r_max: Optional[float] = None,
    n_bins: int = 60,
) -> PairCorrelation:
    """Radial density P(rho) and pair correlation g(rho) = |psi(rho)|^2.

    Both are returned because confusing them is easy and consequential. P(rho)
    is what a histogram of separations shows; it vanishes at the origin purely
    because the 2D measure does, 2 pi rho drho, not because the pair avoids
    contact. g(rho) divides that measure out and is the physical quantity: its
    value at zero is the contact density.

    Binning is uniform in u = rho^2, i.e. equal-AREA annuli. This is the same
    scheme `contact_density` uses. It keeps the statistics per bin roughly
    constant instead of starving the small-rho bins, and it puts finer
    resolution at large rho, which is where a second population sits if the
    pair separates into neighbouring moire minima.

    Do NOT read g(0) off the first bin. That bin reports an average over
    [0, sqrt(du)], which for a decaying g underestimates the value at the
    origin, and the bias grows with the bin width -- here set by the whole
    plotted range rather than by a window chosen for extrapolation. Use
    `contact_density`, which fits a narrow window near the origin for exactly
    this reason.
    """
    rho = np.asarray(rho, dtype=float).ravel()
    rho = rho[np.isfinite(rho)]
    if rho.size < 50:
        raise ValueError(f"Need at least 50 samples, got {rho.size}.")

    if r_max is None:
        r_max = float(rho.max()) * 1.02
    u_max = r_max ** 2

    counts, u_edges = np.histogram(rho ** 2, bins=n_bins, range=(0.0, u_max))
    rho_edges = np.sqrt(u_edges)
    du = u_edges[1] - u_edges[0]

    # density in u is exactly pi |psi|^2, with no Jacobian factor
    density_u = counts / (rho.size * du)
    g = density_u / np.pi

    u_mid = 0.5 * (u_edges[:-1] + u_edges[1:])
    rho_mid = np.sqrt(u_mid)

    # P(rho) is taken directly as the histogram density in rho rather than as
    # 2 pi rho_mid g. The two agree only where rho_mid coincides with the
    # midpoint in rho, which equal-area bins guarantee nowhere and violate
    # badly in the first bin (it spans [0, sqrt(du)], whose rho-midpoint is
    # 0.5 sqrt(du) against rho_mid = 0.707 sqrt(du)). Using the product form
    # overestimates the normalisation by ~14% for a bound pair, concentrated
    # exactly in the bins that carry most of the samples.
    d_rho = np.diff(rho_edges)
    radial = counts / (rho.size * d_rho)

    return PairCorrelation(
        rho_nm=rho_mid, radial_density=radial, g_of_rho=g,
        counts=counts, bin_edges_rho=rho_edges, n_samples=int(rho.size),
    )


def variance_decomposition(rho_per_seed: Sequence[np.ndarray]) -> dict:
    """Split the variance of rho^2 into within-seed and between-seed parts.

    This is the measurement that decides what a bimodal pair correlation means,
    and the two readings are opposite:

      within-seed   a single trajectory visits both the co-located and the
                    separated configuration. The pair genuinely switches, and a
                    two-state description is physical.

      between-seed  each trajectory stays in whichever configuration it started
                    in and the seeds merely disagree. The aggregate looks
                    bimodal, but that is incomplete ergodicity, and any
                    "transition" measured from it is a lottery over initial
                    conditions rather than a response to the field.

    A histogram of per-seed <rho^2>, which is what the manuscript currently
    shows, cannot separate these: it is a distribution of means, and a
    unimodal distribution of means is perfectly compatible with sharply
    bimodal sampling inside every seed.
    """
    if len(rho_per_seed) < 2:
        raise ValueError("Need at least two seeds for a decomposition.")

    r2 = [np.asarray(r, dtype=float).ravel() ** 2 for r in rho_per_seed]
    means = np.array([x.mean() for x in r2])
    within = float(np.mean([x.var() for x in r2]))
    between = float(means.var(ddof=1))
    total = within + between

    return {
        "within_seed_variance": within,
        "between_seed_variance": between,
        "between_seed_fraction": float(between / total) if total > 0 else float("nan"),
        "per_seed_mean_rho2": means,
        "n_seeds": len(r2),
    }


def detect_bimodality(pc: PairCorrelation, min_prominence_frac: float = 0.10) -> dict:
    """Locate peaks in the radial density and measure the dip between them.

    Operates on P(rho) rather than g(rho): g rises monotonically toward the
    origin for a bound pair, so a second population shows up as a shoulder
    there, whereas in P(rho) both populations appear as genuine peaks.

    `min_prominence_frac` is expressed relative to the tallest peak, so the
    criterion does not depend on normalisation or on the sample count.
    """
    from scipy.signal import find_peaks

    p = np.asarray(pc.radial_density, dtype=float)
    if p.max() <= 0:
        return {"n_modes": 0, "peak_rho_nm": [], "dip_depth": float("nan")}

    # find_peaks never reports an endpoint, but the bound-state peak of a
    # tightly bound pair sits in the very first bin. Padding with zeros makes
    # the boundary eligible without perturbing interior prominences.
    padded = np.concatenate(([0.0], p, [0.0]))
    peaks, props = find_peaks(padded, prominence=min_prominence_frac * p.max())
    peaks = peaks - 1
    peak_rho = [float(pc.rho_nm[i]) for i in peaks]

    dip = float("nan")
    if len(peaks) >= 2:
        i, j = peaks[0], peaks[-1]
        valley = p[i:j + 1].min()
        dip = float(1.0 - valley / min(p[i], p[j]))

    return {
        "n_modes": int(len(peaks)),
        "peak_rho_nm": peak_rho,
        "peak_prominence": [float(v) for v in props.get("prominences", [])],
        "dip_depth": dip,
    }
