"""Shared resolution of per-carrier landscape parameters.

Every two-body runner in this project needs the same three things: work out
whether the electron and hole have been given one landscape amplitude or
two, do the same for the effective dipole length, and be able to say which
mode a given run is in. Before this module that logic lived, slightly
differently, in each runner.

Background
----------
Until v1.1 the two-body model was built from a single `moire_amplitude_eV`
and a single `dipole_length_nm`, so V_e and V_h differed only by a registry
offset and by the sign of the Stark term. That is an idealisation. Stage 0
DFT for WSe2/MoSe2 indicates the electron and hole registry amplitudes are
different -- the physically relevant quantity being the modulation of the
gap across the moire cell rather than a common GSFE-derived V0 -- and that
the hole landscape in particular is only correct with spin-orbit coupling
included.

The distinction matters for interpretation, not just for accuracy. A model
in which V_e and V_h share an amplitude carries a residual electron-hole
symmetry, so any conclusion resting on the ABSENCE of such asymmetry -- for
instance a relocation threshold found to be independent of the landscape
amplitude -- may follow from the symmetry of the model rather than from the
physics. Such results must be re-checked with the amplitudes decoupled.

Config conventions
------------------
Each parameter may be supplied either once, under its shared key, or once
per carrier. Per-carrier keys win where both are present::

    "moire_amplitude_eV": 0.04
    "moire_amplitude_e_eV": 0.04, "moire_amplitude_h_eV": 0.02

    "dipole_length_nm": 0.05
    "dipole_length_e_nm": 0.05, "dipole_length_h_nm": 0.08

The moire PERIOD is deliberately not resolvable per carrier: both carriers
inhabit the same superlattice, and the periodic-cell rasterisation assumes
one set of lattice vectors.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "resolve_per_carrier",
    "resolve_amplitudes",
    "resolve_dipole_lengths",
    "describe_landscape_config",
    "format_landscape_banner",
    "landscape_provenance_row",
    "landscape_minima",
]

DEFAULT_DIPOLE_LENGTH_NM = 0.05


def resolve_per_carrier(
    config: Dict[str, Any],
    shared_key: str,
    key_e: str,
    key_h: str,
    default: Optional[float] = None,
) -> Tuple[float, float]:
    """Resolve a config value given either once or once per carrier.

    Raises ValueError rather than guessing when nothing is supplied, so a
    missing parameter fails loudly at config-load time instead of silently
    defaulting mid-run.
    """
    shared = config.get(shared_key, default)
    value_e = config.get(key_e, shared)
    value_h = config.get(key_h, shared)
    if value_e is None or value_h is None:
        raise ValueError(
            f"Config must supply either '{shared_key}' or both "
            f"'{key_e}' and '{key_h}'"
        )
    return float(value_e), float(value_h)


def resolve_amplitudes(config: Dict[str, Any]) -> Tuple[float, float]:
    """(amplitude_e_eV, amplitude_h_eV) from a runner config."""
    return resolve_per_carrier(
        config, "moire_amplitude_eV",
        "moire_amplitude_e_eV", "moire_amplitude_h_eV",
    )


def resolve_dipole_lengths(config: Dict[str, Any]) -> Tuple[float, float]:
    """(dipole_length_e_nm, dipole_length_h_nm) from a runner config."""
    return resolve_per_carrier(
        config, "dipole_length_nm",
        "dipole_length_e_nm", "dipole_length_h_nm",
        default=DEFAULT_DIPOLE_LENGTH_NM,
    )


def describe_landscape_config(config: Dict[str, Any], Fz: Optional[float] = None) -> Dict[str, Any]:
    """Summarise the landscape a config specifies.

    `Fz` may be passed explicitly for runners that sweep the field rather
    than reading it from the config.
    """
    amp_e, amp_h = resolve_amplitudes(config)
    dip_e, dip_h = resolve_dipole_lengths(config)
    if Fz is None:
        Fz = float(config.get("Fz_eV_per_nm", 0.0))
    return {
        "moire_period_nm": float(config["moire_period_nm"]),
        "moire_amplitude_e_eV": amp_e,
        "moire_amplitude_h_eV": amp_h,
        "amplitude_ratio_h_over_e": (amp_h / amp_e if amp_e != 0.0 else float("nan")),
        "dipole_length_e_nm": dip_e,
        "dipole_length_h_nm": dip_h,
        "Fz_eV_per_nm": float(Fz),
        "landscape_symmetric": (amp_e == amp_h) and (dip_e == dip_h),
    }


def format_landscape_banner(desc: Dict[str, Any]) -> str:
    """Human-readable banner for stderr, including the symmetry warning."""
    mode = "SYMMETRIC" if desc["landscape_symmetric"] else "ASYMMETRIC"
    lines = [
        f"Landscape mode: {mode}",
        f"  moire amplitude  e={desc['moire_amplitude_e_eV']:.6g} eV"
        f"   h={desc['moire_amplitude_h_eV']:.6g} eV"
        f"   (ratio h/e = {desc['amplitude_ratio_h_over_e']:.4g})",
        f"  dipole length    e={desc['dipole_length_e_nm']:.6g} nm"
        f"   h={desc['dipole_length_h_nm']:.6g} nm",
        f"  Fz               {desc['Fz_eV_per_nm']:.6g} eV/nm",
    ]
    if desc["landscape_symmetric"]:
        lines += [
            "  NOTE: electron and hole see landscapes of identical amplitude and",
            "  dipole length, differing only by the registry offset and the mass",
            "  ratio. Any result whose interpretation rests on the absence of",
            "  electron-hole asymmetry should be re-run with these decoupled.",
        ]
    return "\n".join(lines)


def landscape_provenance_row(desc: Dict[str, Any]) -> Dict[str, Any]:
    """Columns to attach to every output row, so a result can never be
    misattributed to the wrong model."""
    return {
        "moire_amplitude_e_eV": desc["moire_amplitude_e_eV"],
        "moire_amplitude_h_eV": desc["moire_amplitude_h_eV"],
        "amplitude_ratio_h_over_e": desc["amplitude_ratio_h_over_e"],
        "dipole_length_e_nm": desc["dipole_length_e_nm"],
        "dipole_length_h_nm": desc["dipole_length_h_nm"],
        "Fz_eV_per_nm": desc["Fz_eV_per_nm"],
        "landscape_symmetric": desc["landscape_symmetric"],
    }


def landscape_minima(
    period_nm: float,
    amplitude_e_eV: float,
    amplitude_h_eV: float,
    dipole_e_nm: float,
    dipole_h_nm: float,
    Fz: float,
    n_grid: int = 1201,
):
    """Locate the electron and hole landscape minima on a grid.

    Returned positions are used only to place starting configurations, so a
    grid resolution of a few pm is ample. Separations are computed directly
    and never minimum-imaged: the interaction is aperiodic, and reducing the
    separation into one cell here would silently place the "separated" start
    at the wrong distance.

    The moire term V0 sum cos(G.r) and the Stark texture sum sin(G.r) are
    built on the SAME three reciprocal vectors, so the field does not move
    the minima; it tilts their relative depths and thereby selects which
    sublattice each carrier prefers.

    Unlike the earlier single-amplitude version of this routine, the moire
    term is evaluated with each carrier's OWN amplitude and dipole length.
    With equal amplitudes and equal dipole lengths the result is identical
    to that version.
    """
    G = 4.0 * math.pi / (math.sqrt(3.0) * float(period_nm))
    Gs = [np.array([G, 0.0]),
          np.array([-0.5 * G, math.sqrt(3.0) / 2.0 * G]),
          np.array([-0.5 * G, -math.sqrt(3.0) / 2.0 * G])]

    g = np.linspace(-period_nm, period_nm, n_grid)
    X, Y = np.meshgrid(g, g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=1)

    cos_sum = sum(np.cos(P @ k) for k in Gs)
    sin_sum = sum(np.sin(P @ k) for k in Gs)

    v_e = amplitude_e_eV * cos_sum - Fz * dipole_e_nm * sin_sum
    v_h = amplitude_h_eV * cos_sum + Fz * dipole_h_nm * sin_sum

    r_e = P[np.argmin(v_e)]
    r_h = P[np.argmin(v_h)]
    return tuple(r_e), tuple(r_h)
