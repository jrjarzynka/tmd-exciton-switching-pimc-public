"""Radial Rytova--Keldysh interaction tables for 2D excitons.

Convention used throughout this module
--------------------------------------

    V_RK(r) = -pi C / (2 kappa r0) * [H_0(r/r0) - Y_0(r/r0)]

where ``C = e^2/(4 pi epsilon_0)`` in eV nm, ``kappa`` is the effective
external dielectric constant, ``r0`` is the screening length in nm, ``H_0``
is the Struve function and ``Y_0`` is the Bessel function of the second kind.
With this convention, ``V_RK(r) -> -C/(kappa r)`` at large distance.

The table class uses only NumPy during Monte Carlo.  SciPy special functions
are needed only while constructing or independently checking the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import struve, y0

COULOMB_CONSTANT_EV_NM = 1.43996448
EULER_GAMMA = 0.5772156649015329


def _validate_parameters(
    kappa: float,
    screening_length_nm: float,
    coulomb_constant_eV_nm: float,
) -> None:
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be positive and finite")
    if not np.isfinite(screening_length_nm) or screening_length_nm <= 0.0:
        raise ValueError("screening_length_nm must be positive and finite")
    if not np.isfinite(coulomb_constant_eV_nm) or coulomb_constant_eV_nm <= 0.0:
        raise ValueError("coulomb_constant_eV_nm must be positive and finite")


def rk_short_distance_coefficient_eV(
    *,
    kappa: float,
    screening_length_nm: float,
    coulomb_constant_eV_nm: float = COULOMB_CONSTANT_EV_NM,
) -> float:
    """Return A=C/(kappa*r0) in V(r) ~ A[ln(r/2r0)+gamma]."""
    _validate_parameters(kappa, screening_length_nm, coulomb_constant_eV_nm)
    return float(coulomb_constant_eV_nm / (kappa * screening_length_nm))


def rk_energy_direct_eV(
    r_nm: np.ndarray | float,
    *,
    kappa: float,
    screening_length_nm: float,
    coulomb_constant_eV_nm: float = COULOMB_CONSTANT_EV_NM,
    evaluation_floor_nm: float = 1.0e-12,
    small_x_threshold: float = 1.0e-4,
    large_x_threshold: float = 500.0,
) -> np.ndarray | float:
    """Evaluate the RK interaction with controlled endpoint asymptotics.

    The small-r branch uses

        V(r) = C/(kappa*r0) [ln(r/(2r0)) + gamma - r/r0 + O(x^2 ln x)],

    and the large-r branch uses the Coulomb tail ``-C/(kappa*r)``.  The
    central interval uses SciPy's Struve and Bessel functions directly.
    """
    _validate_parameters(kappa, screening_length_nm, coulomb_constant_eV_nm)
    if evaluation_floor_nm <= 0.0:
        raise ValueError("evaluation_floor_nm must be positive")
    if small_x_threshold <= 0.0:
        raise ValueError("small_x_threshold must be positive")
    if large_x_threshold <= small_x_threshold:
        raise ValueError("large_x_threshold must exceed small_x_threshold")

    original = np.asarray(r_nm, dtype=float)
    scalar = original.ndim == 0
    radii = np.maximum(np.abs(original), float(evaluation_floor_nm))
    x = radii / float(screening_length_nm)
    values = np.empty_like(x, dtype=float)

    small = x < float(small_x_threshold)
    large = x > float(large_x_threshold)
    middle = ~(small | large)

    A = rk_short_distance_coefficient_eV(
        kappa=kappa,
        screening_length_nm=screening_length_nm,
        coulomb_constant_eV_nm=coulomb_constant_eV_nm,
    )
    if np.any(small):
        xs = x[small]
        values[small] = A * (np.log(xs / 2.0) + EULER_GAMMA - xs)
    if np.any(middle):
        xm = x[middle]
        prefactor = -np.pi * float(coulomb_constant_eV_nm) / (
            2.0 * float(kappa) * float(screening_length_nm)
        )
        values[middle] = prefactor * (struve(0, xm) - y0(xm))
    if np.any(large):
        values[large] = -float(coulomb_constant_eV_nm) / (
            float(kappa) * radii[large]
        )

    if not np.all(np.isfinite(values)):
        raise FloatingPointError("RK evaluator produced non-finite values")
    return float(values) if scalar else values


def build_hybrid_radial_grid_nm(
    *,
    r_min_nm: float,
    log_switch_nm: float,
    r_max_nm: float,
    n_log: int,
    n_linear: int,
) -> np.ndarray:
    """Create a dense log-near-origin / linear-far-field radial grid."""
    if not (0.0 < r_min_nm < log_switch_nm < r_max_nm):
        raise ValueError("Require 0 < r_min_nm < log_switch_nm < r_max_nm")
    if n_log < 16 or n_linear < 16:
        raise ValueError("n_log and n_linear must each be >= 16")
    logarithmic = np.geomspace(float(r_min_nm), float(log_switch_nm), int(n_log))
    linear = np.linspace(float(log_switch_nm), float(r_max_nm), int(n_linear))
    grid = np.concatenate([logarithmic, linear[1:]])
    if np.any(np.diff(grid) <= 0.0):
        raise RuntimeError("Constructed radial grid is not strictly increasing")
    return grid


@dataclass(frozen=True)
class RytovaKeldyshTablePotential:
    """Fast radial RK potential using a precomputed one-dimensional table."""

    r_nm: np.ndarray
    V_eV: np.ndarray
    kappa: float
    screening_length_nm: float
    coulomb_constant_eV_nm: float = COULOMB_CONSTANT_EV_NM
    evaluation_floor_nm: float = 1.0e-12

    def __post_init__(self) -> None:
        r = np.asarray(self.r_nm, dtype=float)
        V = np.asarray(self.V_eV, dtype=float)
        _validate_parameters(
            self.kappa, self.screening_length_nm, self.coulomb_constant_eV_nm
        )
        if r.ndim != 1 or V.ndim != 1 or r.shape != V.shape:
            raise ValueError("r_nm and V_eV must be one-dimensional arrays of equal length")
        if r.size < 32:
            raise ValueError("RK table must contain at least 32 points")
        if r[0] <= 0.0 or np.any(np.diff(r) <= 0.0):
            raise ValueError("r_nm must be strictly increasing and positive")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(V)):
            raise ValueError("RK table contains non-finite values")
        if self.evaluation_floor_nm <= 0.0:
            raise ValueError("evaluation_floor_nm must be positive")
        object.__setattr__(self, "r_nm", r)
        object.__setattr__(self, "V_eV", V)

    @property
    def r_min_nm(self) -> float:
        return float(self.r_nm[0])

    @property
    def r_max_nm(self) -> float:
        return float(self.r_nm[-1])

    @property
    def short_distance_coefficient_eV(self) -> float:
        return rk_short_distance_coefficient_eV(
            kappa=self.kappa,
            screening_length_nm=self.screening_length_nm,
            coulomb_constant_eV_nm=self.coulomb_constant_eV_nm,
        )

    def radial_value(self, radii_nm: np.ndarray | float) -> np.ndarray | float:
        original = np.asarray(radii_nm, dtype=float)
        scalar = original.ndim == 0
        radii = np.maximum(np.abs(original), float(self.evaluation_floor_nm))
        values = np.empty_like(radii, dtype=float)

        below = radii < self.r_min_nm
        above = radii > self.r_max_nm
        inside = ~(below | above)

        if np.any(inside):
            values[inside] = np.interp(
                radii[inside], self.r_nm, self.V_eV
            )
        if np.any(below):
            x = radii[below] / float(self.screening_length_nm)
            values[below] = self.short_distance_coefficient_eV * (
                np.log(x / 2.0) + EULER_GAMMA - x
            )
        if np.any(above):
            values[above] = -float(self.coulomb_constant_eV_nm) / (
                float(self.kappa) * radii[above]
            )

        if not np.all(np.isfinite(values)):
            raise FloatingPointError("RK table interpolation produced non-finite values")
        return float(values) if scalar else values

    def value(self, r: np.ndarray) -> np.ndarray:
        points = np.asarray(r, dtype=float)
        was_vector = points.ndim == 1
        if was_vector:
            if points.shape != (2,):
                raise ValueError("A single coordinate must have shape (2,)")
            points = points.reshape(1, 2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Coordinates must have shape (N,2) or (2,)")
        radii = np.sqrt(np.einsum("ij,ij->i", points, points))
        values = np.asarray(self.radial_value(radii), dtype=float)
        return values[0] if was_vector else values


@dataclass(frozen=True)
class RytovaKeldyshWallPotential:
    """RK attraction plus a distant radial wall for finite-T validation."""

    rk: RytovaKeldyshTablePotential
    wall_radius_nm: float
    wall_height_eV: float
    wall_power: int = 8

    def __post_init__(self) -> None:
        if self.wall_radius_nm <= 0.0:
            raise ValueError("wall_radius_nm must be positive")
        if self.wall_height_eV < 0.0:
            raise ValueError("wall_height_eV must be non-negative")
        if self.wall_power < 2 or self.wall_power % 2 != 0:
            raise ValueError("wall_power must be an even integer >= 2")

    def wall_radial_value(self, radii_nm: np.ndarray | float) -> np.ndarray | float:
        radii = np.asarray(radii_nm, dtype=float)
        values = float(self.wall_height_eV) * (
            np.abs(radii) / float(self.wall_radius_nm)
        ) ** int(self.wall_power)
        return float(values) if values.ndim == 0 else values

    def central_value(self, r: np.ndarray) -> np.ndarray:
        return self.rk.value(r)

    def wall_value(self, r: np.ndarray) -> np.ndarray:
        points = np.asarray(r, dtype=float)
        was_vector = points.ndim == 1
        if was_vector:
            points = points.reshape(1, 2)
        radii = np.sqrt(np.einsum("ij,ij->i", points, points))
        values = np.asarray(self.wall_radial_value(radii), dtype=float)
        return values[0] if was_vector else values

    def value(self, r: np.ndarray) -> np.ndarray:
        return self.central_value(r) + self.wall_value(r)


def build_rk_table(
    *,
    kappa: float,
    screening_length_nm: float,
    r_min_nm: float = 1.0e-6,
    log_switch_nm: float | None = None,
    r_max_nm: float = 80.0,
    n_log: int = 4000,
    n_linear: int = 4000,
    coulomb_constant_eV_nm: float = COULOMB_CONSTANT_EV_NM,
    evaluation_floor_nm: float = 1.0e-12,
) -> RytovaKeldyshTablePotential:
    _validate_parameters(kappa, screening_length_nm, coulomb_constant_eV_nm)
    if log_switch_nm is None:
        log_switch_nm = min(0.5 * float(screening_length_nm), 0.2 * float(r_max_nm))
        log_switch_nm = max(log_switch_nm, 100.0 * float(r_min_nm))
    grid = build_hybrid_radial_grid_nm(
        r_min_nm=float(r_min_nm),
        log_switch_nm=float(log_switch_nm),
        r_max_nm=float(r_max_nm),
        n_log=int(n_log),
        n_linear=int(n_linear),
    )
    values = np.asarray(
        rk_energy_direct_eV(
            grid,
            kappa=kappa,
            screening_length_nm=screening_length_nm,
            coulomb_constant_eV_nm=coulomb_constant_eV_nm,
            evaluation_floor_nm=evaluation_floor_nm,
        ),
        dtype=float,
    )
    return RytovaKeldyshTablePotential(
        r_nm=grid,
        V_eV=values,
        kappa=float(kappa),
        screening_length_nm=float(screening_length_nm),
        coulomb_constant_eV_nm=float(coulomb_constant_eV_nm),
        evaluation_floor_nm=float(evaluation_floor_nm),
    )


def save_rk_table_npz(path: str | Path, table: RytovaKeldyshTablePotential) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format_version=np.asarray("rk-radial-table-v1"),
        r_nm=table.r_nm,
        V_eV=table.V_eV,
        kappa=float(table.kappa),
        screening_length_nm=float(table.screening_length_nm),
        coulomb_constant_eV_nm=float(table.coulomb_constant_eV_nm),
        evaluation_floor_nm=float(table.evaluation_floor_nm),
        convention=np.asarray(
            "V=-pi*C/(2*kappa*r0)*(H0(r/r0)-Y0(r/r0)); tail=-C/(kappa*r)"
        ),
    )
    return output


def load_rk_table_npz(path: str | Path) -> RytovaKeldyshTablePotential:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"RK table not found: {source}")
    with np.load(source, allow_pickle=False) as z:
        required = {
            "r_nm",
            "V_eV",
            "kappa",
            "screening_length_nm",
            "coulomb_constant_eV_nm",
            "evaluation_floor_nm",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            raise ValueError(f"RK table {source} is missing keys: {missing}")
        return RytovaKeldyshTablePotential(
            r_nm=np.asarray(z["r_nm"], dtype=float),
            V_eV=np.asarray(z["V_eV"], dtype=float),
            kappa=float(np.asarray(z["kappa"]).reshape(())),
            screening_length_nm=float(
                np.asarray(z["screening_length_nm"]).reshape(())
            ),
            coulomb_constant_eV_nm=float(
                np.asarray(z["coulomb_constant_eV_nm"]).reshape(())
            ),
            evaluation_floor_nm=float(
                np.asarray(z["evaluation_floor_nm"]).reshape(())
            ),
        )


def rk_table_diagnostics(
    table: RytovaKeldyshTablePotential,
    *,
    n_test: int = 20000,
) -> dict[str, Any]:
    """Compare the table against the direct special-function evaluator."""
    if n_test < 100:
        raise ValueError("n_test must be >= 100")
    test_r = np.geomspace(table.r_min_nm, table.r_max_nm, int(n_test))
    direct = np.asarray(
        rk_energy_direct_eV(
            test_r,
            kappa=table.kappa,
            screening_length_nm=table.screening_length_nm,
            coulomb_constant_eV_nm=table.coulomb_constant_eV_nm,
            evaluation_floor_nm=table.evaluation_floor_nm,
        ),
        dtype=float,
    )
    interpolated = np.asarray(table.radial_value(test_r), dtype=float)
    abs_error = np.abs(interpolated - direct)
    rel_error = abs_error / np.maximum(np.abs(direct), 1.0e-14)

    small_probe = max(table.evaluation_floor_nm * 100.0, table.r_min_nm * 0.1)
    large_probe = max(table.r_max_nm * 10.0, table.screening_length_nm * 100.0)
    small_direct = float(
        rk_energy_direct_eV(
            small_probe,
            kappa=table.kappa,
            screening_length_nm=table.screening_length_nm,
            coulomb_constant_eV_nm=table.coulomb_constant_eV_nm,
            evaluation_floor_nm=table.evaluation_floor_nm,
        )
    )
    small_table = float(table.radial_value(small_probe))
    large_table = float(table.radial_value(large_probe))
    large_coulomb = -table.coulomb_constant_eV_nm / (table.kappa * large_probe)

    return {
        "n_table_points": int(table.r_nm.size),
        "r_min_nm": table.r_min_nm,
        "r_max_nm": table.r_max_nm,
        "kappa": float(table.kappa),
        "screening_length_nm": float(table.screening_length_nm),
        "short_distance_coefficient_eV": table.short_distance_coefficient_eV,
        "max_abs_interpolation_error_eV": float(np.max(abs_error)),
        "rms_abs_interpolation_error_eV": float(np.sqrt(np.mean(abs_error**2))),
        "max_relative_interpolation_error": float(np.max(rel_error)),
        "rms_relative_interpolation_error": float(np.sqrt(np.mean(rel_error**2))),
        "small_probe_nm": float(small_probe),
        "small_branch_error_eV": float(small_table - small_direct),
        "large_probe_nm": float(large_probe),
        "large_tail_ratio_to_coulomb": float(large_table / large_coulomb),
        "V_at_rmin_eV": float(table.V_eV[0]),
        "V_at_rmax_eV": float(table.V_eV[-1]),
    }
