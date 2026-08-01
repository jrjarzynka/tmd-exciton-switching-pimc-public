"""Material-specific bilayer Keldysh interaction for direct TMD heterobilayers.

The model describes an interlayer exciton with an electron and a hole confined
in two adjacent monolayers.  Their in-plane relative coordinate is ``rho`` and
the carrier planes are separated by a fixed vertical distance ``D``.

For layer-resolved screening lengths r0_1 and r0_2, the analytical bilayer
Keldysh (BLK) potential is

    V(rho) = -pi C / (2 r0_sum) * [H0(x) - Y0(x)],
    x      = kappa * sqrt(rho**2 + D**2) / r0_sum,
    r0_sum = r0_1 + r0_2,

where C=e^2/(4*pi*epsilon_0)=1.43996448 eV nm.  This is Eq. (5) of
Kamban and Pedersen, Scientific Reports 10, 5537 (2020), with SI units
restored.  The finite D makes V(0) finite.

Distances are in nm and energies in eV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import struve, yv

COULOMB_CONSTANT_EV_NM = 1.43996448


def screening_length_from_chi2d_nm(chi2d_angstrom: float) -> float:
    """Return intrinsic screening length r0=2*pi*chi_2D in nm."""
    chi = float(chi2d_angstrom)
    if not np.isfinite(chi) or chi <= 0.0:
        raise ValueError("chi2d_angstrom must be positive and finite")
    return float(2.0 * np.pi * chi * 0.1)


def _validate_parameters(
    *,
    separation_nm: float,
    screening_length_layer1_nm: float,
    screening_length_layer2_nm: float,
    kappa_environment: float,
    coulomb_constant_eV_nm: float,
) -> None:
    for name, value in (
        ("separation_nm", separation_nm),
        ("screening_length_layer1_nm", screening_length_layer1_nm),
        ("screening_length_layer2_nm", screening_length_layer2_nm),
        ("kappa_environment", kappa_environment),
        ("coulomb_constant_eV_nm", coulomb_constant_eV_nm),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")


def bilayer_keldysh_value_eV(
    radii_nm: np.ndarray | float,
    *,
    separation_nm: float,
    screening_length_layer1_nm: float,
    screening_length_layer2_nm: float,
    kappa_environment: float,
    coulomb_constant_eV_nm: float = COULOMB_CONSTANT_EV_NM,
) -> np.ndarray | float:
    """Evaluate the analytical material-specific BLK potential."""
    _validate_parameters(
        separation_nm=separation_nm,
        screening_length_layer1_nm=screening_length_layer1_nm,
        screening_length_layer2_nm=screening_length_layer2_nm,
        kappa_environment=kappa_environment,
        coulomb_constant_eV_nm=coulomb_constant_eV_nm,
    )
    original = np.asarray(radii_nm, dtype=float)
    scalar = original.ndim == 0
    rho = np.abs(original)
    if not np.all(np.isfinite(rho)):
        raise ValueError("radii_nm contains non-finite values")

    r0_sum = float(screening_length_layer1_nm + screening_length_layer2_nm)
    three_dimensional_distance = np.sqrt(rho * rho + float(separation_nm) ** 2)
    x = float(kappa_environment) * three_dimensional_distance / r0_sum
    values = -(
        np.pi * float(coulomb_constant_eV_nm) / (2.0 * r0_sum)
    ) * (struve(0, x) - yv(0, x))

    if not np.all(np.isfinite(values)):
        raise FloatingPointError("BLK evaluation produced non-finite energies")
    return float(values) if scalar else values


def build_hybrid_radial_grid_nm(
    *,
    r_min_positive_nm: float,
    log_switch_nm: float,
    r_max_nm: float,
    n_log: int,
    n_linear: int,
    include_origin: bool = True,
) -> np.ndarray:
    if not (0.0 < r_min_positive_nm < log_switch_nm < r_max_nm):
        raise ValueError(
            "Require 0 < r_min_positive_nm < log_switch_nm < r_max_nm"
        )
    if n_log < 16 or n_linear < 16:
        raise ValueError("n_log and n_linear must each be >= 16")
    log_grid = np.geomspace(r_min_positive_nm, log_switch_nm, int(n_log))
    linear_grid = np.linspace(log_switch_nm, r_max_nm, int(n_linear))
    pieces = [log_grid, linear_grid[1:]]
    if include_origin:
        pieces.insert(0, np.array([0.0], dtype=float))
    grid = np.concatenate(pieces)
    if np.any(np.diff(grid) <= 0.0):
        raise RuntimeError("Radial grid is not strictly increasing")
    return grid


@dataclass(frozen=True)
class BilayerKeldyshTablePotential:
    """Fast radial table for the analytical material-specific BLK attraction."""

    r_nm: np.ndarray
    V_eV: np.ndarray
    separation_nm: float
    screening_length_layer1_nm: float
    screening_length_layer2_nm: float
    kappa_environment: float
    layer1_name: str = "MoSe2"
    layer2_name: str = "WSe2"
    coulomb_constant_eV_nm: float = COULOMB_CONSTANT_EV_NM

    def __post_init__(self) -> None:
        r = np.asarray(self.r_nm, dtype=float)
        V = np.asarray(self.V_eV, dtype=float)
        _validate_parameters(
            separation_nm=self.separation_nm,
            screening_length_layer1_nm=self.screening_length_layer1_nm,
            screening_length_layer2_nm=self.screening_length_layer2_nm,
            kappa_environment=self.kappa_environment,
            coulomb_constant_eV_nm=self.coulomb_constant_eV_nm,
        )
        if r.ndim != 1 or V.ndim != 1 or r.shape != V.shape:
            raise ValueError("r_nm and V_eV must have identical one-dimensional shapes")
        if r.size < 32 or r[0] != 0.0 or np.any(np.diff(r) <= 0.0):
            raise ValueError("Table must start at r=0 and increase strictly")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(V)):
            raise ValueError("BLK table contains non-finite values")
        object.__setattr__(self, "r_nm", r)
        object.__setattr__(self, "V_eV", V)

    @property
    def screening_length_sum_nm(self) -> float:
        return float(
            self.screening_length_layer1_nm + self.screening_length_layer2_nm
        )

    @property
    def r_max_nm(self) -> float:
        return float(self.r_nm[-1])

    @property
    def value_at_origin_eV(self) -> float:
        return float(self.V_eV[0])

    def analytic_radial_value(
        self, radii_nm: np.ndarray | float
    ) -> np.ndarray | float:
        return bilayer_keldysh_value_eV(
            radii_nm,
            separation_nm=self.separation_nm,
            screening_length_layer1_nm=self.screening_length_layer1_nm,
            screening_length_layer2_nm=self.screening_length_layer2_nm,
            kappa_environment=self.kappa_environment,
            coulomb_constant_eV_nm=self.coulomb_constant_eV_nm,
        )

    def radial_value(self, radii_nm: np.ndarray | float) -> np.ndarray | float:
        original = np.asarray(radii_nm, dtype=float)
        scalar = original.ndim == 0
        radii = np.abs(original)
        values = np.empty_like(radii)
        inside = radii <= self.r_max_nm
        if np.any(inside):
            values[inside] = np.interp(radii[inside], self.r_nm, self.V_eV)
        if np.any(~inside):
            values[~inside] = np.asarray(
                self.analytic_radial_value(radii[~inside]), dtype=float
            )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("BLK table interpolation failed")
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
class BilayerKeldyshWallPotential:
    bilayer: BilayerKeldyshTablePotential
    wall_radius_nm: float
    wall_height_eV: float
    wall_power: int = 8

    def __post_init__(self) -> None:
        if self.wall_radius_nm <= 0.0:
            raise ValueError("wall_radius_nm must be positive")
        if self.wall_height_eV < 0.0:
            raise ValueError("wall_height_eV must be non-negative")
        if self.wall_power < 2 or self.wall_power % 2:
            raise ValueError("wall_power must be an even integer >= 2")

    def wall_radial_value(self, radii_nm: np.ndarray | float) -> np.ndarray | float:
        radii = np.asarray(radii_nm, dtype=float)
        values = float(self.wall_height_eV) * (
            np.abs(radii) / float(self.wall_radius_nm)
        ) ** int(self.wall_power)
        return float(values) if values.ndim == 0 else values

    def central_value(self, r: np.ndarray) -> np.ndarray:
        return self.bilayer.value(r)

    def wall_value(self, r: np.ndarray) -> np.ndarray:
        points = np.asarray(r, dtype=float)
        was_vector = points.ndim == 1
        if was_vector:
            if points.shape != (2,):
                raise ValueError("A single coordinate must have shape (2,)")
            points = points.reshape(1, 2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Coordinates must have shape (N,2) or (2,)")
        radii = np.sqrt(np.einsum("ij,ij->i", points, points))
        values = np.asarray(self.wall_radial_value(radii), dtype=float)
        return values[0] if was_vector else values

    def value(self, r: np.ndarray) -> np.ndarray:
        return self.central_value(r) + self.wall_value(r)


def build_bilayer_keldysh_table(
    *,
    separation_nm: float,
    screening_length_layer1_nm: float,
    screening_length_layer2_nm: float,
    kappa_environment: float,
    layer1_name: str = "MoSe2",
    layer2_name: str = "WSe2",
    r_min_positive_nm: float = 1.0e-4,
    log_switch_nm: float = 0.5,
    r_max_nm: float = 80.0,
    n_log: int = 1000,
    n_linear: int = 2000,
    coulomb_constant_eV_nm: float = COULOMB_CONSTANT_EV_NM,
) -> BilayerKeldyshTablePotential:
    r = build_hybrid_radial_grid_nm(
        r_min_positive_nm=r_min_positive_nm,
        log_switch_nm=log_switch_nm,
        r_max_nm=r_max_nm,
        n_log=n_log,
        n_linear=n_linear,
        include_origin=True,
    )
    V = np.asarray(
        bilayer_keldysh_value_eV(
            r,
            separation_nm=separation_nm,
            screening_length_layer1_nm=screening_length_layer1_nm,
            screening_length_layer2_nm=screening_length_layer2_nm,
            kappa_environment=kappa_environment,
            coulomb_constant_eV_nm=coulomb_constant_eV_nm,
        ),
        dtype=float,
    )
    return BilayerKeldyshTablePotential(
        r_nm=r,
        V_eV=V,
        separation_nm=float(separation_nm),
        screening_length_layer1_nm=float(screening_length_layer1_nm),
        screening_length_layer2_nm=float(screening_length_layer2_nm),
        kappa_environment=float(kappa_environment),
        layer1_name=str(layer1_name),
        layer2_name=str(layer2_name),
        coulomb_constant_eV_nm=float(coulomb_constant_eV_nm),
    )


def save_bilayer_keldysh_table_npz(
    path: str | Path,
    potential: BilayerKeldyshTablePotential,
    **extra_metadata: Any,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "r_nm": potential.r_nm,
        "V_eV": potential.V_eV,
        "separation_nm": potential.separation_nm,
        "screening_length_layer1_nm": potential.screening_length_layer1_nm,
        "screening_length_layer2_nm": potential.screening_length_layer2_nm,
        "screening_length_sum_nm": potential.screening_length_sum_nm,
        "kappa_environment": potential.kappa_environment,
        "layer1_name": np.asarray(potential.layer1_name),
        "layer2_name": np.asarray(potential.layer2_name),
        "coulomb_constant_eV_nm": potential.coulomb_constant_eV_nm,
        "value_at_origin_eV": potential.value_at_origin_eV,
        "model_name": np.asarray("bilayer_keldysh_material_specific"),
    }
    payload.update(extra_metadata)
    np.savez_compressed(output, **payload)


def load_bilayer_keldysh_table_npz(
    path: str | Path,
) -> BilayerKeldyshTablePotential:
    with np.load(Path(path), allow_pickle=False) as data:
        return BilayerKeldyshTablePotential(
            r_nm=np.asarray(data["r_nm"], dtype=float),
            V_eV=np.asarray(data["V_eV"], dtype=float),
            separation_nm=float(data["separation_nm"]),
            screening_length_layer1_nm=float(data["screening_length_layer1_nm"]),
            screening_length_layer2_nm=float(data["screening_length_layer2_nm"]),
            kappa_environment=float(data["kappa_environment"]),
            layer1_name=str(np.asarray(data["layer1_name"]).item()),
            layer2_name=str(np.asarray(data["layer2_name"]).item()),
            coulomb_constant_eV_nm=float(data["coulomb_constant_eV_nm"]),
        )


def bilayer_keldysh_table_diagnostics(
    potential: BilayerKeldyshTablePotential,
    *,
    n_test: int = 128,
) -> dict[str, float | int | str]:
    """Check interpolation, monotonicity, finite origin and Coulomb tail."""
    positive = potential.r_nm[potential.r_nm > 0.0]
    test_r = np.geomspace(
        max(float(positive[0]) * 1.037, 1.0e-4),
        min(float(potential.r_max_nm) * 0.997, 60.0),
        int(n_test),
    )
    interpolated = np.asarray(potential.radial_value(test_r), dtype=float)
    exact = np.asarray(potential.analytic_radial_value(test_r), dtype=float)
    absolute = np.abs(interpolated - exact)
    relative = absolute / np.maximum(np.abs(exact), 1.0e-12)

    diffs = np.diff(potential.V_eV)
    monotonic_violations = int(np.count_nonzero(diffs < -1.0e-12))
    minimum_increment = float(np.min(diffs))

    small_mask = potential.r_nm <= min(0.12, 0.2 * potential.separation_nm)
    small_r = potential.r_nm[small_mask]
    small_V = potential.V_eV[small_mask]
    if small_r.size >= 5:
        design = np.column_stack([np.ones_like(small_r), small_r**2])
        coefficients, *_ = np.linalg.lstsq(design, small_V, rcond=None)
        fitted = design @ coefficients
        small_r_quadratic_max_abs_residual_eV = float(
            np.max(np.abs(small_V - fitted))
        )
        origin_curvature_eV_per_nm2 = float(coefficients[1])
    else:
        small_r_quadratic_max_abs_residual_eV = float("nan")
        origin_curvature_eV_per_nm2 = float("nan")

    tail_r = float(potential.r_max_nm)
    softened_distance = float(np.sqrt(tail_r**2 + potential.separation_nm**2))
    tail_expected = -potential.coulomb_constant_eV_nm / (
        potential.kappa_environment * softened_distance
    )
    tail_actual = float(potential.V_eV[-1])

    return {
        "model_name": "bilayer_keldysh_material_specific",
        "layer1_name": potential.layer1_name,
        "layer2_name": potential.layer2_name,
        "table_points": int(potential.r_nm.size),
        "r_max_nm": potential.r_max_nm,
        "separation_nm": potential.separation_nm,
        "screening_length_layer1_nm": potential.screening_length_layer1_nm,
        "screening_length_layer2_nm": potential.screening_length_layer2_nm,
        "screening_length_sum_nm": potential.screening_length_sum_nm,
        "kappa_environment": potential.kappa_environment,
        "value_at_origin_eV": potential.value_at_origin_eV,
        "max_absolute_interpolation_error_eV": float(np.max(absolute)),
        "max_relative_interpolation_error": float(np.max(relative)),
        "rms_relative_interpolation_error": float(np.sqrt(np.mean(relative**2))),
        "monotonicity_violations": monotonic_violations,
        "minimum_table_increment_eV": minimum_increment,
        "origin_curvature_eV_per_nm2": origin_curvature_eV_per_nm2,
        "small_r_quadratic_max_abs_residual_eV": small_r_quadratic_max_abs_residual_eV,
        "tail_actual_eV": tail_actual,
        "tail_expected_soft_coulomb_eV": float(tail_expected),
        "tail_relative_error": float(abs(tail_actual - tail_expected) / abs(tail_expected)),
    }
