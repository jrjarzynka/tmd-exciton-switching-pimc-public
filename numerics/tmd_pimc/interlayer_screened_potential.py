"""Direct-contact interlayer exciton interaction for TMD bilayers.

This module implements the rotationally symmetric real-space interaction
between an electron and a hole confined to two adjacent TMD monolayers,
separated by a finite vertical distance ``D``.  The potential is generated as
a one-dimensional radial table from a momentum-space screened interaction.

Two environment models are provided:

``suspended``
    Direct bilayer in vacuum.  For equal single-layer screening length ``r0``
    the real-space interaction is

        V(r) = -C int_0^inf dq J0(q r) exp(-q D)
               / [(1+r0 q)^2 - (r0 q)^2 exp(-2 q D)].

``symmetric_encapsulation``
    Direct bilayer surrounded symmetrically by a dielectric with permittivity
    ``epsilon_env``.  The dielectric interfaces are a distance ``d_interface``
    from the nearest TMD carrier plane.  This follows Eq. (1) and Appendix I
    of Tang et al., arXiv:2410.16717, specialized to equal screening lengths in
    the two TMD layers.

The finite interlayer separation makes V(0) finite.  Consequently the
short-distance primitive-action non-normalisability of the intralayer
Rytova--Keldysh logarithmic singularity is absent here.

Units
-----
Distances are in nm, wavevectors in nm^-1 and energies in eV.
``C = e^2/(4 pi epsilon_0)`` is represented by 1.43996448 eV nm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.special import j0

COULOMB_CONSTANT_EV_NM = 1.43996448
EnvironmentModel = Literal["suspended", "symmetric_encapsulation"]


def _validate_parameters(
    *,
    separation_nm: float,
    screening_length_nm: float,
    environment_model: EnvironmentModel,
    epsilon_env: float,
    interface_distance_nm: float,
    coulomb_constant_eV_nm: float,
) -> None:
    if not np.isfinite(separation_nm) or separation_nm <= 0.0:
        raise ValueError("separation_nm must be positive and finite")
    if not np.isfinite(screening_length_nm) or screening_length_nm <= 0.0:
        raise ValueError("screening_length_nm must be positive and finite")
    if environment_model not in ("suspended", "symmetric_encapsulation"):
        raise ValueError(f"Unknown environment_model={environment_model!r}")
    if not np.isfinite(coulomb_constant_eV_nm) or coulomb_constant_eV_nm <= 0.0:
        raise ValueError("coulomb_constant_eV_nm must be positive and finite")
    if environment_model == "symmetric_encapsulation":
        if not np.isfinite(epsilon_env) or epsilon_env <= 1.0:
            raise ValueError("epsilon_env must exceed 1 for encapsulation")
        if not np.isfinite(interface_distance_nm) or interface_distance_nm < 0.0:
            raise ValueError("interface_distance_nm must be finite and non-negative")


def interlayer_screening_kernel(
    q_nm_inv: np.ndarray | float,
    *,
    separation_nm: float,
    screening_length_nm: float,
    environment_model: EnvironmentModel,
    epsilon_env: float = 4.5,
    interface_distance_nm: float = 0.5,
) -> np.ndarray | float:
    """Return the positive radial Hankel kernel K(q).

    The real-space attraction is ``V(r) = -C int dq J0(qr) K(q)``.
    """
    _validate_parameters(
        separation_nm=separation_nm,
        screening_length_nm=screening_length_nm,
        environment_model=environment_model,
        epsilon_env=epsilon_env,
        interface_distance_nm=interface_distance_nm,
        coulomb_constant_eV_nm=COULOMB_CONSTANT_EV_NM,
    )

    original = np.asarray(q_nm_inv, dtype=float)
    scalar = original.ndim == 0
    q = np.abs(original)
    D = float(separation_nm)
    r0 = float(screening_length_nm)
    eD = np.exp(-q * D)

    if environment_model == "suspended":
        denominator = (1.0 + r0 * q) ** 2 - (r0 * q) ** 2 * eD**2
        kernel = eD / denominator
    else:
        eps = float(epsilon_env)
        d = float(interface_distance_nm)
        reflection = (eps - 1.0) / (eps + 1.0)
        M = (1.0 - reflection * np.exp(-2.0 * q * d)) ** 2
        Q = (
            1.0 - reflection * np.exp(-q * (D + 2.0 * d))
        ) * (
            1.0 + reflection * np.exp(-q * (D + 2.0 * d))
        )
        N = (
            1.0 - reflection * np.exp(-2.0 * q * (D + d))
        ) * (
            1.0 - reflection * np.exp(-2.0 * q * d)
        )
        denominator = (
            Q + (N - eD * M) * q * r0
        ) * (
            Q + (N + eD * M) * q * r0
        )
        kernel = eD * M * Q / denominator

    if not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
        raise FloatingPointError("Interlayer screening kernel is invalid")
    return float(kernel) if scalar else kernel


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


def build_q_grid_nm_inv(*, q_max_nm_inv: float, n_q: int) -> np.ndarray:
    if q_max_nm_inv <= 0.0:
        raise ValueError("q_max_nm_inv must be positive")
    if n_q < 1000:
        raise ValueError("n_q must be >= 1000")
    return np.linspace(0.0, float(q_max_nm_inv), int(n_q), dtype=float)


def hankel_transform_table_eV(
    radii_nm: np.ndarray,
    *,
    separation_nm: float,
    screening_length_nm: float,
    environment_model: EnvironmentModel,
    epsilon_env: float = 4.5,
    interface_distance_nm: float = 0.5,
    q_max_nm_inv: float = 50.0,
    n_q: int = 30001,
    chunk_size: int = 64,
    coulomb_constant_eV_nm: float = COULOMB_CONSTANT_EV_NM,
) -> np.ndarray:
    """Evaluate the screened interlayer potential on a radial grid.

    A fixed, uniformly spaced q grid is used so the trapezoidal weights can be
    reused efficiently in chunks of radial points.  The exponential factor
    exp(-qD) makes the transform rapidly convergent for direct TMD bilayers.
    """
    _validate_parameters(
        separation_nm=separation_nm,
        screening_length_nm=screening_length_nm,
        environment_model=environment_model,
        epsilon_env=epsilon_env,
        interface_distance_nm=interface_distance_nm,
        coulomb_constant_eV_nm=coulomb_constant_eV_nm,
    )
    r = np.asarray(radii_nm, dtype=float)
    if r.ndim != 1 or r.size < 16 or np.any(r < 0.0):
        raise ValueError("radii_nm must be a one-dimensional non-negative grid")
    if np.any(np.diff(r) <= 0.0):
        raise ValueError("radii_nm must be strictly increasing")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    q = build_q_grid_nm_inv(q_max_nm_inv=q_max_nm_inv, n_q=n_q)
    kernel = np.asarray(
        interlayer_screening_kernel(
            q,
            separation_nm=separation_nm,
            screening_length_nm=screening_length_nm,
            environment_model=environment_model,
            epsilon_env=epsilon_env,
            interface_distance_nm=interface_distance_nm,
        ),
        dtype=float,
    )

    # Trapezoidal weights for a uniform grid.
    dq = float(q[1] - q[0])
    weights = kernel.copy()
    weights[0] *= 0.5
    weights[-1] *= 0.5
    weights *= dq

    values = np.empty_like(r)
    for start in range(0, r.size, int(chunk_size)):
        stop = min(start + int(chunk_size), r.size)
        B = j0(np.outer(r[start:stop], q))
        values[start:stop] = -float(coulomb_constant_eV_nm) * (B @ weights)

    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Hankel transform produced non-finite energies")
    return values


@dataclass(frozen=True)
class InterlayerScreenedTablePotential:
    """Fast radial table for a finite-separation screened interlayer attraction."""

    r_nm: np.ndarray
    V_eV: np.ndarray
    separation_nm: float
    screening_length_nm: float
    environment_model: EnvironmentModel
    epsilon_env: float = 4.5
    interface_distance_nm: float = 0.5
    coulomb_constant_eV_nm: float = COULOMB_CONSTANT_EV_NM

    def __post_init__(self) -> None:
        r = np.asarray(self.r_nm, dtype=float)
        V = np.asarray(self.V_eV, dtype=float)
        _validate_parameters(
            separation_nm=self.separation_nm,
            screening_length_nm=self.screening_length_nm,
            environment_model=self.environment_model,
            epsilon_env=self.epsilon_env,
            interface_distance_nm=self.interface_distance_nm,
            coulomb_constant_eV_nm=self.coulomb_constant_eV_nm,
        )
        if r.ndim != 1 or V.ndim != 1 or r.shape != V.shape:
            raise ValueError("r_nm and V_eV must have identical 1D shapes")
        if r.size < 32 or r[0] != 0.0 or np.any(np.diff(r) <= 0.0):
            raise ValueError("Table must start at r=0 and increase strictly")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(V)):
            raise ValueError("Interlayer table contains non-finite values")
        object.__setattr__(self, "r_nm", r)
        object.__setattr__(self, "V_eV", V)

    @property
    def r_max_nm(self) -> float:
        return float(self.r_nm[-1])

    @property
    def value_at_origin_eV(self) -> float:
        return float(self.V_eV[0])

    @property
    def far_field_kappa(self) -> float:
        return 1.0 if self.environment_model == "suspended" else float(self.epsilon_env)

    def radial_value(self, radii_nm: np.ndarray | float) -> np.ndarray | float:
        original = np.asarray(radii_nm, dtype=float)
        scalar = original.ndim == 0
        radii = np.abs(original)
        values = np.empty_like(radii)
        inside = radii <= self.r_max_nm
        above = ~inside
        if np.any(inside):
            values[inside] = np.interp(radii[inside], self.r_nm, self.V_eV)
        if np.any(above):
            values[above] = -float(self.coulomb_constant_eV_nm) / (
                self.far_field_kappa * radii[above]
            )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Interlayer table interpolation failed")
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
class InterlayerScreenedWallPotential:
    interlayer: InterlayerScreenedTablePotential
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
        return self.interlayer.value(r)

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


def build_interlayer_table(
    *,
    separation_nm: float,
    screening_length_nm: float,
    environment_model: EnvironmentModel,
    epsilon_env: float = 4.5,
    interface_distance_nm: float = 0.5,
    r_min_positive_nm: float = 1.0e-4,
    log_switch_nm: float = 0.5,
    r_max_nm: float = 80.0,
    n_log: int = 1000,
    n_linear: int = 2000,
    q_max_nm_inv: float = 50.0,
    n_q: int = 30001,
    chunk_size: int = 64,
    coulomb_constant_eV_nm: float = COULOMB_CONSTANT_EV_NM,
) -> InterlayerScreenedTablePotential:
    r = build_hybrid_radial_grid_nm(
        r_min_positive_nm=r_min_positive_nm,
        log_switch_nm=log_switch_nm,
        r_max_nm=r_max_nm,
        n_log=n_log,
        n_linear=n_linear,
        include_origin=True,
    )
    V = hankel_transform_table_eV(
        r,
        separation_nm=separation_nm,
        screening_length_nm=screening_length_nm,
        environment_model=environment_model,
        epsilon_env=epsilon_env,
        interface_distance_nm=interface_distance_nm,
        q_max_nm_inv=q_max_nm_inv,
        n_q=n_q,
        chunk_size=chunk_size,
        coulomb_constant_eV_nm=coulomb_constant_eV_nm,
    )
    return InterlayerScreenedTablePotential(
        r_nm=r,
        V_eV=V,
        separation_nm=float(separation_nm),
        screening_length_nm=float(screening_length_nm),
        environment_model=environment_model,
        epsilon_env=float(epsilon_env),
        interface_distance_nm=float(interface_distance_nm),
        coulomb_constant_eV_nm=float(coulomb_constant_eV_nm),
    )


def save_interlayer_table_npz(
    path: str | Path,
    potential: InterlayerScreenedTablePotential,
    **extra_metadata: Any,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "r_nm": potential.r_nm,
        "V_eV": potential.V_eV,
        "separation_nm": potential.separation_nm,
        "screening_length_nm": potential.screening_length_nm,
        "environment_model": np.asarray(potential.environment_model),
        "epsilon_env": potential.epsilon_env,
        "interface_distance_nm": potential.interface_distance_nm,
        "coulomb_constant_eV_nm": potential.coulomb_constant_eV_nm,
        "value_at_origin_eV": potential.value_at_origin_eV,
    }
    payload.update(extra_metadata)
    np.savez_compressed(output, **payload)


def load_interlayer_table_npz(path: str | Path) -> InterlayerScreenedTablePotential:
    with np.load(Path(path), allow_pickle=False) as data:
        return InterlayerScreenedTablePotential(
            r_nm=np.asarray(data["r_nm"], dtype=float),
            V_eV=np.asarray(data["V_eV"], dtype=float),
            separation_nm=float(data["separation_nm"]),
            screening_length_nm=float(data["screening_length_nm"]),
            environment_model=str(np.asarray(data["environment_model"]).item()),
            epsilon_env=float(data["epsilon_env"]),
            interface_distance_nm=float(data["interface_distance_nm"]),
            coulomb_constant_eV_nm=float(data["coulomb_constant_eV_nm"]),
        )


def interlayer_table_diagnostics(
    potential: InterlayerScreenedTablePotential,
    *,
    q_max_nm_inv: float,
    n_q: int,
    n_test: int = 96,
) -> dict[str, float | int | str]:
    """Check q-grid convergence and the asymptotic Coulomb tail."""
    positive = potential.r_nm[potential.r_nm > 0.0]
    test_r = np.geomspace(
        max(float(positive[0]), 1.0e-4),
        min(float(potential.r_max_nm), 40.0),
        int(n_test),
    )
    coarse = np.asarray(potential.radial_value(test_r), dtype=float)
    refined = hankel_transform_table_eV(
        test_r,
        separation_nm=potential.separation_nm,
        screening_length_nm=potential.screening_length_nm,
        environment_model=potential.environment_model,
        epsilon_env=potential.epsilon_env,
        interface_distance_nm=potential.interface_distance_nm,
        q_max_nm_inv=float(q_max_nm_inv) * 1.25,
        n_q=int(2 * n_q - 1),
        chunk_size=32,
        coulomb_constant_eV_nm=potential.coulomb_constant_eV_nm,
    )
    absolute = np.abs(coarse - refined)
    scale = np.maximum(np.abs(refined), 1.0e-12)
    relative = absolute / scale

    tail_r = float(potential.r_max_nm)
    tail_expected = -potential.coulomb_constant_eV_nm / (
        potential.far_field_kappa * tail_r
    )
    tail_actual = float(potential.V_eV[-1])
    return {
        "environment_model": potential.environment_model,
        "table_points": int(potential.r_nm.size),
        "r_max_nm": potential.r_max_nm,
        "q_max_nm_inv": float(q_max_nm_inv),
        "n_q": int(n_q),
        "value_at_origin_eV": potential.value_at_origin_eV,
        "max_absolute_q_refinement_error_eV": float(np.max(absolute)),
        "max_relative_q_refinement_error": float(np.max(relative)),
        "rms_relative_q_refinement_error": float(np.sqrt(np.mean(relative**2))),
        "tail_actual_eV": tail_actual,
        "tail_expected_eV": float(tail_expected),
        "tail_relative_error": float(abs(tail_actual - tail_expected) / abs(tail_expected)),
    }
