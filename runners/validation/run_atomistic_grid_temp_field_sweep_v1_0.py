"""
v1.0 — Atomistic-grid COM exciton PI-QMC temperature + effective-field sweep

Purpose
-------
This runner is meant for the atomistic WSe2/MoSe2 moiré workflow:

    relaxed / generated structure -> V_grid.npz -> GridPotential2D -> PI-QMC

It preserves the important outputs of the older analytic sweep script:

    - per-seed NPZ files
    - ensemble NPZ files for each (T, Ex)
    - centroid histogram density_H
    - normalised probability_H
    - free_energy and free_energy_masked
    - centroids
    - mean_x, mean_y
    - r2_bead, r2_centroid
    - acceptance diagnostics
    - n_beads, beta, RNG seed, start centre, local/global step
    - p_capped warning flag
    - final CSV summary

The central difference is that the base potential is ALWAYS a grid loaded from
V_grid.npz. Analytic moiré/piezo terms are intentionally not used here.
Optional weak envelope and optional soft-Coulomb terms are available, but OFF/ON
choices should be made consciously to avoid double-counting physics already
encoded in V_grid.npz.

Recommended smoke test from the project root:

    export PYTHONPATH="$PWD/code:$PYTHONPATH"

    python3 runners/validation/run_atomistic_grid_temp_field_sweep_v1_0.py \
      --potential_npz results/wse2_mose2/V_grid.npz \
      --temps 20 \
      --quick \
      --seeds 1 \
      --workers 1

Recommended production-style example:

    python3 runners/validation/run_atomistic_grid_temp_field_sweep_v1_0.py \
      --potential_npz results/wse2_mose2/V_grid.npz \
      --temps 20 50 80 100 120 150 \
      --start_mode grid_cycle \
      --n_steps 80000 \
      --burn_in 20000 \
      --sample_every 50 \
      --seeds 5 \
      --workers 4 \
      --output_tag wse2_mose2_atomistic_grid_v1_0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project root / import handling
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    """Find the project root robustly when the script lives in scripts/."""
    env_root = os.environ.get("TMD_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    candidates = [Path.cwd().resolve()]
    candidates.extend(parent for parent in here.parents[:4])

    for cand in candidates:
        if (cand / "numerics" / "tmd_pimc").exists():
            return cand
        if (cand / "numerics").exists() and (cand / "results").exists():
            return cand

    # Fallback: old layout, script in <root>/scripts/
    return here.parents[1]


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "numerics"))

try:
    from tmd_pimc import (  # noqa: E402
        RingPolymerAction,
        PIMCSamplerJIT,
        HarmonicEnvelopePotential,
        ExternalFieldPotential,
        CompositePotential,
        SoftCoulombPotential,
        GridPotential2D,
        r2_mean_pimc,
        r2_mean_centroid,
        r2_spread_pimc,
        centroids,
        KB_EV_PER_K,
        HBAR2_OVER_2M0,
    )
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise SystemExit(
        "\nERROR: Could not import the grid-ready tmd_pimc package.\n"
        "Make sure code/tmd_pimc has the GridPotential2D patch and run:\n\n"
        "  export PYTHONPATH=\"$PWD/code:$PYTHONPATH\"\n"
        "  python3 -c \"from tmd_pimc import GridPotential2D; print(GridPotential2D)\"\n\n"
        f"Original import error: {exc}\n"
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MASS_M0_DEFAULT = 0.5

# Atomistic workflow defaults: focus around the WSe2/MoSe2 localisation ->
# delocalisation temperature window. Override with --temps for low-T checks.
T_VALUES_DEFAULT = [20.0, 50.0, 80.0, 100.0, 120.0, 150.0]
EX_VALUES_DEFAULT = np.linspace(-0.001, 0.001, 21)  # effective tilt, eV/nm
N_SEEDS_DEFAULT = 5

# Weak confinement, same functional form as the old script, but now optional.
# It is ON by default because Ex + periodic grid is otherwise a tilted torus / finite-window problem.
V0_ENVELOPE_DEFAULT = 0.010  # eV at R_ENVELOPE
R_ENVELOPE_DEFAULT = 30.0    # nm

# Keep this OFF by default for atomistic-grid runs: V_grid may already encode
# excitonic/registry/strain terms, and this effective COM Coulomb term is large
# with the old parameters.
COULOMB_STRENGTH_DEFAULT = 0.15   # eV nm
COULOMB_SOFTENING_DEFAULT = 1.0   # nm

GLOBAL_STEP_NM_DEFAULT = 2.0
LOCAL_STEP_MULTIPLIER_DEFAULT = 0.70

MIN_COUNTS_FOR_FREE_ENERGY_DEFAULT = 5
BASE_SEED_DEFAULT = 1729

StartMode = Literal["grid_cycle", "grid_min", "field", "random_grid", "random_box", "origin", "hex"]
CoordinateOrigin = Literal["center", "native"]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridInfo:
    x_axis: np.ndarray
    y_axis: np.ndarray
    V_eV: np.ndarray
    bounds_nm: tuple[float, float, float, float]
    origin_shift_nm: tuple[float, float]
    min_position_nm: tuple[float, float]
    min_value_eV: float
    max_value_eV: float
    start_centres_nm: list[tuple[float, float]]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def resolve_project_path(path_like: str | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def get_P(T_K: float, p_max: int) -> int:
    """
    Number of imaginary-time slices, capped at p_max to prevent OOM.

    This keeps the old heuristic P = int(400/T). When capped, low-temperature
    results need explicit P-convergence checks before being trusted.
    """
    if T_K <= 0:
        raise ValueError(f"Temperature must be positive, got T_K={T_K}")
    uncapped = int(400 / T_K)
    return max(32, min(p_max, uncapped))


def is_P_capped(T_K: float, p_max: int) -> bool:
    return int(400 / T_K) > p_max


def format_temperature_label(T_K: float) -> str:
    return f"T_{T_K:.1f}K".replace(".", "p")


def format_ex_label(Ex: float) -> str:
    return f"Ex_{Ex:+.4f}".replace(".", "p")


def unique_rng_seed(T_K: float, Ex: float, seed_index: int, base_seed: int) -> int:
    """Deterministic seed mixer preserving reproducibility."""
    t_int = int(round(T_K * 1000.0))
    ex_int = int(round((Ex + 0.01) * 1_000_000.0))
    mixed = (
        int(base_seed)
        + 1_000_003 * int(seed_index)
        + 9_176 * t_int
        + 1_315_423_911 * ex_int
    )
    return int(mixed % (2**32 - 1))


def save_npz(path: Path, compressed: bool, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def make_envelope(v0_eV: float, radius_nm: float):
    k_env = 2.0 * float(v0_eV) / float(radius_nm) ** 2
    return HarmonicEnvelopePotential(k_env_eV_per_nm2=k_env)


# ---------------------------------------------------------------------------
# Grid loading / coordinate handling
# ---------------------------------------------------------------------------

def load_grid_potential_for_simulation(
    potential_npz: str,
    periodic: bool,
    subtract_minimum: bool,
    scale: float,
    coordinate_origin: CoordinateOrigin,
    barrier_eV: float,
) -> GridPotential2D:
    """
    Load V_grid.npz and optionally recenter coordinates around the grid centre.

    coordinate_origin='center' is usually better for Ex sweeps because the
    effective field term -Ex*x should not depend on whether the input grid used
    axes [0,L] or [-L/2,L/2].
    """
    path = resolve_project_path(potential_npz)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Potential grid file not found: {path}")

    gp = GridPotential2D.from_npz(
        path,
        periodic=periodic,
        subtract_minimum=subtract_minimum,
        scale=scale,
        barrier_eV=barrier_eV,
    )

    if coordinate_origin == "native":
        return gp

    if coordinate_origin != "center":
        raise ValueError(f"Unknown coordinate_origin={coordinate_origin!r}")

    x = gp.x_nm_axis
    y = gp.y_nm_axis
    V = gp.V_grid_eV
    x_mid = 0.5 * (float(x[0]) + float(x[-1]))
    y_mid = 0.5 * (float(y[0]) + float(y[-1]))

    return GridPotential2D(
        x_nm=x - x_mid,
        y_nm=y - y_mid,
        V_eV=V,
        periodic=periodic,
        subtract_minimum=False,  # already applied above if requested
        scale=1.0,              # already applied above
        barrier_eV=barrier_eV,
    )


def find_grid_start_centres(
    gp: GridPotential2D,
    n_centres: int,
    min_separation_nm: float,
    periodic: bool,
) -> list[tuple[float, float]]:
    """
    Find low-energy local minima on the loaded grid and greedily de-duplicate
    them by distance. Falls back to the global minimum if local-min detection
    is too restrictive.
    """
    x = gp.x_nm_axis
    y = gp.y_nm_axis
    V = gp.V_grid_eV

    # 8-neighbour local-minimum mask. Use np.roll for periodic maps; for
    # nonperiodic maps, ignore boundary points to avoid artificial edge minima.
    local = np.ones_like(V, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            local &= V <= np.roll(np.roll(V, di, axis=0), dj, axis=1)

    if not periodic and V.shape[0] > 2 and V.shape[1] > 2:
        local[0, :] = False
        local[-1, :] = False
        local[:, 0] = False
        local[:, -1] = False

    candidates = np.argwhere(local)
    if candidates.size == 0:
        candidates = np.array([np.unravel_index(int(np.argmin(V)), V.shape)])

    order = np.argsort(V[candidates[:, 0], candidates[:, 1]])
    candidates = candidates[order]

    selected: list[tuple[float, float]] = []
    for i, j in candidates:
        p = np.array([x[int(i)], y[int(j)]], dtype=float)
        keep = True
        for q in selected:
            if np.linalg.norm(p - np.array(q)) < float(min_separation_nm):
                keep = False
                break
        if keep:
            selected.append((float(p[0]), float(p[1])))
        if len(selected) >= int(n_centres):
            break

    if not selected:
        i, j = np.unravel_index(int(np.argmin(V)), V.shape)
        selected.append((float(x[i]), float(y[j])))

    return selected


def inspect_grid(
    potential_npz: str,
    periodic: bool,
    subtract_minimum: bool,
    scale: float,
    coordinate_origin: CoordinateOrigin,
    barrier_eV: float,
    n_start_centres: int,
    start_min_separation_nm: float,
) -> GridInfo:
    gp = load_grid_potential_for_simulation(
        potential_npz=potential_npz,
        periodic=periodic,
        subtract_minimum=subtract_minimum,
        scale=scale,
        coordinate_origin=coordinate_origin,
        barrier_eV=barrier_eV,
    )
    x = gp.x_nm_axis
    y = gp.y_nm_axis
    V = gp.V_grid_eV
    bounds = gp.bounds_nm
    i_min, j_min = np.unravel_index(int(np.argmin(V)), V.shape)
    start_centres = find_grid_start_centres(
        gp,
        n_centres=n_start_centres,
        min_separation_nm=start_min_separation_nm,
        periodic=periodic,
    )

    # Origin shift relative to native axes. If centered, this is approximately
    # the native mid-point. We report it for reproducibility; workers only need
    # the final centred/native axes.
    if coordinate_origin == "center":
        native = GridPotential2D.from_npz(
            resolve_project_path(potential_npz),
            periodic=periodic,
            subtract_minimum=subtract_minimum,
            scale=scale,
            barrier_eV=barrier_eV,
        )
        xn = native.x_nm_axis
        yn = native.y_nm_axis
        origin_shift = (0.5 * (float(xn[0]) + float(xn[-1])),
                        0.5 * (float(yn[0]) + float(yn[-1])))
    else:
        origin_shift = (0.0, 0.0)

    return GridInfo(
        x_axis=x,
        y_axis=y,
        V_eV=V,
        bounds_nm=bounds,
        origin_shift_nm=origin_shift,
        min_position_nm=(float(x[i_min]), float(y[j_min])),
        min_value_eV=float(np.min(V)),
        max_value_eV=float(np.max(V)),
        start_centres_nm=start_centres,
    )


# ---------------------------------------------------------------------------
# Potential construction
# ---------------------------------------------------------------------------

def build_atomistic_potential(Ex: float, config: dict):
    """Construct V_total = V_grid + optional envelope + field + optional Coulomb."""
    gp = load_grid_potential_for_simulation(
        potential_npz=config["potential_npz"],
        periodic=config["periodic"],
        subtract_minimum=config["subtract_minimum"],
        scale=config["grid_scale"],
        coordinate_origin=config["coordinate_origin"],
        barrier_eV=config["barrier_eV"],
    )

    terms = [gp]

    if config["add_envelope"]:
        terms.append(make_envelope(config["envelope_v0_eV"], config["envelope_radius_nm"]))

    # Same convention as the old script: Ex is an effective energy gradient in eV/nm.
    # ExternalFieldPotential.value(r) = -q_eff * r·E, with q_eff=1 by default.
    terms.append(ExternalFieldPotential(E=(float(Ex), 0.0)))

    if config["add_soft_coulomb"]:
        terms.append(
            SoftCoulombPotential(
                strength_eV_nm=config["coulomb_strength_eV_nm"],
                softening_nm=config["coulomb_softening_nm"],
            )
        )

    return CompositePotential(terms)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def choose_initial_center(
    Ex: float,
    seed_index: int,
    rng_seed: int,
    start_mode: StartMode,
    start_centres_nm: Sequence[tuple[float, float]],
    grid_bounds_nm: tuple[float, float, float, float],
    moire_period_nm: float,
) -> tuple[float, float]:
    """Choose the initial centroid position."""
    centres = list(start_centres_nm)
    if not centres:
        centres = [(0.0, 0.0)]

    if start_mode == "grid_cycle":
        return tuple(map(float, centres[seed_index % len(centres)]))

    if start_mode == "grid_min":
        return tuple(map(float, centres[0]))

    if start_mode == "random_grid":
        rng = np.random.default_rng(rng_seed)
        return tuple(map(float, centres[int(rng.integers(0, len(centres)))]))

    if start_mode == "field":
        if Ex > 0:
            return tuple(map(float, max(centres, key=lambda p: p[0])))
        if Ex < 0:
            return tuple(map(float, min(centres, key=lambda p: p[0])))
        return tuple(map(float, centres[seed_index % len(centres)]))

    if start_mode == "random_box":
        x0, x1, y0, y1 = grid_bounds_nm
        rng = np.random.default_rng(rng_seed)
        return float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))

    if start_mode == "origin":
        return 0.0, 0.0

    if start_mode == "hex":
        r_theory = float(moire_period_nm) / np.sqrt(3.0)
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
        angle = float(angles[seed_index % 6])
        return float(r_theory * np.cos(angle)), float(r_theory * np.sin(angle))

    raise ValueError(f"Unknown start_mode={start_mode!r}")


# ---------------------------------------------------------------------------
# Observables / histograms / free energy
# ---------------------------------------------------------------------------

def compute_histogram_free_energy(
    density_H: np.ndarray,
    T_K: float,
    min_counts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return probability_H, free_energy, free_energy_masked."""
    counts = density_H.astype(np.float64, copy=False)
    total = float(counts.sum())

    probability_H = np.zeros_like(counts)
    free_energy = np.zeros_like(counts)
    free_energy_masked = np.full_like(counts, np.nan)

    if total <= 0.0:
        return probability_H, free_energy, free_energy_masked

    probability_H = counts / total

    P_safe = np.maximum(probability_H, 1e-300)
    free_energy = -KB_EV_PER_K * T_K * np.log(P_safe)
    free_energy -= np.nanmin(free_energy)

    good = counts >= int(min_counts)
    if np.any(good):
        free_energy_masked[good] = -KB_EV_PER_K * T_K * np.log(probability_H[good])
        free_energy_masked -= np.nanmin(free_energy_masked)

    return probability_H, free_energy, free_energy_masked


def make_histogram_edges(config: dict) -> tuple[np.ndarray, np.ndarray]:
    if config["hist_range_nm"] is not None:
        r = float(config["hist_range_nm"])
        x0, x1, y0, y1 = -r, r, -r, r
    else:
        x0, x1, y0, y1 = config["hist_bounds_nm"]
        margin = float(config["hist_margin_nm"])
        x0, x1, y0, y1 = x0 - margin, x1 + margin, y0 - margin, y1 + margin

    n_bins = int(config["hist_bins"])
    return np.linspace(x0, x1, n_bins + 1), np.linspace(y0, y1, n_bins + 1)


def compute_energy_observables(potential, samples: np.ndarray) -> dict:
    """Useful additions beyond the old script: mean V over beads and centroids."""
    cents = centroids(samples)
    bead_points = samples.reshape(-1, 2)
    V_beads = potential.value(bead_points)
    V_centroids = potential.value(cents)
    return {
        "mean_V_bead_eV": float(np.mean(V_beads)),
        "std_V_bead_eV": float(np.std(V_beads)),
        "mean_V_centroid_eV": float(np.mean(V_centroids)),
        "std_V_centroid_eV": float(np.std(V_centroids)),
    }


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def single_run(
    Ex: float,
    seed_index: int,
    n_steps: int,
    burn_in: int,
    sample_every: int,
    T_K: float,
    p_max: int,
    mass_m0: float,
    local_step_multiplier: float,
    global_step_nm: float,
    start_mode: StartMode,
    compressed_npz: bool,
    save_samples: bool,
    output_tag: str,
    config: dict,
) -> tuple[dict, dict, Path]:

    P = get_P(T_K, p_max)
    potential = build_atomistic_potential(Ex, config)
    action = RingPolymerAction(mass_m0, T_K, P, potential)

    beta = 1.0 / (T_K * KB_EV_PER_K)
    tau = beta / P
    lambda_x = HBAR2_OVER_2M0 / mass_m0
    sigma_bead = float((2.0 * lambda_x * tau) ** 0.5)

    rng_seed = unique_rng_seed(T_K, Ex, seed_index, config["base_seed"])
    local_step = float(local_step_multiplier) * sigma_bead

    sampler = PIMCSamplerJIT(
        action,
        rng_seed=rng_seed,
        local_step_nm=local_step,
        global_step_nm=global_step_nm,
        global_move_probability=config["global_move_probability"],
        grid_size=config["jit_grid_size"],
        grid_range_nm=config["jit_grid_range_nm"],
    )

    center = choose_initial_center(
        Ex=Ex,
        seed_index=seed_index,
        rng_seed=rng_seed,
        start_mode=start_mode,
        start_centres_nm=config["start_centres_nm"],
        grid_bounds_nm=config["hist_bounds_nm"],
        moire_period_nm=config["moire_period_nm"],
    )

    res = sampler.run(
        n_steps=n_steps,
        burn_in=burn_in,
        sample_every=sample_every,
        center=center,
    )

    samples = res["samples"]
    cents = centroids(samples)

    mean_x = float(cents[:, 0].mean())
    mean_y = float(cents[:, 1].mean())
    std_x = float(cents[:, 0].std())
    std_y = float(cents[:, 1].std())
    r2_b = float(r2_mean_pimc(samples))
    r2_c = float(r2_mean_centroid(samples))
    r2_spread = float(r2_spread_pimc(samples))

    x_edges, y_edges = make_histogram_edges(config)
    density_H, xedges, yedges = np.histogram2d(
        cents[:, 0], cents[:, 1], bins=[x_edges, y_edges]
    )
    probability_H, free_energy, free_energy_masked = compute_histogram_free_energy(
        density_H=density_H,
        T_K=T_K,
        min_counts=config["min_counts_for_free_energy"],
    )

    energy_obs = compute_energy_observables(potential, samples)

    out_dir = ROOT / "results" / output_tag / format_temperature_label(T_K) / format_ex_label(Ex)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_payload = dict(
        mean_x=mean_x,
        mean_y=mean_y,
        std_x=std_x,
        std_y=std_y,
        r2_bead=r2_b,
        r2_centroid=r2_c,
        r2_spread=r2_spread,
        density_H=density_H,
        probability_H=probability_H,
        density_xedges=xedges,
        density_yedges=yedges,
        free_energy=free_energy,
        free_energy_masked=free_energy_masked,
        centroids=cents,
        T_K=T_K,
        Ex=Ex,
        n_beads=P,
        beta_eV=beta,
        tau_eV_inv=tau,
        mass_m0=mass_m0,
        rng_seed=rng_seed,
        start_center=np.asarray(center, dtype=np.float64),
        local_step_nm=local_step,
        global_step_nm=global_step_nm,
        acceptance_local=float(res.get("acceptance_local", np.nan)),
        acceptance_global=float(res.get("acceptance_global", np.nan)),
        p_capped=int(is_P_capped(T_K, p_max)),
        potential_npz=str(resolve_project_path(config["potential_npz"])),
        grid_scale=config["grid_scale"],
        grid_periodic=int(config["periodic"]),
        grid_subtract_minimum=int(config["subtract_minimum"]),
        coordinate_origin=config["coordinate_origin"],
        origin_shift_nm=np.asarray(config["origin_shift_nm"], dtype=np.float64),
        add_envelope=int(config["add_envelope"]),
        envelope_v0_eV=config["envelope_v0_eV"],
        envelope_radius_nm=config["envelope_radius_nm"],
        add_soft_coulomb=int(config["add_soft_coulomb"]),
        coulomb_strength_eV_nm=config["coulomb_strength_eV_nm"],
        coulomb_softening_nm=config["coulomb_softening_nm"],
        jit_grid_size=config["jit_grid_size"],
        jit_grid_range_nm=config["jit_grid_range_nm"],
        min_counts_for_free_energy=config["min_counts_for_free_energy"],
        mean_V_bead_eV=energy_obs["mean_V_bead_eV"],
        std_V_bead_eV=energy_obs["std_V_bead_eV"],
        mean_V_centroid_eV=energy_obs["mean_V_centroid_eV"],
        std_V_centroid_eV=energy_obs["std_V_centroid_eV"],
    )

    if save_samples:
        npz_payload["samples"] = samples

    save_npz(out_dir / f"results_seed_{seed_index}.npz", compressed_npz, **npz_payload)

    scalar_metrics = {
        "T_K": T_K,
        "Ex": Ex,
        "seed": seed_index,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "std_x": std_x,
        "std_y": std_y,
        "r2_bead": r2_b,
        "r2_centroid": r2_c,
        "r2_spread": r2_spread,
        "r_rms_bead": float(np.sqrt(r2_b)),
        "r_rms_centroid": float(np.sqrt(r2_c)),
        "r_rms_spread": float(np.sqrt(r2_spread)),
        "mean_V_bead_eV": energy_obs["mean_V_bead_eV"],
        "std_V_bead_eV": energy_obs["std_V_bead_eV"],
        "mean_V_centroid_eV": energy_obs["mean_V_centroid_eV"],
        "std_V_centroid_eV": energy_obs["std_V_centroid_eV"],
        "acceptance_local": float(res.get("acceptance_local", np.nan)),
        "acceptance_global": float(res.get("acceptance_global", np.nan)),
        "n_beads": P,
        "beta_eV": beta,
        "tau_eV_inv": tau,
        "mass_m0": mass_m0,
        "rng_seed": rng_seed,
        "start_x": center[0],
        "start_y": center[1],
        "local_step_nm": local_step,
        "global_step_nm": global_step_nm,
        "p_capped": int(is_P_capped(T_K, p_max)),
        "potential_npz": str(resolve_project_path(config["potential_npz"])),
        "coordinate_origin": config["coordinate_origin"],
        "grid_periodic": int(config["periodic"]),
        "grid_scale": config["grid_scale"],
        "add_envelope": int(config["add_envelope"]),
        "add_soft_coulomb": int(config["add_soft_coulomb"]),
    }

    return scalar_metrics, npz_payload, out_dir


# ---------------------------------------------------------------------------
# Task wrapper: all seeds for one (T, Ex) in one worker
# ---------------------------------------------------------------------------

def _run_temp_ex_task(args):
    (
        Ex,
        total_seeds,
        n_steps,
        burn_in,
        sample_every,
        T_K,
        p_max,
        mass_m0,
        local_step_multiplier,
        global_step_nm,
        start_mode,
        compressed_npz,
        save_samples,
        output_tag,
        config,
    ) = args

    try:
        sub_results = []
        agg_density_H = None
        agg_centroids = []
        mean_x_list, mean_y_list = [], []
        std_x_list, std_y_list = [], []
        r2_b_list, r2_c_list, r2_s_list = [], [], []
        acc_local_list, acc_global_list = [], []
        mean_V_bead_list, mean_V_centroid_list = [], []
        final_out_dir = None
        shared_meta = None

        for seed_index in range(total_seeds):
            scalar_metrics, npz_payload, out_dir = single_run(
                Ex=Ex,
                seed_index=seed_index,
                n_steps=n_steps,
                burn_in=burn_in,
                sample_every=sample_every,
                T_K=T_K,
                p_max=p_max,
                mass_m0=mass_m0,
                local_step_multiplier=local_step_multiplier,
                global_step_nm=global_step_nm,
                start_mode=start_mode,
                compressed_npz=compressed_npz,
                save_samples=save_samples,
                output_tag=output_tag,
                config=config,
            )
            sub_results.append(scalar_metrics)

            if agg_density_H is None:
                agg_density_H = npz_payload["density_H"].copy()
                shared_meta = npz_payload
            else:
                agg_density_H += npz_payload["density_H"]

            agg_centroids.append(npz_payload["centroids"])
            mean_x_list.append(npz_payload["mean_x"])
            mean_y_list.append(npz_payload["mean_y"])
            std_x_list.append(npz_payload["std_x"])
            std_y_list.append(npz_payload["std_y"])
            r2_b_list.append(npz_payload["r2_bead"])
            r2_c_list.append(npz_payload["r2_centroid"])
            r2_s_list.append(npz_payload["r2_spread"])
            acc_local_list.append(npz_payload["acceptance_local"])
            acc_global_list.append(npz_payload["acceptance_global"])
            mean_V_bead_list.append(npz_payload["mean_V_bead_eV"])
            mean_V_centroid_list.append(npz_payload["mean_V_centroid_eV"])
            final_out_dir = out_dir

        if final_out_dir is not None and shared_meta is not None:
            probability_H, free_energy, free_energy_masked = compute_histogram_free_energy(
                density_H=agg_density_H,
                T_K=T_K,
                min_counts=config["min_counts_for_free_energy"],
            )
            ensemble_payload = dict(
                mean_x=float(np.mean(mean_x_list)),
                mean_y=float(np.mean(mean_y_list)),
                std_x=float(np.mean(std_x_list)),
                std_y=float(np.mean(std_y_list)),
                r2_bead=float(np.mean(r2_b_list)),
                r2_centroid=float(np.mean(r2_c_list)),
                r2_spread=float(np.mean(r2_s_list)),
                r_rms_bead=float(np.sqrt(np.mean(r2_b_list))),
                r_rms_centroid=float(np.sqrt(np.mean(r2_c_list))),
                r_rms_spread=float(np.sqrt(np.mean(r2_s_list))),
                mean_V_bead_eV=float(np.mean(mean_V_bead_list)),
                mean_V_centroid_eV=float(np.mean(mean_V_centroid_list)),
                acceptance_local=float(np.nanmean(acc_local_list)),
                acceptance_global=float(np.nanmean(acc_global_list)),
                density_H=agg_density_H,
                probability_H=probability_H,
                density_xedges=shared_meta["density_xedges"],
                density_yedges=shared_meta["density_yedges"],
                free_energy=free_energy,
                free_energy_masked=free_energy_masked,
                centroids=np.concatenate(agg_centroids, axis=0),
                T_K=T_K,
                Ex=Ex,
                n_beads=shared_meta["n_beads"],
                beta_eV=shared_meta["beta_eV"],
                tau_eV_inv=shared_meta["tau_eV_inv"],
                mass_m0=shared_meta["mass_m0"],
                potential_npz=shared_meta["potential_npz"],
                grid_scale=shared_meta["grid_scale"],
                grid_periodic=shared_meta["grid_periodic"],
                grid_subtract_minimum=shared_meta["grid_subtract_minimum"],
                coordinate_origin=shared_meta["coordinate_origin"],
                origin_shift_nm=shared_meta["origin_shift_nm"],
                add_envelope=shared_meta["add_envelope"],
                envelope_v0_eV=shared_meta["envelope_v0_eV"],
                envelope_radius_nm=shared_meta["envelope_radius_nm"],
                add_soft_coulomb=shared_meta["add_soft_coulomb"],
                coulomb_strength_eV_nm=shared_meta["coulomb_strength_eV_nm"],
                coulomb_softening_nm=shared_meta["coulomb_softening_nm"],
                jit_grid_size=shared_meta["jit_grid_size"],
                jit_grid_range_nm=shared_meta["jit_grid_range_nm"],
                min_counts_for_free_energy=shared_meta["min_counts_for_free_energy"],
            )
            save_npz(final_out_dir / "results.npz", compressed_npz, **ensemble_payload)

        return sub_results

    except Exception as exc:
        raise RuntimeError(f"Worker failed: Ex={Ex:+.4f} eV/nm, T={T_K} K") from exc


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_temp_field_sweep(
    ex_values: list[float],
    t_values: list[float],
    n_steps: int,
    burn_in: int,
    sample_every: int,
    seeds: int,
    workers: int,
    p_max: int,
    mass_m0: float,
    local_step_multiplier: float,
    global_step_nm: float,
    start_mode: StartMode,
    compressed_npz: bool,
    save_samples: bool,
    output_tag: str,
    config: dict,
) -> pd.DataFrame:

    task_args = [
        (
            float(ex),
            seeds,
            n_steps,
            burn_in,
            sample_every,
            float(temp),
            p_max,
            mass_m0,
            local_step_multiplier,
            global_step_nm,
            start_mode,
            compressed_npz,
            save_samples,
            output_tag,
            config,
        )
        for temp in t_values
        for ex in ex_values
    ]

    n_ex, n_T = len(ex_values), len(t_values)
    total = len(task_args)
    print(f"  Total tasks: {total} ({n_ex} field values × {n_T} temperatures, {seeds} seeds per task)")

    results = []
    completed = 0
    t0 = time.perf_counter()

    if workers == 1:
        for args in task_args:
            results.extend(_run_temp_ex_task(args))
            completed += 1
            if completed % max(1, total // 10) == 0 or completed == total:
                elapsed = time.perf_counter() - t0
                eta = elapsed / completed * (total - completed)
                print(f"  [{completed}/{total}] elapsed {elapsed:.0f}s ETA {eta:.0f}s")
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for seed_outputs in pool.map(_run_temp_ex_task, task_args):
                results.extend(seed_outputs)
                completed += 1
                if completed % max(1, total // 10) == 0 or completed == total:
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / completed * (total - completed)
                    print(f"  [{completed}/{total}] elapsed {elapsed:.0f}s ETA {eta:.0f}s")

    df = pd.DataFrame(results)
    first_cols = [
        "T_K", "Ex", "seed",
        "mean_x", "mean_y", "std_x", "std_y",
        "r2_bead", "r2_centroid", "r2_spread",
        "r_rms_bead", "r_rms_centroid", "r_rms_spread",
        "mean_V_bead_eV", "mean_V_centroid_eV",
        "acceptance_local", "acceptance_global",
        "n_beads", "beta_eV", "tau_eV_inv", "mass_m0", "rng_seed",
        "start_x", "start_y",
        "local_step_nm", "global_step_nm", "p_capped",
        "potential_npz", "coordinate_origin", "grid_periodic", "grid_scale",
        "add_envelope", "add_soft_coulomb",
    ]
    return df[first_cols + [c for c in df.columns if c not in first_cols]]


# ---------------------------------------------------------------------------
# Validation / diagnostics
# ---------------------------------------------------------------------------

def validate_run_parameters(
    n_steps: int,
    burn_in: int,
    sample_every: int,
    workers: int,
    seeds: int,
    p_max: int,
    mass_m0: float,
    hist_bins: int,
) -> None:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    if burn_in >= n_steps:
        raise ValueError("burn_in must be < n_steps")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if p_max < 32:
        raise ValueError("p_max must be >= 32")
    if mass_m0 <= 0.0:
        raise ValueError("mass_m0 must be positive")
    if hist_bins < 10:
        raise ValueError("hist_bins must be >= 10")


def parse_ex_values(args) -> list[float]:
    if args.ex_values is not None:
        return [float(v) for v in args.ex_values]
    return [float(v) for v in np.linspace(args.ex_min, args.ex_max, args.ex_points)]


def auto_jit_grid_range(grid_bounds_nm: tuple[float, float, float, float], margin_factor: float = 1.03) -> float:
    x0, x1, y0, y1 = grid_bounds_nm
    return float(max(abs(x0), abs(x1), abs(y0), abs(y1)) * margin_factor)


def print_start_centres(centres: Sequence[tuple[float, float]], max_rows: int = 8) -> None:
    print(f"  Low-energy start centres found        : {len(centres)}")
    for idx, (x, y) in enumerate(list(centres)[:max_rows]):
        print(f"    #{idx:02d}: x={x:+9.4f} nm, y={y:+9.4f} nm")
    if len(centres) > max_rows:
        print(f"    ... {len(centres) - max_rows} more")


def print_grid_diagnostics(grid_info: GridInfo, config: dict, ex_values: list[float]) -> None:
    x0, x1, y0, y1 = grid_info.bounds_nm
    V = grid_info.V_eV

    print("\n  ── Atomistic grid diagnostics ─────────────────────────────")
    print(f"  Potential file                      : {resolve_project_path(config['potential_npz'])}")
    print(f"  Coordinate origin                   : {config['coordinate_origin']}")
    print(f"  Origin shift [nm]                   : ({grid_info.origin_shift_nm[0]:+.4f}, {grid_info.origin_shift_nm[1]:+.4f})")
    print(f"  Grid shape                          : {V.shape[0]} × {V.shape[1]}")
    print(f"  Grid bounds [nm]                    : x=[{x0:+.4f}, {x1:+.4f}], y=[{y0:+.4f}, {y1:+.4f}]")
    print(f"  Grid periodic                       : {config['periodic']}")
    print(f"  V_grid range after shift/scale      : {grid_info.min_value_eV*1000:+.4f} … {grid_info.max_value_eV*1000:+.4f} meV")
    print(f"  V_grid depth                        : {(grid_info.max_value_eV-grid_info.min_value_eV)*1000:.4f} meV")
    print(f"  Global minimum [nm]                 : x={grid_info.min_position_nm[0]:+.4f}, y={grid_info.min_position_nm[1]:+.4f}")
    print_start_centres(grid_info.start_centres_nm)

    if ex_values:
        ex_abs = max(abs(v) for v in ex_values)
        field_drop_x = ex_abs * (x1 - x0)
        print(f"  Max |Ex| in sweep                    : {ex_abs:.6g} eV/nm")
        print(f"  Field energy drop across grid x      : {field_drop_x*1000:.4f} meV")

    print(f"  Add weak harmonic envelope           : {config['add_envelope']}")
    if config["add_envelope"]:
        print(f"    envelope V0 at R                   : {config['envelope_v0_eV']*1000:.3f} meV at {config['envelope_radius_nm']:.3f} nm")
    print(f"  Add soft-Coulomb COM correction      : {config['add_soft_coulomb']}")
    if config["add_soft_coulomb"]:
        v0 = -config["coulomb_strength_eV_nm"] / config["coulomb_softening_nm"]
        print(f"    V_C(0)                             : {v0*1000:+.3f} meV")
        print("    WARNING: this is an effective COM term, not a two-particle e-h simulation.")
    print(f"  JIT square range [nm]                : [-{config['jit_grid_range_nm']:.3f}, +{config['jit_grid_range_nm']:.3f}]")
    print("  ────────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Atomistic-grid V_grid.npz PI-QMC temperature + effective-field sweep (v1.0)"
    )

    parser.add_argument(
        "--potential_npz",
        type=str,
        default="results/wse2_mose2/V_grid.npz",
        help="Path to atomistic/grid potential NPZ. Default: results/wse2_mose2/V_grid.npz",
    )
    parser.add_argument("--output_tag", type=str, default="atomistic_grid_pimc_v1_0")

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--burn_in", type=int)
    parser.add_argument("--sample_every", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--temps", type=float, nargs="+", default=None)
    parser.add_argument("--quick", action="store_true", help="Short run: 6 000 steps, burn-in 1 500.")

    parser.add_argument("--mass_m0", type=float, default=MASS_M0_DEFAULT)
    parser.add_argument("--p_max", type=int, default=128)
    parser.add_argument("--local_step_multiplier", type=float, default=LOCAL_STEP_MULTIPLIER_DEFAULT)
    parser.add_argument("--global_step_nm", type=float, default=GLOBAL_STEP_NM_DEFAULT)
    parser.add_argument("--global_move_probability", type=float, default=0.20)

    parser.add_argument("--ex_values", type=float, nargs="+", default=None,
                        help="Explicit list of effective field/tilt values in eV/nm.")
    parser.add_argument("--ex_min", type=float, default=float(EX_VALUES_DEFAULT[0]))
    parser.add_argument("--ex_max", type=float, default=float(EX_VALUES_DEFAULT[-1]))
    parser.add_argument("--ex_points", type=int, default=len(EX_VALUES_DEFAULT))

    parser.add_argument(
        "--start_mode",
        choices=["grid_cycle", "grid_min", "field", "random_grid", "random_box", "origin", "hex"],
        default="grid_cycle",
        help=(
            "Initialisation mode. grid_cycle cycles seeds over low-energy minima found in V_grid; "
            "this is the recommended atomistic default."
        ),
    )
    parser.add_argument("--n_start_centres", type=int, default=24)
    parser.add_argument("--start_min_separation_nm", type=float, default=2.0)
    parser.add_argument("--moire_period_nm", type=float, default=20.0,
                        help="Only used by --start_mode hex as analytic fallback.")

    parser.add_argument("--coordinate_origin", choices=["center", "native"], default="center",
                        help="center shifts grid axes to the cell centre; recommended for Ex sweeps.")
    parser.add_argument("--grid_scale", type=float, default=1.0)
    parser.add_argument("--grid_nonperiodic", action="store_true",
                        help="Do not wrap coordinates periodically inside GridPotential2D.")
    parser.add_argument("--grid_keep_absolute_offset", action="store_true",
                        help="Do not subtract min(V_grid). Energy differences are unchanged, but offsets are preserved.")
    parser.add_argument("--barrier_eV", type=float, default=1.0e6,
                        help="Outside-grid barrier used when --grid_nonperiodic is set.")

    parser.add_argument("--no_envelope", action="store_true",
                        help="Disable weak harmonic envelope. Default is ON for field-sweep stability.")
    parser.add_argument("--envelope_v0_eV", type=float, default=V0_ENVELOPE_DEFAULT)
    parser.add_argument("--envelope_radius_nm", type=float, default=R_ENVELOPE_DEFAULT)

    parser.add_argument("--add_soft_coulomb", action="store_true",
                        help="Add old effective soft-Coulomb COM correction. OFF by default to avoid double-counting.")
    parser.add_argument("--coulomb_strength_eV_nm", type=float, default=COULOMB_STRENGTH_DEFAULT)
    parser.add_argument("--coulomb_softening_nm", type=float, default=COULOMB_SOFTENING_DEFAULT)

    parser.add_argument("--hist_bins", type=int, default=200)
    parser.add_argument("--hist_range_nm", type=float, default=None,
                        help="If set, histogram is [-R,R] × [-R,R]. Otherwise grid bounds are used.")
    parser.add_argument("--hist_margin_nm", type=float, default=0.0)
    parser.add_argument("--min_counts_for_free_energy", type=int, default=MIN_COUNTS_FOR_FREE_ENERGY_DEFAULT)

    parser.add_argument("--jit_grid_size", type=int, default=600)
    parser.add_argument("--jit_grid_range_nm", type=float, default=None,
                        help="Square range for PIMCSamplerJIT. Auto from centred/native grid bounds if omitted.")

    parser.add_argument("--base_seed", type=int, default=BASE_SEED_DEFAULT)
    parser.add_argument("--compressed_npz", action="store_true")
    parser.add_argument("--save_samples", action="store_true",
                        help="Also store full bead samples. Large files; old script stored centroids, not full samples.")

    return parser.parse_args()


def main():
    args = parse_args()

    n_steps = args.n_steps or 30_000
    burn_in = args.burn_in if args.burn_in is not None else 5_000
    if args.quick:
        n_steps, burn_in = 6_000, 1_500
        print("--> QUICK mode: 6 000 steps, burn-in 1 500.")

    sample_every = args.sample_every
    seeds = args.seeds
    t_values = list(args.temps or T_VALUES_DEFAULT)
    ex_values = parse_ex_values(args)

    validate_run_parameters(
        n_steps=n_steps,
        burn_in=burn_in,
        sample_every=sample_every,
        workers=args.workers,
        seeds=seeds,
        p_max=args.p_max,
        mass_m0=args.mass_m0,
        hist_bins=args.hist_bins,
    )

    grid_info = inspect_grid(
        potential_npz=args.potential_npz,
        periodic=not args.grid_nonperiodic,
        subtract_minimum=not args.grid_keep_absolute_offset,
        scale=args.grid_scale,
        coordinate_origin=args.coordinate_origin,
        barrier_eV=args.barrier_eV,
        n_start_centres=args.n_start_centres,
        start_min_separation_nm=args.start_min_separation_nm,
    )

    jit_grid_range = args.jit_grid_range_nm
    if jit_grid_range is None:
        jit_grid_range = auto_jit_grid_range(grid_info.bounds_nm)

    config = dict(
        potential_npz=args.potential_npz,
        periodic=not args.grid_nonperiodic,
        subtract_minimum=not args.grid_keep_absolute_offset,
        grid_scale=args.grid_scale,
        coordinate_origin=args.coordinate_origin,
        barrier_eV=args.barrier_eV,
        origin_shift_nm=grid_info.origin_shift_nm,
        start_centres_nm=grid_info.start_centres_nm,
        hist_bounds_nm=grid_info.bounds_nm,
        hist_bins=args.hist_bins,
        hist_range_nm=args.hist_range_nm,
        hist_margin_nm=args.hist_margin_nm,
        min_counts_for_free_energy=args.min_counts_for_free_energy,
        add_envelope=not args.no_envelope,
        envelope_v0_eV=args.envelope_v0_eV,
        envelope_radius_nm=args.envelope_radius_nm,
        add_soft_coulomb=args.add_soft_coulomb,
        coulomb_strength_eV_nm=args.coulomb_strength_eV_nm,
        coulomb_softening_nm=args.coulomb_softening_nm,
        jit_grid_size=args.jit_grid_size,
        jit_grid_range_nm=float(jit_grid_range),
        global_move_probability=args.global_move_probability,
        moire_period_nm=args.moire_period_nm,
        base_seed=args.base_seed,
    )

    # Warn about P cap
    T_p_boundary = 400.0 / args.p_max
    capped_temps = [T for T in t_values if T < T_p_boundary]
    if capped_temps:
        print(
            f"\n  ⚠ WARNING: p_max={args.p_max} caps bead count for T < {T_p_boundary:.2f} K.\n"
            f"  Affected temperatures: {capped_temps} K\n"
            "  Treat these low-T results as under-converged until P-convergence is checked.\n"
        )

    T_min = min(t_values)
    P_at_T_min = get_P(T_min, args.p_max)
    n_samples = 1 + (n_steps - burn_in - 1) // sample_every
    gb_per_run = n_samples * P_at_T_min * 2 * 8 / 1e9

    print("=" * 70)
    print("  Atomistic-grid PI-QMC — Temperature + effective-field sweep (v1.0)")
    print("=" * 70)
    print(f"  Project root                         : {ROOT}")
    print(f"  Temperatures [K]                     : {t_values}")
    print(f"  Effective field Ex                   : {len(ex_values)} points [{ex_values[0]:+.5f} … {ex_values[-1]:+.5f}] eV/nm")
    print(f"  RNG seeds / point                    : {seeds}")
    print(f"  PIMC steps                           : {n_steps:,} (burn-in {burn_in:,}, sample_every {sample_every})")
    print(f"  Bead count range                     : {get_P(max(t_values), args.p_max)}–{P_at_T_min} (p_max={args.p_max})")
    print(f"  Mass                                 : {args.mass_m0:.4f} m0")
    print(f"  Local step multiplier                : {args.local_step_multiplier} × sigma_bead")
    print(f"  Global step                          : {args.global_step_nm:.4f} nm")
    print(f"  Estimated sample array per seed      : {gb_per_run:.4f} GB")
    print(f"  Worker processes                     : {args.workers}")
    print(f"  Start mode                           : {args.start_mode}")
    print(f"  Output tag                           : {args.output_tag}")
    print(f"  Compressed NPZ                       : {args.compressed_npz}")
    print(f"  Save full bead samples               : {args.save_samples}")

    print_grid_diagnostics(grid_info, config, ex_values)

    df = run_temp_field_sweep(
        ex_values=ex_values,
        t_values=t_values,
        n_steps=n_steps,
        burn_in=burn_in,
        sample_every=sample_every,
        seeds=seeds,
        workers=args.workers,
        p_max=args.p_max,
        mass_m0=args.mass_m0,
        local_step_multiplier=args.local_step_multiplier,
        global_step_nm=args.global_step_nm,
        start_mode=args.start_mode,
        compressed_npz=args.compressed_npz,
        save_samples=args.save_samples,
        output_tag=args.output_tag,
        config=config,
    )

    out_dir = ROOT / "results" / args.output_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.output_tag}.csv"
    df.to_csv(csv_path, index=False)

    print("\n--- Ensemble-averaged centroid displacement <x> [nm] ---")
    print(df.groupby("T_K")["mean_x"].agg([("mean", "mean"), ("std", "std")]).to_string())

    print("\n--- Mean intrinsic quantum spread sqrt(<|r_bead-r_centroid|^2>) [nm] ---")
    print(df.groupby("T_K")["r_rms_spread"].agg([("mean", "mean"), ("std", "std")]).to_string())

    print("\n--- Acceptance diagnostics (target local roughly 0.40–0.55) ---")
    acc_cols = [c for c in ["acceptance_local", "acceptance_global"] if c in df]
    if acc_cols:
        print(df.groupby("T_K")[acc_cols].agg(["mean", "min", "max"]).to_string())

    capped_in_output = df[df["p_capped"] == 1]["T_K"].unique()
    if len(capped_in_output):
        print(f"\n  ⚠ P was capped at p_max={args.p_max} for T = {sorted(capped_in_output)} K")

    print("\nRun completed.")
    print(f"  CSV  : {csv_path}")
    print(f"  NPZs : {out_dir}")


if __name__ == "__main__":
    main()
