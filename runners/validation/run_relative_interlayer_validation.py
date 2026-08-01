#!/usr/bin/env python3
"""v1.7 direct-contact interlayer exciton PI-QMC validation.

The same finite-separation screened radial table is used by the independent
2D radial Schroedinger solver and by staging PI-QMC.  The current model keeps
only the relative in-plane coordinate rho and assumes a fixed vertical
separation D between the MoSe2 electron and WSe2 hole planes.

No moire, Ez, piezoelectric or separate one-body band-edge potentials are used.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "v1.7-direct-interlayer-staging"


def find_project_root() -> Path:
    env_root = os.environ.get("TMD_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not (root / "code" / "tmd_pimc").exists():
            raise RuntimeError(f"TMD_PROJECT_ROOT={root} lacks code/tmd_pimc")
        return root
    here = Path(__file__).resolve()
    for candidate in [Path.cwd().resolve(), *here.parents]:
        if (candidate / "code" / "tmd_pimc").exists():
            return candidate
    raise RuntimeError("Could not locate project root containing code/tmd_pimc")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "code"))

try:
    from tmd_pimc.action import RingPolymerAction
    from tmd_pimc.constants import HBAR2_OVER_2M0, KB_EV_PER_K
    from tmd_pimc.interlayer_screened_potential import (
        InterlayerScreenedWallPotential,
        build_interlayer_table,
        interlayer_table_diagnostics,
        load_interlayer_table_npz,
        save_interlayer_table_npz,
    )
    from tmd_pimc.observables import centroids, r2_mean_centroid, r2_spread_pimc
    from tmd_pimc.radial_solver import thermal_radial_reference_2d
    from tmd_pimc.sampler import PIMCSampler, PIMCSamplerJIT, PIMCSamplerStaging
except ImportError as exc:
    raise SystemExit(f"Could not import v1.7 modules from {ROOT}: {exc}") from exc


@dataclass(frozen=True)
class InterlayerTask:
    temperature_K: float
    n_beads: int
    seed_index: int
    rng_seed: int
    reduced_mass_m0: float
    interlayer_table_npz: str
    wall_radius_nm: float
    wall_height_eV: float
    wall_power: int
    n_steps: int
    burn_in: int
    sample_every: int
    local_step_multiplier: float
    global_step_nm: float
    global_move_probability: float
    sampler_kind: str
    staging_segment_lengths: tuple[int, ...]
    staging_moves_per_step: int
    staging_perform_local_sweep: bool
    jit_grid_size: int
    jit_grid_range_nm: float
    radial_edges_nm: tuple[float, ...]
    initial_radius_nm: float
    save_samples: bool
    output_dir: str


def resolve_project_path(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def make_rng_seed(base_seed: int, T: float, P: int, seed_index: int) -> int:
    mixed = (
        int(base_seed)
        + 1_000_003 * int(seed_index)
        + 97_409 * int(P)
        + 9_176 * int(round(float(T) * 1000.0))
    )
    return int(mixed % (2**32 - 1))


def finite_mean(series: pd.Series) -> float:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def finite_std(series: pd.Series) -> float:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def load_config(path_text: str) -> tuple[dict[str, Any], Path]:
    path = resolve_project_path(path_text)
    assert path is not None
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "reduced_mass_m0",
        "interlayer_separation_nm",
        "single_layer_screening_length_nm",
        "environment_model",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config is missing keys: {missing}")
    return config, path


def build_total_potential(
    table_path: str | Path,
    wall_radius_nm: float,
    wall_height_eV: float,
    wall_power: int,
) -> InterlayerScreenedWallPotential:
    return InterlayerScreenedWallPotential(
        interlayer=load_interlayer_table_npz(table_path),
        wall_radius_nm=float(wall_radius_nm),
        wall_height_eV=float(wall_height_eV),
        wall_power=int(wall_power),
    )


def run_single_task(task: InterlayerTask) -> dict[str, Any]:
    potential = build_total_potential(
        task.interlayer_table_npz,
        task.wall_radius_nm,
        task.wall_height_eV,
        task.wall_power,
    )
    T = float(task.temperature_K)
    P = int(task.n_beads)
    beta = 1.0 / (KB_EV_PER_K * T)
    tau = beta / P
    lam = HBAR2_OVER_2M0 / float(task.reduced_mass_m0)
    sigma_link = float(np.sqrt(2.0 * lam * tau))
    local_step = float(task.local_step_multiplier) * sigma_link

    action = RingPolymerAction(
        mass_m0=float(task.reduced_mass_m0),
        temperature_K=T,
        n_beads=P,
        potential=potential,
    )
    if task.sampler_kind == "python":
        sampler = PIMCSampler(
            action=action,
            local_step_nm=local_step,
            global_step_nm=float(task.global_step_nm),
            global_move_probability=float(task.global_move_probability),
            rng_seed=int(task.rng_seed),
        )
    elif task.sampler_kind == "staging":
        sampler = PIMCSamplerStaging(
            action=action,
            local_step_nm=local_step,
            global_step_nm=float(task.global_step_nm),
            global_move_probability=float(task.global_move_probability),
            rng_seed=int(task.rng_seed),
            staging_segment_lengths=task.staging_segment_lengths,
            staging_moves_per_step=int(task.staging_moves_per_step),
            perform_local_sweep=bool(task.staging_perform_local_sweep),
        )
    elif task.sampler_kind == "jit":
        sampler = PIMCSamplerJIT(
            action=action,
            local_step_nm=local_step,
            global_step_nm=float(task.global_step_nm),
            global_move_probability=float(task.global_move_probability),
            rng_seed=int(task.rng_seed),
            grid_size=int(task.jit_grid_size),
            grid_range_nm=float(task.jit_grid_range_nm),
        )
    else:
        raise ValueError(f"Unknown sampler_kind={task.sampler_kind!r}")

    result = sampler.run(
        n_steps=int(task.n_steps),
        burn_in=int(task.burn_in),
        sample_every=int(task.sample_every),
        center=(float(task.initial_radius_nm), 0.0),
    )
    samples = np.asarray(result["samples"], dtype=float)
    if samples.ndim != 3 or samples.shape[1:] != (P, 2):
        raise RuntimeError(f"Unexpected samples shape: {samples.shape}")
    if samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
        raise RuntimeError("Invalid or empty production samples")

    staging_lengths = np.asarray(
        result.get("staging_segment_lengths", []), dtype=np.int64
    )
    staging_attempts = np.asarray(
        result.get("staging_length_attempts", []), dtype=np.int64
    )
    staging_accepted = np.asarray(
        result.get("staging_length_accepted", []), dtype=np.int64
    )
    staging_rates = np.divide(
        staging_accepted,
        staging_attempts,
        out=np.full(staging_lengths.shape, np.nan, dtype=float),
        where=staging_attempts > 0,
    )

    bead_points = samples.reshape(-1, 2)
    radii = np.sqrt(np.einsum("ij,ij->i", bead_points, bead_points))
    r2 = radii**2
    centroid_r2 = float(r2_mean_centroid(samples))
    spread_r2 = float(r2_spread_pimc(samples))
    V_interlayer = np.asarray(potential.central_value(bead_points), dtype=float)
    V_wall = np.asarray(potential.wall_value(bead_points), dtype=float)
    V_total = V_interlayer + V_wall
    edges = np.asarray(task.radial_edges_nm, dtype=float)
    counts, _ = np.histogram(radii, bins=edges)
    lost_fraction = float(np.mean(radii >= edges[-1]))

    if task.save_samples:
        seed_dir = (
            Path(task.output_dir)
            / "samples"
            / f"T_{T:g}K"
            / f"P_{P}"
        )
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            seed_dir / f"seed_{task.seed_index}.npz",
            samples=samples,
            T_K=T,
            n_beads=P,
            rng_seed=task.rng_seed,
            initial_radius_nm=task.initial_radius_nm,
        )

    return {
        "version": VERSION,
        "temperature_K": T,
        "n_beads": P,
        "seed_index": int(task.seed_index),
        "rng_seed": int(task.rng_seed),
        "sampler": task.sampler_kind,
        "n_samples": int(samples.shape[0]),
        "initial_radius_nm": float(task.initial_radius_nm),
        "mean_r_nm": float(np.mean(radii)),
        "mean_r2_nm2": float(np.mean(r2)),
        "r_rms_nm": float(np.sqrt(np.mean(r2))),
        "mean_V_total_eV": float(np.mean(V_total)),
        "mean_V_interlayer_eV": float(np.mean(V_interlayer)),
        "mean_V_wall_eV": float(np.mean(V_wall)),
        "r2_centroid_nm2": centroid_r2,
        "r2_spread_nm2": spread_r2,
        "r2_decomposition_residual_nm2": float(
            np.mean(r2) - centroid_r2 - spread_r2
        ),
        "mean_relative_x_nm": float(np.mean(bead_points[:, 0])),
        "mean_relative_y_nm": float(np.mean(bead_points[:, 1])),
        "min_sampled_radius_nm": float(np.min(radii)),
        "max_sampled_radius_nm": float(np.max(radii)),
        "fraction_beyond_wall_radius": float(
            np.mean(radii >= float(task.wall_radius_nm))
        ),
        "hist_lost_fraction": lost_fraction,
        "acceptance_local": float(result.get("acceptance_local", np.nan)),
        "acceptance_staging": float(result.get("acceptance_staging", np.nan)),
        "acceptance_global": float(result.get("acceptance_global", np.nan)),
        "staging_attempts": int(result.get("staging_attempts", 0)),
        "staging_segment_lengths": ",".join(str(int(v)) for v in staging_lengths),
        "staging_acceptance_by_length": ",".join(
            f"{int(length)}:{rate:.8g}"
            for length, rate in zip(staging_lengths, staging_rates)
        ),
        "local_step_nm": local_step,
        "hist_counts": counts.astype(np.int64),
    }


def build_references(args, potential, output_dir: Path, reduced_mass_m0: float):
    references = {}
    rows = []
    for T in args.temps:
        reference = thermal_radial_reference_2d(
            mass_m0=float(reduced_mass_m0),
            potential=potential,
            temperature_K=float(T),
            r_max_nm=float(args.radial_solver_rmax_nm),
            n_grid=int(args.radial_solver_grid_points),
            m_max=int(args.radial_solver_mmax),
            states_per_m=int(args.radial_solver_states_per_m),
        )
        references[float(T)] = reference
        points = np.column_stack([reference.r_nm, np.zeros_like(reference.r_nm)])
        V_interlayer = np.asarray(potential.central_value(points), dtype=float)
        V_wall = np.asarray(potential.wall_value(points), dtype=float)
        pdf = reference.radial_pdf_per_nm
        dr = reference.dr_nm
        mean_V_interlayer = float(np.sum(pdf * V_interlayer) * dr)
        mean_V_wall = float(np.sum(pdf * V_wall) * dr)
        beyond_wall = float(
            np.sum(pdf[reference.r_nm >= args.wall_radius_nm]) * dr
        )
        row = {
            "temperature_K": float(T),
            "ground_energy_eV": reference.ground_energy_eV,
            "first_excited_energy_eV": reference.first_excited_energy_eV,
            "excitation_gap_eV": reference.excitation_gap_eV,
            "beta_gap": reference.excitation_gap_eV / (KB_EV_PER_K * float(T)),
            "ground_state_weight": reference.ground_state_weight,
            "thermal_mean_energy_eV": reference.mean_energy_eV,
            "thermal_mean_V_total_eV": reference.mean_potential_eV,
            "thermal_mean_V_interlayer_eV": mean_V_interlayer,
            "thermal_mean_V_wall_eV": mean_V_wall,
            "thermal_mean_r_nm": reference.mean_r_nm,
            "thermal_mean_r2_nm2": reference.mean_r2_nm2,
            "probability_beyond_wall_radius": beyond_wall,
            "edge_weight_fraction": reference.edge_weight_fraction,
            "included_state_count": reference.included_state_count,
        }
        rows.append(row)
        np.savez_compressed(
            output_dir / f"interlayer_radial_reference_T_{float(T):g}K.npz",
            r_nm=reference.r_nm,
            radial_pdf_per_nm=reference.radial_pdf_per_nm,
            V_interlayer_eV=V_interlayer,
            V_wall_eV=V_wall,
            **row,
        )
    pd.DataFrame(rows).to_csv(
        output_dir / "interlayer_radial_thermal_references.csv", index=False
    )
    return references


def aggregate_results(
    df: pd.DataFrame,
    raw_results: list[dict[str, Any]],
    references,
    edges: np.ndarray,
    output_dir: Path,
    potential: InterlayerScreenedWallPotential,
) -> pd.DataFrame:
    rows = []
    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    grouped_raw: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for result in raw_results:
        grouped_raw.setdefault(
            (result["temperature_K"], result["n_beads"]), []
        ).append(result)

    for (T, P), group in df.groupby(["temperature_K", "n_beads"], sort=True):
        reference = references[float(T)]
        counts = np.sum(
            [item["hist_counts"] for item in grouped_raw[(float(T), int(P))]],
            axis=0,
        ).astype(float)
        if np.sum(counts) <= 0.0:
            raise RuntimeError("Aggregated radial histogram is empty")
        probability = counts / np.sum(counts)
        pimc_density = probability / widths
        ref_density = np.interp(
            centers,
            reference.r_nm,
            reference.radial_pdf_per_nm,
            left=0.0,
            right=0.0,
        )
        ref_density /= np.sum(ref_density * widths)
        l1 = float(np.sum(np.abs(pimc_density - ref_density) * widths))

        n = len(group)
        mean_r = float(group["mean_r_nm"].mean())
        mean_r2 = float(group["mean_r2_nm2"].mean())
        mean_V_total = float(group["mean_V_total_eV"].mean())
        mean_V_interlayer = float(group["mean_V_interlayer_eV"].mean())
        mean_V_wall = float(group["mean_V_wall_eV"].mean())
        r2_std = finite_std(group["mean_r2_nm2"])
        r2_sem = r2_std / np.sqrt(n) if np.isfinite(r2_std) else float("nan")

        reference_points = np.column_stack(
            [reference.r_nm, np.zeros_like(reference.r_nm)]
        )
        ref_V_interlayer = float(
            np.sum(
                reference.radial_pdf_per_nm
                * np.asarray(potential.central_value(reference_points), dtype=float)
            )
            * reference.dr_nm
        )
        ref_V_wall = float(
            np.sum(
                reference.radial_pdf_per_nm
                * np.asarray(potential.wall_value(reference_points), dtype=float)
            )
            * reference.dr_nm
        )

        rows.append(
            {
                "temperature_K": float(T),
                "n_beads": int(P),
                "n_seeds": int(n),
                "mean_r_pimc_nm": mean_r,
                "mean_r_reference_nm": reference.mean_r_nm,
                "mean_r_error_percent": 100.0
                * (mean_r - reference.mean_r_nm)
                / reference.mean_r_nm,
                "mean_r2_pimc_nm2": mean_r2,
                "mean_r2_seed_std_nm2": r2_std,
                "mean_r2_seed_sem_nm2": r2_sem,
                "mean_r2_reference_nm2": reference.mean_r2_nm2,
                "mean_r2_error_percent": 100.0
                * (mean_r2 - reference.mean_r2_nm2)
                / reference.mean_r2_nm2,
                "mean_V_total_pimc_eV": mean_V_total,
                "mean_V_total_reference_eV": reference.mean_potential_eV,
                "mean_V_total_error_meV": 1000.0
                * (mean_V_total - reference.mean_potential_eV),
                "mean_V_interlayer_pimc_eV": mean_V_interlayer,
                "mean_V_interlayer_reference_eV": ref_V_interlayer,
                "mean_V_interlayer_error_meV": 1000.0
                * (mean_V_interlayer - ref_V_interlayer),
                "mean_V_wall_pimc_eV": mean_V_wall,
                "mean_V_wall_reference_eV": ref_V_wall,
                "radial_pdf_L1_distance": l1,
                "acceptance_local_mean": finite_mean(group["acceptance_local"]),
                "acceptance_staging_mean": finite_mean(group["acceptance_staging"]),
                "acceptance_global_mean": finite_mean(group["acceptance_global"]),
                "min_sampled_radius_nm": float(group["min_sampled_radius_nm"].min()),
                "max_sampled_radius_nm": float(group["max_sampled_radius_nm"].max()),
                "fraction_beyond_wall_radius_max": float(
                    group["fraction_beyond_wall_radius"].max()
                ),
                "hist_lost_fraction_max": float(group["hist_lost_fraction"].max()),
                "r2_decomposition_residual_max_abs_nm2": float(
                    np.max(np.abs(group["r2_decomposition_residual_nm2"]))
                ),
                "reference_ground_state_weight": reference.ground_state_weight,
                "reference_edge_weight_fraction": reference.edge_weight_fraction,
                "within_2_percent_r2": bool(
                    abs(
                        100.0
                        * (mean_r2 - reference.mean_r2_nm2)
                        / reference.mean_r2_nm2
                    )
                    < 2.0
                ),
            }
        )
        np.savez_compressed(
            output_dir / f"interlayer_radial_comparison_T_{float(T):g}K_P_{int(P)}.npz",
            radial_bin_edges_nm=edges,
            radial_bin_centers_nm=centers,
            pimc_radial_density_per_nm=pimc_density,
            reference_radial_density_per_nm=ref_density,
            radial_pdf_L1_distance=l1,
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_tag", default="relative_interlayer_validation_v1_7")
    parser.add_argument("--interlayer_table_npz", default=None)

    parser.add_argument("--table_r_max_nm", type=float, default=80.0)
    parser.add_argument("--table_n_log", type=int, default=1000)
    parser.add_argument("--table_n_linear", type=int, default=2000)
    parser.add_argument("--table_q_max_nm_inv", type=float, default=50.0)
    parser.add_argument("--table_n_q", type=int, default=30001)
    parser.add_argument("--table_chunk_size", type=int, default=64)

    parser.add_argument("--wall_radius_nm", type=float, default=25.0)
    parser.add_argument("--wall_height_eV", type=float, default=1.0)
    parser.add_argument("--wall_power", type=int, default=8)
    parser.add_argument("--temps", type=float, nargs="+", default=[20.0])
    parser.add_argument("--beads", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n_steps", type=int, default=20000)
    parser.add_argument("--burn_in", type=int, default=5000)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sampler", choices=["python", "staging", "jit"], default="staging")
    parser.add_argument("--staging_segment_lengths", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--staging_moves_per_step", type=int, default=2)
    parser.add_argument("--staging_no_local_sweep", action="store_true")
    parser.add_argument("--local_step_multiplier", type=float, default=0.70)
    parser.add_argument("--global_step_nm", type=float, default=0.80)
    parser.add_argument("--global_move_probability", type=float, default=0.20)
    parser.add_argument("--jit_grid_size", type=int, default=1400)
    parser.add_argument("--jit_grid_range_nm", type=float, default=30.0)
    parser.add_argument("--initial_radius_nm", type=float, default=None)

    parser.add_argument("--radial_hist_max_nm", type=float, default=30.0)
    parser.add_argument("--radial_hist_bins", type=int, default=360)
    parser.add_argument("--radial_solver_rmax_nm", type=float, default=40.0)
    parser.add_argument("--radial_solver_grid_points", type=int, default=6000)
    parser.add_argument("--radial_solver_mmax", type=int, default=8)
    parser.add_argument("--radial_solver_states_per_m", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=1729)
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_steps <= args.burn_in or args.burn_in < 0:
        raise ValueError("Require n_steps > burn_in >= 0")
    if args.sample_every <= 0 or args.seeds <= 0 or args.workers <= 0:
        raise ValueError("sample_every, seeds and workers must be positive")
    if any(T <= 0.0 for T in args.temps):
        raise ValueError("Temperatures must be positive")
    if any(P < 2 for P in args.beads):
        raise ValueError("All bead counts must be >= 2")
    if args.wall_radius_nm <= 0.0 or args.wall_height_eV < 0.0:
        raise ValueError("Invalid wall parameters")
    if args.wall_power < 2 or args.wall_power % 2:
        raise ValueError("wall_power must be an even integer >= 2")
    if args.radial_solver_rmax_nm <= args.wall_radius_nm:
        raise ValueError("radial_solver_rmax_nm must exceed wall_radius_nm")
    if args.sampler == "staging":
        if args.staging_moves_per_step <= 0:
            raise ValueError("staging_moves_per_step must be positive")
        if not args.staging_segment_lengths or any(
            value < 2 for value in args.staging_segment_lengths
        ):
            raise ValueError("all staging segment lengths must be >= 2")


def prepare_table(args, config: dict[str, Any], output_dir: Path):
    supplied = resolve_project_path(args.interlayer_table_npz)
    if supplied is None:
        table = build_interlayer_table(
            separation_nm=float(config["interlayer_separation_nm"]),
            screening_length_nm=float(config["single_layer_screening_length_nm"]),
            environment_model=str(config["environment_model"]),
            epsilon_env=float(config.get("epsilon_env", 4.5)),
            interface_distance_nm=float(config.get("interface_distance_nm", 0.5)),
            r_max_nm=float(args.table_r_max_nm),
            n_log=int(args.table_n_log),
            n_linear=int(args.table_n_linear),
            q_max_nm_inv=float(args.table_q_max_nm_inv),
            n_q=int(args.table_n_q),
            chunk_size=int(args.table_chunk_size),
        )
    else:
        if not supplied.is_file():
            raise FileNotFoundError(supplied)
        table = load_interlayer_table_npz(supplied)

    used_path = output_dir / "interlayer_table_used.npz"
    save_interlayer_table_npz(
        used_path,
        table,
        q_max_nm_inv=float(args.table_q_max_nm_inv),
        n_q=int(args.table_n_q),
        source_config_json=json.dumps(config, sort_keys=True),
    )
    diagnostics = interlayer_table_diagnostics(
        table,
        q_max_nm_inv=float(args.table_q_max_nm_inv),
        n_q=int(args.table_n_q),
        n_test=32 if args.quick else 96,
    )
    pd.DataFrame([diagnostics]).to_csv(
        output_dir / "interlayer_table_diagnostics.csv", index=False
    )
    if diagnostics["max_relative_q_refinement_error"] > 1.0e-3:
        raise RuntimeError(
            "Interlayer table q-grid refinement error exceeds 1e-3"
        )
    return table, used_path, diagnostics


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    if args.quick:
        args.temps = [20.0]
        args.beads = [32]
        args.seeds = 1
        args.n_steps = 1500
        args.burn_in = 300
        args.sample_every = 20
        args.radial_solver_grid_points = min(args.radial_solver_grid_points, 1800)
        args.table_n_log = min(args.table_n_log, 160)
        args.table_n_linear = min(args.table_n_linear, 320)
        args.table_n_q = min(args.table_n_q, 8001)
        args.table_q_max_nm_inv = min(args.table_q_max_nm_inv, 35.0)
        args.table_r_max_nm = min(args.table_r_max_nm, 40.0)
        print("QUICK mode enabled")
    args.temps = sorted({float(value) for value in args.temps})
    args.beads = sorted({int(value) for value in args.beads})
    validate_args(args)

    output_dir = ROOT / "results" / args.output_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    table, table_path, table_diagnostics = prepare_table(args, config, output_dir)
    potential = InterlayerScreenedWallPotential(
        interlayer=table,
        wall_radius_nm=args.wall_radius_nm,
        wall_height_eV=args.wall_height_eV,
        wall_power=args.wall_power,
    )
    reduced_mass = float(config["reduced_mass_m0"])

    print("=" * 78)
    print("v1.7 direct-contact interlayer exciton PI-QMC validation")
    print("=" * 78)
    print(f"Project root                    : {ROOT}")
    print(f"Config                          : {config_path}")
    print(f"Output directory                : {output_dir}")
    print(f"Structure                       : {config.get('structure', '')}")
    print(f"Electron / hole layers          : {config.get('electron_layer')} / {config.get('hole_layer')}")
    print(f"Reduced mass                    : {reduced_mass:.8f} m0")
    print(f"Vertical separation D           : {table.separation_nm:.6f} nm")
    print(f"Per-layer screening r0          : {table.screening_length_nm:.6f} nm")
    print(f"Environment model               : {table.environment_model}")
    print(f"epsilon / interface distance    : {table.epsilon_env} / {table.interface_distance_nm} nm")
    print(f"Finite V(0)                     : {table.value_at_origin_eV:.6f} eV")
    print(f"Table points / rmax             : {table.r_nm.size} / {table.r_max_nm:g} nm")
    print(f"Max q-refinement rel. error     : {table_diagnostics['max_relative_q_refinement_error']:.3e}")
    print(f"Tail relative error             : {table_diagnostics['tail_relative_error']:.3e}")
    print(f"Sampler                         : {args.sampler}")
    print(f"Temperatures / beads            : {args.temps} / {args.beads}")

    references = build_references(args, potential, output_dir, reduced_mass)
    for T, reference in references.items():
        print(
            f"Reference T={T:g} K: E0={reference.ground_energy_eV:.6f} eV, "
            f"<r>={reference.mean_r_nm:.6f} nm, "
            f"<r^2>={reference.mean_r2_nm2:.6f} nm^2, "
            f"ground weight={reference.ground_state_weight:.8f}, "
            f"edge weight={reference.edge_weight_fraction:.3e}"
        )

    edges = np.linspace(0.0, args.radial_hist_max_nm, args.radial_hist_bins + 1)
    tasks: list[InterlayerTask] = []
    for T in args.temps:
        initial_radius = (
            float(args.initial_radius_nm)
            if args.initial_radius_nm is not None
            else float(references[float(T)].mean_r_nm)
        )
        for P in args.beads:
            for seed_index in range(args.seeds):
                tasks.append(
                    InterlayerTask(
                        temperature_K=float(T),
                        n_beads=int(P),
                        seed_index=seed_index,
                        rng_seed=make_rng_seed(args.base_seed, T, P, seed_index),
                        reduced_mass_m0=reduced_mass,
                        interlayer_table_npz=str(table_path),
                        wall_radius_nm=args.wall_radius_nm,
                        wall_height_eV=args.wall_height_eV,
                        wall_power=args.wall_power,
                        n_steps=args.n_steps,
                        burn_in=args.burn_in,
                        sample_every=args.sample_every,
                        local_step_multiplier=args.local_step_multiplier,
                        global_step_nm=args.global_step_nm,
                        global_move_probability=args.global_move_probability,
                        sampler_kind=args.sampler,
                        staging_segment_lengths=tuple(args.staging_segment_lengths),
                        staging_moves_per_step=args.staging_moves_per_step,
                        staging_perform_local_sweep=not args.staging_no_local_sweep,
                        jit_grid_size=args.jit_grid_size,
                        jit_grid_range_nm=args.jit_grid_range_nm,
                        radial_edges_nm=tuple(float(value) for value in edges),
                        initial_radius_nm=initial_radius,
                        save_samples=args.save_samples,
                        output_dir=str(output_dir),
                    )
                )

    print(f"\nRunning {len(tasks)} independent tasks with {args.workers} worker(s)...")
    started = time.perf_counter()
    if args.workers == 1:
        raw_results = [run_single_task(task) for task in tasks]
    else:
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context
        ) as pool:
            raw_results = list(pool.map(run_single_task, tasks))
    elapsed = time.perf_counter() - started

    scalar_rows = [
        {key: value for key, value in row.items() if key != "hist_counts"}
        for row in raw_results
    ]
    per_seed = pd.DataFrame(scalar_rows).sort_values(
        ["temperature_K", "n_beads", "seed_index"]
    )
    per_seed.to_csv(output_dir / "relative_interlayer_per_seed.csv", index=False)
    convergence = aggregate_results(
        per_seed, raw_results, references, edges, output_dir, potential
    )
    convergence.to_csv(
        output_dir / "relative_interlayer_convergence.csv", index=False
    )

    metadata = {
        "version": VERSION,
        "project_root": str(ROOT),
        "config_path": str(config_path),
        "config": config,
        "output_directory": str(output_dir),
        "elapsed_seconds": elapsed,
        "reduced_mass_m0": reduced_mass,
        "interlayer_table_path": str(table_path),
        "interlayer_table_diagnostics": table_diagnostics,
        "arguments": vars(args),
        "tasks": len(tasks),
        "task_schema": asdict(tasks[0]) if tasks else {},
        "short_distance_note": (
            "Finite vertical separation D makes V(rho=0) finite; the intralayer "
            "RK tau*A logarithmic-collapse gate is not applicable."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nCompleted in {elapsed:.1f} s")
    print(convergence.to_string(index=False))
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()
