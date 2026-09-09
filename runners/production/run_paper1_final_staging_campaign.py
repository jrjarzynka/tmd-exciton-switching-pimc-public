#!/usr/bin/env python3
"""Final Paper-1 COM staging campaign on the periodic primitive cell.

The scientific execution path is deliberately the validated Paper-1 path:
GridPotential2D -> RingPolymerAction -> PIMCSamplerStagingPeriodicJIT.
"""
from __future__ import annotations

import os

# Set these before importing NumPy/Numba; workers repeat the assignment.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import concurrent.futures
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "numerics"))

from tmd_pimc import GridPotential2D, PIMCSamplerStagingPeriodicJIT, RingPolymerAction
from tmd_pimc.constants import HBAR2_OVER_2M0, KB_EV_PER_K


GRID_SHA256 = "8fbd5d9a9e21c3f7b3655034bf044d396b37f1af8056ae64626fae783564dd65"
GRID_N = 400
MASS_M0 = 0.5
N_STEPS = 640_000
BURN_IN = 40_000
SAMPLE_EVERY = 20
EXPECTED_RETAINED = 30_000
LOCAL_STEP_FACTOR = 0.70
STAGING_MOVES_PER_STEP = 1
GLOBAL_SIGMA_NM = 15.0
GLOBAL_PROBABILITY = 0.20
WARMUP_SEED = 26_090_000
SMOKE_SEED = 26_099_991
SMOKE_N_STEPS = 2_000
SMOKE_BURN_IN = 400
SMOKE_SAMPLE_EVERY = 20
SMOKE_EXPECTED_RETAINED = 80

SCHEDULE = (
    {"T_K": 5.0, "P": 80, "L": 32},
    {"T_K": 10.0, "P": 40, "L": 32},
    {"T_K": 15.0, "P": 32, "L": 16},
    {"T_K": 20.0, "P": 32, "L": 16},
)

# Literal, non-runtime-randomized production inventory.  The first five seeds
# at each temperature start in A and the last five in B.
SEEDS_BY_T = {
    5.0: (26_090_501, 26_090_502, 26_090_503, 26_090_504, 26_090_505,
          26_090_506, 26_090_507, 26_090_508, 26_090_509, 26_090_510),
    10.0: (26_091_001, 26_091_002, 26_091_003, 26_091_004, 26_091_005,
           26_091_006, 26_091_007, 26_091_008, 26_091_009, 26_091_010),
    15.0: (26_091_501, 26_091_502, 26_091_503, 26_091_504, 26_091_505,
           26_091_506, 26_091_507, 26_091_508, 26_091_509, 26_091_510),
    20.0: (26_092_001, 26_092_002, 26_092_003, 26_092_004, 26_092_005,
           26_092_006, 26_092_007, 26_092_008, 26_092_009, 26_092_010),
}

BASIN_UV = {"A": np.array([1.0 / 3.0, 1.0 / 3.0]),
            "B": np.array([2.0 / 3.0, 2.0 / 3.0])}
REQUIRED_GRID_KEYS = {
    "u_fractional", "v_fractional", "V_eV", "lattice_vectors_nm", "origin_nm"
}
SERIES_NAMES = ("q1", "q2", "q3", "R_g2_nm2", "bond_sum_nm2",
                "spring_proxy_eV", "mean_potential_eV")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def retained_count(n_steps: int, burn_in: int, sample_every: int) -> int:
    if n_steps <= burn_in:
        return 0
    return 1 + (n_steps - burn_in - 1) // sample_every


def expected_retained_count() -> int:
    return retained_count(N_STEPS, BURN_IN, SAMPLE_EVERY)


def task_id(task: dict) -> str:
    return (f"paper1_T{int(task['T_K']):03d}_P{task['P']:03d}_L{task['L']:02d}_"
            f"seed{task['seed']}_basin{task['start_basin']}")


def build_inventory(schedule=SCHEDULE, seeds_by_t=SEEDS_BY_T) -> list[dict]:
    tasks = []
    for row in schedule:
        temperature = float(row["T_K"])
        for index, seed in enumerate(seeds_by_t[temperature]):
            task = {"T_K": temperature, "P": int(row["P"]), "L": int(row["L"]),
                    "seed": int(seed), "start_basin": "A" if index < 5 else "B"}
            task["task_id"] = task_id(task)
            tasks.append(task)
    return tasks


TASK_INVENTORY = build_inventory()

SMOKE_TASK = {
    "T_K": 20.0, "P": 32, "L": 16, "seed": SMOKE_SEED, "start_basin": "A",
    "task_id": "SMOKE_T020_P032_L16_seed26099991_basinA",
}
PRODUCTION_RUN = {
    "n_steps": N_STEPS, "burn_in": BURN_IN, "sample_every": SAMPLE_EVERY,
    "expected_retained": EXPECTED_RETAINED, "production": True, "smoke_test": False,
}
SMOKE_RUN = {
    "n_steps": SMOKE_N_STEPS, "burn_in": SMOKE_BURN_IN,
    "sample_every": SMOKE_SAMPLE_EVERY, "expected_retained": SMOKE_EXPECTED_RETAINED,
    "production": False, "smoke_test": True,
}


def _uses_forbidden_historical_seed(seed: int) -> bool:
    text = str(seed)
    return (text.startswith("909") or text.startswith("9103") or
            text.startswith("916") or text.startswith("917") or text.startswith("97"))


def validate_inventory(inventory=TASK_INVENTORY, schedule=SCHEDULE) -> None:
    if len(inventory) != 40:
        raise ValueError(f"expected exactly 40 tasks, found {len(inventory)}")
    seeds = [int(task["seed"]) for task in inventory]
    if len(set(seeds)) != 40:
        raise ValueError("production seeds are not unique")
    if WARMUP_SEED in seeds:
        raise ValueError("warm-up seed overlaps production inventory")
    forbidden = [seed for seed in seeds if _uses_forbidden_historical_seed(seed)]
    if forbidden:
        raise ValueError(f"historical seed range present: {forbidden}")
    if len({task["task_id"] for task in inventory}) != 40:
        raise ValueError("task identifiers are not unique")
    for row in schedule:
        temperature, beads, length = float(row["T_K"]), int(row["P"]), int(row["L"])
        if not 2 <= length < beads:
            raise ValueError(f"invalid staging length at {temperature:g} K: require 2 <= L < P")
        group = [task for task in inventory if task["T_K"] == temperature]
        if len(group) != 10 or sum(x["start_basin"] == "A" for x in group) != 5:
            raise ValueError(f"temperature {temperature:g} K does not have 5 A + 5 B starts")
        if any((task["P"], task["L"]) != (beads, length) for task in group):
            raise ValueError(f"inventory disagrees with schedule at {temperature:g} K")
    if {task["T_K"] for task in inventory} != {float(row["T_K"]) for row in schedule}:
        raise ValueError("inventory contains an unexpected temperature")


def load_grid(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key]) for key in archive.files}


def validate_grid(path: Path, expected_sha256: str = GRID_SHA256) -> dict[str, np.ndarray]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"grid not found: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"grid SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    grid = load_grid(path)
    missing = sorted(REQUIRED_GRID_KEYS.difference(grid))
    if missing:
        raise ValueError(f"grid missing required keys: {missing}")
    expected_axis = np.arange(GRID_N, dtype=float) / GRID_N
    if grid["u_fractional"].shape != (GRID_N,) or grid["v_fractional"].shape != (GRID_N,):
        raise ValueError("grid fractional axes must both have shape (400,)")
    if not (np.array_equal(grid["u_fractional"], expected_axis) and
            np.array_equal(grid["v_fractional"], expected_axis)):
        raise ValueError("grid must be the endpoint-excluded periodic N400 mesh")
    if grid["V_eV"].shape != (GRID_N, GRID_N):
        raise ValueError("V_eV must have shape (400, 400)")
    if grid["lattice_vectors_nm"].shape != (2, 2) or grid["origin_nm"].shape != (2,):
        raise ValueError("invalid primitive-cell geometry arrays")
    if not all(np.all(np.isfinite(grid[key])) for key in REQUIRED_GRID_KEYS):
        raise ValueError("grid contains non-finite values")
    return grid


def validate_output_target(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output target exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"output target must be new or empty: {path}")


def source_provenance() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    files = [
        Path(__file__).resolve(),
        ROOT / "numerics/tmd_pimc/action.py",
        ROOT / "numerics/tmd_pimc/constants.py",
        ROOT / "numerics/tmd_pimc/potentials.py",
        ROOT / "numerics/tmd_pimc/kernels_staging_jit.py",
        ROOT / "numerics/tmd_pimc/sampler_staging_periodic_jit.py",
    ]
    return {"source_commit": commit,
            "source_file_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in files}}


def campaign_configuration() -> dict:
    return {
        "mass_m0": MASS_M0,
        "schedule": [dict(row) for row in SCHEDULE],
        "local_step_nm": "0.70 * sqrt(2 * lambda * tau)",
        "staging_moves_per_step": STAGING_MOVES_PER_STEP,
        "global_translation": "isotropic Gaussian rigid whole-polymer",
        "global_sigma_nm": GLOBAL_SIGMA_NM,
        "global_probability": GLOBAL_PROBABILITY,
        "n_steps": N_STEPS, "burn_in": BURN_IN, "sample_every": SAMPLE_EVERY,
        "expected_retained_per_chain": EXPECTED_RETAINED,
        "chains_per_temperature": 10, "starts_per_basin_per_temperature": 5,
        "thread_environment": {name: "1" for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }


def chain_configuration(run_configuration: dict) -> dict:
    configuration = campaign_configuration()
    configuration.update({
        "n_steps": run_configuration["n_steps"],
        "burn_in": run_configuration["burn_in"],
        "sample_every": run_configuration["sample_every"],
        "expected_retained_per_chain": run_configuration["expected_retained"],
        "production": bool(run_configuration["production"]),
        "smoke_test": bool(run_configuration["smoke_test"]),
    })
    return configuration


def preflight(grid_path: Path, output_path: Path, workers: int) -> dict:
    if not 4 <= workers <= 8:
        raise ValueError("--workers must be in the validated production range 4..8")
    if expected_retained_count() != EXPECTED_RETAINED:
        raise ValueError("sampling arithmetic does not retain exactly 30000 paths")
    validate_inventory()
    validate_grid(grid_path)
    validate_output_target(output_path)
    return {
        "status": "PREFLIGHT_OK", "production_execution_started": False,
        "scientific_chains_run": 0, "grid": str(grid_path.resolve()),
        "grid_sha256": sha256_file(grid_path.resolve()), "output": str(output_path.resolve()),
        "workers": workers, "configuration": campaign_configuration(),
        "task_count": len(TASK_INVENTORY),
        "expected_retained_total": len(TASK_INVENTORY) * EXPECTED_RETAINED,
        "task_inventory": TASK_INVENTORY, "source": source_provenance(),
    }


def smoke_preflight(grid_path: Path, output_path: Path, workers: int) -> dict:
    if not 4 <= workers <= 8:
        raise ValueError("--workers must be in the range 4..8")
    validate_inventory()
    production_seeds = {task["seed"] for task in TASK_INVENTORY}
    if SMOKE_SEED in production_seeds or SMOKE_SEED == WARMUP_SEED:
        raise ValueError("smoke seed overlaps a production or warm-up seed")
    if not 2 <= SMOKE_TASK["L"] < SMOKE_TASK["P"]:
        raise ValueError("smoke task must satisfy 2 <= L < P")
    if retained_count(SMOKE_N_STEPS, SMOKE_BURN_IN, SMOKE_SAMPLE_EVERY) != SMOKE_EXPECTED_RETAINED:
        raise ValueError("smoke sampling arithmetic does not retain exactly 80 paths")
    validate_grid(grid_path)
    validate_output_target(output_path)
    return {
        "status": "SMOKE_PREFLIGHT_OK", "production": False, "smoke_test": True,
        "production_execution_started": False, "scientific_chains_planned": 1,
        "grid": str(grid_path.resolve()), "grid_sha256": sha256_file(grid_path.resolve()),
        "output": str(output_path.resolve()), "workers": workers,
        "task": dict(SMOKE_TASK), "run_configuration": dict(SMOKE_RUN),
        "production_inventory_included": False, "source": source_provenance(),
    }


def make_action(grid: dict[str, np.ndarray], task: dict):
    potential = GridPotential2D(
        grid["u_fractional"], grid["v_fractional"], grid["V_eV"], periodic=True,
        subtract_minimum=False, lattice_vectors_nm=grid["lattice_vectors_nm"],
        origin_nm=grid["origin_nm"],
    )
    return RingPolymerAction(MASS_M0, task["T_K"], task["P"], potential), potential


def local_step_nm(action: RingPolymerAction) -> float:
    return LOCAL_STEP_FACTOR * math.sqrt(2.0 * action._lambda_x * action._tau)


def make_sampler(action: RingPolymerAction, task: dict, seed: int):
    return PIMCSamplerStagingPeriodicJIT(
        action=action, local_step_nm=local_step_nm(action), global_step_nm=GLOBAL_SIGMA_NM,
        global_move_probability=GLOBAL_PROBABILITY, rng_seed=seed,
        staging_segment_lengths=(task["L"],), staging_moves_per_step=STAGING_MOVES_PER_STEP,
    )


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    centered = values - values.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    fft = np.fft.rfft(centered, nfft)
    autocov = np.fft.irfft(fft * np.conj(fft), nfft)[:n] / np.arange(n, 0, -1)
    if not np.isfinite(autocov[0]) or autocov[0] <= 0.0:
        return 1.0
    rho = autocov / autocov[0]
    pair_sum, index = 0.0, 1
    while 2 * index < n:
        pair = rho[2 * index - 1] + rho[2 * index]
        if not np.isfinite(pair) or pair <= 0.0:
            break
        pair_sum += pair
        index += 1
    return float(max(1.0, 1.0 + 2.0 * pair_sum))


def _series_summary(values: np.ndarray, wall_seconds: float) -> dict:
    values = np.asarray(values, dtype=float)
    tau = integrated_autocorrelation_time(values)
    ess = len(values) / tau
    midpoint = len(values) // 2
    first, second = values[:midpoint], values[midpoint:]
    scale = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean": float(np.mean(values)), "std": scale, "IAT": tau, "ESS": float(ess),
        "ESS_per_second": float(ess / wall_seconds),
        "first_half_mean": float(np.mean(first)), "second_half_mean": float(np.mean(second)),
        "half_mean_difference_in_pooled_sd": (float((np.mean(second) - np.mean(first)) / scale)
                                               if scale > 0.0 else 0.0),
    }


def diagnostics(samples: np.ndarray, action, potential, grid: dict, wall_seconds: float) -> dict:
    n, beads, _ = samples.shape
    centroid = np.mean(samples, axis=1)
    relative = samples - centroid[:, None, :]
    rg2 = np.mean(np.sum(relative * relative, axis=2), axis=1)
    del relative
    bonds = np.roll(samples, -1, axis=1) - samples
    bond_sum = np.sum(bonds * bonds, axis=(1, 2))
    del bonds
    spring = bond_sum / (4.0 * action._lambda_x * beads * action._tau * action._tau)
    potential_series = np.mean(
        potential.value(samples.reshape(-1, 2)).reshape(n, beads), axis=1
    )
    bead_index = np.arange(beads, dtype=float)
    q = {}
    for mode in (1, 2, 3):
        phase = np.exp(-2j * np.pi * mode * bead_index / beads) / beads
        amplitude = np.einsum("npd,p->nd", samples, phase, optimize=True)
        q[f"q{mode}"] = np.sum(np.abs(amplitude) ** 2, axis=1)
    series = {**q, "R_g2_nm2": rg2, "bond_sum_nm2": bond_sum,
              "spring_proxy_eV": spring, "mean_potential_eV": potential_series}

    matrix = np.column_stack([grid["lattice_vectors_nm"][0], grid["lattice_vectors_nm"][1]])
    fractional_unwrapped = (centroid - grid["origin_nm"]) @ np.linalg.inv(matrix).T
    fractional_wrapped = fractional_unwrapped - np.floor(fractional_unwrapped)
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)
    distances = []
    for minimum in (BASIN_UV["A"], BASIN_UV["B"]):
        delta_uv = fractional_wrapped[:, None, :] - (minimum[None, None, :] + shifts[None, :, :])
        delta_real = delta_uv @ matrix.T
        distances.append(np.min(np.sum(delta_real * delta_real, axis=2), axis=1))
    basin_labels = np.argmin(np.column_stack(distances), axis=1).astype(np.int8)
    histogram = np.histogram2d(
        fractional_wrapped[:, 0], fractional_wrapped[:, 1], bins=16, range=((0, 1), (0, 1))
    )[0]
    histogram /= histogram.sum()
    summaries = {name: _series_summary(values, wall_seconds) for name, values in series.items()}
    return {
        "centroids_unwrapped_nm": centroid,
        "centroids_wrapped_fractional": fractional_wrapped,
        "series": series, "series_summaries": summaries, "basin_labels": basin_labels,
        "basin_populations": {"A": float(np.mean(basin_labels == 0)),
                              "B": float(np.mean(basin_labels == 1))},
        "basin_label_encoding": {"0": "A", "1": "B"},
        "basin_transitions": int(np.sum(basin_labels[1:] != basin_labels[:-1])),
        "fractional_cell_histogram_16x16": histogram,
        "centroid_fractional_unwrapped_range": {
            "u": [float(np.min(fractional_unwrapped[:, 0])), float(np.max(fractional_unwrapped[:, 0]))],
            "v": [float(np.min(fractional_unwrapped[:, 1])), float(np.max(fractional_unwrapped[:, 1]))],
        },
    }


def _write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def chain_identity(task: dict, run_configuration: dict) -> dict:
    smoke_test = bool(run_configuration["smoke_test"])
    return {
        "task_id": task["task_id"],
        "classification": ("development_only_smoke_test" if smoke_test
                           else "paper1_final_production"),
        "production": bool(run_configuration["production"]),
        "smoke_test": smoke_test,
    }


def run_one(task: dict, grid_path: str, output_path: str, provenance: dict,
            run_configuration: dict = PRODUCTION_RUN) -> dict:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    grid_file, out = Path(grid_path), Path(output_path)
    grid = validate_grid(grid_file)
    action, potential = make_action(grid, task)
    matrix = np.column_stack([grid["lattice_vectors_nm"][0], grid["lattice_vectors_nm"][1]])
    center = grid["origin_nm"] + matrix @ BASIN_UV[task["start_basin"]]

    # Compile on a disposable sampler.  Recreate with the untouched chain seed
    # before timing, exactly as in the validated moire pilot.
    warmup = make_sampler(action, task, WARMUP_SEED)
    warmup.run(n_steps=2, burn_in=0, sample_every=1, center=center)
    sampler = make_sampler(action, task, task["seed"])
    started = time.perf_counter()
    result = sampler.run(
        run_configuration["n_steps"], run_configuration["burn_in"],
        run_configuration["sample_every"], center=center,
    )
    wall_seconds = time.perf_counter() - started
    samples = np.asarray(result["samples"])
    expected_shape = (run_configuration["expected_retained"], task["P"], 2)
    if samples.shape != expected_shape:
        raise RuntimeError(f"unexpected retained sample shape for {task['task_id']}: {samples.shape}")
    diag = diagnostics(samples, action, potential, grid, wall_seconds)
    attempts = np.asarray(result["staging_length_attempts"], dtype=np.int64)
    accepted = np.asarray(result["staging_length_accepted"], dtype=np.int64)
    metadata = {
        **chain_identity(task, run_configuration),
        "task": dict(task), "configuration": chain_configuration(run_configuration),
        "run_configuration": dict(run_configuration),
        "local_step_nm": local_step_nm(action), "wall_seconds": wall_seconds,
        "retained": int(len(samples)), "acceptance_local": float(result["acceptance_local"]),
        "acceptance_staging": float(result["acceptance_staging"]),
        "acceptance_global": float(result["acceptance_global"]),
        "staging_attempts": int(np.sum(attempts)), "staging_accepted": int(np.sum(accepted)),
        "staging_length_attempts": attempts.tolist(), "staging_length_accepted": accepted.tolist(),
        "series_summaries": diag["series_summaries"],
        "basin_label_encoding": diag["basin_label_encoding"],
        "basin_populations": diag["basin_populations"],
        "basin_transitions": diag["basin_transitions"],
        "centroid_fractional_unwrapped_range": diag["centroid_fractional_unwrapped_range"],
        "fractional_cell_histogram_16x16": diag["fractional_cell_histogram_16x16"].tolist(),
        "grid_path": str(grid_file.resolve()), "grid_sha256": GRID_SHA256,
        "source": provenance,
    }
    final_npz = out / f"{task['task_id']}.npz"
    final_json = out / f"{task['task_id']}.json"
    if final_npz.exists() or final_json.exists():
        raise FileExistsError(f"refusing to overwrite chain output: {task['task_id']}")
    temporary_npz = out / f".{task['task_id']}.partial.npz"
    if temporary_npz.exists():
        raise FileExistsError(f"stale partial output exists: {temporary_npz}")
    np.savez_compressed(
        temporary_npz, samples_unwrapped_nm=samples,
        centroids_unwrapped_nm=diag["centroids_unwrapped_nm"],
        centroids_wrapped_fractional=diag["centroids_wrapped_fractional"],
        potential_series_eV=diag["series"]["mean_potential_eV"],
        spring_proxy_series_eV=diag["series"]["spring_proxy_eV"],
        bond_sum_series_nm2=diag["series"]["bond_sum_nm2"],
        R_g2_series_nm2=diag["series"]["R_g2_nm2"],
        q1_series=diag["series"]["q1"], q2_series=diag["series"]["q2"],
        q3_series=diag["series"]["q3"], basin_labels=diag["basin_labels"],
        basin_label_encoding_json=np.array(json.dumps(diag["basin_label_encoding"], sort_keys=True)),
        basin_population_A=diag["basin_populations"]["A"],
        basin_population_B=diag["basin_populations"]["B"],
        basin_transitions=diag["basin_transitions"],
        fractional_cell_histogram_16x16=diag["fractional_cell_histogram_16x16"],
        staging_length_attempts=attempts, staging_length_accepted=accepted,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    )
    # Hard-link publication is atomic and fails if the destination appeared
    # concurrently; unlike os.replace it can never overwrite an existing NPZ.
    os.link(temporary_npz, final_npz)
    temporary_npz.unlink()
    metadata["npz_sha256"] = sha256_file(final_npz)
    try:
        _write_json_exclusive(final_json, metadata)
    except Exception:
        # Keep the NPZ rather than deleting scientific output; a rerun still
        # refuses to overwrite it and exposes the incomplete pair for recovery.
        raise
    return metadata


def aggregate_temperature(temperature: float, output_path: Path) -> dict:
    expected = [task for task in TASK_INVENTORY if task["T_K"] == temperature]
    missing = []
    for task in expected:
        for suffix in (".npz", ".json"):
            path = output_path / f"{task['task_id']}{suffix}"
            if not path.is_file():
                missing.append(path.name)
    if missing:
        raise RuntimeError(f"will not aggregate incomplete {temperature:g} K inventory: {missing}")
    chains = []
    for task in expected:
        json_path = output_path / f"{task['task_id']}.json"
        chain = json.loads(json_path.read_text(encoding="utf-8"))
        if chain["npz_sha256"] != sha256_file(output_path / f"{task['task_id']}.npz"):
            raise RuntimeError(f"NPZ checksum mismatch during aggregation: {task['task_id']}")
        chains.append(chain)
    histogram = np.mean([np.asarray(x["fractional_cell_histogram_16x16"]) for x in chains], axis=0)
    summary = {
        "T_K": temperature, "chain_count": len(chains),
        "task_ids": [x["task_id"] for x in chains],
        "mean_acceptance": {name: float(np.mean([x[f"acceptance_{name}"] for x in chains]))
                            for name in ("local", "staging", "global")},
        "total_staging_attempts": int(sum(x["staging_attempts"] for x in chains)),
        "total_staging_accepted": int(sum(x["staging_accepted"] for x in chains)),
        "total_basin_transitions": int(sum(x["basin_transitions"] for x in chains)),
        "mean_basin_populations": {name: float(np.mean([x["basin_populations"][name] for x in chains]))
                                   for name in ("A", "B")},
        "metrics": {series: {
            "mean_of_chain_means": float(np.mean([x["series_summaries"][series]["mean"] for x in chains])),
            "total_ESS": float(sum(x["series_summaries"][series]["ESS"] for x in chains)),
            "total_ESS_per_total_second": float(
                sum(x["series_summaries"][series]["ESS"] for x in chains) /
                sum(x["wall_seconds"] for x in chains)),
        } for series in SERIES_NAMES},
        "fractional_cell_histogram_16x16": histogram.tolist(),
        "chain_npz_sha256": {x["task_id"]: x["npz_sha256"] for x in chains},
    }
    destination = output_path / f"aggregate_T{int(temperature):03d}.json"
    _write_json_exclusive(destination, summary)
    return summary


def execute_campaign(plan: dict, grid_path: Path, output_path: Path, workers: int) -> None:
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(output_path / "PREFLIGHT.json", plan)
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_one, task, str(grid_path), str(output_path), plan["source"])
                   for task in TASK_INVENTORY]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed.append(row["task_id"])
            print(f"COMPLETE {row['task_id']}", flush=True)
    if set(completed) != {task["task_id"] for task in TASK_INVENTORY}:
        raise RuntimeError("completed task inventory is inconsistent")
    aggregates = [aggregate_temperature(float(row["T_K"]), output_path) for row in SCHEDULE]
    manifest = {"status": "COMPLETE", "configuration": campaign_configuration(),
                "grid_sha256": GRID_SHA256, "source": plan["source"],
                "completed_task_ids": sorted(completed), "temperature_aggregates": aggregates}
    _write_json_exclusive(output_path / "CAMPAIGN_COMPLETE.json", manifest)


def execute_smoke_test(plan: dict, grid_path: Path, output_path: Path) -> dict:
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=False)
    result = run_one(
        SMOKE_TASK, str(grid_path), str(output_path), plan["source"], SMOKE_RUN,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke_test:
        plan = smoke_preflight(args.grid, args.out, args.workers)
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False), flush=True)
        execute_smoke_test(plan, args.grid.resolve(), args.out.resolve())
        return 0
    plan = preflight(args.grid, args.out, args.workers)
    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False), flush=True)
    if args.preflight_only:
        return 0
    execute_campaign(plan, args.grid.resolve(), args.out.resolve(), args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
