#!/usr/bin/env python3
"""
Registry-offset scan for the coupled electron-hole two-body model.

Electron landscape V_e is a MoirePotential pinned at the origin. Hole
landscape V_h is the SAME MoirePotential (same amplitude/period), but
rigidly shifted by shift_nm via ShiftedPotential (potential_helpers.py).
This runner includes robustness improvements: per-seed exception handling,
config validation, safer CSV writing (unified fieldnames), optional dry-run,
and graceful handling of missing SciPy for minima detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import tmd_pimc  # noqa: F401
except ImportError:
    _here = Path(__file__).resolve().parent
    _candidates = [
        _here.parent.parent / "numerics",   # <project_root>/runners/<sub>/ -> <project_root>/numerics
        _here.parent / "numerics",          # <project_root>/runners/ -> <project_root>/numerics
        _here.parent / "code", _here.parent, _here / "code",  # legacy layout fallback
    ]
    for _candidate in _candidates:
        if (_candidate / "tmd_pimc").is_dir():
            sys.path.insert(0, str(_candidate))
            break
    else:
        raise ImportError(
            "Could not locate the tmd_pimc package. Looked in: "
            + ", ".join(str(c) for c in _candidates)
        )

from tmd_pimc.bilayer_keldysh_potential import (
    build_bilayer_keldysh_table,
    BilayerKeldyshWallPotential,
)
from tmd_pimc.potentials import MoirePotential
from tmd_pimc.potential_helpers import ShiftedPotential
from tmd_pimc.two_body_action import TwoBodyRingPolymerAction

try:
    from tmd_pimc.two_body_sampler_periodic_jit import TwoBodyPIMCSamplerStagingPeriodicJIT
    _JIT_AVAILABLE = True
except ImportError:
    _JIT_AVAILABLE = False

# Unit conversion
EV_TO_MEV = 1000.0

REQUIRED_KEYS = [
    "separation_nm",
    "screening_length_layer1_nm",
    "screening_length_layer2_nm",
    "kappa_environment",
    "mass_e_m0",
    "mass_h_m0",
    "temperature_K",
    "n_beads",
    "moire_amplitude_eV",
    "moire_period_nm",
]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = json.load(f)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    return config


def validate_config(cfg: Dict[str, Any]) -> None:
    """Proste sprawdzenia sanityzujące wartości z configu."""
    if int(cfg.get("n_seeds", 1)) < 1:
        raise ValueError("n_seeds must be >= 1")
    if int(cfg.get("n_beads", 1)) < 1:
        raise ValueError("n_beads must be >= 1")
    if float(cfg.get("moire_period_nm", 0.0)) <= 0.0:
        raise ValueError("moire_period_nm must be > 0")
    if int(cfg.get("n_steps", 0)) < 0:
        raise ValueError("n_steps must be non-negative")
    if int(cfg.get("burn_in", 0)) < 0:
        raise ValueError("burn_in must be non-negative")


def build_interaction(config: Dict[str, Any]) -> BilayerKeldyshWallPotential:
    table = build_bilayer_keldysh_table(
        separation_nm=float(config["separation_nm"]),
        screening_length_layer1_nm=float(config["screening_length_layer1_nm"]),
        screening_length_layer2_nm=float(config["screening_length_layer2_nm"]),
        kappa_environment=float(config["kappa_environment"]),
        r_max_nm=float(config.get("r_max_nm", 80.0)),
        n_log=int(config.get("n_log", 1000)),
        n_linear=int(config.get("n_linear", 2000)),
    )
    return BilayerKeldyshWallPotential(
        bilayer=table,
        wall_radius_nm=float(config.get("wall_radius_nm", 15.0)),
        wall_height_eV=float(config.get("wall_height_eV", 0.08)),
        wall_power=int(config.get("wall_power", 8)),
    )


def run_single_seed_shift(
    config: Dict[str, Any],
    interaction: BilayerKeldyshWallPotential,
    shift_nm: Tuple[float, float],
    shift_mag: float,
    seed: int,
) -> Dict[str, Any]:
    if not _JIT_AVAILABLE:
        raise RuntimeError(
            "JIT backend is required for this runner. Ensure Numba and two_body_sampler_jit are installed."
        )

    amplitude = float(config["moire_amplitude_eV"])
    period = float(config["moire_period_nm"])

    # NOTE (2026-07-21 reorg): switched from rasterize-onto-finite-box
    # (TwoBodyPIMCSamplerStagingJIT, landscape_grid_range_nm=40 default) to
    # the periodic-cell sampler. MoirePotential has no overall confinement
    # -- see two_body_kernels_periodic_jit.py module docstring -- so once
    # global_step_nm is set large enough to match the real lattice spacing
    # for correct inter-well mixing, the finite-box kernel can let the pair
    # random-walk into the artificial boundary. The periodic kernel has no
    # boundary at all for the landscape term (only the physical BLK wall
    # confines the interaction, which is intentional). Validated against
    # the already-established registry-scan reference (rho2=4.591-4.603
    # nm^2 at shift=0) before this runner was switched over.
    V_e_potential = MoirePotential(amplitude_eV=amplitude, period_nm=period)  # kept for interaction.value() bookkeeping below only

    action = TwoBodyRingPolymerAction(
        mass_e_m0=float(config["mass_e_m0"]),
        mass_h_m0=float(config["mass_h_m0"]),
        temperature_K=float(config["temperature_K"]),
        n_beads=int(config["n_beads"]),
        potential_e=V_e_potential,
        potential_h=ShiftedPotential(inner=V_e_potential, shift_nm=shift_nm),
        potential_interaction=interaction,
    )

    sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(
        action=action,
        moire_period_nm=period,
        moire_amplitude_eV=amplitude,
        origin_e_nm=(0.0, 0.0),
        origin_h_nm=shift_nm,
        field_e_eV_per_nm=(0.0, 0.0),
        field_h_eV_per_nm=(0.0, 0.0),
        local_step_nm=float(config.get("local_step_nm", 0.15)),
        global_step_nm=float(config.get("global_step_nm", 12.0)),
        global_move_probability=float(config.get("global_move_probability", 0.2)),
        rng_seed=int(seed),
        staging_segment_lengths=tuple(config.get(
            "staging_segment_lengths", [4, 8, 16, 32, 64, 128, 256]
        )),
        staging_moves_per_step=int(config.get("staging_moves_per_step", 2)),
        periodic_cell_grid_size=int(config.get("periodic_cell_grid_size", 200)),
    )

    # MoirePotential has a maximum at the origin; start both particles at a non-trivial offset
    start_offset = (period / (2.0 * np.sqrt(3.0)), 0.0)

    t0 = time.time()
    result = sampler.run(
        n_steps=int(config.get("n_steps", 60000)),
        burn_in=int(config.get("burn_in", 15000)),
        sample_every=int(config.get("sample_every", 20)),
        center_e=start_offset,
        center_h=start_offset,
    )
    elapsed_s = time.time() - t0

    samples_e = result["samples_e"]
    samples_h = result["samples_h"]

    # Vectorized shape reductions for coordinate configurations
    rel = samples_e - samples_h
    rho2_samples = np.sum(rel ** 2, axis=-1)
    rho2_mean = float(np.mean(rho2_samples))
    rho_mean = float(np.mean(np.sqrt(rho2_samples)))

    # interaction.value expects shape (-1, 2)
    v_int_samples = interaction.value(rel.reshape(-1, 2))
    v_int_mean = float(np.mean(v_int_samples))

    cent_e = samples_e.mean(axis=1)
    cent_h = samples_h.mean(axis=1)
    centroid_sep_mean = float(np.mean(np.linalg.norm(cent_e - cent_h, axis=1)))

    return {
        "seed": seed,
        "shift_magnitude_nm": shift_mag,
        "shift_x_nm": shift_nm[0],
        "shift_y_nm": shift_nm[1],
        "rho2_nm2": rho2_mean,
        "rho_nm": rho_mean,
        "v_interaction_meV": v_int_mean * EV_TO_MEV,
        "centroid_separation_nm": centroid_sep_mean,
        "acceptance_local_e": result.get("acceptance_local_e", float("nan")),
        "acceptance_local_h": result.get("acceptance_local_h", float("nan")),
        "acceptance_staging": result.get("acceptance_staging", float("nan")),
        "acceptance_global_joint": result.get("acceptance_global_joint", float("nan")),
        "n_samples": result.get("n_samples", 0),
        "elapsed_s": elapsed_s,
    }


def detect_nearest_neighbor_spacing_nm(
    amplitude_eV: float,
    period_nm: float,
    search_range_nm: Optional[float] = None,
    n_points: int = 2001,
) -> float:
    """Numerically find the real nearest-neighbor minima spacing along x.

    NOTE: this is the spacing between adjacent (physically equivalent but
    not identical) potential wells -- useful for calibrating global_step_nm
    against the physical inter-well hop distance. It is NOT the periodicity
    of the potential as a function of a pure-x origin shift; see
    true_x_periodicity_nm for that (found by direct verification, 2026-07-22,
    to differ by a factor of exactly sqrt(3) for this hexagonal landscape --
    an earlier version of this script conflated the two, incorrectly
    labelling this quantity "a full registry cycle").
    """
    try:
        from scipy.signal import find_peaks
    except Exception as exc:
        raise RuntimeError(
            "detect_nearest_neighbor_spacing_nm requires scipy.signal.find_peaks. "
            "Install SciPy or set search_range_nm manually."
        ) from exc

    if search_range_nm is None:
        search_range_nm = 1.5 * period_nm
    V = MoirePotential(amplitude_eV=amplitude_eV, period_nm=period_nm)
    xs = np.linspace(0.0, search_range_nm, n_points)
    vals = V.value(np.column_stack([xs, np.zeros_like(xs)]))
    minima_idx, _ = find_peaks(-vals)
    if len(minima_idx) < 1:
        raise RuntimeError(
            f"Could not find any potential minima along x within search_range_nm={search_range_nm}; increase it."
        )
    return float(xs[minima_idx[0]])


def true_x_periodicity_nm(period_nm: float) -> float:
    """The exact periodicity of MoirePotential along a pure-x origin shift.

    MoirePotential's primitive real-space lattice vectors (derived from its
    reciprocal vectors G1,G2,G3, same construction as
    two_body_kernels_periodic_jit.build_periodic_cell_grid) are
    a1=(period_nm*sqrt(3)/2, period_nm/2), a2=(0, period_nm) -- neither
    purely along x. The smallest integer combination m*a1+n*a2 with zero
    y-component is (m,n)=(2,-1) (or (-2,1)), giving pure-x magnitude
    period_nm*sqrt(3) exactly. Verified numerically to machine precision
    (max difference ~1e-16 eV over test points) on 2026-07-22; see
    PROJECT_STATE handoff doc for the investigation that found this.
    """
    return float(period_nm) * np.sqrt(3.0)


def write_csv_safe(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Zapis CSV z ujednoliceniem wszystkich kluczy w nagłówku."""
    if not rows:
        return
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write outputs. Defaults to <project_root>/results.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and build potentials but do not run samplers.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print more detailed logs.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}", file=sys.stderr)

    config = load_config(args.config)
    validate_config(config)
    interaction = build_interaction(config)

    period = float(config["moire_period_nm"])
    amplitude = float(config["moire_amplitude_eV"])
    try:
        real_spacing_nm = detect_nearest_neighbor_spacing_nm(amplitude, period)
        true_period_nm = true_x_periodicity_nm(period)
        print(
            f"MoirePotential geometry check: period_nm={period} nm parameter.\n"
            f"  Nearest-neighbor minima spacing along x = {real_spacing_nm:.4f} nm "
            f"(useful for global_step_nm calibration against hop distance).\n"
            f"  TRUE periodicity along a pure-x origin shift = {true_period_nm:.4f} nm "
            f"(= period_nm*sqrt(3); verified to machine precision). A shift scan "
            f"intended to cover 'one full registry cycle' must span this value, "
            f"NOT the nearest-neighbor spacing above -- the two differ by exactly "
            f"a factor of 3 for this landscape and were conflated in an earlier "
            f"version of this script.",
            file=sys.stderr,
        )
    except Exception as exc:
        real_spacing_nm = float("nan")
        print(f"Warning: could not determine real nearest-neighbor spacing: {exc}", file=sys.stderr)

    if "shift_values_nm" in config:
        shift_list_nm = [float(v) for v in config["shift_values_nm"]]
    else:
        fractions = config.get("shift_fractions_of_period", [0.0, 0.25, 0.5])
        shift_list_nm = [float(fraction) * period for fraction in fractions]
        print(
            "WARNING: using legacy shift_fractions_of_period. Prefer shift_values_nm in config.",
            file=sys.stderr,
        )

    axis = config.get("shift_axis", "x")
    n_seeds = int(config.get("n_seeds", 3))
    output_prefix = config.get("output_prefix", "landscape_scan")

    per_seed_rows: List[Dict[str, Any]] = []
    start_time = time.time()
    for shift_mag in shift_list_nm:
        shift_nm = (shift_mag, 0.0) if axis == "x" else (0.0, shift_mag)
        fraction_of_real_spacing = shift_mag / real_spacing_nm if real_spacing_nm > 0 else float("nan")

        print(f"[shift = {shift_mag:.3f} nm = {fraction_of_real_spacing:.3f} x real lattice spacing] running...", file=sys.stderr, flush=True)

        for seed_offset in range(n_seeds):
            seed = 3000 + int(round(shift_mag * 1000)) + seed_offset
            if args.dry_run:
                # In dry-run mode, only record planned run metadata
                per_seed_rows.append({
                    "seed": seed,
                    "shift_magnitude_nm": shift_mag,
                    "shift_x_nm": shift_nm[0],
                    "shift_y_nm": shift_nm[1],
                    "note": "dry_run",
                })
                if args.verbose:
                    print(f"  dry-run seed {seed} prepared", file=sys.stderr)
                continue

            try:
                row = run_single_seed_shift(config, interaction, shift_nm, shift_mag, seed)
            except Exception as exc:
                print(f"  seed {seed}: ERROR: {exc}", file=sys.stderr, flush=True)
                per_seed_rows.append({
                    "seed": seed,
                    "shift_magnitude_nm": shift_mag,
                    "shift_x_nm": shift_nm[0],
                    "shift_y_nm": shift_nm[1],
                    "error": str(exc),
                })
                continue

            per_seed_rows.append(row)
            if args.verbose:
                print(
                    f"  seed {seed}: rho2={row.get('rho2_nm2', float('nan')):.4f} nm^2  "
                    f"centroid_sep={row.get('centroid_separation_nm', float('nan')):.4f} nm  "
                    f"acc(staging)={row.get('acceptance_staging', float('nan')):.3f}  "
                    f"t={row.get('elapsed_s', 0.0):.1f}s",
                    file=sys.stderr, flush=True,
                )
            else:
                print(
                    f"  seed {seed}: rho2={row.get('rho2_nm2', float('nan')):.4f} nm^2  t={row.get('elapsed_s', 0.0):.1f}s",
                    file=sys.stderr, flush=True,
                )

    # Write per-seed CSV (safe)
    per_seed_path = output_dir / f"{output_prefix}_per_seed.csv"
    write_csv_safe(per_seed_path, per_seed_rows)
    print(f"Wrote per-seed results to {per_seed_path}", file=sys.stderr)

    # Build summary rows using only complete results (no 'error' and containing expected keys)
    summary_rows: List[Dict[str, Any]] = []
    for shift_mag in shift_list_nm:
        matching = [r for r in per_seed_rows if np.isclose(r.get("shift_magnitude_nm", float("nan")), shift_mag, atol=1e-9)]
        complete = [r for r in matching if "rho2_nm2" in r and "centroid_separation_nm" in r]
        rho2_values = np.array([r["rho2_nm2"] for r in complete]) if complete else np.array([])
        sep_values = np.array([r["centroid_separation_nm"] for r in complete]) if complete else np.array([])
        n = len(complete)

        mean_v_interaction = float(np.mean([r["v_interaction_meV"] for r in matching if "v_interaction_meV" in r])) if any("v_interaction_meV" in r for r in matching) else float("nan")
        mean_accept_global = float(np.mean([r["acceptance_global_joint"] for r in matching if "acceptance_global_joint" in r])) if any("acceptance_global_joint" in r for r in matching) else float("nan")

        summary_rows.append({
            "shift_nm": shift_mag,
            "shift_fraction_of_real_spacing": shift_mag / real_spacing_nm if real_spacing_nm > 0 else float("nan"),
            "n_seeds": n,
            "mean_rho2_nm2": float(np.mean(rho2_values)) if rho2_values.size else float("nan"),
            "sem_rho2_nm2": float(np.std(rho2_values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "mean_centroid_separation_nm": float(np.mean(sep_values)) if sep_values.size else float("nan"),
            "sem_centroid_separation_nm": float(np.std(sep_values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "mean_v_interaction_meV": mean_v_interaction,
            "mean_acceptance_global_joint": mean_accept_global,
        })

    summary_path = output_dir / f"{output_prefix}_summary.csv"
    write_csv_safe(summary_path, summary_rows)
    print(f"Wrote summary to {summary_path}", file=sys.stderr)

    # Print compact summary to stderr
    print("\n=== Summary (registry-offset scan) ===", file=sys.stderr)
    for row in summary_rows:
        print(
            f"  shift={row['shift_nm']:.3f} nm ({row['shift_fraction_of_real_spacing']:.3f} x real lattice): "
            f"rho2={row['mean_rho2_nm2']:.4f}+/-{row['sem_rho2_nm2']:.4f} nm^2  "
            f"centroid_sep={row['mean_centroid_separation_nm']:.4f}+/-{row['sem_centroid_separation_nm']:.4f} nm",
            file=sys.stderr,
        )

    total_elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {total_elapsed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

