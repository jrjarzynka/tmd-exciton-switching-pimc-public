"""Validation runner for the coupled electron-hole two-body PI-QMC model.

Mirrors the structure of run_relative_material_blk_validation.py: JSON
config in, per-seed CSV + aggregate convergence CSV out. Baseline mode
(default) uses V_e = V_h = 0, so the relative coordinate r_e - r_h should
reproduce the already-validated v1.8a single-body BLK result -- this is
the two-body machinery's sanity check before real per-layer landscapes
(moire, Stark) are wired into V_e / V_h.

Usage
-----
    python run_two_body_validation.py --config two_body_config.json --backend jit

Example two_body_config.json
-----------------------------
{
  "separation_nm": 0.60,
  "screening_length_layer1_nm": 4.479911124019045,
  "screening_length_layer2_nm": 3.4934510307918494,
  "kappa_environment": 4.945,
  "mass_e_m0": 0.58,
  "mass_h_m0": 0.36,
  "temperature_K": 20.0,
  "n_beads": 256,
  "wall_radius_nm": 15.0,
  "wall_height_eV": 0.08,
  "wall_power": 8,
  "n_seeds": 5,
  "n_steps": 60000,
  "burn_in": 15000,
  "sample_every": 20,
  "local_step_nm": 0.15,
  "global_step_nm": 1.0,
  "global_move_probability": 0.2,
  "staging_segment_lengths": [4, 8, 16, 32, 64, 128, 256],
  "staging_moves_per_step": 2,
  "reference_rho2_nm2": 5.3927,
  "reference_rho_nm": 1.9778,
  "output_prefix": "two_body_baseline",
  "potential_e_type": "zero",
  "potential_h_type": "zero"
}
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Auto-resolve path for package discovery
try:
    import tmd_pimc  # noqa: F401
except ImportError:
    _here = Path(__file__).resolve().parent
    _candidates = [
        _here.parent.parent / "numerics",  # <project_root>/runners/<sub>/ -> <project_root>/numerics
        _here.parent / "numerics",         # <project_root>/runners/ -> <project_root>/numerics
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
            + ". If your layout differs, set PYTHONPATH manually."
        )

import numpy as np

from tmd_pimc.bilayer_keldysh_potential import (
    build_bilayer_keldysh_table,
    BilayerKeldyshWallPotential,
)
from tmd_pimc.potentials import Potential2D
from tmd_pimc.two_body_action import TwoBodyRingPolymerAction
from tmd_pimc.two_body_sampler import TwoBodyPIMCSamplerStaging

try:
    from tmd_pimc.two_body_sampler_jit import TwoBodyPIMCSamplerStagingJIT
    _JIT_AVAILABLE = True
except ImportError:
    _JIT_AVAILABLE = False


class ZeroPotential(Potential2D):
    """Placeholder one-body landscape for baseline checks."""
    
    def value(self, r: np.ndarray) -> np.ndarray:
        return np.zeros(r.shape[0])


class BespokePotentialFactory:
    """Factory to dispatch target external single-particle potentials based on configuration."""
    
    @staticmethod
    def create(potential_type: str, config: Dict[str, Any]) -> Potential2D:
        p_type = potential_type.lower().strip()
        if p_type == "zero" or not p_type:
            return ZeroPotential()
        
        # Extensible setup for future real landscapes (e.g., moire, Stark)
        # elif p_type == "moire":
        #     return MoirePotential2D(amplitude=config.get("moire_amp_eV", 0.01), ...)
        
        raise ValueError(f"Unknown single-body potential type specified: '{potential_type}'")


REQUIRED_KEYS = [
    "separation_nm",
    "screening_length_layer1_nm",
    "screening_length_layer2_nm",
    "kappa_environment",
    "mass_e_m0",
    "mass_h_m0",
    "temperature_K",
    "n_beads",
]


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = json.load(f)
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")
    return config


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


def run_single_seed(config: Dict[str, Any], interaction: BilayerKeldyshWallPotential, seed: int, backend: str = "python") -> Dict[str, Any]:
    # Resolve single-particle potentials dynamically
    pot_e_type = config.get("potential_e_type", "zero")
    pot_h_type = config.get("potential_h_type", "zero")
    
    potential_e = BespokePotentialFactory.create(pot_e_type, config)
    potential_h = BespokePotentialFactory.create(pot_h_type, config)

    action = TwoBodyRingPolymerAction(
        mass_e_m0=float(config["mass_e_m0"]),
        mass_h_m0=float(config["mass_h_m0"]),
        temperature_K=float(config["temperature_K"]),
        n_beads=int(config["n_beads"]),
        potential_e=potential_e,
        potential_h=potential_h,
        potential_interaction=interaction,
    )

    sampler_kwargs = dict(
        action=action,
        local_step_nm=float(config.get("local_step_nm", 0.15)),
        global_step_nm=float(config.get("global_step_nm", 1.0)),
        global_move_probability=float(config.get("global_move_probability", 0.2)),
        rng_seed=int(seed),
        staging_segment_lengths=tuple(config.get(
            "staging_segment_lengths", [4, 8, 16, 32, 64, 128, 256]
        )),
        staging_moves_per_step=int(config.get("staging_moves_per_step", 2)),
    )

    if backend == "jit":
        if not _JIT_AVAILABLE:
            raise RuntimeError(
                "backend='jit' requested but tmd_pimc.two_body_sampler_jit "
                "could not be imported. Ensure Numba is configured correctly."
            )
        sampler = TwoBodyPIMCSamplerStagingJIT(**sampler_kwargs)
    elif backend == "python":
        sampler = TwoBodyPIMCSamplerStaging(**sampler_kwargs)
    else:
        raise ValueError(f"Unknown backend '{backend}'; use 'python' or 'jit'")

    t0 = time.time()
    result = sampler.run(
        n_steps=int(config.get("n_steps", 60000)),
        burn_in=int(config.get("burn_in", 15000)),
        sample_every=int(config.get("sample_every", 20)),
        center_e=(1.0, 0.0),
        center_h=(-1.0, 0.0),
    )
    elapsed_s = time.time() - t0

    # Matrix operations to extract coordinates
    rel = result["samples_e"] - result["samples_h"]
    rho2_samples = np.sum(rel ** 2, axis=-1)  # Faster shape reduction (N_samples, P)
    
    rho2_mean = float(np.mean(rho2_samples))
    rho_mean = float(np.mean(np.sqrt(rho2_samples)))

    v_int_samples = interaction.value(rel.reshape(-1, 2))
    v_int_mean = float(np.mean(v_int_samples))

    return {
        "seed": seed,
        "rho2_nm2": rho2_mean,
        "rho_nm": rho_mean,
        "v_interaction_meV": v_int_mean * 1000.0,
        "acceptance_local_e": result["acceptance_local_e"],
        "acceptance_local_h": result["acceptance_local_h"],
        "acceptance_staging": result["acceptance_staging"],
        "acceptance_global_joint": result["acceptance_global_joint"],
        "n_samples": result["n_samples"],
        "elapsed_s": elapsed_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument(
        "--backend", choices=["python", "jit"], default="python",
        help="Sampler backend: 'python' (Reference) or 'jit' (Numba accelerated). Default: python.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write output CSV logs. Defaults to <project_root>/results.",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent.parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}", file=sys.stderr)

    config = load_config(args.config)
    interaction = build_interaction(config)
    
    if args.backend == "jit" and not _JIT_AVAILABLE:
        print("WARNING: JIT backend requested but not available. Execution will fail if run.", file=sys.stderr)

    n_seeds = int(config.get("n_seeds", 5))
    output_prefix = config.get("output_prefix", "two_body_validation")

    per_seed_rows: List[Dict[str, Any]] = []
    for seed_offset in range(n_seeds):
        seed = 1000 + seed_offset
        print(f"[seed {seed_offset + 1}/{n_seeds}] running...", file=sys.stderr, flush=True)
        row = run_single_seed(config, interaction, seed, backend=args.backend)
        per_seed_rows.append(row)
        print(
            f"  rho2={row['rho2_nm2']:.4f} nm^2  "
            f"acc(local e/h)={row['acceptance_local_e']:.3f}/{row['acceptance_local_h']:.3f}  "
            f"acc(staging)={row['acceptance_staging']:.3f}  "
            f"acc(global)={row['acceptance_global_joint']:.3f}  "
            f"t={row['elapsed_s']:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    per_seed_path = output_dir / f"{output_prefix}_per_seed.csv"
    with open(per_seed_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    rho2_values = np.array([r["rho2_nm2"] for r in per_seed_rows])
    rho_values = np.array([r["rho_nm"] for r in per_seed_rows])

    summary = {
        "n_seeds": n_seeds,
        "mean_rho2_nm2": float(np.mean(rho2_values)),
        "sem_rho2_nm2": float(np.std(rho2_values, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0,
        "mean_rho_nm": float(np.mean(rho_values)),
        "sem_rho_nm": float(np.std(rho_values, ddof=1) / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0,
        "mean_acceptance_local_e": float(np.mean([r["acceptance_local_e"] for r in per_seed_rows])),
        "mean_acceptance_local_h": float(np.mean([r["acceptance_local_h"] for r in per_seed_rows])),
        "mean_acceptance_staging": float(np.mean([r["acceptance_staging"] for r in per_seed_rows])),
        "mean_acceptance_global_joint": float(np.mean([r["acceptance_global_joint"] for r in per_seed_rows])),
    }

    # Reference deviation logging if single-body checks are supplied
    if "reference_rho2_nm2" in config:
        ref = float(config["reference_rho2_nm2"])
        summary["reference_rho2_nm2"] = ref
        summary["rho2_error_percent"] = 100.0 * (summary["mean_rho2_nm2"] - ref) / ref
        summary["within_2_percent_rho2"] = bool(abs(summary["rho2_error_percent"]) < 2.0)

    if "reference_rho_nm" in config:
        ref_rho = float(config["reference_rho_nm"])
        summary["reference_rho_nm"] = ref_rho
        summary["rho_error_percent"] = 100.0 * (summary["mean_rho_nm"] - ref_rho) / ref_rho

    summary_path = output_dir / f"{output_prefix}_convergence.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n=== Summary ===", file=sys.stderr)
    for key, value in summary.items():
        print(f"  {key}: {value}", file=sys.stderr)
    print(f"\nWrote {per_seed_path} and {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
