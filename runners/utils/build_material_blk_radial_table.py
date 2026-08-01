#!/usr/bin/env python3
"""Build and diagnose a material-specific bilayer Keldysh radial table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def find_project_root() -> Path:
    env_root = os.environ.get("TMD_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve()
    for candidate in [Path.cwd().resolve(), *here.parents]:
        if (candidate / "code" / "tmd_pimc").exists():
            return candidate
    raise RuntimeError("Could not locate project root containing code/tmd_pimc")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "code"))

from tmd_pimc.bilayer_keldysh_potential import (  # noqa: E402
    bilayer_keldysh_table_diagnostics,
    build_bilayer_keldysh_table,
    save_bilayer_keldysh_table_npz,
    screening_length_from_chi2d_nm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--diagnostics_csv", default=None)
    parser.add_argument("--r_max_nm", type=float, default=80.0)
    parser.add_argument("--n_log", type=int, default=1000)
    parser.add_argument("--n_linear", type=int, default=2000)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def read_and_validate_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "reduced_mass_m0",
        "interlayer_separation_nm",
        "screening_length_mose2_nm",
        "screening_length_wse2_nm",
        "kappa_environment",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config is missing keys: {missing}")
    if bool(config.get("hbn_spacer", False)):
        raise ValueError("v1.8a direct model requires hbn_spacer=false")

    checks = (
        ("chi2d_mose2_angstrom", "screening_length_mose2_nm"),
        ("chi2d_wse2_angstrom", "screening_length_wse2_nm"),
    )
    for chi_key, r0_key in checks:
        if chi_key in config and config[chi_key] is not None:
            converted = screening_length_from_chi2d_nm(float(config[chi_key]))
            stored = float(config[r0_key])
            relative = abs(converted - stored) / converted
            if relative > 2.0e-4:
                raise ValueError(
                    f"{r0_key}={stored} nm is inconsistent with "
                    f"{chi_key}={config[chi_key]} A (2*pi*chi={converted} nm)"
                )
    return config


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = read_and_validate_config(config_path)

    if args.quick:
        args.n_log = min(args.n_log, 180)
        args.n_linear = min(args.n_linear, 360)
        args.r_max_nm = min(args.r_max_nm, 40.0)
        print("QUICK material-BLK table mode enabled")

    potential = build_bilayer_keldysh_table(
        separation_nm=float(config["interlayer_separation_nm"]),
        screening_length_layer1_nm=float(config["screening_length_mose2_nm"]),
        screening_length_layer2_nm=float(config["screening_length_wse2_nm"]),
        kappa_environment=float(config["kappa_environment"]),
        layer1_name=str(config.get("electron_layer", "MoSe2")),
        layer2_name=str(config.get("hole_layer", "WSe2")),
        r_max_nm=float(args.r_max_nm),
        n_log=int(args.n_log),
        n_linear=int(args.n_linear),
    )

    output_path = resolve(args.output_npz)
    save_bilayer_keldysh_table_npz(
        output_path,
        potential,
        source_config_json=json.dumps(config, sort_keys=True),
        chi2d_mose2_angstrom=float(config.get("chi2d_mose2_angstrom", float("nan"))),
        chi2d_wse2_angstrom=float(config.get("chi2d_wse2_angstrom", float("nan"))),
        reduced_mass_m0=float(config["reduced_mass_m0"]),
    )
    diagnostics = bilayer_keldysh_table_diagnostics(
        potential,
        n_test=48 if args.quick else 160,
    )
    diagnostics.update(
        {
            "config_path": str(config_path),
            "output_npz": str(output_path),
            "structure": config.get("structure", ""),
            "hbn_spacer": bool(config.get("hbn_spacer", False)),
            "reduced_mass_m0": float(config["reduced_mass_m0"]),
            "chi2d_mose2_angstrom": config.get("chi2d_mose2_angstrom", float("nan")),
            "chi2d_wse2_angstrom": config.get("chi2d_wse2_angstrom", float("nan")),
        }
    )

    if diagnostics["max_relative_interpolation_error"] > 1.0e-3:
        raise RuntimeError("Material-BLK table interpolation error exceeds 1e-3")
    if diagnostics["monotonicity_violations"] != 0:
        raise RuntimeError("Material-BLK table is not monotonic")

    diagnostics_path = (
        resolve(args.diagnostics_csv)
        if args.diagnostics_csv
        else output_path.with_name(output_path.stem + "_diagnostics.csv")
    )
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([diagnostics]).to_csv(diagnostics_path, index=False)

    print("Saved material-BLK table:", output_path)
    print("Saved diagnostics       :", diagnostics_path)
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
