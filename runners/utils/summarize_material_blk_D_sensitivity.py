#!/usr/bin/env python3
"""Combine radial-only v1.8a D-sensitivity outputs into one CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", nargs="+", help="v1.8a result directories")
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for text in args.result_dirs:
        directory = Path(text).expanduser().resolve()
        ref_path = directory / "material_blk_radial_thermal_references.csv"
        diag_path = directory / "material_blk_table_diagnostics.csv"
        if not ref_path.is_file() or not diag_path.is_file():
            raise FileNotFoundError(
                f"Expected material-BLK reference and diagnostics in {directory}"
            )
        refs = pd.read_csv(ref_path)
        diags = pd.read_csv(diag_path)
        if len(diags) != 1:
            raise ValueError(f"Expected one diagnostics row in {diag_path}")
        diag = diags.iloc[0].to_dict()
        for _, ref in refs.iterrows():
            row = {
                "result_directory": str(directory),
                "temperature_K": float(ref["temperature_K"]),
                "interlayer_separation_nm": float(diag["separation_nm"]),
                "reduced_mass_m0": float(diag.get("reduced_mass_m0", float("nan"))),
                "screening_length_mose2_nm": float(diag["screening_length_layer1_nm"]),
                "screening_length_wse2_nm": float(diag["screening_length_layer2_nm"]),
                "screening_length_sum_nm": float(diag["screening_length_sum_nm"]),
                "kappa_environment": float(diag["kappa_environment"]),
                "V_at_origin_eV": float(diag["value_at_origin_eV"]),
                "ground_energy_eV": float(ref["ground_energy_eV"]),
                "excitation_gap_eV": float(ref["excitation_gap_eV"]),
                "ground_state_weight": float(ref["ground_state_weight"]),
                "thermal_mean_r_nm": float(ref["thermal_mean_r_nm"]),
                "thermal_mean_r2_nm2": float(ref["thermal_mean_r2_nm2"]),
                "thermal_mean_V_material_blk_eV": float(ref["thermal_mean_V_material_blk_eV"]),
                "probability_beyond_wall_radius": float(ref["probability_beyond_wall_radius"]),
                "edge_weight_fraction": float(ref["edge_weight_fraction"]),
                "max_relative_interpolation_error": float(diag["max_relative_interpolation_error"]),
                "tail_relative_error": float(diag["tail_relative_error"]),
            }
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["temperature_K", "interlayer_separation_nm"]
    )
    output = Path(args.output_csv).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
