#!/usr/bin/env python3
"""
Merge RK validation results from multiple run directories into one canonical
validation directory.

Example:
    python3 scripts/merge_rk_validation_results.py \
      --source_dirs \
        results/relative_rk_T20_stage1 \
        results/relative_rk_T20_P256 \
      --target_dir results/relative_rk_T20_validation_complete
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


CSV_FILES = (
    "relative_rk_convergence.csv",
    "relative_rk_per_seed.csv",
)

REFERENCE_FILES = (
    "rk_radial_thermal_references.csv",
    "rk_table_diagnostics.csv",
    "rk_table_used.npz",
)

COMPARISON_GLOB = "rk_radial_comparison_*.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge several RK validation runs into one directory."
    )
    parser.add_argument(
        "--source_dirs",
        nargs="+",
        required=True,
        help="Input result directories containing RK CSV files.",
    )
    parser.add_argument(
        "--target_dir",
        required=True,
        help="Output directory for the merged validation series.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing files in the target directory.",
    )
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def merge_csv_files(
    sources: list[Path],
    target: Path,
    overwrite: bool,
) -> None:
    for filename in CSV_FILES:
        frames: list[pd.DataFrame] = []

        for source in sources:
            csv_path = source / filename
            if not csv_path.is_file():
                raise FileNotFoundError(
                    f"Required input file not found: {csv_path}"
                )

            frame = pd.read_csv(csv_path)
            frame["source_run"] = source.name
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True)

        duplicate_subset = [
            column
            for column in (
                "temperature_K",
                "n_beads",
                "seed_index",
                "source_run",
            )
            if column in combined.columns
        ]
        if duplicate_subset:
            combined = combined.drop_duplicates(
                subset=duplicate_subset,
                keep="last",
            )

        sort_columns = [
            column
            for column in (
                "temperature_K",
                "n_beads",
                "seed_index",
                "source_run",
            )
            if column in combined.columns
        ]
        if sort_columns:
            combined = combined.sort_values(sort_columns).reset_index(drop=True)

        output_path = target / filename
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {output_path}. "
                "Use --overwrite to replace it."
            )

        combined.to_csv(output_path, index=False)
        print(f"Merged: {output_path}")


def copy_reference_files(
    sources: list[Path],
    target: Path,
    overwrite: bool,
) -> None:
    reference_source = sources[0]

    for filename in REFERENCE_FILES:
        source_path = reference_source / filename
        if not source_path.exists():
            print(f"Optional reference file not found, skipping: {source_path}")
            continue

        target_path = target / filename
        if target_path.exists() and not overwrite:
            print(f"Already exists, keeping: {target_path}")
            continue

        shutil.copy2(source_path, target_path)
        print(f"Copied: {target_path}")


def copy_radial_comparisons(
    sources: list[Path],
    target: Path,
    overwrite: bool,
) -> None:
    for source in sources:
        for source_path in source.glob(COMPARISON_GLOB):
            target_path = target / source_path.name

            if target_path.exists() and not overwrite:
                print(f"Already exists, keeping: {target_path}")
                continue

            shutil.copy2(source_path, target_path)
            print(f"Copied: {target_path}")


def main() -> None:
    args = parse_args()
    sources = [resolve_path(path) for path in args.source_dirs]
    target = resolve_path(args.target_dir)

    for source in sources:
        if not source.is_dir():
            raise NotADirectoryError(f"Source directory not found: {source}")

    target.mkdir(parents=True, exist_ok=True)

    merge_csv_files(
        sources=sources,
        target=target,
        overwrite=args.overwrite,
    )
    copy_reference_files(
        sources=sources,
        target=target,
        overwrite=args.overwrite,
    )
    copy_radial_comparisons(
        sources=sources,
        target=target,
        overwrite=args.overwrite,
    )

    print()
    print(f"Complete merged validation directory: {target}")


if __name__ == "__main__":
    main()
