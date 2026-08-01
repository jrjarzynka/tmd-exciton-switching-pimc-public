#!/usr/bin/env python3
"""Quick diagnostic: staging acceptance rate by segment length, per (T, P).

Reads relative_rk_per_seed.csv produced by run_relative_rk_validation.py
(--sampler staging) and reports the mean acceptance rate for each staging
segment length, at each (temperature, n_beads) point, averaged across
seeds. Flags any combination whose mean acceptance falls below a threshold
-- the usual failure mode when P grows and long staging segments become too
rigid to move (this is exactly what to watch for when P=32's segment
lengths [4,8,16] are extended to include 32 at P=64/128).

Usage
-----
    python3 check_staging_acceptance.py results/relative_rk_T20_stage1/relative_rk_per_seed.csv
    python3 check_staging_acceptance.py <csv> --warn_below 0.02 --plot acceptance.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_acceptance_field(text: object) -> dict[int, float]:
    """Parse 'length:rate,length:rate,...' -> {length: rate}. Empty/NaN -> {}."""
    result: dict[int, float] = {}
    if not isinstance(text, str) or not text.strip():
        return result
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        length_str, rate_str = entry.split(":")
        result[int(length_str)] = float(rate_str)
    return result


def build_long_table(df: pd.DataFrame) -> pd.DataFrame:
    if "staging_acceptance_by_length" not in df.columns:
        raise SystemExit(
            "Input CSV has no 'staging_acceptance_by_length' column "
            "(was the run made with --sampler staging?)"
        )
    rows = []
    for _, row in df.iterrows():
        parsed = parse_acceptance_field(row["staging_acceptance_by_length"])
        for length, rate in parsed.items():
            rows.append(
                {
                    "temperature_K": row["temperature_K"],
                    "n_beads": int(row["n_beads"]),
                    "seed_index": int(row["seed_index"]),
                    "segment_length": length,
                    "acceptance_rate": rate,
                }
            )
    if not rows:
        raise SystemExit("No parsable staging acceptance entries found in the CSV.")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=str, help="Path to relative_rk_per_seed.csv")
    parser.add_argument(
        "--warn_below",
        type=float,
        default=0.01,
        help="Flag (T, P, length) combos with mean acceptance below this fraction (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional path to save a PNG plot of acceptance vs segment length",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path}")
    df = pd.read_csv(csv_path)

    long_df = build_long_table(df)

    summary = (
        long_df.groupby(["temperature_K", "n_beads", "segment_length"])["acceptance_rate"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .sort_values(["temperature_K", "n_beads", "segment_length"])
    )

    print("=" * 88)
    print(f"Staging acceptance rate by segment length  (source: {csv_path.name})")
    print("=" * 88)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    flagged = summary[summary["mean"] < args.warn_below]
    print()
    if len(flagged):
        print(
            f"WARNING: {len(flagged)} (T, P, length) combination(s) below "
            f"{args.warn_below:.1%} mean acceptance:"
        )
        print(flagged.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
        print()
        print(
            "Consider: dropping the offending segment length, increasing "
            "--staging_moves_per_step, or retuning local_step_multiplier / "
            "global_step_nm at this P before trusting the P-convergence result."
        )
    else:
        print(f"No segment length falls below {args.warn_below:.1%} mean acceptance. Looks OK.")

    # Largest segment length still above threshold, per (T, P) -- useful at a
    # glance when deciding whether e.g. length=32 is usable at this P.
    print()
    print("Largest acceptable segment length per (T, P):")
    for (T, P), group in summary.groupby(["temperature_K", "n_beads"]):
        ok = group[group["mean"] >= args.warn_below]
        best = int(ok["segment_length"].max()) if len(ok) else None
        print(f"  T={T:g} K, P={P}: {best}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for (T, P), group in summary.groupby(["temperature_K", "n_beads"]):
            group = group.sort_values("segment_length")
            ax.plot(
                group["segment_length"],
                group["mean"],
                marker="o",
                label=f"T={T:g} K, P={int(P)}",
            )
        ax.axhline(
            args.warn_below,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"warn threshold ({args.warn_below:.1%})",
        )
        ax.set_xlabel("Staging segment length")
        ax.set_ylabel("Mean acceptance rate")
        ax.set_yscale("log")
        ax.set_title("Staging acceptance vs segment length")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        plot_path = Path(args.plot).expanduser().resolve()
        fig.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
