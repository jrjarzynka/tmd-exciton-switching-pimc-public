#!/usr/bin/env python3
"""
Aggregate and validate relative-coordinate Rytova--Keldysh convergence runs.

The script scans one or more result directories, reads each
``relative_rk_convergence.csv`` and its optional ``rk_table_diagnostics.csv``,
then evaluates the six validation criteria documented by the RK runner README:

1. At the largest tested P for each (run tag, temperature), |r^2 error| <= 2%.
2. |<r> error|, |<r^2> error|, |<V_RK> error| and radial-PDF L1 do not
   increase as P increases.
3. Seed SEM is smaller than the remaining r^2 finite-P bias.  The ratio
   SEM/|bias| is calculated at every point. Ratios above the configured limit
   are labelled WEAK rather than silently accepted.
4. Wall energy, probability beyond the wall, histogram loss and reference-edge
   weight are negligible.
5. The RK table maximum relative interpolation error is below 1e-3.
6. The primitive short-distance exponent tau*A is below 2 at every point.
   Values below 1 are labelled STRONG; 1 <= tau*A < 2 are labelled OK;
   tau*A >= 2 are INVALID.

Typical use from the project root:

    python3 scripts/check_relative_rk_convergence.py \
      --results_glob 'results/relative_rk_*' \
      --outdir results/relative_rk_convergence_check

The script writes combined point data, per-temperature validation, per-tag
summaries, JSON and a human-readable report.  It returns exit status 2 when a
hard validation criterion fails.  Warnings return zero unless --fail_on_warn
is supplied.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


VERSION = "v1.0-rk-convergence-checker"
CONVERGENCE_FILENAME = "relative_rk_convergence.csv"
TABLE_DIAGNOSTICS_FILENAME = "rk_table_diagnostics.csv"


# ---------------------------------------------------------------------------
# Project paths and input discovery
# ---------------------------------------------------------------------------


def find_project_root() -> Path:
    env_root = os.environ.get("TMD_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    candidates = [Path.cwd().resolve(), *here.parents]
    for candidate in candidates:
        if (candidate / "results").exists() or (candidate / "code" / "tmd_pimc").exists():
            return candidate
    return Path.cwd().resolve()


ROOT = find_project_root()


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def discover_run_inputs(
    results_dirs: list[str] | None,
    results_glob: str,
    convergence_filename: str,
) -> list[tuple[Path, Path]]:
    """Return unique ``(run_dir, convergence_csv)`` pairs.

    A supplied path/glob match may be either a result directory or the
    convergence CSV itself.  Directories are expected to contain the CSV at
    their top level, matching the RK runner output layout.
    """
    candidates: list[Path] = []
    if results_dirs:
        candidates = [resolve_project_path(value) for value in results_dirs]
    else:
        pattern = str(resolve_project_path(results_glob))
        candidates = [Path(value).resolve() for value in sorted(glob.glob(pattern))]

    found: dict[Path, tuple[Path, Path]] = {}
    for candidate in candidates:
        if candidate.is_file():
            if candidate.name != convergence_filename:
                print(
                    f"WARNING: ignoring file that is not {convergence_filename}: {candidate}",
                    file=sys.stderr,
                )
                continue
            run_dir = candidate.parent
            csv_path = candidate
        elif candidate.is_dir():
            run_dir = candidate
            csv_path = run_dir / convergence_filename
            if not csv_path.is_file():
                print(
                    f"WARNING: result directory has no {convergence_filename}: {run_dir}",
                    file=sys.stderr,
                )
                continue
        else:
            print(f"WARNING: input does not exist: {candidate}", file=sys.stderr)
            continue

        found[csv_path.resolve()] = (run_dir.resolve(), csv_path.resolve())

    return [found[key] for key in sorted(found, key=str)]


def make_unique_tags(run_dirs: Iterable[Path], tag_mode: str) -> dict[Path, str]:
    base_tags: dict[Path, str] = {}
    for run_dir in run_dirs:
        if tag_mode == "basename":
            base = run_dir.name
        elif tag_mode == "relative":
            try:
                base = str(run_dir.relative_to(ROOT))
            except ValueError:
                base = str(run_dir)
        else:
            raise ValueError(f"Unknown tag_mode={tag_mode!r}")
        base_tags[run_dir] = base

    counts: dict[str, int] = {}
    tags: dict[Path, str] = {}
    for run_dir in sorted(base_tags, key=str):
        base = base_tags[run_dir]
        counts[base] = counts.get(base, 0) + 1
        tags[run_dir] = base if counts[base] == 1 else f"{base}__{counts[base]}"
    return tags


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Thresholds:
    r2_percent: float
    table_max_relative_error: float
    sem_bias_max_ratio: float
    sem_bias_strong_ratio: float
    wall_max_energy_meV: float
    wall_max_probability: float
    hist_lost_max: float
    reference_edge_weight_max: float
    tau_strong_max: float
    tau_invalid_min: float
    monotonic_rtol: float
    monotonic_atol: float


REQUIRED_COLUMNS = {
    "temperature_K",
    "n_beads",
    "mean_r_pimc_nm",
    "mean_r_reference_nm",
    "mean_r_error_percent",
    "mean_r2_pimc_nm2",
    "mean_r2_seed_sem_nm2",
    "mean_r2_reference_nm2",
    "mean_r2_error_percent",
    "mean_V_rk_pimc_eV",
    "mean_V_rk_reference_eV",
    "mean_V_rk_error_meV",
    "mean_V_wall_pimc_eV",
    "mean_V_wall_reference_eV",
    "radial_pdf_L1_distance",
    "primitive_short_distance_exponent",
    "fraction_beyond_wall_radius_max",
    "hist_lost_fraction_max",
    "reference_edge_weight_fraction",
}


MONOTONIC_METRICS: tuple[tuple[str, str], ...] = (
    ("abs_mean_r_error_percent", "|<r> error| [%]"),
    ("abs_mean_r2_error_percent", "|<r^2> error| [%]"),
    ("abs_mean_V_rk_error_meV", "|<V_RK> error| [meV]"),
    ("radial_pdf_L1_distance", "radial PDF L1"),
)


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def bool_status(value: bool) -> str:
    return "PASS" if bool(value) else "FAIL"


def tau_classification(alpha: float, thresholds: Thresholds) -> str:
    if not math.isfinite(alpha):
        return "MISSING"
    if alpha >= thresholds.tau_invalid_min:
        return "INVALID"
    if alpha < thresholds.tau_strong_max:
        return "STRONG"
    return "OK"


def sem_bias_classification(ratio: float, thresholds: Thresholds) -> str:
    if not math.isfinite(ratio):
        return "WEAK"
    if ratio <= thresholds.sem_bias_strong_ratio:
        return "STRONG"
    if ratio <= thresholds.sem_bias_max_ratio:
        return "PASS"
    return "WEAK"


def is_non_increasing(previous: float, current: float, thresholds: Thresholds) -> bool:
    if not (math.isfinite(previous) and math.isfinite(current)):
        return False
    allowed = previous * (1.0 + thresholds.monotonic_rtol) + thresholds.monotonic_atol
    return current <= allowed


def combine_statuses(statuses: Iterable[str], *, not_tested_is_warn: bool = True) -> str:
    values = list(statuses)
    if any(value in {"FAIL", "INVALID", "MISSING_FAIL"} for value in values):
        return "FAIL"
    warning_values = {"WARN", "WEAK", "MISSING", "NOT_TESTED"}
    if not_tested_is_warn and any(value in warning_values for value in values):
        return "WARN"
    return "PASS"


def validate_and_enrich_points(df: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("Convergence CSV is missing required columns: " + ", ".join(missing))

    out = df.copy()
    numeric_columns = sorted(REQUIRED_COLUMNS - {"temperature_K", "n_beads"})
    for column in ["temperature_K", "n_beads", *numeric_columns]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["abs_mean_r_error_percent"] = out["mean_r_error_percent"].abs()
    out["abs_mean_r2_error_percent"] = out["mean_r2_error_percent"].abs()
    out["abs_mean_V_rk_error_meV"] = out["mean_V_rk_error_meV"].abs()
    out["r2_bias_nm2"] = (out["mean_r2_pimc_nm2"] - out["mean_r2_reference_nm2"]).abs()

    bias = out["r2_bias_nm2"].to_numpy(dtype=float)
    sem = out["mean_r2_seed_sem_nm2"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            sem,
            bias,
            out=np.full_like(sem, np.inf, dtype=float),
            where=bias > np.finfo(float).eps,
        )
    ratio[(bias <= np.finfo(float).eps) & (sem <= np.finfo(float).eps)] = 0.0
    out["sem_to_r2_bias_ratio"] = ratio
    out["sem_bias_status"] = [sem_bias_classification(value, thresholds) for value in ratio]

    out["within_r2_threshold"] = out["abs_mean_r2_error_percent"] <= thresholds.r2_percent
    out["tauA_status"] = [
        tau_classification(value, thresholds)
        for value in out["primitive_short_distance_exponent"].to_numpy(dtype=float)
    ]

    wall_energy_meV = np.maximum(
        out["mean_V_wall_pimc_eV"].abs().to_numpy(dtype=float),
        out["mean_V_wall_reference_eV"].abs().to_numpy(dtype=float),
    ) * 1000.0
    out["wall_energy_max_abs_meV"] = wall_energy_meV
    out["wall_energy_ok"] = wall_energy_meV <= thresholds.wall_max_energy_meV
    out["wall_probability_ok"] = (
        out["fraction_beyond_wall_radius_max"].abs() <= thresholds.wall_max_probability
    )
    out["histogram_loss_ok"] = out["hist_lost_fraction_max"].abs() <= thresholds.hist_lost_max
    out["reference_edge_ok"] = (
        out["reference_edge_weight_fraction"].abs() <= thresholds.reference_edge_weight_max
    )
    out["wall_effects_status"] = [
        bool_status(a and b and c and d)
        for a, b, c, d in zip(
            out["wall_energy_ok"],
            out["wall_probability_ok"],
            out["histogram_loss_ok"],
            out["reference_edge_ok"],
        )
    ]
    return out


def load_table_diagnostics(
    run_dir: Path,
    thresholds: Thresholds,
    allow_missing: bool,
) -> dict[str, Any]:
    path = run_dir / TABLE_DIAGNOSTICS_FILENAME
    if not path.is_file():
        return {
            "table_diagnostics_path": str(path),
            "table_diagnostics_present": False,
            "table_max_relative_interpolation_error": math.nan,
            "table_gate_status": "MISSING" if allow_missing else "MISSING_FAIL",
            "table_gate_detail": "diagnostics file not found",
        }

    diagnostics = pd.read_csv(path)
    if diagnostics.empty or "max_relative_interpolation_error" not in diagnostics.columns:
        return {
            "table_diagnostics_path": str(path),
            "table_diagnostics_present": True,
            "table_max_relative_interpolation_error": math.nan,
            "table_gate_status": "MISSING" if allow_missing else "MISSING_FAIL",
            "table_gate_detail": "max_relative_interpolation_error column missing",
        }

    max_error = float(pd.to_numeric(
        diagnostics["max_relative_interpolation_error"], errors="coerce"
    ).max())
    passed = math.isfinite(max_error) and max_error < thresholds.table_max_relative_error
    return {
        "table_diagnostics_path": str(path),
        "table_diagnostics_present": True,
        "table_max_relative_interpolation_error": max_error,
        "table_gate_status": "PASS" if passed else "FAIL",
        "table_gate_detail": (
            f"max relative interpolation error={max_error:.6g}; "
            f"required < {thresholds.table_max_relative_error:.6g}"
        ),
    }


# ---------------------------------------------------------------------------
# Per-(tag, temperature) validation
# ---------------------------------------------------------------------------


def monotonic_group_check(group: pd.DataFrame, thresholds: Thresholds) -> tuple[str, str, dict[str, str]]:
    ordered = group.sort_values("n_beads").reset_index(drop=True)
    if ordered.shape[0] < 2:
        details = {column: "NOT_TESTED: only one P value" for column, _ in MONOTONIC_METRICS}
        return "NOT_TESTED", "only one P value", details

    metric_details: dict[str, str] = {}
    all_ok = True
    for column, label in MONOTONIC_METRICS:
        values = ordered[column].to_numpy(dtype=float)
        p_values = ordered["n_beads"].to_numpy(dtype=int)
        checks: list[str] = []
        metric_ok = True
        for index in range(1, len(values)):
            ok = is_non_increasing(values[index - 1], values[index], thresholds)
            metric_ok &= ok
            direction = "OK" if ok else "REVERSAL"
            checks.append(
                f"P={p_values[index-1]}:{values[index-1]:.6g} -> "
                f"P={p_values[index]}:{values[index]:.6g} [{direction}]"
            )
        metric_details[column] = f"{label}: " + "; ".join(checks)
        all_ok &= metric_ok

    return (
        "PASS" if all_ok else "FAIL",
        "all monitored errors are non-increasing" if all_ok else "one or more errors increase with P",
        metric_details,
    )


def evaluate_temperature_group(
    tag: str,
    run_dir: Path,
    group: pd.DataFrame,
    table_info: dict[str, Any],
    thresholds: Thresholds,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = group.sort_values("n_beads").reset_index(drop=True)
    highest = ordered.iloc[-1]
    highest_p = int(highest["n_beads"])

    criterion_1 = "PASS" if bool(highest["within_r2_threshold"]) else "FAIL"
    monotonic_status, monotonic_summary, metric_details = monotonic_group_check(ordered, thresholds)

    sem_statuses = ordered["sem_bias_status"].astype(str).tolist()
    criterion_3 = "PASS" if all(value in {"PASS", "STRONG"} for value in sem_statuses) else "WEAK"
    max_sem_ratio = float(ordered["sem_to_r2_bias_ratio"].max())

    criterion_4 = "PASS" if (ordered["wall_effects_status"] == "PASS").all() else "FAIL"
    criterion_5 = str(table_info["table_gate_status"])
    tau_statuses = ordered["tauA_status"].astype(str).tolist()
    criterion_6 = "PASS" if all(value in {"STRONG", "OK"} for value in tau_statuses) else "FAIL"

    overall = combine_statuses(
        [criterion_1, monotonic_status, criterion_3, criterion_4, criterion_5, criterion_6]
    )

    row: dict[str, Any] = {
        "tag": tag,
        "run_dir": str(run_dir),
        "temperature_K": float(highest["temperature_K"]),
        "n_P_values": int(ordered.shape[0]),
        "P_values": ",".join(str(int(value)) for value in ordered["n_beads"]),
        "largest_P": highest_p,
        "largest_P_abs_r2_error_percent": float(highest["abs_mean_r2_error_percent"]),
        "largest_P_within_r2_threshold": bool(highest["within_r2_threshold"]),
        "largest_P_r2_seed_sem_nm2": float(highest["mean_r2_seed_sem_nm2"]),
        "largest_P_r2_bias_nm2": float(highest["r2_bias_nm2"]),
        "largest_P_sem_to_bias_ratio": float(highest["sem_to_r2_bias_ratio"]),
        "max_sem_to_bias_ratio_over_P": max_sem_ratio,
        "largest_P_tauA": float(highest["primitive_short_distance_exponent"]),
        "max_tauA_over_P": float(ordered["primitive_short_distance_exponent"].max()),
        "max_wall_energy_abs_meV_over_P": float(ordered["wall_energy_max_abs_meV"].max()),
        "max_fraction_beyond_wall_over_P": float(ordered["fraction_beyond_wall_radius_max"].abs().max()),
        "max_hist_lost_fraction_over_P": float(ordered["hist_lost_fraction_max"].abs().max()),
        "max_reference_edge_weight_over_P": float(ordered["reference_edge_weight_fraction"].abs().max()),
        "table_max_relative_interpolation_error": table_info["table_max_relative_interpolation_error"],
        "criterion_1_largest_P_r2_within_threshold": criterion_1,
        "criterion_2_monotonic_improvement": monotonic_status,
        "criterion_3_sem_smaller_than_bias": criterion_3,
        "criterion_4_wall_effects_negligible": criterion_4,
        "criterion_5_table_interpolation": criterion_5,
        "criterion_6_tauA_below_2": criterion_6,
        "overall_status": overall,
        "monotonic_summary": monotonic_summary,
        "table_gate_detail": table_info["table_gate_detail"],
    }
    row.update({f"monotonic_{key}_detail": value for key, value in metric_details.items()})

    point_rows: list[dict[str, Any]] = []
    for _, point in ordered.iterrows():
        point_rows.append(
            {
                "tag": tag,
                "run_dir": str(run_dir),
                "temperature_K": float(point["temperature_K"]),
                "n_beads": int(point["n_beads"]),
                "is_largest_P_for_tag_T": int(int(point["n_beads"]) == highest_p),
                "abs_mean_r_error_percent": float(point["abs_mean_r_error_percent"]),
                "abs_mean_r2_error_percent": float(point["abs_mean_r2_error_percent"]),
                "abs_mean_V_rk_error_meV": float(point["abs_mean_V_rk_error_meV"]),
                "radial_pdf_L1_distance": float(point["radial_pdf_L1_distance"]),
                "mean_r2_seed_sem_nm2": float(point["mean_r2_seed_sem_nm2"]),
                "r2_bias_nm2": float(point["r2_bias_nm2"]),
                "sem_to_r2_bias_ratio": float(point["sem_to_r2_bias_ratio"]),
                "sem_bias_status": str(point["sem_bias_status"]),
                "within_r2_threshold": bool(point["within_r2_threshold"]),
                "primitive_short_distance_exponent": float(point["primitive_short_distance_exponent"]),
                "tauA_status": str(point["tauA_status"]),
                "wall_energy_max_abs_meV": float(point["wall_energy_max_abs_meV"]),
                "fraction_beyond_wall_radius_max": float(point["fraction_beyond_wall_radius_max"]),
                "hist_lost_fraction_max": float(point["hist_lost_fraction_max"]),
                "reference_edge_weight_fraction": float(point["reference_edge_weight_fraction"]),
                "wall_effects_status": str(point["wall_effects_status"]),
                "table_gate_status": criterion_5,
            }
        )

    return row, point_rows


def summarise_tags(group_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tag, group in group_summary.groupby("tag", sort=True):
        statuses = group["overall_status"].astype(str).tolist()
        rows.append(
            {
                "tag": tag,
                "run_dir": str(group["run_dir"].iloc[0]),
                "n_temperatures": int(group.shape[0]),
                "temperatures_K": ",".join(f"{value:g}" for value in sorted(group["temperature_K"])),
                "n_pass": int(sum(value == "PASS" for value in statuses)),
                "n_warn": int(sum(value == "WARN" for value in statuses)),
                "n_fail": int(sum(value == "FAIL" for value in statuses)),
                "tag_status": combine_statuses(statuses),
            }
        )
    return pd.DataFrame(rows).sort_values("tag").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_text_report(
    path: Path,
    inputs: list[tuple[Path, Path]],
    point_checks: pd.DataFrame,
    group_summary: pd.DataFrame,
    tag_summary: pd.DataFrame,
    thresholds: Thresholds,
) -> None:
    lines: list[str] = []
    lines.append("Relative-coordinate RK convergence validation")
    lines.append("=" * 88)
    lines.append(f"Checker version: {VERSION}")
    lines.append(f"Input run directories: {len(inputs)}")
    for run_dir, csv_path in inputs:
        lines.append(f"  - {run_dir}  [{csv_path.name}]")
    lines.append("")
    lines.append("Thresholds")
    lines.append("-" * 88)
    lines.append(f"Largest-P |r^2 error|            <= {thresholds.r2_percent:g}%")
    lines.append(f"Table max relative error          < {thresholds.table_max_relative_error:g}")
    lines.append(f"SEM / remaining r^2 bias          <= {thresholds.sem_bias_max_ratio:g} (strong <= {thresholds.sem_bias_strong_ratio:g})")
    lines.append(f"Max |wall energy|                 <= {thresholds.wall_max_energy_meV:g} meV")
    lines.append(f"Probability beyond wall           <= {thresholds.wall_max_probability:g}")
    lines.append(f"Histogram lost fraction           <= {thresholds.hist_lost_max:g}")
    lines.append(f"Reference edge weight             <= {thresholds.reference_edge_weight_max:g}")
    lines.append(f"tau*A classification              STRONG < {thresholds.tau_strong_max:g}; OK < {thresholds.tau_invalid_min:g}; INVALID >= {thresholds.tau_invalid_min:g}")
    lines.append("")

    lines.append("Per-tag summary")
    lines.append("-" * 88)
    if tag_summary.empty:
        lines.append("No tags evaluated.")
    else:
        lines.append(tag_summary.to_string(index=False))
    lines.append("")

    lines.append("Per-(tag, temperature) six-criterion validation")
    lines.append("-" * 88)
    display_columns = [
        "tag",
        "temperature_K",
        "P_values",
        "largest_P_abs_r2_error_percent",
        "largest_P_sem_to_bias_ratio",
        "criterion_1_largest_P_r2_within_threshold",
        "criterion_2_monotonic_improvement",
        "criterion_3_sem_smaller_than_bias",
        "criterion_4_wall_effects_negligible",
        "criterion_5_table_interpolation",
        "criterion_6_tauA_below_2",
        "overall_status",
    ]
    lines.append(group_summary[display_columns].to_string(index=False))
    lines.append("")

    lines.append("Detailed findings")
    lines.append("-" * 88)
    for _, row in group_summary.iterrows():
        lines.append(
            f"[{row['overall_status']}] tag={row['tag']} T={row['temperature_K']:g} K "
            f"P=[{row['P_values']}]"
        )
        lines.append(
            f"  Largest P={int(row['largest_P'])}: |r2 error|="
            f"{row['largest_P_abs_r2_error_percent']:.6g}%, "
            f"SEM/bias={row['largest_P_sem_to_bias_ratio']:.6g}, "
            f"tauA={row['largest_P_tauA']:.6g}"
        )
        lines.append(f"  Monotonic: {row['monotonic_summary']}")
        for metric, _ in MONOTONIC_METRICS:
            column = f"monotonic_{metric}_detail"
            if column in row and pd.notna(row[column]):
                lines.append(f"    {row[column]}")
        lines.append(f"  Table: {row['table_gate_detail']}")
        if row["criterion_3_sem_smaller_than_bias"] == "WEAK":
            weak_points = point_checks[
                (point_checks["tag"] == row["tag"])
                & (point_checks["temperature_K"] == row["temperature_K"])
                & (point_checks["sem_bias_status"] == "WEAK")
            ]
            for _, point in weak_points.iterrows():
                lines.append(
                    f"  WEAK signal at P={int(point['n_beads'])}: "
                    f"SEM/bias={point['sem_to_r2_bias_ratio']:.6g}"
                )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = None if not np.isfinite(value) else float(value)
            elif isinstance(value, float) and not math.isfinite(value):
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and validate relative_rk_convergence.csv files across RK runs."
    )
    parser.add_argument(
        "--results_dirs",
        nargs="+",
        default=None,
        help="Explicit result directories or relative_rk_convergence.csv paths.",
    )
    parser.add_argument(
        "--results_glob",
        default="results/relative_rk_*",
        help="Glob selecting result directories or convergence CSV files.",
    )
    parser.add_argument(
        "--outdir",
        default="results/relative_rk_convergence_check",
    )
    parser.add_argument(
        "--convergence_filename",
        default=CONVERGENCE_FILENAME,
    )
    parser.add_argument(
        "--tag_mode",
        choices=["basename", "relative"],
        default="basename",
        help="Use each result directory basename or project-relative path as the run tag.",
    )

    parser.add_argument("--r2_threshold_percent", type=float, default=2.0)
    parser.add_argument("--table_max_relative_error", type=float, default=1.0e-3)
    parser.add_argument("--sem_bias_max_ratio", type=float, default=1.0)
    parser.add_argument("--sem_bias_strong_ratio", type=float, default=0.5)
    parser.add_argument("--wall_max_energy_meV", type=float, default=0.01)
    parser.add_argument("--wall_max_probability", type=float, default=1.0e-6)
    parser.add_argument("--hist_lost_max", type=float, default=1.0e-6)
    parser.add_argument("--reference_edge_weight_max", type=float, default=1.0e-6)
    parser.add_argument("--tau_strong_max", type=float, default=1.0)
    parser.add_argument("--tau_invalid_min", type=float, default=2.0)
    parser.add_argument(
        "--monotonic_rtol",
        type=float,
        default=0.0,
        help="Relative tolerance allowed in non-increasing error checks.",
    )
    parser.add_argument(
        "--monotonic_atol",
        type=float,
        default=1.0e-12,
        help="Absolute tolerance allowed in non-increasing error checks.",
    )
    parser.add_argument(
        "--allow_missing_table_diagnostics",
        action="store_true",
        help="Downgrade a missing rk_table_diagnostics.csv from FAIL to WARN.",
    )
    parser.add_argument(
        "--fail_on_warn",
        action="store_true",
        help="Return exit status 2 for WARN as well as FAIL.",
    )
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> Thresholds:
    nonnegative = {
        "r2_threshold_percent": args.r2_threshold_percent,
        "table_max_relative_error": args.table_max_relative_error,
        "sem_bias_max_ratio": args.sem_bias_max_ratio,
        "sem_bias_strong_ratio": args.sem_bias_strong_ratio,
        "wall_max_energy_meV": args.wall_max_energy_meV,
        "wall_max_probability": args.wall_max_probability,
        "hist_lost_max": args.hist_lost_max,
        "reference_edge_weight_max": args.reference_edge_weight_max,
        "tau_strong_max": args.tau_strong_max,
        "tau_invalid_min": args.tau_invalid_min,
        "monotonic_rtol": args.monotonic_rtol,
        "monotonic_atol": args.monotonic_atol,
    }
    for name, value in nonnegative.items():
        if value < 0.0:
            raise ValueError(f"--{name} must be non-negative")
    if args.sem_bias_strong_ratio > args.sem_bias_max_ratio:
        raise ValueError("--sem_bias_strong_ratio must be <= --sem_bias_max_ratio")
    if args.tau_strong_max >= args.tau_invalid_min:
        raise ValueError("--tau_strong_max must be < --tau_invalid_min")

    return Thresholds(
        r2_percent=float(args.r2_threshold_percent),
        table_max_relative_error=float(args.table_max_relative_error),
        sem_bias_max_ratio=float(args.sem_bias_max_ratio),
        sem_bias_strong_ratio=float(args.sem_bias_strong_ratio),
        wall_max_energy_meV=float(args.wall_max_energy_meV),
        wall_max_probability=float(args.wall_max_probability),
        hist_lost_max=float(args.hist_lost_max),
        reference_edge_weight_max=float(args.reference_edge_weight_max),
        tau_strong_max=float(args.tau_strong_max),
        tau_invalid_min=float(args.tau_invalid_min),
        monotonic_rtol=float(args.monotonic_rtol),
        monotonic_atol=float(args.monotonic_atol),
    )


def main() -> int:
    args = parse_args()
    thresholds = validate_cli(args)

    inputs = discover_run_inputs(
        args.results_dirs,
        args.results_glob,
        args.convergence_filename,
    )
    if not inputs:
        raise FileNotFoundError(
            "No valid RK result directories found. Use --results_dirs or --results_glob."
        )

    tags = make_unique_tags((run_dir for run_dir, _ in inputs), args.tag_mode)
    combined_frames: list[pd.DataFrame] = []
    group_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    for run_dir, convergence_path in inputs:
        tag = tags[run_dir]
        raw = pd.read_csv(convergence_path)
        enriched = validate_and_enrich_points(raw, thresholds)
        enriched.insert(0, "tag", tag)
        enriched.insert(1, "run_dir", str(run_dir))
        enriched.insert(2, "convergence_csv", str(convergence_path))
        combined_frames.append(enriched)

        table_info = load_table_diagnostics(
            run_dir,
            thresholds,
            allow_missing=args.allow_missing_table_diagnostics,
        )
        for temperature, group in enriched.groupby("temperature_K", sort=True, dropna=False):
            if not math.isfinite(float(temperature)):
                raise ValueError(f"Non-finite temperature in {convergence_path}")
            summary, points = evaluate_temperature_group(
                tag,
                run_dir,
                group,
                table_info,
                thresholds,
            )
            group_rows.append(summary)
            point_rows.extend(points)

    combined = pd.concat(combined_frames, ignore_index=True).sort_values(
        ["tag", "temperature_K", "n_beads"]
    )
    point_checks = pd.DataFrame(point_rows).sort_values(
        ["tag", "temperature_K", "n_beads"]
    )
    group_summary = pd.DataFrame(group_rows).sort_values(
        ["tag", "temperature_K"]
    ).reset_index(drop=True)
    tag_summary = summarise_tags(group_summary)

    outdir = resolve_project_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(outdir / "combined_relative_rk_convergence.csv", index=False)
    point_checks.to_csv(outdir / "rk_point_checks.csv", index=False)
    group_summary.to_csv(outdir / "rk_validation_by_tag_temperature.csv", index=False)
    tag_summary.to_csv(outdir / "rk_validation_by_tag.csv", index=False)
    write_text_report(
        outdir / "rk_convergence_validation_report.txt",
        inputs,
        point_checks,
        group_summary,
        tag_summary,
        thresholds,
    )

    overall_status = combine_statuses(tag_summary["tag_status"].astype(str).tolist())
    payload = {
        "version": VERSION,
        "project_root": str(ROOT),
        "overall_status": overall_status,
        "thresholds": thresholds.__dict__,
        "input_runs": [
            {"tag": tags[run_dir], "run_dir": str(run_dir), "convergence_csv": str(csv_path)}
            for run_dir, csv_path in inputs
        ],
        "tag_summary": dataframe_records(tag_summary),
        "temperature_summary": dataframe_records(group_summary),
    }
    (outdir / "rk_convergence_validation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=" * 88)
    print("Relative-coordinate RK convergence validation completed")
    print("=" * 88)
    print(f"Input runs       : {len(inputs)}")
    print(f"Temperature groups: {group_summary.shape[0]}")
    print(f"Overall status   : {overall_status}")
    print(f"Output directory : {outdir}")
    print()
    display_columns = [
        "tag",
        "temperature_K",
        "P_values",
        "largest_P_abs_r2_error_percent",
        "largest_P_sem_to_bias_ratio",
        "criterion_1_largest_P_r2_within_threshold",
        "criterion_2_monotonic_improvement",
        "criterion_3_sem_smaller_than_bias",
        "criterion_4_wall_effects_negligible",
        "criterion_5_table_interpolation",
        "criterion_6_tauA_below_2",
        "overall_status",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 220, "display.max_columns", None):
        print(group_summary[display_columns].to_string(index=False))
    print()
    print(f"Report            : {outdir / 'rk_convergence_validation_report.txt'}")
    print(f"Group summary     : {outdir / 'rk_validation_by_tag_temperature.csv'}")
    print(f"Point checks      : {outdir / 'rk_point_checks.csv'}")

    has_fail = (tag_summary["tag_status"] == "FAIL").any()
    has_warn = (tag_summary["tag_status"] == "WARN").any()
    if has_fail or (args.fail_on_warn and has_warn):
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(3)
