# ============================================================================
# Reporting
# ============================================================================
from __future__ import annotations

import csv
import json

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .validation import (
    ValidationCase,
    ValidationStatus,
)


def _serialize(obj: Any) -> Any:
    """
    Recursively convert dataclasses/Enums/Paths into JSON-safe types.
    """

    if isinstance(obj, Enum):
        return str(obj)

    if isinstance(obj, Path):
        return str(obj)

    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _serialize(getattr(obj, f.name)) for f in fields(obj)}

    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]

    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}

    return obj


def validation_to_dict(case: ValidationCase) -> dict[str, Any]:
    """
    Convert a single validation case into a JSON-safe dict.
    """

    return _serialize(case)


def campaign_to_dicts(cases: Iterable[ValidationCase]) -> list[dict[str, Any]]:
    """
    Convert a validation campaign into a list of JSON-safe dicts.
    """

    return [validation_to_dict(case) for case in cases]


def _flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """
    Flatten a nested dict into a single-level dict suitable for CSV rows.
    Lists are joined into a single ';'-separated string.
    """

    items: dict[str, Any] = {}

    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.update(_flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):
            items[new_key] = ";".join(str(v) for v in value)
        else:
            items[new_key] = value

    return items


def write_json(
    cases: Iterable[ValidationCase],
    output_directory: str | Path,
    *,
    filename: str = "validation.json",
    indent: int = 4,
) -> Path:
    """
    Write an entire validation campaign to a JSON file.

    Parameters
    ----------
    cases
        Validation cases.

    output_directory
        Output directory.

    filename
        Output JSON filename.

    indent
        JSON indentation.

    Returns
    -------
    Path
        Path to the generated JSON file.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / filename

    data = campaign_to_dicts(cases)

    output_file.write_text(
        json.dumps(
            data,
            indent=indent,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_file


def write_csv(
    cases: Iterable[ValidationCase],
    output_directory: str | Path,
    *,
    filename: str = "validation.csv",
) -> Path:
    """
    Write a validation campaign to CSV.

    Parameters
    ----------
    cases
        Validation cases.

    output_directory
        Output directory.

    filename
        CSV filename.

    Returns
    -------
    Path
        Generated CSV path.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / filename

    rows = [
        _flatten_dict(validation_to_dict(case))
        for case in cases
    ]

    if not rows:
        raise ValueError("No validation cases supplied.")

    fieldnames = sorted(rows[0].keys())

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    return output_file
    
# ============================================================================
# Summary helpers
# ============================================================================


def _count_statuses(
    cases: Iterable[ValidationCase],
) -> dict[str, int]:
    """
    Count validation outcomes.
    """

    counts = {
        "PASS": 0,
        "WARN": 0,
        "FAIL": 0,
    }

    for case in cases:
        counts[str(case.result.status)] += 1

    return counts


def overall_status(
    cases: Iterable[ValidationCase],
) -> ValidationStatus:
    """
    Determine the overall campaign status.
    """

    statuses = [
        case.result.status
        for case in cases
    ]

    if not statuses:
        raise ValueError("No validation cases supplied.")

    if any(
        status is ValidationStatus.FAIL
        for status in statuses
    ):
        return ValidationStatus.FAIL

    if any(
        status is ValidationStatus.WARN
        for status in statuses
    ):
        return ValidationStatus.WARN

    return ValidationStatus.PASS


def _format_float(
    value: float,
    digits: int = 6,
) -> str:
    """
    Format floating-point values.
    """

    return f"{value:.{digits}f}"


def _format_percent(
    value: float,
    digits: int = 2,
) -> str:
    """
    Format percentage values.
    """

    return f"{value:.{digits}f} %"
    
def _case_to_text(
    case: ValidationCase,
) -> str:
    """
    Convert a validation case into a human-readable report.
    """

    run = case.run
    metrics = case.metrics
    result = case.result

    lines: list[str] = []

    lines.append("=" * 72)
    lines.append("Validation Case")
    lines.append("=" * 72)
    lines.append("")

    lines.append(f"Potential      : {run.potential}")
    lines.append(f"Reference      : {run.reference}")
    lines.append(f"Sampler        : {run.sampler}")
    lines.append(
        f"Temperature    : {_format_float(run.temperature_K,2)} K"
    )
    lines.append(f"Beads          : {run.n_beads}")
    lines.append("")

    lines.append(f"Status         : {result.summary}")
    lines.append("")

    lines.append("Metrics")
    lines.append("-" * 72)

    lines.append(
        f"Δ<r>           : {_format_percent(metrics.delta_mean_r_percent)}"
    )

    lines.append(
        f"Δ<r²>          : {_format_percent(metrics.delta_mean_r2_percent)}"
    )

    lines.append(
        f"Δ<V>           : {_format_percent(metrics.delta_mean_v_percent)}"
    )

    lines.append(
        f"PDF L1         : {_format_float(metrics.pdf_l1)}"
    )

    lines.append(
        f"Wall probability : {_format_float(metrics.wall_probability)}"
    )

    lines.append(
        f"Histogram loss   : {_format_float(metrics.histogram_loss)}"
    )

    lines.append("")

    lines.append("Acceptance")
    lines.append("-" * 72)

    lines.append(
        f"Local          : {_format_float(metrics.acceptance_local,3)}"
    )

    lines.append(
        f"Staging        : {_format_float(metrics.acceptance_staging,3)}"
    )

    lines.append(
        f"Global         : {_format_float(metrics.acceptance_global,3)}"
    )

    lines.append("")

    return "\n".join(lines)
    
def write_txt(
    cases: Iterable[ValidationCase],
    output_directory: str | Path,
    *,
    filename: str = "validation.txt",
) -> Path:
    """
    Write a human-readable validation report.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_directory / filename

    cases = list(cases)

    counts = _count_statuses(cases)

    report: list[str] = []

    report.append("=" * 72)
    report.append("Validation Campaign")
    report.append("=" * 72)
    report.append("")

    report.append(f"Cases : {len(cases)}")
    report.append(f"PASS  : {counts['PASS']}")
    report.append(f"WARN  : {counts['WARN']}")
    report.append(f"FAIL  : {counts['FAIL']}")
    report.append("")
    report.append(
        f"Overall status : {overall_status(cases).value if cases else 'EMPTY'}"
    )
    report.append("")
    report.append("")

    for index, case in enumerate(cases, start=1):

        report.append(
            f"Case {index}"
        )

        report.append(
            _case_to_text(case)
        )

        report.append("")
        report.append("")

    output_file.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    return output_file
    
# ============================================================================
# Public reporting API
# ============================================================================


@dataclass(slots=True, frozen=True)
class ReportFiles:
    """
    Paths to generated report files.
    """

    json: Path
    csv: Path
    txt: Path
    
def print_summary(
    cases: Iterable[ValidationCase],
) -> None:
    """
    Print a concise validation summary.
    """

    cases = list(cases)

    counts = _count_statuses(cases)

    print("=" * 72)
    print("Validation summary")
    print("=" * 72)
    print(f"Cases          : {len(cases)}")
    print(f"PASS           : {counts['PASS']}")
    print(f"WARN           : {counts['WARN']}")
    print(f"FAIL           : {counts['FAIL']}")

    if cases:
        print(f"Overall status : {overall_status(cases).value}")
    else:
        print("Overall status : EMPTY")
    
def generate_validation_report(
    cases: Iterable[ValidationCase],
    output_directory: str | Path,
) -> ReportFiles:
    """
    Generate the complete validation report.

    The following files are created

        validation.json
        validation.csv
        validation.txt

    Parameters
    ----------
    cases
        Validation cases.

    output_directory
        Destination directory.

    Returns
    -------
    ReportFiles
        Paths to generated report files.
    """

    cases = list(cases)

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_file = write_json(
        cases,
        output_directory,
    )

    csv_file = write_csv(
        cases,
        output_directory,
    )

    txt_file = write_txt(
        cases,
        output_directory,
    )

    return ReportFiles(
        json=json_file,
        csv=csv_file,
        txt=txt_file,
    )
