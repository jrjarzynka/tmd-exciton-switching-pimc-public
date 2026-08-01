"""
validation.py
=============

Generic validation framework for TMD Path Integral Monte Carlo (PIMC)
simulations.

This module is intentionally independent of any particular interaction
potential (harmonic, Coulomb, Rytova-Keldysh, Bilayer Keldysh, etc.).
Its purpose is to evaluate simulation results against an independent
reference solution and classify the outcome using configurable validation
criteria.

The module contains no Monte Carlo code and no file I/O.

Main components
---------------
ValidationThresholds
    Numerical tolerances used during validation.

ValidationMetrics
    Collection of measured deviations between PIMC and the reference.

ValidationResult
    Final PASS/WARN/FAIL classification.

Functions
---------
evaluate_validation()
evaluate_all()
overall_status()

Notes
-----
This module intentionally performs no file I/O.
Reporting is implemented separately in reporting.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


__all__ = [
    "ValidationStatus",
    "ValidationReason",
    "ValidationThresholds",
    "ValidationRun",
    "ValidationMetrics",
    "ValidationResult",
    "ValidationCase",
    "evaluate_validation",
    "evaluate_all",
    "overall_status",
]


# ============================================================================
# Enums
# ============================================================================


class ValidationStatus(Enum):
    """
    Validation outcome.

    PASS
        All required criteria satisfied.

    WARN
        Minor deviations detected.

    FAIL
        Significant disagreement with the reference.
    """

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

    def __str__(self) -> str:
        return self.value


class ValidationReason(Enum):
    """
    Validation criteria identifiers.
    """

    DELTA_MEAN_R = "delta_mean_r"
    DELTA_MEAN_R2 = "delta_mean_r2"
    DELTA_MEAN_V = "delta_mean_v"

    PDF_L1 = "pdf_l1"

    WALL_PROBABILITY = "wall_probability"
    HISTOGRAM_LOSS = "histogram_loss"

    ACCEPTANCE_LOCAL = "acceptance_local"
    ACCEPTANCE_STAGING = "acceptance_staging"
    ACCEPTANCE_GLOBAL = "acceptance_global"

    def __str__(self) -> str:
        return self.value


# ============================================================================
# Data containers
# ============================================================================


@dataclass(slots=True)
class ValidationThresholds:
    """
    Numerical acceptance criteria.

    All percentage values are expressed in percent,
    e.g. 2.0 means ±2%.
    """

    mean_r_percent: float = 2.0
    mean_r2_percent: float = 2.0
    mean_v_percent: float = 5.0

    pdf_l1: float = 0.05

    wall_probability: float = 1.0e-4
    histogram_loss: float = 1.0e-4

    minimum_local_acceptance: float = 0.15
    minimum_staging_acceptance: float = 0.15
    minimum_global_acceptance: float = 0.01


FRAMEWORK_VERSION: str = "1.8b"


@dataclass(slots=True)
class ValidationRun:
    """
    Descriptive metadata for a single validation run.

    This class intentionally stores only descriptive information.
    Numerical validation metrics are stored separately in
    :class:`ValidationMetrics`.
    """

    temperature_K: float

    n_beads: int

    potential: str

    sampler: str

    reference: str = "radial"

    framework_version: str = FRAMEWORK_VERSION

    notes: str = ""

    random_seed: int | None = None

    n_samples: int | None = None

    output_directory: str | None = None


@dataclass(slots=True)
class ValidationMetrics:
    """
    Validation metrics for one simulation.

    Every quantity represents the comparison between
    one PIMC calculation and one reference solution.
    """

    delta_mean_r_percent: float

    delta_mean_r2_percent: float

    delta_mean_v_percent: float

    pdf_l1: float

    wall_probability: float

    histogram_loss: float

    acceptance_local: float

    acceptance_staging: float

    acceptance_global: float


@dataclass(slots=True)
class ValidationResult:
    """
    Result of a validation test.

    Carries its own :class:`ValidationMetrics` so that it remains a
    self-contained, inspectable object even when used outside of a
    full :class:`ValidationCase` (e.g. calling :func:`evaluate_validation`
    directly in a notebook or debugging script).
    """

    status: ValidationStatus

    reasons: list[ValidationReason] = field(default_factory=list)

    metrics: ValidationMetrics | None = None

    def passed(self) -> bool:
        return self.status is ValidationStatus.PASS

    def warning(self) -> bool:
        return self.status is ValidationStatus.WARN

    def failed(self) -> bool:
        return self.status is ValidationStatus.FAIL

    def __bool__(self) -> bool:
        """
        True only for ValidationStatus.PASS.

        Note: WARN is falsy here, same as FAIL. Use .passed() / .warning()
        / .failed() explicitly if you need to distinguish "did not pass
        cleanly" from "failed outright".
        """
        return self.passed()

    @property
    def summary(self) -> str:
        """
        Human-readable validation summary.

        Examples
        --------
        PASS
        WARN (delta_mean_r2)
        WARN (delta_mean_r2, pdf_l1)
        FAIL (wall_probability)
        """
        if not self.reasons:
            return str(self.status)

        return f"{self.status} ({', '.join(str(r) for r in self.reasons)})"


@dataclass(slots=True)
class ValidationCase:
    """
    Complete validation record.

    A ValidationCase combines

    * run metadata describing the calculation,
    * the final validation result (which itself carries the metrics
      that produced it).

    This is the primary object consumed by reporting.py.
    """

    run: ValidationRun

    result: ValidationResult

    @property
    def metrics(self) -> ValidationMetrics | None:
        """Metrics that produced this case's result (delegates to result)."""
        return self.result.metrics

    @property
    def status(self) -> ValidationStatus:
        return self.result.status

    @property
    def summary(self) -> str:
        return self.result.summary

    @property
    def passed(self) -> bool:
        return self.result.status is ValidationStatus.PASS

    @property
    def warning(self) -> bool:
        return self.result.status is ValidationStatus.WARN

    @property
    def failed(self) -> bool:
        return self.result.status is ValidationStatus.FAIL


# ============================================================================
# Private helper functions
# ============================================================================


def _append_reason(
    reasons: list[ValidationReason],
    reason: ValidationReason,
) -> None:
    """
    Append a validation reason only if it has not already been recorded.
    """
    if reason not in reasons:
        reasons.append(reason)


def _check_upper_limit(
    *,
    value: float,
    limit: float,
    reason: ValidationReason,
    reasons: list[ValidationReason],
) -> bool:
    """
    Check whether a metric is below its maximum allowed value.

    Parameters
    ----------
    value
        Measured metric.

    limit
        Maximum allowed value.

    reason
        Reason stored in ValidationResult.reasons if the test fails.

    reasons
        Mutable list of failure reasons.

    Returns
    -------
    bool
        True if the criterion is satisfied.
    """
    if abs(value) <= limit:
        return True

    _append_reason(reasons, reason)
    return False


def _check_lower_limit(
    *,
    value: float,
    limit: float,
    reason: ValidationReason,
    reasons: list[ValidationReason],
) -> bool:
    """
    Check whether a metric is above its minimum allowed value.
    """
    if value >= limit:
        return True

    _append_reason(reasons, reason)
    return False


def _status_from_reasons(
    reasons: Sequence[ValidationReason],
) -> ValidationStatus:
    """
    Determine the validation status.

    Current policy
    --------------
    PASS
        No violated criteria.

    WARN
        One or more violated criteria.

    FAIL
        Reserved for future use.
    """
    if not reasons:
        return ValidationStatus.PASS

    return ValidationStatus.WARN


# ============================================================================
# Public API
# ============================================================================


def evaluate_validation(
    metrics: ValidationMetrics,
    thresholds: ValidationThresholds | None = None,
) -> ValidationResult:
    """
    Evaluate one PIMC calculation against the validation criteria.

    Parameters
    ----------
    metrics
        Validation metrics.

    thresholds
        Validation thresholds. If None, default thresholds are used.

    Returns
    -------
    ValidationResult

    Notes
    -----
    # TODO: refactor to a declarative rule table once the number of
    # criteria grows (e.g. ESS, autocorrelation, blocking error). Nine
    # explicit checks is still fine; ~15+ will not be.
    """
    if thresholds is None:
        thresholds = ValidationThresholds()

    reasons: list[ValidationReason] = []

    _check_upper_limit(
        value=metrics.delta_mean_r_percent,
        limit=thresholds.mean_r_percent,
        reason=ValidationReason.DELTA_MEAN_R,
        reasons=reasons,
    )

    _check_upper_limit(
        value=metrics.delta_mean_r2_percent,
        limit=thresholds.mean_r2_percent,
        reason=ValidationReason.DELTA_MEAN_R2,
        reasons=reasons,
    )

    _check_upper_limit(
        value=metrics.delta_mean_v_percent,
        limit=thresholds.mean_v_percent,
        reason=ValidationReason.DELTA_MEAN_V,
        reasons=reasons,
    )

    _check_upper_limit(
        value=metrics.pdf_l1,
        limit=thresholds.pdf_l1,
        reason=ValidationReason.PDF_L1,
        reasons=reasons,
    )

    _check_upper_limit(
        value=metrics.wall_probability,
        limit=thresholds.wall_probability,
        reason=ValidationReason.WALL_PROBABILITY,
        reasons=reasons,
    )

    _check_upper_limit(
        value=metrics.histogram_loss,
        limit=thresholds.histogram_loss,
        reason=ValidationReason.HISTOGRAM_LOSS,
        reasons=reasons,
    )

    _check_lower_limit(
        value=metrics.acceptance_local,
        limit=thresholds.minimum_local_acceptance,
        reason=ValidationReason.ACCEPTANCE_LOCAL,
        reasons=reasons,
    )

    _check_lower_limit(
        value=metrics.acceptance_staging,
        limit=thresholds.minimum_staging_acceptance,
        reason=ValidationReason.ACCEPTANCE_STAGING,
        reasons=reasons,
    )

    _check_lower_limit(
        value=metrics.acceptance_global,
        limit=thresholds.minimum_global_acceptance,
        reason=ValidationReason.ACCEPTANCE_GLOBAL,
        reasons=reasons,
    )

    return ValidationResult(
        status=_status_from_reasons(reasons),
        reasons=reasons,
        metrics=metrics,
    )


def evaluate_all(
    metrics: Iterable[ValidationMetrics],
    thresholds: ValidationThresholds | None = None,
) -> list[ValidationResult]:
    """
    Evaluate multiple validation cases.
    """
    if thresholds is None:
        thresholds = ValidationThresholds()

    return [
        evaluate_validation(m, thresholds)
        for m in metrics
    ]


def overall_status(
    results: Iterable[ValidationResult],
) -> ValidationStatus:
    """
    Overall validation status for an entire validation campaign.
    """
    results = list(results)

    if not results:
        raise ValueError("No validation results supplied.")

    statuses = {r.status for r in results}

    if statuses == {ValidationStatus.PASS}:
        return ValidationStatus.PASS

    if ValidationStatus.FAIL in statuses:
        return ValidationStatus.FAIL

    return ValidationStatus.WARN
