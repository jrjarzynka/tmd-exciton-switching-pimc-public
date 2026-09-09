"""Preflight-only tests for the final Paper-1 production runner."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


RUNNER_PATH = (Path(__file__).resolve().parents[1] / "runners/production" /
               "run_paper1_final_staging_campaign.py")
SPEC = importlib.util.spec_from_file_location("paper1_final_staging_campaign", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_literal_inventory_is_complete_fresh_and_balanced():
    runner.validate_inventory()
    assert runner.expected_retained_count() == 30_000
    assert len(runner.TASK_INVENTORY) == 40
    assert len({task["seed"] for task in runner.TASK_INVENTORY}) == 40
    for row in runner.SCHEDULE:
        group = [task for task in runner.TASK_INVENTORY if task["T_K"] == row["T_K"]]
        assert len(group) == 10
        assert [task["start_basin"] for task in group] == ["A"] * 5 + ["B"] * 5
        assert all((task["P"], task["L"]) == (row["P"], row["L"]) for task in group)
        assert 2 <= row["L"] < row["P"]
    assert not any(runner._uses_forbidden_historical_seed(task["seed"])
                   for task in runner.TASK_INVENTORY)


def test_inventory_rejects_L_equal_to_P():
    schedule = tuple(dict(row) for row in runner.SCHEDULE)
    schedule[2]["L"] = schedule[2]["P"]
    inventory = runner.build_inventory(schedule=schedule)
    with pytest.raises(ValueError, match="2 <= L < P"):
        runner.validate_inventory(inventory=inventory, schedule=schedule)


def test_output_target_must_be_new_or_empty(tmp_path):
    new = tmp_path / "new"
    runner.validate_output_target(new)
    empty = tmp_path / "empty"
    empty.mkdir()
    runner.validate_output_target(empty)
    (empty / "unexpected.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        runner.validate_output_target(empty)


def test_grid_validation_checks_periodic_N400_contract(tmp_path):
    axis = np.arange(400, dtype=float) / 400
    grid_path = tmp_path / "grid.npz"
    np.savez(
        grid_path, u_fractional=axis, v_fractional=axis,
        V_eV=np.zeros((400, 400)), lattice_vectors_nm=np.eye(2), origin_nm=np.zeros(2),
    )
    digest = hashlib.sha256(grid_path.read_bytes()).hexdigest()
    grid = runner.validate_grid(grid_path, expected_sha256=digest)
    assert grid["V_eV"].shape == (400, 400)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runner.validate_grid(grid_path, expected_sha256="0" * 64)


def test_preflight_rejects_worker_count_outside_production_range(tmp_path):
    with pytest.raises(ValueError, match="4..8"):
        runner.preflight(tmp_path / "not-used.npz", tmp_path / "out", workers=3)
