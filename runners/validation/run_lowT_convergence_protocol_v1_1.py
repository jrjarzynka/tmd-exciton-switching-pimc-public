#!/usr/bin/env python3
"""
v1.1 — Low-temperature convergence protocol for atomistic-grid PI-QMC.

This script does not replace the PI-QMC runner. It orchestrates repeated calls to
runners/validation/run_atomistic_grid_temp_field_sweep_v1_0.py with different global move
sizes, so we can test whether low-T localisation observables are stable.

By default it prints commands only. Add --execute to actually run them.

Recommended from project root:

  python3 runners/validation/run_lowT_convergence_protocol_v1_0.py \
    --potential_npz results/wse2_mose2/V_grid.npz \
    --runner_script runners/validation/run_atomistic_grid_temp_field_sweep_v1_0.py \
    --analysis_script runners/validation/analyze_atomistic_grid_pimc_v1_0.py \
    --execute

This produces output dirs like:

  results/lowT_conv_v1_0_gstep_1p0/
  results/lowT_conv_v1_0_gstep_2p0/
  results/lowT_conv_v1_0_gstep_3p5/
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def find_project_root() -> Path:
    env_root = os.environ.get("TMD_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve()
    candidates = [Path.cwd().resolve()]
    candidates.extend(parent for parent in here.parents[:5])
    for cand in candidates:
        if (cand / "numerics" / "tmd_pimc").exists() or (cand / "results").exists():
            return cand
    return Path.cwd().resolve()


ROOT = find_project_root()


def resolve_project_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def tag_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def build_runner_command(args, global_step: float, output_tag: str) -> list[str]:
    cmd = [
        sys.executable,
        str(resolve_project_path(args.runner_script)),
        "--potential_npz",
        str(args.potential_npz),
        "--temps",
        *[str(t) for t in args.temps],
        "--ex_values",
        *[str(v) for v in args.ex_values],
        "--start_mode",
        args.start_mode,
        "--n_steps",
        str(args.n_steps),
        "--burn_in",
        str(args.burn_in),
        "--sample_every",
        str(args.sample_every),
        "--seeds",
        str(args.seeds),
        "--workers",
        str(args.workers),
        "--p_max",
        str(args.p_max),
        "--mass_m0",
        str(args.mass_m0),
        "--local_step_multiplier",
        str(args.local_step_multiplier),
        "--global_step_nm",
        str(global_step),
        "--global_move_probability",
        str(args.global_move_probability),
        "--hist_bins",
        str(args.hist_bins),
        "--jit_grid_size",
        str(args.jit_grid_size),
        "--output_tag",
        output_tag,
    ]

    if args.jit_grid_range_nm is not None:
        cmd += ["--jit_grid_range_nm", str(args.jit_grid_range_nm)]
    if args.coordinate_origin:
        cmd += ["--coordinate_origin", args.coordinate_origin]
    if args.grid_nonperiodic:
        cmd += ["--grid_nonperiodic"]
    if args.grid_keep_absolute_offset:
        cmd += ["--grid_keep_absolute_offset"]
    if args.grid_scale != 1.0:
        cmd += ["--grid_scale", str(args.grid_scale)]
    if args.no_envelope:
        cmd += ["--no_envelope"]
    else:
        cmd += ["--envelope_v0_eV", str(args.envelope_v0_eV), "--envelope_radius_nm", str(args.envelope_radius_nm)]
    if args.add_soft_coulomb:
        cmd += [
            "--add_soft_coulomb",
            "--coulomb_strength_eV_nm",
            str(args.coulomb_strength_eV_nm),
            "--coulomb_softening_nm",
            str(args.coulomb_softening_nm),
        ]
    if args.compressed_npz:
        cmd += ["--compressed_npz"]
    if args.save_samples:
        cmd += ["--save_samples"]
    return cmd


def normalise_analysis_maps(value: str | None) -> str | None:
    """
    Translate this wrapper's user-facing --analysis_maps value to the value
    expected by analyze_atomistic_grid_pimc_v1_0.py.

    Backward compatibility:
      - "zero" is accepted here as a legacy alias
      - the analyzer expects "zero_field"
    """
    if value is None:
        return None
    value = str(value)
    if value == "zero":
        return "zero_field"
    return value


def build_analysis_command(args, output_tag: str) -> list[str] | None:
    if not args.run_analysis:
        return None
    analysis_script = resolve_project_path(args.analysis_script)
    if analysis_script is None or not analysis_script.exists():
        raise FileNotFoundError(f"Analysis script not found: {analysis_script}")
    cmd = [
        sys.executable,
        str(analysis_script),
        "--results_dir",
        str(Path("results") / output_tag),
        "--potential_npz",
        str(args.potential_npz),
    ]
    analysis_maps = normalise_analysis_maps(args.analysis_maps)
    if analysis_maps:
        cmd += ["--maps", analysis_maps]
    return cmd


def build_commands(args) -> list[dict]:
    commands = []
    for global_step in args.global_steps:
        output_tag = f"{args.output_prefix}_gstep_{tag_float(float(global_step))}"
        runner_cmd = build_runner_command(args, float(global_step), output_tag)
        commands.append({"kind": "runner", "global_step_nm": float(global_step), "output_tag": output_tag, "cmd": runner_cmd})
        analysis_cmd = build_analysis_command(args, output_tag)
        if analysis_cmd is not None:
            commands.append({"kind": "analysis", "global_step_nm": float(global_step), "output_tag": output_tag, "cmd": analysis_cmd})
    return commands


def write_command_log(commands: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in commands:
        lines.append(f"# {item['kind']} | global_step_nm={item['global_step_nm']} | output_tag={item['output_tag']}")
        lines.append(shell_join(item["cmd"]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(commands, indent=2), encoding="utf-8")


def run_commands(commands: list[dict], continue_on_error: bool) -> None:
    env = os.environ.copy()
    code_path = str(ROOT / "numerics")
    env["PYTHONPATH"] = code_path + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TMD_PROJECT_ROOT", str(ROOT))

    for idx, item in enumerate(commands, start=1):
        print("=" * 80)
        print(f"[{idx}/{len(commands)}] {item['kind']} | gstep={item['global_step_nm']} | tag={item['output_tag']}")
        print(shell_join(item["cmd"]))
        print("=" * 80)
        t0 = time.perf_counter()
        try:
            subprocess.run(item["cmd"], cwd=ROOT, env=env, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: command failed with return code {exc.returncode}")
            if not continue_on_error:
                raise
        dt = time.perf_counter() - t0
        print(f"Completed in {dt/60:.2f} min")


def parse_args():
    p = argparse.ArgumentParser(description="Run low-T convergence protocol for atomistic-grid PI-QMC.")
    p.add_argument("--potential_npz", default="results/wse2_mose2/V_grid.npz")
    p.add_argument("--runner_script", default="runners/validation/run_atomistic_grid_temp_field_sweep_v1_0.py")
    p.add_argument("--analysis_script", default="runners/validation/analyze_atomistic_grid_pimc_v1_0.py")
    p.add_argument("--output_prefix", default="lowT_conv_v1_0")

    p.add_argument("--temps", type=float, nargs="+", default=[20, 30, 40, 50, 60, 80, 100, 120, 150])
    p.add_argument("--ex_values", type=float, nargs="+", default=[0.0])
    p.add_argument("--global_steps", type=float, nargs="+", default=[1.0, 2.0, 3.5])

    p.add_argument("--n_steps", type=int, default=160_000)
    p.add_argument("--burn_in", type=int, default=40_000)
    p.add_argument("--sample_every", type=int, default=50)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--p_max", type=int, default=128)
    p.add_argument("--mass_m0", type=float, default=0.5)
    p.add_argument("--local_step_multiplier", type=float, default=0.70)
    p.add_argument("--global_move_probability", type=float, default=0.20)
    p.add_argument("--start_mode", choices=["grid_cycle", "grid_min", "field", "random_grid", "random_box", "origin", "hex"], default="grid_cycle")

    p.add_argument("--coordinate_origin", choices=["center", "native"], default="center")
    p.add_argument("--grid_scale", type=float, default=1.0)
    p.add_argument("--grid_nonperiodic", action="store_true")
    p.add_argument("--grid_keep_absolute_offset", action="store_true")

    p.add_argument("--no_envelope", action="store_true")
    p.add_argument("--envelope_v0_eV", type=float, default=0.010)
    p.add_argument("--envelope_radius_nm", type=float, default=30.0)
    p.add_argument("--add_soft_coulomb", action="store_true")
    p.add_argument("--coulomb_strength_eV_nm", type=float, default=0.15)
    p.add_argument("--coulomb_softening_nm", type=float, default=1.0)

    p.add_argument("--hist_bins", type=int, default=200)
    p.add_argument("--jit_grid_size", type=int, default=600)
    p.add_argument("--jit_grid_range_nm", type=float, default=None)
    p.add_argument("--compressed_npz", action="store_true")
    p.add_argument("--save_samples", action="store_true")

    p.add_argument("--run_analysis", action="store_true", help="Run analyze_atomistic_grid_pimc_v1_0.py after each runner job.")
    p.add_argument(
        "--analysis_maps",
        choices=["all", "zero", "zero_field", "selected", "none"],
        default="zero_field",
        help=(
            "Map plotting mode passed to analyze_atomistic_grid_pimc_v1_0.py. "
            "Legacy alias accepted: zero -> zero_field."
        ),
    )
    p.add_argument("--execute", action="store_true", help="Actually run commands. Without this, commands are only printed.")
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--command_log", default="results/lowT_convergence_protocol_v1_0_commands.sh")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runner = resolve_project_path(args.runner_script)
    potential = resolve_project_path(args.potential_npz)
    if runner is None or not runner.exists():
        raise FileNotFoundError(f"Runner script not found: {runner}")
    if potential is None or not potential.exists():
        raise FileNotFoundError(f"Potential file not found: {potential}")

    commands = build_commands(args)
    log_path = resolve_project_path(args.command_log)
    write_command_log(commands, log_path)

    print("=" * 80)
    print("Low-T convergence protocol v1.1")
    print("=" * 80)
    print(f"Project root : {ROOT}")
    print(f"Potential    : {potential}")
    print(f"Runner       : {runner}")
    print(f"Temps        : {args.temps}")
    print(f"Ex values    : {args.ex_values}")
    print(f"Global steps : {args.global_steps}")
    print(f"Seeds        : {args.seeds}")
    print(f"Steps        : {args.n_steps} burn-in {args.burn_in} sample_every {args.sample_every}")
    if args.run_analysis:
        print(f"Analysis maps: {args.analysis_maps} -> {normalise_analysis_maps(args.analysis_maps)}")
    print(f"Command log  : {log_path}")
    print("")
    for item in commands:
        print(f"# {item['kind']} | gstep={item['global_step_nm']} | tag={item['output_tag']}")
        print(shell_join(item["cmd"]))
        print("")

    if args.execute:
        run_commands(commands, continue_on_error=args.continue_on_error)
    else:
        print("Dry run only. Add --execute to run the protocol.")


if __name__ == "__main__":
    main()
