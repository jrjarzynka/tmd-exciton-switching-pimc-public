#!/usr/bin/env python3
"""
v1.0 — Compare low-T convergence protocol outputs across global move sizes.

Input can be either:
  1. result directories produced by run_lowT_convergence_protocol_v1_0.py, after
     running analyze_atomistic_grid_pimc_v1_0.py, or
  2. raw runner output directories containing T_*/Ex_*/results.npz.

The script combines zero-field observables and asks:

  - do A_eff_entropy and A_eff_IPR change when global_step_nm is changed?
  - does acceptance_global improve at low T?
  - are observables stable enough to claim convergence?

Recommended:

  python3 runners/validation/compare_lowT_convergence_v1_0.py \
    --results_glob 'results/lowT_conv_v1_0_gstep_*' \
    --outdir results/lowT_conv_v1_0_comparison
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_project_root() -> Path:
    env_root = os.environ.get("TMD_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve()
    candidates = [Path.cwd().resolve()]
    candidates.extend(parent for parent in here.parents[:5])
    for cand in candidates:
        if (cand / "results").exists() or (cand / "numerics").exists():
            return cand
    return Path.cwd().resolve()


ROOT = find_project_root()


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def parse_gstep_from_tag(tag: str) -> float | None:
    m = re.search(r"gstep_([0-9]+(?:p[0-9]+)?)", tag)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def safe_scalar(npz, key: str, default=np.nan):
    if key not in npz:
        return default
    val = npz[key]
    try:
        arr = np.asarray(val)
        if arr.shape == ():
            return arr.item()
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
        return val
    except Exception:
        return default


def metrics_from_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        T = float(safe_scalar(z, "T_K"))
        Ex = float(safe_scalar(z, "Ex"))
        density = np.asarray(z["density_H"], dtype=float) if "density_H" in z else None
        prob = np.asarray(z["probability_H"], dtype=float) if "probability_H" in z else None
        xedges = np.asarray(z["density_xedges"], dtype=float) if "density_xedges" in z else None
        yedges = np.asarray(z["density_yedges"], dtype=float) if "density_yedges" in z else None
        free_energy_masked = np.asarray(z["free_energy_masked"], dtype=float) if "free_energy_masked" in z else None
        cents = np.asarray(z["centroids"], dtype=float) if "centroids" in z else None

        row = {
            "T_K": T,
            "Ex": Ex,
            "npz_path": str(path),
            "mean_x": float(safe_scalar(z, "mean_x")),
            "mean_y": float(safe_scalar(z, "mean_y")),
            "r2_bead": float(safe_scalar(z, "r2_bead")),
            "r2_centroid": float(safe_scalar(z, "r2_centroid")),
            "r2_spread": float(safe_scalar(z, "r2_spread")),
            "r_rms_bead": float(safe_scalar(z, "r_rms_bead", np.sqrt(float(safe_scalar(z, "r2_bead"))))),
            "r_rms_centroid": float(safe_scalar(z, "r_rms_centroid", np.sqrt(float(safe_scalar(z, "r2_centroid"))))),
            "r_rms_spread": float(safe_scalar(z, "r_rms_spread", np.sqrt(float(safe_scalar(z, "r2_spread"))))),
            "mean_V_bead_eV": float(safe_scalar(z, "mean_V_bead_eV")),
            "mean_V_centroid_eV": float(safe_scalar(z, "mean_V_centroid_eV")),
            "acceptance_local": float(safe_scalar(z, "acceptance_local")),
            "acceptance_global": float(safe_scalar(z, "acceptance_global")),
            "n_beads": float(safe_scalar(z, "n_beads")),
            "mass_m0": float(safe_scalar(z, "mass_m0")),
            "global_step_nm": float(safe_scalar(z, "global_step_nm")),
            "local_step_nm": float(safe_scalar(z, "local_step_nm")),
        }

    if prob is None and density is not None and np.sum(density) > 0:
        prob = density / float(np.sum(density))

    if prob is not None and xedges is not None and yedges is not None:
        p = prob.astype(float).ravel()
        p = p[p > 0]
        dx = float(np.mean(np.diff(xedges))) if len(xedges) > 1 else np.nan
        dy = float(np.mean(np.diff(yedges))) if len(yedges) > 1 else np.nan
        area = abs(dx * dy) if np.isfinite(dx) and np.isfinite(dy) else np.nan
        entropy = -float(np.sum(p * np.log(p))) if len(p) else np.nan
        n_eff_entropy = float(np.exp(entropy)) if np.isfinite(entropy) else np.nan
        n_eff_ipr = float(1.0 / np.sum(p * p)) if len(p) else np.nan
        row.update(
            {
                "hist_bin_area_nm2": area,
                "entropy": entropy,
                "N_eff_entropy": n_eff_entropy,
                "A_eff_entropy_nm2": n_eff_entropy * area if np.isfinite(area) else np.nan,
                "N_eff_ipr": n_eff_ipr,
                "A_eff_ipr_nm2": n_eff_ipr * area if np.isfinite(area) else np.nan,
                "hist_pmax": float(np.max(prob)),
            }
        )

    if free_energy_masked is not None:
        F = free_energy_masked[np.isfinite(free_energy_masked)] * 1000.0
        if F.size:
            row.update(
                {
                    "F_p50_meV": float(np.percentile(F, 50)),
                    "F_p90_meV": float(np.percentile(F, 90)),
                    "F_p95_meV": float(np.percentile(F, 95)),
                    "F_max_meV": float(np.max(F)),
                }
            )

    if cents is not None and cents.ndim == 2 and cents.shape[1] == 2 and len(cents) >= 3:
        cov = np.cov(cents.T)
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(np.maximum(eig, 0.0))[::-1]
        sigma_major = float(np.sqrt(eig[0])) if len(eig) else np.nan
        sigma_minor = float(np.sqrt(eig[1])) if len(eig) > 1 else np.nan
        row.update(
            {
                "centroid_count": int(len(cents)),
                "centroid_std_x_nm": float(np.std(cents[:, 0])),
                "centroid_std_y_nm": float(np.std(cents[:, 1])),
                "centroid_sigma_major_nm": sigma_major,
                "centroid_sigma_minor_nm": sigma_minor,
                "centroid_anisotropy": sigma_major / max(sigma_minor, 1e-12) if np.isfinite(sigma_major) and np.isfinite(sigma_minor) else np.nan,
            }
        )

    return row


def read_analysis_or_raw(results_dir: Path) -> pd.DataFrame:
    candidates = [
        results_dir / "analysis_v1_0" / "tables" / "ensemble_observables.csv",
        results_dir / "analysis" / "tables" / "ensemble_observables.csv",
    ]
    for c in candidates:
        if c.exists():
            df = pd.read_csv(c)
            df["source_table"] = str(c)
            return df

    # Fallback: raw ensemble NPZs.
    rows = []
    for npz in sorted(results_dir.glob("T_*K/Ex_*/results.npz")):
        try:
            rows.append(metrics_from_npz(npz))
        except Exception as exc:
            print(f"WARNING: could not read {npz}: {exc}")
    if not rows:
        raise FileNotFoundError(
            f"No analysis table or raw T_*/Ex_*/results.npz files found in {results_dir}"
        )
    return pd.DataFrame(rows)


def collect_results(results_dirs: list[Path], ex_target: float, ex_tol: float) -> pd.DataFrame:
    frames = []
    for rd in results_dirs:
        df = read_analysis_or_raw(rd)
        df = df.copy()
        tag = rd.name
        df["run_tag"] = tag
        gstep = parse_gstep_from_tag(tag)
        if gstep is not None:
            df["global_step_nm_from_tag"] = gstep
            if "global_step_nm" not in df or df["global_step_nm"].isna().all():
                df["global_step_nm"] = gstep
        if "Ex" in df:
            df = df[np.abs(df["Ex"].astype(float) - float(ex_target)) <= float(ex_tol)].copy()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined.sort_values(["T_K", "run_tag"]).reset_index(drop=True)


def summarise_convergence(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for T, group in df.groupby("T_K"):
        row = {"T_K": T, "n_runs": int(group["run_tag"].nunique())}
        for metric in metrics:
            if metric not in group.columns:
                continue
            vals = group[metric].astype(float).to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            mean = float(np.mean(vals))
            mn = float(np.min(vals))
            mx = float(np.max(vals))
            sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std_across_runs"] = sd
            row[f"{metric}_min"] = mn
            row[f"{metric}_max"] = mx
            row[f"{metric}_rel_range_pct"] = float((mx - mn) / abs(mean) * 100.0) if abs(mean) > 1e-12 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("T_K")


def plot_metric_vs_T(df: pd.DataFrame, metric: str, out: Path) -> None:
    if metric not in df.columns:
        return
    plt.figure(figsize=(7.0, 4.8))
    for tag, group in df.groupby("run_tag"):
        g = group.sort_values("T_K")
        plt.plot(g["T_K"], g[metric], marker="o", label=str(tag))
    plt.xlabel("T (K)")
    plt.ylabel(metric)
    plt.title(f"Convergence check: {metric}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_rel_range(summary: pd.DataFrame, metric: str, out: Path) -> None:
    col = f"{metric}_rel_range_pct"
    if col not in summary.columns:
        return
    plt.figure(figsize=(7.0, 4.8))
    plt.plot(summary["T_K"], summary[col], marker="o")
    plt.axhline(10.0, linestyle="--", linewidth=1)
    plt.xlabel("T (K)")
    plt.ylabel("relative range across runs (%)")
    plt.title(f"Across-global-step stability: {metric}")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def write_report(path: Path, combined: pd.DataFrame, summary: pd.DataFrame, metrics: list[str]) -> None:
    lines = []
    lines.append("Low-T convergence comparison v1.0")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Runs: {', '.join(sorted(combined['run_tag'].unique()))}")
    lines.append(f"Temperatures: {sorted(combined['T_K'].unique())}")
    lines.append("")
    lines.append("Rule of thumb")
    lines.append("-" * 78)
    lines.append("For the same T, A_eff/IPR varying by <~10% across global_step_nm is a good sign.")
    lines.append("Large variation at low T means basin mixing is still influencing the result.")
    lines.append("Acceptance_global itself does not need to be high, but observables must be stable.")
    lines.append("")
    lines.append("Relative range across global-step runs")
    lines.append("-" * 78)
    cols = ["T_K", "n_runs"]
    for metric in metrics:
        col = f"{metric}_rel_range_pct"
        if col in summary.columns:
            cols.append(col)
    if len(cols) > 2:
        lines.append(summary[cols].to_string(index=False))
    else:
        lines.append("No comparable metric columns found.")
    lines.append("")
    lines.append("Potential red flags")
    lines.append("-" * 78)
    for _, row in summary.iterrows():
        T = row["T_K"]
        flags = []
        for metric in ["A_eff_entropy_nm2", "A_eff_ipr_nm2", "F_p95_meV", "r_rms_centroid"]:
            col = f"{metric}_rel_range_pct"
            if col in row and pd.notna(row[col]) and float(row[col]) > 15.0:
                flags.append(f"{metric} range {row[col]:.1f}%")
        if flags:
            lines.append(f"T={T:g} K: " + "; ".join(flags))
    if not any("T=" in line and "K:" in line for line in lines[-len(summary)-2:]):
        lines.append("No large >15% relative-range flags found in the selected metrics.")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Compare low-T convergence outputs across global step sizes.")
    p.add_argument("--results_dirs", nargs="+", default=None)
    p.add_argument("--results_glob", default="results/lowT_conv_v1_0_gstep_*")
    p.add_argument("--outdir", default="results/lowT_conv_v1_0_comparison")
    p.add_argument("--ex_target", type=float, default=0.0)
    p.add_argument("--ex_tol", type=float, default=1e-12)
    p.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "A_eff_entropy_nm2",
            "A_eff_ipr_nm2",
            "support_1pct_area_nm2",
            "F_p95_meV",
            "r_rms_centroid",
            "r_rms_spread",
            "acceptance_global",
            "acceptance_local",
            "mean_V_centroid_eV",
            "centroid_anisotropy",
        ],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.results_dirs:
        results_dirs = [resolve_project_path(p) for p in args.results_dirs]
    else:
        pattern = str(resolve_project_path(args.results_glob))
        results_dirs = [Path(p).resolve() for p in sorted(glob.glob(pattern)) if Path(p).is_dir()]
    if not results_dirs:
        raise FileNotFoundError("No result directories found. Use --results_dirs or --results_glob.")

    outdir = resolve_project_path(args.outdir)
    tables = outdir / "tables"
    figures = outdir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    combined = collect_results(results_dirs, ex_target=args.ex_target, ex_tol=args.ex_tol)
    if combined.empty:
        raise RuntimeError("No rows found after Ex filtering.")
    combined.to_csv(tables / "combined_zero_field_observables.csv", index=False)

    summary = summarise_convergence(combined, args.metrics)
    summary.to_csv(tables / "convergence_summary_by_T.csv", index=False)

    for metric in args.metrics:
        if metric in combined.columns:
            plot_metric_vs_T(combined, metric, figures / f"{metric}_vs_T_by_run.png")
            plot_rel_range(summary, metric, figures / f"{metric}_relative_range_vs_T.png")

    write_report(outdir / "lowT_convergence_report.txt", combined, summary, args.metrics)

    print("=" * 78)
    print("Low-T convergence comparison completed")
    print("=" * 78)
    print(f"Input runs : {len(results_dirs)}")
    for rd in results_dirs:
        print(f"  - {rd}")
    print(f"Outdir     : {outdir}")
    print(f"Report     : {outdir / 'lowT_convergence_report.txt'}")
    print(f"Summary    : {tables / 'convergence_summary_by_T.csv'}")


if __name__ == "__main__":
    main()
