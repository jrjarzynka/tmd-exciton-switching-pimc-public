#!/usr/bin/env python3
"""
v1.0 — Analysis script for atomistic-grid moiré exciton PI-QMC results

This script analyzes outputs produced by:

    run_atomistic_grid_temp_field_sweep_v1_0.py

It is designed for the atomistic WSe2/MoSe2 workflow:

    V_grid.npz -> GridPotential2D -> PI-QMC -> results/<output_tag>/...

What it reads
-------------
1. The per-seed CSV written by the runner, e.g.

       results/wse2_mose2_atomistic_grid_v1_0/wse2_mose2_atomistic_grid_v1_0.csv

2. The ensemble NPZ files written for every (T, Ex), e.g.

       results/<output_tag>/T_20p0K/Ex_+0p0000/results.npz

What it writes
--------------
A compact analysis folder, by default:

       results/<output_tag>/analysis_v1_0/

containing:

       tables/ensemble_observables.csv
       tables/seed_summary_by_T_Ex.csv
       tables/transition_summary_zero_field.csv
       figures/metric_*.png
       figures/heatmap_*.png
       figures/maps/*.png

Main observables
----------------
- mean_x, mean_y
- std_x, std_y
- r2_bead, r2_centroid, r2_spread
- r_rms_bead, r_rms_centroid, r_rms_spread
- mean potential energy on beads / centroids when stored by the runner
- local/global acceptance
- centroid histogram probability_H
- free_energy / free_energy_masked
- entropy effective area and IPR effective area from probability_H
- centroid covariance eigenwidths and anisotropy

Recommended usage from the project root
---------------------------------------

    export PYTHONPATH="$PWD/code:$PYTHONPATH"

    python3 runners/validation/analyze_atomistic_grid_pimc_v1_0.py \
      --results_dir results/wse2_mose2_atomistic_grid_v1_0 \
      --potential_npz results/wse2_mose2/V_grid.npz

For a very quick run without map figures:

    python3 runners/validation/analyze_atomistic_grid_pimc_v1_0.py \
      --results_dir results/wse2_mose2_atomistic_grid_v1_0 \
      --maps none
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Project-root handling
# ---------------------------------------------------------------------------


def find_project_root() -> Path:
    """Find the project root when this script lives in scripts/."""
    env_root = os.environ.get("TMD_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    candidates = [Path.cwd().resolve()]
    candidates.extend(parent for parent in here.parents[:5])

    for cand in candidates:
        if (cand / "results").exists() or (cand / "numerics" / "tmd_pimc").exists():
            return cand
    return here.parents[1]


ROOT = find_project_root()


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def resolve_path(path_like: str | Path | None, base: Path | None = None) -> Path | None:
    if path_like is None:
        return None
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (base or ROOT) / path
    return path.resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_float_from_npz(npz, key: str, default: float = np.nan) -> float:
    if key not in npz.files:
        return float(default)
    arr = np.asarray(npz[key])
    if arr.size == 0:
        return float(default)
    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default)


def safe_str_from_npz(npz, key: str, default: str = "") -> str:
    if key not in npz.files:
        return default
    arr = np.asarray(npz[key])
    if arr.size == 0:
        return default
    try:
        return str(arr.reshape(-1)[0])
    except Exception:
        return default


def parse_T_from_dir(path: Path) -> float | None:
    for part in path.parts[::-1]:
        m = re.fullmatch(r"T_([0-9]+(?:p[0-9]+)?)K", part)
        if m:
            return float(m.group(1).replace("p", "."))
    return None


def parse_Ex_from_dir(path: Path) -> float | None:
    for part in path.parts[::-1]:
        m = re.fullmatch(r"Ex_([+-][0-9]+p[0-9]+)", part)
        if m:
            return float(m.group(1).replace("p", "."))
    return None


def format_temperature_label(T_K: float) -> str:
    return f"T_{float(T_K):.1f}K".replace(".", "p")


def format_ex_label(Ex: float) -> str:
    return f"Ex_{float(Ex):+.4f}".replace(".", "p")


def bin_centres(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def finite_or_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values


def nearest_value(values: Sequence[float], target: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot choose nearest value from an empty sequence")
    return float(arr[np.nanargmin(np.abs(arr - float(target)))])


def maybe_read_csv(results_dir: Path, csv_path: Path | None) -> pd.DataFrame | None:
    if csv_path is not None and csv_path.exists():
        return pd.read_csv(csv_path)

    candidates = sorted(results_dir.glob("*.csv"))
    if candidates:
        # Prefer a CSV whose stem matches the results directory name.
        for cand in candidates:
            if cand.stem == results_dir.name:
                return pd.read_csv(cand)
        return pd.read_csv(candidates[0])
    return None


# ---------------------------------------------------------------------------
# Grid-potential loader for optional plotting
# ---------------------------------------------------------------------------


@dataclass
class PotentialGrid:
    x_nm: np.ndarray
    y_nm: np.ndarray
    V_eV: np.ndarray
    path: Path


def first_existing_key(npz, keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in npz.files:
            return key
    return None


def load_potential_grid(path: Path, coordinate_origin: str = "center") -> PotentialGrid:
    """Load V_grid.npz for plotting. Supports several key conventions."""
    with np.load(path, allow_pickle=True) as data:
        x_key = first_existing_key(data, ["x_nm", "x", "X_nm", "X", "x_axis_nm", "x_nm_axis"])
        y_key = first_existing_key(data, ["y_nm", "y", "Y_nm", "Y", "y_axis_nm", "y_nm_axis"])
        v_key = first_existing_key(
            data,
            [
                "V_eV", "V", "potential_eV", "V_grid_eV", "v_eV",
                "V_meV", "potential_meV", "V_grid_meV", "v_meV",
            ],
        )
        if x_key is None or y_key is None or v_key is None:
            raise KeyError(
                f"Could not find x/y/V keys in {path}. Available keys: {data.files}"
            )
        x = np.asarray(data[x_key], dtype=float).squeeze()
        y = np.asarray(data[y_key], dtype=float).squeeze()
        V = np.asarray(data[v_key], dtype=float)

    if x.ndim == 2:
        x = x[:, 0]
    if y.ndim == 2:
        y = y[0, :]

    if "meV" in v_key.lower():
        V = V / 1000.0

    if V.shape == (len(y), len(x)):
        # Convert to runner convention: V[x_index, y_index]
        V = V.T

    if V.shape != (len(x), len(y)):
        raise ValueError(
            f"Potential grid shape mismatch: V.shape={V.shape}, len(x)={len(x)}, len(y)={len(y)}"
        )

    if coordinate_origin == "center":
        x = x - 0.5 * (float(x[0]) + float(x[-1]))
        y = y - 0.5 * (float(y[0]) + float(y[-1]))
    elif coordinate_origin == "native":
        pass
    else:
        raise ValueError(f"Unknown coordinate_origin={coordinate_origin!r}")

    V = V - np.nanmin(V)
    return PotentialGrid(x_nm=x, y_nm=y, V_eV=V, path=path)


# ---------------------------------------------------------------------------
# Ensemble NPZ analysis
# ---------------------------------------------------------------------------


def find_ensemble_npzs(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("T_*K/Ex_*/results.npz"))


def histogram_observables(npz) -> dict:
    """Compute localization/delocalization metrics from probability_H."""
    out: dict[str, float] = {}

    if "probability_H" not in npz.files:
        return out

    P = np.asarray(npz["probability_H"], dtype=float)
    P = np.maximum(P, 0.0)
    total = float(np.sum(P))
    if total <= 0.0:
        return {
            "hist_total_probability": 0.0,
            "hist_pmax": np.nan,
            "entropy": np.nan,
            "N_eff_entropy": np.nan,
            "A_eff_entropy_nm2": np.nan,
            "N_eff_ipr": np.nan,
            "A_eff_ipr_nm2": np.nan,
            "support_1pct_area_nm2": np.nan,
        }

    P = P / total
    p_nonzero = P[P > 0]

    dx = dy = np.nan
    area_bin = np.nan
    if "density_xedges" in npz.files and "density_yedges" in npz.files:
        xedges = np.asarray(npz["density_xedges"], dtype=float)
        yedges = np.asarray(npz["density_yedges"], dtype=float)
        if len(xedges) > 1 and len(yedges) > 1:
            dx = float(np.mean(np.diff(xedges)))
            dy = float(np.mean(np.diff(yedges)))
            area_bin = abs(dx * dy)

    entropy = float(-np.sum(p_nonzero * np.log(p_nonzero)))
    n_eff_entropy = float(np.exp(entropy))
    ipr = float(np.sum(P**2))
    n_eff_ipr = float(1.0 / ipr) if ipr > 0 else np.nan
    pmax = float(np.max(P))
    support_1pct_bins = int(np.sum(P >= 0.01 * pmax)) if pmax > 0 else 0
    support_5pct_bins = int(np.sum(P >= 0.05 * pmax)) if pmax > 0 else 0

    out.update(
        hist_total_probability=total,
        hist_dx_nm=dx,
        hist_dy_nm=dy,
        hist_bin_area_nm2=area_bin,
        hist_pmax=pmax,
        entropy=entropy,
        N_eff_entropy=n_eff_entropy,
        A_eff_entropy_nm2=n_eff_entropy * area_bin if np.isfinite(area_bin) else np.nan,
        N_eff_ipr=n_eff_ipr,
        A_eff_ipr_nm2=n_eff_ipr * area_bin if np.isfinite(area_bin) else np.nan,
        support_1pct_bins=support_1pct_bins,
        support_1pct_area_nm2=support_1pct_bins * area_bin if np.isfinite(area_bin) else np.nan,
        support_5pct_bins=support_5pct_bins,
        support_5pct_area_nm2=support_5pct_bins * area_bin if np.isfinite(area_bin) else np.nan,
    )

    return out


def free_energy_observables(npz) -> dict:
    out: dict[str, float] = {}
    if "free_energy_masked" in npz.files:
        F = np.asarray(npz["free_energy_masked"], dtype=float)
    elif "free_energy" in npz.files:
        F = np.asarray(npz["free_energy"], dtype=float)
    else:
        return out

    finite = F[np.isfinite(F)]
    if finite.size == 0:
        return {
            "F_finite_bins": 0,
            "F_p50_meV": np.nan,
            "F_p90_meV": np.nan,
            "F_p95_meV": np.nan,
            "F_max_meV": np.nan,
        }

    finite = finite - np.nanmin(finite)
    out.update(
        F_finite_bins=int(finite.size),
        F_p50_meV=float(np.nanpercentile(finite, 50.0) * 1000.0),
        F_p90_meV=float(np.nanpercentile(finite, 90.0) * 1000.0),
        F_p95_meV=float(np.nanpercentile(finite, 95.0) * 1000.0),
        F_max_meV=float(np.nanmax(finite) * 1000.0),
    )
    return out


def centroid_observables(npz) -> dict:
    out: dict[str, float] = {}
    if "centroids" not in npz.files:
        return out
    C = np.asarray(npz["centroids"], dtype=float)
    if C.ndim != 2 or C.shape[1] != 2 or C.shape[0] < 2:
        return out

    x = C[:, 0]
    y = C[:, 1]
    r = np.sqrt(x**2 + y**2)
    cov = np.cov(C.T)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.sort(np.maximum(evals, 0.0))[::-1]
    sigma_major = float(np.sqrt(evals[0]))
    sigma_minor = float(np.sqrt(evals[1])) if len(evals) > 1 else np.nan
    anisotropy = sigma_major / sigma_minor if sigma_minor and sigma_minor > 0 else np.nan

    out.update(
        centroid_count=int(C.shape[0]),
        centroid_mean_x_nm=float(np.mean(x)),
        centroid_mean_y_nm=float(np.mean(y)),
        centroid_std_x_nm=float(np.std(x)),
        centroid_std_y_nm=float(np.std(y)),
        centroid_mean_r_nm=float(np.mean(r)),
        centroid_std_r_nm=float(np.std(r)),
        centroid_r_p50_nm=float(np.percentile(r, 50.0)),
        centroid_r_p90_nm=float(np.percentile(r, 90.0)),
        centroid_sigma_major_nm=sigma_major,
        centroid_sigma_minor_nm=sigma_minor,
        centroid_anisotropy=anisotropy,
    )
    return out


def analyze_one_npz(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=True) as z:
        T = safe_float_from_npz(z, "T_K", default=parse_T_from_dir(npz_path) or np.nan)
        Ex = safe_float_from_npz(z, "Ex", default=parse_Ex_from_dir(npz_path) or np.nan)

        row = {
            "T_K": T,
            "Ex": Ex,
            "npz_path": str(npz_path),
            "mean_x": safe_float_from_npz(z, "mean_x"),
            "mean_y": safe_float_from_npz(z, "mean_y"),
            "std_x": safe_float_from_npz(z, "std_x"),
            "std_y": safe_float_from_npz(z, "std_y"),
            "r2_bead": safe_float_from_npz(z, "r2_bead"),
            "r2_centroid": safe_float_from_npz(z, "r2_centroid"),
            "r2_spread": safe_float_from_npz(z, "r2_spread"),
            "r_rms_bead": safe_float_from_npz(z, "r_rms_bead"),
            "r_rms_centroid": safe_float_from_npz(z, "r_rms_centroid"),
            "r_rms_spread": safe_float_from_npz(z, "r_rms_spread"),
            "mean_V_bead_eV": safe_float_from_npz(z, "mean_V_bead_eV"),
            "mean_V_centroid_eV": safe_float_from_npz(z, "mean_V_centroid_eV"),
            "acceptance_local": safe_float_from_npz(z, "acceptance_local"),
            "acceptance_global": safe_float_from_npz(z, "acceptance_global"),
            "n_beads": safe_float_from_npz(z, "n_beads"),
            "beta_eV": safe_float_from_npz(z, "beta_eV"),
            "mass_m0": safe_float_from_npz(z, "mass_m0"),
            "p_capped": safe_float_from_npz(z, "p_capped", default=0.0),
            "coordinate_origin": safe_str_from_npz(z, "coordinate_origin"),
            "potential_npz": safe_str_from_npz(z, "potential_npz"),
            "grid_periodic": safe_float_from_npz(z, "grid_periodic"),
            "grid_scale": safe_float_from_npz(z, "grid_scale"),
            "add_envelope": safe_float_from_npz(z, "add_envelope"),
            "add_soft_coulomb": safe_float_from_npz(z, "add_soft_coulomb"),
        }
        row.update(histogram_observables(z))
        row.update(free_energy_observables(z))
        row.update(centroid_observables(z))
        return row


def build_ensemble_table(npz_paths: Sequence[Path]) -> pd.DataFrame:
    rows = [analyze_one_npz(path) for path in npz_paths]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["T_K", "Ex"]).reset_index(drop=True)
    return df


def build_seed_summary(seed_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if seed_df is None or seed_df.empty:
        return None
    if not {"T_K", "Ex"}.issubset(seed_df.columns):
        return None

    numeric_cols = [c for c in seed_df.columns if pd.api.types.is_numeric_dtype(seed_df[c])]
    metric_cols = [c for c in numeric_cols if c not in {"T_K", "Ex", "seed", "rng_seed"}]
    if not metric_cols:
        return None

    grouped = seed_df.groupby(["T_K", "Ex"], as_index=False)[metric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = ["_".join([str(x) for x in col if str(x)]) for col in grouped.columns.to_flat_index()]
    grouped = grouped.reset_index()
    return grouped


def zero_field_summary(ensemble_df: pd.DataFrame) -> pd.DataFrame:
    if ensemble_df.empty or not {"T_K", "Ex"}.issubset(ensemble_df.columns):
        return pd.DataFrame()
    rows = []
    for T, sub in ensemble_df.groupby("T_K"):
        idx = (sub["Ex"].abs()).idxmin()
        rows.append(ensemble_df.loc[idx].to_dict())
    return pd.DataFrame(rows).sort_values("T_K").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def save_current_figure(path: Path, dpi: int) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_metric_vs_ex(df: pd.DataFrame, metric: str, outdir: Path, dpi: int) -> None:
    if df.empty or metric not in df.columns:
        return
    plt.figure(figsize=(7.0, 4.5))
    for T, sub in df.groupby("T_K"):
        sub = sub.sort_values("Ex")
        plt.plot(sub["Ex"].to_numpy(), sub[metric].to_numpy(), marker="o", label=f"{T:g} K")
    plt.xlabel("Effective field Ex [eV/nm]")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Ex")
    plt.legend(title="T", fontsize=8)
    plt.grid(True, alpha=0.3)
    save_current_figure(outdir / f"metric_{metric}_vs_Ex.png", dpi)


def plot_metric_vs_T_zero_field(df0: pd.DataFrame, metric: str, outdir: Path, dpi: int) -> None:
    if df0.empty or metric not in df0.columns:
        return
    sub = df0.sort_values("T_K")
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(sub["T_K"].to_numpy(), sub[metric].to_numpy(), marker="o")
    plt.xlabel("Temperature [K]")
    plt.ylabel(metric)
    plt.title(f"Zero-field {metric} vs T")
    plt.grid(True, alpha=0.3)
    save_current_figure(outdir / f"zero_field_{metric}_vs_T.png", dpi)


def plot_heatmap(df: pd.DataFrame, metric: str, outdir: Path, dpi: int) -> None:
    if df.empty or metric not in df.columns:
        return
    pivot = df.pivot_table(index="T_K", columns="Ex", values=metric, aggfunc="mean")
    if pivot.empty:
        return
    T_values = pivot.index.to_numpy(dtype=float)
    Ex_values = pivot.columns.to_numpy(dtype=float)
    Z = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(7.2, 4.8))
    if len(Ex_values) > 1 and len(T_values) > 1:
        extent = [Ex_values.min(), Ex_values.max(), T_values.min(), T_values.max()]
        plt.imshow(Z, origin="lower", aspect="auto", extent=extent)
        plt.xlabel("Effective field Ex [eV/nm]")
        plt.ylabel("Temperature [K]")
    else:
        plt.imshow(Z, origin="lower", aspect="auto")
        plt.xlabel("Ex index")
        plt.ylabel("T index")
        plt.xticks(range(len(Ex_values)), [f"{v:.3g}" for v in Ex_values])
        plt.yticks(range(len(T_values)), [f"{v:g}" for v in T_values])
    plt.colorbar(label=metric)
    plt.title(f"Heatmap: {metric}")
    save_current_figure(outdir / f"heatmap_{metric}.png", dpi)


def plot_acceptance_from_seed_csv(seed_df: pd.DataFrame, outdir: Path, dpi: int) -> None:
    if seed_df is None or seed_df.empty:
        return
    for metric in ["acceptance_local", "acceptance_global"]:
        if metric not in seed_df.columns:
            continue
        plt.figure(figsize=(7.0, 4.5))
        grouped = seed_df.groupby(["T_K", "Ex"], as_index=False)[metric].mean()
        for T, sub in grouped.groupby("T_K"):
            sub = sub.sort_values("Ex")
            plt.plot(sub["Ex"], sub[metric], marker="o", label=f"{T:g} K")
        plt.axhline(0.40, linestyle="--", linewidth=1.0)
        plt.axhline(0.55, linestyle="--", linewidth=1.0)
        plt.xlabel("Effective field Ex [eV/nm]")
        plt.ylabel(metric)
        plt.title(f"{metric} from per-seed CSV")
        plt.grid(True, alpha=0.3)
        plt.legend(title="T", fontsize=8)
        save_current_figure(outdir / f"seed_{metric}_vs_Ex.png", dpi)


def select_npzs_for_maps(
    npz_paths: Sequence[Path],
    mode: str,
    selected_temps: Sequence[float] | None,
    selected_ex: Sequence[float] | None,
) -> list[Path]:
    if mode == "none":
        return []
    if mode == "all":
        return list(npz_paths)

    records = []
    for path in npz_paths:
        T = parse_T_from_dir(path)
        Ex = parse_Ex_from_dir(path)
        if T is None or Ex is None:
            with np.load(path, allow_pickle=True) as z:
                T = safe_float_from_npz(z, "T_K")
                Ex = safe_float_from_npz(z, "Ex")
        records.append((float(T), float(Ex), path))

    if not records:
        return []

    if mode == "zero_field":
        out = []
        for T in sorted(set(r[0] for r in records)):
            sub = [r for r in records if r[0] == T]
            out.append(min(sub, key=lambda r: abs(r[1]))[2])
        return out

    if mode == "selected":
        if not selected_temps and not selected_ex:
            # If no explicit selection was given, use the same useful default as zero_field.
            return select_npzs_for_maps(npz_paths, "zero_field", None, None)
        wanted = []
        all_T = sorted(set(r[0] for r in records))
        all_Ex = sorted(set(r[1] for r in records))
        temps = [nearest_value(all_T, t) for t in selected_temps] if selected_temps else all_T
        exs = [nearest_value(all_Ex, e) for e in selected_ex] if selected_ex else [nearest_value(all_Ex, 0.0)]
        for T in temps:
            for Ex in exs:
                candidates = [r for r in records if r[0] == T]
                if not candidates:
                    continue
                wanted.append(min(candidates, key=lambda r: abs(r[1] - Ex))[2])
        # De-duplicate while preserving order.
        seen = set()
        out = []
        for path in wanted:
            if path not in seen:
                out.append(path)
                seen.add(path)
        return out

    raise ValueError(f"Unknown maps mode: {mode}")


def decimated_centroids(C: np.ndarray, max_points: int, seed: int = 12345) -> np.ndarray:
    if C.shape[0] <= max_points:
        return C
    rng = np.random.default_rng(seed)
    idx = rng.choice(C.shape[0], size=max_points, replace=False)
    return C[np.sort(idx)]


def plot_one_map_panel(
    npz_path: Path,
    outdir: Path,
    dpi: int,
    potential_grid: PotentialGrid | None,
    max_scatter_points: int,
    free_energy_max_meV: float | None,
) -> None:
    with np.load(npz_path, allow_pickle=True) as z:
        T = safe_float_from_npz(z, "T_K", default=parse_T_from_dir(npz_path) or np.nan)
        Ex = safe_float_from_npz(z, "Ex", default=parse_Ex_from_dir(npz_path) or np.nan)
        xedges = np.asarray(z["density_xedges"], dtype=float)
        yedges = np.asarray(z["density_yedges"], dtype=float)
        H = np.asarray(z["density_H"], dtype=float)
        P = np.asarray(z["probability_H"], dtype=float) if "probability_H" in z.files else None
        F = np.asarray(z["free_energy_masked"], dtype=float) if "free_energy_masked" in z.files else None
        C = np.asarray(z["centroids"], dtype=float) if "centroids" in z.files else None

    label = f"{format_temperature_label(T)}_{format_ex_label(Ex)}"
    extent = [float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])]

    # Density/counts map
    plt.figure(figsize=(6.2, 5.2))
    plt.imshow(np.log10(H.T + 1.0), origin="lower", aspect="equal", extent=extent)
    plt.colorbar(label="log10(counts + 1)")
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    plt.title(f"Centroid density, T={T:g} K, Ex={Ex:+.4g} eV/nm")
    save_current_figure(outdir / f"{label}_density_log_counts.png", dpi)

    # Probability map
    if P is not None:
        plt.figure(figsize=(6.2, 5.2))
        plt.imshow(P.T, origin="lower", aspect="equal", extent=extent)
        plt.colorbar(label="probability per bin")
        plt.xlabel("x [nm]")
        plt.ylabel("y [nm]")
        plt.title(f"Centroid probability, T={T:g} K, Ex={Ex:+.4g} eV/nm")
        save_current_figure(outdir / f"{label}_probability.png", dpi)

    # Free-energy map in meV
    if F is not None:
        F_meV = F * 1000.0
        if np.any(np.isfinite(F_meV)):
            F_meV = F_meV - np.nanmin(F_meV)
        vmax = free_energy_max_meV
        if vmax is None and np.any(np.isfinite(F_meV)):
            vmax = float(np.nanpercentile(F_meV, 95.0))
        plt.figure(figsize=(6.2, 5.2))
        plt.imshow(F_meV.T, origin="lower", aspect="equal", extent=extent, vmin=0.0, vmax=vmax)
        plt.colorbar(label="F = -kBT ln P [meV]")
        plt.xlabel("x [nm]")
        plt.ylabel("y [nm]")
        plt.title(f"Centroid free energy, T={T:g} K, Ex={Ex:+.4g} eV/nm")
        save_current_figure(outdir / f"{label}_free_energy_meV.png", dpi)

    # Centroid scatter over optional potential contours
    if C is not None and C.ndim == 2 and C.shape[1] == 2:
        Cplot = decimated_centroids(C, max_scatter_points)
        plt.figure(figsize=(6.2, 5.2))
        if potential_grid is not None:
            V_meV = potential_grid.V_eV * 1000.0
            v_extent = [
                float(potential_grid.x_nm[0]), float(potential_grid.x_nm[-1]),
                float(potential_grid.y_nm[0]), float(potential_grid.y_nm[-1]),
            ]
            plt.imshow(V_meV.T, origin="lower", aspect="equal", extent=v_extent, alpha=0.55)
            plt.colorbar(label="V_grid [meV]")
        plt.scatter(Cplot[:, 0], Cplot[:, 1], s=3, alpha=0.45)
        plt.xlabel("x [nm]")
        plt.ylabel("y [nm]")
        plt.title(f"Centroids, T={T:g} K, Ex={Ex:+.4g} eV/nm")
        save_current_figure(outdir / f"{label}_centroids.png", dpi)


def plot_potential_grid(potential_grid: PotentialGrid, outdir: Path, dpi: int) -> None:
    V_meV = potential_grid.V_eV * 1000.0
    extent = [
        float(potential_grid.x_nm[0]), float(potential_grid.x_nm[-1]),
        float(potential_grid.y_nm[0]), float(potential_grid.y_nm[-1]),
    ]
    plt.figure(figsize=(6.2, 5.2))
    plt.imshow(V_meV.T, origin="lower", aspect="equal", extent=extent)
    plt.colorbar(label="V_grid - min(V_grid) [meV]")
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    plt.title("Input atomistic/grid potential")
    save_current_figure(outdir / "input_V_grid_meV.png", dpi)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def write_report(
    out_path: Path,
    results_dir: Path,
    ensemble_df: pd.DataFrame,
    seed_df: pd.DataFrame | None,
    zero_df: pd.DataFrame,
    npz_count: int,
    potential_npz: Path | None,
) -> None:
    lines = []
    lines.append("Atomistic-grid PI-QMC analysis report")
    lines.append("=" * 42)
    lines.append("")
    lines.append(f"Results directory: {results_dir}")
    lines.append(f"Ensemble NPZ files: {npz_count}")
    if potential_npz is not None:
        lines.append(f"Potential grid: {potential_npz}")
    if seed_df is not None:
        lines.append(f"Per-seed CSV rows: {len(seed_df)}")
    lines.append("")

    if not ensemble_df.empty:
        T_vals = sorted(ensemble_df["T_K"].dropna().unique())
        Ex_vals = sorted(ensemble_df["Ex"].dropna().unique())
        lines.append(f"Temperatures [K]: {T_vals}")
        lines.append(f"Field values Ex [eV/nm]: {len(Ex_vals)} values from {min(Ex_vals):+.6g} to {max(Ex_vals):+.6g}")
        lines.append("")

        key_cols = [
            "T_K", "Ex", "mean_x", "r_rms_centroid", "r_rms_spread",
            "A_eff_entropy_nm2", "A_eff_ipr_nm2", "F_p95_meV",
            "acceptance_local", "acceptance_global",
        ]
        present = [c for c in key_cols if c in zero_df.columns]
        if present and not zero_df.empty:
            lines.append("Zero-field / nearest-zero-field transition table:")
            lines.append(zero_df[present].to_string(index=False))
            lines.append("")

        if "acceptance_local" in ensemble_df.columns:
            bad_acc = ensemble_df[(ensemble_df["acceptance_local"] < 0.25) | (ensemble_df["acceptance_local"] > 0.75)]
            if len(bad_acc):
                lines.append("WARNING: Some local acceptance values are far from the rough target window.")
                lines.append(bad_acc[["T_K", "Ex", "acceptance_local"]].to_string(index=False))
                lines.append("")

        if "p_capped" in ensemble_df.columns and np.nanmax(ensemble_df["p_capped"].to_numpy(dtype=float)) > 0:
            capped = ensemble_df[ensemble_df["p_capped"] > 0][["T_K", "Ex", "n_beads"]]
            lines.append("WARNING: P was capped for these entries; low-T imaginary-time convergence may be weak:")
            lines.append(capped.to_string(index=False))
            lines.append("")

    lines.append("Interpretation notes:")
    lines.append("- A_eff_entropy_nm2 and A_eff_ipr_nm2 are centroid-probability spread metrics, not geometric dot sizes.")
    lines.append("- r_rms_spread is the intrinsic ring-polymer quantum spread around the centroid.")
    lines.append("- r_rms_centroid and A_eff_* track centroid delocalisation over the moiré landscape.")
    lines.append("- F_p95_meV is a robust finite-bin free-energy spread; it depends on histogram binning and sampling.")
    lines.append("")

    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze atomistic-grid PI-QMC sweep results."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/atomistic_grid_pimc_v1_0",
        help="Directory containing the runner output_tag results."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional explicit per-seed CSV path. If omitted, the script searches results_dir/*.csv."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output analysis directory. Default: <results_dir>/analysis_v1_0."
    )
    parser.add_argument(
        "--potential_npz",
        type=str,
        default=None,
        help="Optional V_grid.npz for plotting the input potential and centroid overlays."
    )
    parser.add_argument(
        "--coordinate_origin",
        choices=["center", "native"],
        default="center",
        help="How to display the optional potential grid. Use the same origin as the runner."
    )
    parser.add_argument(
        "--maps",
        choices=["none", "zero_field", "selected", "all"],
        default="zero_field",
        help="Which ensemble maps to render. Default: nearest Ex=0 for each temperature."
    )
    parser.add_argument("--selected_temps", type=float, nargs="+", default=None)
    parser.add_argument("--selected_ex", type=float, nargs="+", default=None)
    parser.add_argument("--max_scatter_points", type=int, default=5000)
    parser.add_argument("--free_energy_max_meV", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = resolve_path(args.results_dir)
    if results_dir is None or not results_dir.exists():
        raise SystemExit(f"ERROR: results_dir does not exist: {results_dir}")

    outdir = resolve_path(args.outdir) if args.outdir else results_dir / "analysis_v1_0"
    tables_dir = ensure_dir(outdir / "tables")
    figures_dir = ensure_dir(outdir / "figures")
    maps_dir = ensure_dir(figures_dir / "maps")

    csv_path = resolve_path(args.csv) if args.csv else None
    seed_df = maybe_read_csv(results_dir, csv_path)

    npz_paths = find_ensemble_npzs(results_dir)
    if not npz_paths:
        raise SystemExit(
            f"ERROR: no ensemble results.npz files found under {results_dir}/T_*K/Ex_*/results.npz"
        )

    print(f"Found {len(npz_paths)} ensemble NPZ files.")
    ensemble_df = build_ensemble_table(npz_paths)
    ensemble_csv = tables_dir / "ensemble_observables.csv"
    ensemble_df.to_csv(ensemble_csv, index=False)
    print(f"Wrote {ensemble_csv}")

    seed_summary = build_seed_summary(seed_df)
    if seed_summary is not None:
        seed_summary_csv = tables_dir / "seed_summary_by_T_Ex.csv"
        seed_summary.to_csv(seed_summary_csv, index=False)
        print(f"Wrote {seed_summary_csv}")

    zero_df = zero_field_summary(ensemble_df)
    zero_csv = tables_dir / "transition_summary_zero_field.csv"
    zero_df.to_csv(zero_csv, index=False)
    print(f"Wrote {zero_csv}")

    # Optional potential grid.
    potential_grid = None
    potential_npz = resolve_path(args.potential_npz) if args.potential_npz else None
    if potential_npz is not None:
        if potential_npz.exists():
            try:
                potential_grid = load_potential_grid(potential_npz, coordinate_origin=args.coordinate_origin)
                plot_potential_grid(potential_grid, figures_dir, args.dpi)
                print(f"Loaded and plotted potential grid: {potential_npz}")
            except Exception as exc:
                print(f"WARNING: could not load/plot potential grid {potential_npz}: {exc}", file=sys.stderr)
        else:
            print(f"WARNING: potential_npz does not exist: {potential_npz}", file=sys.stderr)

    # Summary plots.
    line_metrics = [
        "mean_x", "mean_y",
        "r_rms_centroid", "r_rms_spread", "r_rms_bead",
        "A_eff_entropy_nm2", "A_eff_ipr_nm2",
        "F_p95_meV", "mean_V_centroid_eV",
        "acceptance_local", "acceptance_global",
    ]
    for metric in line_metrics:
        plot_metric_vs_ex(ensemble_df, metric, figures_dir, args.dpi)
        plot_metric_vs_T_zero_field(zero_df, metric, figures_dir, args.dpi)

    heatmap_metrics = [
        "mean_x", "r_rms_centroid", "r_rms_spread",
        "A_eff_entropy_nm2", "A_eff_ipr_nm2",
        "hist_pmax", "F_p95_meV",
        "acceptance_local", "mean_V_centroid_eV",
    ]
    for metric in heatmap_metrics:
        plot_heatmap(ensemble_df, metric, figures_dir, args.dpi)

    if seed_df is not None:
        plot_acceptance_from_seed_csv(seed_df, figures_dir, args.dpi)

    selected_maps = select_npzs_for_maps(
        npz_paths=npz_paths,
        mode=args.maps,
        selected_temps=args.selected_temps,
        selected_ex=args.selected_ex,
    )
    print(f"Rendering {len(selected_maps)} map set(s), mode={args.maps}.")
    for path in selected_maps:
        plot_one_map_panel(
            npz_path=path,
            outdir=maps_dir,
            dpi=args.dpi,
            potential_grid=potential_grid,
            max_scatter_points=args.max_scatter_points,
            free_energy_max_meV=args.free_energy_max_meV,
        )

    write_report(
        out_path=outdir / "analysis_report.txt",
        results_dir=results_dir,
        ensemble_df=ensemble_df,
        seed_df=seed_df,
        zero_df=zero_df,
        npz_count=len(npz_paths),
        potential_npz=potential_npz,
    )

    print("\nAnalysis completed.")
    print(f"  Output directory : {outdir}")
    print(f"  Tables           : {tables_dir}")
    print(f"  Figures          : {figures_dir}")
    print(f"  Report           : {outdir / 'analysis_report.txt'}")


if __name__ == "__main__":
    main()
