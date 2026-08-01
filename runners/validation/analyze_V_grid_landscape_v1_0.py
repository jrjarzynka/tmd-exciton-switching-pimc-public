#!/usr/bin/env python3
"""
v1.0 — Audit an atomistic/grid exciton potential V_grid.npz.

Purpose
-------
Before trusting PI-QMC localisation/delocalisation results, we need to know what
landscape was actually sampled. This script reads V_grid.npz and measures:

  - grid shape, bounds, spacing and uniformity
  - V_min, V_max, depth, percentiles and histogram
  - local minima and their coordinates
  - topographic/persistence-style escape barriers from minima
  - dominant spatial wavelengths from the 2D FFT
  - estimated triangular moire cell area from the dominant wavelength
  - comparison of energy scales with k_B T

It is intentionally independent of tmd_pimc, so it can be run before/after the
GridPotential2D patch. It accepts common key names used in the project:

  axes: x_nm/y_nm, x/y, X_nm/Y_nm, X/Y
  V:    V_eV, V, potential_eV, V_grid_eV, V_meV, potential_meV, V_grid_meV

Recommended use from project root:

  python3 runners/validation/analyze_V_grid_landscape_v1_0.py \
    --potential_npz results/wse2_mose2/V_grid.npz \
    --outdir results/wse2_mose2/V_grid_audit_v1_0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KB_EV_PER_K = 8.617333262e-5


# -----------------------------------------------------------------------------
# Robust loading
# -----------------------------------------------------------------------------

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


def _first_existing_key(data: np.lib.npyio.NpzFile, keys: Iterable[str]) -> str | None:
    available = set(data.files)
    for key in keys:
        if key in available:
            return key
    return None


def _extract_axis_or_mesh(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    v_key = _first_existing_key(
        data,
        [
            "V_eV",
            "V",
            "potential_eV",
            "V_grid_eV",
            "energy_eV",
            "V_meV",
            "potential_meV",
            "V_grid_meV",
            "energy_meV",
        ],
    )
    if v_key is None:
        raise KeyError(f"Could not find potential array in NPZ. Available keys: {data.files}")

    V = np.asarray(data[v_key], dtype=float)
    unit_note = "eV"
    if v_key.lower().endswith("mev") or "mev" in v_key.lower():
        V = V / 1000.0
        unit_note = "meV converted to eV"

    # 1D axes are preferred.
    x_key = _first_existing_key(data, ["x_nm", "x", "X_nm", "X", "x_axis_nm", "x_axis"])
    y_key = _first_existing_key(data, ["y_nm", "y", "Y_nm", "Y", "y_axis_nm", "y_axis"])
    if x_key is not None and y_key is not None:
        x_raw = np.asarray(data[x_key], dtype=float)
        y_raw = np.asarray(data[y_key], dtype=float)
        if x_raw.ndim == 1 and y_raw.ndim == 1:
            x = x_raw
            y = y_raw
        elif x_raw.ndim == 2 and y_raw.ndim == 2:
            # Meshgrid case. Try to infer axes.
            if x_raw.shape != V.shape or y_raw.shape != V.shape:
                raise ValueError("2D X/Y mesh arrays do not match V shape.")
            x = x_raw[:, 0]
            if np.allclose(x_raw, x[:, None]):
                y = y_raw[0, :]
            else:
                x = x_raw[0, :]
                y = y_raw[:, 0]
                V = V.T
        else:
            raise ValueError(f"Unsupported axis dimensions: {x_key}{x_raw.shape}, {y_key}{y_raw.shape}")
    else:
        raise KeyError(
            "Could not find x/y axes in NPZ. Expected x_nm/y_nm, x/y, X_nm/Y_nm, or X/Y. "
            f"Available keys: {data.files}"
        )

    if V.ndim != 2:
        raise ValueError(f"Potential array must be 2D, got shape {V.shape}.")

    # Accept either V[nx, ny] or V[ny, nx]. Normalise to V[nx, ny].
    if V.shape == (len(x), len(y)):
        pass
    elif V.shape == (len(y), len(x)):
        V = V.T
    else:
        raise ValueError(
            f"V shape {V.shape} is incompatible with len(x)={len(x)}, len(y)={len(y)}."
        )

    # Make axes increasing; flip V accordingly.
    if len(x) > 1 and x[1] < x[0]:
        x = x[::-1]
        V = V[::-1, :]
    if len(y) > 1 and y[1] < y[0]:
        y = y[::-1]
        V = V[:, ::-1]

    return x.astype(float), y.astype(float), V.astype(float), f"{v_key} ({unit_note})"


def load_grid(path: Path, coordinate_origin: str, subtract_minimum: bool, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=True) as data:
        x, y, V, source_note = _extract_axis_or_mesh(data)
        original_keys = list(data.files)

    V = V * float(scale)
    if subtract_minimum:
        V = V - float(np.nanmin(V))

    origin_shift = (0.0, 0.0)
    if coordinate_origin == "center":
        x_mid = 0.5 * (float(x[0]) + float(x[-1]))
        y_mid = 0.5 * (float(y[0]) + float(y[-1]))
        x = x - x_mid
        y = y - y_mid
        origin_shift = (x_mid, y_mid)
    elif coordinate_origin != "native":
        raise ValueError("coordinate_origin must be 'center' or 'native'")

    meta = {
        "source_note": source_note,
        "original_keys": original_keys,
        "origin_shift_x_nm": origin_shift[0],
        "origin_shift_y_nm": origin_shift[1],
        "scale": float(scale),
        "subtract_minimum": bool(subtract_minimum),
        "coordinate_origin": coordinate_origin,
    }
    return x, y, V, meta


# -----------------------------------------------------------------------------
# Minima and persistence barriers
# -----------------------------------------------------------------------------

@dataclass
class MinimumRecord:
    min_id: int
    i: int
    j: int
    x_nm: float
    y_nm: float
    V_eV: float
    escape_barrier_eV: float | float("nan")
    saddle_eV: float | float("nan")
    basin_cells: int


class UnionFind:
    def __init__(self, n: int, V_flat: np.ndarray):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        self.min_index = np.arange(n, dtype=np.int64)
        self.min_value = V_flat.copy()
        self.active = np.zeros(n, dtype=bool)

    def find(self, a: int) -> int:
        parent = self.parent
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    def activate(self, a: int) -> None:
        self.active[a] = True
        self.parent[a] = a
        self.size[a] = 1
        self.min_index[a] = a

    def union_at_energy(self, a: int, b: int, energy: float, barriers: dict[int, tuple[float, float]]) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra

        # Component with lower minimum survives; the higher minimum dies here.
        va = float(self.min_value[ra])
        vb = float(self.min_value[rb])
        if va < vb or (va == vb and self.size[ra] >= self.size[rb]):
            survivor, loser = ra, rb
        else:
            survivor, loser = rb, ra

        loser_min_idx = int(self.min_index[loser])
        loser_min_val = float(self.min_value[loser])
        if loser_min_idx not in barriers:
            barriers[loser_min_idx] = (float(energy - loser_min_val), float(energy))

        self.parent[loser] = survivor
        self.size[survivor] += self.size[loser]
        # Survivor already has lower minimum.
        return int(survivor)


def _neighbours(i: int, j: int, nx: int, ny: int, periodic: bool) -> Iterable[tuple[int, int]]:
    # 4-neighbour connectivity is more conservative for saddle barriers than 8-neighbour.
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ii = i + di
        jj = j + dj
        if periodic:
            ii %= nx
            jj %= ny
            yield ii, jj
        elif 0 <= ii < nx and 0 <= jj < ny:
            yield ii, jj


def persistence_barriers(V: np.ndarray, periodic: bool) -> tuple[dict[int, tuple[float, float]], np.ndarray, dict[int, int]]:
    """
    Flood landscape from low to high energy. Each minimum becomes a component;
    when components merge, the higher minimum receives a persistence barrier.

    Returns:
      barriers: flat_min_index -> (escape_barrier_eV, saddle_eV)
      root_for_cell: final UF root for each flat cell
      final_basin_size_by_min_index: min flat index -> basin cell count for surviving roots
    """
    nx, ny = V.shape
    N = nx * ny
    V_flat = V.reshape(-1)
    order = np.argsort(V_flat, kind="mergesort")
    uf = UnionFind(N, V_flat)
    barriers: dict[int, tuple[float, float]] = {}

    for flat in order:
        flat = int(flat)
        uf.activate(flat)
        i, j = divmod(flat, ny)
        for ii, jj in _neighbours(i, j, nx, ny, periodic=periodic):
            nb = ii * ny + jj
            if uf.active[nb]:
                uf.union_at_energy(flat, nb, float(V_flat[flat]), barriers)

    # Assign infinite/NaN escape barrier to the global survivor if it never died.
    root_for_cell = np.full(N, -1, dtype=np.int64)
    for idx in range(N):
        if uf.active[idx]:
            root_for_cell[idx] = uf.find(idx)

    basin_size_by_min_index: dict[int, int] = {}
    unique_roots, counts = np.unique(root_for_cell[root_for_cell >= 0], return_counts=True)
    for root, count in zip(unique_roots, counts):
        min_idx = int(uf.min_index[int(root)])
        basin_size_by_min_index[min_idx] = int(count)

    return barriers, root_for_cell, basin_size_by_min_index


def local_minima_mask(V: np.ndarray, periodic: bool) -> np.ndarray:
    local = np.ones_like(V, dtype=bool)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if periodic:
        for di, dj in shifts:
            local &= V <= np.roll(np.roll(V, di, axis=0), dj, axis=1)
    else:
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                v = V[i, j]
                for ii in range(max(0, i - 1), min(V.shape[0], i + 2)):
                    for jj in range(max(0, j - 1), min(V.shape[1], j + 2)):
                        if ii == i and jj == j:
                            continue
                        if V[ii, jj] < v:
                            local[i, j] = False
                            break
                    if not local[i, j]:
                        break
        if V.shape[0] > 2 and V.shape[1] > 2:
            local[0, :] = False
            local[-1, :] = False
            local[:, 0] = False
            local[:, -1] = False
    return local


def greedily_select_minima(
    x: np.ndarray,
    y: np.ndarray,
    V: np.ndarray,
    periodic: bool,
    min_separation_nm: float,
    max_minima: int,
) -> pd.DataFrame:
    mask = local_minima_mask(V, periodic=periodic)
    candidates = np.argwhere(mask)
    if candidates.size == 0:
        candidates = np.array([np.unravel_index(int(np.argmin(V)), V.shape)])
    order = np.argsort(V[candidates[:, 0], candidates[:, 1]])
    candidates = candidates[order]

    selected: list[tuple[int, int]] = []
    for i, j in candidates:
        p = np.array([x[int(i)], y[int(j)]], dtype=float)
        keep = True
        for ii, jj in selected:
            q = np.array([x[ii], y[jj]], dtype=float)
            if float(np.linalg.norm(p - q)) < float(min_separation_nm):
                keep = False
                break
        if keep:
            selected.append((int(i), int(j)))
        if len(selected) >= int(max_minima):
            break

    barriers, _root, basin_sizes = persistence_barriers(V, periodic=periodic)
    rows = []
    ny = V.shape[1]
    for min_id, (i, j) in enumerate(selected):
        flat = i * ny + j
        barrier, saddle = barriers.get(flat, (np.nan, np.nan))
        rows.append(
            {
                "min_id": min_id,
                "i": i,
                "j": j,
                "x_nm": float(x[i]),
                "y_nm": float(y[j]),
                "V_eV": float(V[i, j]),
                "V_meV": float(V[i, j] * 1000.0),
                "escape_barrier_eV": float(barrier),
                "escape_barrier_meV": float(barrier * 1000.0) if np.isfinite(barrier) else np.nan,
                "saddle_eV": float(saddle),
                "saddle_meV": float(saddle * 1000.0) if np.isfinite(saddle) else np.nan,
                "basin_cells_final_survivor_only": int(basin_sizes.get(flat, 0)),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# FFT wavelength estimate
# -----------------------------------------------------------------------------

def fft_peaks(
    x: np.ndarray,
    y: np.ndarray,
    V: np.ndarray,
    n_peaks: int,
    exclude_low_k_bins: int = 2,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = V.shape
    dx = float(np.mean(np.diff(x))) if nx > 1 else 1.0
    dy = float(np.mean(np.diff(y))) if ny > 1 else 1.0
    V0 = V - float(np.mean(V))
    window_x = np.hanning(nx) if nx > 2 else np.ones(nx)
    window_y = np.hanning(ny) if ny > 2 else np.ones(ny)
    Vw = V0 * window_x[:, None] * window_y[None, :]
    F = np.fft.fftshift(np.fft.fft2(Vw))
    power = np.abs(F) ** 2

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))  # cycles/nm
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    kr = np.sqrt(FX**2 + FY**2)

    centre_i = nx // 2
    centre_j = ny // 2
    mask = np.ones_like(power, dtype=bool)
    mask[
        max(0, centre_i - exclude_low_k_bins): min(nx, centre_i + exclude_low_k_bins + 1),
        max(0, centre_j - exclude_low_k_bins): min(ny, centre_j + exclude_low_k_bins + 1),
    ] = False
    mask &= kr > 0

    flat_indices = np.argsort(power[mask].ravel())[::-1]
    candidate_coords = np.argwhere(mask)
    rows = []
    seen: list[tuple[float, float]] = []
    for idx in flat_indices:
        i, j = map(int, candidate_coords[int(idx)])
        fxi = float(FX[i, j])
        fyj = float(FY[i, j])
        k = float(kr[i, j])
        if k <= 0:
            continue
        # De-duplicate Friedel pairs and nearby equivalent peaks.
        skip = False
        for sx, sy in seen:
            if math.hypot(fxi - sx, fyj - sy) < max(abs(fx[1] - fx[0]) if len(fx) > 1 else 0.0, abs(fy[1] - fy[0]) if len(fy) > 1 else 0.0) * 1.5:
                skip = True
                break
            if math.hypot(fxi + sx, fyj + sy) < max(abs(fx[1] - fx[0]) if len(fx) > 1 else 0.0, abs(fy[1] - fy[0]) if len(fy) > 1 else 0.0) * 1.5:
                skip = True
                break
        if skip:
            continue
        seen.append((fxi, fyj))
        rows.append(
            {
                "peak_rank": len(rows) + 1,
                "fx_cycles_per_nm": fxi,
                "fy_cycles_per_nm": fyj,
                "k_cycles_per_nm": k,
                "wavelength_nm": 1.0 / k,
                "power": float(power[i, j]),
            }
        )
        if len(rows) >= n_peaks:
            break

    return pd.DataFrame(rows), FX, FY, power


def estimate_radial_spectrum(FX: np.ndarray, FY: np.ndarray, power: np.ndarray, n_bins: int = 200) -> pd.DataFrame:
    kr = np.sqrt(FX**2 + FY**2).ravel()
    p = power.ravel()
    mask = kr > 0
    kr = kr[mask]
    p = p[mask]
    if len(kr) == 0:
        return pd.DataFrame(columns=["k_cycles_per_nm", "wavelength_nm", "power_mean"])
    bins = np.linspace(float(np.min(kr)), float(np.max(kr)), n_bins + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    sums = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    inds = np.searchsorted(bins, kr, side="right") - 1
    good = (inds >= 0) & (inds < n_bins)
    np.add.at(sums, inds[good], p[good])
    np.add.at(counts, inds[good], 1)
    mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    wavelength = np.divide(1.0, centres, out=np.full_like(centres, np.nan), where=centres > 0)
    return pd.DataFrame({"k_cycles_per_nm": centres, "wavelength_nm": wavelength, "power_mean": mean, "count": counts})


# -----------------------------------------------------------------------------
# Plots and outputs
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_summary_csv(summary: dict, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def plot_v_map(x: np.ndarray, y: np.ndarray, V: np.ndarray, minima: pd.DataFrame, out: Path, max_minima: int) -> None:
    plt.figure(figsize=(7.2, 6.2))
    extent = [float(y[0]), float(y[-1]), float(x[0]), float(x[-1])]
    im = plt.imshow(V * 1000.0, origin="lower", extent=extent, aspect="equal")
    plt.colorbar(im, label="V (meV)")
    if not minima.empty:
        m = minima.head(max_minima)
        plt.scatter(m["y_nm"], m["x_nm"], marker="x", s=35, label="selected minima")
        plt.legend(loc="best")
    plt.xlabel("y (nm)")
    plt.ylabel("x (nm)")
    plt.title("Input V_grid landscape")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_histogram(V: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(6.8, 4.6))
    plt.hist((V.ravel() - float(np.nanmin(V))) * 1000.0, bins=80)
    plt.xlabel("V - V_min (meV)")
    plt.ylabel("grid-cell count")
    plt.title("Potential energy histogram")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_fft_power(FX: np.ndarray, FY: np.ndarray, power: np.ndarray, out: Path) -> None:
    plt.figure(figsize=(6.2, 5.5))
    p = np.log10(power + max(float(np.max(power)) * 1e-12, 1e-300))
    extent = [float(FY.min()), float(FY.max()), float(FX.min()), float(FX.max())]
    im = plt.imshow(p, origin="lower", extent=extent, aspect="equal")
    plt.colorbar(im, label="log10 FFT power")
    plt.xlabel("f_y (cycles/nm)")
    plt.ylabel("f_x (cycles/nm)")
    plt.title("2D FFT power spectrum")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_radial_spectrum(radial: pd.DataFrame, out: Path) -> None:
    if radial.empty:
        return
    d = radial[(radial["wavelength_nm"] > 0) & np.isfinite(radial["power_mean"])]
    if d.empty:
        return
    plt.figure(figsize=(7.0, 4.6))
    plt.plot(d["wavelength_nm"], d["power_mean"])
    plt.xlabel("wavelength (nm)")
    plt.ylabel("mean FFT power")
    plt.title("Radial FFT spectrum")
    plt.xlim(0, np.nanpercentile(d["wavelength_nm"], 95))
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_barriers(minima: pd.DataFrame, out: Path) -> None:
    if minima.empty or "escape_barrier_meV" not in minima:
        return
    d = minima[np.isfinite(minima["escape_barrier_meV"])].copy()
    if d.empty:
        return
    plt.figure(figsize=(6.8, 4.6))
    plt.scatter(d["V_meV"] - float(d["V_meV"].min()), d["escape_barrier_meV"])
    plt.xlabel("minimum energy above lowest minimum (meV)")
    plt.ylabel("escape barrier / persistence (meV)")
    plt.title("Local minima persistence barriers")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def write_report(path: Path, summary: dict, minima: pd.DataFrame, fft_df: pd.DataFrame, temps: list[float]) -> None:
    lines = []
    lines.append("V_grid landscape audit v1.0")
    lines.append("=" * 72)
    lines.append("")
    for key in [
        "potential_npz",
        "grid_shape",
        "bounds_nm",
        "spacing_nm",
        "coordinate_origin",
        "origin_shift_nm",
        "V_min_meV",
        "V_max_meV",
        "V_depth_meV",
        "V_std_meV",
        "n_selected_minima",
        "dominant_wavelength_nm",
        "estimated_triangular_cell_area_nm2",
        "median_escape_barrier_meV",
        "lowest_min_escape_barrier_meV",
    ]:
        lines.append(f"{key}: {summary.get(key, '')}")
    lines.append("")
    lines.append("Energy-scale comparison")
    lines.append("-" * 72)
    for T in temps:
        lines.append(f"T={T:g} K: kBT = {KB_EV_PER_K*T*1000:.3f} meV")
    lines.append("")
    lines.append("Lowest selected minima")
    lines.append("-" * 72)
    if minima.empty:
        lines.append("No minima found.")
    else:
        show_cols = ["min_id", "x_nm", "y_nm", "V_meV", "escape_barrier_meV", "saddle_meV"]
        lines.append(minima.head(12)[show_cols].to_string(index=False))
    lines.append("")
    lines.append("Dominant FFT peaks")
    lines.append("-" * 72)
    if fft_df.empty:
        lines.append("No FFT peaks found.")
    else:
        lines.append(fft_df.head(8).to_string(index=False))
    lines.append("")
    lines.append("Interpretation hints")
    lines.append("-" * 72)
    lines.append("1. Compare kBT with the escape barriers, not only with V_depth.")
    lines.append("2. If kBT at the apparent transition is far below barriers, check sampling/mixing.")
    lines.append("3. If barriers are only a few meV, delocalisation below 100 K is plausible.")
    lines.append("4. Use FFT wavelength/cell area to normalise PI-QMC A_eff: eta = A_eff / A_cell.")
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Audit atomistic/grid moire potential landscape V_grid.npz.")
    p.add_argument("--potential_npz", default="results/wse2_mose2/V_grid.npz")
    p.add_argument("--outdir", default=None)
    p.add_argument("--coordinate_origin", choices=["center", "native"], default="center")
    p.add_argument("--grid_keep_absolute_offset", action="store_true")
    p.add_argument("--grid_scale", type=float, default=1.0)
    p.add_argument("--periodic", action="store_true", help="Use periodic connectivity for minima/barrier analysis.")
    p.add_argument("--max_minima", type=int, default=64)
    p.add_argument("--min_separation_nm", type=float, default=2.0)
    p.add_argument("--fft_peaks", type=int, default=12)
    p.add_argument("--temperature_K", type=float, nargs="+", default=[20, 50, 80, 100, 120, 150])
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    potential_path = resolve_project_path(args.potential_npz)
    if not potential_path.exists():
        raise FileNotFoundError(f"Potential file not found: {potential_path}")

    if args.outdir is None:
        outdir = potential_path.parent / "V_grid_audit_v1_0"
    else:
        outdir = resolve_project_path(args.outdir)
    tables_dir = outdir / "tables"
    figures_dir = outdir / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    x, y, V, meta = load_grid(
        potential_path,
        coordinate_origin=args.coordinate_origin,
        subtract_minimum=not args.grid_keep_absolute_offset,
        scale=args.grid_scale,
    )

    dx = np.diff(x)
    dy = np.diff(y)
    dx_mean = float(np.mean(dx)) if len(dx) else np.nan
    dy_mean = float(np.mean(dy)) if len(dy) else np.nan
    dx_std = float(np.std(dx)) if len(dx) else np.nan
    dy_std = float(np.std(dy)) if len(dy) else np.nan

    minima = greedily_select_minima(
        x=x,
        y=y,
        V=V,
        periodic=args.periodic,
        min_separation_nm=args.min_separation_nm,
        max_minima=args.max_minima,
    )
    minima.to_csv(tables_dir / "selected_local_minima.csv", index=False)

    fft_df, FX, FY, power = fft_peaks(x, y, V, n_peaks=args.fft_peaks)
    fft_df.to_csv(tables_dir / "fft_peaks.csv", index=False)
    radial = estimate_radial_spectrum(FX, FY, power)
    radial.to_csv(tables_dir / "fft_radial_spectrum.csv", index=False)

    percentiles = {p: float(np.nanpercentile(V, p) * 1000.0) for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]}
    pd.DataFrame({"percentile": list(percentiles.keys()), "V_meV": list(percentiles.values())}).to_csv(
        tables_dir / "V_percentiles.csv", index=False
    )

    dominant_wavelength = float(fft_df.iloc[0]["wavelength_nm"]) if not fft_df.empty else np.nan
    cell_area = float(np.sqrt(3.0) / 2.0 * dominant_wavelength**2) if np.isfinite(dominant_wavelength) else np.nan

    finite_barriers = minima["escape_barrier_meV"].to_numpy(dtype=float) if not minima.empty else np.array([])
    finite_barriers = finite_barriers[np.isfinite(finite_barriers)]
    lowest_min_barrier = np.nan
    if not minima.empty:
        lowest_row = minima.sort_values("V_eV").iloc[0]
        lowest_min_barrier = float(lowest_row.get("escape_barrier_meV", np.nan))

    summary = {
        "potential_npz": str(potential_path),
        "source_array": meta["source_note"],
        "npz_keys": json.dumps(meta["original_keys"]),
        "grid_shape": f"{V.shape[0]} x {V.shape[1]}",
        "nx": int(V.shape[0]),
        "ny": int(V.shape[1]),
        "bounds_nm": f"x=[{x[0]:+.6g}, {x[-1]:+.6g}], y=[{y[0]:+.6g}, {y[-1]:+.6g}]",
        "x_min_nm": float(x[0]),
        "x_max_nm": float(x[-1]),
        "y_min_nm": float(y[0]),
        "y_max_nm": float(y[-1]),
        "spacing_nm": f"dx={dx_mean:.6g}±{dx_std:.3g}, dy={dy_mean:.6g}±{dy_std:.3g}",
        "dx_mean_nm": dx_mean,
        "dy_mean_nm": dy_mean,
        "dx_std_nm": dx_std,
        "dy_std_nm": dy_std,
        "coordinate_origin": args.coordinate_origin,
        "origin_shift_nm": f"({meta['origin_shift_x_nm']:+.6g}, {meta['origin_shift_y_nm']:+.6g})",
        "origin_shift_x_nm": meta["origin_shift_x_nm"],
        "origin_shift_y_nm": meta["origin_shift_y_nm"],
        "grid_scale": args.grid_scale,
        "subtract_minimum": not args.grid_keep_absolute_offset,
        "periodic_barrier_analysis": bool(args.periodic),
        "V_min_meV": float(np.nanmin(V) * 1000.0),
        "V_max_meV": float(np.nanmax(V) * 1000.0),
        "V_depth_meV": float((np.nanmax(V) - np.nanmin(V)) * 1000.0),
        "V_mean_meV": float(np.nanmean(V) * 1000.0),
        "V_std_meV": float(np.nanstd(V) * 1000.0),
        "V_p05_meV": percentiles[5],
        "V_p50_meV": percentiles[50],
        "V_p95_meV": percentiles[95],
        "n_selected_minima": int(len(minima)),
        "dominant_wavelength_nm": dominant_wavelength,
        "estimated_triangular_cell_area_nm2": cell_area,
        "median_escape_barrier_meV": float(np.nanmedian(finite_barriers)) if finite_barriers.size else np.nan,
        "mean_escape_barrier_meV": float(np.nanmean(finite_barriers)) if finite_barriers.size else np.nan,
        "min_escape_barrier_meV": float(np.nanmin(finite_barriers)) if finite_barriers.size else np.nan,
        "max_escape_barrier_meV": float(np.nanmax(finite_barriers)) if finite_barriers.size else np.nan,
        "lowest_min_escape_barrier_meV": lowest_min_barrier,
    }
    for T in args.temperature_K:
        summary[f"kBT_meV_at_{T:g}K"] = float(KB_EV_PER_K * T * 1000.0)
        if np.isfinite(summary["median_escape_barrier_meV"]):
            summary[f"median_barrier_over_kBT_at_{T:g}K"] = float(summary["median_escape_barrier_meV"] / (KB_EV_PER_K * T * 1000.0))

    save_summary_csv(summary, tables_dir / "V_grid_landscape_summary.csv")
    (outdir / "V_grid_landscape_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_plots:
        plot_v_map(x, y, V, minima, figures_dir / "V_grid_meV_with_minima.png", max_minima=min(32, args.max_minima))
        plot_histogram(V, figures_dir / "V_grid_histogram_meV.png")
        plot_fft_power(FX, FY, power, figures_dir / "V_grid_fft_power.png")
        plot_radial_spectrum(radial, figures_dir / "V_grid_fft_radial_spectrum.png")
        plot_barriers(minima, figures_dir / "V_grid_minima_barriers.png")

    write_report(
        outdir / "V_grid_landscape_report.txt",
        summary=summary,
        minima=minima,
        fft_df=fft_df,
        temps=[float(t) for t in args.temperature_K],
    )

    print("=" * 72)
    print("V_grid landscape audit completed")
    print("=" * 72)
    print(f"Potential : {potential_path}")
    print(f"Outdir    : {outdir}")
    print(f"V depth   : {summary['V_depth_meV']:.3f} meV")
    print(f"Minima    : {summary['n_selected_minima']}")
    print(f"FFT lambda: {dominant_wavelength:.3f} nm" if np.isfinite(dominant_wavelength) else "FFT lambda: n/a")
    print(f"A_cell    : {cell_area:.3f} nm^2" if np.isfinite(cell_area) else "A_cell    : n/a")
    if np.isfinite(summary["median_escape_barrier_meV"]):
        print(f"Median escape barrier: {summary['median_escape_barrier_meV']:.3f} meV")
    print(f"Report    : {outdir / 'V_grid_landscape_report.txt'}")


if __name__ == "__main__":
    main()
