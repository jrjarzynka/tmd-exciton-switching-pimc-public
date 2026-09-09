#!/usr/bin/env python3
"""Audit final Paper-1 NPZ files and plot primitive-cell centroid densities."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


GRID_SHA256 = "8fbd5d9a9e21c3f7b3655034bf044d396b37f1af8056ae64626fae783564dd65"
SOURCE_COMMIT = "c8719094355fe7a8d23039c2254e11e77131459d"
TEMPERATURES = (5.0, 10.0, 15.0, 20.0)
EXPECTED_CHAINS_PER_T = 10
EXPECTED_RETAINED = 30_000
REQUIRED_NPZ_ARRAYS = {
    "samples_unwrapped_nm", "centroids_unwrapped_nm", "centroids_wrapped_fractional",
    "potential_series_eV", "spring_proxy_series_eV", "bond_sum_series_nm2",
    "R_g2_series_nm2", "q1_series", "q2_series", "q3_series", "basin_labels",
    "fractional_cell_histogram_16x16", "staging_length_attempts",
    "staging_length_accepted", "metadata_json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_new_or_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output must be new or empty: {path}")


def circular_summary(values: np.ndarray) -> dict:
    phase = np.exp(2j * np.pi * values)
    mean_phase = np.mean(phase)
    return {
        "mean_fractional": float((np.angle(mean_phase) / (2 * np.pi)) % 1.0),
        "resultant_magnitude": float(abs(mean_phase)),
    }


def histogram_probability(uv: np.ndarray, bins: int) -> np.ndarray:
    counts = np.histogram2d(uv[:, 0], uv[:, 1], bins=bins, range=((0, 1), (0, 1)))[0]
    return counts / counts.sum()


def total_variation(first: np.ndarray, second: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(first - second)))


def classify_basins(uv: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=2)), dtype=float)
    distances = []
    for minimum in (np.array([1 / 3, 1 / 3]), np.array([2 / 3, 2 / 3])):
        delta_uv = uv[:, None, :] - (minimum[None, None, :] + shifts[None, :, :])
        delta_real = delta_uv @ matrix.T
        distances.append(np.min(np.sum(delta_real * delta_real, axis=2), axis=1))
    return np.argmin(np.column_stack(distances), axis=1)


def geometry_audit(grid: dict[str, np.ndarray]) -> dict:
    vectors = np.asarray(grid["lattice_vectors_nm"], dtype=float)
    lengths = np.linalg.norm(vectors, axis=1)
    cosine = np.dot(vectors[0], vectors[1]) / np.prod(lengths)
    angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    determinant = float(np.linalg.det(np.column_stack([vectors[0], vectors[1]])))
    equal_rhombus = bool(np.isclose(lengths[0], lengths[1], rtol=1e-12, atol=1e-12))
    return {
        "lattice_vectors_nm": vectors.tolist(),
        "origin_nm": np.asarray(grid["origin_nm"], dtype=float).tolist(),
        "length_a1_nm": float(lengths[0]), "length_a2_nm": float(lengths[1]),
        "angle_a1_a2_deg": angle, "primitive_cell_area_nm2": abs(determinant),
        "determinant_nm2": determinant,
        "determinant_orientation": "positive/counterclockwise" if determinant > 0 else "negative/clockwise",
        "u_fractional_range": [float(grid["u_fractional"][0]), float(grid["u_fractional"][-1])],
        "v_fractional_range": [float(grid["v_fractional"][0]), float(grid["v_fractional"][-1])],
        "endpoint_excluded": bool(np.asarray(grid["endpoint_excluded"]).item()),
        "grid_shape": list(grid["V_eV"].shape),
        "V_eV_min": float(np.min(grid["V_eV"])),
        "V_eV_max": float(np.max(grid["V_eV"])),
        "V_eV_peak_to_peak": float(np.ptp(grid["V_eV"])),
        "inferred_moire_period_nm": float(np.mean(lengths)) if equal_rhombus else None,
        "period_inference": ("common primitive-vector length of the 60-degree rhombus"
                             if equal_rhombus and np.isclose(angle, 60.0) else "not unique"),
    }


def load_grid(path: Path) -> dict[str, np.ndarray]:
    if sha256_file(path) != GRID_SHA256:
        raise ValueError("grid SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def audit_campaign(campaign: Path, grid_path: Path, bins: int):
    grid = load_grid(grid_path)
    matrix = np.column_stack([grid["lattice_vectors_nm"][0], grid["lattice_vectors_nm"][1]])
    inverse = np.linalg.inv(matrix)
    preflight = json.loads((campaign / "PREFLIGHT.json").read_text(encoding="utf-8"))
    complete = json.loads((campaign / "CAMPAIGN_COMPLETE.json").read_text(encoding="utf-8"))
    expected_tasks = {row["task_id"]: row for row in preflight["task_inventory"]}
    npz_paths = sorted(campaign.glob("paper1_T*.npz"))
    if len(expected_tasks) != 40 or len(npz_paths) != 40:
        raise ValueError(f"expected 40 inventory tasks and NPZ files; got {len(expected_tasks)}, {len(npz_paths)}")
    actual_ids = {path.stem for path in npz_paths}
    if actual_ids != set(expected_tasks) or set(complete["completed_task_ids"]) != set(expected_tasks):
        raise ValueError("NPZ, preflight, and completion inventories disagree")
    if preflight["grid_sha256"] != GRID_SHA256 or complete["grid_sha256"] != GRID_SHA256:
        raise ValueError("campaign-level grid hash mismatch")

    pooled = {temperature: [] for temperature in TEMPERATURES}
    chain_rows = []
    max_centroid_difference = 0.0
    max_fractional_difference = 0.0
    for path in npz_paths:
        task = expected_tasks[path.stem]
        provenance_path = path.with_suffix(".json")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        actual_hash = sha256_file(path)
        if provenance["npz_sha256"] != actual_hash:
            raise ValueError(f"NPZ provenance hash mismatch: {path.name}")
        if provenance["grid_sha256"] != GRID_SHA256:
            raise ValueError(f"chain grid hash mismatch: {path.name}")
        if provenance["source"]["source_commit"] != SOURCE_COMMIT:
            raise ValueError(f"chain source commit mismatch: {path.name}")
        if provenance["retained"] != EXPECTED_RETAINED or not provenance["production"] or provenance["smoke_test"]:
            raise ValueError(f"chain metadata is not final production: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            missing = REQUIRED_NPZ_ARRAYS.difference(archive.files)
            if missing:
                raise ValueError(f"missing arrays in {path.name}: {sorted(missing)}")
            samples = np.asarray(archive["samples_unwrapped_nm"])
            expected_shape = (EXPECTED_RETAINED, int(task["P"]), 2)
            if samples.shape != expected_shape:
                raise ValueError(f"wrong raw shape in {path.name}: {samples.shape}")
            centroid = np.mean(samples, axis=1)
            uv_unwrapped = (centroid - grid["origin_nm"]) @ inverse.T
            uv = uv_unwrapped - np.floor(uv_unwrapped)
            max_centroid_difference = max(
                max_centroid_difference,
                float(np.max(np.abs(centroid - archive["centroids_unwrapped_nm"]))),
            )
            wrapped_delta = uv - archive["centroids_wrapped_fractional"]
            wrapped_delta -= np.rint(wrapped_delta)
            max_fractional_difference = max(
                max_fractional_difference, float(np.max(np.abs(wrapped_delta)))
            )
        temperature = float(task["T_K"])
        pooled[temperature].append({"task": task, "uv": uv,
                                    "histogram": histogram_probability(uv, bins),
                                    "npz": path.name, "npz_sha256": actual_hash})
        chain_rows.append({
            "task_id": path.stem, "T_K": temperature, "P": int(task["P"]), "L": int(task["L"]),
            "seed": int(task["seed"]), "start_basin": task["start_basin"],
            "retained": EXPECTED_RETAINED, "raw_shape": list(expected_shape),
            "npz": path.name, "json": provenance_path.name, "npz_sha256": actual_hash,
        })

    diagnostics = {}
    histograms = {}
    pooled_uv = {}
    for temperature in TEMPERATURES:
        chains = pooled[temperature]
        if len(chains) != EXPECTED_CHAINS_PER_T:
            raise ValueError(f"expected 10 chains at {temperature:g} K")
        starts = [row["task"]["start_basin"] for row in chains]
        if starts.count("A") != 5 or starts.count("B") != 5:
            raise ValueError(f"start-basin imbalance at {temperature:g} K")
        uv = np.concatenate([row["uv"] for row in chains])
        histogram = histogram_probability(uv, bins)
        histograms[temperature], pooled_uv[temperature] = histogram, uv
        labels = classify_basins(uv, matrix)
        hist_a = histogram_probability(np.concatenate([row["uv"] for row in chains if row["task"]["start_basin"] == "A"]), bins)
        hist_b = histogram_probability(np.concatenate([row["uv"] for row in chains if row["task"]["start_basin"] == "B"]), bins)
        pairwise = [total_variation(x["histogram"], y["histogram"])
                    for x, y in itertools.combinations(chains, 2)]
        diagnostics[str(int(temperature))] = {
            "T_K": temperature, "pooled_centroid_count": int(len(uv)),
            "pA": float(np.mean(labels == 0)), "pB": float(np.mean(labels == 1)),
            "histogram_bins": [bins, bins], "histogram_probability_sum": float(histogram.sum()),
            "circular_u": circular_summary(uv[:, 0]), "circular_v": circular_summary(uv[:, 1]),
            "start_A_vs_start_B_histogram_TV": total_variation(hist_a, hist_b),
            "chain_pairwise_histogram_TV": {
                "pair_count": len(pairwise), "mean": float(np.mean(pairwise)),
                "std": float(np.std(pairwise, ddof=1)), "median": float(np.median(pairwise)),
                "min": float(np.min(pairwise)), "max": float(np.max(pairwise)),
            },
        }
    verification = {
        "status": "VERIFIED", "npz_count": len(npz_paths), "chains_per_temperature": {
            str(int(t)): len(pooled[t]) for t in TEMPERATURES},
        "retained_per_chain": EXPECTED_RETAINED, "total_retained": len(npz_paths) * EXPECTED_RETAINED,
        "all_npz_hashes_match_json": True, "all_grid_hashes_match": True,
        "preflight_inventory_complete": True, "campaign_complete_inventory_match": True,
        "source_commit": SOURCE_COMMIT, "max_stored_vs_recomputed_centroid_abs_difference_nm": max_centroid_difference,
        "max_stored_vs_recomputed_wrapped_fractional_difference": max_fractional_difference,
    }
    return grid, histograms, pooled_uv, diagnostics, chain_rows, verification


def plot_density(histograms: dict, bins: int, output: Path) -> list[Path]:
    positives = np.concatenate([hist[hist > 0] for hist in histograms.values()])
    norm = LogNorm(vmin=float(np.min(positives)), vmax=float(max(np.max(x) for x in histograms.values())))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")
    edges = np.linspace(0, 1, bins + 1)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(7.4, 6.8))
    layout = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.055), wspace=0.18, hspace=0.32)
    axes = np.array([
        [fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[0, 1])],
        [fig.add_subplot(layout[1, 0]), fig.add_subplot(layout[1, 1])],
    ])
    colorbar_axis = fig.add_subplot(layout[:, 2])
    image = None
    for label, temperature, axis in zip("abcd", TEMPERATURES, axes.flat):
        masked = np.ma.masked_equal(histograms[temperature].T, 0.0)
        image = axis.pcolormesh(edges, edges, masked, cmap=cmap, norm=norm, shading="flat")
        axis.set_aspect("equal")
        axis.set_title(f"({label})  $T={temperature:g}$ K", loc="left", pad=5)
        axis.set_xlim(0, 1); axis.set_ylim(0, 1)
        axis.set_xticks((0, 0.25, 0.5, 0.75, 1))
        axis.set_yticks((0, 0.25, 0.5, 0.75, 1))
    axes[1, 0].set_xlabel("Fractional coordinate $u$")
    axes[1, 1].set_xlabel("Fractional coordinate $u$")
    axes[0, 0].set_ylabel("Fractional coordinate $v$")
    axes[1, 0].set_ylabel("Fractional coordinate $v$")
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(f"Centroid probability per ${bins}\\times{bins}$ bin")
    paths = [output / "fig_paper1_final_primitive_cell_centroids_2x2.pdf",
             output / "fig_paper1_final_primitive_cell_centroids_2x2.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_marginals(pooled_uv: dict, bins: int, output: Path) -> list[Path]:
    edges = np.linspace(0, 1, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), sharex=True, constrained_layout=True)
    for temperature, axis in zip(TEMPERATURES, axes.flat):
        uv = pooled_uv[temperature]
        density_u = np.histogram(uv[:, 0], bins=edges, density=True)[0]
        density_v = np.histogram(uv[:, 1], bins=edges, density=True)[0]
        axis.plot(centers, density_u, label="$u$", lw=1.25)
        axis.plot(centers, density_v, label="$v$", lw=1.25)
        axis.set_title(f"$T={temperature:g}$ K")
        axis.set_xlim(0, 1); axis.set_ylabel("Marginal density")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].set_xlabel("Wrapped fractional coordinate")
    axes[1, 1].set_xlabel("Wrapped fractional coordinate")
    paths = [output / "diagnostic_paper1_final_centroid_marginals.pdf",
             output / "diagnostic_paper1_final_centroid_marginals.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=400, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()
    if args.bins < 2:
        raise ValueError("--bins must be at least 2")
    require_new_or_empty(args.out)
    grid, histograms, pooled_uv, diagnostics, inventory, verification = audit_campaign(
        args.campaign.resolve(), args.grid.resolve(), args.bins)
    args.out.mkdir(parents=True, exist_ok=True)
    figures = plot_density(histograms, args.bins, args.out)
    marginals = plot_marginals(pooled_uv, args.bins, args.out)
    payload = {
        "classification": "postproduction_analysis_no_new_MC",
        "campaign": str(args.campaign.resolve()), "grid": str(args.grid.resolve()),
        "grid_sha256": GRID_SHA256, "histogram_resolution": [args.bins, args.bins],
        "verification": verification, "primitive_cell_geometry": geometry_audit(grid),
        "diagnostics_by_temperature": diagnostics, "inventory": inventory,
        "figure_files": [path.name for path in figures],
        "optional_marginal_files": [path.name for path in marginals],
    }
    report = args.out / "paper1_final_centroid_audit.json"
    with report.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
