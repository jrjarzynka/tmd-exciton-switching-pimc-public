#!/usr/bin/env python3
# adaptive_dense_field_scan.py
"""
Adaptacyjny gęsty skan pola Fz (prostopadłego, out-of-plane) + analiza dla
dwuciałowego modelu elektron-dziura.

NOTE (2026-07-22): przełączone z wcześniejszego pola w płaszczyźnie
(V=-q_eff*E.r, addytywny, nieperiodyczny człon) na fizycznie umotywowane
pole prostopadłe Fz, sprzężone przez zależny-od-rejestru efektywny dipol
międzywarstwowy (OutOfPlaneStarkPotential w potentials.py -- ten sam
mechanizm co V_Stark^COM(R;Fz) w Paper 1 Eq. 6, teraz uogólniony na dwa
niezależnie propagujące się ciała). To pole JEST periodyczne w tej samej
sieci co landscape rejestru, więc zawija się tym samym, już zwalidowanym
kernelem periodycznym -- bez potrzeby nowego kodu numerycznego.

Przepływ:
  1) Coarse scan: szeroki zakres Fz, mało seedów (szybkie).
  2) Lokalizacja okna przejścia (gdzie frakcja dysocjacji rośnie od ~p_low do ~p_high).
  3) Refined scan: w oknie przejścia uruchamiamy więcej seedów i (opcjonalnie) dłuższe przebiegi.
  4) Zapis wyników, checkpointing, analiza i wykresy.

Pliki wyjściowe:
  - per-seed CSV: wszystkie metadane i ścieżki do .npz z time series
  - per-seed .npz: rho2_t (średnie po beadach per sample), samples_e/h, meta
  - summary CSV: statystyki per (shift, field)
  - wykresy PNG: dissoc fraction vs field, histograms, survival curves

Wymaga periodycznego samplera (domyślne, --no-use-periodic nieobsługiwane
dla Fz != 0 -- OutOfPlaneStarkPotential nie był walidowany na skończonym
pudełku, patrz worker_run_point).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- ensure local package is importable even when running from scripts/ ---
# Search a few parent levels for numerics/tmd_pimc (robust to runners/<sub>/ nesting).
_here_dir = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = None
_walk = _here_dir
for _ in range(5):
    _candidate = os.path.join(_walk, "numerics")
    if os.path.isdir(os.path.join(_candidate, "tmd_pimc")):
        _CODE_DIR = _candidate
        break
    _legacy = os.path.join(_walk, "code")
    if os.path.isdir(os.path.join(_legacy, "tmd_pimc")):
        _CODE_DIR = _legacy
        break
    _walk = os.path.dirname(_walk)
if _CODE_DIR is None:
    raise ImportError(
        f"Could not locate numerics/tmd_pimc within 5 parent levels of {_here_dir}."
    )
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
# ---------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom


# --- Helpery statystyczne ---------------------------------------------------
def bootstrap_sem(x: np.ndarray, nboot: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(nboot)]
    return float(np.std(means, ddof=1))

def binomial_95_ci(k: int, n: int) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    low, high = binom.interval(0.95, n, k / n)
    return low / n, high / n

def autocorrelation(x: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)
    x = x - x.mean()
    ss = np.dot(x, x)
    if np.isclose(ss, 0.0):
        L = n - 1 if max_lag is None else min(n - 1, int(max_lag))
        out = np.zeros(L + 1, dtype=np.float64)
        out[0] = 1.0
        return out
    n_fft = 1 << int(np.ceil(np.log2(2 * n - 1)))
    fx = np.fft.fft(x, n=n_fft)
    acov = np.fft.ifft(fx * np.conjugate(fx)).real[:n]
    rho = acov / ss
    rho[0] = 1.0
    if max_lag is not None:
        L = min(n - 1, int(max_lag))
        rho = rho[: L + 1]
    return rho

def integrated_autocorrelation_time(rho: np.ndarray, cutoff: str = "first_negative") -> float:
    rho = np.asarray(rho, dtype=np.float64)
    if rho.size <= 1:
        return 0.5
    if cutoff == "first_negative":
        neg = np.where(rho[1:] < 0)[0]
        k_max = neg[0] + 1 if neg.size > 0 else rho.size
        s = np.sum(rho[1:k_max])
        return max(0.5 + float(s), 0.5)
    raise ValueError("Unknown cutoff")

def effective_sample_size(n_samples: int, tau_int: float) -> float:
    if tau_int <= 0:
        return float(n_samples)
    return float(n_samples) / (2.0 * float(tau_int))

# --- Worker: uruchomienie jednego (shift, field, seed) -----------------------
def worker_run_point(args_tuple) -> Dict[str, Any]:
    """
    args_tuple = (config_path, shift_mag, field_mag, seed, output_dir, use_periodic, save_full_samples)
    Zwraca dict z metadanymi i 'ts_path' do zapisanego .npz lub 'error'.
    """
    config_path, shift_mag, field_mag, seed, output_dir, use_periodic, save_full_samples = args_tuple
    out: Dict[str, Any] = {"seed": int(seed), "shift_magnitude_nm": float(shift_mag), "field_eV_per_nm": float(field_mag)}
    try:
        # Wczytaj config
        with open(config_path, "r") as f:
            config = json.load(f)

        # Zbuduj interakcję BLK
        from tmd_pimc.bilayer_keldysh_potential import build_bilayer_keldysh_table, BilayerKeldyshWallPotential
        table = build_bilayer_keldysh_table(
            separation_nm=float(config["separation_nm"]),
            screening_length_layer1_nm=float(config["screening_length_layer1_nm"]),
            screening_length_layer2_nm=float(config["screening_length_layer2_nm"]),
            kappa_environment=float(config["kappa_environment"]),
            r_max_nm=float(config.get("r_max_nm", 80.0)),
            n_log=int(config.get("n_log", 1000)),
            n_linear=int(config.get("n_linear", 2000)),
        )
        interaction = BilayerKeldyshWallPotential(
            bilayer=table,
            wall_radius_nm=float(config.get("wall_radius_nm", 15.0)),
            wall_height_eV=float(config.get("wall_height_eV", 0.08)),
            wall_power=int(config.get("wall_power", 8)),
        )

        # shift (rejestr elektron/dziura); field_mag interpretowane teraz
        # jako Fz (out-of-plane), nie jako pole w plaszczyznie -- nie ma
        # juz "field_axis", Fz nie ma kierunku w plaszczyznie x,y.
        axis = config.get("shift_axis", "x")
        shift_nm = (shift_mag, 0.0) if axis == "x" else (0.0, shift_mag)

        # Potencjaly-placeholdery dla TwoBodyRingPolymerAction (wymaga
        # niepustych Potential2D, ale periodyczny sampler ich NIE czyta --
        # buduje wlasny, polaczony rejestr+Stark landscape wewnetrznie z
        # moire_period_nm/moire_amplitude_eV/origin_*/Fz_eV_per_nm ponizej).
        from tmd_pimc.potentials import MoirePotential
        from tmd_pimc.potential_helpers import ShiftedPotential
        amplitude = float(config["moire_amplitude_eV"])
        period = float(config["moire_period_nm"])
        V_e_placeholder = MoirePotential(amplitude_eV=amplitude, period_nm=period)
        V_h_placeholder = ShiftedPotential(inner=MoirePotential(amplitude_eV=amplitude, period_nm=period), shift_nm=shift_nm)

        # Action
        from tmd_pimc.two_body_action import TwoBodyRingPolymerAction
        action = TwoBodyRingPolymerAction(
            mass_e_m0=float(config["mass_e_m0"]),
            mass_h_m0=float(config["mass_h_m0"]),
            temperature_K=float(config["temperature_K"]),
            n_beads=int(config["n_beads"]),
            potential_e=V_e_placeholder,
            potential_h=V_h_placeholder,
            potential_interaction=interaction,
        )

        # Wybor samplera: periodyczny wrapper jesli wlaczone (domyslnie).
        # NOTE (2026-07-21 reorg): previously this silently fell back to the
        # finite-box TwoBodyPIMCSamplerStagingJIT (landscape_grid_range_nm=40)
        # on ANY exception during periodic-sampler construction, with no
        # record of which backend actually ran. Now a construction failure
        # with use_periodic=True is fatal (fails loudly), and the backend
        # actually used is always recorded as "sampler_backend" below.
        #
        # NOTE (2026-07-22): field_mag now drives Fz_eV_per_nm (out-of-plane
        # Stark), not an in-plane force. The finite-box (--no-use-periodic)
        # backend does not support this -- OutOfPlaneStarkPotential was only
        # validated on the periodic-cell path -- so it is now a hard error
        # rather than a silently different (and unvalidated) physics path.
        if not use_periodic:
            raise RuntimeError(
                "Fz-driven dissociation scan requires the periodic sampler "
                "(OutOfPlaneStarkPotential has not been validated on the "
                "finite-box backend). Do not pass --no-use-periodic for "
                "this script."
            )

        from tmd_pimc.two_body_sampler_periodic_jit import TwoBodyPIMCSamplerStagingPeriodicJIT
        dipole_length_nm = float(config.get("dipole_length_nm", 0.05))
        sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(
            action=action,
            moire_period_nm=period,
            moire_amplitude_eV=amplitude,
            origin_e_nm=(0.0, 0.0),
            origin_h_nm=shift_nm,
            Fz_eV_per_nm=float(field_mag),
            dipole_length_nm=dipole_length_nm,
            rng_seed=int(seed),
            local_step_nm=float(config.get("local_step_nm", 0.15)),
            global_step_nm=float(config.get("global_step_nm", 12.0)),
            global_move_probability=float(config.get("global_move_probability", 0.2)),
            staging_segment_lengths=tuple(config.get("staging_segment_lengths", [4,8,16,32])),
            staging_moves_per_step=int(config.get("staging_moves_per_step", 2)),
            periodic_cell_grid_size=int(config.get("periodic_cell_grid_size", 200)),
        )
        sampler_backend = "periodic"

        # Parametry uruchomienia
        n_steps = int(config.get("n_steps", 60000))
        burn_in = int(config.get("burn_in", 15000))
        sample_every = int(config.get("sample_every", 20))
        period = float(config["moire_period_nm"])
        start_offset = (period / (2.0 * math.sqrt(3.0)), 0.0)

        t0 = time.time()
        result = sampler.run(
            n_steps=n_steps,
            burn_in=burn_in,
            sample_every=sample_every,
            center_e=start_offset,
            center_h=start_offset,
        )
        elapsed_s = time.time() - t0

        samples_e = result["samples_e"]
        samples_h = result["samples_h"]

        # rho2 per sample (średnia po beadach)
        rel = samples_e - samples_h
        rho2_per_bead = np.sum(rel ** 2, axis=-1)  # shape (n_samples, p_beads)
        rho2_t_mean = np.mean(rho2_per_bead, axis=1)  # shape (n_samples,)

        # metryki
        rho2_mean = float(np.mean(rho2_t_mean))
        rho_mean = float(np.mean(np.sqrt(rho2_t_mean)))
        v_int_samples = interaction.value(rel.reshape(-1, 2))
        v_int_mean = float(np.mean(v_int_samples))
        cent_e = samples_e.mean(axis=1)
        cent_h = samples_h.mean(axis=1)
        centroid_sep_mean = float(np.mean(np.linalg.norm(cent_e - cent_h, axis=1)))
        centroid_sep_x_mean = float(np.mean(cent_e[:, 0] - cent_h[:, 0]))
        centroid_e_x_mean = float(np.mean(cent_e[:, 0]))
        centroid_h_x_mean = float(np.mean(cent_h[:, 0]))

        max_abs_coord_nm = float(max(
            np.max(np.abs(samples_e[:, :, 0])), np.max(np.abs(samples_e[:, :, 1])),
            np.max(np.abs(samples_h[:, :, 0])), np.max(np.abs(samples_h[:, :, 1])),
        ))
        # grid_margin_nm only means "distance to an artificial boundary" for
        # the finite_box backend; the periodic backend has no boundary for
        # the landscape+field term, so max_abs_coord_nm being large is
        # expected and not itself a warning sign there.
        if sampler_backend == "finite_box":
            landscape_grid_range_nm = float(config.get("landscape_grid_range_nm", 40.0))
            grid_margin_nm = landscape_grid_range_nm - max_abs_coord_nm
        else:
            grid_margin_nm = float("nan")

        # zapisz time series i snapshoty (opt-in, patrz --save-full-samples --
        # nic w tym repo nie czyta tych plików z powrotem, a potrafią
        # urosnąć do dziesiątek GB na pełny skan)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_path = None
        if save_full_samples:
            ts_name = f"ts_shift{shift_mag:.6f}_field{field_mag:.6e}_seed{seed}.npz"
            ts_path = out_dir / ts_name
            np.savez_compressed(str(ts_path),
                                rho2_t=rho2_t_mean,
                                samples_e=samples_e,
                                samples_h=samples_h,
                                meta={
                                    "shift_magnitude_nm": shift_mag,
                                    "field_eV_per_nm": field_mag,
                                    "seed": seed,
                                    "n_steps": n_steps,
                                    "burn_in": burn_in,
                                    "sample_every": sample_every,
                                })

        out.update({
            "sampler_backend": sampler_backend,
            "rho2_nm2": rho2_mean,
            "rho_nm": rho_mean,
            "v_interaction_meV": v_int_mean * 1000.0,
            "centroid_separation_nm": centroid_sep_mean,
            "centroid_separation_x_nm": centroid_sep_x_mean,
            "centroid_e_x_nm": centroid_e_x_mean,
            "centroid_h_x_nm": centroid_h_x_mean,
            "max_abs_coord_nm": max_abs_coord_nm,
            "grid_margin_nm": grid_margin_nm,
            "acceptance_local_e": float(result.get("acceptance_local_e", np.nan)),
            "acceptance_local_h": float(result.get("acceptance_local_h", np.nan)),
            "acceptance_staging": float(result.get("acceptance_staging", np.nan)),
            "acceptance_global_joint": float(result.get("acceptance_global_joint", np.nan)),
            "n_samples": int(result.get("n_samples", 0)),
            "elapsed_s": float(elapsed_s),
            "ts_path": str(ts_path),
        })
    except Exception as exc:
        tb = traceback.format_exc()
        # natychmiastowy, czytelny komunikat w logu procesu (stderr)
        print(f"[Seed {seed}] Worker Error: {exc}", file=sys.stderr)
        print(tb, file=sys.stderr)
        # zapisz pełny traceback do CSV/wyjścia workera
        out["error"] = tb

    return out

# --- Orkiestrator: coarse -> refine -> analiza ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Adaptive dense field scan + analysis")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--output-dir", default="adaptive_dense_results", help="Output directory")
    parser.add_argument("--field-min", type=float, required=True, help="Min Fz, out-of-plane field (eV/nm)")
    parser.add_argument("--field-max", type=float, required=True, help="Max Fz, out-of-plane field (eV/nm)")
    parser.add_argument("--coarse-step", type=float, required=True, help="Coarse step (eV/nm)")
    parser.add_argument("--coarse-seeds", type=int, default=8, help="Seeds per coarse point")
    parser.add_argument("--refine-window_frac", type=float, default=0.2, help="Fraction of coarse range to refine around transition")
    parser.add_argument("--refine-step", type=float, default=0.0005, help="Refined step (eV/nm)")
    parser.add_argument("--refine-seeds", type=int, default=40, help="Seeds per refined point")
    parser.add_argument("--dissoc-threshold", type=float, default=100.0, help="rho2 threshold for dissociation")
    parser.add_argument("--dissoc_frac_low", type=float, default=0.1, help="lower fraction to define transition start")
    parser.add_argument("--dissoc_frac_high", type=float, default=0.9, help="upper fraction to define transition end")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers")
    parser.add_argument("--use-periodic", dest="use_periodic", action="store_true", default=True,
                         help="Use the periodic-cell sampler (default: True; required for this "
                              "script's Fz-driven Stark coupling, see module docstring).")
    parser.add_argument("--no-use-periodic", dest="use_periodic", action="store_false",
                         help="NOT SUPPORTED by this script: OutOfPlaneStarkPotential has not "
                              "been validated on the finite-box backend, so passing this raises "
                              "a RuntimeError rather than silently using an unvalidated path.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output-dir (checkpointing)")
    parser.add_argument(
        "--save-full-samples", dest="save_full_samples", action="store_true", default=False,
        help="Also write per-seed ts_*.npz files containing the full samples_e/samples_h "
             "trajectory arrays. OFF by default: nothing downstream in this repo reads "
             "these files (confirmed via grep), and they balloon to tens of GB across a "
             "full adaptive scan (46 GB observed for a single production rerun) while "
             "every aggregate quantity actually used (rho2, centroid separation, "
             "dissoc_frac, etc.) is already written to the per-seed/summary CSVs "
             "regardless of this flag. Enable only if you specifically need raw "
             "per-bead trajectories for offline inspection of a handful of points.",
    )
    args = parser.parse_args()

    cfg_path = args.config
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load base config to get shifts
    with open(cfg_path, "r") as f:
        base_cfg = json.load(f)
    shift_list = [float(v) for v in base_cfg.get("shift_values_nm", [0.0])]

    # prepare coarse field grid
    field_vals_coarse = np.arange(args.field_min, args.field_max + 1e-12, args.coarse_step)
    print(f"Coarse grid: {len(field_vals_coarse)} fields, shifts: {len(shift_list)}")

    # checkpoint: load existing per-seed CSV if resume
    per_seed_csv = out_dir / "adaptive_per_seed.csv"
    existing_results: List[Dict[str, Any]] = []
    if args.resume and per_seed_csv.exists():
        with open(per_seed_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # convert numeric fields where possible
                parsed = {}
                for k, v in row.items():
                    if v == "":
                        parsed[k] = None
                        continue
                    try:
                        parsed[k] = float(v)
                        # if integer-like, cast to int for seed/n_samples
                        if k in ("seed", "n_samples"):
                            parsed[k] = int(parsed[k])
                    except Exception:
                        parsed[k] = v
                existing_results.append(parsed)
        print(f"Loaded {len(existing_results)} existing per-seed rows from checkpoint")

    # helper to check if (shift, field, seed) already done
    def already_done(shift, field, seed):
        for r in existing_results:
            if (r.get("seed") == seed and
                np.isclose(float(r.get("shift_magnitude_nm", np.nan)), float(shift), atol=1e-9) and
                np.isclose(float(r.get("field_eV_per_nm", np.nan)), float(field), atol=1e-12)):
                return True
        return False

    # run coarse scan
    tasks = []
    for shift in shift_list:
        for field in field_vals_coarse:
            for s_off in range(args.coarse_seeds):
                seed = 20000 + int(round(shift * 1000)) + int(round(field * 1e7)) + s_off
                if args.resume and already_done(shift, field, seed):
                    continue
                tasks.append((cfg_path, shift, float(field), seed, str(out_dir), args.use_periodic, args.save_full_samples))

    print(f"Submitting {len(tasks)} coarse tasks with {args.workers} workers")
    coarse_results: List[Dict[str, Any]] = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(worker_run_point, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            coarse_results.append(res)
            # append to checkpoint file incrementally
            existing_results.append(res)
            # write incremental CSV
            keys = sorted({k for r in existing_results for k in r.keys()})
            with open(per_seed_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(existing_results)
            if i % 10 == 0 or i == len(futures):
                print(f"Coarse completed {i}/{len(futures)}")

    elapsed = time.time() - start
    print(f"Coarse scan finished in {elapsed:.1f}s")

    # aggregate coarse summary: compute dissoc fraction per (shift, field)
    summary_coarse = []
    for shift in shift_list:
        for field in field_vals_coarse:
            matching = [r for r in existing_results
                        if np.isclose(float(r.get("shift_magnitude_nm", np.nan)), shift, atol=1e-9)
                        and np.isclose(float(r.get("field_eV_per_nm", np.nan)), float(field), atol=1e-12)
                        and "rho2_nm2" in r]
            n = len(matching)
            if n == 0:
                continue
            rho2_vals = np.array([float(r["rho2_nm2"]) for r in matching])
            diss_flags = np.array([1 if float(r["rho2_nm2"]) > args.dissoc_threshold else 0 for r in matching])
            k = int(diss_flags.sum())
            frac = k / n
            ci_low, ci_high = binomial_95_ci(k, n)
            summary_coarse.append({
                "shift": shift, "field": float(field), "n": n,
                "diss_frac": frac, "ci_low": ci_low, "ci_high": ci_high,
                "mean_rho2": float(np.mean(rho2_vals)),
            })

    # find transition windows where diss_frac crosses [dissoc_frac_low, dissoc_frac_high]
    refine_windows: List[Tuple[float, float]] = []
    for shift in shift_list:
        rows = sorted([r for r in summary_coarse if r["shift"] == shift], key=lambda x: x["field"])
        if not rows:
            continue
        # find first field where diss_frac >= low and first where >= high
        low_idx = next((i for i, rr in enumerate(rows) if rr["diss_frac"] >= args.dissoc_frac_low), None)
        high_idx = next((i for i, rr in enumerate(rows) if rr["diss_frac"] >= args.dissoc_frac_high), None)
        if low_idx is None:
            continue
        if high_idx is None:
            # extend window a bit beyond low point
            f_low = rows[low_idx]["field"]
            delta = args.refine_window_frac * (args.field_max - args.field_min)
            refine_windows.append((max(args.field_min, f_low - delta), min(args.field_max, f_low + delta)))
        else:
            f_low = rows[low_idx]["field"]
            f_high = rows[high_idx]["field"]
            pad = args.refine_window_frac * (f_high - f_low + 1e-12)
            refine_windows.append((max(args.field_min, f_low - pad), min(args.field_max, f_high + pad)))

    # merge overlapping windows
    refine_windows_sorted = sorted(refine_windows, key=lambda x: x[0])
    merged_windows: List[Tuple[float, float]] = []
    for w in refine_windows_sorted:
        if not merged_windows:
            merged_windows.append(w)
        else:
            a, b = merged_windows[-1]
            if w[0] <= b + 1e-12:
                merged_windows[-1] = (a, max(b, w[1]))
            else:
                merged_windows.append(w)
    print(f"Refine windows: {merged_windows}")

    # prepare refined grid tasks
    refine_tasks = []
    for shift in shift_list:
        for (f0, f1) in merged_windows:
            field_vals_ref = np.arange(f0, f1 + 1e-12, args.refine_step)
            for field in field_vals_ref:
                for s_off in range(args.refine_seeds):
                    seed = 30000 + int(round(shift * 1000)) + int(round(field * 1e7)) + s_off
                    if args.resume and already_done(shift, field, seed):
                        continue
                    refine_tasks.append((cfg_path, shift, float(field), seed, str(out_dir), args.use_periodic, args.save_full_samples))

    print(f"Submitting {len(refine_tasks)} refine tasks")
    refine_results: List[Dict[str, Any]] = []
    if refine_tasks:
        start = time.time()
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(worker_run_point, t): t for t in refine_tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                refine_results.append(res)
                existing_results.append(res)
                # incremental checkpoint write
                keys = sorted({k for r in existing_results for k in r.keys()})
                with open(per_seed_csv, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(existing_results)
                if i % 10 == 0 or i == len(futures):
                    print(f"Refine completed {i}/{len(futures)}")
        elapsed = time.time() - start
        print(f"Refined scan finished in {elapsed:.1f}s")
    else:
        print("No refine tasks (no transition windows found)")

    # Final aggregation and summary CSV
    summary_rows = []
    for shift in shift_list:
        # collect all unique fields present in results for this shift
        fields = sorted({float(r["field_eV_per_nm"]) for r in existing_results if np.isclose(float(r.get("shift_magnitude_nm", np.nan)), shift, atol=1e-9)})
        for field in fields:
            matching = [r for r in existing_results
                        if np.isclose(float(r.get("shift_magnitude_nm", np.nan)), shift, atol=1e-9)
                        and np.isclose(float(r.get("field_eV_per_nm", np.nan)), float(field), atol=1e-12)
                        and "rho2_nm2" in r]
            n = len(matching)
            if n == 0:
                continue
            rho2_vals = np.array([float(r["rho2_nm2"]) for r in matching])
            mean_rho2 = float(np.mean(rho2_vals))
            sem_boot = bootstrap_sem(rho2_vals, nboot=2000, seed=0)
            diss_flags = np.array([1 if float(r["rho2_nm2"]) > args.dissoc_threshold else 0 for r in matching])
            k = int(diss_flags.sum())
            frac = k / n
            ci_low, ci_high = binomial_95_ci(k, n)
            # median time to dissociation (samples -> steps)
            times_to_diss = []
            for r in matching:
                ts_path = r.get("ts_path")
                if ts_path and Path(ts_path).exists():
                    data = np.load(ts_path)
                    rho2_t = data["rho2_t"]
                    M = 3
                    consec = 0
                    found = None
                    for idx, v in enumerate(rho2_t):
                        if v > args.dissoc_threshold:
                            consec += 1
                            if consec >= M:
                                found = idx
                                break
                        else:
                            consec = 0
                    if found is not None:
                        sample_every = int(base_cfg.get("sample_every", 20))
                        burn_in = int(base_cfg.get("burn_in", 15000))
                        t_steps = burn_in + found * sample_every
                        times_to_diss.append(t_steps)
            median_time = float(np.median(times_to_diss)) if times_to_diss else float("nan")

            summary_rows.append({
                "shift_nm": float(shift),
                "field_eV_per_nm": float(field),
                "n_seeds": n,
                "mean_rho2_nm2": mean_rho2,
                "sem_rho2_boot": sem_boot,
                "dissoc_frac": frac,
                "dissoc_frac_ci_low": ci_low,
                "dissoc_frac_ci_high": ci_high,
                "median_time_to_diss_steps": median_time,
            })

    # write summary CSV
    summary_csv = out_dir / "adaptive_summary.csv"
    if summary_rows:
        keys = sorted(summary_rows[0].keys())
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Wrote summary CSV: {summary_csv}")
    else:
        print("No summary rows to write")

    # Quick plots: dissoc fraction vs field per shift
    try:
        for shift in shift_list:
            rows = sorted([r for r in summary_rows if np.isclose(r["shift_nm"], shift)], key=lambda x: x["field_eV_per_nm"])
            if not rows:
                continue
            fields = [r["field_eV_per_nm"] for r in rows]
            fracs = [r["dissoc_frac"] for r in rows]
            lows = [r["dissoc_frac_ci_low"] for r in rows]
            highs = [r["dissoc_frac_ci_high"] for r in rows]
            plt.errorbar(fields, fracs, yerr=[np.array(fracs)-np.array(lows), np.array(highs)-np.array(fracs)],
                         fmt='o-', label=f"shift={shift:.3f} nm")
        plt.xlabel("Field (eV/nm)")
        plt.ylabel("Dissociation fraction")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = out_dir / "adaptive_dissoc_fraction_vs_field.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Wrote plot: {plot_path}")
    except Exception as exc:
        print(f"Plotting failed: {exc}")

    print("Adaptive scan complete.")

if __name__ == "__main__":
    main()
