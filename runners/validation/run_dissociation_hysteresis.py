#!/usr/bin/env python3
"""
run_dissociation_hysteresis.py

Bound the equilibrium dissociation fraction from both sides, and record the
rho^2(t) trace of every seed.

Why initialise from both states
-------------------------------
run_dissociation_time_convergence.py showed the apparent threshold sliding from
3.71 to 3.27 eV/nm as the run lengthened, with no sign of stopping. That leaves
the measurement without a value: Monte Carlo steps are not physical time -- the
sampler is a device for drawing from the equilibrium distribution, not a model
of the dynamics -- so "the threshold at 180k steps" is a statement about the
algorithm, not about the material.

Starting half the seeds bound and half already separated fixes this. At
equilibrium the two must agree; while they disagree they BRACKET the answer,
which is a rigorous statement that a one-sided run cannot make. Where the two
branches cross is the equilibrium field, independent of run length. This is the
standard heating/cooling construction used at first-order transitions.

The reverse branch also supplies a control that no forward-only run can. If
seeds started in the separated state never return to bound even at a field well
below any plausible transition -- where the bound state is unambiguously
favoured -- then "escape is irreversible" is a property of the move set, not of
the physics, and the whole forward measurement is uninterpretable. Without that
control the irreversibility hypothesis cannot be falsified.

Starting configurations
-----------------------
The moire term V0 sum cos(G.r) and the Stark texture sum sin(G.r) are built on
the SAME three reciprocal vectors, so the field does not move the minima; it
only tilts their relative depths and thereby selects which sublattice each
carrier prefers. Numerically the effective minima are separated by 11.560 nm at
every field from 1.8 to 4.0 eV/nm, against L/sqrt(3) = 11.547 nm at zero field.
The "field-relaxed" and "geometric nearest-neighbour" separated states are
therefore the same configuration, and the separation of the dissociated state
is fixed by the lattice rather than by the field -- which is why <rho^2>
saturates at ~133 nm^2 however hard the field is driven.

A second separated start along a different C3 direction is included as a
control: agreeing branches show the choice of neighbour does not matter,
disagreeing ones would indicate the sampler cannot rotate between equivalent
separated configurations.

Traces
------
sampler.run() already returns samples of shape (n_samples, P, 2), so rho^2 per
sample IS a time series -- no kernel change is needed, only reducing over the
bead axis instead of over everything. A seed that ratchets once from 4 to 133
nm^2 and stays there is escaping; one that oscillates between them is mixing.
Only the latter is sampling equilibrium.

Usage
-----
    python run_dissociation_hysteresis.py \\
        --config configs/two_body/calib_dft_shift0000.json \\
        --fields 1.8 2.1 2.4 2.7 3.0 3.4 \\
        --n-steps 180000 --n-seeds 12 \\
        --out results/hysteresis_s0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Optional, Sequence

import numpy as np


DEFAULT_THRESHOLD_NM2 = 30.0
DEFAULT_BURN_IN_FRACTION = 0.25


def landscape_minima(period_nm: float, amplitude_eV: float, dipole_nm: float,
                     Fz: float, n_grid: int = 1201):
    """Locate the electron and hole landscape minima on a grid.

    Returned positions are used only to place the starting configurations, so
    grid resolution of a few pm is ample. Separations are computed directly and
    never minimum-imaged: the interaction is aperiodic, and reducing the
    separation into one cell here would silently place the "separated" start at
    the wrong distance.
    """
    G = 4.0 * math.pi / (math.sqrt(3.0) * period_nm)
    Gs = [np.array([G, 0.0]),
          np.array([-0.5 * G, math.sqrt(3.0) / 2.0 * G]),
          np.array([-0.5 * G, -math.sqrt(3.0) / 2.0 * G])]

    g = np.linspace(-period_nm, period_nm, n_grid)
    X, Y = np.meshgrid(g, g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=1)

    moire = amplitude_eV * sum(np.cos(P @ k) for k in Gs)
    stark = sum(np.sin(P @ k) for k in Gs)

    r_e = P[np.argmin(moire - Fz * dipole_nm * stark)]
    r_h = P[np.argmin(moire + Fz * dipole_nm * stark)]
    return tuple(r_e), tuple(r_h)


def rotate_c3(r, k: int = 1):
    """Rotate a position by k * 120 degrees about the origin."""
    a = 2.0 * math.pi * k / 3.0
    c, s = math.cos(a), math.sin(a)
    return (c * r[0] - s * r[1], s * r[0] + c * r[1])


def starting_configurations(cfg: dict, Fz: float) -> dict:
    """The three initial conditions, as (center_e, center_h) pairs."""
    r_e, r_h = landscape_minima(cfg["moire_period_nm"], cfg["moire_amplitude_eV"],
                                cfg["dipole_length_nm"], max(Fz, 1e-6))
    return {
        "bound": (r_h, r_h),
        "separated": (r_e, r_h),
        "separated_c3": (rotate_c3(r_e), rotate_c3(r_h)),
    }


def _interaction(cfg: dict, r_max_nm: float):
    from tmd_pimc.bilayer_keldysh_potential import (
        BilayerKeldyshTablePotential, bilayer_keldysh_value_eV,
    )
    kw = dict(
        separation_nm=cfg["separation_nm"],
        screening_length_layer1_nm=cfg["screening_length_layer1_nm"],
        screening_length_layer2_nm=cfg["screening_length_layer2_nm"],
        kappa_environment=cfg["kappa_environment"],
    )
    r = np.linspace(0.0, r_max_nm, 40001)
    return BilayerKeldyshTablePotential(
        r_nm=r, V_eV=bilayer_keldysh_value_eV(r, **kw), **kw
    )


def count_crossings(trace: np.ndarray, threshold: float, hysteresis: float = 0.5) -> int:
    """Number of times a trace crosses the threshold, with a dead band.

    The dead band suppresses the double-counting that a bare threshold produces
    when a trace hovers near it. A trace must fall below threshold*(1-h) and
    rise above threshold*(1+h) to register successive crossings.

    This is the quantity that distinguishes escape from mixing: one crossing is
    a one-way escape and samples relaxation, many crossings mean the seed is
    visiting both states and samples equilibrium.
    """
    lo = threshold * (1.0 - hysteresis)
    hi = threshold * (1.0 + hysteresis)
    state = None
    n = 0
    for v in trace:
        if v < lo:
            if state == "high":
                n += 1
            state = "low"
        elif v > hi:
            if state == "low":
                n += 1
            state = "high"
    return n


def run_branch(cfg: dict, Fz: float, start, n_steps: int, seeds: Sequence[int],
               threshold: float, burn_in_fraction: float, r_max_nm: float,
               sample_every: int) -> dict:
    from tmd_pimc import (
        CompositePotential, TwoBodyRingPolymerAction,
        TwoBodyPIMCSamplerStagingPeriodicJIT, pair_separations,
    )

    center_e, center_h = start
    zero = CompositePotential(terms=[])
    action = TwoBodyRingPolymerAction(
        mass_e_m0=cfg["mass_e_m0"], mass_h_m0=cfg["mass_h_m0"],
        temperature_K=cfg["temperature_K"], n_beads=int(cfg["n_beads"]),
        potential_e=zero, potential_h=zero,
        potential_interaction=_interaction(cfg, r_max_nm),
    )
    burn_in = int(round(burn_in_fraction * n_steps))

    traces, rho2, acc = [], [], {"local_e": [], "local_h": [], "staging": [], "global": []}
    max_sep = 0.0

    for seed in seeds:
        sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(
            action=action,
            moire_period_nm=cfg["moire_period_nm"],
            moire_amplitude_eV=cfg["moire_amplitude_eV"],
            Fz_eV_per_nm=Fz,
            dipole_length_nm=cfg["dipole_length_nm"],
            local_step_nm=cfg["local_step_nm"],
            global_step_nm=cfg["global_step_nm"],
            interaction_table_r_max_nm=r_max_nm,
            rng_seed=seed,
        )
        out = sampler.run(n_steps=n_steps, burn_in=burn_in,
                          sample_every=sample_every,
                          center_e=center_e, center_h=center_h)
        rho = pair_separations(out["samples_e"], out["samples_h"])
        trace = np.mean(rho ** 2, axis=1)          # per sample, averaged over beads
        traces.append(trace)
        rho2.append(float(trace.mean()))
        max_sep = max(max_sep, float(rho.max()))
        acc["local_e"].append(float(out["acceptance_local_e"]))
        acc["local_h"].append(float(out["acceptance_local_h"]))
        acc["staging"].append(float(out["acceptance_staging"]))
        acc["global"].append(float(out["acceptance_global_joint"]))

    crossings = [count_crossings(t, threshold) for t in traces]
    return {
        "fraction": float(np.mean([v > threshold for v in rho2])),
        "rho2_per_seed": rho2,
        "traces": traces,
        "crossings": crossings,
        "max_separation_nm": max_sep,
        "acceptance": {k: float(np.mean(v)) for k, v in acc.items()},
        "cutoff_ok": max_sep < r_max_nm,
    }


def crossing_field(fields, forward, reverse) -> Optional[float]:
    """Field where the forward and reverse branches meet, by linear interpolation."""
    f = np.asarray(fields, float)
    d = np.asarray(reverse, float) - np.asarray(forward, float)
    o = np.argsort(f)
    f, d = f[o], d[o]
    if np.all(d > 0) or np.all(d < 0):
        return None
    for i in range(len(f) - 1):
        if d[i] * d[i + 1] <= 0:
            if d[i + 1] == d[i]:
                return float(f[i])
            t = -d[i] / (d[i + 1] - d[i])
            return float(f[i] + t * (f[i + 1] - f[i]))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fields", type=float, nargs="+", required=True)
    ap.add_argument("--n-steps", type=int, default=180000)
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_NM2)
    ap.add_argument("--burn-in-fraction", type=float, default=DEFAULT_BURN_IN_FRACTION)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--interaction-r-max-nm", type=float, default=300.0)
    ap.add_argument("--branches", nargs="+",
                    default=["bound", "separated", "separated_c3"])
    ap.add_argument("--out", default=None, help="output prefix (writes .csv and .npz)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    with open(args.config) as fh:
        cfg = json.load(fh)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    print(f"config      : {args.config}")
    print(f"  V0        : {cfg['moire_amplitude_eV'] * 1000:.3f} meV")
    print(f"  d_p       : {cfg['dipole_length_nm']:.5f} nm")
    print(f"  n_steps   : {args.n_steps}   seeds/branch: {len(seeds)}")
    e0, h0 = landscape_minima(cfg["moire_period_nm"], cfg["moire_amplitude_eV"],
                              cfg["dipole_length_nm"], max(args.fields))
    print(f"  minima    : e ({e0[0]:.3f}, {e0[1]:.3f})  h ({h0[0]:.3f}, {h0[1]:.3f})"
          f"   separation {math.dist(e0, h0):.3f} nm\n")

    rows, store = [], {}
    frac = {b: [] for b in args.branches}

    for Fz in args.fields:
        starts = starting_configurations(cfg, Fz)
        print(f"Fz = {Fz:.3f}")
        for b in args.branches:
            r = run_branch(cfg, Fz, starts[b], args.n_steps, seeds, args.threshold,
                           args.burn_in_fraction, args.interaction_r_max_nm,
                           args.sample_every)
            frac[b].append(r["fraction"])
            store[f"{Fz:.4f}_{b}"] = np.array(r["traces"])
            nx = r["crossings"]
            print(f"  {b:14s} frac={r['fraction']:5.2f}  "
                  f"<rho^2>=[{min(r['rho2_per_seed']):7.1f},{max(r['rho2_per_seed']):8.1f}]  "
                  f"crossings/seed: med={int(np.median(nx))} max={max(nx)}  "
                  f"acc L/S/G = {r['acceptance']['local_e'] * 100:.1f}/"
                  f"{r['acceptance']['staging'] * 100:.1f}/"
                  f"{r['acceptance']['global'] * 100:.2f}%")
            if not r["cutoff_ok"]:
                print(f"      [warning] separations reached {r['max_separation_nm']:.1f} nm "
                      f"against a {args.interaction_r_max_nm:.0f} nm table cutoff")
            rows.append(dict(
                Fz_eV_per_nm=Fz, branch=b, n_steps=args.n_steps, n_seeds=len(seeds),
                fraction=r["fraction"],
                rho2_per_seed=";".join(f"{v:.6f}" for v in r["rho2_per_seed"]),
                crossings=";".join(str(v) for v in nx),
                crossings_median=float(np.median(nx)),
                max_separation_nm=r["max_separation_nm"],
                cutoff_ok=int(r["cutoff_ok"]),
                acc_local_e=r["acceptance"]["local_e"],
                acc_staging=r["acceptance"]["staging"],
                acc_global=r["acceptance"]["global"]))
        print()

    print("=" * 72)
    print("HYSTERESIS")
    print("=" * 72)
    print(f"{'Fz':>7}" + "".join(f"{b:>16}" for b in args.branches) + f"{'gap':>8}")
    for i, Fz in enumerate(args.fields):
        gap = (abs(frac[args.branches[0]][i] - frac[args.branches[1]][i])
               if len(args.branches) > 1 else float("nan"))
        print(f"{Fz:7.3f}" + "".join(f"{frac[b][i]:16.2f}" for b in args.branches)
              + f"{gap:8.2f}")

    if "bound" in frac and "separated" in frac:
        xf = crossing_field(args.fields, frac["bound"], frac["separated"])
        gaps = [abs(a - b) for a, b in zip(frac["bound"], frac["separated"])]
        print()
        if max(gaps) < 0.15:
            print("  Branches agree everywhere: the sampler is equilibrated at this "
                  "run length and the fraction is a genuine equilibrium quantity.")
        else:
            print(f"  Branches disagree by up to {max(gaps):.2f}. The equilibrium "
                  f"fraction is BRACKETED between them, not measured. Longer runs "
                  f"or umbrella sampling are needed for a value.")
        if xf is not None:
            print(f"  Branches cross at Fz = {xf:.3f} eV/nm -- the run-length-"
                  f"independent estimate of the equilibrium switching field.")

        # the control that makes irreversibility falsifiable
        low = min(range(len(args.fields)), key=lambda i: args.fields[i])
        if frac["separated"][low] > 0.9:
            print(f"\n  [control FAILED] at the lowest field tested "
                  f"({args.fields[low]:.2f} eV/nm) seeds started separated never "
                  f"returned. Irreversibility is then a property of the move set, "
                  f"not of the physics, and the forward branch cannot be "
                  f"interpreted. Test a lower field before drawing conclusions.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out + ".csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        np.savez_compressed(args.out + "_traces.npz", **store)
        print(f"\nWrote {args.out}.csv and {args.out}_traces.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
