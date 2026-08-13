#!/usr/bin/env python3
"""
run_dissociation_time_convergence.py

Does the field-driven dissociation threshold F_z,50 converge in sampling length?

Why this is not a formality
---------------------------
The dissociation fraction is defined as the fraction of independent seeds whose
relative-coordinate spread exceeds a threshold within a finite imaginary-time
Monte Carlo run. That is an escape probability, not a thermodynamic average --
the manuscript says so explicitly -- and escape probabilities depend on how long
one waits. Two scenarios are possible and they have opposite consequences:

  SATURATING   Below some field the bound pair is the global free-energy
               minimum, not merely long-lived. The escape probability then
               approaches a finite limit and F_z,50 converges to a physical
               number.

  DRIFTING     The bound state is metastable at every nonzero field. The escape
               probability then approaches 1 for any field given enough time,
               F_z,50 drifts downward without limit, and the quoted threshold is
               a statement about run length rather than about the material.

Distinguishing them is the point of this runner. A saturating fraction plateaus;
a metastable one grows roughly linearly in log(n_steps), because escape from a
barrier is close to a Poisson process and P(escape) = 1 - exp(-t/tau).
The analysis reports the log-slope alongside the plateau test so the two are not
confused.

If the answer is DRIFTING, the result is not worthless -- it just has to be
quoted differently, as a threshold at a stated observation time, the way a
coercive field or a blocking temperature is quoted with its measurement rate.

Relation to the equilibrium runs
--------------------------------
Distinct from a landscape-resolution or bead-count convergence check. Those probe
a discretisation that has a well-defined limit. This one probes whether the
observable itself has a limit, which is a question about the physics rather than
about the numerics.

Practical notes
---------------
The bilayer Keldysh tail is shallow -- around 0.4 meV/nm at 20 nm separation --
so a pair that separates returns slowly. Runs must therefore be long enough that
the pair samples both outcomes, and the interaction table must extend well beyond
the largest separation reached or the force is clamped and the pair is
artificially unbound. Both are checked and reported per run.

Usage
-----
    python run_dissociation_time_convergence.py \\
        --config configs/two_body/calib_dft_shift0000.json \\
        --fields 8 10 12 14 16 \\
        --n-steps 60000 120000 240000 \\
        --n-seeds 24 --out results/time_convergence.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field as dc_field
from typing import Optional, Sequence

import numpy as np

# Threshold on <rho^2> above which a seed counts as dissociated, in nm^2.
# Matches the manuscript's classification threshold.
DEFAULT_DISSOCIATION_THRESHOLD_NM2 = 30.0

# Burn-in as a fraction of n_steps. Held proportional rather than fixed, so that
# every point in the sweep discards the same fraction of its trajectory. A fixed
# burn-in would make longer runs sample a proportionally later window and would
# confound the very time dependence being measured.
DEFAULT_BURN_IN_FRACTION = 0.25


def moire_minimum_nm(period_nm: float) -> tuple:
    """Cartesian position of a true minimum of V0 * sum_i cos(G_i . r).

    The origin is the global MAXIMUM of this landscape (+3 V0 against -1.5 V0 at
    the minima, a span of 4.5 V0 = 547 meV at the DFT-calibrated depth), so
    starting a run there puts both particles on a hilltop. They then slide apart
    for reasons that have nothing to do with the applied field, and the measured
    threshold is meaningless. MoirePotential's own docstring warns about this.

    The minima of the three-cosine form sit at |r| = period / sqrt(3) along the
    Cartesian axes, where every cosine equals -1/2.
    """
    return (period_nm / math.sqrt(3.0), 0.0)


def landscape_value_at(period_nm: float, amplitude_eV: float, r) -> float:
    """Value of the analytic moire landscape at r, in meV. Used for the start check."""
    G = 4.0 * math.pi / (math.sqrt(3.0) * period_nm)
    x, y = float(r[0]), float(r[1])
    s = (math.cos(G * x)
         + math.cos(-0.5 * G * x + math.sqrt(3.0) / 2.0 * G * y)
         + math.cos(-0.5 * G * x - math.sqrt(3.0) / 2.0 * G * y))
    return s * amplitude_eV * 1000.0


@dataclass(slots=True)
class PointResult:
    n_steps: int
    Fz_eV_per_nm: float
    n_seeds: int
    dissociated_fraction: float
    rho2_per_seed: list
    rho2_median: float
    max_separation_nm: float
    interaction_cutoff_nm: float
    cutoff_ok: bool
    acceptance_global: float
    psi0_sq: Optional[float] = None
    warnings: list = dc_field(default_factory=list)


# =============================================================================
# ANALYSIS -- pure functions, unit-testable without a sampler
# =============================================================================


def interpolate_threshold_field(
    fields: Sequence[float], fractions: Sequence[float], level: float = 0.5
) -> Optional[float]:
    """Field at which the dissociation fraction first crosses `level`.

    Linear interpolation between the bracketing points. Returns None when the
    scan does not bracket the crossing, rather than extrapolating: an
    extrapolated threshold from a scan that never reached 50% is exactly the
    kind of number that looks quotable and is not.
    """
    f = np.asarray(fields, dtype=float)
    p = np.asarray(fractions, dtype=float)
    order = np.argsort(f)
    f, p = f[order], p[order]

    if p.min() > level or p.max() < level:
        return None

    for i in range(len(f) - 1):
        if (p[i] - level) * (p[i + 1] - level) <= 0:
            if p[i + 1] == p[i]:
                return float(f[i])
            t = (level - p[i]) / (p[i + 1] - p[i])
            return float(f[i] + t * (f[i + 1] - f[i]))
    return None


def classify_time_dependence(
    n_steps: Sequence[int], fractions: Sequence[float], stderr: Optional[Sequence[float]] = None
) -> dict:
    """Decide whether a dissociation fraction is saturating or still drifting.

    Fits the fraction against log(n_steps). A Poisson escape process gives
    P = 1 - exp(-t/tau), which over any two-decade window looks close to linear
    in log t with a slope set by tau; a saturated fraction gives a slope
    consistent with zero.

    The returned `verdict` is deliberately three-valued. Reporting 'saturated'
    when the data merely lack the power to see a drift would convert a sampling
    limitation into a physical claim.
    """
    n = np.asarray(n_steps, dtype=float)
    p = np.asarray(fractions, dtype=float)
    if len(n) < 3:
        return {"verdict": "insufficient", "slope_per_decade": float("nan"),
                "slope_stderr": float("nan"), "total_change": float("nan")}

    x = np.log10(n)
    w = None
    if stderr is not None:
        s = np.asarray(stderr, dtype=float)
        if np.all(s > 0):
            w = 1.0 / s

    coeffs, cov = np.polyfit(x, p, 1, w=w, cov=True)
    slope = float(coeffs[0])
    slope_err = float(np.sqrt(cov[0, 0]))
    total = float(p[np.argmax(n)] - p[np.argmin(n)])

    if not math.isfinite(slope_err):
        verdict = "insufficient"
    elif abs(slope) > 2.0 * slope_err and abs(slope) > 0.02:
        verdict = "drifting"
    elif abs(slope) + 2.0 * slope_err < 0.05:
        # Saturation is claimed only when the UPPER bound on the drift is small.
        # Testing the point estimate alone would report saturation whenever the
        # error bars are wide, converting a lack of statistics into a physical
        # statement about the material.
        verdict = "saturated"
    else:
        verdict = "inconclusive"

    return {
        "verdict": verdict,
        "slope_per_decade": slope,
        "slope_stderr": slope_err,
        "total_change": total,
    }


def binomial_stderr(fraction: float, n: int) -> float:
    """Standard error of a fraction estimated from n independent seeds."""
    if n <= 0:
        return float("nan")
    p = min(max(fraction, 0.0), 1.0)
    return math.sqrt(max(p * (1.0 - p), 1.0 / (4.0 * n)) / n)


# =============================================================================
# SAMPLING
# =============================================================================


def _build_interaction(cfg: dict, r_max_nm: float):
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


def run_point(
    cfg: dict,
    Fz: float,
    n_steps: int,
    seeds: Sequence[int],
    shift_nm: float,
    dissociation_threshold: float,
    burn_in_fraction: float,
    interaction_r_max_nm: float,
    sample_every: int,
    with_contact_density: bool,
    start_nm: tuple,
) -> PointResult:
    from tmd_pimc import (
        CompositePotential, TwoBodyRingPolymerAction,
        TwoBodyPIMCSamplerStagingPeriodicJIT, pair_separations,
    )

    zero = CompositePotential(terms=[])
    action = TwoBodyRingPolymerAction(
        mass_e_m0=cfg["mass_e_m0"], mass_h_m0=cfg["mass_h_m0"],
        temperature_K=cfg["temperature_K"], n_beads=int(cfg["n_beads"]),
        potential_e=zero, potential_h=zero,
        potential_interaction=_build_interaction(cfg, interaction_r_max_nm),
    )

    burn_in = int(round(burn_in_fraction * n_steps))
    rho2_seeds, max_sep, acc_global = [], 0.0, []
    all_rho = []

    for seed in seeds:
        sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(
            action=action,
            moire_period_nm=cfg["moire_period_nm"],
            moire_amplitude_eV=cfg["moire_amplitude_eV"],
            origin_h_nm=(shift_nm, 0.0),
            Fz_eV_per_nm=Fz,
            dipole_length_nm=cfg["dipole_length_nm"],
            local_step_nm=cfg["local_step_nm"],
            global_step_nm=cfg["global_step_nm"],
            interaction_table_r_max_nm=interaction_r_max_nm,
            rng_seed=seed,
        )
        out = sampler.run(n_steps=n_steps, burn_in=burn_in,
                          sample_every=sample_every,
                          center_e=start_nm, center_h=start_nm)
        rho = pair_separations(out["samples_e"], out["samples_h"])
        rho2_seeds.append(float(np.mean(rho ** 2)))
        max_sep = max(max_sep, float(rho.max()))
        acc_global.append(float(out["acceptance_global_joint"]))
        if with_contact_density:
            all_rho.append(rho.ravel())

    frac = float(np.mean([r2 > dissociation_threshold for r2 in rho2_seeds]))

    warnings: list[str] = []
    cutoff_ok = max_sep < interaction_r_max_nm
    if not cutoff_ok:
        warnings.append(
            f"separations reached {max_sep:.1f} nm against a table cutoff of "
            f"{interaction_r_max_nm:.1f} nm: the pair left the tabulated region, "
            f"so it is unbound by construction and this point is meaningless."
        )
    elif max_sep > 0.5 * interaction_r_max_nm:
        warnings.append(
            f"largest separation {max_sep:.1f} nm exceeds half the table cutoff; "
            f"widen interaction_table_r_max_nm before trusting the tail."
        )

    psi0 = None
    if with_contact_density and all_rho:
        try:
            from tmd_pimc import contact_density
            psi0 = float(contact_density(np.concatenate(all_rho)).psi0_sq)
        except Exception as exc:                      # noqa: BLE001
            warnings.append(f"contact density unavailable: {exc}")

    return PointResult(
        n_steps=n_steps, Fz_eV_per_nm=Fz, n_seeds=len(seeds),
        dissociated_fraction=frac, rho2_per_seed=rho2_seeds,
        rho2_median=float(np.median(rho2_seeds)),
        max_separation_nm=max_sep, interaction_cutoff_nm=interaction_r_max_nm,
        cutoff_ok=cutoff_ok,
        acceptance_global=float(np.mean(acc_global)),
        psi0_sq=psi0, warnings=warnings,
    )


# =============================================================================
# CLI
# =============================================================================


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fields", type=float, nargs="+", required=True,
                    help="Fz values in eV/nm; must bracket the transition")
    ap.add_argument("--n-steps", type=int, nargs="+",
                    default=[60000, 120000, 240000],
                    help="sampling lengths to compare (default: %(default)s)")
    ap.add_argument("--n-seeds", type=int, default=24)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--shift-nm", type=float, default=0.0,
                    help="registry offset between electron and hole landscapes")
    ap.add_argument("--dissociation-threshold", type=float,
                    default=DEFAULT_DISSOCIATION_THRESHOLD_NM2,
                    help="<rho^2> above which a seed counts as dissociated, nm^2")
    ap.add_argument("--burn-in-fraction", type=float,
                    default=DEFAULT_BURN_IN_FRACTION)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--interaction-r-max-nm", type=float, default=200.0,
                    help="interaction table range; must exceed the largest "
                         "separation reached (default: %(default)s)")
    ap.add_argument("--contact-density", action="store_true",
                    help="also estimate |psi(0)|^2 at each point")
    ap.add_argument("--start-nm", type=float, nargs=2, default=None,
                    metavar=("X", "Y"),
                    help="starting position of both chains, nm. Defaults to a "
                         "true landscape minimum. The ORIGIN IS A MAXIMUM of the "
                         "three-cosine moire potential, so do not pass (0, 0).")
    ap.add_argument("--out", default=None, help="CSV output path")
    args = ap.parse_args(argv)

    # Line-buffer stdout so that `tail -f` on a redirected log shows progress.
    # Without this a multi-hour run looks dead until it finishes.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    with open(args.config) as fh:
        cfg = json.load(fh)

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    start = (tuple(args.start_nm) if args.start_nm is not None
             else moire_minimum_nm(cfg["moire_period_nm"]))
    v_start = landscape_value_at(cfg["moire_period_nm"], cfg["moire_amplitude_eV"], start)
    v_min = -1.5 * cfg["moire_amplitude_eV"] * 1000.0
    above = v_start - v_min

    print(f"config                 : {args.config}")
    print(f"  V0 (amplitude)       : {cfg['moire_amplitude_eV'] * 1000:.3f} meV")
    print(f"  d_p                  : {cfg['dipole_length_nm']:.5f} nm")
    print(f"  period / T / P       : {cfg['moire_period_nm']} nm / "
          f"{cfg['temperature_K']} K / {cfg['n_beads']}")
    print(f"  registry offset      : {args.shift_nm} nm")
    print(f"  dissociation at      : <rho^2> > {args.dissociation_threshold} nm^2")
    print(f"  seeds per point      : {len(seeds)}")
    print(f"  start position       : ({start[0]:.4f}, {start[1]:.4f}) nm, "
          f"{above:+.2f} meV above the landscape minimum")
    if above > 0.25 * 4.5 * cfg["moire_amplitude_eV"] * 1000.0:
        print("  [warning] the pair starts well above the landscape minimum. It "
              "will slide downhill and separate for reasons unrelated to the "
              "applied field, and the threshold will be meaningless.")
    print()

    rows: list[PointResult] = []
    by_n: dict[int, list[PointResult]] = {}

    for n_steps in args.n_steps:
        print(f"n_steps = {n_steps}")
        pts = []
        for Fz in args.fields:
            r = run_point(
                cfg, Fz, n_steps, seeds, args.shift_nm,
                args.dissociation_threshold, args.burn_in_fraction,
                args.interaction_r_max_nm, args.sample_every,
                args.contact_density, start,
            )
            pts.append(r)
            rows.append(r)
            se = binomial_stderr(r.dissociated_fraction, r.n_seeds)
            extra = f"  |psi(0)|^2={r.psi0_sq:.5f}" if r.psi0_sq else ""
            lo = min(r.rho2_per_seed); hi = max(r.rho2_per_seed)
            print(f"  Fz={Fz:7.3f}  frac={r.dissociated_fraction:5.2f} +/-{se:4.2f}"
                  f"  <rho^2> med={r.rho2_median:8.2f} "
                  f"[{lo:7.2f}, {hi:8.2f}]"
                  f"  max rho={r.max_separation_nm:6.2f}{extra}")
            for w in r.warnings:
                print(f"      [warning] {w}")
        by_n[n_steps] = pts
        f50 = interpolate_threshold_field([p.Fz_eV_per_nm for p in pts],
                                          [p.dissociated_fraction for p in pts])
        print(f"  -> F_z,50 = {'not bracketed' if f50 is None else f'{f50:.4f} eV/nm'}\n")

    print("=" * 72)
    print("TIME DEPENDENCE AT FIXED FIELD")
    print("=" * 72)
    for Fz in args.fields:
        ns = sorted(by_n)
        fr = [next(p.dissociated_fraction for p in by_n[n] if p.Fz_eV_per_nm == Fz)
              for n in ns]
        se = [binomial_stderr(f, args.n_seeds) for f in fr]
        c = classify_time_dependence(ns, fr, se)
        trail = "  ".join(f"{n // 1000}k:{f:.2f}" for n, f in zip(ns, fr))
        print(f"  Fz={Fz:7.3f}  {trail}   slope/decade={c['slope_per_decade']:+.3f}"
              f" +/-{c['slope_stderr']:.3f}   {c['verdict'].upper()}")

    print()
    thresholds = {n: interpolate_threshold_field(
        [p.Fz_eV_per_nm for p in by_n[n]],
        [p.dissociated_fraction for p in by_n[n]]) for n in sorted(by_n)}
    print("F_z,50 versus sampling length:")
    for n, t in thresholds.items():
        print(f"  {n:>8} steps : {'not bracketed' if t is None else f'{t:.4f} eV/nm'}")

    vals = [t for t in thresholds.values() if t is not None]
    if len(vals) >= 2:
        drift = (vals[-1] - vals[0]) / vals[0] * 100.0
        print(f"\n  drift across the sweep: {drift:+.1f}%")
        if abs(drift) > 5.0:
            print("  F_z,50 is NOT converged in sampling length. Quote it at a "
                  "stated observation time, or extrapolate, rather than as a "
                  "material constant.")
        else:
            print("  F_z,50 is stable across the sweep.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            # rho2_per_seed is recorded because the aggregate statistics
            # cannot answer the question the scan is for. A sharp 0 -> 1
            # transition has two readings -- every seed switching at the same
            # field, or seeds locked in whichever state they started in -- and
            # only the per-seed distribution separates them. Reconstructing it
            # later is impossible: the runner keeps no raw samples.
            w.writerow(["n_steps", "Fz_eV_per_nm", "n_seeds",
                        "dissociated_fraction", "stderr", "rho2_median",
                        "rho2_per_seed", "rho2_seed_spread",
                        "max_separation_nm", "interaction_cutoff_nm",
                        "cutoff_ok", "acceptance_global", "psi0_sq",
                        "moire_amplitude_eV", "dipole_length_nm",
                        "shift_nm", "dissociation_threshold_nm2"])
            for r in rows:
                seed_vals = ";".join(f"{v:.6f}" for v in r.rho2_per_seed)
                spread = (float(np.std(r.rho2_per_seed, ddof=1))
                          if len(r.rho2_per_seed) > 1 else float("nan"))
                w.writerow([r.n_steps, r.Fz_eV_per_nm, r.n_seeds,
                            f"{r.dissociated_fraction:.6f}",
                            f"{binomial_stderr(r.dissociated_fraction, r.n_seeds):.6f}",
                            f"{r.rho2_median:.6f}", seed_vals, f"{spread:.6f}",
                            f"{r.max_separation_nm:.6f}",
                            r.interaction_cutoff_nm, int(r.cutoff_ok),
                            f"{r.acceptance_global:.6f}",
                            "" if r.psi0_sq is None else f"{r.psi0_sq:.8f}",
                            cfg["moire_amplitude_eV"], cfg["dipole_length_nm"],
                            args.shift_nm, args.dissociation_threshold])
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
