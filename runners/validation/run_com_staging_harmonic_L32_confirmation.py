#!/usr/bin/env python3
"""Development-only confirmation of the current L=32 staging implementation.

Purpose
-------
Confirm that the *current repo code*, after the staging refactor, still:
1. samples the exact finite-P 2D harmonic distribution,
2. gives consistent Python/JIT L=32 results,
3. substantially reduces low-mode autocorrelation relative to local-only Python.

This is not Paper-1 production evidence.  It is a short confirmation run before
the 5 K / P=80 moire development pilot.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "numerics"))

from tmd_pimc import (  # noqa: E402
    HarmonicPotential,
    PIMCSampler,
    PIMCSamplerStaging,
    RingPolymerAction,
)
from tmd_pimc.sampler import PIMCSamplerStagingJIT  # noqa: E402
from tmd_pimc.constants import HBAR2_OVER_2M0, KB_EV_PER_K  # noqa: E402


T_K = 5.0
P = 80
MASS_M0 = 0.5
K_HARM_EV_PER_NM2 = 0.0002
L = 32

DEFAULT_SEEDS = (910301, 910302)

# Declared development-confirmation criteria.  These are practical effect-size
# and sampling requirements, not Paper-1 production gates.
MAX_L32_MODE_RELATIVE_BIAS = 0.02
MAX_L32_OBSERVABLE_RELATIVE_BIAS = 0.02
MAX_PYTHON_JIT_MODE_RELATIVE_DIFFERENCE = 0.02
MIN_L32_LOW_MODE_ESS = 200.0
MIN_PYTHON_L32_ESS_PER_SECOND_GAIN = 1.25


def iat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return 1.0
    y = x - x.mean()
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(y, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[:n] / np.arange(n, 0, -1)
    if not np.isfinite(ac[0]) or ac[0] <= 0:
        return 1.0
    rho = ac / ac[0]
    total = 0.0
    m = 1
    while 2 * m < n:
        pair = rho[2 * m - 1] + rho[2 * m]
        if not np.isfinite(pair) or pair <= 0:
            break
        total += pair
        m += 1
    return float(max(1.0, 1.0 + 2.0 * total))


def exact_finite_p():
    beta = 1.0 / (KB_EV_PER_K * T_K)
    tau = beta / P
    lam = HBAR2_OVER_2M0 / MASS_M0
    kpf = 1.0 / (4.0 * lam * tau)

    k = np.arange(P, dtype=float)
    omega = 4.0 * np.sin(np.pi * k / P) ** 2
    C = kpf * omega + tau * K_HARM_EV_PER_NM2 / 2.0

    # q_k = (1/P) sum_j r_j exp(-2 pi i j k/P)
    # total 2D power <|q_k,x|^2 + |q_k,y|^2>
    power = 1.0 / (P * C)

    obs = {
        "R_g2_nm2": float(power[1:].sum()),
        "bond_squared_sum_nm2": float(P * np.sum(omega * power)),
        "mean_potential_eV": float(0.5 * K_HARM_EV_PER_NM2 * np.sum(power)),
        "spring_proxy_eV": float(np.sum(omega * power) / (4.0 * lam * tau * tau)),
    }
    return power, obs


def sample_series(samples: np.ndarray, action):
    n = len(samples)
    F = np.fft.fft(samples, axis=1) / P
    mode_power = np.sum(np.abs(F[:, :41, :]) ** 2, axis=2)

    center = samples.mean(axis=1)
    relative = samples - center[:, None, :]
    rg2 = np.mean(np.sum(relative * relative, axis=2), axis=1)

    bonds = np.roll(samples, -1, axis=1) - samples
    bond_sum = np.sum(bonds * bonds, axis=(1, 2))

    r2 = np.mean(np.sum(samples * samples, axis=2), axis=1)
    potential = 0.5 * K_HARM_EV_PER_NM2 * r2
    spring = bond_sum / (4.0 * action._lambda_x * P * action._tau * action._tau)

    return mode_power, {
        "R_g2_nm2": rg2,
        "bond_squared_sum_nm2": bond_sum,
        "mean_potential_eV": potential,
        "spring_proxy_eV": spring,
    }


def make_action():
    potential = HarmonicPotential(k_eV_per_nm2=K_HARM_EV_PER_NM2)
    return RingPolymerAction(
        mass_m0=MASS_M0,
        temperature_K=T_K,
        n_beads=P,
        potential=potential,
    )


def run_chain(kind: str, seed: int, n_steps: int, burn_in: int, sample_every: int):
    action = make_action()
    tau = 1.0 / (KB_EV_PER_K * T_K) / P
    lam = HBAR2_OVER_2M0 / MASS_M0
    local_step = 1.471881828  # same 5 K / P=80 development value used previously

    common = dict(
        action=action,
        local_step_nm=local_step,
        global_step_nm=1.0,
        global_move_probability=0.0,  # isolate internal-mode mixing
        rng_seed=seed,
    )

    if kind == "python_baseline":
        sampler = PIMCSampler(**common)
    elif kind == "python_L32":
        sampler = PIMCSamplerStaging(
            **common,
            staging_segment_lengths=(L,),
            staging_moves_per_step=1,
            perform_local_sweep=True,
        )
    elif kind == "jit_L32":
        sampler = PIMCSamplerStagingJIT(
            **common,
            staging_segment_lengths=(L,),
            staging_moves_per_step=1,
            perform_local_sweep=True,
        )
        # compile outside timed region, then recreate to restore fresh RNG state
        sampler.run(n_steps=2, burn_in=0, sample_every=1)
        sampler = PIMCSamplerStagingJIT(
            **common,
            staging_segment_lengths=(L,),
            staging_moves_per_step=1,
            perform_local_sweep=True,
        )
    else:
        raise ValueError(kind)

    t0 = time.perf_counter()
    result = sampler.run(
        n_steps=n_steps,
        burn_in=burn_in,
        sample_every=sample_every,
    )
    seconds = time.perf_counter() - t0

    samples = np.asarray(result["samples"], dtype=float)
    powers, obs = sample_series(samples, action)

    out = {
        "kind": kind,
        "seed": seed,
        "wall_seconds": seconds,
        "n_retained": int(len(samples)),
        "acceptance_local": float(result["acceptance_local"]),
        "acceptance_staging": (
            None if "acceptance_staging" not in result
            else float(result["acceptance_staging"])
        ),
        "modes": {},
        "observables": {},
    }

    for k in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40):
        t = iat(powers[:, k])
        ess = len(samples) / t
        out["modes"][str(k)] = {
            "mean": float(powers[:, k].mean()),
            "IAT": t,
            "ESS": ess,
            "ESS_per_second": ess / seconds,
        }

    for name, series in obs.items():
        t = iat(series)
        ess = len(series) / t
        out["observables"][name] = {
            "mean": float(series.mean()),
            "IAT": t,
            "ESS": ess,
            "ESS_per_second": ess / seconds,
        }

    return out


def aggregate(chains, exact_power, exact_obs):
    result = {}
    for kind in ("python_baseline", "python_L32", "jit_L32"):
        js = [x for x in chains if x["kind"] == kind]
        modes = {}
        for k in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40):
            vals = np.array([j["modes"][str(k)]["mean"] for j in js])
            mean = float(vals.mean())
            ex = float(exact_power[k])
            modes[str(k)] = {
                "exact": ex,
                "mean": mean,
                "relative_bias": float(abs(mean - ex) / ex),
                "mean_IAT": float(np.mean([j["modes"][str(k)]["IAT"] for j in js])),
                "ESS_per_second": float(
                    sum(j["modes"][str(k)]["ESS"] for j in js)
                    / sum(j["wall_seconds"] for j in js)
                ),
            }

        obs = {}
        for name, ex in exact_obs.items():
            vals = np.array([j["observables"][name]["mean"] for j in js])
            mean = float(vals.mean())
            obs[name] = {
                "exact": float(ex),
                "mean": mean,
                "relative_bias": float(abs(mean - ex) / abs(ex)),
                "mean_IAT": float(np.mean([j["observables"][name]["IAT"] for j in js])),
                "ESS_per_second": float(
                    sum(j["observables"][name]["ESS"] for j in js)
                    / sum(j["wall_seconds"] for j in js)
                ),
            }

        result[kind] = {
            "mean_wall_seconds": float(np.mean([j["wall_seconds"] for j in js])),
            "mean_acceptance_staging": (
                None if kind == "python_baseline"
                else float(np.mean([j["acceptance_staging"] for j in js]))
            ),
            "modes": modes,
            "observables": obs,
        }

    # Development effect-size checks, deliberately not chain-SEM gates.
    result["python_L32_over_baseline_ESS_per_second"] = {
        "q1": result["python_L32"]["modes"]["1"]["ESS_per_second"]
              / result["python_baseline"]["modes"]["1"]["ESS_per_second"],
        "q2": result["python_L32"]["modes"]["2"]["ESS_per_second"]
              / result["python_baseline"]["modes"]["2"]["ESS_per_second"],
        "q3": result["python_L32"]["modes"]["3"]["ESS_per_second"]
              / result["python_baseline"]["modes"]["3"]["ESS_per_second"],
        "R_g2": result["python_L32"]["observables"]["R_g2_nm2"]["ESS_per_second"]
                / result["python_baseline"]["observables"]["R_g2_nm2"]["ESS_per_second"],
    }

    result["python_vs_jit_L32"] = {}
    for k in (0, 1, 2, 3, 4, 5, 10, 20, 30, 40):
        a = result["python_L32"]["modes"][str(k)]["mean"]
        b = result["jit_L32"]["modes"][str(k)]["mean"]
        ex = exact_power[k]
        result["python_vs_jit_L32"][f"q{k}"] = {
            "absolute_difference": float(a - b),
            "relative_difference_vs_exact": float(abs(a - b) / ex),
        }

    max_bias = max(
        x["relative_bias"]
        for kind in ("python_L32", "jit_L32")
        for x in result[kind]["modes"].values()
    )
    max_obs_bias = max(
        x["relative_bias"]
        for kind in ("python_L32", "jit_L32")
        for x in result[kind]["observables"].values()
    )
    max_python_jit_difference = max(
        x["relative_difference_vs_exact"]
        for x in result["python_vs_jit_L32"].values()
    )
    min_low_mode_ess = min(
        j["modes"][str(k)]["ESS"]
        for j in chains
        if j["kind"] in ("python_L32", "jit_L32")
        for k in (1, 2, 3)
    )
    min_low_mode_ess = min(
        min_low_mode_ess,
        min(
            j["observables"]["R_g2_nm2"]["ESS"]
            for j in chains
            if j["kind"] in ("python_L32", "jit_L32")
        ),
    )
    staging_acceptances = [
        j["acceptance_staging"]
        for j in chains
        if j["kind"] in ("python_L32", "jit_L32")
    ]
    min_staging_acceptance = float(np.min(staging_acceptances))
    staging_acceptance_positive = bool(
        np.all(np.isfinite(staging_acceptances)) and min_staging_acceptance > 0.0
    )
    min_gain = min(result["python_L32_over_baseline_ESS_per_second"].values())

    checks = {
        "mode_relative_bias": bool(max_bias <= MAX_L32_MODE_RELATIVE_BIAS),
        "observable_relative_bias": bool(
            max_obs_bias <= MAX_L32_OBSERVABLE_RELATIVE_BIAS
        ),
        "python_jit_mode_agreement": bool(
            max_python_jit_difference
            <= MAX_PYTHON_JIT_MODE_RELATIVE_DIFFERENCE
        ),
        "low_mode_ESS": bool(min_low_mode_ess >= MIN_L32_LOW_MODE_ESS),
        "ESS_per_second_gain": bool(
            min_gain >= MIN_PYTHON_L32_ESS_PER_SECOND_GAIN
        ),
        "staging_acceptance_positive": staging_acceptance_positive,
    }

    result["summary"] = {
        "max_L32_mode_relative_bias": float(max_bias),
        "max_L32_observable_relative_bias": float(max_obs_bias),
        "max_python_jit_mode_relative_difference_vs_exact": float(
            max_python_jit_difference
        ),
        "minimum_L32_low_mode_or_Rg2_ESS": float(min_low_mode_ess),
        "minimum_L32_staging_acceptance": min_staging_acceptance,
        "minimum_python_L32_ESS_per_second_gain": float(min_gain),
        "criteria": {
            "max_L32_mode_relative_bias": MAX_L32_MODE_RELATIVE_BIAS,
            "max_L32_observable_relative_bias": (
                MAX_L32_OBSERVABLE_RELATIVE_BIAS
            ),
            "max_python_jit_mode_relative_difference_vs_exact": (
                MAX_PYTHON_JIT_MODE_RELATIVE_DIFFERENCE
            ),
            "minimum_L32_low_mode_or_Rg2_ESS": MIN_L32_LOW_MODE_ESS,
            "minimum_python_L32_ESS_per_second_gain": (
                MIN_PYTHON_L32_ESS_PER_SECOND_GAIN
            ),
            "staging_acceptance_must_be_finite_and_positive": True,
        },
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "REVIEW",
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-steps", type=int, default=160_000)
    ap.add_argument("--burn-in", type=int, default=20_000)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    exact_power, exact_obs = exact_finite_p()

    chains = []
    for kind in ("python_baseline", "python_L32", "jit_L32"):
        for seed in args.seeds:
            print(f"Running {kind}, seed={seed} ...", flush=True)
            row = run_chain(
                kind, seed, args.n_steps, args.burn_in, args.sample_every
            )
            chains.append(row)
            print(
                f"  wall={row['wall_seconds']:.2f}s "
                f"acc_stage={row['acceptance_staging']}",
                flush=True,
            )

    result = {
        "title": "COM_STAGING_L32_HARMONIC_CONFIRMATION",
        "classification": "development_only_not_paper1_evidence",
        "configuration": {
            "T_K": T_K,
            "P": P,
            "mass_m0": MASS_M0,
            "k_harm_eV_per_nm2": K_HARM_EV_PER_NM2,
            "L": L,
            "n_steps": args.n_steps,
            "burn_in": args.burn_in,
            "sample_every": args.sample_every,
            "seeds": args.seeds,
            "global_moves": False,
        },
        "exact": {
            "mode_power": [float(x) for x in exact_power],
            "observables": exact_obs,
        },
        "chains": chains,
        "aggregate": aggregate(chains, exact_power, exact_obs),
    }

    out = args.out / "COM_STAGING_L32_HARMONIC_CONFIRMATION.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("\nSaved:", out)
    print(json.dumps(result["aggregate"]["summary"], indent=2))


if __name__ == "__main__":
    main()
