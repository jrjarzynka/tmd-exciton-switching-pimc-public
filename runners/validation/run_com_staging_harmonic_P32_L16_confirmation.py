#!/usr/bin/env python3
"""Development-only harmonic confirmation for the Paper-1 P=32/L=16 schedule."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "numerics"))

from tmd_pimc import HarmonicPotential, PIMCSampler, PIMCSamplerStaging, RingPolymerAction
from tmd_pimc.constants import HBAR2_OVER_2M0, KB_EV_PER_K
from tmd_pimc.sampler import PIMCSamplerStagingJIT

TEMPERATURES = (15.0, 20.0)
P = 32
L = 16
MASS_M0 = 0.5
K_HARM = 0.0002
MODES = (1, 2, 3, 4, 6, 8, 12, 16)
KINDS = ("python_baseline", "python_L16", "jit_L16")
DEFAULT_SEEDS = (916151, 916152)


def iat(x):
    x = np.asarray(x, float); n = len(x); y = x - x.mean()
    nfft = 1 << (2*n - 1).bit_length()
    ac = np.fft.irfft(np.fft.rfft(y, nfft) * np.conj(np.fft.rfft(y, nfft)), nfft)[:n] / np.arange(n, 0, -1)
    if n < 4 or not np.isfinite(ac[0]) or ac[0] <= 0: return 1.0
    rho = ac/ac[0]; total = 0.0; m = 1
    while 2*m < n:
        pair = rho[2*m-1] + rho[2*m]
        if not np.isfinite(pair) or pair <= 0: break
        total += pair; m += 1
    return float(max(1.0, 1.0 + 2.0*total))


def exact_finite_p(T):
    tau = 1.0/(KB_EV_PER_K*T)/P; lam = HBAR2_OVER_2M0/MASS_M0
    omega = 4.0*np.sin(np.pi*np.arange(P)/P)**2
    coeff = omega/(4.0*lam*tau) + tau*K_HARM/2.0
    power = 1.0/(P*coeff)
    obs = {
        "R_g2_nm2": float(power[1:].sum()),
        "bond_squared_sum_nm2": float(P*np.sum(omega*power)),
        "mean_potential_eV": float(0.5*K_HARM*np.sum(power)),
        "spring_proxy_eV": float(np.sum(omega*power)/(4.0*lam*tau*tau)),
    }
    return power, obs


def series(samples, action):
    fourier = np.fft.fft(samples, axis=1)/P
    powers = np.sum(np.abs(fourier)**2, axis=2)
    center = samples.mean(axis=1); rel = samples-center[:, None, :]
    rg2 = np.mean(np.sum(rel*rel, axis=2), axis=1)
    bonds = np.roll(samples, -1, axis=1)-samples
    bond = np.sum(bonds*bonds, axis=(1, 2))
    r2 = np.mean(np.sum(samples*samples, axis=2), axis=1)
    return powers, {
        "R_g2_nm2": rg2,
        "bond_squared_sum_nm2": bond,
        "mean_potential_eV": 0.5*K_HARM*r2,
        "spring_proxy_eV": bond/(4.0*action._lambda_x*P*action._tau*action._tau),
    }


def run_chain(job):
    T, kind, seed, n_steps, burn_in, every = job
    action = RingPolymerAction(MASS_M0, T, P, HarmonicPotential(k_eV_per_nm2=K_HARM))
    local = 0.70*math.sqrt(2.0*(HBAR2_OVER_2M0/MASS_M0)*action._tau)
    common = dict(action=action, local_step_nm=local, global_step_nm=1.0,
                  global_move_probability=0.0, rng_seed=seed)
    def make():
        if kind == "python_baseline": return PIMCSampler(**common)
        cls = PIMCSamplerStaging if kind == "python_L16" else PIMCSamplerStagingJIT
        return cls(**common, staging_segment_lengths=(L,), staging_moves_per_step=1,
                   perform_local_sweep=True)
    sampler = make()
    if kind == "jit_L16":
        sampler.run(n_steps=2, burn_in=0, sample_every=1)
        sampler = make()
    t0 = time.perf_counter(); result = sampler.run(n_steps, burn_in, every); elapsed = time.perf_counter()-t0
    samples = np.asarray(result["samples"]); powers, observables = series(samples, action)
    row = {"T_K": T, "P": P, "L": None if kind == "python_baseline" else L,
           "kind": kind, "seed": seed, "wall_seconds": elapsed,
           "n_retained": len(samples), "local_step_nm": local,
           "acceptance_local": float(result["acceptance_local"]),
           "acceptance_staging": None if kind == "python_baseline" else float(result["acceptance_staging"]),
           "modes": {}, "observables": {}}
    for k in MODES:
        tau = iat(powers[:, k]); row["modes"][str(k)] = {
            "mean": float(powers[:, k].mean()), "IAT": tau,
            "ESS": float(len(samples)/tau), "ESS_per_second": float(len(samples)/tau/elapsed)}
    for name, values in observables.items():
        tau = iat(values); row["observables"][name] = {
            "mean": float(values.mean()), "IAT": tau,
            "ESS": float(len(values)/tau), "ESS_per_second": float(len(values)/tau/elapsed)}
    return row


def aggregate_temperature(T, rows):
    exact_power, exact_obs = exact_finite_p(T); out = {}
    for kind in KINDS:
        chains = [r for r in rows if r["kind"] == kind]
        out[kind] = {"mean_wall_seconds": float(np.mean([r["wall_seconds"] for r in chains])),
                     "mean_acceptance_staging": None if kind == "python_baseline" else float(np.mean([r["acceptance_staging"] for r in chains])),
                     "modes": {}, "observables": {}}
        for k in MODES:
            mean = float(np.mean([r["modes"][str(k)]["mean"] for r in chains])); exact = float(exact_power[k])
            out[kind]["modes"][str(k)] = {"exact": exact, "mean": mean,
                "relative_bias": abs(mean-exact)/exact,
                "mean_IAT": float(np.mean([r["modes"][str(k)]["IAT"] for r in chains])),
                "total_ESS": float(sum(r["modes"][str(k)]["ESS"] for r in chains)),
                "ESS_per_second": float(sum(r["modes"][str(k)]["ESS"] for r in chains)/sum(r["wall_seconds"] for r in chains))}
        for name, exact in exact_obs.items():
            mean = float(np.mean([r["observables"][name]["mean"] for r in chains]))
            out[kind]["observables"][name] = {"exact": exact, "mean": mean,
                "relative_bias": abs(mean-exact)/abs(exact),
                "mean_IAT": float(np.mean([r["observables"][name]["IAT"] for r in chains])),
                "total_ESS": float(sum(r["observables"][name]["ESS"] for r in chains)),
                "ESS_per_second": float(sum(r["observables"][name]["ESS"] for r in chains)/sum(r["wall_seconds"] for r in chains))}
    gains = {f"q{k}": out["python_L16"]["modes"][str(k)]["ESS_per_second"] / out["python_baseline"]["modes"][str(k)]["ESS_per_second"] for k in (1,2,3)}
    gains["R_g2"] = out["python_L16"]["observables"]["R_g2_nm2"]["ESS_per_second"] / out["python_baseline"]["observables"]["R_g2_nm2"]["ESS_per_second"]
    agreement = {f"q{k}": abs(out["python_L16"]["modes"][str(k)]["mean"]-out["jit_L16"]["modes"][str(k)]["mean"])/exact_power[k] for k in MODES}
    agreement.update({name: abs(out["python_L16"]["observables"][name]["mean"]-out["jit_L16"]["observables"][name]["mean"])/abs(exact) for name, exact in exact_obs.items()})
    stage_bias = [out[k]["modes"][str(m)]["relative_bias"] for k in ("python_L16","jit_L16") for m in MODES]
    obs_bias = [out[k]["observables"][n]["relative_bias"] for k in ("python_L16","jit_L16") for n in exact_obs]
    acc = [r["acceptance_staging"] for r in rows if r["kind"] != "python_baseline"]
    low_gains = [gains[x] for x in ("q1","q2","q3","R_g2")]
    checks = {"max_mode_bias_le_5pct": bool(max(stage_bias) <= 0.05),
              "max_observable_bias_le_3pct": bool(max(obs_bias) <= 0.03),
              "python_jit_agreement_le_5pct_exact_scale": bool(max(agreement.values()) <= 0.05),
              "staging_acceptance_positive": bool(min(acc) > 0.0),
              "median_low_mode_ESS_per_second_gain_ge_1p25": bool(float(np.median(low_gains)) >= 1.25),
              "at_least_three_of_four_low_mode_gains_gt_1": bool(sum(x > 1.0 for x in low_gains) >= 3)}
    out["comparison"] = {"ESS_per_second_gains_python_L16_over_baseline": gains,
                         "python_vs_jit_relative_difference": agreement,
                         "max_staging_mode_relative_bias": max(stage_bias),
                         "max_staging_observable_relative_bias": max(obs_bias),
                         "minimum_staging_acceptance": min(acc), "checks": checks,
                         "verdict": "PASS" if all(checks.values()) else "REVIEW"}
    return out, exact_power, exact_obs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-steps", type=int, default=160_000); ap.add_argument("--burn-in", type=int, default=20_000)
    ap.add_argument("--sample-every", type=int, default=20); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS)); args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(T,k,s,args.n_steps,args.burn_in,args.sample_every) for T in TEMPERATURES for k in KINDS for s in args.seeds]
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_chain,j) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            row=future.result(); rows.append(row)
            chain_path=args.out/f"harmonic_T{int(row['T_K']):03d}_{row['kind']}_seed{row['seed']}.json"
            chain_path.write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
            print(f"T={row['T_K']:g} {row['kind']} seed={row['seed']} wall={row['wall_seconds']:.2f}s",flush=True)
    aggregate={}; exact={}
    for T in TEMPERATURES:
        aggregate[str(int(T))], power, obs = aggregate_temperature(T,[r for r in rows if r["T_K"]==T])
        exact[str(int(T))]={"mode_power":[float(x) for x in power],"observables":obs}
    verdict="PASS" if all(aggregate[str(int(T))]["comparison"]["verdict"]=="PASS" for T in TEMPERATURES) else "REVIEW"
    result={"title":"COM_STAGING_P32_L16_HARMONIC_CONFIRMATION","classification":"development_only_not_paper1_evidence",
            "configuration":{"temperatures_K":list(TEMPERATURES),"P":P,"L":L,"mass_m0":MASS_M0,"k_harm_eV_per_nm2":K_HARM,
                             "n_steps":args.n_steps,"burn_in":args.burn_in,"sample_every":args.sample_every,"seeds":args.seeds,"global_moves":False},
            "exact":exact,"chains":sorted(rows,key=lambda r:(r["T_K"],r["kind"],r["seed"])),"aggregate":aggregate,"verdict":verdict}
    path=args.out/"COM_STAGING_P32_L16_HARMONIC_CONFIRMATION.json"; path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"path":str(path),"verdict":verdict,"per_temperature":{k:v["comparison"] for k,v in aggregate.items()}},indent=2))


if __name__ == "__main__": main()
