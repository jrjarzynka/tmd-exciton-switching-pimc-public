"""
Test 5 v2: jak v1, ale (a) N_STEPS rosnie z P (stala docelowa liczba
niezaleznych probek), (b) usredniamy po kilku seedach na punkt, zeby miec
error bary i stabilne dopasowanie potegowe.
"""
import numpy as np
from tmd_pimc import RingPolymerAction, HarmonicPotential, PIMCSampler, PIMCSamplerStaging
from tmd_pimc.observables import r2_time_series
from tmd_pimc.statistics import analyze_timeseries

MASS_M0 = 0.5
K_EV_PER_NM2 = 0.010
T_K = 20.0

P_VALUES = [10, 20, 40, 80, 160]
N_SEEDS = 4
TARGET_ESS = 150   # docelowa liczba niezaleznych probek -- N_STEPS dobierany tak,
                    # zeby N_STEPS/(2*tau_szacowane) ~ TARGET_ESS
TAU_GUESS_LOCAL = {10: 35, 20: 80, 40: 170, 80: 350, 160: 700}  # zgrubne, do
                    # doboru dlugosci runu; nadpisywane realnym pomiarem

def n_steps_for(P, sampler_kind):
    tau_guess = TAU_GUESS_LOCAL[P] if sampler_kind == "local" else 25
    n = int(2 * TARGET_ESS * tau_guess) + 3000  # + zapas na burn-in
    return max(n, 8000)

def measure_tau(sampler_cls, P, seed, n_steps, burn_in):
    potential = HarmonicPotential(k_eV_per_nm2=K_EV_PER_NM2)
    action = RingPolymerAction(mass_m0=MASS_M0, temperature_K=T_K,
                                n_beads=P, potential=potential)
    sampler = sampler_cls(action=action, rng_seed=seed)
    out = sampler.run(n_steps=n_steps, burn_in=burn_in, sample_every=1)
    total, centroid, spread = r2_time_series(out["samples"])
    res = analyze_timeseries(total, max_lag=min(4000, len(total)//4))
    return res["tau_int"], res["ESS"]

rows = []
for P in P_VALUES:
    for kind, cls in [("local", PIMCSampler), ("staging", PIMCSamplerStaging)]:
        n_steps = n_steps_for(P, kind)
        burn_in = max(1000, n_steps // 10)
        taus = []
        for seed in range(N_SEEDS):
            tau, ess = measure_tau(cls, P, seed=seed, n_steps=n_steps, burn_in=burn_in)
            taus.append(tau)
            print(f"P={P:4d} {kind:8s} seed={seed} n_steps={n_steps:7d} "
                  f"tau={tau:8.2f} ESS={ess:7.1f}")
        taus = np.array(taus)
        rows.append((P, kind, taus.mean(), taus.std(ddof=1)/np.sqrt(N_SEEDS)))
        print(f"  -> P={P} {kind}: tau_mean={taus.mean():.2f} +/- {taus.std(ddof=1)/np.sqrt(N_SEEDS):.2f}")

import csv
with open("test5_IAT_results_v2.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["P", "sampler", "tau_mean", "tau_sem"])
    w.writerows(rows)

# dopasowanie potegowe na srednich
import itertools
for kind in ["local", "staging"]:
    Ps = np.array([r[0] for r in rows if r[1] == kind], dtype=float)
    taus = np.array([r[2] for r in rows if r[1] == kind])
    z, c = np.polyfit(np.log(Ps), np.log(taus), 1)
    print(f"z_{kind} = {z:.4f}")
