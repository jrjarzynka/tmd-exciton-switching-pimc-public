"""
Test 7: chi_E(T, Ex) -- rozrzut seed-to-seed centroidu <X_c>, Eq. 9 artykulu.
Definiowane od poczatku dokumentu, nigdy dotad nie policzone.

chi_E = Var_s[ <X_c>_s ]  (wariancja SREDNIEGO polozenia centroidu miedzy
niezaleznymi seedami przy tych samych T, Ex)

Mniejsza siatka Ex niz Test 6 (3 zamiast 5 punktow) -- celem jest sprawdzenie
powtarzalnosci, nie gestej rozdzielczosci pola.
"""
import numpy as np
from tmd_pimc import (RingPolymerAction, DoubleGaussianWellPotential,
                       ExternalFieldPotential, CompositePotential,
                       PIMCSamplerStaging)
from tmd_pimc.observables import centroids

MASS_M0 = 0.5
SEPARATION_NM = 10.0
SIGMA_NM = 3.0
V0_EV = 0.05
Q_EFF = 1.0

T_VALUES = [5.0, 10.0, 15.0, 20.0]
EX_VALUES = [-0.000517, 0.0, 0.000517]
N_SEEDS = 5

def n_beads_for_T(T_K):
    return max(32, int(400 // T_K))

def run(T_K, Ex, seed, n_steps=50000, burn_in=12000, sample_every=20):
    P = n_beads_for_T(T_K)
    dw = DoubleGaussianWellPotential(V0_eV=V0_EV, sigma_nm=SIGMA_NM,
                                      separation_nm=SEPARATION_NM,
                                      asymmetry_eV=0.0)
    field = ExternalFieldPotential(E=(Ex, 0.0), q_eff=Q_EFF)
    potential = CompositePotential(terms=[dw, field])
    action = RingPolymerAction(mass_m0=MASS_M0, temperature_K=T_K,
                                n_beads=P, potential=potential)
    sampler = PIMCSamplerStaging(action=action, rng_seed=seed)
    out = sampler.run(n_steps=n_steps, burn_in=burn_in, sample_every=sample_every)
    return centroids(out["samples"])[:, 0].mean()  # <X_c> for this seed

seed_counter = 0
results = []
for T in T_VALUES:
    for Ex in EX_VALUES:
        mean_x_per_seed = []
        for s in range(N_SEEDS):
            seed_counter += 1
            mx = run(T, Ex, seed=seed_counter)
            mean_x_per_seed.append(mx)
        mean_x_per_seed = np.array(mean_x_per_seed)
        chi_E = float(np.var(mean_x_per_seed, ddof=1))
        results.append((T, Ex, chi_E, mean_x_per_seed.tolist()))
        print(f"T={T:5.1f}K  Ex={Ex:+.6f}  <X_c> per seed={np.round(mean_x_per_seed,3)}  "
              f"chi_E={chi_E:.4f} nm^2")

import csv
with open("test7_chi_E_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["T_K", "Ex_eV_per_nm", "chi_E_nm2"])
    for T, Ex, chi_E, _ in results:
        w.writerow([T, Ex, chi_E])
print("saved test7_chi_E_results.csv")
