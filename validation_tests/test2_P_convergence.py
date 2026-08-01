"""
Test 2: zbieznosc <R^2> wzgledem liczby paciorkow P, przy ustalonym T=5K
(najostrzejszy przypadek adaptacyjnej reguly P(T), Eq. 8 w artykule).
Pokazuje plateau potwierdzajace ze P=80 (uzyte w produkcji przy 5K) jest
wystarczajace, nie za male.
"""
import numpy as np
from tmd_pimc import RingPolymerAction, HarmonicPotential, PIMCSamplerStaging
from tmd_pimc.analytic import harmonic_r2_analytic, harmonic_r2_primitive_finite_P
from tmd_pimc.observables import r2_mean_pimc

MASS_M0 = 0.5
K_EV_PER_NM2 = 0.010
T_K = 5.0
P_VALUES = [20, 40, 60, 80, 100, 140, 200]   # obejmuje P=80 uzyte w produkcji

results = []
for P in P_VALUES:
    potential = HarmonicPotential(k_eV_per_nm2=K_EV_PER_NM2)
    action = RingPolymerAction(mass_m0=MASS_M0, temperature_K=T_K,
                                n_beads=P, potential=potential)
    sampler = PIMCSamplerStaging(action=action, rng_seed=42)
    out = sampler.run(n_steps=60000, burn_in=15000, sample_every=20)
    r2_pimc = r2_mean_pimc(out["samples"])
    r2_exact_finiteP = harmonic_r2_primitive_finite_P(MASS_M0, K_EV_PER_NM2, T_K, P)
    results.append((P, r2_pimc, r2_exact_finiteP))
    print(f"P={P:4d}  R2_pimc={r2_pimc:.4f}  R2_exact(sameP)={r2_exact_finiteP:.4f}")

r2_infP = harmonic_r2_analytic(MASS_M0, K_EV_PER_NM2, T_K)
print(f"\nGranica P->inf (analityczna): R2 = {r2_infP:.4f}")
print("Plateau powinno zblizac sie do tej wartosci wraz ze wzrostem P.")

import csv
with open("test2_P_convergence_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["P_beads", "R2_pimc_nm2", "R2_exact_finiteP_nm2"])
    w.writerows(results)
