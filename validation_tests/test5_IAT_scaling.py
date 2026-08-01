"""
Test 5: skalowanie IAT (integrated autocorrelation time) z P dla samplera COM.
Odtwarza dokladnie ten sam typ testu, ktory juz macie dla interakcji BLK
(z~1.594 lokalny vs z~0.01 staging), ale dla silnika COM -- centralnego
elementu tego artykulu.

WAZNE: sample_every=1 (surowy szereg czasowy) -- dziesiatkowanie zniszczyloby
pomiar autokorelacji.
"""
import numpy as np
from tmd_pimc import RingPolymerAction, HarmonicPotential, PIMCSampler, PIMCSamplerStaging
from tmd_pimc.observables import r2_time_series
from tmd_pimc.statistics import analyze_timeseries

MASS_M0 = 0.5
K_EV_PER_NM2 = 0.010
T_K = 20.0

P_VALUES = [10, 20, 40, 80, 160]
N_STEPS = 20000     # surowy szereg -- sample_every=1
BURN_IN = 2000

def measure_tau(sampler_cls, P, **extra_kwargs):
    potential = HarmonicPotential(k_eV_per_nm2=K_EV_PER_NM2)
    action = RingPolymerAction(mass_m0=MASS_M0, temperature_K=T_K,
                                n_beads=P, potential=potential)
    sampler = sampler_cls(action=action, rng_seed=1, **extra_kwargs)
    out = sampler.run(n_steps=N_STEPS, burn_in=BURN_IN, sample_every=1)
    total, centroid, spread = r2_time_series(out["samples"])
    res = analyze_timeseries(total, max_lag=min(3000, len(total)//4))
    return res["tau_int"]

results_local, results_staging = [], []
for P in P_VALUES:
    tau_local = measure_tau(PIMCSampler, P)
    tau_staging = measure_tau(PIMCSamplerStaging, P)
    results_local.append((P, tau_local))
    results_staging.append((P, tau_staging))
    print(f"P={P:4d}  tau_local={tau_local:10.2f}  tau_staging={tau_staging:8.3f}")

# dopasowanie potegowe: log(tau) = z*log(P) + const
P_arr = np.array([p for p, _ in results_local], dtype=float)
tau_local_arr = np.array([t for _, t in results_local])
tau_staging_arr = np.array([t for _, t in results_staging])

z_local, c_local = np.polyfit(np.log(P_arr), np.log(tau_local_arr), 1)
z_staging, c_staging = np.polyfit(np.log(P_arr), np.log(tau_staging_arr), 1)

print(f"\nz_local   = {z_local:.4f}  (oczekiwane ~1.5-2, jak dla BLK z~1.594)")
print(f"z_staging = {z_staging:.4f}  (oczekiwane ~0, jak dla BLK z~0.01)")

import csv
with open("test5_IAT_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["P", "tau_local", "tau_staging"])
    for (P, tl), (_, ts) in zip(results_local, results_staging):
        w.writerow([P, tl, ts])
    w.writerow([])
    w.writerow(["z_local", z_local])
    w.writerow(["z_staging", z_staging])
