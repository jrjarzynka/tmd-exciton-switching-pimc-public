"""
Test 3 v3: podwojna studnia SYMETRYCZNA (asymmetry_eV=0), pole Ex skanowane
PRZEZ ZERO (ujemne i dodatnie). Przejscie obsadzenia przez 50% musi wypasc
dokladnie przy Ex=0 -- to wynika z czystej symetrii lewo-prawo potencjalu,
bez zadnych dopasowywanych parametrow. Prostszy i solidniejszy test niz v2.
"""
import numpy as np
from tmd_pimc import (RingPolymerAction, DoubleGaussianWellPotential,
                       ExternalFieldPotential, CompositePotential,
                       PIMCSamplerStaging)
from tmd_pimc.observables import centroids

MASS_M0 = 0.5
T_K = 20.0
N_BEADS = 32
SEPARATION_NM = 10.0
SIGMA_NM = 3.0
V0_EV = 0.05
Q_EFF = 1.0

def run(Ex, seed, n_steps=60000, burn_in=15000, sample_every=20):
    dw = DoubleGaussianWellPotential(V0_eV=V0_EV, sigma_nm=SIGMA_NM,
                                      separation_nm=SEPARATION_NM,
                                      asymmetry_eV=0.0)   # <-- symetryczna
    field = ExternalFieldPotential(E=(Ex, 0.0), q_eff=Q_EFF)
    potential = CompositePotential(terms=[dw, field])
    action = RingPolymerAction(mass_m0=MASS_M0, temperature_K=T_K,
                                n_beads=N_BEADS, potential=potential)
    sampler = PIMCSamplerStaging(action=action, rng_seed=seed)
    out = sampler.run(n_steps=n_steps, burn_in=burn_in, sample_every=sample_every)
    return out["samples"]

# Skala pola: dobierz tak, zeby kT ~ porownywalne z modulacja energii na
# skalu separation_nm, np. Ex_char = kT/(q*separation) daje rozsadny zakres.
KB = 8.617333262e-5
kT = KB*T_K
Ex_char = kT / (Q_EFF * SEPARATION_NM)
print(f"kT={kT*1000:.3f} meV, Ex_char={Ex_char:.6f} eV/nm (skala charakterystyczna)")

Ex_scan = np.linspace(-3*Ex_char, 3*Ex_char, 9)
occ_scan = []
for i, Ex in enumerate(Ex_scan):
    samples = run(Ex, seed=200+i)
    cents = centroids(samples)
    frac_right = float(np.mean(cents[:, 0] > 0))
    occ_scan.append((Ex, frac_right))
    print(f"Ex={Ex:+.6f}  frac_right={frac_right:.4f}")

import csv
with open("occ_scan_v3.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Ex_eV_per_nm", "fraction_right"]); w.writerows(occ_scan)

# Trzy migawki: silnie w lewo, symetria (Ex=0), silnie w prawo
for label, Ex in {"left": -2.5*Ex_char, "symmetric": 0.0, "right": 2.5*Ex_char}.items():
    samples = run(Ex, seed=888)
    np.save(f"samples_v3_{label}.npy", samples)
    cents = centroids(samples)
    np.save(f"centroids_v3_{label}.npy", cents)
    print(f"{label}: Ex={Ex:+.6f} mean_x={cents[:,0].mean():.3f} frac_right={np.mean(cents[:,0]>0):.3f}")
