"""
Test 6: operational stability map, A_eff^S(T, Ex) na siatce podwojnej
studni -- domyka koncepcje juz zdefiniowana w Eq. 13 artykulu (A_eff^IPR,
A_eff^S), nigdy wczesniej nie policzona jako rysunek.

A_eff^S = exp[-integral P(R) ln P(R) d^2R]  (efektywna powierzchnia entropijna)

Uzywa tego samego, juz zwalidowanego silnika co Test 3 (symetryczna
podwojna studnia + pole w plaszczyznie), teraz przeskanowanego rowniez po T.
Liczba paciorkow P(T) z Eq. 8 artykulu (adaptive_beads), zgodnie z reszta
pracy: P = max(32, floor(400/T)).
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
# te same wartosci pola co w occ_scan_v3.csv (podzbior 5 z 9)
EX_VALUES = [-0.000517, -0.000259, 0.000000, 0.000259, 0.000517]

def n_beads_for_T(T_K):
    return max(32, int(400 // T_K))

def A_eff_S(cents, bins=40, x_range=(-12, 12), y_range=(-9, 9)):
    H, xedges, yedges = np.histogram2d(cents[:,0], cents[:,1], bins=bins,
                                        range=[x_range, y_range])
    N = H.sum()
    if N == 0:
        return np.nan
    dx = xedges[1]-xedges[0]
    dy = yedges[1]-yedges[0]
    bin_area = dx*dy
    p_density = H / (N * bin_area)   # probability density per nm^2
    mask = p_density > 0
    entropy = -np.sum(p_density[mask] * np.log(p_density[mask])) * bin_area
    return float(np.exp(entropy))

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
    return centroids(out["samples"]), P

seed_counter = 0
results = []
for T in T_VALUES:
    for Ex in EX_VALUES:
        seed_counter += 1
        cents, P = run(T, Ex, seed=seed_counter)
        area = A_eff_S(cents)
        results.append((T, Ex, P, area))
        print(f"T={T:5.1f}K  Ex={Ex:+.6f}  P={P:3d}  A_eff_S={area:8.2f} nm^2")

import csv
with open("test6_stability_map_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["T_K", "Ex_eV_per_nm", "P_beads", "A_eff_S_nm2"])
    w.writerows(results)
print("saved test6_stability_map_results.csv")
