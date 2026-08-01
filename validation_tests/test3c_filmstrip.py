"""
Test 3c: "filmstrip" -- centroidy zapisane dla WSZYSTKICH 9 punktow skanu
pola (te same wartosci Ex co w occ_scan_v3.csv), zeby zrobic plynny ciag
map gestosci pokazujacych przejscie z lewej do prawej studni.

Lzejsze niz pelne sciezki (samples_v3_*.npy) -- zapisujemy tylko centroidy,
wystarczajace do map gestosci COM.
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

KB = 8.617333262e-5
kT = KB * T_K
Ex_char = kT / (Q_EFF * SEPARATION_NM)

def run(Ex, seed, n_steps=60000, burn_in=15000, sample_every=20):
    dw = DoubleGaussianWellPotential(V0_eV=V0_EV, sigma_nm=SIGMA_NM,
                                      separation_nm=SEPARATION_NM,
                                      asymmetry_eV=0.0)
    field = ExternalFieldPotential(E=(Ex, 0.0), q_eff=Q_EFF)
    potential = CompositePotential(terms=[dw, field])
    action = RingPolymerAction(mass_m0=MASS_M0, temperature_K=T_K,
                                n_beads=N_BEADS, potential=potential)
    sampler = PIMCSamplerStaging(action=action, rng_seed=seed)
    out = sampler.run(n_steps=n_steps, burn_in=burn_in, sample_every=sample_every)
    return centroids(out["samples"])

Ex_scan = np.linspace(-3*Ex_char, 3*Ex_char, 9)  # identyczne jak w v3

all_centroids = {}
for i, Ex in enumerate(Ex_scan):
    cents = run(Ex, seed=300+i)
    all_centroids[f"Ex_{i}"] = cents
    print(f"[{i}] Ex={Ex:+.6f}  mean_x={cents[:,0].mean():.3f}  frac_right={np.mean(cents[:,0]>0):.4f}")

np.savez("filmstrip_centroids.npz", Ex_values=Ex_scan, **all_centroids)
print("saved filmstrip_centroids.npz")
