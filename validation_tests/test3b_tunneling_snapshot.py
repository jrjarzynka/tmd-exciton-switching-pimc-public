"""
Migawka tunelowania kwantowego: przy Ex=0 (symetryczna podwojna studnia),
pojedynczy ring polymer (wszystkie P paciorkow jednej probki) czasem
rozciaga sie przez OBIE studnie jednoczesnie -- to jest bezposredni,
wizualny dowod tunelowania, ktorego klasyczne MC nie pokaze.
"""
import numpy as np
from tmd_pimc import (RingPolymerAction, DoubleGaussianWellPotential,
                       PIMCSamplerStaging)

MASS_M0 = 0.5
T_K = 20.0
N_BEADS = 32
SEPARATION_NM = 10.0
SIGMA_NM = 3.0
V0_EV = 0.05

dw = DoubleGaussianWellPotential(V0_eV=V0_EV, sigma_nm=SIGMA_NM,
                                  separation_nm=SEPARATION_NM,
                                  asymmetry_eV=0.0)  # symetryczna, Ex=0
action = RingPolymerAction(mass_m0=MASS_M0, temperature_K=T_K,
                            n_beads=N_BEADS, potential=dw)
sampler = PIMCSamplerStaging(action=action, rng_seed=2024)
out = sampler.run(n_steps=60000, burn_in=15000, sample_every=20)
samples = out["samples"]  # (n_samples, P, 2)

np.save("samples_symmetric_Ex0.npy", samples)

# Metryka "rozciagniecia miedzy studniami" dla kazdej probki:
# ile paciorkow lezy w lewej (x<0) vs prawej (x>0) studni w TEJ SAMEJ probce.
left_frac_per_sample = np.mean(samples[:, :, 0] < 0, axis=1)   # (n_samples,)
straddling = (left_frac_per_sample > 0.05) & (left_frac_per_sample < 0.95)
print(f"Frakcja probek 'rozciagnietych' miedzy studniami (tunelujacych): "
      f"{np.mean(straddling)*100:.1f}%")

# Znajdz kilka najbardziej wyraznych przykladow tunelowania do narysowania
idx_sorted = np.argsort(np.abs(left_frac_per_sample - 0.5))
best_tunneling_examples = idx_sorted[:5]
np.save("tunneling_example_indices.npy", best_tunneling_examples)
for idx in best_tunneling_examples:
    print(f"probka {idx}: lewa_frakcja={left_frac_per_sample[idx]:.2f}")
