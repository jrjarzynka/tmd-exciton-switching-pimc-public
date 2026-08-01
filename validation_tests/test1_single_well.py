"""
Test 1: pojedyncza studnia harmoniczna -- PIMC vs dokladne analityczne <R^2>.
Rozdziela blad Monte Carlo (porownanie z harmonic_r2_analytic, ktore jest
dokladnym wynikiem w granicy P->inf) od bledu dyskretyzacji Trottera
(porownanie z harmonic_r2_primitive_finite_P, ktore jest dokladnym wynikiem
DLA TEGO SAMEGO skonczonego P co uzywa sampler).
"""
import numpy as np
from tmd_pimc import RingPolymerAction, HarmonicPotential, PIMCSamplerStaging
from tmd_pimc.analytic import harmonic_r2_analytic, harmonic_r2_primitive_finite_P
from tmd_pimc.observables import r2_mean_pimc

MASS_M0 = 0.5          # ta sama masa COM co w produkcji (M_X)
K_EV_PER_NM2 = 0.010    # dobierz tak, zeby dac sensowna skale dlugosci (~kilka nm)
TEMPERATURES_K = [5, 10, 15, 20, 50, 100]
N_BEADS = 80            # najostrzejszy przypadek z Twojej tabeli kriogenicznej

results = []
for T in TEMPERATURES_K:
    potential = HarmonicPotential(k_eV_per_nm2=K_EV_PER_NM2)  # SPRAWDZ nazwe argumentu w potentials.py
    action = RingPolymerAction(mass_m0=MASS_M0, temperature_K=T,
                                n_beads=N_BEADS, potential=potential)
    sampler = PIMCSamplerStaging(action=action, rng_seed=42)  # SPRAWDZ pozostale domyslne parametry

    out = sampler.run(n_steps=60000, burn_in=15000, sample_every=20)
    r2_pimc = r2_mean_pimc(out["samples"])  # SPRAWDZ klucz zwracany przez run() -- moze byc "samples_e" itp.

    r2_exact_infP = harmonic_r2_analytic(MASS_M0, K_EV_PER_NM2, T)
    r2_exact_finiteP = harmonic_r2_primitive_finite_P(MASS_M0, K_EV_PER_NM2, T, N_BEADS)

    err_vs_infP = 100*(r2_pimc - r2_exact_infP)/r2_exact_infP
    err_vs_finiteP = 100*(r2_pimc - r2_exact_finiteP)/r2_exact_finiteP

    results.append((T, r2_pimc, r2_exact_infP, r2_exact_finiteP, err_vs_infP, err_vs_finiteP))
    print(f"T={T:5.1f}K  PIMC={r2_pimc:.4f}  exact(P=inf)={r2_exact_infP:.4f}  "
          f"exact(P={N_BEADS})={r2_exact_finiteP:.4f}  "
          f"err_vs_infP={err_vs_infP:+.2f}%  err_vs_finiteP={err_vs_finiteP:+.3f}%")

# Zapisz jako tabele do artykulu (analogiczna do Tab. 2 dla BLK)
import csv
with open("test1_single_well_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["T_K", "R2_pimc_nm2", "R2_exact_infP_nm2", "R2_exact_finiteP_nm2",
                "err_vs_infP_pct", "err_vs_finiteP_pct"])
    w.writerows(results)
