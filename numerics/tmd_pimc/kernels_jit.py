"""
v0.9.1 — Skompilowane jądra obliczeniowe PIMC przy użyciu Numba JIT.
Brak struktur obiektowych Pythona, zerowa alokacja pamięci wewnątrz pętli.

Patch v0.9.1:
- acceptance_global liczony tylko dla prób globalnych po burn-in,
  tak samo jak accepted_global. Wcześniej mianownik obejmował burn-in,
  co zaniżało raportowaną akceptację globalnych translacji.
  
Patch v0.9.2
- Przy długich przebiegach produkcyjnych (rzędu $10^6$ kroków) i dużej liczbie koralików ($P$), tablice te zajmują gigabajty pamięci RAM. Procesor zamiast wykonywać operacje zmiennoprzecinkowe, bezustannie czeka na załadowanie kolejnych bloków losowych z wolnej pamięci operacyjnej do szybkiej pamięci podręcznej L1/L2 (zjawisko cache thrashing).Numba natywnie wspiera funkcje z modułu np.random bezpośrednio wewnątrz bloków @njit. Przenosząc generowanie losowości bezpośrednio do pętli, redukujemy alokację pamięci do absolutnego zera i pozwalamy, aby współrzędne i zmienne losowe mieściły się w rejestrach procesora.

W tej wersji usunąłem z sygnatury cztery wielkie tablice losowe. Kroki translacji oraz liczby pseudolosowe są generowane skalarnie na bieżąco. Dodatkowo wprowadzona została funkcja inicjalizacji ziarna losowości Numba, gwarantująca pełną replikowalność wyników.  

Patch v0.9.3
- acc_global_rate zwraca teraz np.nan (zamiast 0.0), gdy global_attempts == 0.
  Ujednolica to semantykę z PIMCSampler.run() w sampler.py oraz z finite_mean()
  w runnerze sweepów, który używa np.nanmean przy uśrednianiu po seedach —
  wcześniej brak prób globalnych (np. przy krótkich runach --quick lub bardzo
  niskim global_move_probability) był cicho liczony jako "zero akceptacji"
  zamiast być pominięty jako brak danych.

Patch v1.16 (2026-07-14)
- Dodano run_pimc_core_staging_harmonic_jit: JIT-owe jądro dla
  PIMCSamplerStaging, UŻYWANE WYŁĄCZNIE dla harmonicznego testu
  walidacyjnego (v1.6 relative-coordinate testbed). Celowo NIE używa
  interpolacji siatki (bilinear_interpolate_2d) — potencjał harmoniczny
  V(r)=0.5*k*r^2 jest ewaluowany analitycznie wewnątrz jądra, żeby nie
  wprowadzać dodatkowego błędu interpolacji do porównania względem
  dokładnego wyniku dyskretyzacji skończonego P (harmonic_r2_primitive_finite_P
  w analytic.py). To jądro NIE jest zamiennikiem run_pimc_core_jit dla pracy
  na siatce moire — do tego nadal służy PIMCSamplerJIT.
  Zwalidowane numerycznie (samodzielny test, poza tym plikiem) przeciwko
  harmonic_r2_primitive_finite_P dla T=100K, P=16/128/320: błąd względny
  +0.36% / -0.61% / -0.42%, SEM płaski w P (0.055/0.054/0.078) —
  odtwarza sygnaturę czystego Pythona PIMCSamplerStaging, nie blow-up
  obserwowany w lokalnym samplerze przy dużym P.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def bilinear_interpolate_2d(x, y, v_grid, x_min, x_max, y_min, y_max):
    """Ultraszybka interpolacja dwuliniowa energii potencjalnej z siatki."""
    nx, ny = v_grid.shape
    
    if x <= x_min or x >= x_max or y <= y_min or y >= y_max:
        return 1e6  # Miękka bariera (poza zakresem siatki)
        
    fx = (x - x_min) / (x_max - x_min) * (nx - 1)
    fy = (y - y_min) / (y_max - y_min) * (ny - 1)
    
    ix = int(fx)
    iy = int(fy)
    
    wx = fx - ix
    wy = fy - iy
    
    v00 = v_grid[ix, iy]
    v10 = v_grid[ix + 1, iy]
    v01 = v_grid[ix, iy + 1]
    v11 = v_grid[ix + 1, iy + 1]
    
    return (1.0 - wx) * (1.0 - wy) * v00 + wx * (1.0 - wy) * v10 + (1.0 - wx) * wy * v01 + wx * wy * v11

@njit(cache=True)
def run_pimc_core_jit(n_steps, burn_in, sample_every, p_beads, path, 
                      v_grid, x_min, x_max, y_min, y_max,
                      kpf, tau, local_step_nm, global_step_nm, global_move_prob, seed):
    """
    Główne jądro Metropolis-Hastings PIMC z optymalizacją rejestrową.
    Zerowa alokacja pamięci RAM w trakcie wykonywania pętli.
    """
    # Inicjalizacja strumienia losowego Numba dla danego wątku/wywołania
    np.random.seed(seed)

    n_samples = 0
    if n_steps > burn_in:
        n_samples = 1 + (n_steps - burn_in - 1) // sample_every
        
    samples = np.empty((n_samples, p_beads, 2), dtype=np.float64)
    sample_idx = 0
    
    accepted_local = 0
    accepted_global = 0
    global_attempts = 0
    
    for step in range(n_steps):
        
        # --- LOCAL SWEEP ---
        for j in range(p_beads):
            jm = (j - 1) % p_beads
            jp = (j + 1) % p_beads
            
            r_old = path[j]
            
            # Generowanie propozycji przemieszczenia bezpośrednio w rejestrach
            dx_prop = local_step_nm * np.random.normal()
            dy_prop = local_step_nm * np.random.normal()
            
            r_new_x = r_old[0] + dx_prop
            r_new_y = r_old[1] + dy_prop
            
            # Zmiana akcji kinetycznej (sprężyny)
            dx_old_m = path[jm, 0] - r_old[0]; dy_old_m = path[jm, 1] - r_old[1]
            dx_old_p = path[jp, 0] - r_old[0]; dy_old_p = path[jp, 1] - r_old[1]
            old_spring = (dx_old_m**2 + dy_old_m**2) + (dx_old_p**2 + dy_old_p**2)
            
            dx_new_m = path[jm, 0] - r_new_x; dy_new_m = path[jm, 1] - r_new_y
            dx_new_p = path[jp, 0] - r_new_x; dy_new_p = path[jp, 1] - r_new_y
            new_spring = (dx_new_m**2 + dy_new_m**2) + (dx_new_p**2 + dy_new_p**2)
            
            dS_kinetic = kpf * (new_spring - old_spring)
            
            # Zmiana akcji potencjalnej przez interpolację siatki
            v_old = bilinear_interpolate_2d(r_old[0], r_old[1], v_grid, x_min, x_max, y_min, y_max)
            v_new = bilinear_interpolate_2d(r_new_x, r_new_y, v_grid, x_min, x_max, y_min, y_max)
            dS_potential = tau * (v_new - v_old)
            
            dS = dS_kinetic + dS_potential
            
            if dS < 0.0 or np.random.random() < np.exp(-dS):
                path[j, 0] = r_new_x
                path[j, 1] = r_new_y
                if step >= burn_in:
                    accepted_local += 1
                    
        # --- GLOBAL TRANSLATION ---
        if np.random.random() < global_move_prob:
            if step >= burn_in:
                global_attempts += 1

            disp_x = global_step_nm * np.random.normal()
            disp_y = global_step_nm * np.random.normal()
            
            v_sum_old = 0.0
            v_sum_new = 0.0
            for b in range(p_beads):
                v_sum_old += bilinear_interpolate_2d(path[b, 0], path[b, 1], v_grid, x_min, x_max, y_min, y_max)
                v_sum_new += bilinear_interpolate_2d(path[b, 0] + disp_x, path[b, 1] + disp_y, v_grid, x_min, x_max, y_min, y_max)
                
            dS_global = tau * (v_sum_new - v_sum_old)
            
            if dS_global < 0.0 or np.random.random() < np.exp(-dS_global):
                for b in range(p_beads):
                    path[b, 0] += disp_x
                    path[b, 1] += disp_y
                if step >= burn_in:
                    accepted_global += 1
                    
        # --- RECORD SAMPLE ---
        if step >= burn_in and (step - burn_in) % sample_every == 0:
            samples[sample_idx, :, 0] = path[:, 0]
            samples[sample_idx, :, 1] = path[:, 1]
            sample_idx += 1
            
    eff_steps = max(1, n_steps - burn_in)
    acc_local_rate = accepted_local / (eff_steps * p_beads)
    acc_global_rate = accepted_global / global_attempts if global_attempts > 0 else np.nan
    
    return samples, acc_local_rate, acc_global_rate



@njit(cache=True)
def bilinear_interpolate_periodic_cell(
    x, y, v_grid,
    origin_x, origin_y,
    ainv00, ainv01, ainv10, ainv11,
):
    """Periodic bilinear interpolation in fractional lattice coordinates.

    ``v_grid`` samples one complete periodic cell on an endpoint-excluded
    regular (u, v) grid. Cartesian coordinates are converted using the same
    lattice convention as GridPotential2D and wrapped with u % 1, v % 1.
    """
    nx, ny = v_grid.shape

    rel_x = x - origin_x
    rel_y = y - origin_y
    u = ainv00 * rel_x + ainv01 * rel_y
    v = ainv10 * rel_x + ainv11 * rel_y
    u = u - np.floor(u)
    v = v - np.floor(v)

    fx = u * nx
    fy = v * ny
    ix = int(np.floor(fx))
    iy = int(np.floor(fy))
    if ix >= nx:
        ix = 0
    if iy >= ny:
        iy = 0
    wx = fx - ix
    wy = fy - iy
    ix1 = (ix + 1) % nx
    iy1 = (iy + 1) % ny

    v00 = v_grid[ix, iy]
    v10 = v_grid[ix1, iy]
    v01 = v_grid[ix, iy1]
    v11 = v_grid[ix1, iy1]

    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v10
        + (1.0 - wx) * wy * v01
        + wx * wy * v11
    )


@njit(cache=True)
def run_pimc_core_jit_periodic_cell(
    n_steps, burn_in, sample_every, p_beads, path,
    v_grid,
    origin_x, origin_y,
    ainv00, ainv01, ainv10, ainv11,
    kpf, tau, local_step_nm, global_step_nm, global_move_prob, seed,
):
    """JIT local/global sampler on one periodically wrapped lattice cell.

    Unlike ``run_pimc_core_jit``, this kernel has no finite-square barrier.
    It evaluates every bead after wrapping in the supplied lattice cell, so it
    is directly comparable with a Python sampler using periodic
    ``GridPotential2D.value``. Unwrapped coordinates may diffuse through many
    cells; use wrapped-cell observables for localization analysis.
    """
    np.random.seed(seed)

    n_samples = 0
    if n_steps > burn_in:
        n_samples = 1 + (n_steps - burn_in - 1) // sample_every

    samples = np.empty((n_samples, p_beads, 2), dtype=np.float64)
    sample_idx = 0
    accepted_local = 0
    accepted_global = 0
    global_attempts = 0

    for step in range(n_steps):
        for j in range(p_beads):
            jm = (j - 1) % p_beads
            jp = (j + 1) % p_beads

            r_old_x = path[j, 0]
            r_old_y = path[j, 1]
            r_new_x = r_old_x + local_step_nm * np.random.normal()
            r_new_y = r_old_y + local_step_nm * np.random.normal()

            dx_old_m = path[jm, 0] - r_old_x
            dy_old_m = path[jm, 1] - r_old_y
            dx_old_p = path[jp, 0] - r_old_x
            dy_old_p = path[jp, 1] - r_old_y
            old_spring = (
                dx_old_m * dx_old_m + dy_old_m * dy_old_m
                + dx_old_p * dx_old_p + dy_old_p * dy_old_p
            )

            dx_new_m = path[jm, 0] - r_new_x
            dy_new_m = path[jm, 1] - r_new_y
            dx_new_p = path[jp, 0] - r_new_x
            dy_new_p = path[jp, 1] - r_new_y
            new_spring = (
                dx_new_m * dx_new_m + dy_new_m * dy_new_m
                + dx_new_p * dx_new_p + dy_new_p * dy_new_p
            )

            v_old = bilinear_interpolate_periodic_cell(
                r_old_x, r_old_y, v_grid,
                origin_x, origin_y,
                ainv00, ainv01, ainv10, ainv11,
            )
            v_new = bilinear_interpolate_periodic_cell(
                r_new_x, r_new_y, v_grid,
                origin_x, origin_y,
                ainv00, ainv01, ainv10, ainv11,
            )
            dS = kpf * (new_spring - old_spring) + tau * (v_new - v_old)

            if dS < 0.0 or np.random.random() < np.exp(-dS):
                path[j, 0] = r_new_x
                path[j, 1] = r_new_y
                if step >= burn_in:
                    accepted_local += 1

        if np.random.random() < global_move_prob:
            if step >= burn_in:
                global_attempts += 1

            disp_x = global_step_nm * np.random.normal()
            disp_y = global_step_nm * np.random.normal()
            v_sum_old = 0.0
            v_sum_new = 0.0
            for b in range(p_beads):
                v_sum_old += bilinear_interpolate_periodic_cell(
                    path[b, 0], path[b, 1], v_grid,
                    origin_x, origin_y,
                    ainv00, ainv01, ainv10, ainv11,
                )
                v_sum_new += bilinear_interpolate_periodic_cell(
                    path[b, 0] + disp_x, path[b, 1] + disp_y, v_grid,
                    origin_x, origin_y,
                    ainv00, ainv01, ainv10, ainv11,
                )

            dS_global = tau * (v_sum_new - v_sum_old)
            if dS_global < 0.0 or np.random.random() < np.exp(-dS_global):
                for b in range(p_beads):
                    path[b, 0] += disp_x
                    path[b, 1] += disp_y
                if step >= burn_in:
                    accepted_global += 1

        if step >= burn_in and (step - burn_in) % sample_every == 0:
            samples[sample_idx, :, 0] = path[:, 0]
            samples[sample_idx, :, 1] = path[:, 1]
            sample_idx += 1

    eff_steps = max(1, n_steps - burn_in)
    acc_local_rate = accepted_local / (eff_steps * p_beads)
    acc_global_rate = (
        accepted_global / global_attempts if global_attempts > 0 else np.nan
    )
    return samples, acc_local_rate, acc_global_rate

@njit(cache=True)
def run_pimc_core_staging_harmonic_jit(
    n_steps, burn_in, sample_every, p_beads, path,
    k_spring, kpf, tau, lam,
    local_step_nm, global_step_nm, global_move_prob,
    staging_segment_lengths, staging_moves_per_step, perform_local_sweep,
    seed,
):
    """
    JIT-owe jądro dla PIMCSamplerStaging — WYŁĄCZNIE potencjał harmoniczny.

    Celowo nie reużywa bilinear_interpolate_2d/v_grid z run_pimc_core_jit:
    V(r) = 0.5*k_spring*r^2 jest tu ewaluowane analitycznie, żeby błąd
    interpolacji siatki nie zanieczyszczał porównania z dokładnym wynikiem
    dyskretyzacji skończonego P (harmonic_r2_primitive_finite_P). Do pracy
    na siatce moire nadal służy PIMCSamplerJIT / run_pimc_core_jit.

    Zwraca to samo co Pythonowy PIMCSamplerStaging.run(), tyle że
    length_attempts/length_accepted jako tablice int64 (indeksowane po
    pozycji w staging_segment_lengths), nie po słowniku.
    """
    np.random.seed(seed)

    n_samples = 0
    if n_steps > burn_in:
        n_samples = 1 + (n_steps - burn_in - 1) // sample_every
    samples = np.empty((n_samples, p_beads, 2), dtype=np.float64)
    sample_idx = 0

    accepted_local = 0
    accepted_global = 0
    global_attempts = 0
    accepted_staging = 0
    staging_attempts = 0

    n_lengths = staging_segment_lengths.shape[0]
    length_attempts = np.zeros(n_lengths, dtype=np.int64)
    length_accepted = np.zeros(n_lengths, dtype=np.int64)

    max_interior = int(np.max(staging_segment_lengths)) - 1
    buf_x = np.empty(max_interior, dtype=np.float64)
    buf_y = np.empty(max_interior, dtype=np.float64)

    free_link_var = 2.0 * lam * tau

    for step in range(n_steps):
        # --- LOCAL SWEEP (analityczny potencjał harmoniczny, bez siatki) ---
        if perform_local_sweep:
            for j in range(p_beads):
                jm = (j - 1) % p_beads
                jp = (j + 1) % p_beads
                r_old_x = path[j, 0]
                r_old_y = path[j, 1]

                dx_prop = local_step_nm * np.random.normal()
                dy_prop = local_step_nm * np.random.normal()
                r_new_x = r_old_x + dx_prop
                r_new_y = r_old_y + dy_prop

                dx_old_m = path[jm, 0] - r_old_x; dy_old_m = path[jm, 1] - r_old_y
                dx_old_p = path[jp, 0] - r_old_x; dy_old_p = path[jp, 1] - r_old_y
                old_spring = (dx_old_m**2 + dy_old_m**2) + (dx_old_p**2 + dy_old_p**2)

                dx_new_m = path[jm, 0] - r_new_x; dy_new_m = path[jm, 1] - r_new_y
                dx_new_p = path[jp, 0] - r_new_x; dy_new_p = path[jp, 1] - r_new_y
                new_spring = (dx_new_m**2 + dy_new_m**2) + (dx_new_p**2 + dy_new_p**2)

                dS_kinetic = kpf * (new_spring - old_spring)

                v_old = 0.5 * k_spring * (r_old_x * r_old_x + r_old_y * r_old_y)
                v_new = 0.5 * k_spring * (r_new_x * r_new_x + r_new_y * r_new_y)
                dS_potential = tau * (v_new - v_old)

                dS = dS_kinetic + dS_potential
                if dS < 0.0 or np.random.random() < np.exp(-dS):
                    path[j, 0] = r_new_x
                    path[j, 1] = r_new_y
                    if step >= burn_in:
                        accepted_local += 1

        # --- STAGING SWEEP ---
        for _mv in range(staging_moves_per_step):
            len_idx = np.random.randint(0, n_lengths)
            L = staging_segment_lengths[len_idx]
            start = np.random.randint(0, p_beads)

            prev_x = path[start, 0]
            prev_y = path[start, 1]
            end_idx = (start + L) % p_beads
            end_x = path[end_idx, 0]
            end_y = path[end_idx, 1]

            # Warunkowy most swobodnej cząstki (Brownian bridge)
            for offset in range(1, L):
                remaining = L - offset
                denom = remaining + 1.0
                mean_x = (remaining * prev_x + end_x) / denom
                mean_y = (remaining * prev_y + end_y) / denom
                var = free_link_var * remaining / denom
                sd = np.sqrt(var)
                new_x = mean_x + sd * np.random.normal()
                new_y = mean_y + sd * np.random.normal()
                buf_x[offset - 1] = new_x
                buf_y[offset - 1] = new_y
                prev_x = new_x
                prev_y = new_y

            old_V = 0.0
            new_V = 0.0
            for offset in range(1, L):
                idx = (start + offset) % p_beads
                ox = path[idx, 0]; oy = path[idx, 1]
                old_V += 0.5 * k_spring * (ox * ox + oy * oy)
                nx = buf_x[offset - 1]; ny = buf_y[offset - 1]
                new_V += 0.5 * k_spring * (nx * nx + ny * ny)

            dS_pot = tau * (new_V - old_V)

            accept = dS_pot < 0.0 or np.random.random() < np.exp(-dS_pot)
            if step >= burn_in:
                staging_attempts += 1
                length_attempts[len_idx] += 1
            if accept:
                for offset in range(1, L):
                    idx = (start + offset) % p_beads
                    path[idx, 0] = buf_x[offset - 1]
                    path[idx, 1] = buf_y[offset - 1]
                if step >= burn_in:
                    accepted_staging += 1
                    length_accepted[len_idx] += 1

        # --- GLOBAL TRANSLATION (analityczny potencjał) ---
        if np.random.random() < global_move_prob:
            if step >= burn_in:
                global_attempts += 1
            disp_x = global_step_nm * np.random.normal()
            disp_y = global_step_nm * np.random.normal()

            v_sum_old = 0.0
            v_sum_new = 0.0
            for b in range(p_beads):
                bx = path[b, 0]; by = path[b, 1]
                v_sum_old += 0.5 * k_spring * (bx * bx + by * by)
                nbx = bx + disp_x; nby = by + disp_y
                v_sum_new += 0.5 * k_spring * (nbx * nbx + nby * nby)

            dS_global = tau * (v_sum_new - v_sum_old)
            if dS_global < 0.0 or np.random.random() < np.exp(-dS_global):
                for b in range(p_beads):
                    path[b, 0] += disp_x
                    path[b, 1] += disp_y
                if step >= burn_in:
                    accepted_global += 1

        # --- RECORD SAMPLE ---
        if step >= burn_in and (step - burn_in) % sample_every == 0:
            samples[sample_idx, :, 0] = path[:, 0]
            samples[sample_idx, :, 1] = path[:, 1]
            sample_idx += 1

    eff_steps = max(1, n_steps - burn_in)
    acc_local_rate = accepted_local / (eff_steps * p_beads) if perform_local_sweep else np.nan
    acc_staging_rate = accepted_staging / staging_attempts if staging_attempts > 0 else np.nan
    acc_global_rate = accepted_global / global_attempts if global_attempts > 0 else np.nan

    return samples, acc_local_rate, acc_staging_rate, acc_global_rate, length_attempts, length_accepted
