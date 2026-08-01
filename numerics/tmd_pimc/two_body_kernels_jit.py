"""JIT-compiled kernel for the coupled electron-hole staging sampler.

Generalises run_pimc_core_staging_harmonic_jit (kernels_jit.py) from one
ring polymer with an analytic harmonic potential to two independently
propagating ring polymers (mass_e != mass_h in general) coupled through:

  - a radial interaction table V_int(rho), rho = |r_e - r_h| -- built by
    resampling BilayerKeldyshWallPotential (or any Potential2D with radial
    symmetry) onto a UNIFORM grid once, in plain Python, before the JIT
    loop starts. The production BLK table (bilayer_keldysh_potential.py)
    uses a hybrid log+linear grid for adaptive resolution near rho=0; that
    is excellent for scipy.interp but not for O(1) njit indexing, so
    build_uniform_interaction_table() below re-evaluates the *existing,
    already-validated* potential object on a dense uniform grid instead of
    reusing its internal non-uniform table directly.

  - optional independent one-body landscapes V_e(r_e), V_h(r_h) on ordinary
    2D grids, reusing bilinear_interpolate_2d from kernels_jit.py. Pass
    has_landscape_e=False / has_landscape_h=False (with any 1x1 dummy grid)
    to reproduce the current validated baseline (V_e = V_h = 0), exactly as
    tested by run_two_body_validation.py / two_body_action.ZeroPotential.

Not yet validated against the pure-Python TwoBodyPIMCSamplerStaging -- run
the baseline comparison (V_e = V_h = 0, same config as
two_body_baseline.json) before trusting this for production, the same way
run_pimc_core_staging_harmonic_jit was checked against
harmonic_r2_primitive_finite_P before being trusted.
"""

import numpy as np
from numba import njit

from .kernels_jit import bilinear_interpolate_2d


def build_uniform_interaction_table(interaction_potential, r_max_nm=80.0, n_points=20000):
    """Resample any radially symmetric Potential2D onto a uniform 1D grid.

    Returns (v_table, r_min, r_max) ready for radial_table_interpolate.
    Evaluates the *existing* potential object (e.g. BilayerKeldyshWallPotential,
    already validated in v1.8a/v1.8b) rather than re-deriving the physics --
    this function only changes the grid the values are stored on.
    """
    r_uniform = np.linspace(0.0, float(r_max_nm), int(n_points))
    points = np.column_stack([r_uniform, np.zeros_like(r_uniform)])
    v_uniform = np.asarray(interaction_potential.value(points), dtype=np.float64)
    if not np.all(np.isfinite(v_uniform)):
        raise FloatingPointError(
            "Interaction potential produced non-finite values while building "
            "the uniform JIT lookup table."
        )
    return v_uniform, 0.0, float(r_max_nm)


@njit(cache=True, inline="always")
def radial_table_interpolate(rho, table, r_min, r_max):
    """O(1) linear interpolation on a uniform radial grid, with edge clamping.

    Clamping (rather than the 1e6 soft barrier used by bilinear_interpolate_2d
    for off-grid Cartesian points) is intentional here: the BLK interaction
    table already has a soft confining wall baked in via
    BilayerKeldyshWallPotential before build_uniform_interaction_table is
    called, so rho reaching r_max should not happen in a converged run: if it
    does, clamping to the last (already very large, wall-dominated) table
    value is the correct fallback rather than an artificial hard cutoff.
    """
    n = table.shape[0]
    if rho <= r_min:
        return table[0]
    if rho >= r_max:
        return table[n - 1]
    f = (rho - r_min) / (r_max - r_min) * (n - 1)
    i = int(f)
    if i >= n - 1:
        return table[n - 1]
    w = f - i
    return (1.0 - w) * table[i] + w * table[i + 1]


@njit(cache=True)
def run_pimc_core_two_body_staging_jit(
    n_steps, burn_in, sample_every, p_beads,
    path_e, path_h,
    kpf_e, kpf_h, tau,
    v_int_table, r_int_min, r_int_max,
    has_landscape_e, v_grid_e, xe_min, xe_max, ye_min, ye_max,
    has_landscape_h, v_grid_h, xh_min, xh_max, yh_min, yh_max,
    local_step_nm, global_step_nm, global_move_prob,
    staging_segment_lengths, staging_moves_per_step, perform_local_sweep,
    seed,
):
    """Two coupled ring polymers, staging + local + joint-global moves.

    Returns the same quantities as the pure-Python TwoBodyPIMCSamplerStaging
    .run(): samples_e, samples_h, acceptance_local_e/h, acceptance_staging,
    acceptance_global_joint, plus per-length staging diagnostics.
    """
    np.random.seed(seed)

    n_samples = 0
    if n_steps > burn_in:
        n_samples = 1 + (n_steps - burn_in - 1) // sample_every
    samples_e = np.empty((n_samples, p_beads, 2), dtype=np.float64)
    samples_h = np.empty((n_samples, p_beads, 2), dtype=np.float64)
    sample_idx = 0

    accepted_local_e = 0
    accepted_local_h = 0
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

    free_link_var_e = 0.5 / kpf_e
    free_link_var_h = 0.5 / kpf_h

    for step in range(n_steps):
        # --- LOCAL SWEEP: electron chain ---
        if perform_local_sweep:
            for j in range(p_beads):
                jm = (j - 1) % p_beads
                jp = (j + 1) % p_beads
                r_old_x = path_e[j, 0]; r_old_y = path_e[j, 1]

                dx_prop = local_step_nm * np.random.normal()
                dy_prop = local_step_nm * np.random.normal()
                r_new_x = r_old_x + dx_prop
                r_new_y = r_old_y + dy_prop

                dx_old_m = path_e[jm, 0] - r_old_x; dy_old_m = path_e[jm, 1] - r_old_y
                dx_old_p = path_e[jp, 0] - r_old_x; dy_old_p = path_e[jp, 1] - r_old_y
                old_spring = (dx_old_m**2 + dy_old_m**2) + (dx_old_p**2 + dy_old_p**2)

                dx_new_m = path_e[jm, 0] - r_new_x; dy_new_m = path_e[jm, 1] - r_new_y
                dx_new_p = path_e[jp, 0] - r_new_x; dy_new_p = path_e[jp, 1] - r_new_y
                new_spring = (dx_new_m**2 + dy_new_m**2) + (dx_new_p**2 + dy_new_p**2)

                dS_kinetic = kpf_e * (new_spring - old_spring)

                rh_x = path_h[j, 0]; rh_y = path_h[j, 1]
                rho_old = np.sqrt((r_old_x - rh_x)**2 + (r_old_y - rh_y)**2)
                rho_new = np.sqrt((r_new_x - rh_x)**2 + (r_new_y - rh_y)**2)
                v_old = radial_table_interpolate(rho_old, v_int_table, r_int_min, r_int_max)
                v_new = radial_table_interpolate(rho_new, v_int_table, r_int_min, r_int_max)
                if has_landscape_e:
                    v_old += bilinear_interpolate_2d(r_old_x, r_old_y, v_grid_e, xe_min, xe_max, ye_min, ye_max)
                    v_new += bilinear_interpolate_2d(r_new_x, r_new_y, v_grid_e, xe_min, xe_max, ye_min, ye_max)

                dS = dS_kinetic + tau * (v_new - v_old)
                if dS < 0.0 or np.random.random() < np.exp(-dS):
                    path_e[j, 0] = r_new_x
                    path_e[j, 1] = r_new_y
                    if step >= burn_in:
                        accepted_local_e += 1

            # --- LOCAL SWEEP: hole chain ---
            for j in range(p_beads):
                jm = (j - 1) % p_beads
                jp = (j + 1) % p_beads
                r_old_x = path_h[j, 0]; r_old_y = path_h[j, 1]

                dx_prop = local_step_nm * np.random.normal()
                dy_prop = local_step_nm * np.random.normal()
                r_new_x = r_old_x + dx_prop
                r_new_y = r_old_y + dy_prop

                dx_old_m = path_h[jm, 0] - r_old_x; dy_old_m = path_h[jm, 1] - r_old_y
                dx_old_p = path_h[jp, 0] - r_old_x; dy_old_p = path_h[jp, 1] - r_old_y
                old_spring = (dx_old_m**2 + dy_old_m**2) + (dx_old_p**2 + dy_old_p**2)

                dx_new_m = path_h[jm, 0] - r_new_x; dy_new_m = path_h[jm, 1] - r_new_y
                dx_new_p = path_h[jp, 0] - r_new_x; dy_new_p = path_h[jp, 1] - r_new_y
                new_spring = (dx_new_m**2 + dy_new_m**2) + (dx_new_p**2 + dy_new_p**2)

                dS_kinetic = kpf_h * (new_spring - old_spring)

                re_x = path_e[j, 0]; re_y = path_e[j, 1]
                rho_old = np.sqrt((re_x - r_old_x)**2 + (re_y - r_old_y)**2)
                rho_new = np.sqrt((re_x - r_new_x)**2 + (re_y - r_new_y)**2)
                v_old = radial_table_interpolate(rho_old, v_int_table, r_int_min, r_int_max)
                v_new = radial_table_interpolate(rho_new, v_int_table, r_int_min, r_int_max)
                if has_landscape_h:
                    v_old += bilinear_interpolate_2d(r_old_x, r_old_y, v_grid_h, xh_min, xh_max, yh_min, yh_max)
                    v_new += bilinear_interpolate_2d(r_new_x, r_new_y, v_grid_h, xh_min, xh_max, yh_min, yh_max)

                dS = dS_kinetic + tau * (v_new - v_old)
                if dS < 0.0 or np.random.random() < np.exp(-dS):
                    path_h[j, 0] = r_new_x
                    path_h[j, 1] = r_new_y
                    if step >= burn_in:
                        accepted_local_h += 1

        # --- STAGING SWEEP: alternate electron / hole segment updates ---
        for _mv in range(staging_moves_per_step):
            # electron segment
            len_idx = np.random.randint(0, n_lengths)
            L = staging_segment_lengths[len_idx]
            start = np.random.randint(0, p_beads)
            prev_x = path_e[start, 0]; prev_y = path_e[start, 1]
            end_idx = (start + L) % p_beads
            end_x = path_e[end_idx, 0]; end_y = path_e[end_idx, 1]
            for offset in range(1, L):
                remaining = L - offset
                denom = remaining + 1.0
                mean_x = (remaining * prev_x + end_x) / denom
                mean_y = (remaining * prev_y + end_y) / denom
                var = free_link_var_e * remaining / denom
                sd = np.sqrt(var)
                new_x = mean_x + sd * np.random.normal()
                new_y = mean_y + sd * np.random.normal()
                buf_x[offset - 1] = new_x
                buf_y[offset - 1] = new_y
                prev_x = new_x; prev_y = new_y

            old_V = 0.0
            new_V = 0.0
            for offset in range(1, L):
                idx = (start + offset) % p_beads
                ox = path_e[idx, 0]; oy = path_e[idx, 1]
                rhx = path_h[idx, 0]; rhy = path_h[idx, 1]
                old_rho = np.sqrt((ox - rhx)**2 + (oy - rhy)**2)
                old_V += radial_table_interpolate(old_rho, v_int_table, r_int_min, r_int_max)
                nx = buf_x[offset - 1]; ny = buf_y[offset - 1]
                new_rho = np.sqrt((nx - rhx)**2 + (ny - rhy)**2)
                new_V += radial_table_interpolate(new_rho, v_int_table, r_int_min, r_int_max)
                if has_landscape_e:
                    old_V += bilinear_interpolate_2d(ox, oy, v_grid_e, xe_min, xe_max, ye_min, ye_max)
                    new_V += bilinear_interpolate_2d(nx, ny, v_grid_e, xe_min, xe_max, ye_min, ye_max)

            dS_pot = tau * (new_V - old_V)
            accept = dS_pot < 0.0 or np.random.random() < np.exp(-dS_pot)
            if step >= burn_in:
                staging_attempts += 1
                length_attempts[len_idx] += 1
            if accept:
                for offset in range(1, L):
                    idx = (start + offset) % p_beads
                    path_e[idx, 0] = buf_x[offset - 1]
                    path_e[idx, 1] = buf_y[offset - 1]
                if step >= burn_in:
                    accepted_staging += 1
                    length_accepted[len_idx] += 1

            # hole segment
            len_idx = np.random.randint(0, n_lengths)
            L = staging_segment_lengths[len_idx]
            start = np.random.randint(0, p_beads)
            prev_x = path_h[start, 0]; prev_y = path_h[start, 1]
            end_idx = (start + L) % p_beads
            end_x = path_h[end_idx, 0]; end_y = path_h[end_idx, 1]
            for offset in range(1, L):
                remaining = L - offset
                denom = remaining + 1.0
                mean_x = (remaining * prev_x + end_x) / denom
                mean_y = (remaining * prev_y + end_y) / denom
                var = free_link_var_h * remaining / denom
                sd = np.sqrt(var)
                new_x = mean_x + sd * np.random.normal()
                new_y = mean_y + sd * np.random.normal()
                buf_x[offset - 1] = new_x
                buf_y[offset - 1] = new_y
                prev_x = new_x; prev_y = new_y

            old_V = 0.0
            new_V = 0.0
            for offset in range(1, L):
                idx = (start + offset) % p_beads
                ohx = path_h[idx, 0]; ohy = path_h[idx, 1]
                rex = path_e[idx, 0]; rey = path_e[idx, 1]
                old_rho = np.sqrt((rex - ohx)**2 + (rey - ohy)**2)
                old_V += radial_table_interpolate(old_rho, v_int_table, r_int_min, r_int_max)
                nx = buf_x[offset - 1]; ny = buf_y[offset - 1]
                new_rho = np.sqrt((rex - nx)**2 + (rey - ny)**2)
                new_V += radial_table_interpolate(new_rho, v_int_table, r_int_min, r_int_max)
                if has_landscape_h:
                    old_V += bilinear_interpolate_2d(ohx, ohy, v_grid_h, xh_min, xh_max, yh_min, yh_max)
                    new_V += bilinear_interpolate_2d(nx, ny, v_grid_h, xh_min, xh_max, yh_min, yh_max)

            dS_pot = tau * (new_V - old_V)
            accept = dS_pot < 0.0 or np.random.random() < np.exp(-dS_pot)
            if step >= burn_in:
                staging_attempts += 1
                length_attempts[len_idx] += 1
            if accept:
                for offset in range(1, L):
                    idx = (start + offset) % p_beads
                    path_h[idx, 0] = buf_x[offset - 1]
                    path_h[idx, 1] = buf_y[offset - 1]
                if step >= burn_in:
                    accepted_staging += 1
                    length_accepted[len_idx] += 1

        # --- GLOBAL JOINT TRANSLATION (interaction-invariant by construction) ---
        if np.random.random() < global_move_prob:
            if step >= burn_in:
                global_attempts += 1
            disp_x = global_step_nm * np.random.normal()
            disp_y = global_step_nm * np.random.normal()

            v_sum_old = 0.0
            v_sum_new = 0.0
            if has_landscape_e:
                for b in range(p_beads):
                    bx = path_e[b, 0]; by = path_e[b, 1]
                    v_sum_old += bilinear_interpolate_2d(bx, by, v_grid_e, xe_min, xe_max, ye_min, ye_max)
                    v_sum_new += bilinear_interpolate_2d(bx + disp_x, by + disp_y, v_grid_e, xe_min, xe_max, ye_min, ye_max)
            if has_landscape_h:
                for b in range(p_beads):
                    bx = path_h[b, 0]; by = path_h[b, 1]
                    v_sum_old += bilinear_interpolate_2d(bx, by, v_grid_h, xh_min, xh_max, yh_min, yh_max)
                    v_sum_new += bilinear_interpolate_2d(bx + disp_x, by + disp_y, v_grid_h, xh_min, xh_max, yh_min, yh_max)

            dS_global = tau * (v_sum_new - v_sum_old)
            if dS_global < 0.0 or np.random.random() < np.exp(-dS_global):
                for b in range(p_beads):
                    path_e[b, 0] += disp_x
                    path_e[b, 1] += disp_y
                    path_h[b, 0] += disp_x
                    path_h[b, 1] += disp_y
                if step >= burn_in:
                    accepted_global += 1

        # --- RECORD SAMPLE ---
        if step >= burn_in and (step - burn_in) % sample_every == 0:
            samples_e[sample_idx, :, 0] = path_e[:, 0]
            samples_e[sample_idx, :, 1] = path_e[:, 1]
            samples_h[sample_idx, :, 0] = path_h[:, 0]
            samples_h[sample_idx, :, 1] = path_h[:, 1]
            sample_idx += 1

    eff_steps = max(1, n_steps - burn_in)
    acc_local_e = accepted_local_e / (eff_steps * p_beads) if perform_local_sweep else np.nan
    acc_local_h = accepted_local_h / (eff_steps * p_beads) if perform_local_sweep else np.nan
    acc_staging = accepted_staging / staging_attempts if staging_attempts > 0 else np.nan
    acc_global = accepted_global / global_attempts if global_attempts > 0 else np.nan

    return (
        samples_e, samples_h,
        acc_local_e, acc_local_h, acc_staging, acc_global,
        length_attempts, length_accepted,
    )
