import numpy as np
from numba import njit

from .kernels_jit import bilinear_interpolate_periodic_cell
from .two_body_kernels_jit import radial_table_interpolate  # used for interaction lookup

def build_periodic_cell_grid(potential, period_nm, origin_nm=(0.0, 0.0), grid_size=200):
    """
    Rasterize one periodic cell of a MoirePotential-like Potential2D.

    Returns dict with:
      - v_grid: (grid_size, grid_size) float64 array
      - origin_x, origin_y: floats
      - ainv00, ainv01, ainv10, ainv11: floats (inverse lattice matrix entries)
    """
    G = 4.0 * np.pi / (np.sqrt(3.0) * period_nm)
    G1 = np.array([G, 0.0], dtype=np.float64)
    G2 = np.array([-0.5 * G, np.sqrt(3.0) / 2.0 * G], dtype=np.float64)
    Gmat = np.column_stack([G1, G2])
    A = 2.0 * np.pi * np.linalg.inv(Gmat).T
    Ainv = np.linalg.inv(A)

    n = int(grid_size)
    if n < 2:
        raise ValueError("grid_size must be >= 2")

    u_ticks = np.arange(n, dtype=np.float64) / n
    v_ticks = np.arange(n, dtype=np.float64) / n
    U, V = np.meshgrid(u_ticks, v_ticks, indexing="ij")
    ox, oy = float(origin_nm[0]), float(origin_nm[1])
    X = ox + U * A[0, 0] + V * A[0, 1]
    Y = oy + U * A[1, 0] + V * A[1, 1]
    pts = np.column_stack([X.ravel(), Y.ravel()])

    v_flat = np.asarray(potential.value(pts), dtype=np.float64)
    if v_flat.size != n * n:
        raise RuntimeError("Rasterized potential returned unexpected number of values")
    if not np.all(np.isfinite(v_flat)):
        raise FloatingPointError("Non-finite values encountered while rasterizing periodic cell")

    v_grid = v_flat.reshape(n, n)

    return {
        "v_grid": v_grid,
        "origin_x": float(ox),
        "origin_y": float(oy),
        "ainv00": float(Ainv[0, 0]),
        "ainv01": float(Ainv[0, 1]),
        "ainv10": float(Ainv[1, 0]),
        "ainv11": float(Ainv[1, 1]),
    }


@njit(cache=True)
def run_pimc_core_two_body_periodic_staging_jit(
    n_steps, burn_in, sample_every, p_beads,
    path_e, path_h,
    kpf_e, kpf_h, tau,
    v_int_table, r_int_min, r_int_max,
    v_grid_e, origin_e_x, origin_e_y, ainv_e00, ainv_e01, ainv_e10, ainv_e11,
    field_e_x, field_e_y,
    v_grid_h, origin_h_x, origin_h_y, ainv_h00, ainv_h01, ainv_h10, ainv_h11,
    field_h_x, field_h_y,
    local_step_nm, global_step_nm, global_move_prob,
    staging_segment_lengths, staging_moves_per_step, perform_local_sweep,
    seed,
):
    """
    Numba-jitted core PIMC loop for two-body periodic landscapes plus linear field.

    Returns:
      samples_e, samples_h, acc_local_e, acc_local_h, acc_staging, acc_global
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
    max_interior = int(np.max(staging_segment_lengths)) - 1
    buf_x = np.empty(max_interior, dtype=np.float64)
    buf_y = np.empty(max_interior, dtype=np.float64)

    free_link_var_e = 0.5 / kpf_e
    free_link_var_h = 0.5 / kpf_h

    for step in range(n_steps):
        # LOCAL SWEEP electron
        if perform_local_sweep:
            for j in range(p_beads):
                jm = (j - 1) % p_beads
                jp = (j + 1) % p_beads
                r_old_x = path_e[j, 0]; r_old_y = path_e[j, 1]

                r_new_x = r_old_x + local_step_nm * np.random.normal()
                r_new_y = r_old_y + local_step_nm * np.random.normal()

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
                v_old += bilinear_interpolate_periodic_cell(
                    r_old_x, r_old_y, v_grid_e, origin_e_x, origin_e_y,
                    ainv_e00, ainv_e01, ainv_e10, ainv_e11,
                ) + field_e_x * r_old_x + field_e_y * r_old_y
                v_new += bilinear_interpolate_periodic_cell(
                    r_new_x, r_new_y, v_grid_e, origin_e_x, origin_e_y,
                    ainv_e00, ainv_e01, ainv_e10, ainv_e11,
                ) + field_e_x * r_new_x + field_e_y * r_new_y

                dS = dS_kinetic + tau * (v_new - v_old)
                if dS < 0.0 or np.random.random() < np.exp(-dS):
                    path_e[j, 0] = r_new_x
                    path_e[j, 1] = r_new_y
                    if step >= burn_in:
                        accepted_local_e += 1

            # LOCAL SWEEP hole
            for j in range(p_beads):
                jm = (j - 1) % p_beads
                jp = (j + 1) % p_beads
                r_old_x = path_h[j, 0]; r_old_y = path_h[j, 1]

                r_new_x = r_old_x + local_step_nm * np.random.normal()
                r_new_y = r_old_y + local_step_nm * np.random.normal()

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
                v_old += bilinear_interpolate_periodic_cell(
                    r_old_x, r_old_y, v_grid_h, origin_h_x, origin_h_y,
                    ainv_h00, ainv_h01, ainv_h10, ainv_h11,
                ) + field_h_x * r_old_x + field_h_y * r_old_y
                v_new += bilinear_interpolate_periodic_cell(
                    r_new_x, r_new_y, v_grid_h, origin_h_x, origin_h_y,
                    ainv_h00, ainv_h01, ainv_h10, ainv_h11,
                ) + field_h_x * r_new_x + field_h_y * r_new_y

                dS = dS_kinetic + tau * (v_new - v_old)
                if dS < 0.0 or np.random.random() < np.exp(-dS):
                    path_h[j, 0] = r_new_x
                    path_h[j, 1] = r_new_y
                    if step >= burn_in:
                        accepted_local_h += 1

        # STAGING SWEEP (electron then hole)
        for _mv in range(staging_moves_per_step):
            # electron staging
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
                sd = np.sqrt(free_link_var_e * remaining / denom)
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
                old_V += bilinear_interpolate_periodic_cell(
                    ox, oy, v_grid_e, origin_e_x, origin_e_y,
                    ainv_e00, ainv_e01, ainv_e10, ainv_e11,
                ) + field_e_x * ox + field_e_y * oy
                nx = buf_x[offset - 1]; ny = buf_y[offset - 1]
                new_rho = np.sqrt((nx - rhx)**2 + (ny - rhy)**2)
                new_V += radial_table_interpolate(new_rho, v_int_table, r_int_min, r_int_max)
                new_V += bilinear_interpolate_periodic_cell(
                    nx, ny, v_grid_e, origin_e_x, origin_e_y,
                    ainv_e00, ainv_e01, ainv_e10, ainv_e11,
                ) + field_e_x * nx + field_e_y * ny

            dS_pot = tau * (new_V - old_V)
            if step >= burn_in:
                staging_attempts += 1
            accept = dS_pot < 0.0 or np.random.random() < np.exp(-dS_pot)
            if accept:
                for offset in range(1, L):
                    idx = (start + offset) % p_beads
                    path_e[idx, 0] = buf_x[offset - 1]
                    path_e[idx, 1] = buf_y[offset - 1]
                if step >= burn_in:
                    accepted_staging += 1

            # hole staging (mirror of electron)
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
                sd = np.sqrt(free_link_var_h * remaining / denom)
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
                old_V += bilinear_interpolate_periodic_cell(
                    ohx, ohy, v_grid_h, origin_h_x, origin_h_y,
                    ainv_h00, ainv_h01, ainv_h10, ainv_h11,
                ) + field_h_x * ohx + field_h_y * ohy
                nx = buf_x[offset - 1]; ny = buf_y[offset - 1]
                new_rho = np.sqrt((rex - nx)**2 + (rey - ny)**2)
                new_V += radial_table_interpolate(new_rho, v_int_table, r_int_min, r_int_max)
                new_V += bilinear_interpolate_periodic_cell(
                    nx, ny, v_grid_h, origin_h_x, origin_h_y,
                    ainv_h00, ainv_h01, ainv_h10, ainv_h11,
                ) + field_h_x * nx + field_h_y * ny

            dS_pot = tau * (new_V - old_V)
            if step >= burn_in:
                staging_attempts += 1
            accept = dS_pot < 0.0 or np.random.random() < np.exp(-dS_pot)
            if accept:
                for offset in range(1, L):
                    idx = (start + offset) % p_beads
                    path_h[idx, 0] = buf_x[offset - 1]
                    path_h[idx, 1] = buf_y[offset - 1]
                if step >= burn_in:
                    accepted_staging += 1

        # GLOBAL JOINT TRANSLATION
        if np.random.random() < global_move_prob:
            if step >= burn_in:
                global_attempts += 1
            disp_x = global_step_nm * np.random.normal()
            disp_y = global_step_nm * np.random.normal()

            v_sum_old = 0.0
            v_sum_new = 0.0
            for b in range(p_beads):
                bx = path_e[b, 0]; by = path_e[b, 1]
                v_sum_old += bilinear_interpolate_periodic_cell(
                    bx, by, v_grid_e, origin_e_x, origin_e_y,
                    ainv_e00, ainv_e01, ainv_e10, ainv_e11,
                ) + field_e_x * bx + field_e_y * by
                nbx, nby = bx + disp_x, by + disp_y
                v_sum_new += bilinear_interpolate_periodic_cell(
                    nbx, nby, v_grid_e, origin_e_x, origin_e_y,
                    ainv_e00, ainv_e01, ainv_e10, ainv_e11,
                ) + field_e_x * nbx + field_e_y * nby
            for b in range(p_beads):
                bx = path_h[b, 0]; by = path_h[b, 1]
                v_sum_old += bilinear_interpolate_periodic_cell(
                    bx, by, v_grid_h, origin_h_x, origin_h_y,
                    ainv_h00, ainv_h01, ainv_h10, ainv_h11,
                ) + field_h_x * bx + field_h_y * by
                nbx, nby = bx + disp_x, by + disp_y
                v_sum_new += bilinear_interpolate_periodic_cell(
                    nbx, nby, v_grid_h, origin_h_x, origin_h_y,
                    ainv_h00, ainv_h01, ainv_h10, ainv_h11,
                ) + field_h_x * nbx + field_h_y * nby

            dS_global = tau * (v_sum_new - v_sum_old)
            if dS_global < 0.0 or np.random.random() < np.exp(-dS_global):
                for b in range(p_beads):
                    path_e[b, 0] += disp_x
                    path_e[b, 1] += disp_y
                    path_h[b, 0] += disp_x
                    path_h[b, 1] += disp_y
                if step >= burn_in:
                    accepted_global += 1

        # RECORD SAMPLE
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
    )

