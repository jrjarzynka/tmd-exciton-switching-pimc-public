"""JIT-compiled counterpart of TwoBodyPIMCSamplerStaging.

Rasterizes action.potential_e and action.potential_h onto uniform 2D grids
once at construction time, mirroring PIMCSamplerJIT._build_potential_grid
in sampler.py exactly (same np.linspace(x_min, x_max, grid_size, endpoint
implied True) + meshgrid(indexing="ij") convention, so bilinear_interpolate_2d
behaves identically to the already-validated single-body JIT sampler).

CAUTION (found by regression-testing this file before shipping it): a true
zero potential rasterized onto a *finite* grid is NOT equivalent to "no
potential". bilinear_interpolate_2d imposes a hard 1e6 penalty outside
[x_min, x_max], so an unconfined composite pair that genuinely random-walks
past the box edge over a long run gets artificially caged. With
rasterize_e/h=True (previously the only option) on a true ZeroPotential,
acceptance_global_joint dropped from the expected exact 1.000 to ~0.97-0.98
over 60000 steps because some walkers reached the +/-40nm edge -- a small
but real unphysical artifact. Pass rasterize_e=False / rasterize_h=False to
recover the exact original unconfined behaviour (acceptance_global_joint
verified back to exactly 1.000 after this fix); use rasterize=True only for
genuinely confining/bounded landscapes (moire, Stark, strain, ...), where
the physical potential itself keeps the pair away from the grid edge.

Usage:

    action = TwoBodyRingPolymerAction(..., potential_e=V_e, potential_h=V_h,
                                       potential_interaction=interaction)
    sampler = TwoBodyPIMCSamplerStagingJIT(action, rng_seed=1000,
                                            landscape_grid_size=400,
                                            landscape_grid_range_nm=40.0)
    result = sampler.run(n_steps=60000, burn_in=15000, sample_every=20)

    # true V_e = V_h = 0 baseline (no artificial box):
    sampler = TwoBodyPIMCSamplerStagingJIT(action, rasterize_e=False, rasterize_h=False)
"""

import numpy as np

from .two_body_kernels_jit import (
    build_uniform_interaction_table,
    run_pimc_core_two_body_staging_jit,
)


class TwoBodyPIMCSamplerStagingJIT:
    def __init__(self, action, local_step_nm=0.20, global_step_nm=1.00,
                 global_move_probability=0.20, rng_seed=1234,
                 staging_segment_lengths=(4, 8, 16, 32, 64, 128, 256),
                 staging_moves_per_step=2, perform_local_sweep=True,
                 interaction_table_r_max_nm=80.0,
                 interaction_table_n_points=20000,
                 landscape_grid_size=400,
                 landscape_grid_range_nm=40.0,
                 rasterize_e=True,
                 rasterize_h=True):
        self.action = action
        self.local_step_nm = local_step_nm
        self.global_step_nm = global_step_nm
        self.global_move_probability = global_move_probability
        self.rng_seed = rng_seed
        self.staging_moves_per_step = int(staging_moves_per_step)
        self.perform_local_sweep = bool(perform_local_sweep)
        self.rasterize_e = bool(rasterize_e)
        self.rasterize_h = bool(rasterize_h)

        P = int(action.n_beads)
        if P < 3:
            raise ValueError("TwoBodyPIMCSamplerStagingJIT requires n_beads >= 3")
        if landscape_grid_size < 4:
            raise ValueError("landscape_grid_size must be >= 4")
        if landscape_grid_range_nm <= 0.0:
            raise ValueError("landscape_grid_range_nm must be positive")

        lengths = sorted(
            {int(L) for L in staging_segment_lengths if 2 <= int(L) < P}
        )
        if not lengths:
            lengths = [P - 1]
        self._valid_segment_lengths = np.asarray(lengths, dtype=np.int64)

        self.rng = np.random.default_rng(self.rng_seed)

        # Resample the (already-validated) interaction potential onto a
        # uniform radial grid once, up front -- not inside the JIT loop.
        v_table, r_min, r_max = build_uniform_interaction_table(
            action.potential_interaction,
            r_max_nm=interaction_table_r_max_nm,
            n_points=interaction_table_n_points,
        )
        self._v_int_table = v_table
        self._r_int_min = r_min
        self._r_int_max = r_max

        self.landscape_grid_size = int(landscape_grid_size)
        self.landscape_grid_range_nm = float(landscape_grid_range_nm)
        self._dummy_grid = {"v_grid": np.zeros((1, 1), dtype=np.float64),
                             "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
        self._grid_e = self._rasterize(action.potential_e) if self.rasterize_e else self._dummy_grid
        self._grid_h = self._rasterize(action.potential_h) if self.rasterize_h else self._dummy_grid

    def _rasterize(self, potential):
        """Evaluate a Potential2D on a uniform square grid for bilinear_interpolate_2d.

        Same convention as PIMCSamplerJIT._build_potential_grid (finite_square
        mode) in sampler.py: np.linspace(x_min, x_max, grid_size) with
        meshgrid(indexing="ij"), so an already-validated interpolation
        routine is reused unmodified.
        """
        n = self.landscape_grid_size
        rng = self.landscape_grid_range_nm
        x_min, x_max = -rng, rng
        y_min, y_max = -rng, rng
        x_ticks = np.linspace(x_min, x_max, n)
        y_ticks = np.linspace(y_min, y_max, n)
        X, Y = np.meshgrid(x_ticks, y_ticks, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        V_flat = np.asarray(potential.value(pts), dtype=np.float64)
        v_grid = V_flat.reshape(n, n)
        if not np.all(np.isfinite(v_grid)):
            raise FloatingPointError(
                f"{type(potential).__name__} produced non-finite values while "
                "rasterizing onto the JIT landscape grid."
            )
        return {"v_grid": v_grid, "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def initialize_paths(self, center_e=(0.0, 0.0), center_h=(0.0, 0.0), spread_nm=0.1):
        P = self.action.n_beads
        path_e = np.array(center_e, dtype=np.float64) + spread_nm * self.rng.standard_normal((P, 2))
        path_h = np.array(center_h, dtype=np.float64) + spread_nm * self.rng.standard_normal((P, 2))
        return path_e, path_h

    def run(self, n_steps=10000, burn_in=2500, sample_every=20,
            center_e=(0.0, 0.0), center_h=(0.0, 0.0)):
        path_e, path_h = self.initialize_paths(center_e=center_e, center_h=center_h)

        (
            samples_e, samples_h,
            acc_local_e, acc_local_h, acc_staging, acc_global,
            len_attempts, len_accepted,
        ) = run_pimc_core_two_body_staging_jit(
            n_steps=n_steps,
            burn_in=burn_in,
            sample_every=sample_every,
            p_beads=self.action.n_beads,
            path_e=path_e,
            path_h=path_h,
            kpf_e=self.action._kpf_e,
            kpf_h=self.action._kpf_h,
            tau=self.action._tau,
            v_int_table=self._v_int_table,
            r_int_min=self._r_int_min,
            r_int_max=self._r_int_max,
            has_landscape_e=self.rasterize_e,
            v_grid_e=self._grid_e["v_grid"],
            xe_min=self._grid_e["x_min"], xe_max=self._grid_e["x_max"],
            ye_min=self._grid_e["y_min"], ye_max=self._grid_e["y_max"],
            has_landscape_h=self.rasterize_h,
            v_grid_h=self._grid_h["v_grid"],
            xh_min=self._grid_h["x_min"], xh_max=self._grid_h["x_max"],
            yh_min=self._grid_h["y_min"], yh_max=self._grid_h["y_max"],
            local_step_nm=self.local_step_nm,
            global_step_nm=self.global_step_nm,
            global_move_prob=self.global_move_probability,
            staging_segment_lengths=self._valid_segment_lengths,
            staging_moves_per_step=self.staging_moves_per_step,
            perform_local_sweep=self.perform_local_sweep,
            seed=self.rng_seed,
        )

        return {
            "samples_e": samples_e,
            "samples_h": samples_h,
            "acceptance_local_e": float(acc_local_e),
            "acceptance_local_h": float(acc_local_h),
            "acceptance_staging": float(acc_staging),
            "acceptance_global_joint": float(acc_global),
            "staging_segment_lengths": self._valid_segment_lengths,
            "staging_length_attempts": len_attempts,
            "staging_length_accepted": len_accepted,
            "n_samples": int(samples_e.shape[0]),
        }

