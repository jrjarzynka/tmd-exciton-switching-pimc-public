from dataclasses import dataclass
from typing import Sequence
import numpy as np
from .kernels_jit import (
    run_pimc_core_jit,
    run_pimc_core_jit_periodic_cell,
    run_pimc_core_staging_harmonic_jit,
)


@dataclass
class PIMCSampler:
    action: object
    local_step_nm: float = 0.20
    global_step_nm: float = 1.00
    global_move_probability: float = 0.20
    rng_seed: int = 1234

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)

    def initialize_path(self, center=(0.0, 0.0), spread_nm=0.1):
        return (np.array(center, dtype=float)
                + spread_nm * self.rng.standard_normal((self.action.n_beads, 2)))

    def local_sweep(self, path):
        P         = self.action.n_beads
        accepted  = 0
        proposals = self.local_step_nm * self.rng.standard_normal((P, 2))
        for j in range(P):
            r_new = path[j] + proposals[j]
            dS    = self.action.delta_action_bead_move(path, j, r_new)
            if dS < 0.0 or self.rng.random() < np.exp(-dS):
                path[j] = r_new
                accepted += 1
        return path, accepted / P

    def global_translation_move(self, path):
        old_S    = self.action.total_action(path)
        proposed = path + self.global_step_nm * self.rng.standard_normal(2)
        dS       = self.action.total_action(proposed) - old_S
        if dS < 0.0 or self.rng.random() < np.exp(-dS):
            return proposed, 1.0
        return path, 0.0

    @staticmethod
    def _num_samples(n_steps, burn_in, sample_every):
        if n_steps <= burn_in:
            return 0
        return 1 + (n_steps - burn_in - 1) // sample_every

    def run(self, n_steps=10000, burn_in=2500, sample_every=20, center=(0.0, 0.0)):
        path       = self.initialize_path(center=center)
        n_samples  = self._num_samples(n_steps, burn_in, sample_every)
        samples    = np.empty((n_samples, self.action.n_beads, 2), dtype=float)
        sample_idx = 0
        # FIX: acceptance stats are now accumulated only for step >= burn_in,
        # matching run_pimc_core_jit's semantics (kernels_jit.py, patch v0.9.1).
        # Previously local_acc/global_acc included burn-in steps here, so
        # acceptance_local/acceptance_global from this sampler were NOT
        # comparable to the JIT sampler's numbers (e.g. when validating
        # PIMCSamplerJIT physics against this reference implementation).
        local_acc  = []
        global_acc = []

        for step in range(n_steps):
            path, acc_l = self.local_sweep(path)
            if step >= burn_in:
                local_acc.append(acc_l)
            if self.rng.random() < self.global_move_probability:
                path, acc_g = self.global_translation_move(path)
                if step >= burn_in:
                    global_acc.append(acc_g)
            if step >= burn_in and (step - burn_in) % sample_every == 0:
                samples[sample_idx] = path
                sample_idx += 1

        if sample_idx != n_samples:
            samples = samples[:sample_idx]

        # FIX (bonus): unified dict return so both samplers work with the
        # same single_run() in run_moire_switching_v0_8.py.
        # Original returned a (samples, stats_dict) tuple — incompatible with
        # res["samples"] / res["acceptance_local"] in the run script.
        return {
            "samples":           samples,
            "acceptance_local":  float(np.mean(local_acc)) if local_acc else float("nan"),
            "acceptance_global": float(np.mean(global_acc)) if global_acc else float("nan"),
            "global_attempts":   int(len(global_acc)),
            "n_samples":         int(sample_idx),
        }


class PIMCSamplerJIT:
    """
    Zoptymalizowana wersja samplera PIMC z kompilacją JIT (v1.5).
    Ewaluuje dowolny złożony potencjał na gęstej siatce 2D przed symulacją,
    zapewniając zerową alokację i maksymalne wykorzystanie pamięci podręcznej CPU.
    """
    def __init__(self, action, local_step_nm=0.20, global_step_nm=1.00,
                 global_move_probability=0.20, rng_seed=1234,
                 grid_size=600, grid_range_nm=40.0,
                 boundary_mode="finite_square",
                 periodic_cell_vectors_nm=None,
                 periodic_cell_origin_nm=None,
                 global_disp_vectors_nm=None,
                 directed_move_frac=0.0,
                 directed_jitter_nm=0.5):
        self.action = action
        self.local_step_nm = local_step_nm
        self.global_step_nm = global_step_nm
        self.global_move_probability = global_move_probability

        # Kierunkowe propozycje ruchu globalnego (opcjonalne; domyslnie
        # wylaczone, wiec zachowanie jest bit-w-bit zgodne z poprzednia
        # wersja). Zbior MUSI byc domkniety na negacje -- patrz docstring
        # run_pimc_core_jit; helper moire_hop_vectors_nm to gwarantuje.
        if global_disp_vectors_nm is None:
            self.global_disp_vectors_nm = np.empty((0, 2), dtype=np.float64)
        else:
            vecs = np.asarray(global_disp_vectors_nm, dtype=np.float64)
            if vecs.ndim != 2 or vecs.shape[1] != 2:
                raise ValueError("global_disp_vectors_nm must have shape (n_vec, 2)")
            for v in vecs:
                if not any(np.allclose(-v, w, atol=1e-9) for w in vecs):
                    raise ValueError(
                        "global_disp_vectors_nm must be closed under negation "
                        "(for every v it must also contain -v), otherwise the "
                        "Monte Carlo proposal is not symmetric and the plain "
                        "Metropolis acceptance criterion is invalid."
                    )
            self.global_disp_vectors_nm = vecs
        self.directed_move_frac = float(directed_move_frac)
        if not (0.0 <= self.directed_move_frac <= 1.0):
            raise ValueError("directed_move_frac must lie in [0, 1]")
        self.directed_jitter_nm = float(directed_jitter_nm)
        self.rng_seed = rng_seed
        self.grid_size = int(grid_size)
        self.grid_range_nm = float(grid_range_nm)
        self.boundary_mode = str(boundary_mode)
        self.rng = np.random.default_rng(self.rng_seed)

        if self.boundary_mode not in ("finite_square", "periodic_cell"):
            raise ValueError(
                "boundary_mode must be 'finite_square' or 'periodic_cell'"
            )
        if self.grid_size < 4:
            raise ValueError("grid_size must be >= 4")
        if self.grid_range_nm <= 0.0:
            raise ValueError("grid_range_nm must be positive")

        self.periodic_cell_vectors_nm = None
        self.periodic_cell_origin_nm = None
        self._periodic_Ainv = None
        if self.boundary_mode == "periodic_cell":
            if periodic_cell_vectors_nm is None or periodic_cell_origin_nm is None:
                raise ValueError(
                    "periodic_cell mode requires periodic_cell_vectors_nm and "
                    "periodic_cell_origin_nm"
                )
            vectors = np.asarray(periodic_cell_vectors_nm, dtype=np.float64)
            origin = np.asarray(periodic_cell_origin_nm, dtype=np.float64)
            if vectors.shape != (2, 2):
                raise ValueError("periodic_cell_vectors_nm must have shape (2, 2)")
            if origin.shape != (2,):
                raise ValueError("periodic_cell_origin_nm must have shape (2,)")
            A = np.column_stack([vectors[0], vectors[1]])
            if abs(float(np.linalg.det(A))) < 1e-14:
                raise ValueError("periodic cell vectors are degenerate")
            self.periodic_cell_vectors_nm = vectors
            self.periodic_cell_origin_nm = origin
            self._periodic_Ainv = np.linalg.inv(A)

    def initialize_path(self, center=(0.0, 0.0), spread_nm=0.1):
        return (np.array(center, dtype=np.float64)
                + spread_nm * self.rng.standard_normal((self.action.n_beads, 2)))

    def _build_potential_grid(self):
        """Build either the legacy finite square or one periodic lattice cell."""
        if self.boundary_mode == "finite_square":
            x_min, x_max = -self.grid_range_nm, self.grid_range_nm
            y_min, y_max = -self.grid_range_nm, self.grid_range_nm
            x_ticks = np.linspace(x_min, x_max, self.grid_size)
            y_ticks = np.linspace(y_min, y_max, self.grid_size)
            X, Y = np.meshgrid(x_ticks, y_ticks, indexing="ij")
            pts = np.stack([X.ravel(), Y.ravel()], axis=1)
            V_flat = self.action.potential.value(pts)
            v_grid = V_flat.reshape(self.grid_size, self.grid_size).astype(np.float64)
            return {
                "v_grid": v_grid,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            }

        # Endpoint-excluded fractional grid. The periodic interpolation kernel
        # wraps the neighbour of the last sample back to index zero.
        uv = np.arange(self.grid_size, dtype=np.float64) / self.grid_size
        U, V = np.meshgrid(uv, uv, indexing="ij")
        a1 = self.periodic_cell_vectors_nm[0]
        a2 = self.periodic_cell_vectors_nm[1]
        origin = self.periodic_cell_origin_nm
        pts = (
            origin[None, :]
            + U.reshape(-1, 1) * a1[None, :]
            + V.reshape(-1, 1) * a2[None, :]
        )
        V_flat = self.action.potential.value(pts)
        v_grid = V_flat.reshape(self.grid_size, self.grid_size).astype(np.float64)
        return {"v_grid": v_grid}

    def run(self, n_steps=10000, burn_in=2500, sample_every=20, center=(0.0, 0.0)):
        path = self.initialize_path(center=center)
        P    = self.action.n_beads

        kpf = self.action._kpf
        tau = self.action._tau

        grid = self._build_potential_grid()

        if self.boundary_mode == "finite_square":
            samples, acc_local, acc_global = run_pimc_core_jit(
                n_steps=n_steps,
                burn_in=burn_in,
                sample_every=sample_every,
                p_beads=P,
                path=path,
                v_grid=grid["v_grid"],
                x_min=grid["x_min"],
                x_max=grid["x_max"],
                y_min=grid["y_min"],
                y_max=grid["y_max"],
                kpf=kpf,
                tau=tau,
                local_step_nm=self.local_step_nm,
                global_step_nm=self.global_step_nm,
                global_move_prob=self.global_move_probability,
                seed=self.rng_seed,
                global_disp_vectors=self.global_disp_vectors_nm,
                directed_move_frac=self.directed_move_frac,
                directed_jitter_nm=self.directed_jitter_nm,
            )
        else:
            Ainv = self._periodic_Ainv
            origin = self.periodic_cell_origin_nm
            samples, acc_local, acc_global = run_pimc_core_jit_periodic_cell(
                n_steps=n_steps,
                burn_in=burn_in,
                sample_every=sample_every,
                p_beads=P,
                path=path,
                v_grid=grid["v_grid"],
                origin_x=float(origin[0]),
                origin_y=float(origin[1]),
                ainv00=float(Ainv[0, 0]),
                ainv01=float(Ainv[0, 1]),
                ainv10=float(Ainv[1, 0]),
                ainv11=float(Ainv[1, 1]),
                kpf=kpf,
                tau=tau,
                local_step_nm=self.local_step_nm,
                global_step_nm=self.global_step_nm,
                global_move_prob=self.global_move_probability,
                seed=self.rng_seed,
            )

        return {
            "samples": samples,
            "acceptance_local": acc_local,
            "acceptance_global": acc_global,
            "jit_boundary_mode": self.boundary_mode,
        }


@dataclass
class PIMCSamplerStaging:
    """Python PIMC sampler with exact free-particle staging proposals.

    A staging move selects a contiguous segment of ``L`` imaginary-time links,
    fixes its endpoint beads, and redraws the ``L-1`` interior beads from the
    exact free-particle Brownian-bridge distribution.  Since the proposal
    already contains the complete kinetic spring weight, the Metropolis ratio
    contains only the change in potential action.

    ``PIMCSampler`` and ``PIMCSamplerJIT`` above are intentionally unchanged;
    this is a separate validation target.
    """

    action: object
    local_step_nm: float = 0.20
    global_step_nm: float = 1.00
    global_move_probability: float = 0.20
    rng_seed: int = 1234
    staging_segment_lengths: Sequence[int] = (4, 8, 16, 32, 64, 128, 256)
    staging_moves_per_step: int = 2
    perform_local_sweep: bool = True

    def __post_init__(self):
        self.rng = np.random.default_rng(self.rng_seed)
        P = int(self.action.n_beads)
        if P < 3:
            raise ValueError("PIMCSamplerStaging requires n_beads >= 3")
        if self.local_step_nm <= 0.0:
            raise ValueError("local_step_nm must be positive")
        if self.global_step_nm <= 0.0:
            raise ValueError("global_step_nm must be positive")
        if not 0.0 <= self.global_move_probability <= 1.0:
            raise ValueError("global_move_probability must lie in [0, 1]")
        if self.staging_moves_per_step <= 0:
            raise ValueError("staging_moves_per_step must be positive")

        lengths = sorted(
            {
                int(length)
                for length in self.staging_segment_lengths
                if 2 <= int(length) < P
            }
        )
        if not lengths:
            lengths = [P - 1]
        self._valid_segment_lengths = tuple(lengths)

        # Link density: exp[-|dr|^2/(4 lambda tau)], therefore each Cartesian
        # free-particle link has variance 2 lambda tau.
        self._free_link_variance_nm2 = float(
            2.0 * self.action._lambda_x * self.action._tau
        )

    @property
    def valid_staging_segment_lengths(self) -> tuple[int, ...]:
        return self._valid_segment_lengths

    def initialize_path(self, center=(0.0, 0.0), spread_nm=0.1):
        return (
            np.array(center, dtype=float)
            + spread_nm * self.rng.standard_normal((self.action.n_beads, 2))
        )

    def local_sweep(self, path):
        P = self.action.n_beads
        accepted = 0
        proposals = self.local_step_nm * self.rng.standard_normal((P, 2))
        for j in range(P):
            r_new = path[j] + proposals[j]
            dS = self.action.delta_action_bead_move(path, j, r_new)
            if dS < 0.0 or self.rng.random() < np.exp(-dS):
                path[j] = r_new
                accepted += 1
        return path, accepted / P

    def global_translation_move(self, path):
        old_S = self.action.total_action(path)
        proposed = path + self.global_step_nm * self.rng.standard_normal(2)
        dS = self.action.total_action(proposed) - old_S
        if dS < 0.0 or self.rng.random() < np.exp(-dS):
            return proposed, 1.0
        return path, 0.0

    def staging_move(self, path, segment_length=None):
        """Attempt one Brownian-bridge segment update."""
        P = int(self.action.n_beads)
        if segment_length is None:
            L = int(self.rng.choice(self._valid_segment_lengths))
        else:
            L = int(segment_length)
            if L < 2 or L >= P:
                raise ValueError(
                    f"segment_length must satisfy 2 <= L < P; got L={L}, P={P}"
                )

        start = int(self.rng.integers(0, P))
        indices = (start + np.arange(L + 1, dtype=np.int64)) % P
        interior_indices = indices[1:-1]

        old_interior = np.asarray(path[interior_indices], dtype=float).copy()
        new_interior = np.empty_like(old_interior)
        previous = np.asarray(path[indices[0]], dtype=float).copy()
        endpoint = np.asarray(path[indices[-1]], dtype=float)

        # Conditional free-particle bridge.  For the next point, with R links
        # remaining after it, variance = 2 lambda tau * R/(R+1).
        for interior_offset in range(1, L):
            remaining = L - interior_offset
            denominator = remaining + 1.0
            mean = (remaining * previous + endpoint) / denominator
            variance = self._free_link_variance_nm2 * remaining / denominator
            proposed_point = mean + np.sqrt(variance) * self.rng.standard_normal(2)
            new_interior[interior_offset - 1] = proposed_point
            previous = proposed_point

        old_V = float(np.sum(self.action.potential.value(old_interior)))
        new_V = float(np.sum(self.action.potential.value(new_interior)))
        dS_potential = self.action._tau * (new_V - old_V)

        if dS_potential < 0.0 or self.rng.random() < np.exp(-dS_potential):
            path[interior_indices] = new_interior
            return path, 1, L
        return path, 0, L

    def staging_sweep(self, path):
        outcomes = []
        for _ in range(self.staging_moves_per_step):
            path, accepted, length = self.staging_move(path)
            outcomes.append((length, accepted))
        return path, outcomes

    @staticmethod
    def _num_samples(n_steps, burn_in, sample_every):
        if n_steps <= burn_in:
            return 0
        return 1 + (n_steps - burn_in - 1) // sample_every

    def run(self, n_steps=10000, burn_in=2500, sample_every=20, center=(0.0, 0.0)):
        path = self.initialize_path(center=center)
        n_samples = self._num_samples(n_steps, burn_in, sample_every)
        samples = np.empty((n_samples, self.action.n_beads, 2), dtype=float)
        sample_idx = 0

        local_accepted = 0.0
        local_attempts = 0
        staging_accepted = 0
        staging_attempts = 0
        global_accepted = 0
        global_attempts = 0
        length_attempts = {length: 0 for length in self._valid_segment_lengths}
        length_accepted = {length: 0 for length in self._valid_segment_lengths}

        for step in range(n_steps):
            if self.perform_local_sweep:
                path, acc_l = self.local_sweep(path)
                if step >= burn_in:
                    local_accepted += acc_l * self.action.n_beads
                    local_attempts += self.action.n_beads

            path, staging_outcomes = self.staging_sweep(path)
            if step >= burn_in:
                staging_attempts += len(staging_outcomes)
                for length, accepted in staging_outcomes:
                    staging_accepted += accepted
                    length_attempts[length] += 1
                    length_accepted[length] += accepted

            if self.rng.random() < self.global_move_probability:
                path, acc_g = self.global_translation_move(path)
                if step >= burn_in:
                    global_attempts += 1
                    global_accepted += int(acc_g)

            if step >= burn_in and (step - burn_in) % sample_every == 0:
                samples[sample_idx] = path
                sample_idx += 1

        if sample_idx != n_samples:
            samples = samples[:sample_idx]

        return {
            "samples": samples,
            "acceptance_local": (
                float(local_accepted / local_attempts)
                if local_attempts > 0
                else float("nan")
            ),
            "acceptance_staging": (
                float(staging_accepted / staging_attempts)
                if staging_attempts > 0
                else float("nan")
            ),
            "acceptance_global": (
                float(global_accepted / global_attempts)
                if global_attempts > 0
                else float("nan")
            ),
            "global_attempts": int(global_attempts),
            "staging_attempts": int(staging_attempts),
            "staging_segment_lengths": np.asarray(
                self._valid_segment_lengths, dtype=np.int64
            ),
            "staging_length_attempts": np.asarray(
                [length_attempts[L] for L in self._valid_segment_lengths],
                dtype=np.int64,
            ),
            "staging_length_accepted": np.asarray(
                [length_accepted[L] for L in self._valid_segment_lengths],
                dtype=np.int64,
            ),
            "n_samples": int(sample_idx),
        }


class PIMCSamplerStagingJIT:
    """JIT-compiled counterpart of PIMCSamplerStaging.

    HARMONIC POTENTIAL ONLY — see run_pimc_core_staging_harmonic_jit in
    kernels_jit.py for why this deliberately does not use grid interpolation.
    Not a drop-in replacement for PIMCSamplerJIT on the moire grid.

    ATTENTION: reads the spring constant from
    ``self.action.potential.k_eV_per_nm2``. If HarmonicPotential in
    potentials.py exposes the spring constant under a different attribute
    name, update ``_read_k_spring`` below to match — this is the one thing
    not verified against your actual potentials.py in this pass.
    """

    def __init__(self, action, local_step_nm=0.20, global_step_nm=1.00,
                 global_move_probability=0.20, rng_seed=1234,
                 staging_segment_lengths=(4, 8, 16, 32, 64, 128, 256),
                 staging_moves_per_step=2, perform_local_sweep=True):
        self.action = action
        self.local_step_nm = local_step_nm
        self.global_step_nm = global_step_nm
        self.global_move_probability = global_move_probability
        self.rng_seed = rng_seed
        self.staging_moves_per_step = int(staging_moves_per_step)
        self.perform_local_sweep = bool(perform_local_sweep)

        P = int(action.n_beads)
        if P < 3:
            raise ValueError("PIMCSamplerStagingJIT requires n_beads >= 3")

        lengths = sorted(
            {int(L) for L in staging_segment_lengths if 2 <= int(L) < P}
        )
        if not lengths:
            lengths = [P - 1]
        self._valid_segment_lengths = np.asarray(lengths, dtype=np.int64)

        self.rng = np.random.default_rng(self.rng_seed)

    def _read_k_spring(self):
        """Extract the harmonic spring constant from the action's potential.

        Tries a couple of plausible attribute names so a mismatch fails
        loudly with a clear message rather than silently reading garbage.
        """
        pot = self.action.potential
        for attr in ("k_eV_per_nm2", "k", "spring_k_eV_per_nm2"):
            if hasattr(pot, attr):
                return float(getattr(pot, attr))
        raise AttributeError(
            "Could not find the harmonic spring constant on "
            f"{type(pot).__name__}. Checked: k_eV_per_nm2, k, "
            "spring_k_eV_per_nm2. Update PIMCSamplerStagingJIT._read_k_spring "
            "to match the actual attribute name in potentials.py."
        )

    def initialize_path(self, center=(0.0, 0.0), spread_nm=0.1):
        return (np.array(center, dtype=np.float64)
                + spread_nm * self.rng.standard_normal((self.action.n_beads, 2)))

    def run(self, n_steps=10000, burn_in=2500, sample_every=20, center=(0.0, 0.0)):
        path = self.initialize_path(center=center)
        k_spring = self._read_k_spring()

        samples, acc_local, acc_staging, acc_global, len_attempts, len_accepted = (
            run_pimc_core_staging_harmonic_jit(
                n_steps=n_steps,
                burn_in=burn_in,
                sample_every=sample_every,
                p_beads=self.action.n_beads,
                path=path,
                k_spring=k_spring,
                kpf=self.action._kpf,
                tau=self.action._tau,
                lam=self.action._lambda_x,
                local_step_nm=self.local_step_nm,
                global_step_nm=self.global_step_nm,
                global_move_prob=self.global_move_probability,
                staging_segment_lengths=self._valid_segment_lengths,
                staging_moves_per_step=self.staging_moves_per_step,
                perform_local_sweep=self.perform_local_sweep,
                seed=self.rng_seed,
            )
        )

        return {
            "samples":                 samples,
            "acceptance_local":        float(acc_local),
            "acceptance_staging":      float(acc_staging),
            "acceptance_global":       float(acc_global),
            "staging_segment_lengths": self._valid_segment_lengths,
            "staging_length_attempts": len_attempts,
            "staging_length_accepted": len_accepted,
            "n_samples":               int(samples.shape[0]),
        }
