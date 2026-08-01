"""Staging PIMC sampler for the coupled electron-hole two-body model.

Generalises PIMCSamplerStaging (sampler.py) from one ring polymer to two,
coupled only through the potential (see two_body_action.py). Move set:

  1. local_sweep_e / local_sweep_h
       Single-bead Gaussian proposals on each chain independently -- direct
       analogue of PIMCSamplerStaging.local_sweep.
  2. staging_move_e / staging_move_h
       Brownian-bridge segment updates on each chain independently. The
       proposal already carries the exact free-particle kinetic weight for
       that chain, so (as in the single-body sampler) the Metropolis ratio
       needs only the potential-action difference -- but now that includes
       the interaction term evaluated against the *other* chain's fixed
       slices at the same imaginary-time indices.
  3. global_translation_joint  (NEW relative to the single-body sampler)
       Rigidly translates BOTH path_e and path_h by the same random vector.
       Because r_e - r_h is invariant under this move, the interaction
       term contributes exactly zero to the Metropolis ratio -- only
       Delta V_e + Delta V_h enters. This is the move that lets the bound
       e-h pair explore the moire landscape as a composite object without
       relying on the two internal-mode chains randomly drifting together;
       omitting it would make basin-to-basin transport of the *exciton*
       (as opposed to the *relative coordinate*) extremely slow.

perform_local_sweep=False for both chains reproduces a "staging + joint
global move only" scheme, analogous to how PIMCSamplerStaging is normally
run with perform_local_sweep=True but can be tested without it.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class TwoBodyPIMCSamplerStaging:
    action: object  # TwoBodyRingPolymerAction
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
            raise ValueError("TwoBodyPIMCSamplerStaging requires n_beads >= 3")
        if self.local_step_nm <= 0.0:
            raise ValueError("local_step_nm must be positive")
        if self.global_step_nm <= 0.0:
            raise ValueError("global_step_nm must be positive")
        if not 0.0 <= self.global_move_probability <= 1.0:
            raise ValueError("global_move_probability must lie in [0, 1]")
        if self.staging_moves_per_step <= 0:
            raise ValueError("staging_moves_per_step must be positive")

        lengths = sorted(
            {int(length) for length in self.staging_segment_lengths if 2 <= int(length) < P}
        )
        if not lengths:
            lengths = [P - 1]
        self._valid_segment_lengths = tuple(lengths)

        # Each chain has its own free-particle link variance, since m_e != m_h
        # in general -> lambda_e != lambda_h.
        self._free_link_variance_e_nm2 = float(2.0 * self.action._lambda_e * self.action._tau)
        self._free_link_variance_h_nm2 = float(2.0 * self.action._lambda_h * self.action._tau)

    @property
    def valid_staging_segment_lengths(self) -> tuple[int, ...]:
        return self._valid_segment_lengths

    def initialize_paths(self, center_e=(0.0, 0.0), center_h=(0.0, 0.0), spread_nm=0.1):
        P = self.action.n_beads
        path_e = np.array(center_e, dtype=float) + spread_nm * self.rng.standard_normal((P, 2))
        path_h = np.array(center_h, dtype=float) + spread_nm * self.rng.standard_normal((P, 2))
        return path_e, path_h

    # -- local sweeps --------------------------------------------------------

    def local_sweep_e(self, path_e, path_h):
        P = self.action.n_beads
        accepted = 0
        proposals = self.local_step_nm * self.rng.standard_normal((P, 2))
        for j in range(P):
            r_new = path_e[j] + proposals[j]
            dS = self.action.delta_action_bead_move_e(path_e, path_h, j, r_new)
            if dS < 0.0 or self.rng.random() < np.exp(-dS):
                path_e[j] = r_new
                accepted += 1
        return path_e, accepted / P

    def local_sweep_h(self, path_e, path_h):
        P = self.action.n_beads
        accepted = 0
        proposals = self.local_step_nm * self.rng.standard_normal((P, 2))
        for j in range(P):
            r_new = path_h[j] + proposals[j]
            dS = self.action.delta_action_bead_move_h(path_e, path_h, j, r_new)
            if dS < 0.0 or self.rng.random() < np.exp(-dS):
                path_h[j] = r_new
                accepted += 1
        return path_h, accepted / P

    # -- staging (Brownian-bridge) moves -------------------------------------

    def _staging_proposal(self, path, indices, free_link_variance_nm2):
        """Exact free-particle bridge proposal for the interior of a segment.

        Identical maths to PIMCSamplerStaging.staging_move in sampler.py,
        factored out so it can be reused for either chain with its own
        free_link_variance.
        """
        L = len(indices) - 1
        old_interior = np.asarray(path[indices[1:-1]], dtype=float).copy()
        new_interior = np.empty_like(old_interior)
        previous = np.asarray(path[indices[0]], dtype=float).copy()
        endpoint = np.asarray(path[indices[-1]], dtype=float)

        for interior_offset in range(1, L):
            remaining = L - interior_offset
            denominator = remaining + 1.0
            mean = (remaining * previous + endpoint) / denominator
            variance = free_link_variance_nm2 * remaining / denominator
            proposed_point = mean + np.sqrt(variance) * self.rng.standard_normal(2)
            new_interior[interior_offset - 1] = proposed_point
            previous = proposed_point
        return old_interior, new_interior

    def staging_move_e(self, path_e, path_h, segment_length=None):
        P = int(self.action.n_beads)
        L = int(self.rng.choice(self._valid_segment_lengths)) if segment_length is None else int(segment_length)
        if L < 2 or L >= P:
            raise ValueError(f"segment_length must satisfy 2 <= L < P; got L={L}, P={P}")
        start = int(self.rng.integers(0, P))
        indices = (start + np.arange(L + 1, dtype=np.int64)) % P
        interior_indices = indices[1:-1]

        old_interior, new_interior = self._staging_proposal(
            path_e, indices, self._free_link_variance_e_nm2
        )
        dS = self.action.delta_action_segment_e(
            old_interior, new_interior, path_h, interior_indices
        )
        if dS < 0.0 or self.rng.random() < np.exp(-dS):
            path_e[interior_indices] = new_interior
            return path_e, 1, L
        return path_e, 0, L

    def staging_move_h(self, path_e, path_h, segment_length=None):
        P = int(self.action.n_beads)
        L = int(self.rng.choice(self._valid_segment_lengths)) if segment_length is None else int(segment_length)
        if L < 2 or L >= P:
            raise ValueError(f"segment_length must satisfy 2 <= L < P; got L={L}, P={P}")
        start = int(self.rng.integers(0, P))
        indices = (start + np.arange(L + 1, dtype=np.int64)) % P
        interior_indices = indices[1:-1]

        old_interior, new_interior = self._staging_proposal(
            path_h, indices, self._free_link_variance_h_nm2
        )
        dS = self.action.delta_action_segment_h(
            old_interior, new_interior, path_e, interior_indices
        )
        if dS < 0.0 or self.rng.random() < np.exp(-dS):
            path_h[interior_indices] = new_interior
            return path_h, 1, L
        return path_h, 0, L

    def staging_sweep(self, path_e, path_h):
        outcomes_e, outcomes_h = [], []
        for _ in range(self.staging_moves_per_step):
            path_e, accepted, length = self.staging_move_e(path_e, path_h)
            outcomes_e.append((length, accepted))
            path_h, accepted, length = self.staging_move_h(path_e, path_h)
            outcomes_h.append((length, accepted))
        return path_e, path_h, outcomes_e, outcomes_h

    # -- global moves ---------------------------------------------------------

    def global_translation_joint(self, path_e, path_h):
        """Rigidly translate both chains together (interaction-invariant)."""
        old_S = (
            float(np.sum(self.action.potential_e.value(path_e)))
            + float(np.sum(self.action.potential_h.value(path_h)))
        )
        shift = self.global_step_nm * self.rng.standard_normal(2)
        proposed_e = path_e + shift
        proposed_h = path_h + shift
        new_S = (
            float(np.sum(self.action.potential_e.value(proposed_e)))
            + float(np.sum(self.action.potential_h.value(proposed_h)))
        )
        dS = self.action._tau * (new_S - old_S)
        if dS < 0.0 or self.rng.random() < np.exp(-dS):
            return proposed_e, proposed_h, 1.0
        return path_e, path_h, 0.0

    # -- driver ----------------------------------------------------------------

    @staticmethod
    def _num_samples(n_steps, burn_in, sample_every):
        if n_steps <= burn_in:
            return 0
        return 1 + (n_steps - burn_in - 1) // sample_every

    def run(self, n_steps=10000, burn_in=2500, sample_every=20,
            center_e=(0.0, 0.0), center_h=(0.0, 0.0)):
        path_e, path_h = self.initialize_paths(center_e=center_e, center_h=center_h)
        n_samples = self._num_samples(n_steps, burn_in, sample_every)
        P = self.action.n_beads
        samples_e = np.empty((n_samples, P, 2), dtype=float)
        samples_h = np.empty((n_samples, P, 2), dtype=float)
        sample_idx = 0

        local_acc_e, local_acc_h = [], []
        staging_accepted = 0
        staging_attempts = 0
        global_accepted = 0
        global_attempts = 0

        for step in range(n_steps):
            if self.perform_local_sweep:
                path_e, acc_e = self.local_sweep_e(path_e, path_h)
                path_h, acc_h = self.local_sweep_h(path_e, path_h)
                if step >= burn_in:
                    local_acc_e.append(acc_e)
                    local_acc_h.append(acc_h)

            path_e, path_h, outcomes_e, outcomes_h = self.staging_sweep(path_e, path_h)
            if step >= burn_in:
                for _, accepted in outcomes_e + outcomes_h:
                    staging_attempts += 1
                    staging_accepted += accepted

            if self.rng.random() < self.global_move_probability:
                path_e, path_h, acc_g = self.global_translation_joint(path_e, path_h)
                if step >= burn_in:
                    global_attempts += 1
                    global_accepted += int(acc_g)

            if step >= burn_in and (step - burn_in) % sample_every == 0:
                samples_e[sample_idx] = path_e
                samples_h[sample_idx] = path_h
                sample_idx += 1

        if sample_idx != n_samples:
            samples_e = samples_e[:sample_idx]
            samples_h = samples_h[:sample_idx]

        return {
            "samples_e": samples_e,
            "samples_h": samples_h,
            "acceptance_local_e": float(np.mean(local_acc_e)) if local_acc_e else float("nan"),
            "acceptance_local_h": float(np.mean(local_acc_h)) if local_acc_h else float("nan"),
            "acceptance_staging": (
                float(staging_accepted / staging_attempts) if staging_attempts > 0 else float("nan")
            ),
            "acceptance_global_joint": (
                float(global_accepted / global_attempts) if global_attempts > 0 else float("nan")
            ),
            "global_attempts": int(global_attempts),
            "staging_attempts": int(staging_attempts),
            "n_samples": int(sample_idx),
        }
