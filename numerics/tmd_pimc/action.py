from dataclasses import dataclass, field
import numpy as np
from .constants import HBAR2_OVER_2M0, KB_EV_PER_K
from .potentials import Potential2D


@dataclass
class RingPolymerAction:
    mass_m0: float
    temperature_K: float
    n_beads: int
    potential: Potential2D

    _beta: float = field(init=False, repr=False)
    _tau: float = field(init=False, repr=False)
    _lambda_x: float = field(init=False, repr=False)
    _kpf: float = field(init=False, repr=False)

    def __post_init__(self):
        self._beta = 1.0 / (KB_EV_PER_K * self.temperature_K)
        self._tau = self._beta / self.n_beads
        self._lambda_x = HBAR2_OVER_2M0 / self.mass_m0
        self._kpf = 1.0 / (4.0 * self._lambda_x * self._tau)

    @staticmethod
    def _norm2(v):
        return float(np.dot(v, v))

    def delta_action_bead_move(self, path, j, r_new):
        P = self.n_beads
        jm, jp = (j - 1) % P, (j + 1) % P
        r_old  = path[j]
        r_prev = path[jm]
        r_next = path[jp]
        old_spring = self._norm2(r_prev - r_old) + self._norm2(r_old - r_next)
        new_spring = self._norm2(r_prev - r_new) + self._norm2(r_new - r_next)
        old_v = self.potential.value(r_old.reshape(1, 2))[0]
        new_v = self.potential.value(r_new.reshape(1, 2))[0]
        return float(self._kpf * (new_spring - old_spring)
                     + self._tau * (new_v - old_v))

    def total_action(self, path):
        diffs  = path - np.roll(path, -1, axis=0)
        spring = self._kpf * np.sum(np.einsum("ij,ij->i", diffs, diffs))
        pot    = self._tau * np.sum(self.potential.value(path))
        return float(spring + pot)
