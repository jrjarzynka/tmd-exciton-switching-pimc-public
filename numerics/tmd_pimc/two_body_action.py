"""Coupled electron-hole ring-polymer action.

Direct generalisation of ``RingPolymerAction`` (action.py) from a single
relative coordinate with reduced mass mu, to two independently propagating
ring polymers -- one for the electron, one for the hole -- with their own
physical masses m_e, m_h, coupled only through the potential term.

Design note
-----------
The two ring polymers are *kinetically independent*: the electron spring
term only involves electron beads, the hole spring term only involves hole
beads. Coupling enters exclusively through

    V_total(r_e, r_h) = V_e(r_e) + V_h(r_h) + V_int(r_e - r_h)

evaluated slice-by-slice at matched imaginary-time index j. This mirrors
Eq. (22) of the manuscript (H_eh) and is a strict generalisation: setting
V_int = 0 decouples the two polymers into two independent RingPolymerAction
instances; setting V_e = V_h = 0 and using the *relative* coordinate alone
recovers the current v1.7/v1.8a single-body model (up to an unconstrained
centre of mass, which the two-body model now also samples).

KNOWN GAP (flag for the caller): m_e and m_h are not yet available anywhere
in the existing v1.7/v1.8a config pipeline, which only stores a single
`reduced_mass_m0`. mu alone does not determine (m_e, m_h) individually --
callers must supply physically justified values (e.g. from Kormanyos et al.
2015 k.p parameters for the specific layer pair) rather than back-deriving
them from mu.
"""

from dataclasses import dataclass, field
import numpy as np
from .constants import HBAR2_OVER_2M0, KB_EV_PER_K
from .potentials import Potential2D


@dataclass
class TwoBodyRingPolymerAction:
    """Action for two coupled ring polymers (electron and hole).

    Parameters
    ----------
    mass_e_m0, mass_h_m0:
        Physical electron and hole effective masses, in units of m0.
        NOT interchangeable with a single reduced mass -- see module note.
    potential_e, potential_h:
        One-body landscapes for the electron and the hole (e.g. separate
        moire/Stark potentials per layer). Use a zero potential
        (CompositePotential(terms=[])) if a bare interaction-only test is
        wanted, matching the current v1.7/v1.8a relative-coordinate model.
    potential_interaction:
        Screened electron-hole interaction, evaluated at r = r_e - r_h.
        Pass a BilayerKeldyshTablePotential (or wrapped *WallPotential) here
        directly -- its ``value(r)`` signature is unchanged from how it is
        used today on the single relative-coordinate path.
    """

    mass_e_m0: float
    mass_h_m0: float
    temperature_K: float
    n_beads: int
    potential_e: Potential2D
    potential_h: Potential2D
    potential_interaction: Potential2D

    _beta: float = field(init=False, repr=False)
    _tau: float = field(init=False, repr=False)
    _lambda_e: float = field(init=False, repr=False)
    _lambda_h: float = field(init=False, repr=False)
    _kpf_e: float = field(init=False, repr=False)
    _kpf_h: float = field(init=False, repr=False)

    def __post_init__(self):
        if self.mass_e_m0 <= 0.0 or self.mass_h_m0 <= 0.0:
            raise ValueError("mass_e_m0 and mass_h_m0 must be positive")
        self._beta = 1.0 / (KB_EV_PER_K * self.temperature_K)
        self._tau = self._beta / self.n_beads
        self._lambda_e = HBAR2_OVER_2M0 / self.mass_e_m0
        self._lambda_h = HBAR2_OVER_2M0 / self.mass_h_m0
        self._kpf_e = 1.0 / (4.0 * self._lambda_e * self._tau)
        self._kpf_h = 1.0 / (4.0 * self._lambda_h * self._tau)

    @staticmethod
    def _norm2(v):
        return float(np.dot(v, v))

    def _one_slice_interaction(self, r_e_point, r_h_point):
        rel = np.atleast_2d(r_e_point - r_h_point)
        return float(self.potential_interaction.value(rel)[0])

    # -- single-bead Metropolis deltas, one per chain ----------------------

    def delta_action_bead_move_e(self, path_e, path_h, j, r_new):
        """Delta-action for moving electron bead j; hole path fixed."""
        P = self.n_beads
        jm, jp = (j - 1) % P, (j + 1) % P
        r_old = path_e[j]
        r_prev, r_next = path_e[jm], path_e[jp]
        old_spring = self._norm2(r_prev - r_old) + self._norm2(r_old - r_next)
        new_spring = self._norm2(r_prev - r_new) + self._norm2(r_new - r_next)

        r_h_j = path_h[j]
        old_v = float(self.potential_e.value(r_old.reshape(1, 2))[0]) \
            + self._one_slice_interaction(r_old, r_h_j)
        new_v = float(self.potential_e.value(r_new.reshape(1, 2))[0]) \
            + self._one_slice_interaction(r_new, r_h_j)

        return float(self._kpf_e * (new_spring - old_spring)
                     + self._tau * (new_v - old_v))

    def delta_action_bead_move_h(self, path_e, path_h, j, r_new):
        """Delta-action for moving hole bead j; electron path fixed."""
        P = self.n_beads
        jm, jp = (j - 1) % P, (j + 1) % P
        r_old = path_h[j]
        r_prev, r_next = path_h[jm], path_h[jp]
        old_spring = self._norm2(r_prev - r_old) + self._norm2(r_old - r_next)
        new_spring = self._norm2(r_prev - r_new) + self._norm2(r_new - r_next)

        r_e_j = path_e[j]
        old_v = float(self.potential_h.value(r_old.reshape(1, 2))[0]) \
            + self._one_slice_interaction(r_e_j, r_old)
        new_v = float(self.potential_h.value(r_new.reshape(1, 2))[0]) \
            + self._one_slice_interaction(r_e_j, r_new)

        return float(self._kpf_h * (new_spring - old_spring)
                     + self._tau * (new_v - old_v))

    # -- segment (staging) potential-only deltas, one per chain ------------

    def delta_action_segment_e(self, old_segment, new_segment, path_h, indices):
        """tau * sum[V_e + V_int] over a contiguous electron segment.

        ``indices`` are the imaginary-time slice indices of the segment
        (matching path_h rows to couple against). Kinetic part is exact
        by construction of the Brownian-bridge proposal, so only the
        potential difference enters the Metropolis ratio (mirrors
        PIMCSamplerStaging.staging_move in sampler.py).
        """
        r_h_slices = path_h[indices]
        old_v = np.sum(self.potential_e.value(old_segment)) + np.sum(
            self.potential_interaction.value(old_segment - r_h_slices)
        )
        new_v = np.sum(self.potential_e.value(new_segment)) + np.sum(
            self.potential_interaction.value(new_segment - r_h_slices)
        )
        return float(self._tau * (new_v - old_v))

    def delta_action_segment_h(self, old_segment, new_segment, path_e, indices):
        """Mirror of delta_action_segment_e for a hole-chain segment."""
        r_e_slices = path_e[indices]
        old_v = np.sum(self.potential_h.value(old_segment)) + np.sum(
            self.potential_interaction.value(r_e_slices - old_segment)
        )
        new_v = np.sum(self.potential_h.value(new_segment)) + np.sum(
            self.potential_interaction.value(r_e_slices - new_segment)
        )
        return float(self._tau * (new_v - old_v))

    # -- full action, for global/joint moves and diagnostics ----------------

    def total_action(self, path_e, path_h):
        diffs_e = path_e - np.roll(path_e, -1, axis=0)
        diffs_h = path_h - np.roll(path_h, -1, axis=0)
        spring_e = self._kpf_e * np.sum(np.einsum("ij,ij->i", diffs_e, diffs_e))
        spring_h = self._kpf_h * np.sum(np.einsum("ij,ij->i", diffs_h, diffs_h))

        pot_e = np.sum(self.potential_e.value(path_e))
        pot_h = np.sum(self.potential_h.value(path_h))
        pot_int = np.sum(self.potential_interaction.value(path_e - path_h))

        pot = self._tau * (pot_e + pot_h + pot_int)
        return float(spring_e + spring_h + pot)
