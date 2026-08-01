"""Small composable Potential2D helpers, additive to potentials.py.

Kept in a separate module rather than edited into potentials.py so the
already-validated file stays untouched.
"""

from dataclasses import dataclass, field
import numpy as np

from .potentials import Potential2D


@dataclass
class ShiftedPotential(Potential2D):
    """Evaluate an inner potential at r - shift_nm instead of at r.

    Used to give the electron and hole independent moire registries: e.g.
    V_e = MoirePotential(...) (registry pinned at the origin) and
    V_h = ShiftedPotential(MoirePotential(...), shift_nm=(dx, dy)) places
    the hole's registry minima at an offset representing a different
    local stacking (e.g. AB vs BA) relative to the electron's landscape.
    Amplitude/period/symmetry of the inner potential are unchanged; only
    its phase/origin moves.
    """

    inner: Potential2D
    shift_nm: tuple[float, float] = (0.0, 0.0)
    _shift: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._shift = np.asarray(self.shift_nm, dtype=float)

    def value(self, r: np.ndarray) -> np.ndarray:
        return self.inner.value(np.asarray(r, dtype=float) - self._shift)
