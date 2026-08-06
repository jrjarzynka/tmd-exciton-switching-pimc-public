from .action import RingPolymerAction
from .sampler import PIMCSampler, PIMCSamplerStaging, PIMCSamplerJIT
from .potentials import (
    HarmonicPotential,
    HarmonicEnvelopePotential,
    SoftWallBoxPotential,
    GaussianWellPotential,
    DoubleGaussianWellPotential,
    ExternalFieldPotential,
    CompositePotential,
    GridPotential2D,
    MoirePotential,
    moire_hop_vectors_nm,
    StrainPiezoelectricPotential,
    SoftCoulombPotential,
)
from .observables import (
    centroids,
    all_beads,
    r2_mean_pimc,
    r2_mean_centroid,
    r2_spread_pimc,
    r_rms_bead,
    r_rms_centroid,
    r_rms_spread,
    r2_time_series,
    mean_position,
    switching_contrast,
)
from .analytic import (
    harmonic_r2_analytic,
    harmonic_r2_classical,
    harmonic_r2_zeropoint,
    harmonic_hbar_omega,
    harmonic_r2_primitive_finite_P,
)
from .constants import HBAR2_OVER_2M0, KB_EV_PER_K

# Two-body (electron-hole) extension. Imported at the end and guarded so
# that importing the base single-body package never requires numba (the
# JIT-backed two_body_*_jit modules) if it isn't installed -- only importing
# tmd_pimc.two_body_* explicitly (or the JIT variants specifically) does.
from .two_body_action import TwoBodyRingPolymerAction
from .two_body_sampler import TwoBodyPIMCSamplerStaging
from .potential_helpers import ShiftedPotential

try:
    from .two_body_sampler_jit import TwoBodyPIMCSamplerStagingJIT
    from .two_body_sampler_periodic_jit import (
        TwoBodyPIMCSamplerStagingPeriodicJIT,
        field_coefficient,
    )
except ImportError:
    # numba not installed -- pure-Python TwoBodyPIMCSamplerStaging above
    # still works; JIT variants stay unavailable until numba is installed.
    pass
