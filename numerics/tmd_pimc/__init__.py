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

# Contact density |psi(0)|^2 of the relative coordinate, and its exact
# references. Pure numpy: no numba, no sampler dependency, so it sits with the
# other observables rather than behind the two-body guard below. It consumes
# separations from any source, including the pure-Python sampler.
from .contact import (
    ContactDensityResult,
    contact_density,
    contact_density_gaussian_exact,
    contact_density_exponential_exact,
    relative_radiative_rate,
)

# Two-body (electron-hole) extension. Imported at the end and guarded so
# that importing the base single-body package never requires numba (the
# JIT-backed two_body_*_jit modules) if it isn't installed -- only importing
# tmd_pimc.two_body_* explicitly (or the JIT variants specifically) does.
from .two_body_action import TwoBodyRingPolymerAction
from .two_body_sampler import TwoBodyPIMCSamplerStaging
from .potential_helpers import ShiftedPotential

# Observable layer over the two-body samplers: slice matching, interaction-range
# diagnostics, and registry-resolved contact density. Deliberately OUTSIDE the
# numba guard below -- it only post-processes the (n_configs, P, 2) sample
# arrays and works identically for the pure-Python TwoBodyPIMCSamplerStaging.
# Placing it inside the try/except would silently make these observables
# unavailable on a numba-less install that can still produce the samples.
from .pair_observables import (
    SeparationDiagnostics,
    PairCorrelation,
    pair_separations,
    separation_diagnostics,
    assert_interaction_is_aperiodic,
    contact_density_from_samples,
    contact_density_by_registry,
    pair_correlation,
    variance_decomposition,
    detect_bimodality,
)

try:
    from .sampler_staging_periodic_jit import PIMCSamplerStagingPeriodicJIT
    from .two_body_sampler_jit import TwoBodyPIMCSamplerStagingJIT
    from .two_body_sampler_periodic_jit import (
        TwoBodyPIMCSamplerStagingPeriodicJIT,
        field_coefficient,
    )
except ImportError:
    # numba not installed -- pure-Python TwoBodyPIMCSamplerStaging above
    # still works; JIT variants stay unavailable until numba is installed.
    pass
