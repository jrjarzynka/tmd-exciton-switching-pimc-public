# Atomistic bridge placeholder

Scaffold for the **required technical upgrade** identified in the
manuscript (`Table 2`, row "Lattice-periodic atomistic-grid COM PI-QMC" --
"required technical upgrade") and in Sec. 4.4 ("Required lattice-vector-
periodic implementation"): a commensurate rhombic moire supercell must be
represented by fractional-coordinate wrapping with real lattice vectors,
not independent Cartesian `x mod Lx`/`y mod Ly` wrapping.

## Status: NOT yet connected to `numerics/tmd_pimc`

Nothing in `numerics/tmd_pimc/` imports anything from this directory.
These files are a placeholder pipeline (structure relaxation ->
potential-map generation -> PI-QMC integration hooks) for when this project
moves from analytic `MoirePotential` (current two-body work in
`runners/scans/`) to a real relaxed atomistic moire cell.

## Validation status (2026-08-02)

- `moire_pipeline/generate_potential_map.py::registry_energy` had a bug
  where the `phase` parameter did **not** implement a physical registry
  (stacking-offset) shift: it added a single scalar to only one of the
  three moire G-vector cosine terms, which distorts the pattern shape
  rather than translating it (verified: ~18% residual RMS vs. a true
  rigid shift, even at the best-fit scalar value). This also broke
  `registry_norm_mode="global"`'s hardcoded value range for any nonzero
  `phase`, causing silent clipping. Fixed by adding a proper
  `registry_shift_nm=(dx_nm, dy_nm)` parameter (implemented as a genuine
  per-G_k phase of `-G_k . shift`). The old `phase` parameter now raises
  `ValueError` for any nonzero value instead of silently returning a
  distorted landscape; `phase=0.0` (the only value any config here ever
  used) is unaffected, so `make_placeholder_grid.py` and the existing
  `V_grid.npz` placeholder are **not** impacted by this fix.
- Regression tests now exist for this directory: `conftest.py` +
  `tests/test_registry_shift.py` (10 tests -- coordinate-shift
  equivalence, global-mode range invariance under shift, rejection of the
  legacy `phase` parameter, a half-period-shift sanity check). Run with
  `pytest tests/` from this directory.
- Nothing else in this pipeline (`structure_relaxation.py`,
  `potential.py`, `pimc_integration.py`) has been reviewed/tested to this
  level yet -- see "Known gaps" below.

## Relation to the periodic two-body work already done

`numerics/tmd_pimc/two_body_kernels_periodic_jit.py` already solves the
*wrapping* half of this problem for the analytic case: it derives real-space
primitive lattice vectors `a1, a2` from `MoirePotential`'s reciprocal
vectors and does exact periodic-cell lookup (`bilinear_interpolate_periodic_cell`,
reused from `kernels_jit.py`), validated to `2e-5 eV` max error 25 periods
from the origin. The missing piece for a *real* atomistic device
prediction is not the wrapping math -- it's:

1. **Structure relaxation** (`structure_relaxation.py`): commensurate
   supercell construction with a *validated* atomistic output -- exported
   `.xyz`/`.lammps.data` must be checked for valid line breaks, atom
   records, cell info, species assignments before anything downstream
   trusts them (this was flagged as an open audit item on the COM/TMD_Ez
   side). **Correction (2026-08-02): this file currently only contains
   `build_structure` (fixed-`theta_deg` twisted bilayer on a square
   window) and a clearly-marked-non-physical `toy_relax`. It does NOT yet
   contain `find_commensurate_pair`/`build_commensurate_moire` -- those
   exist in the separate `TMD_Ez`/`relative_coordinates_1_6` project
   trees, not here. An earlier version of this README implied they were
   already present in this file; they still need to be ported over (or
   this file needs to be pointed at that existing builder) before a real
   commensurate `theta ~ 2.0046 deg` cell can be constructed.**
2. **Potential map generation** (`generate_potential_map.py`): calibrate a
   real `V_e(r)`, `V_h(r)` from stacking-dependent band edges, local
   dipoles, and dielectric screening -- replacing the analytic `V0 * sum
   cos(G_i . r)` form with an electronic-structure-derived grid. The
   registry-shift mechanism needed for this (`registry_shift_nm`) is now
   correct -- see "Validation status" above -- but the actual electronic-
   structure calibration (band edges, dipoles, screening) is still
   entirely unimplemented; `registry_energy`'s cosine-sum form is itself
   still just an analytic stand-in, not DFT-derived.
3. **PI-QMC integration** (`pimc_integration.py`): feed the resulting grid
   through the *same* periodic-cell machinery already validated in
   `two_body_kernels_periodic_jit.py`, generalized from the current
   hexagonal analytic lattice vectors to the actual (possibly lower-
   symmetry, strain-distorted) relaxed supercell's lattice vectors. Note:
   `build_pimc_grid_potential` currently catches a bare `except Exception`
   around the `tmd_pimc` import/`CompositePotential` construction and
   silently falls back to a raw `GridPotential2D` on ANY failure, not just
   `ImportError` -- a real bug there (e.g. a bad kwarg) would currently be
   mislabeled as "tmd_pimc not available" and silently produce an
   incomplete potential. Worth narrowing before this pipeline is trusted
   for real output.

## Before treating any of this as production code

- These are placeholders (per the original directory name); read them
  before assuming they do what their filenames suggest.
- The `V_grid.npz` caveat from the COM/TMD_Ez side of the project applies
  here too if this pipeline is ever pointed at that grid: the only
  available grid at time of writing is `theta_deg=0.5`,
  `disable_deformation=True` -- a placeholder registry-only test grid, NOT
  the production WSe2/MoSe2 structure (theta ~ 2.0046 deg, m=17/n=-16,
  ~4902 atoms, with heterostrain). Do not write material-specific results
  into a paper from this grid without regenerating the real one first.

## Suggested next step when this work resumes

1. Port `find_commensurate_pair`/`build_commensurate_moire` from the
   `TMD_Ez`/`relative_coordinates_1_6` trees into (or alongside)
   `structure_relaxation.py` -- see the correction under "Structure
   relaxation" above; the commensurate builder needed for a real
   `theta ~ 2.0046 deg` cell is not in this directory yet.
2. Narrow the bare `except Exception` in `pimc_integration.py`'s
   `build_pimc_grid_potential` to `ImportError`.
3. Start from `two_body_kernels_periodic_jit.py`'s lattice-vector
   derivation (`build_periodic_cell_grid`) and generalize it to accept
   arbitrary `a1, a2` (not just the hexagonal `G = 4*pi/(sqrt(3)*period_nm)`
   form) read from a relaxed structure's actual cell vectors, rather than
   writing a parallel wrapping implementation from scratch here.
