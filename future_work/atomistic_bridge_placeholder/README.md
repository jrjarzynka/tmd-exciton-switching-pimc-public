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

## Relation to the periodic two-body work already done

`numerics/tmd_pimc/two_body_kernels_periodic_jit.py` already solves the
*wrapping* half of this problem for the analytic case: it derives real-space
primitive lattice vectors `a1, a2` from `MoirePotential`'s reciprocal
vectors and does exact periodic-cell lookup (`bilinear_interpolate_periodic_cell`,
reused from `kernels_jit.py`), validated to `2e-5 eV` max error 25 periods
from the origin. The missing piece for a *real* atomistic device
prediction is not the wrapping math -- it's:

1. **Structure relaxation** (`structure_relaxation.py`): commensurate
   supercell construction (`find_commensurate_pair`, `build_commensurate_moire`
   per the project's existing ASE-based builder, per prior context) with a
   *validated* atomistic output -- exported `.xyz`/`.lammps.data` must be
   checked for valid line breaks, atom records, cell info, species
   assignments before anything downstream trusts them (this was flagged as
   an open audit item on the COM/TMD_Ez side).
2. **Potential map generation** (`generate_potential_map.py`): calibrate a
   real `V_e(r)`, `V_h(r)` from stacking-dependent band edges, local
   dipoles, and dielectric screening -- replacing the analytic `V0 * sum
   cos(G_i . r)` form with an electronic-structure-derived grid.
3. **PI-QMC integration** (`pimc_integration.py`): feed the resulting grid
   through the *same* periodic-cell machinery already validated in
   `two_body_kernels_periodic_jit.py`, generalized from the current
   hexagonal analytic lattice vectors to the actual (possibly lower-
   symmetry, strain-distorted) relaxed supercell's lattice vectors.

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

Start from `two_body_kernels_periodic_jit.py`'s lattice-vector derivation
(`build_periodic_cell_grid`) and generalize it to accept arbitrary
`a1, a2` (not just the hexagonal `G = 4*pi/(sqrt(3)*period_nm)` form) read
from a relaxed structure's actual cell vectors, rather than writing a
parallel wrapping implementation from scratch here.
