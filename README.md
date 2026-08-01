# Numerical-grid centre-of-mass Path-Integral Monte Carlo for moiré exciton relocation

Code and validation scripts accompanying:

> J. R. Jarzynka, *Validated numerical-grid centre-of-mass Path-Integral Monte Carlo
> for field-driven exciton-centroid relocation in moiré-scale landscapes*,
> submitted to Computational Materials Science.

This repository implements and validates a Path-Integral Quantum Monte Carlo (PI-QMC)
engine for the centre-of-mass (COM) dynamics of an exciton in a moiré-scale potential
landscape, including a staging (Brownian-bridge) sampler, JIT-compiled kernels for
numerical-grid landscapes, and a battery of validation tests against exact or
cross-checked references.

## Installation

```bash
git clone https://github.com/jrjarzynka/tmd-exciton-switching-pimc.git
cd tmd-exciton-switching-pimc
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `tmd_pimc` package (source in `numerics/tmd_pimc/`)
in editable mode, so changes to the source are picked up without reinstalling.

## Project structure

* `numerics/tmd_pimc/` — core package: ring-polymer action (`action.py`), samplers
  (`sampler.py`, local/staging/JIT variants), potentials (`potentials.py`,
  `potential_helpers.py`), observables (`observables.py`), the primitive/virial
  energy estimators (`energy_estimators.py`), exact benchmark references
  (`analytic.py`), and the two-body electron–hole extension
  (`two_body_action.py`, `two_body_sampler*.py`) used in a separate, forthcoming
  manuscript.
* `runners/validation/` — scripts reproducing the validation sections of the paper
  (see table below).
* `runners/scans/`, `runners/utils/` — production scan drivers and support scripts,
  primarily for the two-body extension.
* `validation_tests/` — earlier, self-contained validation scripts (Validation I–III
  and the autocorrelation/staging test of Sec. 3.2), each with a short docstring
  describing exactly what it checks.
* `tests/` — unit tests (`pytest`) for the screened-interaction potentials and the
  radial solver used in the two-body extension.
* `configs/` — JSON configuration templates for production scans. `configs/two_body/legacy/`
  contains two deliberately mislabelled configs (`*_COMPROMISED_*`, `*_SUPERSEDED_*`)
  kept as a documented record of a diagnosed and fixed sampling artifact, not for reuse.
* `future_work/atomistic_bridge_placeholder/` — scaffold for the planned atomistic
  (DFT/relaxation-derived) landscape extension. **Explicitly out of scope for this
  submission**: PI-QMC's numerical-grid interface can already consume any properly
  formatted `V_grid.npz` (see `runners/validation/run_atomistic_grid_temp_field_sweep_v1_0.py`),
  but no relaxed, material-specific structure has yet been generated or validated. This
  is the identified next step of the project (Sec. "Relevance" of the manuscript),
  not a claim made by the current submission.

## Reproducing the validation results

| Paper section | Script(s) | Notes |
|---|---|---|
| Validation I: harmonic benchmark | `validation_tests/test1_single_well.py` | PIMC vs. exact `harmonic_r2_analytic` (`P→∞`) and `harmonic_r2_primitive_finite_P` (exact at the *same* finite `P`) |
| Validation I: bead-count convergence | `validation_tests/test2_P_convergence.py` | `T=5`K scan confirming `P=80` is on a converged plateau |
| Sec. 3.2: autocorrelation / staging | `validation_tests/test5_IAT_scaling.py`, `test5_IAT_scaling_v2.py` | `z_local` vs. `z_staging` power-law fit |
| Validation II: field-driven double-well relocation | `validation_tests/test3_double_well_v3.py` | Symmetric double well, occupation crossover at `Ex=0` |
| Validation II: tunnelling / delocalization signature | `validation_tests/test3b_tunneling_snapshot.py`, `test3c_filmstrip.py` | Ring-polymer snapshots and the 9-point field filmstrip |
| Validation III: periodicity fix | `validation_tests/test4_periodicity_artifact.py` | Reconstructs the pre-fix bilinear-interpolation artifact from the fixed code, for direct before/after comparison |
| Validation III: cryogenic registry-grid stability (Table 5) | `runners/validation/run_atomistic_grid_temp_field_sweep_v1_0.py` (runner), `run_lowT_convergence_protocol_v1_1.py` (orchestrator across `global_step_nm`), `analyze_atomistic_grid_pimc_v1_0.py` (per-run analysis), `compare_lowT_convergence_v1_0.py` (produces the Table 5 relative-range summary) | Also: `analyze_V_grid_landscape_v1_0.py` and `check_moire_geometry_v1_1.py` audit the input `V_grid.npz` itself (grid metadata, minima, dominant wavelength). **Note:** the `V_grid.npz` shipped with the exploratory runs in this validation is the `theta=0.5°`, `disable_deformation=True` placeholder grid described in Sec. 6, not a relaxed, material-specific structure — see `future_work/` below. |
| Validation IV: primitive vs. virial energy estimator | `runners/validation/run_energy_validation_harmonic.py`, `run_energy_validation_doublewell.py` | Uses `tmd_pimc.energy_estimators` |
| Validation V: interpolation grid-resolution convergence | `runners/validation/run_grid_resolution_convergence.py` | Seed-averaged, reports both histogram-based and histogram-free diagnostics |
| Validation VI: Trotter convergence at additional spring constants | `runners/validation/run_trotter_convergence_second_k.py` | Bracketing `k` values around the production `k=0.010 eV/nm²` |

Additional (not currently referenced in the published manuscript, kept as
available diagnostics): `validation_tests/test6_stability_map.py`
(`A_eff^S(T,Ex)`) and `test7_chi_E.py` (seed-to-seed centroid spread `χ_E`).

Each `runners/validation/run_*.py` script accepts `--help` for its full parameter
list and writes a CSV to `results/` (git-ignored; not tracked in this repository).

## Two-body extension

`numerics/tmd_pimc/two_body_*.py`, `runners/scans/*two_body*`, and
`configs/two_body/` implement and drive a separate, coupled electron–hole
extension of this framework, the subject of a forthcoming, currently unpublished
manuscript. It is included here because it shares the validated single-body
engine, but its production results are not part of the Computational Materials
Science submission this repository accompanies.

## License

MIT — see `LICENSE`.
