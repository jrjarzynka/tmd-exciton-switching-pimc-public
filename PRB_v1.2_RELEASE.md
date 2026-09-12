# PRB reviewer-hardening release: confined double well and bead-primary observable

This release updates the reproducibility package for the PRB manuscript *Finite-temperature centre-of-mass path-integral Monte Carlo of field-biased exciton relocation and delocalization in moiré landscapes*.

## Scientific changes

- Replaces the formally unconfined finite double-Gaussian benchmark by the same two-well landscape plus a weak `r^8` thermodynamic regulator (`0.08 eV`, `R_w=25 nm`, power 8).
- Reruns the symmetric field scan, asymmetric crossing analysis, targeted `P=32,64,128` audit, regulator-radius sensitivity, and primitive/virial energy cross-check.
- Promotes the bead-averaged diagonal right-basin probability to the primary physical relocation observable; centroid occupation remains a secondary path diagnostic.
- Adds deterministic finite-temperature diagonalization directly benchmarking the bead occupation.
- Regenerates all figures depending on the double-well model.

## Key checks

- The regulator shifts the lowest deterministic levels by only a few `10^-3 meV` while repairing the asymptotic thermodynamics.
- At `E_x=+0.258519998 meV/nm`, deterministic `p_R=0.771136` and independent `P=128` PI-QMC gives `0.77119 ± 0.01784`.
- Zero-field 5–95% path-spanning fraction remains about 25%.
- Largest-asymmetry crossing remains stable across bead counts and regulator radii within chain-level uncertainties.
- Primitive and HBB virial energy estimators agree within one standard error on the confined non-harmonic benchmark.

See `paper1_prb_confined_double_well/REVISION_NOTES.md` and `RERUN_MANIFEST.md` for provenance and exact source files.

## Archive continuity

This release is the successor to GitHub release `v1.1`. The existing Zenodo concept DOI for all versions is `10.5281/zenodo.21935778`. A version-specific DOI for `v1.2` should be inserted after Zenodo archives the release.
