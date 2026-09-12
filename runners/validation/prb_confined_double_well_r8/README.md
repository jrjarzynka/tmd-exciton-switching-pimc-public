# PRB confined double-well r^8 rerun

This directory contains the reproducibility runners for the reviewer-hardening rerun of Paper 1.

The authoritative production scripts use the repository's tested `tmd_pimc.PIMCSamplerStaging` implementation directly. The Hamiltonian is the two-Gaussian double well plus the auxiliary radial regulator `0.08 eV (r/25 nm)^8` and the linear lateral energy-gradient term. The primary relocation observable is the bead-averaged right-basin occupation.

`fast_staging.py` is retained only as an optional campaign-specific Numba accelerator. It is not the source of truth for the reported reviewer-hardening results. `tests/test_prb_confined_double_well_r8.py` checks its analytic potential against the generic `CompositePotential` and validates the added analytic gradients by finite differences.

The older untracked/local `run_paper1_targeted_p_convergence.py` campaign used the formally unconfined Gaussian double well and is historical only; it must not be used to reproduce the revised confined-double-well claims.

Compact reference outputs, deterministic audits, revised manuscript sources and regenerated figures are under `paper1_prb_confined_double_well/`. Representative full-path NPZ archives are distributed in the Zenodo release rather than tracked in Git.
