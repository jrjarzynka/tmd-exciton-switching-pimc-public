# Rerun manifest

Production double-well regulator: `Vwall=0.08 eV`, `Rw=25 nm`, power `8`.

Core summary files:

- `rerun_data/symmetric_scan_summary.csv` — 9-field symmetric P=32 scan.
- `rerun_data/asymmetry_crossings.csv` — primary asymmetric crossing results and fit inputs.
- `rerun_data/asymmetry_scan_summary.csv` — field-resolved asymmetric occupations.
- `rerun_data/p_audit_summary.csv` — P=32,64,128 occupation/path/continuous-observable audit.
- `rerun_data/p_crossing_refinement.csv` — P-dependent largest-asymmetry crossing refinement.
- `rerun_data/wall_sensitivity_summary.csv` and `wall_sensitivity_crossing.csv` — Rw=20,25,30 nm audit.
- `rerun_data/energy_confined_summary.csv` — primitive/HBB comparison.
- `audits/deterministic_wall_audit.csv` — local spectrum and regulator perturbation.
- `audits/deterministic_occupation_convergence.csv` — grid/box/eigenstate convergence of p_R^(diag).

Representative full paths for all nine symmetric fields are the `symmetric_repr_field*.npz` files and are retained to reproduce the revised filmstrip/path/free-energy figures.
