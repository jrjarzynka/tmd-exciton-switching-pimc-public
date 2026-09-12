# Confined double-well PRB revision

This directory documents the reviewer-hardening reruns and deterministic audits for the PRB manuscript

> *Finite-temperature centre-of-mass path-integral Monte Carlo of field-biased exciton relocation and delocalization in moiré landscapes*

The revision replaces the formally unconfined finite double-Gaussian benchmark by

\[
V(X,Y)=V_{\rm dw}(X,Y)+0.08\,{\rm eV}\left(\frac{\sqrt{X^2+Y^2}}{25\,{\rm nm}}\right)^8-E_xX.
\]

The eighth-power term is an auxiliary thermodynamic regulator. It is negligible in the two-well region but makes the canonical equilibrium problem normalizable and the linearly biased Hamiltonian bounded below. The primary relocation observable in the revised manuscript is the bead-averaged diagonal right-basin probability; centroid quantities are retained only as path diagnostics.

The full rerun package (code, compact CSV outputs, regenerated figures, and representative path archives) is distributed with the versioned Zenodo release. The GitHub branch contains the core estimator patch and release metadata; compact source/results are available in the archived release asset.

## Core numerical results

- Zero-field bead occupation at `P=32`: `0.4982 ± 0.0224`.
- Zero-field 5–95% path-spanning fraction: `25.18 ± 0.40%`.
- At `E_x=+0.258519998 meV/nm`, deterministic diagonalization gives `p_R=0.771136`; the independent `P=128` PI-QMC audit gives `0.77119 ± 0.01784`.
- Largest-asymmetry bead crossing is stable across the tested bead counts and regulator radii.
- Primitive and HBB virial energy estimators agree within one standard error on the confined non-harmonic benchmark.

The unchanged primitive-cell production campaign retains its earlier provenance; this release changes only the controlled confined-double-well sector and associated manuscript/validation artifacts.
