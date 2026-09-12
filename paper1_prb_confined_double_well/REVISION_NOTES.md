# PRB reviewer-hardening revision — confined double well

## Scientific correction

The original finite double-Gaussian potential approaches zero at large radius and therefore does not define a normalizable single-particle canonical ensemble on the unbounded plane. With a linear bias it is also unbounded below in one direction. The revised controlled double-well Hamiltonian is

\[
V(X,Y)=V_{\rm dw}(X,Y)+0.08\,{\rm eV}\left(\frac{R}{25\,{\rm nm}}\right)^8-E_xX,
\qquad R=\sqrt{X^2+Y^2}.
\]

The eighth-power wall is an auxiliary thermodynamic regulator. It is negligible in the two-well region but dominates the linear field asymptotically. All double-well claims in the revised manuscript are based on reruns with this confined Hamiltonian.

## Principal revised results

- Primary relocation observable: bead-averaged diagonal right-basin probability `p_R^(b)`; centroid occupation is now secondary.
- Symmetric P=32 scan, eight chains/field: `p_R^(b)=0.4982 +/- 0.0224` at zero field and a smooth 0.0793 -> 0.9144 crossover over the scanned field interval.
- Independent deterministic finite-temperature diagonalization agrees directly with the bead observable. At `E_x=+0.258519998 meV/nm`, `p_R^(diag)=0.771136`; the independent P=128 PI-QMC audit gives `0.77119 +/- 0.01784`.
- Zero-bias 5--95% path-spanning fraction: `25.18 +/- 0.40%` in the symmetric production scan; independent P=32,64,128 audit gives `24.95 +/- 0.28%`, `23.34 +/- 0.28%`, `23.46 +/- 0.44%`.
- Asymmetric bead-defined crossings for actual zero-bias offsets 1.9917, 3.9835, 5.9752 meV: `-0.1386 +/- 0.0089`, `-0.2837 +/- 0.0207`, `-0.4666 +/- 0.0147 meV/nm`. Origin-constrained slope `0.7491`, descriptive `R^2=0.9876`.
- Regulator-radius audit Rw=20,25,30 nm: zero-field and steep-point occupations and the largest-asymmetry crossing remain statistically compatible.
- Confined-double-well primitive vs HBB energy estimator audit: 0.94 sigma agreement at P=32 and 0.50 sigma at P=80.

## Deterministic regulator audit

For the production wall Rw=25 nm, relative to the formally unconfined double-Gaussian potential represented on the same finite difference box:

- shift of E0: +0.00261 meV
- shift of E1: +0.00410 meV
- shift of the minimum: ~0.00019 meV

The local two-well physics is therefore essentially unchanged while the asymptotic thermodynamics is repaired.

At the representative field, deterministic truncation errors in the diagonal occupation are below 1e-4 across grid, box, and spectral tests, versus a P=128 PI-QMC uncertainty of ~1.8e-2.

## New figure files

The revision replaces the old double-well figures with:

- `fig8_occupation_scurve.pdf`
- `fig13_filmstrip.pdf`
- `fig10_tunneling_paths.pdf`
- `fig12_free_energy_map.pdf`

All other pre-existing figure assets remain unchanged and must remain available beside the revised main TeX when compiling the full manuscript.

## Compilation audit

Both revised TeX files were compiled twice after replacing figure inclusions by neutral boxes so that syntax, labels and cross-references could be checked independently of unavailable unchanged figure assets. No fatal LaTeX errors or unresolved internal references remain. The Supplement has no overfull boxes; the main source retains only a pre-existing ~0.78 pt cosmetic overfull box in an unchanged section.

## One external submission gate

The currently published GitHub/Zenodo release predates these confined-double-well reruns. The revised Data Availability statement therefore does **not** falsely claim that the new artifacts are already in that archive. Before pressing Submit, create a new versioned repository/archive release containing this package and replace the temporary Data Availability wording with the final release/DOI information. No additional scientific rerun is required for that gate.
