"""gsfe_fit.py

Parses pw.x output total energies from a GSFE scan and fits them to the
SAME leading-order three-reciprocal-vector Fourier form used by
registry_energy() in generate_potential_map.py -- but evaluated with the
COMMON MONOLAYER's reciprocal vectors (fast, primitive-cell periodicity)
rather than the small moire G-vectors (slow, moire-period periodicity).

WHY THE SAME FUNCTIONAL FORM APPLIES AT BOTH SCALES
-------------------------------------------------------
registry_energy(X, Y, ...) computes, at each ABSOLUTE position (X,Y) in
the moire supercell, the LOCAL registry offset implied by the twist/
mismatch, and evaluates a three-cosine sum at that offset. The three
moire G-vectors it uses are the natural "slow" wavevectors that convert
absolute position into local registry as you sweep across one moire
period. But the FUNCTIONAL SHAPE (three cosines, C3-symmetric, one
leading harmonic) is a general property of the hexagonal stacking
problem -- the same shape appears if you instead directly compute energy
vs. registry offset within ONE primitive cell (a GSFE/gamma-surface scan)
using the reciprocal vectors of the COMMON (average) monolayer lattice.
Fitting a GSFE(u,v) DFT scan to this form both (a) gives the
`registry_depth_meV` amplitude to plug directly into
generate_potential_map(), and (b) checks whether the real DFT landscape
is well described by a single leading harmonic at all, or needs the
model extended with higher moire harmonics -- registry_energy() currently
assumes the former.

VALIDATION ON SYNTHETIC DATA
-------------------------------
No pw.x is available in this environment, so this module is validated
here by fitting its own exact analytic model back to itself (recovers
the injected depth_eV to numerical precision) and by checking the
residual-based "is a single harmonic enough" diagnostic correctly reports
near-zero residual for pure single-harmonic synthetic data and a nonzero
residual once higher-harmonic content is injected. Real DFT output is
still required before trusting a fitted registry_depth_meV for the paper.
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np

from .gsfe_geometry import avec

RY_TO_EV = 13.605693122994  # CODATA


def parse_pw_total_energy_eV(pw_output_path: str | Path) -> float:
    """Extract the final SCF/relax total energy from a pw.x stdout file
    (the '!    total energy              =  ... Ry' line; for a `relax`
    run, the LAST such line is the converged, relaxed-geometry energy)."""
    text = Path(pw_output_path).read_text()
    matches = re.findall(r"!\s+total energy\s*=\s*(-?\d+\.\d+)\s*Ry", text)
    if not matches:
        raise ValueError(f"No '!    total energy' line found in {pw_output_path}")
    return float(matches[-1]) * RY_TO_EV


def _recip_common(a_common_A: float):
    """Reciprocal vectors of the common (average) monolayer lattice --
    same construction as generate_potential_map.recip(), applied to a
    single shared lattice constant instead of two mismatched ones."""
    a1, a2 = avec(a_common_A)
    A = np.column_stack([a1, a2])
    B = 2.0 * np.pi * np.linalg.inv(A).T
    b1, b2 = B[:, 0], B[:, 1]
    b3 = -(b1 + b2)
    return np.array([b1, b2, b3])


def _raw_model(u: np.ndarray, v: np.ndarray, a_common_A: float) -> np.ndarray:
    a1, a2 = avec(a_common_A)
    delta = np.outer(u, a1) + np.outer(v, a2)  # (N,2), Angstrom
    B = _recip_common(a_common_A)
    raw = np.zeros(len(u))
    for b in B:
        raw += np.cos(delta @ b)
    return raw


def fit_registry_depth(u: np.ndarray, v: np.ndarray, E_eV: np.ndarray, a_common_A: float):
    """Least-squares fit E(u,v) ~= E0 + depth_eV * raw_model(u,v), where
    raw_model is the exact three-cosine shape (analytically bounded in
    [-1.5, 3.0], matching registry_energy's "global" convention).

    Returns: (E0_eV, depth_eV, residual_rms_eV, r_squared)
    """
    raw = _raw_model(u, v, a_common_A)
    A_design = np.column_stack([np.ones_like(raw), raw])
    coeffs, *_ = np.linalg.lstsq(A_design, E_eV, rcond=None)
    E0, depth_eV = coeffs
    predicted = A_design @ coeffs
    residual = E_eV - predicted
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((E_eV - E_eV.mean())**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(E0), float(depth_eV), residual_rms, r_squared


def collate_and_fit_tier1(manifest: dict, output_dir: str | Path):
    """Convenience wrapper: for every cheap rigid-grid job in a
    run_gsfe_scan.py manifest (tier == '1_rigid'), parse the corresponding
    pw.x output file (assumed to live at output_dir / <same stem as
    input, .out>), then fit.
    """
    output_dir = Path(output_dir)
    u_list, v_list, E_list = [], [], []
    for job in manifest["jobs"]:
        if job["tier"] != "1_rigid":
            continue
        in_path = Path(job["path"])
        out_path = output_dir / in_path.with_suffix(".out").name
        if not out_path.exists():
            continue
        E_list.append(parse_pw_total_energy_eV(out_path))
        u_list.append(job["u"])
        v_list.append(job["v"])
    if not E_list:
        raise FileNotFoundError(
            f"No pw.x output files found under {output_dir} matching the tier1_rigid manifest jobs"
        )
    u = np.array(u_list)
    v = np.array(v_list)
    E = np.array(E_list)
    return fit_registry_depth(u, v, E, manifest["a_common_A"])
