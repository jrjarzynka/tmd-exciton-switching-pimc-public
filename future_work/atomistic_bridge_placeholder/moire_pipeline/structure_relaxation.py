# moire_pipeline/structure_relaxation.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
from typing import Tuple, Dict, Any
from .utils import save_npz_with_meta, logger

MASS_AMU = {"Mo": 95.95, "W": 183.84, "S": 32.06, "Se": 78.96}


@dataclass(frozen=True)
class TMDLayer:
    formula: str
    metal: str
    chalcogen: str
    a_A: float
    chalcogen_z_A: float


TMD = {
    "MoSe2": TMDLayer("MoSe2", "Mo", "Se", 3.288, 1.67),
    "WSe2": TMDLayer("WSe2", "W", "Se", 3.282, 1.67),
    "MoS2": TMDLayer("MoS2", "Mo", "S", 3.160, 1.56),
    "WS2": TMDLayer("WS2", "W", "S", 3.153, 1.56),
}


def rot(th: float) -> np.ndarray:
    c = math.cos(th)
    s = math.sin(th)
    return np.array([[c, -s], [s, c]], float)


def avec(a: float) -> Tuple[np.ndarray, np.ndarray]:
    return np.array([a, 0.0]), np.array([0.5 * a, 0.5 * math.sqrt(3) * a])


def monolayer(layer: TMDLayer, box_nm: float, theta_deg: float, z0_A: float, lname: str):
    """Generate atomic positions for a monolayer within a square box (Angstrom units)."""
    a1, a2 = avec(layer.a_A)
    R = rot(math.radians(theta_deg))
    half = 0.5 * box_nm * 10.0
    nmax = int(math.ceil((math.sqrt(2) * half + 10.0) / layer.a_A)) + 4
    # BUGFIX (2026-08-02): chalcogen_top and chalcogen_bottom previously
    # used DIFFERENT in-plane fractional positions ((1/3,2/3) and
    # (2/3,1/3) respectively). This is wrong for 2H-phase TMDs: the
    # defining feature of trigonal PRISMATIC coordination is that the top
    # and bottom chalcogen triangles around each metal atom are ECLIPSED
    # (stacked directly on top of each other, same in-plane position,
    # differing only in z) -- "two equilateral triangles, one directly
    # above the other," not rotated/staggered relative to each other.
    # Verified numerically: (1/3,2/3) alone gives a metal's nearest
    # chalcogen neighbor at an unphysical ~2.00 A with NO threefold
    # degeneracy (2nd/3rd neighbors at different, larger distances) --
    # not a valid trigonal-prismatic motif at all. The correct shared
    # in-plane offset (1/3,1/3) gives exactly three degenerate
    # nearest-neighbor M-X bonds at 2.528 A, matching the experimental
    # Mo-Se bond length (~2.53 A) used elsewhere in this project's
    # DFT-calibration notes.
    basis = [
        (layer.metal, "metal", np.array([0.0, 0.0]), 0.0),
        (layer.chalcogen, "chalcogen_top", (1.0 / 3.0) * a1 + (1.0 / 3.0) * a2, +layer.chalcogen_z_A),
        (layer.chalcogen, "chalcogen_bottom", (1.0 / 3.0) * a1 + (1.0 / 3.0) * a2, -layer.chalcogen_z_A),
    ]
    species = []
    layers = []
    sub = []
    pos = []
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            cell = i * a1 + j * a2
            for el, su, bxy, bz in basis:
                xy = R @ (cell + bxy)
                if abs(xy[0]) <= half and abs(xy[1]) <= half:
                    species.append(str(el))
                    layers.append(str(lname))
                    sub.append(str(su))
                    pos.append([xy[0], xy[1], z0_A + bz])
    species = np.asarray(species, dtype=str)
    layers = np.asarray(layers, dtype=str)
    sub = np.asarray(sub, dtype=str)
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must be shape (N,3) in Angstrom")
    return species, layers, sub, pos


def build_structure(
    theta_deg: float = 0.50,
    box_nm: float = 25.0,
    bottom_material: str = "MoSe2",
    top_material: str = "WSe2",
    interlayer_midplane_A: float = 6.5,
) -> Dict[str, Any]:
    bottom = TMD[bottom_material]
    top = TMD[top_material]
    sb, lb, ub, pb = monolayer(bottom, box_nm, 0.0, -0.5 * interlayer_midplane_A, "bottom")
    st, lt, ut, pt = monolayer(top, box_nm, theta_deg, +0.5 * interlayer_midplane_A, "top")
    species = np.r_[sb, st]
    layer = np.r_[lb, lt]
    sub = np.r_[ub, ut]
    pos = np.vstack([pb, pt])
    assert len(species) == pos.shape[0]
    return {
        "species": species,
        "layer": layer,
        "sublattice": sub,
        "positions_initial_A": pos.copy(),
        "positions_relaxed_A": pos.copy(),
        "theta_deg": float(theta_deg),
        "box_nm": float(box_nm),
        "interlayer_midplane_A": float(interlayer_midplane_A),
        "bottom_material": np.asarray(bottom_material),
        "top_material": np.asarray(top_material),
        "a_bottom_A": float(bottom.a_A),
        "a_top_A": float(top.a_A),
        "relaxation_status": np.asarray("unrelaxed"),
        "relaxation_engine": np.asarray("none"),
    }


def toy_relax(structure: Dict[str, Any], amplitude_A: float = 0.03, seed: int | None = None) -> Dict[str, Any]:
    """Apply deterministic smooth displacement pattern for testing (not physical)."""
    out = dict(structure)
    pos = out["positions_relaxed_A"].copy()
    rng_offset = 0.0 if seed is None else float(seed % 1000) * 2.0 * math.pi / 1000.0
    G = _moire_G(float(out["a_bottom_A"]), float(out["a_top_A"]), float(out["theta_deg"]))
    phase = (pos[:, :2] @ G[0]) + rng_offset
    disp = amplitude_A * np.c_[np.sin(phase), np.cos(phase), np.zeros_like(phase)]
    top_mask = out["layer"] == "top"
    bottom_mask = out["layer"] == "bottom"
    pos[top_mask] += disp[top_mask]
    pos[bottom_mask] -= 0.3 * disp[bottom_mask]
    out["positions_relaxed_A"] = pos
    out["relaxation_status"] = np.asarray("toy_relaxed_not_physical")
    out["relaxation_engine"] = np.asarray("toy_internal")
    return out


def _moire_G(a_bottom: float, a_top: float, theta_deg: float):
    # local helper to avoid circular import
    from math import radians, cos, sin, sqrt

    def rot_local(th):
        c = cos(th)
        s = sin(th)
        return np.array([[c, -s], [s, c]], float)

    def avec_local(a):
        return np.array([a, 0.0]), np.array([0.5 * a, 0.5 * math.sqrt(3) * a])

    def recip_local(a):
        a1, a2 = avec_local(a)
        A = np.column_stack([a1, a2])
        B = 2.0 * math.pi * np.linalg.inv(A).T
        b1, b2 = B[:, 0], B[:, 1]
        return b1, b2, -(b1 + b2)

    R = rot_local(math.radians(theta_deg))
    return np.array([R @ bt - bb for bt, bb in zip(recip_local(a_top), recip_local(a_bottom))])


def write_xyz(path: str | Path, species, pos, comment: str = ""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    species = np.asarray(species, dtype=str)
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must be shape (N,3)")
    with path.open("w") as f:
        f.write(f"{len(species)}\n{comment}\n")
        for s, r in zip(species, pos):
            f.write(f"{s} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f}\n")


def write_lammps_data(path: str | Path, st: Dict[str, Any]):
    path = Path(path)
    sp = np.asarray(st["species"], dtype=str)
    la = np.asarray(st["layer"], dtype=str)
    su = np.asarray(st["sublattice"], dtype=str)
    pos = np.asarray(st["positions_initial_A"], dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions_initial_A must be shape (N,3)")
    L = float(st["box_nm"]) * 10.0  # Angstrom
    Lz = 100.0
    keys = []
    for k in (f"{s}_{l}_{u}" for s, l, u in zip(sp, la, su)):
        if k not in keys:
            keys.append(k)
    tid = {k: i + 1 for i, k in enumerate(keys)}
    with path.open("w") as f:
        f.write("LAMMPS data file via moire_pipeline\n\n")
        f.write(f"{len(sp)} atoms\n{len(keys)} atom types\n\n")
        f.write("# units: positions in Angstrom, masses in amu\n")
        f.write(f"{-L/2:.10f} {L/2:.10f} xlo xhi\n")
        f.write(f"{-L/2:.10f} {L/2:.10f} ylo yhi\n")
        f.write(f"{-Lz/2:.10f} {Lz/2:.10f} zlo zhi\n\n")
        f.write("Masses\n\n")
        for k in keys:
            element = k.split("_")[0]
            mass = MASS_AMU.get(element)
            if mass is None:
                raise KeyError(f"Unknown element mass for {element}")
            f.write(f"{tid[k]} {mass:.8f} # {k}\n")
        f.write("\nAtoms # atomic\n\n")
        for i, (s, l, u, r) in enumerate(zip(sp, la, su, pos), 1):
            k = f"{s}_{l}_{u}"
            f.write(f"{i} {tid[k]} {r[0]:.10f} {r[1]:.10f} {r[2]:.10f} # {k}\n")


def save_bundle(st: Dict[str, Any], outdir: str | Path) -> Dict[str, str]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = dict(st)
    out_npz = outdir / "structure_relaxed_or_initial.npz"
    meta = {
        "theta_deg": float(st["theta_deg"]),
        "box_nm": float(st["box_nm"]),
        "a_bottom_A": float(st["a_bottom_A"]),
        "a_top_A": float(st["a_top_A"]),
        "relaxation_status": str(st.get("relaxation_status", "unknown")),
    }
    save_npz_with_meta(out_npz, payload, meta)
    write_xyz(outdir / "structure_initial.xyz", st["species"], st["positions_initial_A"], "initial")
    write_xyz(outdir / "structure_relaxed_or_initial.xyz", st["species"], st["positions_relaxed_A"], str(st["relaxation_status"]))
    write_lammps_data(outdir / "structure_unrelaxed.lammps.data", st)
    # template for LAMMPS relax (user must fill force field)
    (outdir / "in.relax.template").write_text(
        "# LAMMPS relaxation template\nunits metal\natom_style atomic\nread_data structure_unrelaxed.lammps.data\n# TODO: set pair_style and pair_coeff\nneighbor 2.0 bin\nneigh_modify every 1 delay 0 check yes\nthermo 100\nmin_style cg\nminimize 1.0e-10 1.0e-12 10000 100000\nwrite_data structure_relaxed.lammps.data\nwrite_dump all xyz structure_relaxed.xyz\n"
    )
    logger.info("Saved structure bundle to %s", outdir)
    return {
        "npz": str(out_npz),
        "xyz": str(outdir / "structure_relaxed_or_initial.xyz"),
        "lammps_data": str(outdir / "structure_unrelaxed.lammps.data"),
        "lammps_template": str(outdir / "in.relax.template"),
    }

