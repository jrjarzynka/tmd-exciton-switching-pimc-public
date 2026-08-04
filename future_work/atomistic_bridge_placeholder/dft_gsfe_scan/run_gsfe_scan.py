"""run_gsfe_scan.py

Driver: writes the full GSFE scan plan to disk, split by COST not just by
vdW treatment -- interlayer-z relaxation (calculation='relax', BFGS)
costs roughly (1 + n_bfgs_steps) full SCF cycles per point, ~10-11x a
single rigid SCF in practice (measured: point_000 took 11 SCF cycles /
10 BFGS steps, ~43 min wall time on a 4-core run). At that cost, relaxing
all 49 grid points would take ~35 hours -- not practical for a first
pass. So:

  - tier1_rigid  (cheap, full 2D grid, 49 points): Grimme-D3, NO z
    relaxation (single SCF each) -- gives the GSFE *shape* quickly.
  - tier1_relaxed_subset (3 points: AA, bridge, AB): Grimme-D3, WITH z
    relaxation -- lets you compute a rigid->relaxed correction (e.g. a
    constant offset, or check the correction is registry-independent
    enough to apply uniformly) without relaxing all 49 points.
  - tier2_validation (3 points: AA, bridge, AB): nonlocal vdW-DF (rVV10
    by default), WITH z relaxation -- cross-check against the D3-relaxed
    subset above, at the same 3 points, rather than the full 19-point
    path (which would cost as much as the entire rest of the scan
    combined at this per-point runtime).

Usage (from future_work/atomistic_bridge_placeholder/, i.e. one level
above this package, so that the relative imports resolve as part of the
dft_gsfe_scan package):

    python3 -m dft_gsfe_scan.run_gsfe_scan
"""
from pathlib import Path
import json

from .gsfe_geometry import build_gsfe_cell, registry_shift_grid, high_symmetry_path
from .gsfe_qe_input import write_pw_input

OUT_ROOT = Path("gsfe_scan_inputs")
BOTTOM, TOP = "MoSe2", "WSe2"
GRID_N = 7                  # 7x7 = 49 points, cheap rigid tier
VALIDATION_VDW = "rvv10"    # or 'vdw-df2-b86r'

# The 3 core high-symmetry stacking points, used for both the D3-relaxed
# calibration subset and the rVV10 validation subset.
HIGH_SYM_POINTS = {"AA": (0.0, 0.0), "bridge": (1.0 / 3.0, 1.0 / 6.0), "AB": (1.0 / 3.0, 2.0 / 3.0)}


def main():
    cell = build_gsfe_cell(BOTTOM, TOP)
    manifest = {"bottom": BOTTOM, "top": TOP, "a_common_A": cell.a_common_A, "jobs": []}

    # Tier 1a: cheap full-grid D3 scan, RIGID (no z relaxation) -- gives
    # the GSFE shape fast. Single SCF per point.
    grid = registry_shift_grid(GRID_N)
    for i, (u, v) in enumerate(grid):
        rel = f"tier1_rigid/point_{i:03d}.in"
        write_pw_input(OUT_ROOT / rel, cell, (float(u), float(v)),
                        vdw_mode="grimme-d3", relax_interlayer_z=False)
        manifest["jobs"].append({"tier": "1_rigid", "u": float(u), "v": float(v),
                                  "vdw_mode": "grimme-d3", "relaxed": False, "path": rel})

    # Tier 1b: small D3 subset WITH z relaxation, at the 3 high-symmetry
    # points -- lets you calibrate a rigid->relaxed correction without
    # paying the ~11x relax cost on all 49 grid points.
    for lab, (u, v) in HIGH_SYM_POINTS.items():
        rel = f"tier1_relaxed_subset/{lab}.in"
        write_pw_input(OUT_ROOT / rel, cell, (u, v),
                        vdw_mode="grimme-d3", relax_interlayer_z=True)
        manifest["jobs"].append({"tier": "1_relaxed_subset", "u": u, "v": v,
                                  "vdw_mode": "grimme-d3", "relaxed": True,
                                  "label": lab, "path": rel})

    # Tier 2: nonlocal vdW-DF validation, WITH z relaxation, at the SAME 3
    # high-symmetry points (not the full path -- see module docstring).
    for lab, (u, v) in HIGH_SYM_POINTS.items():
        rel = f"tier2_{VALIDATION_VDW}/{lab}.in"
        write_pw_input(OUT_ROOT / rel, cell, (u, v),
                        vdw_mode=VALIDATION_VDW, relax_interlayer_z=True)
        manifest["jobs"].append({"tier": "2_validation", "u": u, "v": v,
                                  "vdw_mode": VALIDATION_VDW, "relaxed": True,
                                  "label": lab, "path": rel})

    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    n_rigid = sum(1 for j in manifest["jobs"] if j["tier"] == "1_rigid")
    n_relaxed = sum(1 for j in manifest["jobs"] if j["relaxed"])
    print(f"Wrote {len(manifest['jobs'])} pw.x input files under {OUT_ROOT}/")
    print(f"  {n_rigid} cheap rigid points (tier1_rigid/, single SCF each)")
    print(f"  {n_relaxed} expensive relaxed points (tier1_relaxed_subset/ + tier2_*/, "
          f"~11x cost each based on point_000's measured 11 SCF cycles)")
    print(f"Manifest: {manifest_path}")
    print()
    print("Before submitting: edit PSEUDO_FILENAMES in gsfe_qe_input.py to match your")
    print("actual installed pseudopotentials, and place them under gsfe_scan_inputs/pseudo/")
    print("(or adjust pseudo_dir in the generated inputs).")


if __name__ == "__main__":
    main()

