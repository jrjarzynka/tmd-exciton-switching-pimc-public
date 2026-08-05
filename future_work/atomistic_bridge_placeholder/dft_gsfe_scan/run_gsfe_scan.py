"""run_gsfe_scan.py

Driver: writes the full GSFE scan plan to disk, split by COST not just by
vdW treatment -- interlayer-z relaxation (calculation='relax', BFGS)
costs roughly (1 + n_bfgs_steps) full SCF cycles per point, ~10-11x a
single rigid SCF in practice (measured: point_000 took 11 SCF cycles /
10 BFGS steps). At that cost, relaxing all 49 grid points would take
~35 hours on a poorly-threaded 4-core run -- not practical for a first
pass. So the DEFAULT plan is:

  - tier1_rigid  (cheap, full 2D grid, 49 points): Grimme-D3, NO z
    relaxation (single SCF each) -- gives the GSFE *shape* quickly.
  - tier1_relaxed_subset (3 points: AA, bridge, AB): Grimme-D3, WITH z
    relaxation -- lets you compute a rigid->relaxed correction without
    paying the ~11x relax cost on all 49 grid points.
  - tier2_validation (3 points: AA, bridge, AB): nonlocal vdW-DF (rVV10
    by default), WITH z relaxation -- cross-check against the D3-relaxed
    subset above.

IMPORTANT FINDING (2026-08-05): comparing tier1_rigid's full-grid fit
(depth=43.66 meV) against the tier1_relaxed_subset/tier2 3-point fits
(D3-relaxed: 113.79 meV; rVV10-relaxed: 121.81 meV) shows interlayer-z
relaxation changes the corrugation amplitude by nearly 3x -- far more
than the choice of vdW functional (D3 vs rVV10 agree to ~7% once BOTH
are relaxed). The rigid-grid number is very likely NOT a trustworthy
final registry_depth_meV; a full 49-point RELAXED grid is needed. This
was previously judged too expensive (~35h), but with the OMP_NUM_THREADS
threading bug fixed (each relaxed point now takes ~9-10 min instead of
15+ hours -- see the project session log), a full 49-point relaxed grid
now costs only ~8h wall time on an 8-core machine, which is tractable as
a single overnight cloud run. Use --full-relaxed-grid to generate it.

Usage (from future_work/atomistic_bridge_placeholder/, i.e. one level
above this package, so that the relative imports resolve as part of the
dft_gsfe_scan package):

    python3 -m dft_gsfe_scan.run_gsfe_scan                     # default 3-tier plan
    python3 -m dft_gsfe_scan.run_gsfe_scan --full-relaxed-grid  # + full 49-pt relaxed D3 grid

CRITICAL REMINDER when running on a cloud VM: set OMP_NUM_THREADS=1
before mpirun, and confirm `echo $TMUX` is non-empty before starting a
long-running loop -- both bit us during this project's first cloud run
(see session log).
"""
import argparse
from pathlib import Path
import json

from .gsfe_geometry import build_gsfe_cell, registry_shift_grid, high_symmetry_path
from .gsfe_qe_input import write_pw_input

OUT_ROOT = Path("gsfe_scan_inputs")
BOTTOM, TOP = "MoSe2", "WSe2"
GRID_N = 7                  # 7x7 = 49 points
VALIDATION_VDW = "rvv10"    # or 'vdw-df2-b86r'

# The 3 core high-symmetry stacking points, used for both the D3-relaxed
# calibration subset and the rVV10 validation subset.
HIGH_SYM_POINTS = {"AA": (0.0, 0.0), "bridge": (1.0 / 3.0, 1.0 / 6.0), "AB": (1.0 / 3.0, 2.0 / 3.0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-relaxed-grid", action="store_true", default=False,
        help="Also write the full 49-point D3 grid WITH z relaxation "
             "(tier1_relaxed_full/). ~8h wall time on 8 cores with "
             "OMP_NUM_THREADS=1 set correctly; see module docstring.",
    )
    args = parser.parse_args()

    cell = build_gsfe_cell(BOTTOM, TOP)
    manifest = {"bottom": BOTTOM, "top": TOP, "a_common_A": cell.a_common_A, "jobs": []}

    # Tier 1a: cheap full-grid D3 scan, RIGID (no z relaxation) -- gives
    # the GSFE shape fast, but see module docstring: the amplitude from
    # this tier alone is likely NOT trustworthy as a final answer.
    grid = registry_shift_grid(GRID_N)
    for i, (u, v) in enumerate(grid):
        rel = f"tier1_rigid/point_{i:03d}.in"
        write_pw_input(OUT_ROOT / rel, cell, (float(u), float(v)),
                        vdw_mode="grimme-d3", relax_interlayer_z=False)
        manifest["jobs"].append({"tier": "1_rigid", "u": float(u), "v": float(v),
                                  "vdw_mode": "grimme-d3", "relaxed": False, "path": rel})

    # Tier 1b: small D3 subset WITH z relaxation, at the 3 high-symmetry
    # points -- lets you sanity-check a rigid->relaxed correction cheaply.
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

    # Tier 3 (opt-in): the full 49-point grid, D3, WITH z relaxation --
    # the actual trustworthy answer per the 2026-08-05 finding above.
    if args.full_relaxed_grid:
        for i, (u, v) in enumerate(grid):
            rel = f"tier1_relaxed_full/point_{i:03d}.in"
            write_pw_input(OUT_ROOT / rel, cell, (float(u), float(v)),
                            vdw_mode="grimme-d3", relax_interlayer_z=True)
            manifest["jobs"].append({"tier": "1_relaxed_full", "u": float(u), "v": float(v),
                                      "vdw_mode": "grimme-d3", "relaxed": True, "path": rel})

    manifest_path = OUT_ROOT / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    n_rigid = sum(1 for j in manifest["jobs"] if j["tier"] == "1_rigid")
    n_relaxed = sum(1 for j in manifest["jobs"] if j["relaxed"])
    print(f"Wrote {len(manifest['jobs'])} pw.x input files under {OUT_ROOT}/")
    print(f"  {n_rigid} cheap rigid points (tier1_rigid/, single SCF each)")
    print(f"  {n_relaxed} expensive relaxed points (~9-10 min each with "
          f"OMP_NUM_THREADS=1 set correctly on 8 cores)")
    if args.full_relaxed_grid:
        print(f"  Full relaxed grid included (tier1_relaxed_full/): ~49 x 9-10 min "
              f"~= 8h wall time on 8 cores.")
    print(f"Manifest: {manifest_path}")
    print()
    print("Before submitting: edit PSEUDO_FILENAMES in gsfe_qe_input.py to match your")
    print("actual installed pseudopotentials, and place them under gsfe_scan_inputs/pseudo/")
    print("(or adjust pseudo_dir in the generated inputs).")
    print()
    print("REMINDER for cloud VMs: export OMP_NUM_THREADS=1 before mpirun, and confirm")
    print("`echo $TMUX` is non-empty before starting the loop -- both caused real")
    print("problems on the first cloud run of this project (session log 2026-08-05).")


if __name__ == "__main__":
    main()

