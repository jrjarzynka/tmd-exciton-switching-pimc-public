import sys
sys.path.insert(0, "scratch/moire_pipeline_placeholder")

from moire_pipeline.structure_relaxation import build_structure, save_bundle
from moire_pipeline.generate_potential_map import generate_potential_map

st = build_structure(theta_deg=0.50, box_nm=100.0,
                      bottom_material="MoSe2", top_material="WSe2")
paths = save_bundle(st, "results/structure_wse2_mose2")

payload = generate_potential_map(
    structure_npz=paths["npz"],
    out_npz="results/wse2_mose2/V_grid.npz",
    grid_n=400,
    registry_depth_meV=90.0,
    disable_deformation=True,
)
print("box_to_moire_ratio:", payload["box_to_moire_ratio"])
print("warnings:", payload["warnings"])
