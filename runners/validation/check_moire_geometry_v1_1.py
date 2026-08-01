#!/usr/bin/env python3
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Check structure_npz/V_grid_npz moire geometry metadata.")
    p.add_argument("--structure_npz")
    p.add_argument("--potential_npz")
    a = p.parse_args()

    if not a.structure_npz and not a.potential_npz:
        raise SystemExit("Provide --structure_npz and/or --potential_npz")

    def scalar(z, key):
        v = z[key]
        try:
            return float(v)
        except Exception:
            return v.item() if getattr(v, 'shape', None) == () else v

    if a.structure_npz:
        z = np.load(a.structure_npz, allow_pickle=True)
        print("STRUCTURE")
        for k in ["theta_deg", "box_nm", "a_bottom_A", "a_top_A", "bottom_material", "top_material", "relaxation_status", "relaxation_engine"]:
            if k in z.files:
                print(f"  {k:28s}: {z[k]}")

    if a.potential_npz:
        z = np.load(a.potential_npz, allow_pickle=True)
        print("V_GRID")
        for k in ["theta_deg", "box_nm", "grid_n", "estimated_moire_period_nm", "box_to_moire_ratio", "registry_depth_meV", "registry_norm_mode", "disable_deformation"]:
            if k in z.files:
                print(f"  {k:28s}: {z[k]}")
        if "x_nm" in z.files and "y_nm" in z.files and "V_eV" in z.files:
            x, y, V = z["x_nm"], z["y_nm"], z["V_eV"]
            print(f"  grid_shape                  : {V.shape}")
            print(f"  x_bounds_nm                 : {x.min():.6g} to {x.max():.6g}")
            print(f"  y_bounds_nm                 : {y.min():.6g} to {y.max():.6g}")
            print(f"  dx_dy_nm                    : {x[1]-x[0]:.6g}, {y[1]-y[0]:.6g}")
            print(f"  V_range_meV                 : {np.nanmin(V)*1000:.6g} to {np.nanmax(V)*1000:.6g}")
        if "warnings" in z.files and len(z["warnings"]):
            print("  warnings:")
            for w in z["warnings"]:
                print(f"    - {w}")


if __name__ == "__main__":
    main()
