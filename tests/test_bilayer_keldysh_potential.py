from pathlib import Path

import numpy as np

from tmd_pimc.bilayer_keldysh_potential import (
    bilayer_keldysh_table_diagnostics,
    bilayer_keldysh_value_eV,
    build_bilayer_keldysh_table,
    load_bilayer_keldysh_table_npz,
    save_bilayer_keldysh_table_npz,
    screening_length_from_chi2d_nm,
)


def test_chi2d_conversion() -> None:
    assert np.isclose(
        screening_length_from_chi2d_nm(7.13),
        4.479911124019045,
        rtol=0.0,
        atol=1.0e-13,
    )
    assert np.isclose(
        screening_length_from_chi2d_nm(5.56),
        3.4934510307918494,
        rtol=0.0,
        atol=1.0e-13,
    )


def test_material_blk_is_finite_and_monotonic() -> None:
    radii = np.linspace(0.0, 40.0, 2001)
    values = np.asarray(
        bilayer_keldysh_value_eV(
            radii,
            separation_nm=0.60,
            screening_length_layer1_nm=4.479911124019045,
            screening_length_layer2_nm=3.4934510307918494,
            kappa_environment=4.945,
        )
    )
    assert np.all(np.isfinite(values))
    assert values[0] < 0.0
    assert np.all(np.diff(values) >= -1.0e-12)


def test_table_diagnostics_and_roundtrip(tmp_path: Path) -> None:
    table = build_bilayer_keldysh_table(
        separation_nm=0.60,
        screening_length_layer1_nm=4.479911124019045,
        screening_length_layer2_nm=3.4934510307918494,
        kappa_environment=4.945,
        r_max_nm=40.0,
        n_log=180,
        n_linear=360,
    )
    diagnostics = bilayer_keldysh_table_diagnostics(table, n_test=64)
    assert diagnostics["monotonicity_violations"] == 0
    assert diagnostics["max_relative_interpolation_error"] < 1.0e-3
    assert diagnostics["tail_relative_error"] < 3.0e-3
    assert np.isclose(table.screening_length_sum_nm, 7.973362154810894)

    path = tmp_path / "material_blk_table.npz"
    save_bilayer_keldysh_table_npz(path, table)
    loaded = load_bilayer_keldysh_table_npz(path)
    assert np.array_equal(loaded.r_nm, table.r_nm)
    assert np.array_equal(loaded.V_eV, table.V_eV)
    assert loaded.layer1_name == table.layer1_name
    assert loaded.layer2_name == table.layer2_name
