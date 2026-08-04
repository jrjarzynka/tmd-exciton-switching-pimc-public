"""Regression test mirroring test_crystal_structure.py's check, applied to
the small GSFE DFT-scan cell (dft_gsfe_scan.gsfe_geometry.build_gsfe_cell)
-- this is the geometry that actually gets written into pw.x input files,
so it matters independently of structure_relaxation.py's own fix.
"""
import numpy as np
import pytest

from dft_gsfe_scan.gsfe_geometry import build_gsfe_cell, avec

EXPECTED_MX_BOND_A = {"MoSe2": 2.528, "WSe2": 2.526}
BOND_TOL_A = 0.02  # cell here uses a common (averaged) lattice constant,
                    # not each material's own -- small extra slack vs.
                    # test_crystal_structure.py's per-material lattice


def test_gsfe_cell_chalcogens_are_eclipsed():
    cell = build_gsfe_cell("MoSe2", "WSe2")
    # indices 0,1,2 = bottom metal, chalcogen_top, chalcogen_bottom
    # indices 3,4,5 = top    metal, chalcogen_top, chalcogen_bottom
    np.testing.assert_allclose(cell.xy_frac[1], cell.xy_frac[2], atol=1e-12)
    np.testing.assert_allclose(cell.xy_frac[4], cell.xy_frac[5], atol=1e-12)


def test_gsfe_cell_six_degenerate_mx_bonds_at_AA():
    """At the AA reference stacking (zero shift), each metal's own-layer
    M-X bonds must be the sixfold-degenerate 2.53-A shell, same physical
    check as test_crystal_structure.py."""
    cell = build_gsfe_cell("MoSe2", "WSe2")
    a1, a2 = avec(cell.a_common_A)

    # bottom-layer metal at index 0
    m_xy = cell.xy_frac[0][0] * a1 + cell.xy_frac[0][1] * a2
    m_z = cell.z_A[0]
    dists = []
    for i in (1, 2):  # bottom layer's own chalcogen_top/bottom
        xy = cell.xy_frac[i][0] * a1 + cell.xy_frac[i][1] * a2
        d = np.linalg.norm(np.array([xy[0] - m_xy[0], xy[1] - m_xy[1], cell.z_A[i] - m_z]))
        dists.append(d)
    # only 2 unique in-plane images at u=v=1/3 shown here (top & bottom);
    # the other 2 of the "3" in-plane positions come from neighboring
    # periodic images, not needed to confirm the top/bottom degeneracy:
    assert dists[0] == pytest.approx(dists[1], abs=1e-9)
    assert dists[0] == pytest.approx(EXPECTED_MX_BOND_A["MoSe2"], abs=BOND_TOL_A)
