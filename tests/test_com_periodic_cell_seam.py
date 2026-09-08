"""Regression tests for the single-body COM periodic-cell interpolator.

The cell and numerical tolerance mirror the deterministic GRID-PBC G0
validation.  In particular, the tests exercise the floating-point case where
a slightly negative fractional coordinate becomes exactly 1.0 after wrapping.
"""

import numpy as np
import pytest

from tmd_pimc.kernels_jit import bilinear_interpolate_periodic_cell
from tmd_pimc.potentials import GridPotential2D


ENERGY_TOL_EV = 1.0e-12
GRID_SIZE = 400
LATTICE_VECTORS_NM = np.array(
    [
        [7.390626585097831, 36.09621110343181],
        [-27.564922503389045, 24.44857592429495],
    ],
    dtype=np.float64,
)
ORIGIN_NM = np.zeros(2, dtype=np.float64)


@pytest.fixture(scope="module")
def periodic_grid():
    axis = np.arange(GRID_SIZE, dtype=np.float64) / GRID_SIZE
    u, v = np.meshgrid(axis, axis, indexing="ij")
    values = 0.020 * (
        np.cos(2.0 * np.pi * u)
        + 0.37 * np.sin(2.0 * np.pi * v)
        + 0.23 * np.cos(2.0 * np.pi * (u + v))
    )
    potential = GridPotential2D(
        x_nm=axis,
        y_nm=axis,
        V_eV=values,
        periodic=True,
        subtract_minimum=False,
        lattice_vectors_nm=LATTICE_VECTORS_NM,
        origin_nm=ORIGIN_NM,
    )
    matrix = np.column_stack(
        [LATTICE_VECTORS_NM[0], LATTICE_VECTORS_NM[1]]
    )
    return potential, values, matrix, np.linalg.inv(matrix)


def _cartesian(uv, matrix):
    return ORIGIN_NM + matrix @ np.asarray(uv, dtype=np.float64)


def _jit_value(point, values, inverse):
    return float(
        bilinear_interpolate_periodic_cell(
            float(point[0]),
            float(point[1]),
            values,
            float(ORIGIN_NM[0]),
            float(ORIGIN_NM[1]),
            float(inverse[0, 0]),
            float(inverse[0, 1]),
            float(inverse[1, 0]),
            float(inverse[1, 1]),
        )
    )


def _g0_or_32ulp_tolerance(a, b):
    scale = np.float64(max(abs(float(a)), abs(float(b))))
    ulp32 = 32.0 * abs(float(np.spacing(scale)))
    return max(ENERGY_TOL_EV, ulp32)


def _assert_energy_close(actual, expected):
    assert abs(float(actual) - float(expected)) <= _g0_or_32ulp_tolerance(
        actual, expected
    )


@pytest.mark.parametrize("eps", [1.0e-12, 1.0e-9, 1.0e-6])
def test_seam_continuity_in_both_fractional_directions(periodic_grid, eps):
    _, values, matrix, inverse = periodic_grid
    for transverse in (0.0, 0.1, 0.37, 0.9):
        for left, right in (
            ((1.0 - eps, transverse), (-eps, transverse)),
            ((transverse, 1.0 - eps), (transverse, -eps)),
        ):
            value_left = _jit_value(_cartesian(left, matrix), values, inverse)
            value_right = _jit_value(_cartesian(right, matrix), values, inverse)
            _assert_energy_close(value_left, value_right)


def test_all_four_periodic_corner_representations(periodic_grid):
    potential, values, matrix, inverse = periodic_grid
    tiny_negative = np.nextafter(np.float64(0.0), np.float64(-1.0))
    corner_representations = (
        (tiny_negative, tiny_negative),
        (tiny_negative, 1.0),
        (1.0, tiny_negative),
        (1.0, 1.0),
    )
    for uv in corner_representations:
        point = _cartesian(uv, matrix)
        expected = float(potential.value(point.reshape(1, 2))[0])
        actual = _jit_value(point, values, inverse)
        _assert_energy_close(actual, expected)


def test_tiny_negative_fraction_can_wrap_to_exactly_one(periodic_grid):
    potential, values, matrix, inverse = periodic_grid
    tiny_negative = np.nextafter(np.float64(0.0), np.float64(-1.0))
    point = _cartesian((tiny_negative, 0.1), matrix)

    rel_x = float(point[0] - ORIGIN_NM[0])
    rel_y = float(point[1] - ORIGIN_NM[1])
    recovered_u = inverse[0, 0] * rel_x + inverse[0, 1] * rel_y
    wrapped_u = recovered_u - np.floor(recovered_u)
    assert recovered_u < 0.0
    assert wrapped_u == 1.0

    expected = float(potential.value(point.reshape(1, 2))[0])
    actual = _jit_value(point, values, inverse)
    _assert_energy_close(actual, expected)


@pytest.mark.parametrize(
    ("m", "n"),
    [(37, -29), (-41, 53), (127, -113), (-211, 197)],
)
def test_large_signed_lattice_translations(periodic_grid, m, n):
    potential, values, matrix, inverse = periodic_grid
    base_uv = np.array([0.3141592653589793, 0.2718281828459045])
    base_point = _cartesian(base_uv, matrix)
    shifted = base_point + m * LATTICE_VECTORS_NM[0] + n * LATTICE_VECTORS_NM[1]

    base_jit = _jit_value(base_point, values, inverse)
    shifted_jit = _jit_value(shifted, values, inverse)
    shifted_python = float(potential.value(shifted.reshape(1, 2))[0])
    _assert_energy_close(shifted_jit, base_jit)
    _assert_energy_close(shifted_jit, shifted_python)


def test_python_grid_matches_jit_on_identical_endpoint_excluded_grid(periodic_grid):
    potential, values, matrix, inverse = periodic_grid
    rng = np.random.default_rng(2026090702)
    fractional = rng.uniform(-3.0, 4.0, size=(4096, 2))
    points = ORIGIN_NM + fractional @ matrix.T
    python_values = potential.value(points)
    jit_values = np.array(
        [_jit_value(point, values, inverse) for point in points], dtype=np.float64
    )

    differences = np.abs(jit_values - python_values)
    tolerances = np.array(
        [
            _g0_or_32ulp_tolerance(jit_value, python_value)
            for jit_value, python_value in zip(jit_values, python_values)
        ]
    )
    assert np.all(differences <= tolerances), differences.max()
