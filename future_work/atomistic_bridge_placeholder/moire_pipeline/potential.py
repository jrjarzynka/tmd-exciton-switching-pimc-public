# moire_pipeline/potential.py
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
from typing import Any

@dataclass
class GridPotential2D:
    x_nm: np.ndarray
    y_nm: np.ndarray
    V_eV: np.ndarray
    periodic: bool = True
    subtract_minimum: bool = True
    scale: float = 1.0
    barrier_eV: float = 1e6
    outside_value: float | None = None
    _x: np.ndarray = field(init=False, repr=False)
    _y: np.ndarray = field(init=False, repr=False)
    _V: np.ndarray = field(init=False, repr=False)
    _nx: int = field(init=False, repr=False)
    _ny: int = field(init=False, repr=False)
    _dx: float = field(init=False, repr=False)
    _dy: float = field(init=False, repr=False)
    _x0: float = field(init=False, repr=False)
    _y0: float = field(init=False, repr=False)
    _Lx: float = field(init=False, repr=False)
    _Ly: float = field(init=False, repr=False)

    def __post_init__(self):
        self._x = np.asarray(self.x_nm, dtype=float).ravel()
        self._y = np.asarray(self.y_nm, dtype=float).ravel()
        self._V = np.asarray(self.V_eV, dtype=float) * float(self.scale)
        if self._x.ndim != 1 or self._y.ndim != 1:
            raise ValueError("x_nm and y_nm must be 1-D arrays")
        if self._V.shape != (len(self._x), len(self._y)):
            raise ValueError("V_eV shape must be (len(x_nm), len(y_nm))")
        if len(self._x) < 2 or len(self._y) < 2:
            raise ValueError("x_nm and y_nm must each contain at least two points")
        dxs = np.diff(self._x)
        dys = np.diff(self._y)
        if not (np.all(dxs > 0) and np.all(dys > 0)):
            raise ValueError("x_nm and y_nm must be strictly increasing")
        if not (np.allclose(dxs, dxs[0]) and np.allclose(dys, dys[0])):
            import warnings
            warnings.warn("x_nm or y_nm are not uniformly spaced; interpolation assumes near-uniform grid", UserWarning)
        self._dx = float(dxs.mean())
        self._dy = float(dys.mean())
        self._nx = len(self._x)
        self._ny = len(self._y)
        self._x0 = float(self._x[0])
        self._y0 = float(self._y[0])
        self._Lx = self._dx * self._nx
        self._Ly = self._dy * self._ny
        if self.subtract_minimum:
            vmin = float(np.nanmin(self._V))
            self._V = self._V - vmin
        if self.outside_value is None:
            self.outside_value = float(self.barrier_eV)

    @classmethod
    def from_npz(cls, path: str | Path, periodic: bool = True, subtract_minimum: bool = True, scale: float = 1.0, barrier_eV: float = 1e6, outside_value: float | None = None):
        d = np.load(path, allow_pickle=False)
        missing = [k for k in ("x_nm", "y_nm", "V_eV") if k not in d]
        if missing:
            raise KeyError(f"{path} missing {missing}")
        return cls(d["x_nm"], d["y_nm"], d["V_eV"], periodic, subtract_minimum, scale, barrier_eV, outside_value)

    def __repr__(self):
        return f"GridPotential2D(nx={self._nx}, ny={self._ny}, periodic={self.periodic})"

    def value(self, r):
        r = np.asarray(r, dtype=float)
        scalar_input = False
        if r.ndim == 1 and r.size == 2:
            r = r.reshape(1, 2)
            scalar_input = True
        if r.ndim != 2 or r.shape[1] != 2:
            raise ValueError("value expects (N,2) or (2,)")
        x = r[:, 0]
        y = r[:, 1]
        if self.periodic:
            ux = ((x - self._x0) % self._Lx) / self._dx
            uy = ((y - self._y0) % self._Ly) / self._dy
            outside = np.zeros_like(x, dtype=bool)
            i0 = np.floor(ux).astype(np.int64) % self._nx
            j0 = np.floor(uy).astype(np.int64) % self._ny
            i1 = (i0 + 1) % self._nx
            j1 = (j0 + 1) % self._ny
        else:
            ux = (x - self._x0) / self._dx
            uy = (y - self._y0) / self._dy
            outside = (ux < 0) | (uy < 0) | (ux > (self._nx - 1)) | (uy > (self._ny - 1))
            max_u_x = np.nextafter(self._nx - 1, -np.inf)
            max_u_y = np.nextafter(self._ny - 1, -np.inf)
            ux = np.clip(ux, 0.0, max_u_x)
            uy = np.clip(uy, 0.0, max_u_y)
            i0 = np.floor(ux).astype(np.int64)
            j0 = np.floor(uy).astype(np.int64)
            i1 = np.minimum(i0 + 1, self._nx - 1)
            j1 = np.minimum(j0 + 1, self._ny - 1)
        tx = ux - np.floor(ux)
        ty = uy - np.floor(uy)
        v00 = self._V[i0, j0]
        v10 = self._V[i1, j0]
        v01 = self._V[i0, j1]
        v11 = self._V[i1, j1]
        out = (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11
        if not self.periodic and outside.any():
            out = out.copy()
            out[outside] = self.outside_value
        return out[0] if scalar_input else out

    __call__ = value

