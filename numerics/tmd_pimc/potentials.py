from dataclasses import dataclass, field
from typing import Sequence, Optional, Tuple
import numpy as np


class Potential2D:
    def value(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class HarmonicPotential(Potential2D):
    k_eV_per_nm2: float = 0.002
    _half_k: float = field(init=False, repr=False)

    def __post_init__(self):
        self._half_k = 0.5 * self.k_eV_per_nm2

    def value(self, r):
        return self._half_k * np.einsum("ij,ij->i", r, r)


@dataclass
class HarmonicEnvelopePotential(Potential2D):
    k_env_eV_per_nm2: float = 0.00015
    _half_k: float = field(init=False, repr=False)

    def __post_init__(self):
        self._half_k = 0.5 * self.k_env_eV_per_nm2

    def value(self, r):
        return self._half_k * np.einsum("ij,ij->i", r, r)


@dataclass
class SoftWallBoxPotential(Potential2D):
    R_box_nm: float = 15.0
    V_wall0_eV: float = 0.08
    power: int = 8
    _inv_R: float = field(init=False, repr=False)

    def __post_init__(self):
        self._inv_R = 1.0 / self.R_box_nm

    def value(self, r):
        rad = np.sqrt(np.einsum("ij,ij->i", r, r))
        return self.V_wall0_eV * (rad * self._inv_R) ** self.power


@dataclass
class GaussianWellPotential(Potential2D):
    V0_eV: float = 0.05
    sigma_nm: float = 4.0
    center: Sequence[float] = (0.0, 0.0)
    _center: np.ndarray = field(init=False, repr=False)
    _inv2s2: float = field(init=False, repr=False)

    def __post_init__(self):
        self._center  = np.array(self.center, dtype=float)
        self._inv2s2  = 1.0 / (2.0 * self.sigma_nm ** 2)

    def value(self, r):
        dr = r - self._center
        return -self.V0_eV * np.exp(-np.einsum("ij,ij->i", dr, dr) * self._inv2s2)


@dataclass
class DoubleGaussianWellPotential(Potential2D):
    V0_eV: float = 0.05
    sigma_nm: float = 3.0
    separation_nm: float = 10.0
    asymmetry_eV: float = 0.0
    _left: np.ndarray  = field(init=False, repr=False)
    _right: np.ndarray = field(init=False, repr=False)
    _inv2s2: float     = field(init=False, repr=False)
    _right_depth: float = field(init=False, repr=False)

    def __post_init__(self):
        self._left  = np.array([-self.separation_nm / 2.0, 0.0])
        self._right = np.array([ self.separation_nm / 2.0, 0.0])
        self._inv2s2 = 1.0 / (2.0 * self.sigma_nm ** 2)
        self._right_depth = self.V0_eV + self.asymmetry_eV

    def value(self, r):
        dl = r - self._left
        dr = r - self._right
        return (
            -self.V0_eV       * np.exp(-np.einsum("ij,ij->i", dl, dl) * self._inv2s2)
            -self._right_depth * np.exp(-np.einsum("ij,ij->i", dr, dr) * self._inv2s2)
        )


@dataclass
class ExternalFieldPotential(Potential2D):
    E: Sequence[float] = (0.0, 0.0)
    q_eff: float = 1.0
    _E: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._E = np.array(self.E, dtype=float)

    def value(self, r):
        return -self.q_eff * (r @ self._E)


@dataclass
class CompositePotential(Potential2D):
    terms: Sequence[Potential2D]

    def value(self, r):
        total = np.zeros(r.shape[0])
        for term in self.terms:
            total += term.value(r)
        return total
        

@dataclass
class GridPotential2D(Potential2D):
    """
    2D potential read from a regular grid and evaluated by bilinear interpolation.

    Expected units:
      x_nm, y_nm : nm
      V_eV       : eV

    The public API is intentionally the same as the analytic potentials in this
    package: value(r), where r has shape (N, 2) and columns are x,y in nm.

    Parameters
    ----------
    periodic:
        If True, coordinates are wrapped into the grid box before interpolation.
        This is usually the right choice for a moiré supercell / periodic map.
    subtract_minimum:
        If True, shift V so that min(V)=0. This is useful for Metropolis/PIMC
        because only energy differences matter and it avoids large offsets.
    scale:
        Multiplicative factor applied after optional minimum subtraction.
    barrier_eV:
        Value returned outside the grid when periodic=False.
    """

    x_nm: Sequence[float]
    y_nm: Sequence[float]
    V_eV: np.ndarray
    periodic: bool = True
    subtract_minimum: bool = True
    scale: float = 1.0
    barrier_eV: float = 1.0e6
    lattice_vectors_nm: Optional[np.ndarray] = None
    origin_nm: Optional[np.ndarray] = None

    _x: np.ndarray = field(init=False, repr=False)
    _y: np.ndarray = field(init=False, repr=False)
    _V: np.ndarray = field(init=False, repr=False)
    _xmin: float = field(init=False, repr=False)
    _xmax: float = field(init=False, repr=False)
    _ymin: float = field(init=False, repr=False)
    _ymax: float = field(init=False, repr=False)
    _dx: float = field(init=False, repr=False)
    _dy: float = field(init=False, repr=False)
    _Lx: float = field(init=False, repr=False)
    _Ly: float = field(init=False, repr=False)
    _nx: int = field(init=False, repr=False)
    _ny: int = field(init=False, repr=False)
    _A: np.ndarray = field(init=False, repr=False)
    _Ainv: np.ndarray = field(init=False, repr=False)
    _origin: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        x = np.asarray(self.x_nm, dtype=float)
        y = np.asarray(self.y_nm, dtype=float)
        V = np.asarray(self.V_eV, dtype=float)

        x, y, V = self._normalise_grid_arrays(x, y, V)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x_nm and y_nm must be 1D arrays after normalisation")
        if x.size < 2 or y.size < 2:
            raise ValueError("GridPotential2D requires at least 2 x-points and 2 y-points")
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(V)):
            raise ValueError("x_nm, y_nm and V_eV must contain only finite values")
        if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
            raise ValueError("x_nm and y_nm must be strictly increasing")

        # Accept both common conventions:
        #   V.shape == (len(x), len(y))  -> indexing='ij'
        #   V.shape == (len(y), len(x))  -> indexing='xy' / image-like
        if V.shape == (x.size, y.size):
            pass
        elif V.shape == (y.size, x.size):
            V = V.T
        else:
            raise ValueError(
                "V_eV shape must be (len(x_nm), len(y_nm)) or "
                "(len(y_nm), len(x_nm)); got "
                f"{V.shape}, with len(x)={x.size}, len(y)={y.size}"
            )

        # Most grids here are regular. Allow tiny floating point noise, but fail
        # loudly if the grid is not rectangular/regular.
        dxs = np.diff(x)
        dys = np.diff(y)
        dx = float(np.mean(dxs))
        dy = float(np.mean(dys))
        if not np.allclose(dxs, dx, rtol=1e-5, atol=1e-10):
            raise ValueError("x_nm must be regularly spaced for fast bilinear interpolation")
        if not np.allclose(dys, dy, rtol=1e-5, atol=1e-10):
            raise ValueError("y_nm must be regularly spaced for fast bilinear interpolation")

        if self.subtract_minimum:
            V = V - float(np.min(V))
        V = float(self.scale) * V

        self._x = x
        self._y = y
        self._V = np.ascontiguousarray(V, dtype=float)
        self._xmin = float(x[0])
        self._xmax = float(x[-1])
        self._ymin = float(y[0])
        self._ymax = float(y[-1])
        self._dx = dx
        self._dy = dy
        # NOTE: grids produced by generate_potential_map.py use
        # np.linspace(..., endpoint=False), so the true physical period is
        # N*dx, not (N-1)*dx = x[-1]-x[0]. Using x[-1]-x[0] here under-counts
        # the period by exactly one grid cell and introduces a small phase
        # slip every time a PIMC path wraps across the periodic boundary.
        self._Lx = float(dx * x.size)
        self._Ly = float(dy * y.size)
        self._nx = int(x.size)
        self._ny = int(y.size)

        # General periodic-cell support: x_nm/y_nm + V_eV are always stored on a
        # regular index grid (nx, ny). By default that index grid is interpreted
        # as an axis-aligned rectangle of size Lx x Ly (unchanged old behaviour).
        # If lattice_vectors_nm is supplied, the SAME index grid is instead
        # interpreted as spanning one full period along two arbitrary (possibly
        # non-orthogonal, e.g. ~60/120 deg for a true commensurate moire cell)
        # lattice vectors a1, a2. Wrapping happens in fractional (u, v) cell
        # coordinates -- u,v in [0,1) -- which is always "rectangular" regardless
        # of the real-space cell shape, so a single interpolation routine covers
        # both the legacy square-box grids and true rhombic supercells.
        if self.lattice_vectors_nm is None:
            a1 = np.array([self._Lx, 0.0])
            a2 = np.array([0.0, self._Ly])
        else:
            lv = np.asarray(self.lattice_vectors_nm, dtype=float)
            if lv.shape != (2, 2):
                raise ValueError(
                    "lattice_vectors_nm must have shape (2,2): rows are "
                    "[a1, a2] in nm (the two periodic supercell vectors)."
                )
            a1, a2 = lv[0], lv[1]

        self._A = np.column_stack([a1, a2])  # Cartesian_rel = A @ [u, v]
        det = float(np.linalg.det(self._A))
        if abs(det) < 1e-12:
            raise ValueError("lattice_vectors_nm are degenerate (zero-area periodic cell)")
        self._Ainv = np.linalg.inv(self._A)
        self._origin = (np.asarray(self.origin_nm, dtype=float)
                         if self.origin_nm is not None
                         else np.array([self._xmin, self._ymin]))

    @staticmethod
    def _normalise_grid_arrays(x: np.ndarray, y: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert 2D meshgrid-style x/y arrays to 1D coordinate axes."""
        if x.ndim == 2:
            # indexing='ij': x varies down rows; indexing='xy': x varies across columns
            if np.allclose(x, x[:, :1]):
                x = x[:, 0]
            elif np.allclose(x, x[:1, :]):
                x = x[0, :]
            else:
                raise ValueError("2D x_nm is not a regular meshgrid coordinate array")
        if y.ndim == 2:
            if np.allclose(y, y[:1, :]):
                y = y[0, :]
            elif np.allclose(y, y[:, :1]):
                y = y[:, 0]
            else:
                raise ValueError("2D y_nm is not a regular meshgrid coordinate array")
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float), V

    @classmethod
    def from_npz(cls,
                 path,
                 periodic: bool = True,
                 subtract_minimum: bool = True,
                 scale: float = 1.0,
                 barrier_eV: float = 1.0e6,
                 x_key: Optional[str] = None,
                 y_key: Optional[str] = None,
                 v_key: Optional[str] = None):
        """
        Load a grid potential from .npz.

        Preferred keys are:
            x_nm, y_nm, V_eV

        Also supported for convenience:
            x/y/V, X_nm/Y_nm/V_eV, V_meV, V, potential_eV,
            potential_meV, V_grid_eV, V_grid_meV.
        """
        data = np.load(path)
        keys = set(data.files)

        def choose(explicit, candidates, what):
            if explicit is not None:
                if explicit not in keys:
                    raise KeyError(f"Requested {what} key {explicit!r} not found in {path}; keys={sorted(keys)}")
                return explicit
            for key in candidates:
                if key in keys:
                    return key
            raise KeyError(f"Could not find {what} in {path}; available keys={sorted(keys)}")

        kx = choose(x_key, ("x_nm", "x", "X_nm", "X", "grid_x_nm", "x_grid_nm"), "x grid")
        ky = choose(y_key, ("y_nm", "y", "Y_nm", "Y", "grid_y_nm", "y_grid_nm"), "y grid")
        kv = choose(v_key, (
            "V_eV", "V", "potential_eV", "V_grid_eV",
            "V_meV", "potential_meV", "V_grid_meV"
        ), "potential grid")

        V = np.asarray(data[kv], dtype=float)
        if kv.lower().endswith("mev") or "mev" in kv.lower():
            V = 1.0e-3 * V

        lattice_vectors_nm = (np.asarray(data["lattice_vectors_nm"], dtype=float)
                               if "lattice_vectors_nm" in keys else None)
        origin_nm = (np.asarray(data["origin_nm"], dtype=float)
                     if "origin_nm" in keys else None)

        return cls(
            x_nm=np.asarray(data[kx], dtype=float),
            y_nm=np.asarray(data[ky], dtype=float),
            V_eV=V,
            periodic=periodic,
            subtract_minimum=subtract_minimum,
            scale=scale,
            barrier_eV=barrier_eV,
            lattice_vectors_nm=lattice_vectors_nm,
            origin_nm=origin_nm,
        )

    @property
    def x_nm_axis(self) -> np.ndarray:
        return self._x.copy()

    @property
    def y_nm_axis(self) -> np.ndarray:
        return self._y.copy()

    @property
    def V_grid_eV(self) -> np.ndarray:
        return self._V.copy()

    @property
    def bounds_nm(self) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) in nm."""
        return self._xmin, self._xmax, self._ymin, self._ymax

    @property
    def lattice_vectors(self) -> np.ndarray:
        """(2,2) array [a1, a2] in nm -- the two periodic cell vectors actually
        in use, whether inferred from an orthogonal box or supplied explicitly."""
        return np.array([self._A[:, 0], self._A[:, 1]])

    @property
    def cell_origin_nm(self) -> np.ndarray:
        """Cartesian origin used for fractional periodic-cell coordinates."""
        return self._origin.copy()

    @property
    def cell_bounds_nm(self) -> Tuple[float, float, float, float]:
        """Axis-aligned Cartesian bounding box of the actual lattice cell."""
        a1, a2 = self.lattice_vectors
        corners = np.asarray([
            self._origin,
            self._origin + a1,
            self._origin + a2,
            self._origin + a1 + a2,
        ])
        return (
            float(np.min(corners[:, 0])),
            float(np.max(corners[:, 0])),
            float(np.min(corners[:, 1])),
            float(np.max(corners[:, 1])),
        )

    def fractional_coordinates(self, r: np.ndarray, wrap: bool = False) -> np.ndarray:
        """Convert Cartesian points to fractional cell coordinates (u, v)."""
        pts = np.asarray(r, dtype=float)
        single = pts.ndim == 1
        if single:
            if pts.size != 2:
                raise ValueError("A single coordinate must have shape (2,)")
            pts = pts.reshape(1, 2)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("r must have shape (N, 2)")
        frac = (pts - self._origin) @ self._Ainv.T
        if wrap:
            frac = frac % 1.0
        return frac[0] if single else frac

    def cartesian_from_fractional(self, uv: np.ndarray) -> np.ndarray:
        """Convert fractional cell coordinates to Cartesian coordinates."""
        frac = np.asarray(uv, dtype=float)
        single = frac.ndim == 1
        if single:
            if frac.size != 2:
                raise ValueError("A single fractional coordinate must have shape (2,)")
            frac = frac.reshape(1, 2)
        if frac.ndim != 2 or frac.shape[1] != 2:
            raise ValueError("uv must have shape (N, 2)")
        pts = self._origin + frac @ self._A.T
        return pts[0] if single else pts

    def wrap_points(self, r: np.ndarray) -> np.ndarray:
        """Wrap Cartesian points into the primary periodic cell."""
        return self.cartesian_from_fractional(
            self.fractional_coordinates(r, wrap=True)
        )

    def value(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        if r.ndim == 1:
            if r.size != 2:
                raise ValueError("A single coordinate must have shape (2,)")
            r = r.reshape(1, 2)
        if r.ndim != 2 or r.shape[1] != 2:
            raise ValueError("r must have shape (N, 2)")

        xq = r[:, 0].astype(float, copy=True)
        yq = r[:, 1].astype(float, copy=True)

        # Cartesian -> fractional lattice ("cell") coordinates. u,v in [0,1)
        # span one full period along a1, a2 respectively. For the default
        # orthogonal case this reduces exactly to the old (x-xmin)/Lx style
        # wrapping; for a general lattice_vectors_nm it correctly wraps along
        # the true (possibly non-orthogonal) periodic directions instead of
        # doing an incorrect independent x%Lx, y%Ly rectangle wrap.
        rel_x = xq - self._origin[0]
        rel_y = yq - self._origin[1]
        u = self._Ainv[0, 0] * rel_x + self._Ainv[0, 1] * rel_y
        v = self._Ainv[1, 0] * rel_x + self._Ainv[1, 1] * rel_y

        if self.periodic:
            u = u % 1.0
            v = v % 1.0
            inside = np.ones(r.shape[0], dtype=bool)
        else:
            inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)

        out = np.full(r.shape[0], float(self.barrier_eV), dtype=float)
        if not np.any(inside):
            return out

        ui = u[inside]
        vi = v[inside]

        # Fractional grid index within the (nx, ny) sample array.
        fx = ui * self._nx
        fy = vi * self._ny

        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        ix = np.clip(ix, 0, self._nx - 1)
        iy = np.clip(iy, 0, self._ny - 1)

        wx = fx - ix
        wy = fy - iy

        if self.periodic:
            # FIX: the neighbouring sample for the last grid cell wraps around
            # to index 0 (that IS the periodic image), rather than being
            # clamped to nx-2/ny-2. The old clamping silently used stale
            # (ix-1, ix) data for every periodic wrap, i.e. every repeat of
            # the cell away from the origin -- which is most of the domain
            # once GridPotential2D is periodically tiled to build the dense
            # PIMC lookup table.
            ix1 = (ix + 1) % self._nx
            iy1 = (iy + 1) % self._ny
        else:
            ix1 = np.clip(ix + 1, 0, self._nx - 1)
            iy1 = np.clip(iy + 1, 0, self._ny - 1)

        v00 = self._V[ix, iy]
        v10 = self._V[ix1, iy]
        v01 = self._V[ix, iy1]
        v11 = self._V[ix1, iy1]

        out[inside] = ((1.0 - wx) * (1.0 - wy) * v00 +
                       wx         * (1.0 - wy) * v10 +
                       (1.0 - wx) * wy         * v01 +
                       wx         * wy         * v11)
        return out

    # Aliases useful for old/new runner scripts.
    def potential(self, r: np.ndarray) -> np.ndarray:
        return self.value(r)

    def energy(self, r: np.ndarray) -> np.ndarray:
        return self.value(r)

@dataclass
class MoirePotential(Potential2D):
    """
    Simple hexagonal moiré potential.

    V(r) = V0 [cos(G1·r) + cos(G2·r) + cos(G3·r)]

    The three reciprocal lattice vectors G1, G2, G3 form the first shell
    of the hexagonal moiré superlattice with period `period_nm`.

    Note: V has a maximum at r = (0, 0) and minima offset into the unit cell.
    Initialise the sampler at a true minimum (e.g. center=(period_nm/2, 0))
    rather than at the origin to avoid starting on a potential hill.
    """

    amplitude_eV: float = 0.020
    period_nm:    float = 20.0

    # reciprocal vectors and cached amplitude — set in __post_init__
    _V0: float        = field(init=False, repr=False)
    _G1: np.ndarray   = field(init=False, repr=False)
    _G2: np.ndarray   = field(init=False, repr=False)
    _G3: np.ndarray   = field(init=False, repr=False)

    def __post_init__(self):
        self._V0 = self.amplitude_eV
        G = 4.0 * np.pi / (np.sqrt(3.0) * self.period_nm)
        self._G1 = np.array([ G,                    0.0])
        self._G2 = np.array([-0.5 * G,  np.sqrt(3.0) / 2.0 * G])
        self._G3 = np.array([-0.5 * G, -np.sqrt(3.0) / 2.0 * G])

    def value(self, r: np.ndarray) -> np.ndarray:
        p1 = r @ self._G1
        p2 = r @ self._G2
        p3 = r @ self._G3
        return self._V0 * (np.cos(p1) + np.cos(p2) + np.cos(p3))

@dataclass
class StrainPiezoelectricPotential(Potential2D):
    """
    Effective 2D piezoelectric / strain-induced moire potential.

    This is still a COM-level phenomenological potential, not a full
    electron-hole piezoelectric model.  In a true e-h model the electron and
    hole should generally receive separate band-edge/piezoelectric potentials.

    The important improvement relative to the old version is `anchor_nm`.
    The phase of the sinusoidal moire pattern can now be pinned to a physically
    meaningful point, e.g. a low-energy registry minimum found from V_grid,
    instead of being implicitly pinned to the arbitrary coordinate origin.

        V_piezo(r) = A * basis(r - anchor)

    Parameters
    ----------
    amplitude_eV:
        Piezo modulation amplitude. With normalisation="harmonic" this is the
        amplitude of each sine harmonic, reproducing the old behaviour. With
        normalisation="peak" the three-sine basis is divided by 3, so this is
        approximately the peak amplitude of the total modulation.
    period_nm:
        Moire period in nm. Use the period measured from the atomistic grid.
    phase_rad:
        Global phase shift of the sinusoidal modulation.
    anchor_nm:
        Real-space phase anchor in nm. Use a registry/minimum coordinate rather
        than the bounding-box centre when possible.
    normalisation:
        "harmonic" = old behaviour, no division of the three-sine sum.
        "peak"     = divide the three-sine sum by 3.
    """

    amplitude_eV: float = 0.012
    period_nm: float = 20.0
    phase_rad: float = 0.0
    anchor_nm: Sequence[float] = (0.0, 0.0)
    normalisation: str = "harmonic"

    _V0: float = field(init=False, repr=False)
    _anchor: np.ndarray = field(init=False, repr=False)
    _G1: np.ndarray = field(init=False, repr=False)
    _G2: np.ndarray = field(init=False, repr=False)
    _G3: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.period_nm <= 0.0:
            raise ValueError(f"period_nm must be positive, got {self.period_nm}")
        if self.normalisation not in ("harmonic", "peak"):
            raise ValueError("normalisation must be 'harmonic' or 'peak'")

        self._V0 = float(self.amplitude_eV)
        self._anchor = np.asarray(self.anchor_nm, dtype=float)
        if self._anchor.shape != (2,):
            raise ValueError("anchor_nm must be a 2-element coordinate (x_nm, y_nm)")

        G = 4.0 * np.pi / (np.sqrt(3.0) * float(self.period_nm))
        self._G1 = np.array([ G,                    0.0], dtype=float)
        self._G2 = np.array([-0.5 * G,  np.sqrt(3.0) / 2.0 * G], dtype=float)
        self._G3 = np.array([-0.5 * G, -np.sqrt(3.0) / 2.0 * G], dtype=float)

    def _moire_basis(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        if r.ndim == 1:
            if r.size != 2:
                raise ValueError("A single coordinate must have shape (2,)")
            r = r.reshape(1, 2)
        if r.ndim != 2 or r.shape[1] != 2:
            raise ValueError("r must have shape (N, 2)")

        r_shifted = r - self._anchor
        p1 = r_shifted @ self._G1
        p2 = r_shifted @ self._G2
        p3 = r_shifted @ self._G3

        basis = (
            np.sin(p1 + self.phase_rad) +
            np.sin(p2 + self.phase_rad) +
            np.sin(p3 + self.phase_rad)
        )

        if self.normalisation == "peak":
            basis = basis / 3.0
        return basis

    def value(self, r: np.ndarray) -> np.ndarray:
        return self._V0 * self._moire_basis(r)
        
@dataclass
class OutOfPlaneStarkPotential(Potential2D):
    """
    Effective COM-level out-of-plane (vertical, Ez) Stark potential.

    This represents the -p_z(r)*Fz term in V_eff^COM(R;Fz)
    (registry + Stark), i.e. the coupling of a spatially varying effective
    interlayer dipole p_z(r) to a uniform vertical field Fz. It is still a
    COM-level phenomenological term, not a separate electron/hole dipole
    model.

    As with StrainPiezoelectricPotential, the phase of the dipole-texture
    modulation is pinned to `anchor_nm` rather than to the coordinate
    origin. Getting this anchor right matters: the origin of an atomistic
    V_grid.npz is an arbitrary Cartesian choice, not a physically meaningful
    point, so an unanchored Stark term would tilt the landscape around the
    wrong point and bias which basin the field appears to favour.

    Recommended anchor convention (chosen by the caller, e.g. the runner
    script via --stark_anchor_mode):
      - "grid_min"  : anchor_nm = grid_info.start_centres_nm[0], i.e. the
                       lowest-energy registry well found on the atomistic
                       grid. This is the recommended default.
      - "explicit"  : anchor_nm supplied directly via --stark_anchor_nm.
      - "origin"    : anchor_nm = (0.0, 0.0); kept only for backward
                       compatibility / debugging, not recommended for
                       production runs.

        V_stark(r; Fz) = -Fz * dipole_length_nm * basis(r - anchor)

    Parameters
    ----------
    Fz_eV_per_nm:
        Applied vertical field, in eV/nm (same effective-field convention
        as ExternalFieldPotential's E).
    dipole_length_nm:
        Effective interlayer dipole length p_z/e, in nm. Together with Fz
        this sets the peak Stark modulation amplitude, Fz*dipole_length_nm.
        A reasonable starting point is the registry-averaged interlayer
        distance modulation from V_grid.npz's interlayer_distance_nm field,
        not an assumed constant -- see generate_potential_map.py.
    period_nm:
        Moire period in nm; use the period measured from the atomistic
        grid, not an assumed analytic value.
    phase_rad:
        Global phase shift of the dipole-texture modulation.
    anchor_nm:
        Real-space phase anchor in nm. See anchor convention above.
    normalisation:
        "harmonic" = no division of the three-sine sum (matches
        StrainPiezoelectricPotential's default).
        "peak"     = divide the three-sine sum by 3.
    """

    Fz_eV_per_nm: float = 0.0
    dipole_length_nm: float = 0.05
    period_nm: float = 20.0
    phase_rad: float = 0.0
    anchor_nm: Sequence[float] = (0.0, 0.0)
    normalisation: str = "harmonic"

    _V0: float = field(init=False, repr=False)
    _anchor: np.ndarray = field(init=False, repr=False)
    _G1: np.ndarray = field(init=False, repr=False)
    _G2: np.ndarray = field(init=False, repr=False)
    _G3: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.period_nm <= 0.0:
            raise ValueError(f"period_nm must be positive, got {self.period_nm}")
        if self.normalisation not in ("harmonic", "peak"):
            raise ValueError("normalisation must be 'harmonic' or 'peak'")

        # V0 carries the sign: positive Fz pulls the exciton toward the
        # anchor's dipole-favoured basin, matching the sign convention of
        # ExternalFieldPotential.value(r) = -q_eff * (r . E).
        self._V0 = -float(self.Fz_eV_per_nm) * float(self.dipole_length_nm)
        self._anchor = np.asarray(self.anchor_nm, dtype=float)
        if self._anchor.shape != (2,):
            raise ValueError("anchor_nm must be a 2-element coordinate (x_nm, y_nm)")

        G = 4.0 * np.pi / (np.sqrt(3.0) * float(self.period_nm))
        self._G1 = np.array([ G,                    0.0], dtype=float)
        self._G2 = np.array([-0.5 * G,  np.sqrt(3.0) / 2.0 * G], dtype=float)
        self._G3 = np.array([-0.5 * G, -np.sqrt(3.0) / 2.0 * G], dtype=float)

    def _dipole_basis(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        if r.ndim == 1:
            if r.size != 2:
                raise ValueError("A single coordinate must have shape (2,)")
            r = r.reshape(1, 2)
        if r.ndim != 2 or r.shape[1] != 2:
            raise ValueError("r must have shape (N, 2)")

        r_shifted = r - self._anchor
        p1 = r_shifted @ self._G1
        p2 = r_shifted @ self._G2
        p3 = r_shifted @ self._G3

        basis = (
            np.sin(p1 + self.phase_rad) +
            np.sin(p2 + self.phase_rad) +
            np.sin(p3 + self.phase_rad)
        )

        if self.normalisation == "peak":
            basis = basis / 3.0
        return basis

    def value(self, r: np.ndarray) -> np.ndarray:
        return self._V0 * self._dipole_basis(r)


@dataclass
class SoftCoulombPotential(Potential2D):
    """
    Effective exciton binding energy.

    This is NOT yet a true electron-hole simulation.
    It is an effective attractive Coulomb correction.

    V(r) = -A / sqrt(r² + a²)
    """

    strength_eV_nm: float = 0.15
    softening_nm: float = 1.0

    def value(self, r):
        r2 = np.einsum("ij,ij->i", r, r)
        return -self.strength_eV_nm / np.sqrt(
            r2 + self.softening_nm**2
        )


def moire_hop_vectors_nm(period_nm: float, include_bravais: bool = True) -> np.ndarray:
    """Wektory przemieszczenia łączące równoważne minima MoirePotential.

    Dla V(r) = (V0/3) sum_i cos(G_i . r) z trzema G_i rozstawionymi co 120
    stopni, minima tworzą sieć PLASTRA MIODU (dwa minima na komórkę
    prymitywną), a nie sieć Bravais. W konsekwencji istnieją dwie istotne
    powłoki wektorów przeskoku:

      * 3 wektory najbliższych sąsiadów o długości ``period_nm / sqrt(3)``
        (łączą dwie podsieci plastra miodu),
      * 6 wektorów sieci Bravais o długości ``period_nm``.

    Zwracany zbiór jest DOMKNIĘTY NA NEGACJĘ, co jest warunkiem koniecznym
    symetryczności propozycji Monte Carlo (patrz docstring
    ``run_pimc_core_jit``): dla wektorów plastra miodu oznacza to, że z
    danego węzła połowa propozycji trafia w równoważne minimum, a połowa
    nie -- to strata wydajności rzędu 2x, ale zachowuje równowagę
    szczegółową bez czynnika Hastingsa.

    Parameters
    ----------
    period_nm : float
        Okres potencjału moiré (ten sam, którego używa ``MoirePotential``).
    include_bravais : bool
        Czy dołączyć 6 wektorów sieci Bravais oprócz 6 wektorów plastra
        miodu. Domyślnie True.

    Returns
    -------
    np.ndarray, kształt (n_vec, 2), dtype float64
    """
    period_nm = float(period_nm)
    if period_nm <= 0.0:
        raise ValueError("period_nm must be positive")

    # Wektory sieci rzeczywistej dualne do G_i (G_i . a_j = 2 pi delta_ij).
    G = 4.0 * np.pi / (np.sqrt(3.0) * period_nm)
    ang = np.deg2rad([0.0, 120.0])
    Gm = np.stack([G * np.cos(ang), G * np.sin(ang)], axis=1)
    A = 2.0 * np.pi * np.linalg.inv(Gm).T
    a1, a2 = A[0], A[1]

    vecs = []

    # Powłoka plastra miodu: 3 najkrótsze wektory laczace podsieci, +/-.
    # Wyznaczone jako (a1 + a2)/3 i jego obroty o +/-120 stopni.
    d1 = (a1 + a2) / 3.0
    for theta in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        d = R @ d1
        vecs.append(d)
        vecs.append(-d)

    if include_bravais:
        # 6 najblizszych sasiadow Bravais dla siatki heksagonalnej
        # (kat 60 stopni miedzy a1 i a2): +/-a1, +/-a2, +/-(a1-a2), wszystkie
        # o dlugosci period_nm. Uwaga: (1,1) daje |a1+a2| = period*sqrt(3),
        # czyli DALSZA powloke -- nie nalezy jej tu uzywac.
        for (i, j) in ((1, 0), (0, 1), (1, -1)):
            v = i * a1 + j * a2
            vecs.append(v)
            vecs.append(-v)

    return np.asarray(vecs, dtype=np.float64)
