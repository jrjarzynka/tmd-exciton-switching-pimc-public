# moire_pipeline/generate_potential_map.py
from __future__ import annotations
from pathlib import Path
import math
import warnings
import numpy as np
from typing import Dict, Any
from .utils import save_npz_with_meta, logger

# --- geometry helpers -----------------------------------------------------
def rot(th: float) -> np.ndarray:
    c = math.cos(th)
    s = math.sin(th)
    return np.array([[c, -s], [s, c]], float)


def avec(a: float):
    return np.array([a, 0.0]), np.array([0.5 * a, 0.5 * math.sqrt(3) * a])


def recip(a: float):
    a1, a2 = avec(a)
    A = np.column_stack([a1, a2])
    B = 2.0 * math.pi * np.linalg.inv(A).T
    b1, b2 = B[:, 0], B[:, 1]
    return b1, b2, -(b1 + b2)


def moire_G(a_bottom: float, a_top: float, theta_deg: float) -> np.ndarray:
    R = rot(math.radians(theta_deg))
    return np.array([R @ bt - bb for bt, bb in zip(recip(a_top), recip(a_bottom))], float)


# --- registry normalization -----------------------------------------------
def _normalise_raw_registry(raw: np.ndarray, depth_eV: float, mode: str) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "local":
        v = raw - np.nanmin(raw)
        m = np.nanmax(v)
        return np.zeros_like(v) if m <= 0 else depth_eV * v / m
    if mode == "global":
        rmin, rmax = -1.5, 3.0
        v = (raw - rmin) / (rmax - rmin)
        return depth_eV * np.clip(v, 0.0, 1.0)
    raise ValueError("registry_norm_mode must be 'global' or 'local'")


def registry_energy(
    X_nm: np.ndarray,
    Y_nm: np.ndarray,
    a_bottom: float,
    a_top: float,
    theta_deg: float,
    depth_eV: float,
    phase: float = 0.0,
    registry_shift_nm: tuple[float, float] | None = None,
    invert: bool = False,
    registry_norm_mode: str = "global",
) -> np.ndarray:
    """Registry (stacking-offset) energy landscape.

    registry_shift_nm is the physically correct way to specify a registry
    offset: a rigid in-plane translation of the top layer relative to the
    bottom layer by the given (dx_nm, dy_nm) vector. Internally this is
    implemented as raw(r) = sum_k cos(G_k . (r - shift)), i.e. a genuine
    per-G_k phase of -G_k . shift for EACH of the three moire G-vectors.

    BUGFIX (2026-08-02): the legacy `phase` parameter added a single scalar
    offset to only the k=0 cosine term, e.g.
        raw = cos(G1.r + phase) + cos(G2.r) + cos(G3.r).
    This does NOT correspond to any rigid translation of the registry
    pattern -- verified numerically: even the best-fit scalar `phase`
    reproduces a true rigid shift only to ~18% residual RMS (relative to
    the pattern's own amplitude), i.e. it distorts the pattern shape
    instead of translating it. A translated pattern requires a DIFFERENT
    phase offset -G_k.shift for each of the three G_k (they are not
    parallel, so a single scalar cannot represent that). `phase` is kept
    only for exact backward compatibility with old configs that pinned it
    at the harmless default of 0.0 (no config in this codebase ever used a
    nonzero value); passing a nonzero `phase` now raises, forcing a switch
    to `registry_shift_nm`, since silently returning a distorted-not-
    translated landscape is worse than an explicit error.

    A useful invariant that falls out of doing this correctly: since a
    rigid translation of a periodic function does not change the SET of
    values it attains (only where each value occurs), the analytic
    registry_norm_mode="global" bounds (raw in [-1.5, 3.0], or [-3.0, 1.5]
    if invert=True) hold for ANY registry_shift_nm, not just the origin --
    unlike the old buggy `phase` parameter, which (also verified
    numerically) pushed raw outside of [-1.5, 3.0] and caused silent
    clipping in "global" mode for any nonzero value.
    """
    if phase != 0.0 and registry_shift_nm is not None:
        raise ValueError("pass either phase (legacy, must be 0.0) or registry_shift_nm, not both")
    if phase != 0.0:
        raise ValueError(
            "phase != 0.0 does not implement a physical registry shift (see "
            "registry_energy docstring) and has been disabled to prevent "
            "silently generating a distorted, not translated, landscape. "
            "Use registry_shift_nm=(dx_nm, dy_nm) instead."
        )
    X_A = 10.0 * X_nm
    Y_A = 10.0 * Y_nm
    G = moire_G(a_bottom, a_top, theta_deg)
    if registry_shift_nm is not None:
        shift_A = np.asarray(registry_shift_nm, dtype=float) * 10.0
        phase_k = [-(float(gx) * shift_A[0] + float(gy) * shift_A[1]) for gx, gy in G]
    else:
        phase_k = [0.0, 0.0, 0.0]
    raw = np.zeros_like(X_A)
    for k, (gx, gy) in enumerate(G):
        raw += np.cos(gx * X_A + gy * Y_A + phase_k[k])
    if invert:
        raw = -raw
    if str(registry_norm_mode).lower() == "global":
        rmin, rmax = (-3.0, 1.5) if invert else (-1.5, 3.0)
        v = (raw - rmin) / (rmax - rmin)
        return depth_eV * np.clip(v, 0.0, 1.0)
    return _normalise_raw_registry(raw, depth_eV, registry_norm_mode)


# --- IDW interpolation (KDTree fallback) ----------------------------------
def _tree():
    try:
        from scipy.spatial import cKDTree
        return cKDTree
    except Exception:
        return None


def idw(points, values, q, k=8, power=2.0, chunk=15000):
    points = np.asarray(points, float)
    values = np.asarray(values, float)
    q = np.asarray(q, float)
    if values.ndim == 1:
        values = values[:, None]
        squeeze = True
    else:
        squeeze = False
    k = max(1, min(k, len(points)))
    Tree = _tree()
    if Tree is not None:
        tr = Tree(points)
        d, idx = tr.query(q, k=k)
        if k == 1:
            d = d[:, None]
            idx = idx[:, None]
        # handle exact matches
        w = 1.0 / np.maximum(d, 1e-12) ** power
        w_sum = w.sum(axis=1, keepdims=True)
        w = w / w_sum
        out = np.einsum("gk,gkm->gm", w, values[idx])
        exact_mask = (d == 0.0)
        if exact_mask.any():
            rows, cols = np.where(exact_mask)
            for r, c in zip(rows, cols):
                out[r] = values[idx[r, c]]
        return out[:, 0] if squeeze else out
    # fallback (chunked)
    out = np.empty((len(q), values.shape[1]))
    for a in range(0, len(q), chunk):
        b = min(a + chunk, len(q))
        d2 = ((q[a:b, None, :] - points[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        d = np.sqrt(np.take_along_axis(d2, idx, axis=1))
        w = 1.0 / np.maximum(d, 1e-12) ** power
        w /= w.sum(axis=1, keepdims=True)
        out[a:b] = np.einsum("gk,gkm->gm", w, values[idx])
    return out[:, 0] if squeeze else out


# --- strain and interlayer maps -------------------------------------------
def strain_maps(st: Dict[str, Any], x_nm: np.ndarray, y_nm: np.ndarray, lname: str, k=8):
    la, sub = st["layer"], st["sublattice"]
    r0, r1 = st["positions_initial_A"], st["positions_relaxed_A"]
    mask = (la == lname) & (sub == "metal")
    pts = r1[mask, :2] / 10.0
    disp = (r1[mask, :2] - r0[mask, :2]) / 10.0
    X, Y = np.meshgrid(x_nm, y_nm, indexing="ij")
    q = np.c_[X.ravel(), Y.ravel()]
    u = idw(pts, disp, q, k=k)
    ux = u[:, 0].reshape(len(x_nm), len(y_nm))
    uy = u[:, 1].reshape(len(x_nm), len(y_nm))
    dux_dx = np.gradient(ux, x_nm, axis=0)
    dux_dy = np.gradient(ux, y_nm, axis=1)
    duy_dx = np.gradient(uy, x_nm, axis=0)
    duy_dy = np.gradient(uy, y_nm, axis=1)
    return {
        "ux": ux,
        "uy": uy,
        "xx": dux_dx,
        "yy": duy_dy,
        "xy": 0.5 * (dux_dy + duy_dx),
        "hydro": dux_dx + duy_dy,
    }


def interlayer_distance(st: Dict[str, Any], x_nm: np.ndarray, y_nm: np.ndarray, k=8):
    la, sub, r = st["layer"], st["sublattice"], st["positions_relaxed_A"]
    mb = (la == "bottom") & (sub == "metal")
    mt = (la == "top") & (sub == "metal")
    X, Y = np.meshgrid(x_nm, y_nm, indexing="ij")
    q = np.c_[X.ravel(), Y.ravel()]
    zb = idw(r[mb, :2] / 10.0, r[mb, 2] / 10.0, q, k=k)
    zt = idw(r[mt, :2] / 10.0, r[mt, 2] / 10.0, q, k=k)
    return (zt - zb).reshape(len(x_nm), len(y_nm))


# --- main generator -------------------------------------------------------
def generate_potential_map(
    structure_npz: str | Path,
    out_npz: str | Path,
    grid_n: int = 400,
    registry_depth_meV: float = 90.0,
    deformation_c_eV: float = -5.0,
    deformation_v_eV: float = -3.0,
    interlayer_alpha_eV_per_nm: float = 0.0,
    phase_rad: float = 0.0,
    registry_shift_nm: tuple[float, float] | None = None,
    invert_registry: bool = False,
    k_neighbors: int = 8,
    registry_norm_mode: str = "global",
    allow_small_box: bool = False,
    min_moire_cells: float = 1.0,
    disable_deformation: bool = False,
) -> Dict[str, Any]:
    z = np.load(structure_npz, allow_pickle=True)
    st = {k: z[k] for k in z.files}
    required = [
        "box_nm",
        "theta_deg",
        "a_bottom_A",
        "a_top_A",
        "positions_initial_A",
        "positions_relaxed_A",
        "layer",
        "sublattice",
    ]
    missing = [k for k in required if k not in st]
    if missing:
        raise KeyError(f"structure_npz missing required keys: {missing}")

    box = float(st["box_nm"])
    th = float(st["theta_deg"])
    ab = float(st["a_bottom_A"])
    at = float(st["a_top_A"])
    Lm = moire_G(ab, at, th)
    # estimate moire period from G vectors
    g = float(np.mean(np.linalg.norm(Lm, axis=1))) if Lm.size else float("nan")
    estimated_moire_period_nm = 4.0 * math.pi / (math.sqrt(3) * g) / 10.0 if g > 0 else float("nan")
    box_to_moire_ratio = box / estimated_moire_period_nm if estimated_moire_period_nm > 0 else float("nan")

    warnings_list = []
    if box_to_moire_ratio < float(min_moire_cells):
        msg = (
            f"box_nm={box:.3f} nm is only {box_to_moire_ratio:.3f} moire periods "
            f"for theta={th:.4f} deg; L_moire={estimated_moire_period_nm:.3f} nm. "
            "Use a larger box for physical PI-QMC, or pass allow_small_box=True for toy maps."
        )
        warnings_list.append(msg)
        if not allow_small_box:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning)
        logger.warning(msg)

    x = np.linspace(-box / 2.0, box / 2.0, grid_n, endpoint=False)
    y = np.linspace(-box / 2.0, box / 2.0, grid_n, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")

    Vreg = registry_energy(
        X,
        Y,
        ab,
        at,
        th,
        registry_depth_meV / 1000.0,
        phase=phase_rad,
        registry_shift_nm=registry_shift_nm,
        invert=invert_registry,
        registry_norm_mode=registry_norm_mode,
    )

    bm = strain_maps(st, x, y, "bottom", k_neighbors)
    tm = strain_maps(st, x, y, "top", k_neighbors)
    if disable_deformation:
        Vdef = np.zeros_like(Vreg)
    else:
        Vdef = deformation_c_eV * bm["hydro"] - deformation_v_eV * tm["hydro"]

    d = interlayer_distance(st, x, y, k_neighbors)
    Vd = interlayer_alpha_eV_per_nm * (d - np.nanmean(d))

    V_raw = Vreg + Vdef + Vd
    V = V_raw - float(np.nanmin(V_raw))

    payload = {
        "x_nm": x,
        "y_nm": y,
        "V_eV": V,
        "V_raw_eV": V_raw,
        "V_registry_eV": Vreg,
        "V_deformation_eV": Vdef,
        "V_interlayer_eV": Vd,
        "interlayer_distance_nm": d,
        "hydro_bottom": bm["hydro"],
        "hydro_top": tm["hydro"],
        "eps_bottom_xx": bm["xx"],
        "eps_bottom_yy": bm["yy"],
        "eps_bottom_xy": bm["xy"],
        "eps_top_xx": tm["xx"],
        "eps_top_yy": tm["yy"],
        "eps_top_xy": tm["xy"],
        "theta_deg": th,
        "a_bottom_A": ab,
        "a_top_A": at,
        "box_nm": box,
        "grid_n": int(grid_n),
        "registry_depth_meV": float(registry_depth_meV),
        "registry_norm_mode": str(registry_norm_mode),
        "registry_shift_nm": np.asarray(registry_shift_nm if registry_shift_nm is not None else (0.0, 0.0), dtype=float),
        "deformation_c_eV": float(deformation_c_eV),
        "deformation_v_eV": float(deformation_v_eV),
        "disable_deformation": bool(disable_deformation),
        "interlayer_alpha_eV_per_nm": float(interlayer_alpha_eV_per_nm),
        "estimated_moire_period_nm": float(estimated_moire_period_nm),
        "box_to_moire_ratio": float(box_to_moire_ratio),
        "min_moire_cells_requested": float(min_moire_cells),
        "allow_small_box": bool(allow_small_box),
        "warnings": np.asarray(warnings_list, dtype=str),
        "source_structure_npz": str(structure_npz),
    }
    outp = save_npz_with_meta(out_npz, payload, {"theta_deg": th, "box_nm": box, "grid_n": grid_n})
    return payload

