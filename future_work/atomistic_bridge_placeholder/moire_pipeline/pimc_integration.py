# moire_pipeline/pimc_integration.py
from __future__ import annotations
from pathlib import Path
import logging
from .potential import GridPotential2D

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_pimc_grid_potential(
    potential_npz: str | Path,
    Ex: float = 0.0,
    k_env_eV_per_nm2: float = 2 * 0.010 / 30.0**2,
    include_envelope: bool = True,
    include_coulomb: bool = False,
    coulomb_strength_eV_nm: float = 0.15,
    coulomb_softening_nm: float = 1.0,
    periodic: bool = True,
):
    """Build a tmd_pimc-compatible potential from a V_grid.npz file.

    Returns either a CompositePotential from tmd_pimc (if available) or a GridPotential2D.
    """
    grid = GridPotential2D.from_npz(potential_npz, periodic=periodic, subtract_minimum=True)
    try:
        from tmd_pimc import HarmonicEnvelopePotential, ExternalFieldPotential, SoftCoulombPotential, CompositePotential

        terms = []
        if include_envelope and k_env_eV_per_nm2:
            terms.append(HarmonicEnvelopePotential(k_env_eV_per_nm2=k_env_eV_per_nm2))
        terms.append(grid)
        if Ex != 0.0:
            terms.append(ExternalFieldPotential(E=(Ex, 0.0)))
        if include_coulomb:
            terms.append(SoftCoulombPotential(strength_eV_nm=coulomb_strength_eV_nm, softening_nm=coulomb_softening_nm))
        logger.info("Built CompositePotential with %d terms", len(terms))
        return CompositePotential(terms) if len(terms) > 1 else terms[0]
    except Exception as exc:
        logger.exception("tmd_pimc not available or failed to build CompositePotential: %s", exc)
        logger.info("Returning raw GridPotential2D as fallback")
        return grid

