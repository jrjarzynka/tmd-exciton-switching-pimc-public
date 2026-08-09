"""
Regression test: the electron-hole interaction must not be minimum-imaged.

The periodic two-body sampler wraps the LANDSCAPE lookup into one moire cell
using fractional lattice coordinates, while the electron-hole interaction is
evaluated from the raw Cartesian separation. That split is deliberate: the
moire landscape is a crystal property and IS periodic, but there is one
electron and one hole in an infinite plane, so the interaction is NOT. Wrapping
rho would replace a single exciton in a moire potential with a lattice of
excitons, one per moire cell.

What minimum imaging would actually look like
---------------------------------------------
Not faster binding, which is the intuitive guess and is wrong. Under wrapping,
a separation of exactly one lattice vector maps to rho = 0, the DEEPEST point
of the bilayer Keldysh well (-253 meV against -14 meV for the true separation).
So rho = L would become a stable minimum and the pair would sit there:

      rho (nm)     V_true (meV)     V_wrapped (meV)
        15            -19.2            -54.0
        18            -16.1           -112.6
        20            -14.5           -252.6     <- global minimum
        22            -13.2           -112.6

The signature is therefore DWELLING at rho ~ L, and this test asserts its
absence. That criterion is what makes the test stable. Starting a pair one
lattice vector apart sets off a race between early collapse and early escape
whose winner depends on the random stream: observed outcomes range from
<rho^2> = 2.7 nm^2 to 1118 nm^2 across bead counts and run lengths. Both
falsify wrapping, so a threshold on <rho^2> would be measuring the race rather
than the physics, while a threshold on dwelling is insensitive to it.

The samples themselves are always raw unwrapped coordinates, so the kernel's
internal convention can only be inferred from the dynamics it produces. That is
why this test runs the sampler rather than inspecting a table.
"""

import json
import os

import numpy as np
import pytest

from tmd_pimc import (
    CompositePotential,
    TwoBodyRingPolymerAction,
    pair_separations,
)
from tmd_pimc.bilayer_keldysh_potential import (
    BilayerKeldyshTablePotential,
    bilayer_keldysh_value_eV,
)

pytest.importorskip("numba", reason="periodic two-body sampler is JIT-backed")
from tmd_pimc import TwoBodyPIMCSamplerStagingPeriodicJIT  # noqa: E402


CONFIG = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "configs", "two_body", "dense_shift0000.json",
)

N_STEPS = 1500
BURN_IN = 300
N_BEADS = 32
SEEDS = (7, 11, 23)

# A separation counts as "at the lattice vector" within this fractional band.
DWELL_BAND = 0.15
# Under wrapping the pair is pinned there, so the dwelling fraction would be
# near 1.0. Under the correct interaction it merely passes through. The gap is
# wide enough that this threshold needs no tuning.
MAX_DWELL_FRACTION = 0.5


def _load_config():
    with open(CONFIG) as fh:
        return json.load(fh)


def _interaction(cfg, r_max_nm):
    kw = dict(
        separation_nm=cfg["separation_nm"],
        screening_length_layer1_nm=cfg["screening_length_layer1_nm"],
        screening_length_layer2_nm=cfg["screening_length_layer2_nm"],
        kappa_environment=cfg["kappa_environment"],
    )
    # The table must start exactly at r = 0, which is well defined here: the
    # bilayer Keldysh form goes through sqrt(rho^2 + D^2) with D the interlayer
    # separation, so it stays finite at zero in-plane separation.
    r = np.linspace(0.0, r_max_nm, 20001)
    return BilayerKeldyshTablePotential(
        r_nm=r, V_eV=bilayer_keldysh_value_eV(r, **kw), **kw
    )


def _run(cfg, center_h, r_max_nm, seed):
    zero = CompositePotential(terms=[])
    action = TwoBodyRingPolymerAction(
        mass_e_m0=cfg["mass_e_m0"],
        mass_h_m0=cfg["mass_h_m0"],
        temperature_K=cfg["temperature_K"],
        n_beads=N_BEADS,
        # The periodic sampler builds its own landscape grids from
        # moire_period_nm / moire_amplitude_eV and never reads these, so a null
        # landscape here keeps the test about the interaction alone.
        potential_e=zero,
        potential_h=zero,
        potential_interaction=_interaction(cfg, r_max_nm),
    )
    sampler = TwoBodyPIMCSamplerStagingPeriodicJIT(
        action=action,
        moire_period_nm=cfg["moire_period_nm"],
        moire_amplitude_eV=cfg["moire_amplitude_eV"],
        local_step_nm=cfg["local_step_nm"],
        global_step_nm=cfg["global_step_nm"],
        interaction_table_r_max_nm=r_max_nm,
        rng_seed=seed,
    )
    out = sampler.run(
        n_steps=N_STEPS, burn_in=BURN_IN, sample_every=20,
        center_e=(0.0, 0.0), center_h=center_h,
    )
    return pair_separations(out["samples_e"], out["samples_h"])


def test_pair_does_not_dwell_at_one_lattice_vector():
    cfg = _load_config()
    L = cfg["moire_period_nm"]
    r_max = 6.0 * L

    fractions = []
    for seed in SEEDS:
        rho = _run(cfg, (L, 0.0), r_max, seed)
        assert float(rho.max()) < r_max, (
            f"seed {seed}: separations reached {float(rho.max()):.1f} nm against a "
            f"table cutoff of {r_max:.1f} nm; widen the table before trusting this."
        )
        fractions.append(float(np.mean(np.abs(rho - L) < DWELL_BAND * L)))

    worst = max(fractions)
    assert worst < MAX_DWELL_FRACTION, (
        f"The pair dwelt within {DWELL_BAND:.0%} of one lattice vector for "
        f"{worst:.0%} of samples (per-seed: "
        f"{', '.join(f'{f:.0%}' for f in fractions)}). Under minimum imaging a "
        f"separation of exactly L wraps to zero, the deepest point of the "
        f"interaction, so the pair becomes pinned there. That is the signature "
        f"of a wrapped electron-hole interaction."
    )


def test_bound_control_stays_bound():
    """Sanity control: a pair started together must remain bound.

    Without this, the test above could pass simply because the sampler is
    broken and the pair wanders everywhere regardless of the interaction.
    """
    cfg = _load_config()
    rho = _run(cfg, (0.0, 0.0), 6.0 * cfg["moire_period_nm"], SEEDS[0])
    r2 = float(np.mean(rho ** 2))
    assert r2 < 25.0, (
        f"Pair started together did not stay bound: <rho^2> = {r2:.2f} nm^2"
    )
