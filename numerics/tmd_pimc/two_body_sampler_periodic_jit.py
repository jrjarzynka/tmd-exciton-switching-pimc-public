"""Periodic-cell two-body sampler with an out-of-plane (Fz) Stark coupling
and independently parameterised electron and hole landscapes.

BACKGROUND (why this file changed): an earlier version of this sampler
only supported an in-plane driving field, V(r) = -q_eff*(E . r), added
analytically outside the periodic landscape. That field has no physical
grounding in this project's own COM formalism, which drives relocation
with an OUT-OF-PLANE field Fz coupled through a registry-dependent
effective interlayer dipole, V_Stark(r;Fz) = -Fz*dipole_length_nm*basis(r),
via OutOfPlaneStarkPotential in potentials.py. Crucially, V_Stark has the
SAME lattice periodicity as the registry potential itself (same G1,G2,G3),
so -- unlike the old in-plane linear force -- it can be rasterized into the
SAME periodic cell as the registry landscape, using the existing,
already-validated build_periodic_cell_grid machinery unchanged.

Sign convention for the two-body generalization: OutOfPlaneStarkPotential's
own docstring ties its sign to ExternalFieldPotential's q_eff convention
("positive Fz pulls the exciton toward the anchor's dipole-favoured
basin, matching ... -q_eff*(r.E)"), so the natural two-body split mirrors
the electron/hole q_eff=-1/+1 split: the electron's Stark term uses
-Fz_eV_per_nm, the hole's uses +Fz_eV_per_nm, each evaluated at that
layer's OWN registry anchor by default.

CHANGE (v1.1): per-carrier landscape parameters
-----------------------------------------------
Previously a single `moire_amplitude_eV` built one `bare_moire`, which was
then rigidly shifted to produce both V_e and V_h. The two landscapes
therefore differed ONLY by a registry offset: identical amplitude,
identical period, identical shape. Likewise a single `dipole_length_nm`
was shared by the electron and hole Stark terms, so the dipole texture
differed only in sign.

That is an idealisation, not a physical result. Stage 0 DFT for
WSe2/MoSe2 indicates that the electron and hole landscapes have DIFFERENT
registry amplitudes (the relevant quantity being the gap modulation across
the moire cell, not a common GSFE-derived V0), and that the hole landscape
in particular depends critically on spin-orbit coupling. A model in which
V_e and V_h share an amplitude carries a residual symmetry between the two
carriers, and any result whose interpretation depends on the ABSENCE of
electron-hole asymmetry -- for instance a dissociation threshold found to
be independent of V0 -- must be re-checked with the amplitudes decoupled
before that independence can be called physical.

This version therefore accepts, for both the registry amplitude and the
Stark dipole length, either a single shared value (previous behaviour) or
one value per carrier. The moire PERIOD remains shared by construction:
both carriers live in the same moire superlattice, and the periodic-cell
rasterisation assumes a single set of lattice vectors.

Backward compatibility
----------------------
- Fz_eV_per_nm=0.0 (the default) reproduces the pure-registry behaviour
  bit-for-bit, since the Stark term then contributes exactly zero.
- Passing only `moire_amplitude_eV` and `dipole_length_nm`, as before,
  reproduces the previous behaviour bit-for-bit.
- The per-carrier arguments default to None, meaning "fall back to the
  shared value".

Use `describe_landscape()` to confirm which mode a given run is in; a
symmetric model is a legitimate choice but should be a deliberate one.
"""

import numpy as np

from .two_body_kernels_jit import build_uniform_interaction_table
from .two_body_kernels_periodic_jit import (
    build_periodic_cell_grid,
    run_pimc_core_two_body_periodic_staging_jit,
)


def field_coefficient(E_vec, q_eff):
    """c = -q_eff * E, linear coefficient for V_field(r) = c . r.

    Retained for the (no longer default) in-plane driving-field path; see
    module docstring for why the out-of-plane Fz path below is now
    preferred for anything connecting to this project's own COM formalism.
    """
    return (-q_eff * float(E_vec[0]), -q_eff * float(E_vec[1]))


def _resolve_per_carrier(shared, value_e, value_h, name):
    """Resolve a parameter that may be given once or once per carrier.

    Returns (value_e, value_h). A per-carrier value of None falls back to
    `shared`. Raises if `shared` is also None, so a missing parameter fails
    loudly rather than silently defaulting.
    """
    resolved_e = shared if value_e is None else value_e
    resolved_h = shared if value_h is None else value_h
    if resolved_e is None or resolved_h is None:
        raise ValueError(
            f"{name}: supply either the shared value or both per-carrier values"
        )
    return float(resolved_e), float(resolved_h)


class TwoBodyPIMCSamplerStagingPeriodicJIT:
    def __init__(
        self,
        action,
        moire_period_nm,
        moire_amplitude_eV=None,
        moire_amplitude_e_eV=None,
        moire_amplitude_h_eV=None,
        origin_e_nm=(0.0, 0.0),
        origin_h_nm=(0.0, 0.0),
        field_e_eV_per_nm=(0.0, 0.0),
        field_h_eV_per_nm=(0.0, 0.0),
        Fz_eV_per_nm=0.0,
        dipole_length_nm=0.05,
        dipole_length_e_nm=None,
        dipole_length_h_nm=None,
        stark_period_nm=None,
        stark_anchor_e_nm=None,
        stark_anchor_h_nm=None,
        stark_phase_rad=0.0,
        stark_normalisation="harmonic",
        local_step_nm=0.20,
        global_step_nm=1.00,
        global_move_probability=0.20,
        rng_seed=1234,
        staging_segment_lengths=(4, 8, 16, 32, 64, 128, 256),
        staging_moves_per_step=2,
        perform_local_sweep=True,
        interaction_table_r_max_nm=80.0,
        interaction_table_n_points=20000,
        periodic_cell_grid_size=200,
    ):
        self.action = action
        self.local_step_nm = float(local_step_nm)
        self.global_step_nm = float(global_step_nm)
        self.global_move_probability = float(global_move_probability)
        self.rng_seed = int(rng_seed)
        self.staging_moves_per_step = int(staging_moves_per_step)
        self.perform_local_sweep = bool(perform_local_sweep)
        # In-plane field path retained for backward compatibility only;
        # not used when Fz_eV_per_nm != 0 (both mechanisms could in
        # principle be combined, but that has not been validated -- avoid
        # setting both nonzero without checking the physics makes sense).
        self.field_e = tuple(float(v) for v in field_e_eV_per_nm)
        self.field_h = tuple(float(v) for v in field_h_eV_per_nm)
        self.Fz_eV_per_nm = float(Fz_eV_per_nm)

        self.moire_period_nm = float(moire_period_nm)
        self.moire_amplitude_e_eV, self.moire_amplitude_h_eV = _resolve_per_carrier(
            moire_amplitude_eV, moire_amplitude_e_eV, moire_amplitude_h_eV,
            "moire amplitude",
        )
        self.dipole_length_e_nm, self.dipole_length_h_nm = _resolve_per_carrier(
            dipole_length_nm, dipole_length_e_nm, dipole_length_h_nm,
            "dipole length",
        )
        # Retained so that existing code reading `.dipole_length_nm` keeps
        # working in the symmetric case; None signals an asymmetric model,
        # where a single shared value would be misleading.
        self.dipole_length_nm = (
            self.dipole_length_e_nm
            if self.dipole_length_e_nm == self.dipole_length_h_nm
            else None
        )

        P = int(action.n_beads)
        if P < 3:
            raise ValueError("n_beads must be >= 3")

        lengths = sorted({int(L) for L in staging_segment_lengths if 2 <= int(L) < P})
        if not lengths:
            lengths = [P - 1]
        self._valid_segment_lengths = np.asarray(lengths, dtype=np.int64)

        self.rng = np.random.default_rng(self.rng_seed)

        v_table, r_min, r_max = build_uniform_interaction_table(
            action.potential_interaction,
            r_max_nm=interaction_table_r_max_nm,
            n_points=interaction_table_n_points,
        )
        self._v_int_table = v_table
        self._r_int_min = r_min
        self._r_int_max = r_max

        from .potentials import MoirePotential, OutOfPlaneStarkPotential, CompositePotential
        from .potential_helpers import ShiftedPotential

        # One MoirePotential per carrier. In the symmetric case these are
        # two equal-valued objects, which is numerically identical to the
        # single shared object used previously.
        bare_moire_e = MoirePotential(
            amplitude_eV=self.moire_amplitude_e_eV, period_nm=self.moire_period_nm
        )
        bare_moire_h = MoirePotential(
            amplitude_eV=self.moire_amplitude_h_eV, period_nm=self.moire_period_nm
        )

        # BUGFIX (2026-08-02): origin_e_nm/origin_h_nm used to be forwarded
        # straight into build_periodic_cell_grid's `origin_nm` argument,
        # which only relocates the rasterization *window*. Because the moire
        # potential is exactly periodic under the same lattice vectors used
        # for the periodic wrap, moving the rasterization window and then
        # wrapping the query point with that same window cancels out
        # completely -- verified numerically (differences at the ~1e-6 eV
        # interpolation-noise level, vs ~0.04 eV, i.e. order amplitude, for a
        # genuine shift). Net effect: the electron and hole were being
        # sampled on the IDENTICAL, unshifted moire registry regardless of
        # origin_h_nm, silently invalidating any "registry offset" result
        # produced this way (only the OutOfPlaneStarkPotential anchor, which
        # bakes its shift directly into value(r) and is therefore immune to
        # this cancellation, was ever actually shifted).
        #
        # Fix: bake the registry shift into the potential itself via
        # ShiftedPotential (evaluates inner.value(r - shift), a genuine
        # query-point shift) BEFORE rasterizing, then rasterize with
        # origin_nm=(0,0) so the window-placement trick is never relied on
        # for anything except window placement. Stark terms are NOT
        # wrapped again here since their own anchor_nm already performs the
        # equivalent shift internally; wrapping the composite a second time
        # would double-shift the Stark phase.
        registry_e_nm = tuple(origin_e_nm)
        registry_h_nm = tuple(origin_h_nm)
        moire_e = (ShiftedPotential(inner=bare_moire_e, shift_nm=registry_e_nm)
                   if registry_e_nm != (0.0, 0.0) else bare_moire_e)
        moire_h = (ShiftedPotential(inner=bare_moire_h, shift_nm=registry_h_nm)
                   if registry_h_nm != (0.0, 0.0) else bare_moire_h)

        if self.Fz_eV_per_nm != 0.0:
            _stark_period = (
                float(stark_period_nm) if stark_period_nm is not None
                else self.moire_period_nm
            )
            _anchor_e = tuple(stark_anchor_e_nm) if stark_anchor_e_nm is not None else registry_e_nm
            _anchor_h = tuple(stark_anchor_h_nm) if stark_anchor_h_nm is not None else registry_h_nm

            stark_e = OutOfPlaneStarkPotential(
                Fz_eV_per_nm=-self.Fz_eV_per_nm,  # q_eff = -1 (electron), matches
                                                   # ExternalFieldPotential convention
                dipole_length_nm=self.dipole_length_e_nm,
                period_nm=_stark_period,
                phase_rad=float(stark_phase_rad),
                anchor_nm=_anchor_e,
                normalisation=stark_normalisation,
            )
            stark_h = OutOfPlaneStarkPotential(
                Fz_eV_per_nm=+self.Fz_eV_per_nm,  # q_eff = +1 (hole)
                dipole_length_nm=self.dipole_length_h_nm,
                period_nm=_stark_period,
                phase_rad=float(stark_phase_rad),
                anchor_nm=_anchor_h,
                normalisation=stark_normalisation,
            )
            potential_e_for_grid = CompositePotential(terms=[moire_e, stark_e])
            potential_h_for_grid = CompositePotential(terms=[moire_h, stark_h])
            self._stark_e = stark_e  # kept for introspection/testing
            self._stark_h = stark_h
        else:
            potential_e_for_grid = moire_e
            potential_h_for_grid = moire_h
            self._stark_e = None
            self._stark_h = None

        self._potential_e_for_grid = potential_e_for_grid
        self._potential_h_for_grid = potential_h_for_grid

        self._grid_e = build_periodic_cell_grid(
            potential_e_for_grid, self.moire_period_nm,
            origin_nm=(0.0, 0.0), grid_size=periodic_cell_grid_size,
        )
        self._grid_h = build_periodic_cell_grid(
            potential_h_for_grid, self.moire_period_nm,
            origin_nm=(0.0, 0.0), grid_size=periodic_cell_grid_size,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def describe_landscape(self):
        """Report how the electron and hole landscapes differ.

        Answers, for a configured sampler, the question "are V_e and V_h
        genuinely two different landscapes, or the same one shifted?".
        `landscape_symmetric` is True when the two carriers see potentials
        of identical amplitude and dipole length, so that the only
        electron-hole asymmetry in the model comes from the registry
        offset and the mass ratio.
        """
        max_abs_diff = float(
            np.max(np.abs(self._grid_e["v_grid"] - self._grid_h["v_grid"]))
        )
        return {
            "moire_period_nm": self.moire_period_nm,
            "moire_amplitude_e_eV": self.moire_amplitude_e_eV,
            "moire_amplitude_h_eV": self.moire_amplitude_h_eV,
            "amplitude_ratio_h_over_e": (
                self.moire_amplitude_h_eV / self.moire_amplitude_e_eV
                if self.moire_amplitude_e_eV != 0.0 else float("nan")
            ),
            "dipole_length_e_nm": self.dipole_length_e_nm,
            "dipole_length_h_nm": self.dipole_length_h_nm,
            "Fz_eV_per_nm": self.Fz_eV_per_nm,
            "stark_active": self._stark_e is not None,
            "landscape_symmetric": (
                self.moire_amplitude_e_eV == self.moire_amplitude_h_eV
                and self.dipole_length_e_nm == self.dipole_length_h_nm
            ),
            "max_abs_grid_difference_eV": max_abs_diff,
        }

    def initialize_paths(self, center_e=(0.0, 0.0), center_h=(0.0, 0.0), spread_nm=0.1):
        P = int(self.action.n_beads)
        path_e = np.array(center_e, dtype=np.float64) + spread_nm * self.rng.standard_normal((P, 2))
        path_h = np.array(center_h, dtype=np.float64) + spread_nm * self.rng.standard_normal((P, 2))
        return path_e, path_h

    def run(self, n_steps=10000, burn_in=2500, sample_every=20, center_e=(0.0, 0.0), center_h=(0.0, 0.0)):
        path_e, path_h = self.initialize_paths(center_e=center_e, center_h=center_h)

        (
            samples_e, samples_h,
            acc_local_e, acc_local_h, acc_staging, acc_global,
        ) = run_pimc_core_two_body_periodic_staging_jit(
            n_steps=int(n_steps), burn_in=int(burn_in), sample_every=int(sample_every),
            p_beads=int(self.action.n_beads),
            path_e=path_e, path_h=path_h,
            kpf_e=float(self.action._kpf_e), kpf_h=float(self.action._kpf_h), tau=float(self.action._tau),
            v_int_table=self._v_int_table, r_int_min=self._r_int_min, r_int_max=self._r_int_max,
            v_grid_e=self._grid_e["v_grid"],
            origin_e_x=self._grid_e["origin_x"], origin_e_y=self._grid_e["origin_y"],
            ainv_e00=self._grid_e["ainv00"], ainv_e01=self._grid_e["ainv01"],
            ainv_e10=self._grid_e["ainv10"], ainv_e11=self._grid_e["ainv11"],
            field_e_x=self.field_e[0], field_e_y=self.field_e[1],
            v_grid_h=self._grid_h["v_grid"],
            origin_h_x=self._grid_h["origin_x"], origin_h_y=self._grid_h["origin_y"],
            ainv_h00=self._grid_h["ainv00"], ainv_h01=self._grid_h["ainv01"],
            ainv_h10=self._grid_h["ainv10"], ainv_h11=self._grid_h["ainv11"],
            field_h_x=self.field_h[0], field_h_y=self.field_h[1],
            local_step_nm=self.local_step_nm, global_step_nm=self.global_step_nm,
            global_move_prob=self.global_move_probability,
            staging_segment_lengths=self._valid_segment_lengths,
            staging_moves_per_step=self.staging_moves_per_step,
            perform_local_sweep=self.perform_local_sweep,
            seed=self.rng_seed,
        )

        return {
            "samples_e": samples_e,
            "samples_h": samples_h,
            "acceptance_local_e": float(acc_local_e),
            "acceptance_local_h": float(acc_local_h),
            "acceptance_staging": float(acc_staging),
            "acceptance_global_joint": float(acc_global),
            "n_samples": int(samples_e.shape[0]),
        }
