"""Shared model/observable definitions for the PRB confined-double-well rerun.

Scientific change relative to the historical PRB scripts:
    V_total(R) = V_doubleGaussian(R) + V_wall(R) - q_eff E_x X
    V_wall(R)  = Vw (|R|/Rw)^8,
with Vw=80 meV and Rw=25 nm.

The r^8 wall is an auxiliary thermodynamic regulator, not a microscopic force.
It is negligible in the two-well region but dominates any finite linear tilt at
large radius, making the Hamiltonian bounded below and the canonical partition
function finite on R^2.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar

from tmd_pimc import (
    RingPolymerAction,
    PIMCSamplerStaging,
    DoubleGaussianWellPotential,
    SoftWallBoxPotential,
    ExternalFieldPotential,
    CompositePotential,
)

MASS_M0 = 0.5
T_K = 20.0
SEPARATION_NM = 10.0
SIGMA_NM = 3.0
V0_EV = 0.05
Q_EFF = 1.0
WALL_R_NM = 25.0
WALL_V0_EV = 0.08
WALL_POWER = 8

N_STEPS = 60000
BURN_IN = 15000
SAMPLE_EVERY = 20


def build_potential(Ex_eV_per_nm=0.0, asymmetry_eV=0.0,
                    wall_R_nm=WALL_R_NM, wall_V0_eV=WALL_V0_EV,
                    wall_power=WALL_POWER):
    dw = DoubleGaussianWellPotential(
        V0_eV=V0_EV, sigma_nm=SIGMA_NM,
        separation_nm=SEPARATION_NM, asymmetry_eV=asymmetry_eV,
    )
    wall = SoftWallBoxPotential(
        R_box_nm=wall_R_nm, V_wall0_eV=wall_V0_eV, power=wall_power,
    )
    field = ExternalFieldPotential(E=(Ex_eV_per_nm, 0.0), q_eff=Q_EFF)
    return CompositePotential(terms=[dw, wall, field])


def make_sampler(P, seed, Ex_eV_per_nm=0.0, asymmetry_eV=0.0,
                 wall_R_nm=WALL_R_NM,
                 staging_length=None):
    potential = build_potential(Ex_eV_per_nm, asymmetry_eV, wall_R_nm)
    action = RingPolymerAction(
        mass_m0=MASS_M0, temperature_K=T_K, n_beads=int(P), potential=potential,
    )
    if staging_length is None:
        staging_length = 16 if int(P) == 32 else 32
    staging_length = min(int(staging_length), int(P)-1)
    sampler = PIMCSamplerStaging(
        action=action,
        local_step_nm=0.20*np.sqrt(32.0/int(P)),
        global_step_nm=1.0,
        global_move_probability=0.20,
        rng_seed=int(seed),
        staging_segment_lengths=(staging_length,),
        staging_moves_per_step=2,
        perform_local_sweep=True,
    )
    return potential, action, sampler


def summarize_samples(samples, potential, thresholds=(0.05, 0.10, 0.15, 0.20)):
    samples = np.asarray(samples)
    cents = samples.mean(axis=1)
    x = samples[:, :, 0]
    p_centroid = float(np.mean(cents[:, 0] > 0.0))
    p_bead = float(np.mean(x > 0.0))
    frac_left = np.mean(x < 0.0, axis=1)
    span = {
        float(t): float(np.mean((frac_left > t) & (frac_left < 1.0-t)))
        for t in thresholds
    }
    delta = samples - cents[:, None, :]
    rg2 = float(np.mean(np.sum(delta*delta, axis=2)))
    V = potential.value(samples.reshape(-1,2)).reshape(samples.shape[0],samples.shape[1])
    return {
        'p_centroid': p_centroid,
        'p_bead': p_bead,
        'span_05': span.get(0.05, float('nan')),
        'span_10': span.get(0.10, float('nan')),
        'span_15': span.get(0.15, float('nan')),
        'span_20': span.get(0.20, float('nan')),
        'Rg2_nm2': rg2,
        'Vmean_eV': float(V.mean()),
        'mean_x_centroid_nm': float(cents[:,0].mean()),
        'mean_y_centroid_nm': float(cents[:,1].mean()),
        'max_bead_radius_nm': float(np.sqrt(np.max(np.sum(samples*samples,axis=2)))),
    }


def local_minima_zero_field(asymmetry_eV=0.0, wall_R_nm=WALL_R_NM):
    pot = build_potential(0.0, asymmetry_eV, wall_R_nm)
    def v1(x):
        return float(pot.value(np.array([[x,0.0]],dtype=float))[0])
    left = minimize_scalar(v1,bounds=(-8.0,-2.0),method='bounded',options={'xatol':1e-13})
    right = minimize_scalar(v1,bounds=(2.0,8.0),method='bounded',options={'xatol':1e-13})
    return {
        'xA_nm': float(left.x), 'EA_eV': float(left.fun),
        'xB_nm': float(right.x), 'EB_eV': float(right.fun),
        'Delta0_eV': float(left.fun-right.fun),
    }


def minimum_to_minimum_crossing(asymmetry_eV=0.0, wall_R_nm=WALL_R_NM):
    m = local_minima_zero_field(asymmetry_eV, wall_R_nm)
    dx = m['xB_nm'] - m['xA_nm']
    return -m['Delta0_eV']/(Q_EFF*dx), m
