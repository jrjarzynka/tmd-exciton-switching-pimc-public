"""Paper-1 single-body periodic sampler using reusable staging primitives.

This wrapper is intentionally a COM/single-body consumer, not the definition
of the staging algorithm; explicit e-h samplers should reuse the bridge layer.
"""
from __future__ import annotations

import numpy as np
from .kernels_staging_jit import run_periodic_staging_jit


class PIMCSamplerStagingPeriodicJIT:
    """Numba COM sampler for a periodic ``GridPotential2D`` action."""

    def __init__(self, action, local_step_nm=0.20, global_step_nm=1.0,
                 global_move_probability=0.20, rng_seed=1234,
                 staging_segment_lengths=(4,8,16,32),
                 staging_moves_per_step=1):
        self.action=action; self.local_step_nm=float(local_step_nm)
        self.global_step_nm=float(global_step_nm)
        self.global_move_probability=float(global_move_probability)
        self.rng_seed=int(rng_seed); self.staging_moves_per_step=int(staging_moves_per_step)
        p=int(action.n_beads)
        lengths=sorted({int(x) for x in staging_segment_lengths if 2<=int(x)<p})
        if not lengths: raise ValueError("no valid staging segment length")
        if self.staging_moves_per_step<1: raise ValueError("staging_moves_per_step must be >= 1")
        if not 0<=self.global_move_probability<=1: raise ValueError("global_move_probability must be in [0,1]")
        potential=action.potential
        required=("_V","_origin","_Ainv")
        if not all(hasattr(potential,x) for x in required):
            raise TypeError("periodic JIT staging requires GridPotential2D-compatible _V/_origin/_Ainv")
        self.segment_lengths=np.asarray(lengths,dtype=np.int64)
        self._grid=np.ascontiguousarray(potential._V,dtype=np.float64)
        self._origin=np.asarray(potential._origin,dtype=np.float64)
        self._ainv=np.asarray(potential._Ainv,dtype=np.float64)
        self._rng=np.random.default_rng(self.rng_seed)

    def initialize_path(self,center=(0.0,0.0),spread_nm=0.1):
        return np.asarray(center,dtype=np.float64)+spread_nm*self._rng.standard_normal((self.action.n_beads,2))

    def run(self,n_steps=10000,burn_in=2500,sample_every=20,center=(0.0,0.0)):
        path=self.initialize_path(center)
        values=run_periodic_staging_jit(n_steps,burn_in,sample_every,path,self._grid,
            self._origin[0],self._origin[1],self._ainv[0,0],self._ainv[0,1],
            self._ainv[1,0],self._ainv[1,1],self.action._kpf,self.action._tau,
            self.action._lambda_x,self.local_step_nm,self.global_step_nm,
            self.global_move_probability,self.segment_lengths,
            self.staging_moves_per_step,self.rng_seed)
        samples,local,staging,global_acc,attempts,accepted=values
        return {"samples":samples,"acceptance_local":float(local),
                "acceptance_staging":float(staging),"acceptance_global":float(global_acc),
                "staging_segment_lengths":self.segment_lengths.copy(),
                "staging_length_attempts":attempts,"staging_length_accepted":accepted,
                "n_samples":int(samples.shape[0])}
