"""Smoke and periodic-cover checks for the integrated COM staging sampler."""
import numpy as np

from tmd_pimc import GridPotential2D, RingPolymerAction, PIMCSamplerStagingPeriodicJIT
from tmd_pimc.kernels_staging_jit import periodic_grid_value, staging_delta_potential_jit
from tmd_pimc.staging import free_bridge_proposal, segment_indices


def fixture():
    n=48; axis=np.arange(n,dtype=float)/n; u,v=np.meshgrid(axis,axis,indexing="ij")
    values=.02*(np.cos(2*np.pi*u)+.3*np.sin(2*np.pi*v)+.2*np.cos(2*np.pi*(u+v)))
    lattice=np.array([[7.4,36.1],[-27.6,24.45]]); origin=np.array([1.5,-2.0])
    potential=GridPotential2D(axis,axis,values,periodic=True,subtract_minimum=False,
                              lattice_vectors_nm=lattice,origin_nm=origin)
    return potential,values,lattice,origin


def test_staging_delta_uses_only_periodic_external_potential():
    potential,grid,lattice,origin=fixture(); action=RingPolymerAction(.5,5.,80,potential)
    matrix=np.column_stack([lattice[0],lattice[1]]); inverse=np.linalg.inv(matrix)
    beads=np.arange(80,dtype=float); path=origin+np.column_stack((.2*beads,.13*beads))@matrix.T
    start=71; length=16; z=np.random.default_rng(71).normal(size=(length-1,2))
    new=free_bridge_proposal(path,start,length,z,2*action._lambda_x*action._tau)
    idx=segment_indices(start,length,80)[1:-1]
    expected=action._tau*np.sum(potential.value(new[idx])-potential.value(path[idx]))
    actual=staging_delta_potential_jit(path,new,start,length,grid,origin[0],origin[1],
                                       inverse[0,0],inverse[0,1],inverse[1,0],inverse[1,1],action._tau)
    np.testing.assert_allclose(actual,expected,rtol=0.0,atol=1e-11)
    # Proposed coordinates remain unwrapped over many real-space cells.
    assert np.max(np.abs(new))>np.max(np.abs(lattice))


def test_integrated_sampler_supports_local_staging_and_global_moves():
    potential,_,lattice,origin=fixture(); action=RingPolymerAction(.5,5.,80,potential)
    center=origin+np.column_stack([lattice[0],lattice[1]])@np.array([1/3,1/3])
    sampler=PIMCSamplerStagingPeriodicJIT(action,local_step_nm=.5,global_step_nm=3.,
        global_move_probability=.2,rng_seed=91821,staging_segment_lengths=(4,8,16,32),
        staging_moves_per_step=1)
    result=sampler.run(n_steps=300,burn_in=100,sample_every=20,center=center)
    assert result["samples"].shape==(10,80,2)
    assert 0<=result["acceptance_local"]<=1
    assert 0<=result["acceptance_staging"]<=1
    assert 0<=result["acceptance_global"]<=1
    assert result["staging_length_attempts"].sum()==200
    assert np.all(result["staging_length_accepted"]<=result["staging_length_attempts"])
