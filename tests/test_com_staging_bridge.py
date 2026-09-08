"""Mathematical and backend-equivalence tests for COM staging proposals."""
import numpy as np
import pytest

from tmd_pimc.staging import (
    bridge_conditional_moments,
    bridge_log_density,
    free_bridge_proposal,
    segment_indices,
)
from tmd_pimc.kernels_staging_jit import free_bridge_proposal_jit


P=80
FREE_LINK_VARIANCE=4.419


@pytest.mark.parametrize("length",[4,8,16,32])
@pytest.mark.parametrize("start",[7,73,79])
def test_python_jit_common_random_bridge(length,start):
    beads=np.arange(P,dtype=np.float64)
    path=np.column_stack((13*np.sin(.19*beads)+.7*beads,
                          -9*np.cos(.11*beads)-.4*beads))
    path += np.array([73.0,-58.0])  # several lattice-cell spans
    z=np.random.default_rng(1000+length+start).standard_normal((length-1,2))
    python=free_bridge_proposal(path,start,length,z,FREE_LINK_VARIANCE)
    jit=free_bridge_proposal_jit(path,start,length,z,FREE_LINK_VARIANCE)
    np.testing.assert_allclose(jit,python,rtol=0.0,atol=1e-12)
    indices=segment_indices(start,length,P)
    np.testing.assert_array_equal(python[indices[[0,-1]]],path[indices[[0,-1]]])


@pytest.mark.parametrize("length",[4,8,16,32])
def test_bridge_linear_map_matches_conditioned_covariance(length):
    path=np.zeros((P,2)); path[0]=[3.25,-7.5]; path[length]=[31.0,12.75]
    mean,cov=bridge_conditional_moments(path[0],path[length],length,FREE_LINK_VARIANCE)
    zero=np.zeros((length-1,2)); observed_mean=free_bridge_proposal(path,0,length,zero,FREE_LINK_VARIANCE)[1:length]
    transform=np.empty((length-1,length-1))
    for q in range(length-1):
        z=np.zeros((length-1,2)); z[q,0]=1.0
        transform[:,q]=free_bridge_proposal(path,0,length,z,FREE_LINK_VARIANCE)[1:length,0]-mean[:,0]
    np.testing.assert_allclose(observed_mean,mean,rtol=0.0,atol=1e-12)
    np.testing.assert_allclose(transform@transform.T,cov,rtol=0.0,atol=1e-12)


@pytest.mark.parametrize("length",[4,8,16,32])
def test_hastings_spring_cancellation(length):
    rng=np.random.default_rng(401+length); path=rng.normal(size=(P,2)); start=P-length//2
    new=free_bridge_proposal(path,start,length,rng.normal(size=(length-1,2)),FREE_LINK_VARIANCE)
    idx=segment_indices(start,length,P); kpf=1/(2*FREE_LINK_VARIANCE)
    def spring(x):
        links=x[idx[1:]]-x[idx[:-1]]
        return kpf*np.sum(links*links)
    log_reverse_minus_forward=bridge_log_density(path,start,length,FREE_LINK_VARIANCE)-bridge_log_density(new,start,length,FREE_LINK_VARIANCE)
    np.testing.assert_allclose(log_reverse_minus_forward,spring(new)-spring(path),rtol=0.0,atol=1e-12)
