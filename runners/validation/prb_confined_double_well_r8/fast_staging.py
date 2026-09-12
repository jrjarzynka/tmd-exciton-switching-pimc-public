"""Numba-specialized staging sampler for the PRB confined Gaussian double well.

This module implements the SAME primitive action and Brownian-bridge staging
proposal as tmd_pimc.PIMCSamplerStaging, but evaluates the specific
analytic potential directly in Numba. It is retained as an optional campaign-specific accelerator. The authoritative
reviewer-hardening rerun scripts in this directory use the general
tmd_pimc.PIMCSamplerStaging implementation directly.
"""
from __future__ import annotations
import numpy as np
from numba import njit

HBAR2_OVER_2M0 = 0.0380998
KB_EV_PER_K = 8.617333262e-5
MASS_M0 = 0.5
T_K = 20.0
V0_EV = 0.05
SIGMA_NM = 3.0
SEP_NM = 10.0
Q_EFF = 1.0
WALL_R_NM = 25.0
WALL_V0_EV = 0.08
WALL_POWER = 8

@njit(cache=True, inline='always')
def v_scalar(x, y, Ex, asym, wall_R=WALL_R_NM, wall_V=WALL_V0_EV):
    s2 = SIGMA_NM*SIGMA_NM
    dxl = x + 0.5*SEP_NM
    dxr = x - 0.5*SEP_NM
    vl = -V0_EV*np.exp(-(dxl*dxl+y*y)/(2.0*s2))
    vr = -(V0_EV+asym)*np.exp(-(dxr*dxr+y*y)/(2.0*s2))
    r2 = x*x+y*y
    # p=8 -> r^8=(r^2)^4; avoids sqrt and is exact for this chosen regulator.
    wall = wall_V*(r2*r2*r2*r2)/(wall_R**8)
    field = -Q_EFF*Ex*x
    return vl+vr+wall+field

@njit(cache=True)
def run_chain_fast(P, Ex, asym, seed, n_steps, burn_in, sample_every,
                   local_step_nm, global_step_nm, global_prob,
                   segment_lengths, wall_R, wall_V, staging_moves_per_step=2):
    np.random.seed(seed)
    beta = 1.0/(KB_EV_PER_K*T_K)
    tau = beta/P
    lam = HBAR2_OVER_2M0/MASS_M0
    kpf = 1.0/(4.0*lam*tau)
    free_var = 2.0*lam*tau

    path = 0.1*np.random.normal(0.0,1.0,(P,2))
    if n_steps <= burn_in:
        nsamp=0
    else:
        nsamp=1+(n_steps-burn_in-1)//sample_every
    samples=np.empty((nsamp,P,2),dtype=np.float64)
    si=0
    local_acc=0; local_att=0
    stag_acc=0; stag_att=0
    glob_acc=0; glob_att=0
    maxL=0
    for q in range(segment_lengths.shape[0]):
        if segment_lengths[q]>maxL:maxL=segment_lengths[q]
    prop=np.empty((maxL-1,2),dtype=np.float64)
    idx=np.empty(maxL+1,dtype=np.int64)

    for step in range(n_steps):
        # local sweep
        for j in range(P):
            jm=(j-1)%P; jp=(j+1)%P
            ox=path[j,0]; oy=path[j,1]
            nx=ox+local_step_nm*np.random.normal(); ny=oy+local_step_nm*np.random.normal()
            dx=path[jm,0]-ox;dy=path[jm,1]-oy
            dx2=ox-path[jp,0];dy2=oy-path[jp,1]
            oldsp=dx*dx+dy*dy+dx2*dx2+dy2*dy2
            dx=path[jm,0]-nx;dy=path[jm,1]-ny
            dx2=nx-path[jp,0];dy2=ny-path[jp,1]
            newsp=dx*dx+dy*dy+dx2*dx2+dy2*dy2
            dS=kpf*(newsp-oldsp)+tau*(v_scalar(nx,ny,Ex,asym,wall_R,wall_V)-v_scalar(ox,oy,Ex,asym,wall_R,wall_V))
            if dS<0.0 or np.random.random()<np.exp(-dS):
                path[j,0]=nx;path[j,1]=ny
                if step>=burn_in:local_acc+=1
            if step>=burn_in:local_att+=1

        # staging moves
        for sm in range(staging_moves_per_step):
            iq=np.random.randint(0,segment_lengths.shape[0])
            L=int(segment_lengths[iq])
            start=np.random.randint(0,P)
            for k in range(L+1):idx[k]=(start+k)%P
            prevx=path[idx[0],0];prevy=path[idx[0],1]
            endx=path[idx[L],0];endy=path[idx[L],1]
            oldV=0.0;newV=0.0
            for off in range(1,L):
                rem=L-off; denom=rem+1.0
                mx=(rem*prevx+endx)/denom;my=(rem*prevy+endy)/denom
                var=free_var*rem/denom
                px=mx+np.sqrt(var)*np.random.normal();py=my+np.sqrt(var)*np.random.normal()
                prop[off-1,0]=px;prop[off-1,1]=py
                ii=idx[off]
                oldV+=v_scalar(path[ii,0],path[ii,1],Ex,asym,wall_R,wall_V)
                newV+=v_scalar(px,py,Ex,asym,wall_R,wall_V)
                prevx=px;prevy=py
            dS=tau*(newV-oldV)
            accepted=(dS<0.0 or np.random.random()<np.exp(-dS))
            if accepted:
                for off in range(1,L):
                    ii=idx[off];path[ii,0]=prop[off-1,0];path[ii,1]=prop[off-1,1]
                if step>=burn_in:stag_acc+=1
            if step>=burn_in:stag_att+=1

        # whole-path translation
        if np.random.random()<global_prob:
            dx=global_step_nm*np.random.normal();dy=global_step_nm*np.random.normal()
            oldV=0.0;newV=0.0
            for j in range(P):
                oldV+=v_scalar(path[j,0],path[j,1],Ex,asym,wall_R,wall_V)
                newV+=v_scalar(path[j,0]+dx,path[j,1]+dy,Ex,asym,wall_R,wall_V)
            dS=tau*(newV-oldV)
            accepted=(dS<0.0 or np.random.random()<np.exp(-dS))
            if accepted:
                for j in range(P):
                    path[j,0]+=dx;path[j,1]+=dy
                if step>=burn_in:glob_acc+=1
            if step>=burn_in:glob_att+=1

        if step>=burn_in and (step-burn_in)%sample_every==0:
            samples[si,:,:]=path;si+=1

    return (samples,
            local_acc/local_att if local_att>0 else np.nan,
            stag_acc/stag_att if stag_att>0 else np.nan,
            glob_acc/glob_att if glob_att>0 else np.nan)


def run(P=32, Ex=0.0, asym=0.0, seed=1, n_steps=60000, burn_in=15000,
        sample_every=20, local_step_nm=None, global_step_nm=1.0,
        global_prob=0.20, segment_lengths=None, staging_moves_per_step=2,
        wall_R_nm=WALL_R_NM, wall_V0_eV=WALL_V0_EV):
    if local_step_nm is None:
        local_step_nm=0.20*np.sqrt(32.0/P)
    if segment_lengths is None:
        segment_lengths=np.array([L for L in (4,8,16,32,64,128,256) if 2<=L<P],dtype=np.int64)
    else:
        segment_lengths=np.asarray(segment_lengths,dtype=np.int64)
    return run_chain_fast(int(P),float(Ex),float(asym),int(seed),int(n_steps),int(burn_in),int(sample_every),float(local_step_nm),float(global_step_nm),float(global_prob),segment_lengths,float(wall_R_nm),float(wall_V0_eV),int(staging_moves_per_step))
