"""Numba staging kernels.

The Brownian-bridge primitive is reusable; ``run_periodic_staging_jit`` is the
optimized single-body periodic-grid consumer used by Paper 1.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True)
def free_bridge_proposal_jit(path, start, length, gaussian_variates, free_link_variance_nm2):
    """JIT counterpart of staging.free_bridge_proposal with caller RNG data."""
    proposed = path.copy()
    p = path.shape[0]
    previous_x = path[start, 0]
    previous_y = path[start, 1]
    endpoint = (start + length) % p
    endpoint_x = path[endpoint, 0]
    endpoint_y = path[endpoint, 1]
    for offset in range(1, length):
        remaining = length - offset
        denominator = remaining + 1.0
        sd = math.sqrt(free_link_variance_nm2 * remaining / denominator)
        x = (remaining * previous_x + endpoint_x) / denominator
        y = (remaining * previous_y + endpoint_y) / denominator
        x += sd * gaussian_variates[offset - 1, 0]
        y += sd * gaussian_variates[offset - 1, 1]
        index = (start + offset) % p
        proposed[index, 0] = x
        proposed[index, 1] = y
        previous_x, previous_y = x, y
    return proposed


@njit(cache=True)
def periodic_grid_value(x, y, grid, ox, oy, a00, a01, a10, a11):
    rx, ry = x - ox, y - oy
    u = a00 * rx + a01 * ry
    v = a10 * rx + a11 * ry
    u -= math.floor(u)
    v -= math.floor(v)
    nx, ny = grid.shape
    fx, fy = u * nx, v * ny
    if fx >= nx: fx = 0.0
    if fy >= ny: fy = 0.0
    if fx < 0.0: fx += nx
    if fy < 0.0: fy += ny
    ix, iy = int(math.floor(fx)), int(math.floor(fy))
    wx, wy = fx - ix, fy - iy
    ix1, iy1 = (ix + 1) % nx, (iy + 1) % ny
    return ((1-wx)*(1-wy)*grid[ix,iy] + wx*(1-wy)*grid[ix1,iy]
            + (1-wx)*wy*grid[ix,iy1] + wx*wy*grid[ix1,iy1])


@njit(cache=True)
def staging_delta_potential_jit(old, new, start, length, grid, ox, oy,
                                a00, a01, a10, a11, tau):
    delta = 0.0
    p = old.shape[0]
    for offset in range(1, length):
        index = (start + offset) % p
        delta += periodic_grid_value(new[index,0],new[index,1],grid,ox,oy,a00,a01,a10,a11)
        delta -= periodic_grid_value(old[index,0],old[index,1],grid,ox,oy,a00,a01,a10,a11)
    return tau * delta


@njit(cache=True)
def run_periodic_staging_jit(n_steps,burn_in,sample_every,path,grid,ox,oy,
    a00,a01,a10,a11,kpf,tau,lam,local_step_nm,global_step_nm,
    global_move_probability,segment_lengths,staging_moves_per_step,seed):
    np.random.seed(seed)
    p=path.shape[0]
    n_samples=1+(n_steps-burn_in-1)//sample_every if n_steps>burn_in else 0
    samples=np.empty((n_samples,p,2),dtype=np.float64); sample_index=0
    local_accepted=global_accepted=global_attempts=staging_accepted=staging_attempts=0
    length_attempts=np.zeros(len(segment_lengths),dtype=np.int64)
    length_accepted=np.zeros(len(segment_lengths),dtype=np.int64)
    max_length=int(np.max(segment_lengths)); zx=np.empty(max_length-1); zy=np.empty(max_length-1)
    free_link_variance=2.0*lam*tau
    for step in range(n_steps):
        for j in range(p):
            jm,jp=(j-1)%p,(j+1)%p; x,y=path[j,0],path[j,1]
            nx=x+local_step_nm*np.random.normal(); ny=y+local_step_nm*np.random.normal()
            old=((path[jm,0]-x)**2+(path[jm,1]-y)**2+(path[jp,0]-x)**2+(path[jp,1]-y)**2)
            new=((path[jm,0]-nx)**2+(path[jm,1]-ny)**2+(path[jp,0]-nx)**2+(path[jp,1]-ny)**2)
            ds=kpf*(new-old)+tau*(periodic_grid_value(nx,ny,grid,ox,oy,a00,a01,a10,a11)-periodic_grid_value(x,y,grid,ox,oy,a00,a01,a10,a11))
            if ds<0.0 or np.random.random()<math.exp(-ds):
                path[j,0],path[j,1]=nx,ny
                if step>=burn_in: local_accepted+=1
        for _ in range(staging_moves_per_step):
            li=np.random.randint(0,len(segment_lengths)); length=int(segment_lengths[li]); start=np.random.randint(0,p)
            endpoint=(start+length)%p; px,py=path[start,0],path[start,1]; ex,ey=path[endpoint,0],path[endpoint,1]
            for offset in range(1,length):
                remaining=length-offset; denominator=remaining+1.0
                sd=math.sqrt(free_link_variance*remaining/denominator)
                x=(remaining*px+ex)/denominator+sd*np.random.normal()
                y=(remaining*py+ey)/denominator+sd*np.random.normal()
                zx[offset-1],zy[offset-1]=x,y; px,py=x,y
            delta=0.0
            for offset in range(1,length):
                index=(start+offset)%p
                delta += periodic_grid_value(zx[offset-1],zy[offset-1],grid,ox,oy,a00,a01,a10,a11)
                delta -= periodic_grid_value(path[index,0],path[index,1],grid,ox,oy,a00,a01,a10,a11)
            ds=tau*delta
            if step>=burn_in: staging_attempts+=1; length_attempts[li]+=1
            if ds<0.0 or np.random.random()<math.exp(-ds):
                for offset in range(1,length):
                    index=(start+offset)%p; path[index,0],path[index,1]=zx[offset-1],zy[offset-1]
                if step>=burn_in: staging_accepted+=1; length_accepted[li]+=1
        if np.random.random()<global_move_probability:
            dx=global_step_nm*np.random.normal(); dy=global_step_nm*np.random.normal(); delta=0.0
            if step>=burn_in: global_attempts+=1
            for j in range(p):
                delta += periodic_grid_value(path[j,0]+dx,path[j,1]+dy,grid,ox,oy,a00,a01,a10,a11)
                delta -= periodic_grid_value(path[j,0],path[j,1],grid,ox,oy,a00,a01,a10,a11)
            ds=tau*delta
            if ds<0.0 or np.random.random()<math.exp(-ds):
                for j in range(p): path[j,0]+=dx; path[j,1]+=dy
                if step>=burn_in: global_accepted+=1
        if step>=burn_in and (step-burn_in)%sample_every==0:
            samples[sample_index]=path; sample_index+=1
    effective=max(1,n_steps-burn_in)
    return (samples,local_accepted/(effective*p),
            staging_accepted/staging_attempts if staging_attempts else np.nan,
            global_accepted/global_attempts if global_attempts else np.nan,
            length_attempts,length_accepted)
