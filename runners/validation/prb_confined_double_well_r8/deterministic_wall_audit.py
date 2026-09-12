#!/usr/bin/env python3
import csv
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar
from pathlib import Path

HBAR2_OVER_2M0_EV_NM2 = 0.0380998
MASS_M0 = 0.5
KIN = HBAR2_OVER_2M0_EV_NM2 / MASS_M0
V0 = 0.05
SIGMA = 3.0
SEP = 10.0
V_WALL = 0.08
POWER = 8


def v_dw_xy(x, y, asym=0.0):
    left = -V0*np.exp(-((x+SEP/2)**2+y*y)/(2*SIGMA**2))
    right = -(V0+asym)*np.exp(-((x-SEP/2)**2+y*y)/(2*SIGMA**2))
    return left+right


def v_wall_xy(x,y,R):
    if np.isinf(R):
        return np.zeros_like(np.asarray(x, dtype=float)+np.asarray(y, dtype=float))
    r2 = x*x+y*y
    return V_WALL * (np.sqrt(r2)/R)**POWER


def v_total_xy(x,y,R):
    return v_dw_xy(x,y)+v_wall_xy(x,y,R)


def local_geometry(R):
    f=lambda x: float(v_total_xy(x,0.0,R))
    res=minimize_scalar(f,bounds=(-8,-2),method='bounded',options={'xatol':1e-13})
    xmin=res.x; vmin=res.fun
    saddle=f(0.0)
    return xmin,vmin,saddle,saddle-vmin


def eigenspectrum(R, halfwidth=20.0, ngrid=128, nev=30):
    # Match documented protocol: full grid has ngrid including Dirichlet boundaries.
    x=np.linspace(-halfwidth, halfwidth, ngrid)
    dx=x[1]-x[0]
    xi=x[1:-1]
    n=len(xi)
    main=-2*np.ones(n); off=np.ones(n-1)
    D2=diags([off,main,off],[-1,0,1],format='csr')/dx**2
    I=eye(n,format='csr')
    Lap=kron(I,D2)+kron(D2,I)
    X,Y=np.meshgrid(xi,xi,indexing='xy')
    V=v_total_xy(X,Y,R).ravel()
    H=(-KIN)*Lap+diags(V,0,format='csr')
    vals,vecs=eigsh(H,k=nev,which='SA',tol=1e-11,maxiter=100000)
    order=np.argsort(vals); vals=vals[order]; vecs=vecs[:,order]
    return vals,vecs,xi,dx


def thermal_pr(vals,vecs,xi,T=20.0,nstates=30):
    kb=8.617333262e-5
    vals=vals[:nstates]; vecs=vecs[:,:nstates]
    w=np.exp(-(vals-vals[0])/(kb*T))
    X,Y=np.meshgrid(xi,xi,indexing='xy')
    mask=(X.ravel()>0)
    # eigsh vectors normalized in Euclidean norm; ratio of spatial sums is sufficient.
    pr=np.sum((vecs[mask,:]**2),axis=0)/np.sum(vecs**2,axis=0)
    return float(np.dot(w,pr)/w.sum())


def main():
    radii=[np.inf,20.0,25.0,30.0]
    rows=[]
    spectra={}
    for R in radii:
        xmin,vmin,saddle,barrier=local_geometry(R)
        vals,vecs,xi,dx=eigenspectrum(R)
        spectra[R]=vals
        rows.append({
            'R_wall_nm':'inf' if np.isinf(R) else f'{R:g}',
            'Vwall_eV':0.0 if np.isinf(R) else V_WALL,
            'power':POWER,
            'xmin_nm':xmin,
            'Vmin_meV':vmin*1000,
            'Vsaddle_meV':saddle*1000,
            'barrier_meV':barrier*1000,
            'E0_meV':vals[0]*1000,
            'E1_meV':vals[1]*1000,
            'E0_above_min_meV':(vals[0]-vmin)*1000,
            'saddle_minus_E0_meV':(saddle-vals[0])*1000,
            'pR_diag_T20':thermal_pr(vals,vecs,xi),
        })
    base=spectra[np.inf]
    for row,R in zip(rows,radii):
        vals=spectra[R]
        row['dE0_vs_unconf_meV']=(vals[0]-base[0])*1000
        row['dE1_vs_unconf_meV']=(vals[1]-base[1])*1000
        row['dE2_vs_unconf_meV']=(vals[2]-base[2])*1000
    out = Path(__file__).resolve().parents[3] / 'paper1_prb_confined_double_well' / 'audits' / 'deterministic_wall_audit.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for row in rows:
        print(row)
    # local wall scales for chosen R=25
    R=25.0
    print('\nChosen wall R=25 nm, V0=80 meV, p=8:')
    for r in [5,10,15,20,25,30]:
        print(f'r={r:2d} nm: Vwall={V_WALL*(r/R)**POWER*1000:.9f} meV')

if __name__=='__main__': main()
