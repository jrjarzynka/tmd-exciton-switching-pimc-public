#!/usr/bin/env python3
import csv, time
from pathlib import Path
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh
HBAR2_OVER_2M0_EV_NM2=0.0380998; MASS=0.5; KIN=HBAR2_OVER_2M0_EV_NM2/MASS
V0=0.05; SIG=3.0; SEP=10.0; VW=0.08; RW=25.0; POW=8
KB=8.617333262e-5; T=20.0; FIELD=0.258519998/1000.0

def V(x,y):
    dw=-V0*np.exp(-((x+SEP/2)**2+y*y)/(2*SIG**2))-V0*np.exp(-((x-SEP/2)**2+y*y)/(2*SIG**2))
    wall=VW*(np.sqrt(x*x+y*y)/RW)**POW
    return dw+wall-FIELD*x

def calc(halfwidth,ngrid,nev):
    x=np.linspace(-halfwidth,halfwidth,ngrid); dx=x[1]-x[0]; xi=x[1:-1]; n=len(xi)
    D2=diags([np.ones(n-1),-2*np.ones(n),np.ones(n-1)],[-1,0,1],format='csr')/dx**2
    I=eye(n,format='csr'); Lap=kron(I,D2)+kron(D2,I)
    X,Y=np.meshgrid(xi,xi,indexing='xy')
    H=(-KIN)*Lap+diags(V(X,Y).ravel(),0,format='csr')
    vals,vecs=eigsh(H,k=nev,which='SA',tol=3e-10,maxiter=100000)
    o=np.argsort(vals); vals=vals[o]; vecs=vecs[:,o]
    w=np.exp(-(vals-vals[0])/(KB*T)); mask=(X.ravel()>0)
    pr=np.sum(vecs[mask,:]**2,axis=0)/np.sum(vecs**2,axis=0)
    p=float(np.dot(w,pr)/w.sum()); tail=float(w[-1]/w.sum())
    return p,float(vals[0]*1000),float(vals[-1]*1000),dx,tail
cases=[('coarse',25.,128,30),('production',25.,160,30),('fine',25.,192,30),('small_box',20.,160,30),('large_box',30.,160,30),('n20',25.,160,20),('n40',25.,160,40)]
rows=[]
for name,hw,ng,nev in cases:
    t=time.time(); p,e0,el,dx,tail=calc(hw,ng,nev); dt=time.time()-t
    row=dict(case=name,halfwidth_nm=hw,ngrid=ng,nev=nev,dx_nm=dx,pR_diag=p,E0_meV=e0,E_last_meV=el,last_weight_fraction=tail,runtime_s=dt)
    rows.append(row); print(row,flush=True)
out = Path(__file__).resolve().parents[3] / 'paper1_prb_confined_double_well' / 'audits' / 'deterministic_occupation_convergence.csv'
out.parent.mkdir(parents=True, exist_ok=True)
with open(out,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
base=next(r for r in rows if r['case']=='production')['pR_diag']
print('production',base)
for r in rows: print(r['case'],r['pR_diag']-base)
