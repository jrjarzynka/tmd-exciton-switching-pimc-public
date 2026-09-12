#!/usr/bin/env python3
"""Asymmetric-well field scans with chain-aware bootstrap crossing uncertainties."""
import argparse,csv,os,sys,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0,os.path.dirname(__file__))
from model import *

ASYM_EV=(0.002,0.004,0.006)
FACTORS=np.array([0.45,0.55,0.65,0.75,0.85,0.95,1.10])
SEED_BASE=45000

def scan_fields(asym):
    pred,_=minimum_to_minimum_crossing(asym)
    return np.sort(pred*FACTORS)

def one(job):
    ia,ifld,is_,n_steps,burn=job; asym=ASYM_EV[ia]; fields=scan_fields(asym); Ex=float(fields[ifld]); seed=SEED_BASE+10000*ia+100*ifld+is_
    pot,action,sampler=make_sampler(32,seed,Ex_eV_per_nm=Ex,asymmetry_eV=asym,staging_length=16)
    out=sampler.run(n_steps=n_steps,burn_in=burn,sample_every=20)
    s=summarize_samples(out['samples'],pot)
    s.update(asymmetry_eV=asym,field_index=ifld,Ex_eV_per_nm=Ex,seed=seed,n_samples=out['n_samples'])
    return s

def interp_crossing(fields,vals):
    order=np.argsort(fields); x=np.asarray(fields)[order];y=np.asarray(vals)[order]
    d=y-0.5
    for i in range(len(x)-1):
        if d[i]==0:return float(x[i])
        if d[i]*d[i+1] <= 0 and y[i+1]!=y[i]:
            return float(x[i]+(0.5-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i]))
    return np.nan

def bootstrap(rr,metric,nboot=20000,seed=123):
    rng=np.random.default_rng(seed); fields=sorted({r['Ex_eV_per_nm'] for r in rr}); groups=[[r[metric] for r in rr if r['Ex_eV_per_nm']==x] for x in fields]
    mean=[np.mean(g) for g in groups]; point=interp_crossing(fields,mean); boots=[]
    for _ in range(nboot):
        vals=[np.mean(rng.choice(g,size=len(g),replace=True)) for g in groups]; c=interp_crossing(fields,vals)
        if np.isfinite(c):boots.append(c)
    b=np.asarray(boots); return point,(np.quantile(b,.025) if len(b) else np.nan),(np.quantile(b,.975) if len(b) else np.nan),len(b)/nboot

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=5);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--bootstrap',type=int,default=20000);ap.add_argument('--quick',action='store_true')
    args=ap.parse_args();steps,burn=(60000,15000) if not args.quick else (2500,500);nseed=args.seeds if not args.quick else min(args.seeds,2)
    jobs=[(ia,ifld,s,steps,burn) for ia in range(3) for ifld in range(len(FACTORS)) for s in range(nseed)]
    rows=[];t=time.time()
    if args.workers>1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(one,jobs):rows.append(r);print(r,flush=True)
    else:
        for j in jobs:r=one(j);rows.append(r);print(r,flush=True)
    with open('asymmetry_confined_chain.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    out=[]
    for asym in ASYM_EV:
        rr=[r for r in rows if r['asymmetry_eV']==asym]; pred,m=minimum_to_minimum_crossing(asym)
        o={'asymmetry_nominal_meV':1000*asym,'Delta0_min_to_min_meV':1000*m['Delta0_eV'],'xA_nm':m['xA_nm'],'xB_nm':m['xB_nm'],'Ecrit_minimum_meV_per_nm':1000*pred}
        for metric in ('p_bead','p_centroid'):
            c,lo,hi,success=bootstrap(rr,metric,args.bootstrap,123+int(1e6*asym)+(0 if metric=='p_bead' else 1))
            o[metric+'_cross_meV_per_nm']=1000*c;o[metric+'_cross_ci_lo']=1000*lo;o[metric+'_cross_ci_hi']=1000*hi;o[metric+'_bootstrap_bracket_fraction']=success
        out.append(o)
    with open('asymmetry_confined_crossings.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys()));w.writeheader();w.writerows(out)
    print(f'elapsed {time.time()-t:.1f}s; crossings:');[print(x) for x in out]
if __name__=='__main__':main()
