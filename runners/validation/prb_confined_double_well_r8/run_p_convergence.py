#!/usr/bin/env python3
"""P=32,64,128 targeted non-harmonic audit for the confined Hamiltonian."""
import argparse,csv,os,sys,time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0,os.path.dirname(__file__))
from model import *

SEED_BASE=35000
TESTS=[('steep',0.258519998/1000.0),('zero',0.0)]

def one(job):
    test_i,P,seed_i,n_steps,burn,sample_every=job
    label,Ex=TESTS[test_i]; seed=SEED_BASE+10000*test_i+100*P+seed_i
    L={32:16,64:32,128:32}.get(P,min(32,P-1))
    pot,action,sampler=make_sampler(P,seed,Ex_eV_per_nm=Ex,staging_length=L)
    out=sampler.run(n_steps=n_steps,burn_in=burn,sample_every=sample_every)
    s=summarize_samples(out['samples'],pot)
    s.update(test=label,Ex_eV_per_nm=Ex,P=P,seed=seed,n_samples=out['n_samples'],
             acceptance_local=out['acceptance_local'],acceptance_staging=out['acceptance_staging'],acceptance_global=out['acceptance_global'])
    return s

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=8);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--quick',action='store_true')
    args=ap.parse_args(); steps,burn=(60000,15000) if not args.quick else (2500,500); nseed=args.seeds if not args.quick else min(args.seeds,2)
    jobs=[(ti,P,s,steps,burn,20) for ti in range(2) for P in (32,64,128) for s in range(nseed)]
    rows=[];t=time.time()
    if args.workers>1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(one,jobs): rows.append(r); print(r,flush=True)
    else:
        for j in jobs:r=one(j);rows.append(r);print(r,flush=True)
    with open('p_audit_confined_chain.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    summary=[]
    for label,_ in TESTS:
        for P in (32,64,128):
            rr=[r for r in rows if r['test']==label and r['P']==P]; o={'test':label,'P':P,'n_chains':len(rr)}
            for key in ['p_centroid','p_bead','span_05','span_10','span_15','span_20','Rg2_nm2','Vmean_eV']:
                a=np.array([x[key] for x in rr]);o[key+'_mean']=a.mean();o[key+'_sem']=a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.nan
            summary.append(o)
    with open('p_audit_confined_summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys()));w.writeheader();w.writerows(summary)
    print(f'elapsed {time.time()-t:.1f}s')
if __name__=='__main__':main()
