#!/usr/bin/env python3
"""Production-style symmetric field scan for the thermodynamically confined model."""
import argparse, csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import *

KB=8.617333262e-5
EX_CHAR=KB*T_K/(Q_EFF*SEPARATION_NM)
EX_SCAN=np.linspace(-3*EX_CHAR,3*EX_CHAR,9)
SEED_BASE=25000

def one(job):
    i_field,i_seed,P,n_steps,burn,sample_every,save_samples=job
    Ex=float(EX_SCAN[i_field]); seed=SEED_BASE+100*i_field+i_seed
    pot,action,sampler=make_sampler(P,seed,Ex_eV_per_nm=Ex)
    out=sampler.run(n_steps=n_steps,burn_in=burn,sample_every=sample_every)
    s=summarize_samples(out['samples'],pot)
    s.update(i_field=i_field,i_seed=i_seed,seed=seed,P=P,Ex_eV_per_nm=Ex,
             n_samples=out['n_samples'],acceptance_local=out['acceptance_local'],
             acceptance_staging=out['acceptance_staging'],acceptance_global=out['acceptance_global'])
    if save_samples and i_field==4:
        np.savez_compressed(f'zero_bias_samples_P{P}_seed{seed}.npz',samples=out['samples'])
    return s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=8)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--P',type=int,default=32)
    ap.add_argument('--steps',type=int,default=N_STEPS)
    ap.add_argument('--burn-in',type=int,default=BURN_IN)
    ap.add_argument('--sample-every',type=int,default=SAMPLE_EVERY)
    ap.add_argument('--only-index',type=int)
    ap.add_argument('--save-zero-samples',action='store_true')
    ap.add_argument('--quick',action='store_true')
    args=ap.parse_args()
    if args.quick:
        args.steps,args.burn_in,args.seeds=2500,500,min(args.seeds,2)
    fields=[args.only_index] if args.only_index is not None else list(range(9))
    jobs=[(i,s,args.P,args.steps,args.burn_in,args.sample_every,args.save_zero_samples)
          for i in fields for s in range(args.seeds)]
    t=time.time(); rows=[]
    if args.workers>1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(one,jobs): rows.append(r); print(r,flush=True)
    else:
        for j in jobs: r=one(j); rows.append(r); print(r,flush=True)
    fields_out=list(rows[0].keys())
    with open('symmetric_confined_chain.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields_out);w.writeheader();w.writerows(sorted(rows,key=lambda z:(z['i_field'],z['i_seed'])))
    summary=[]
    for i in sorted({r['i_field'] for r in rows}):
        rr=[r for r in rows if r['i_field']==i]
        o={'i_field':i,'Ex_eV_per_nm':rr[0]['Ex_eV_per_nm'],'Ex_meV_per_nm':1000*rr[0]['Ex_eV_per_nm'],'n_chains':len(rr)}
        for key in ['p_bead','p_centroid','span_05','span_10','span_15','span_20','Rg2_nm2','Vmean_eV','max_bead_radius_nm']:
            a=np.array([x[key] for x in rr],float); o[key+'_mean']=a.mean(); o[key+'_sem']=a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.nan
        summary.append(o)
    with open('symmetric_confined_summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys()));w.writeheader();w.writerows(summary)
    print(f'elapsed {time.time()-t:.1f}s; wrote symmetric_confined_chain.csv and symmetric_confined_summary.csv')

if __name__=='__main__': main()
