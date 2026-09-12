#!/usr/bin/env python3
"""Primitive vs HBB virial energy cross-check on the *confined* zero-bias double well."""
import argparse,csv,os,sys
import numpy as np
sys.path.insert(0,os.path.dirname(__file__))
from model import *
from tmd_pimc.energy_estimators import primitive_energy_series, virial_energy_series


def block_sem(x,nblocks=20):
    x=np.asarray(x,float); n=(len(x)//nblocks)*nblocks
    if n<2*nblocks:return float(np.mean(x)),float(np.std(x,ddof=1)/np.sqrt(len(x)))
    b=x[:n].reshape(nblocks,-1).mean(axis=1)
    return float(b.mean()),float(b.std(ddof=1)/np.sqrt(nblocks))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=9000);ap.add_argument('--quick',action='store_true');args=ap.parse_args()
    rows=[]
    for P,seed in [(32,61032),(80,61080)]:
        nsteps=1000 if args.quick else (15000+20*args.samples)
        burn=200 if args.quick else 15000
        pot,action,sampler=make_sampler(P,seed,0.0,0.0,staging_length=(16 if P==32 else 32))
        out=sampler.run(n_steps=nsteps,burn_in=burn,sample_every=20)
        ep=primitive_energy_series(out['samples'],pot,MASS_M0,T_K);ev=virial_energy_series(out['samples'],pot)
        mp,sp=block_sem(ep);mv,sv=block_sem(ev); z=abs(mp-mv)/np.sqrt(sp*sp+sv*sv)
        rows.append({'P':P,'n_samples':len(ep),'Eprim_eV':mp,'SEMprim_eV':sp,'Evir_eV':mv,'SEMvir_eV':sv,'agreement_sigma':z,'max_bead_radius_nm':summarize_samples(out['samples'],pot)['max_bead_radius_nm']})
        print(rows[-1])
    with open('energy_confined.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
if __name__=='__main__':main()
