#!/usr/bin/env python3
"""Development-only COM staging pilot on the 5 K/P=80 periodic moire target."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"numerics"))
from tmd_pimc import (GridPotential2D,RingPolymerAction,PIMCSampler,
                      PIMCSamplerJIT,PIMCSamplerStaging,
                      PIMCSamplerStagingPeriodicJIT)
from tmd_pimc.constants import HBAR2_OVER_2M0,KB_EV_PER_K

T=5.0; P=80; MASS=.5; N_STEPS=160000; BURN=20000; EVERY=20
SEEDS=(909201,909202); BASINS={909201:("A",np.array([1/3,1/3])),909202:("B",np.array([2/3,2/3]))}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()

def iat(x):
 x=np.asarray(x,float); n=len(x); y=x-x.mean(); nfft=1<<(2*n-1).bit_length()
 f=np.fft.rfft(y,nfft); ac=np.fft.irfft(f*np.conj(f),nfft)[:n]/np.arange(n,0,-1)
 if ac[0]<=0:return 1.0
 rho=ac/ac[0]; total=0.; m=1
 while 2*m<n:
  pair=rho[2*m-1]+rho[2*m]
  if not np.isfinite(pair) or pair<=0: break
  total+=pair; m+=1
 return float(max(1.,1+2*total))

def load_grid(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k]) for k in z.files}

def make_action(grid):
 pot=GridPotential2D(grid["u_fractional"],grid["v_fractional"],grid["V_eV"],
     periodic=True,subtract_minimum=False,lattice_vectors_nm=grid["lattice_vectors_nm"],
     origin_nm=grid["origin_nm"])
 return RingPolymerAction(MASS,T,P,pot),pot

def diagnostics(samples,action,pot,grid,seconds):
 n=len(samples); F=np.fft.fft(samples,axis=1)/P; power=np.sum(np.abs(F[:,:4,:])**2,axis=2)
 center=samples.mean(axis=1); relative=samples-center[:,None,:]
 rg=np.mean(np.sum(relative*relative,axis=2),axis=1)
 bonds=np.roll(samples,-1,axis=1)-samples; bond=np.sum(bonds*bonds,axis=(1,2))
 potential=np.mean(pot.value(samples.reshape(-1,2)).reshape(n,P),axis=1)
 spring=bond/(4*action._lambda_x*P*action._tau*action._tau)
 A=np.column_stack([grid["lattice_vectors_nm"][0],grid["lattice_vectors_nm"][1]])
 uv=(center-grid["origin_nm"])@np.linalg.inv(A).T; uv-=np.floor(uv)
 shifts=np.array([(i,j) for i in (-1,0,1) for j in (-1,0,1)],float); d=[]
 for minimum in (np.array([1/3,1/3]),np.array([2/3,2/3])):
  du=uv[:,None,:]-(minimum[None,None,:]+shifts[None,:,:]); dr=du@A.T
  d.append(np.min(np.sum(dr*dr,axis=2),axis=1))
 labels=np.argmin(np.column_stack(d),axis=1); hist=np.histogram2d(uv[:,0],uv[:,1],bins=16,range=[[0,1],[0,1]])[0]; hist/=hist.sum()
 series={"q1":power[:,1],"q2":power[:,2],"q3":power[:,3],"R_g2":rg,
         "mean_potential_eV":potential,"spring_proxy_eV":spring}
 metrics={}
 for key,x in series.items():
  t=iat(x); e=n/t; metrics[key]={"mean":float(np.mean(x)),"IAT":t,"ESS":e,"ESS_per_second":e/seconds}
 return metrics,{"A":float(np.mean(labels==0)),"B":float(np.mean(labels==1))},int(np.sum(labels[1:]!=labels[:-1])),hist

def run_one(backend,configuration,seed,grid_path,outdir):
 grid=load_grid(Path(grid_path)); action,pot=make_action(grid); basin,uv=BASINS[seed]
 A=np.column_stack([grid["lattice_vectors_nm"][0],grid["lattice_vectors_nm"][1]])
 center=grid["origin_nm"]+A@uv; tau=1/(KB_EV_PER_K*T)/P; lam=HBAR2_OVER_2M0/MASS
 local=.70*math.sqrt(2*lam*tau); common=dict(action=action,local_step_nm=local,
      global_step_nm=15.,global_move_probability=.20,rng_seed=seed)
 def make_sampler():
  if backend=="python" and configuration=="baseline": return PIMCSampler(**common)
  if backend=="python": return PIMCSamplerStaging(**common,staging_segment_lengths=(32,),staging_moves_per_step=1)
  if backend=="jit" and configuration=="baseline":
   return PIMCSamplerJIT(**common,grid_size=400,grid_range_nm=1.,boundary_mode="periodic_cell",
       periodic_cell_vectors_nm=grid["lattice_vectors_nm"],periodic_cell_origin_nm=grid["origin_nm"],
       global_disp_vectors_nm=None,directed_move_frac=0.0)
  return PIMCSamplerStagingPeriodicJIT(**common,staging_segment_lengths=(32,),staging_moves_per_step=1)
 sampler=make_sampler()
 # Warm JIT compilation outside timing, then recreate the sampler so the timed
 # run starts from the same seed/RNG state as the corresponding fresh run.
 if backend=="jit":
  sampler.run(n_steps=2,burn_in=0,sample_every=1,center=center)
  sampler=make_sampler()
 t0=time.perf_counter(); result=sampler.run(N_STEPS,BURN,EVERY,center=center); seconds=time.perf_counter()-t0
 samples=np.asarray(result["samples"]); metrics,pop,trans,hist=diagnostics(samples,action,pot,grid,seconds)
 task=f"{backend}_{configuration}_T005_P080_seed{seed}_basin{basin}"
 npz=Path(outdir)/(task+".npz"); np.savez_compressed(npz,samples_unwrapped_nm=samples,centroid_histogram_16x16=hist)
 summary={"task_id":task,"classification":"development_only_not_paper1_evidence","backend":backend,
   "configuration":configuration,"segment_length":None if configuration=="baseline" else 32,
   "seed":seed,"start_basin":basin,"n_steps":N_STEPS,"burn_in":BURN,"sample_every":EVERY,
   "retained":len(samples),"wall_seconds":seconds,"acceptance_local":result["acceptance_local"],
   "acceptance_staging":result.get("acceptance_staging"),"acceptance_global":result["acceptance_global"],
   "metrics":metrics,"basin_populations":pop,"basin_transitions":trans,
   "centroid_histogram_16x16":hist.tolist(),"npz_sha256":sha(npz)}
 jp=npz.with_suffix(".json"); jp.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); return summary

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--grid",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--workers",type=int,default=4); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
 jobs=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
  futures=[ex.submit(run_one,b,c,s,str(args.grid),str(args.out)) for b in ("python","jit") for c in ("baseline","staging_L32") for s in SEEDS]
  for f in concurrent.futures.as_completed(futures): jobs.append(f.result()); print(jobs[-1]["task_id"],flush=True)
 aggregate={}
 for b in ("python","jit"):
  aggregate[b]={}
  for c in ("baseline","staging_L32"):
   js=[j for j in jobs if j["backend"]==b and j["configuration"]==c]; aggregate[b][c]={
    "mean_wall_seconds":float(np.mean([j["wall_seconds"] for j in js])),
    "mean_acceptance_staging":None if c=="baseline" else float(np.mean([j["acceptance_staging"] for j in js])),
    "total_basin_transitions":sum(j["basin_transitions"] for j in js),
    "mean_basin_populations":{x:float(np.mean([j["basin_populations"][x] for j in js])) for x in ("A","B")},
    "metrics":{x:{"mean_IAT":float(np.mean([j["metrics"][x]["IAT"] for j in js])),"total_ESS":float(sum(j["metrics"][x]["ESS"] for j in js)),"total_ESS_per_total_second":float(sum(j["metrics"][x]["ESS"] for j in js)/sum(j["wall_seconds"] for j in js)),"mean":float(np.mean([j["metrics"][x]["mean"] for j in js]))} for x in ("q1","q2","q3","R_g2","mean_potential_eV","spring_proxy_eV")}}
 for b in aggregate:
  aggregate[b]["staging_over_baseline_ESS_per_second"]={x:aggregate[b]["staging_L32"]["metrics"][x]["total_ESS_per_total_second"]/aggregate[b]["baseline"]["metrics"][x]["total_ESS_per_total_second"] for x in aggregate[b]["baseline"]["metrics"]}
  base_jobs=[j for j in jobs if j["backend"]==b and j["configuration"]=="baseline"]
  stage_jobs=[j for j in jobs if j["backend"]==b and j["configuration"]=="staging_L32"]
  rho_base=np.mean([np.asarray(j["centroid_histogram_16x16"],float) for j in base_jobs],axis=0)
  rho_stage=np.mean([np.asarray(j["centroid_histogram_16x16"],float) for j in stage_jobs],axis=0)
  aggregate[b]["density_TV_baseline_vs_staging_L32"]=float(.5*np.sum(np.abs(rho_base-rho_stage)))
  aggregate[b]["staging_minus_baseline_mean"]={}
  for x in aggregate[b]["baseline"]["metrics"]:
   m0=aggregate[b]["baseline"]["metrics"][x]["mean"]; m1=aggregate[b]["staging_L32"]["metrics"][x]["mean"]
   aggregate[b]["staging_minus_baseline_mean"][x]={
    "absolute":float(m1-m0),
    "relative_to_baseline":float((m1-m0)/abs(m0)) if m0!=0 else None,
   }
 final={"title":"COM_STAGING_L32_MOIRE_DEVELOPMENT_PILOT","classification":"development_only_not_paper1_evidence","configuration":{"T_K":T,"P":P,"mass_m0":MASS,"n_steps":N_STEPS,"burn_in":BURN,"sample_every":EVERY,"seeds":list(SEEDS),"global_sigma_nm":15.,"global_probability":.2},"chains":sorted(jobs,key=lambda x:x["task_id"]),"aggregate":aggregate,"production_campaign_run":False}
 path=args.out/"COM_STAGING_L32_MOIRE_PILOT.json"; path.write_text(json.dumps(final,indent=2,sort_keys=True)+"\n"); print(path,sha(path))
if __name__=="__main__":main()
