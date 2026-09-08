#!/usr/bin/env python3
"""Development-only P=32/L=16 moire pilot for the 15 K and 20 K schedule."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"numerics"))
from tmd_pimc import GridPotential2D,RingPolymerAction,PIMCSampler,PIMCSamplerJIT,PIMCSamplerStaging,PIMCSamplerStagingPeriodicJIT
from tmd_pimc.constants import HBAR2_OVER_2M0,KB_EV_PER_K

GRID_SHA="8fbd5d9a9e21c3f7b3655034bf044d396b37f1af8056ae64626fae783564dd65"
TEMPERATURES=(15.0,20.0); P=32; L=16; MASS=.5
N_STEPS=160_000; BURN=20_000; EVERY=20
MODES=(1,2,3,4,6,8,12,16)
JOBS={15.0:((917151,"A",np.array([1/3,1/3])),(917152,"B",np.array([2/3,2/3]))),
      20.0:((917201,"A",np.array([1/3,1/3])),(917202,"B",np.array([2/3,2/3])))}
CONFIGS=(("python","baseline"),("python","staging_L16"),("jit","baseline"),("jit","staging_L16"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""): h.update(block)
    return h.hexdigest()


def iat(x):
    x=np.asarray(x,float); n=len(x); y=x-x.mean(); nfft=1<<(2*n-1).bit_length()
    ac=np.fft.irfft(np.fft.rfft(y,nfft)*np.conj(np.fft.rfft(y,nfft)),nfft)[:n]/np.arange(n,0,-1)
    if n<4 or not np.isfinite(ac[0]) or ac[0]<=0:return 1.0
    rho=ac/ac[0]; total=0.; m=1
    while 2*m<n:
        pair=rho[2*m-1]+rho[2*m]
        if not np.isfinite(pair) or pair<=0:break
        total+=pair;m+=1
    return float(max(1.,1+2*total))


def load_grid(path):
    if sha(path)!=GRID_SHA: raise RuntimeError("grid SHA-256 mismatch")
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k]) for k in z.files}


def make_action(grid,T):
    potential=GridPotential2D(grid["u_fractional"],grid["v_fractional"],grid["V_eV"],periodic=True,
        subtract_minimum=False,lattice_vectors_nm=grid["lattice_vectors_nm"],origin_nm=grid["origin_nm"])
    return RingPolymerAction(MASS,T,P,potential),potential


def diagnostics(samples,action,potential,grid,seconds):
    n=len(samples); fourier=np.fft.fft(samples,axis=1)/P
    powers=np.sum(np.abs(fourier[:,list(MODES),:])**2,axis=2)
    center=samples.mean(axis=1); relative=samples-center[:,None,:]
    rg2=np.mean(np.sum(relative*relative,axis=2),axis=1)
    bonds=np.roll(samples,-1,axis=1)-samples; bond=np.sum(bonds*bonds,axis=(1,2))
    potential_series=np.mean(potential.value(samples.reshape(-1,2)).reshape(n,P),axis=1)
    spring=bond/(4*action._lambda_x*P*action._tau*action._tau)
    A=np.column_stack([grid["lattice_vectors_nm"][0],grid["lattice_vectors_nm"][1]])
    uv_unwrapped=(center-grid["origin_nm"])@np.linalg.inv(A).T; uv=uv_unwrapped-np.floor(uv_unwrapped)
    shifts=np.array([(i,j) for i in (-1,0,1) for j in (-1,0,1)],float); distances=[]
    for minimum in (np.array([1/3,1/3]),np.array([2/3,2/3])):
        du=uv[:,None,:]-(minimum[None,None,:]+shifts[None,:,:]); dr=du@A.T
        distances.append(np.min(np.sum(dr*dr,axis=2),axis=1))
    labels=np.argmin(np.column_stack(distances),axis=1).astype(np.int8)
    hist=np.histogram2d(uv[:,0],uv[:,1],bins=16,range=[[0,1],[0,1]])[0].astype(float);hist/=hist.sum()
    named={f"q{k}":powers[:,i] for i,k in enumerate(MODES)}
    named.update({"R_g2_nm2":rg2,"bond_squared_sum_nm2":bond,"mean_potential_eV":potential_series,"spring_proxy_eV":spring})
    metrics={}
    for key,values in named.items():
        tau=iat(values);metrics[key]={"mean":float(np.mean(values)),"IAT":tau,"ESS":float(n/tau),"ESS_per_second":float(n/tau/seconds)}
    circular={axis:{"mean_fractional":float((np.angle(np.mean(np.exp(2j*np.pi*uv[:,i])))%(2*np.pi))/(2*np.pi)),
                    "resultant":float(abs(np.mean(np.exp(2j*np.pi*uv[:,i]))))} for i,axis in enumerate(("u","v"))}
    arrays={"centroids_unwrapped_nm":center,"centroids_fractional_wrapped":uv,"mode_power":powers,"R_g2_nm2":rg2,
            "bond_squared_sum_nm2":bond,"mean_potential_eV_per_sample":potential_series,"spring_proxy_eV_per_sample":spring,
            "basin_label":labels,"hist_fractional":hist}
    summary={"metrics":metrics,"basin_populations":{"A":float(np.mean(labels==0)),"B":float(np.mean(labels==1))},
             "basin_transitions":int(np.count_nonzero(labels[1:]!=labels[:-1])),"circular_centroid":circular,
             "unwrapped_cell_index_ranges":{"u":[int(np.floor(uv_unwrapped[:,0]).min()),int(np.floor(uv_unwrapped[:,0]).max())],
                                             "v":[int(np.floor(uv_unwrapped[:,1]).min()),int(np.floor(uv_unwrapped[:,1]).max())]}}
    return arrays,summary


def run_one(job):
    T,backend,configuration,seed,basin,uv,grid_path,outdir,n_steps,burn,every=job
    grid=load_grid(Path(grid_path)); action,potential=make_action(grid,T)
    A=np.column_stack([grid["lattice_vectors_nm"][0],grid["lattice_vectors_nm"][1]])
    center=grid["origin_nm"]+A@uv; local=.70*math.sqrt(2*(HBAR2_OVER_2M0/MASS)*action._tau)
    common=dict(action=action,local_step_nm=local,global_step_nm=15.,global_move_probability=.20,rng_seed=seed)
    def make():
        if backend=="python" and configuration=="baseline":return PIMCSampler(**common)
        if backend=="python":return PIMCSamplerStaging(**common,staging_segment_lengths=(L,),staging_moves_per_step=1)
        if configuration=="baseline":return PIMCSamplerJIT(**common,grid_size=400,grid_range_nm=1.,boundary_mode="periodic_cell",
            periodic_cell_vectors_nm=grid["lattice_vectors_nm"],periodic_cell_origin_nm=grid["origin_nm"],global_disp_vectors_nm=None,directed_move_frac=0.)
        return PIMCSamplerStagingPeriodicJIT(**common,staging_segment_lengths=(L,),staging_moves_per_step=1)
    sampler=make()
    if backend=="jit":sampler.run(2,0,1,center=center);sampler=make()
    t0=time.perf_counter();result=sampler.run(n_steps,burn,every,center=center);seconds=time.perf_counter()-t0
    samples=np.asarray(result["samples"],dtype=np.float64);arrays,summary=diagnostics(samples,action,potential,grid,seconds)
    task=f"{backend}_{configuration}_T{int(T):03d}_P032_seed{seed}_basin{basin}"
    metadata={"task_id":task,"classification":"development_only_not_paper1_evidence","T_K":T,"P":P,
        "L":None if configuration=="baseline" else L,"backend":backend,"configuration":configuration,"seed":seed,"start_basin":basin,
        "initial_center_nm":center.tolist(),"mass_m0":MASS,"n_steps":n_steps,"burn_in":burn,"sample_every":every,"retained":len(samples),
        "local_step_nm":local,"global_sigma_nm":15.,"global_probability":.20,"wall_seconds":seconds,"grid_sha256":GRID_SHA,
        "acceptance_local":float(result["acceptance_local"]),"acceptance_staging":None if configuration=="baseline" else float(result["acceptance_staging"]),
        "acceptance_global":float(result["acceptance_global"]),**summary}
    npz=Path(outdir)/(task+".npz");np.savez_compressed(npz,samples_unwrapped_nm=samples,metadata_json=np.array(json.dumps(metadata,sort_keys=True)),**arrays)
    metadata["npz_sha256"]=sha(npz);(npz.with_suffix(".json")).write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    return metadata


def tv(a,b):return .5*float(np.sum(np.abs(a-b)))


def compare(T,jobs,outdir):
    by={(j["backend"],j["configuration"]):[x for x in jobs if x["backend"]==j["backend"] and x["configuration"]==j["configuration"]] for j in jobs}
    def pooled(key):
        paths=[Path(outdir)/(j["task_id"]+".npz") for j in by[key]]
        with_arrays=[]
        for p in paths:
            with np.load(p,allow_pickle=False) as z:with_arrays.append(np.array(z["hist_fractional"]))
        h=np.mean(with_arrays,axis=0);return h/h.sum()
    comparisons={}; gains={}
    for backend in ("python","jit"):
        base=by[(backend,"baseline")];stage=by[(backend,"staging_L16")]
        comparisons[f"{backend}_staging_vs_baseline"]={
            "density_TV":tv(pooled((backend,"baseline")),pooled((backend,"staging_L16"))),
            "basin_B_abs_difference":abs(np.mean([x["basin_populations"]["B"] for x in stage])-np.mean([x["basin_populations"]["B"] for x in base])),
            "relative_mean_differences":{name:abs(np.mean([x["metrics"][name]["mean"] for x in stage])-np.mean([x["metrics"][name]["mean"] for x in base]))/max(abs(np.mean([x["metrics"][name]["mean"] for x in base])),1e-15) for name in ("R_g2_nm2","mean_potential_eV","spring_proxy_eV")}}
        gains[backend]={name:sum(x["metrics"][name]["ESS"] for x in stage)/sum(x["wall_seconds"] for x in stage)/(sum(x["metrics"][name]["ESS"] for x in base)/sum(x["wall_seconds"] for x in base)) for name in ("q1","q2","q3","R_g2_nm2")}
    py=by[("python","staging_L16")];ji=by[("jit","staging_L16")]
    comparisons["python_vs_jit_staging"]={"density_TV":tv(pooled(("python","staging_L16")),pooled(("jit","staging_L16"))),
        "basin_B_abs_difference":abs(np.mean([x["basin_populations"]["B"] for x in py])-np.mean([x["basin_populations"]["B"] for x in ji])),
        "relative_mean_differences":{name:abs(np.mean([x["metrics"][name]["mean"] for x in py])-np.mean([x["metrics"][name]["mean"] for x in ji]))/max(abs(np.mean([x["metrics"][name]["mean"] for x in py])),1e-15) for name in ("R_g2_nm2","mean_potential_eV","spring_proxy_eV")}}
    stage=[x for x in jobs if x["configuration"]=="staging_L16"]
    eq=list(comparisons.values())
    checks={"staging_acceptance_positive":bool(min(x["acceptance_staging"] for x in stage)>0),
            "density_TV_le_0p15":bool(max(x["density_TV"] for x in eq)<=.15),
            "basin_B_difference_le_0p15":bool(max(x["basin_B_abs_difference"] for x in eq)<=.15),
            "Rg2_relative_difference_le_0p10":bool(max(x["relative_mean_differences"]["R_g2_nm2"] for x in eq)<=.10),
            "spring_relative_difference_le_0p05":bool(max(x["relative_mean_differences"]["spring_proxy_eV"] for x in eq)<=.05),
            "median_low_mode_gain_ge_1p10_both_backends":bool(min(np.median(list(gains[b].values())) for b in gains)>=1.10)}
    return {"T_K":T,"comparisons":comparisons,"ESS_per_second_gains":gains,"minimum_staging_acceptance":min(x["acceptance_staging"] for x in stage),
            "checks":checks,"verdict":"PASS" if all(checks.values()) else "REVIEW"}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--grid",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--n-steps",type=int,default=N_STEPS);ap.add_argument("--burn-in",type=int,default=BURN);ap.add_argument("--sample-every",type=int,default=EVERY);args=ap.parse_args()
    if sha(args.grid)!=GRID_SHA:raise SystemExit("grid SHA-256 mismatch")
    args.out.mkdir(parents=True,exist_ok=True);jobs=[]
    task_args=[(T,b,c,s,basin,uv,str(args.grid),str(args.out),args.n_steps,args.burn_in,args.sample_every) for T in TEMPERATURES for b,c in CONFIGS for s,basin,uv in JOBS[T]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_one,j) for j in task_args]
        for future in concurrent.futures.as_completed(futures):
            row=future.result();jobs.append(row);print(f"{row['task_id']} wall={row['wall_seconds']:.2f}s",flush=True)
    evaluations={str(int(T)):compare(T,[x for x in jobs if x["T_K"]==T],args.out) for T in TEMPERATURES}
    verdict="PASS" if all(x["verdict"]=="PASS" for x in evaluations.values()) else "REVIEW"
    result={"title":"COM_STAGING_P32_L16_MOIRE_PILOT","classification":"development_only_not_paper1_evidence","production_campaign_run":False,
        "configuration":{"temperatures_K":list(TEMPERATURES),"P":P,"L":L,"mass_m0":MASS,"n_steps":args.n_steps,"burn_in":args.burn_in,
                         "sample_every":args.sample_every,"global_sigma_nm":15.,"global_probability":.2,"grid_sha256":GRID_SHA,
                         "seeds_by_temperature":{str(int(T)):[x[0] for x in JOBS[T]] for T in TEMPERATURES}},
        "chains":sorted(jobs,key=lambda x:x["task_id"]),"evaluations":evaluations,"verdict":verdict}
    path=args.out/"COM_STAGING_P32_L16_MOIRE_PILOT.json";path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"path":str(path),"verdict":verdict,"evaluations":evaluations},indent=2))


if __name__=="__main__":main()
