"""Reproducible OpenFOAM POD/DMD analysis."""
from pathlib import Path
import json, re, sys
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from foam_io import discover_parallel_data, reconstruct_field, read_internal_field
from rom_methods import fit_dmd, dmd_reconstruct, pod_reconstruct, choose_rank, field_metrics, time_metrics, spatial_metrics, track_shock, attach_snapshots, to_lab_grid, shock_error, eigenvalue_table

CASE=Path("/mnt/f/Jahidul_Islam/DetFOAM_1D_NoReact_for_ROM")
OUT=CASE/"ROM_DMD_Journal_Work"; RANKS=[1,2,3,5,10,15,20,30,40,48]; TARGET=.999; TRAIN=30e-6
LABEL={"p":"pressure [Pa]","rho":"density [kg m^-3]","U":"velocity [m s^-1]"}

def setup():
 for n in ("01_case_audit","02_data_extraction","03_preprocessing","04_pod","05_standard_dmd","06_shock_attached_dmd","07_validation","08_comparison","09_figures","10_tables","11_results","12_manuscript","13_reproducibility","src","config"): (OUT/n).mkdir(parents=True,exist_ok=True)
def txt(p,s): p.write_text(s.strip()+"\n",encoding="utf8")
def fig(n):
 for e in ("png","pdf"): plt.savefig(OUT/"09_figures"/(n+"."+e),dpi=220,bbox_inches="tight")
 plt.close()
def ranks(n): return [r for r in RANKS if r<=n]+([] if n in RANKS else [n])
def audit_log():
 t=(CASE/"log.shockTube").read_text(errors="replace")
 get=lambda pat: np.array([float(x) for x in re.findall(pat,t,re.M)])
 c=get(r"Max Courant Number\s*=\s*([0-9.eE+-]+)"); d=get(r"^deltaT\s*=\s*([0-9.eE+-]+)"); z=get(r"^Time\s*=\s*([0-9.eE+-]+)")
 return {"cfl_samples":len(c),"cfl_min":float(c.min()),"cfl_max":float(c.max()),"dt_min_s":float(d.min()),"dt_max_s":float(d.max()),"final_log_time_s":float(z.max())}

def run_frame(method,x,t,Q,xs):
 attached=method=="shock_attached"; dt=float(np.diff(t).mean()); errs=[]; sv=[]; ev=[]; se=[]; keep={}
 for field,q in Q.items():
  xi,work=attach_snapshots(x,q,xs) if attached else (x,q)
  probe=fit_dmd(work,dt,1); mx=min(work.shape[1]-1,len(probe.singular_values)); sel=min(choose_rank(probe.singular_values[:mx],TARGET),mx); effective_sel=fit_dmd(work,dt,sel).rank
  cum=np.cumsum(probe.singular_values**2)/np.sum(probe.singular_values**2)
  for i,s in enumerate(probe.singular_values,1): sv.append({"method":method,"field":field,"rank":i,"singular_value":s,"energy_fraction":s*s/np.sum(probe.singular_values**2),"cumulative_energy":cum[i-1]})
  saved={}
  for r in sorted(set(ranks(mx)+[sel])):
   m=fit_dmd(work,dt,r); a=dmd_reconstruct(m,len(t)); b=to_lab_grid(x,xi,a,xs) if attached else a
   errs.append({"method":method,"field":field,"rank":m.rank,"selected_rank":effective_sel,"support_fraction":float(np.isfinite(b).mean()),"fit_seconds":m.elapsed_s,"storage_real_values":int(m.mean.size+2*m.modes.size+2*m.eigenvalues.size+2*m.amplitudes.size),**field_metrics(q,b)})
   if field=="p":
    est=np.array([track_shock(xi,a[:,j])[0]+xs[j] for j in range(len(t))]) if attached else np.array([track_shock(x,b[:,j])[0] for j in range(len(t))])
    se.append({"method":method,"rank":m.rank,**shock_error(xs,est)})
    if r==sel:saved["shock"]=est
   if r==sel:
    saved.update({"model":m,"xi":xi,"rec_work":a,"rec":b})
    e=eigenvalue_table(m);e.insert(0,"method",method);e.insert(1,"field",field);ev.append(e)
    z=time_metrics(q,b);z["time_s"]=t;z.to_csv(OUT/"07_validation"/(method+"_"+field+"_time_errors.csv"),index=False)
    z=spatial_metrics(q,b);z["x_m"]=x;z.to_csv(OUT/"07_validation"/(method+"_"+field+"_spatial_errors.csv"),index=False)
  if "model" not in saved:
   m=fit_dmd(work,dt,min(sel,mx))
   a=dmd_reconstruct(m,len(t))
   b=to_lab_grid(x,xi,a,xs) if attached else a
   saved.update({"model":m,"xi":xi,"rec_work":a,"rec":b})
   if field=="p":
    saved["shock"]=np.array([track_shock(xi,a[:,j])[0]+xs[j] for j in range(len(t))]) if attached else np.array([track_shock(x,b[:,j])[0] for j in range(len(t))])
   e=eigenvalue_table(m);e.insert(0,"method",method);e.insert(1,"field",field);ev.append(e)
  keep[field]=saved
 return pd.DataFrame(errs),pd.DataFrame(sv),pd.concat(ev,ignore_index=True),pd.DataFrame(se),keep

def run_pod(Q):
 rows=[];spec=[]
 for f,q in Q.items():
  _,s=pod_reconstruct(q,1);n=min(len(s),q.shape[1]);sel=choose_rank(s[:n],TARGET);cu=np.cumsum(s*s)/np.sum(s*s)
  for i,v in enumerate(s,1):spec.append({"field":f,"rank":i,"singular_value":v,"cumulative_energy":cu[i-1]})
  for r in ranks(n):
   a,_=pod_reconstruct(q,r); rows.append({"method":"pod","field":f,"rank":r,"selected_rank":sel,"storage_real_values":q.shape[0]*r+r*q.shape[1]+q.shape[0],**field_metrics(q,a)})
 return pd.DataFrame(rows),pd.DataFrame(spec)

def combo(method,x,t,Q,xs):
 att=method=="shock_attached"; parts=[]; meta={}; start=0;xi=x
 for f,q in Q.items():
  xi,a=attach_snapshots(x,q,xs) if att else (x,q);mean=a.mean(1,keepdims=True);scale=float(np.std(a-mean));parts.append((a-mean)/scale);meta[f]=(start,start+a.shape[0],mean,scale);start+=a.shape[0]
 state=np.vstack(parts);dt=float(np.diff(t).mean());p=fit_dmd(state,dt,1);r=min(choose_rank(p.singular_values,TARGET),state.shape[1]-1);m=fit_dmd(state,dt,r);rec=dmd_reconstruct(m,len(t));rows=[]
 for f,(i,j,mean,scale) in meta.items():
  a=rec[i:j]*scale+mean;b=to_lab_grid(x,xi,a,xs) if att else a;rows.append({"method":method,"state":"combined_scaled_p_rho_u","field":f,"rank":r,"selected_rank":r,**field_metrics(Q[f],b)})
 return pd.DataFrame(rows)

def predict(x,t,p,xs,rs,ra):
 tr=t<=TRAIN+1e-15;te=~tr;dt=float(np.diff(t).mean());n=tr.sum();rows=[]
 m=fit_dmd(p[:,tr],dt,min(rs,n-1));a=dmd_reconstruct(m,len(t));z=np.array([track_shock(x,a[:,j])[0] for j in range(len(t))])
 xi,w=attach_snapshots(x,p[:,tr],xs[tr]);ma=fit_dmd(w,dt,min(ra,n-1));b=dmd_reconstruct(ma,len(t));line=np.polyfit(t[tr],xs[tr],1);frame=np.polyval(line,t);c=to_lab_grid(x,xi,b,frame);za=np.array([track_shock(xi,b[:,j])[0]+frame[j] for j in range(len(t))])
 for scope,mask in (("train",tr),("test",te)):
  rows += [{"method":"standard_dmd","scope":scope,"rank":m.rank,"n_snapshots":int(mask.sum()),**field_metrics(p[:,mask],a[:,mask])},{"method":"standard_dmd_shock","scope":scope,"rank":m.rank,"n_snapshots":int(mask.sum()),**shock_error(xs[mask],z[mask])},{"method":"shock_attached_dmd","scope":scope,"rank":ma.rank,"n_snapshots":int(mask.sum()),**field_metrics(p[:,mask],c[:,mask])},{"method":"shock_attached_dmd_shock","scope":scope,"rank":ma.rank,"n_snapshots":int(mask.sum()),**shock_error(xs[mask],za[mask])}]
 return pd.DataFrame(rows),a,c,z,za,frame

def plots(x,t,Q,xs,S,A,sv,err,se,pred):
 for f,q in Q.items():
  plt.figure(figsize=(7.2,3.2)); im=plt.pcolormesh(x,t*1e6,q.T,shading="auto",rasterized=True);plt.colorbar(im,label=LABEL[f]);plt.xlabel("x [m]");plt.ylabel("t [microseconds]");plt.title("Full-order "+f+" evolution");fig("full_order_"+f+"_evolution")
 try:
  plt.figure(figsize=(7,3.2));plt.plot(x,read_internal_field(CASE/"0"/"p")/1000);plt.xlabel("x [m]");plt.ylabel("p [kPa]");plt.title("Verified initial pressure field");fig("initial_shock_tube_configuration")
 except Exception:pass
 plt.figure(figsize=(7,3.3));plt.plot(t*1e6,xs*1e3,"k",label="FOM");plt.plot(t*1e6,S["p"]["shock"]*1e3,"--",label="standard");plt.plot(t*1e6,A["p"]["shock"]*1e3,":",lw=2,label="attached");plt.legend();plt.xlabel("t [microseconds]");plt.ylabel("shock x [mm]");plt.title("Shock position");fig("shock_position_validation")
 plt.figure(figsize=(7,3.3))
 for m,st in (("standard","-o"),("shock_attached","--s")):
  d=sv[(sv.method==m)&(sv.field=="p")];plt.semilogy(d["rank"],d["singular_value"],st,ms=3,label=m)
 plt.legend();plt.xlabel("rank");plt.ylabel("singular value");plt.title("Pressure singular spectrum");fig("singular_value_spectrum")
 m=S["p"]["model"];e=m.eigenvalues;th=np.linspace(0,2*np.pi,400);plt.figure(figsize=(4.5,4.3));plt.plot(np.cos(th),np.sin(th),"k:");plt.scatter(e.real,e.imag,c=np.abs(m.amplitudes));plt.gca().set_aspect("equal");plt.xlabel("real lambda");plt.ylabel("imag lambda");plt.title("DMD eigenvalues");fig("dmd_eigenvalue_spectrum")
 ind=np.argsort(np.abs(m.amplitudes))[::-1][:min(6,m.rank)];ff,ax=plt.subplots(len(ind),1,figsize=(7,1.4*len(ind)),sharex=True)
 for z,k in zip(np.atleast_1d(ax),ind):y=m.modes[:,k].real;z.plot(x,y/max(abs(y).max(),1e-30));z.set_ylabel(str(k+1))
 np.atleast_1d(ax)[-1].set_xlabel("x [m]");ff.suptitle("Amplitude-ranked real pressure modes");fig("dominant_dmd_pressure_modes")
 ff,ax=plt.subplots(1,3,figsize=(11,3.1))
 for z,f in zip(ax,Q):
  for m in ("standard","shock_attached","pod"):
   d=err[(err.method==m)&(err.field==f)];z.semilogy(d["rank"],d["relative_frobenius"],"-o",ms=3,label=m)
  z.set_title(f);z.set_xlabel("rank");z.set_ylabel("relative Frobenius");z.legend(fontsize=7)
 fig("reconstruction_error_vs_rank")
 plt.figure(figsize=(6.8,3.3))
 for m,st in (("standard","-o"),("shock_attached","--s")):
  d=se[se.method==m];plt.semilogy(d["rank"],d["rmse_m"]*1e3,st,label=m)
 plt.legend();plt.xlabel("rank");plt.ylabel("shock RMSE [mm]");plt.title("Shock error by rank");fig("shock_location_error_vs_rank")
 ii=[np.argmin(abs(t-k*1e-6)) for k in (1,10,20,30,40,50)]
 for f,q in Q.items():
  ff,ax=plt.subplots(2,3,figsize=(11,5.2),sharex=True)
  for z,j in zip(ax.ravel(),ii):
   z.plot(x,q[:,j],"k",label="FOM");z.plot(x,S[f]["rec"][:,j],"--",label="standard");z.plot(x,A[f]["rec"][:,j],":",label="attached");z.set_title(str(round(t[j]*1e6))+" us")
  ax[0,0].legend(fontsize=7);ff.supxlabel("x [m]");ff.supylabel(LABEL[f]);fig("fom_rom_"+f+"_representative_times")
 plt.figure(figsize=(7.2,3.2));im=plt.pcolormesh(x,t*1e6,(abs(Q["p"]-S["p"]["rec"])/(Q["p"].max()-Q["p"].min())).T,shading="auto",cmap="magma");plt.colorbar(im,label="normalized abs error");plt.xlabel("x [m]");plt.ylabel("t [microseconds]");plt.title("Standard-DMD pressure error");fig("standard_dmd_pressure_error_map")
 ps,pa,*_=pred;plt.figure(figsize=(7,3.3));plt.plot(x,Q["p"][:,-1]/1e3,"k",label="FOM");plt.plot(x,ps[:,-1]/1e3,"--",label="standard");plt.plot(x,pa[:,-1]/1e3,":",label="attached");plt.legend();plt.xlabel("x [m]");plt.ylabel("p [kPa]");plt.title("Held-out result at 50 us");fig("training_testing_prediction")
 d=err[(err["rank"]==err["selected_rank"])&err.method.isin(["standard","shock_attached","pod"])];plt.figure(figsize=(7,3.3));plt.bar(range(len(d)),d.relative_frobenius);plt.xticks(range(len(d)),[a.method+" "+a.field for a in d.itertuples()],rotation=35,ha="right");plt.ylabel("relative Frobenius");plt.title("Selected rank comparison");fig("standard_vs_shock_attached_comparison")

def docs(snap,E,SE,P,log):
 pick=lambda m:E[(E.method==m)&(E.field=="p")&(E["rank"]==E["selected_rank"])].iloc[0]
 a,b=pick("standard"),pick("shock_attached");sa=SE[(SE.method=="standard")&(SE["rank"]==a["rank"])].iloc[0];sb=SE[(SE.method=="shock_attached")&(SE["rank"]==b["rank"])].iloc[0]
 txt(OUT/"README.md",f"# ROM/DMD workflow\n\nRun python3 src/run_workflow.py. The package reconstructs {snap['n_snapshots']} common p/rho/U snapshots on {snap['n_cells']} cells through cellProcAddressing, not raw processor concatenation.")
 txt(OUT/"01_case_audit"/"case_audit.md",f"""# Case audit\n\nOpenFOAM headers: control/mesh version 8 and fvSchemes/fvSolution version 7. Application detonationFoam. Flux HLLCP; ddt Euler; gradients cellLimited Gauss linear 0.7; div default Gauss linear with vanLeer for rhoPhi Yi_h; Minmod/MinmodV reconstructions. Thermo is hePsiThermo multiComponentMixture Sutherland JANAF perfectGas sensibleInternalEnergy. chemistry is off and combustionModel none. Mesh 5000x1x1 over x=-0.05 to 0.05 m. Eight processor maps cover every global cell exactly once. Output is {snap['n_snapshots']} common snapshots from 1 to 50 us at dt={snap['dt_s']:.6g} s. setFields and time-zero fields verify p_L=100000 Pa, T_L=348.4 K, u_L=0 and p_R=10000 Pa, T_R=278.7 K, u_R=0. Log Cfl min/max {log['cfl_min']:.6g}/{log['cfl_max']:.6g}, adaptive dt {log['dt_min_s']:.6g} to {log['dt_max_s']:.6g}. Its leading-shock diagnostic stays zero, so field-based tracking is used.""")
 txt(OUT/"11_results"/"methodology.md","# Methodology\n\nExact DMD uses temporally mean-subtracted snapshots X and X-prime with SVD truncation. The combined p/rho/u state standardizes each centered variable by its global space-time standard deviation. POD is the separate energy-optimal centered projection. The shock reference is maximum positive dp/dx with parabolic subcell refinement; alternate pressure/density candidates are saved. Attached DMD uses xi=x-xs. Its in-sample inverse map uses measured xs and thus tests conditional representation; held-out attached results use only a training-fit linear xs coordinate.")
 summary=f"""# Results summary\n\nSelected rank is first at {TARGET:.1%} singular energy.\n\n|model|p rank|relative Frobenius|shock RMSE mm|\n|-|-:|-:|-:|\n|standard|{int(a['rank'])}|{a['relative_frobenius']:.6g}|{sa['rmse_m']*1e3:.6g}|\n|attached|{int(b['rank'])}|{b['relative_frobenius']:.6g}|{sb['rmse_m']*1e3:.6g}|\n\nInterpret attached metrics with support fraction and known-frame qualification. Held-out records are in 10_tables/training_testing_performance.csv.\n"""
 txt(OUT/"11_results"/"results_summary.md",summary)
 txt(OUT/"11_results"/"limitations.md","# Limitations\n\nOne mesh prevents grid-convergence evidence. rho is absent at time zero, so common data start at 1 us. No external exact Riemann solution is supplied or fabricated; JANAF thermodynamics makes constant-gamma comparison an additional approximation. In-sample is reconstruction, not prediction. Attached inverse mapping requires a translation coordinate. No claim extends beyond the recorded 30/50 us held-out experiment.")
 txt(OUT/"13_reproducibility"/"reproducibility.md","# Reproducibility\n\nInstall requirements.txt and run python3 src/run_workflow.py from this directory. It validates shared time directories, field availability, finite values, uniform time spacing, cell-map coverage and ordered x. Exact extracted data are 02_data_extraction/snapshots_p_rho_u.npz.")
 txt(OUT/"12_manuscript"/"manuscript_draft.md",f"""# Reduced-order representation of translating compressible shock dynamics using standard and shock-attached DMD\n\n## Introduction\nThis is a reproducible numerical study, not a novelty claim.\n## Problem formulation\nThe 0.1 m, 5000-cell audited shock tube has verified initialized states in case_audit.md.\n## Numerical method\nThe audited solver uses HLLCP and Euler; chemistry is off.\n## Snapshot generation\nFifty parallel p/rho/u fields are rebuilt through cellProcAddressing.\n## DMD methodology\nCentered exact DMD, scaled combined state and POD are compared.\n## Shock-attached transformation\nxi=x-xs(t) is used with explicit translation coordinate.\n## Validation\nField rank sweeps, shock errors, and training-through-30-us testing are reported.\n## Results\nSelected pressure: standard rank {int(a['rank'])}, error {a['relative_frobenius']:.6g}; attached rank {int(b['rank'])}, error {b['relative_frobenius']:.6g}.\n## Discussion\nThe CSV comparison determines whether alignment reduces rank; it is not presumed.\n## Limitations\nSee 11_results/limitations.md.\n## Conclusions\nThe package supports reconstruction and one scoped held-out test, not mesh-converged general prediction.\n## References to verify\nNo references are fabricated; independently verify DMD, POD, shock alignment and Riemann-problem sources before use.""")
 txt(OUT/"FINAL_RESEARCH_SUMMARY.md",f"""# Final research summary\n\n## Research question\nCan standard and attached DMD represent audited translating shock snapshots?\n## Case description\nNon-reacting configured detonationFoam HLLCP case, 5000 cells, 50 p/rho/u snapshots.\n## Methods\nPOD, centered DMD, attached DMD, shock tracking, rank sweeps, and 30/50-us test.\n## Main numerical findings\nStandard p: rank {int(a['rank'])}, error {a['relative_frobenius']:.6g}, shock RMSE {sa['rmse_m']:.6g} m. Attached p: rank {int(b['rank'])}, error {b['relative_frobenius']:.6g}, shock RMSE {sb['rmse_m']:.6g} m.\n## Best model\nMetric- and coordinate-availability dependent; tables are authoritative.\n## Training/testing\nSee 10_tables/training_testing_performance.csv; no larger predictive claim is made.\n## Limitations and next step\nAdd mesh convergence and an explicitly justified exact/high-fidelity Riemann reference, then test a richer shock-coordinate predictor.""")

def main():
 setup();info=discover_parallel_data(CASE,["p","rho","U"]);x,t=info["x"],info["times"]
 if len(t)<3 or not np.allclose(np.diff(t),np.diff(t).mean(),rtol=1e-8,atol=1e-14):raise ValueError("Time spacing unsuitable for discrete DMD")
 V={"p":[],"rho":[],"U":[]};quality=[]
 for z in t:
  p=reconstruct_field(info,float(z),"p");r=reconstruct_field(info,float(z),"rho");u=reconstruct_field(info,float(z),"U")[:,0]
  for f,a in (("p",p),("rho",r),("U",u)):V[f].append(a);quality.append({"time_s":z,"field":f,"n_values":len(a),"finite":bool(np.isfinite(a).all()),"min":a.min(),"max":a.max()})
 Q={f:np.column_stack(a) for f,a in V.items()};shock=[];cand=[]
 for j,z in enumerate(t):a,c=track_shock(x,Q["p"][:,j],Q["rho"][:,j]);shock.append(a);cand.append({"time_s":z,"x_s_m":a,**c})
 xs=np.array(shock);snap={"n_snapshots":len(t),"n_cells":len(x),"processors":len(info["processors"]),"time_min_s":float(t.min()),"time_max_s":float(t.max()),"dt_s":float(np.diff(t).mean()),"dx_m":float(np.median(np.diff(x)))}
 log=audit_log();(OUT/"01_case_audit"/"case_audit.json").write_text(json.dumps({"snapshot":snap,"log":log,"missing":{str(k):v for k,v in info["missing"].items()}},indent=2))
 pd.DataFrame([snap]).to_csv(OUT/"10_tables"/"snapshot_information.csv",index=False);pd.DataFrame(quality).to_csv(OUT/"03_preprocessing"/"data_quality_checks.csv",index=False);pd.DataFrame(cand).to_csv(OUT/"10_tables"/"shock_candidates.csv",index=False)
 np.savez_compressed(OUT/"02_data_extraction"/"snapshots_p_rho_u.npz",x_m=x,time_s=t,p_pa=Q["p"],rho_kg_m3=Q["rho"],u_m_s=Q["U"],shock_x_m=xs)
 S=run_frame("standard",x,t,Q,xs);A=run_frame("shock_attached",x,t,Q,xs);P,PS=run_pod(Q);E=pd.concat([S[0],A[0],P],ignore_index=True);SV=pd.concat([S[1],A[1]],ignore_index=True);EV=pd.concat([S[2],A[2]],ignore_index=True);SE=pd.concat([S[3],A[3]],ignore_index=True);C=pd.concat([combo("standard",x,t,Q,xs),combo("shock_attached",x,t,Q,xs)])
 PR=predict(x,t,Q["p"],xs,S[4]["p"]["model"].rank,A[4]["p"]["model"].rank)
 for d,n in ((E,"reconstruction_errors.csv"),(SV,"dmd_singular_values.csv"),(PS,"pod_singular_values.csv"),(EV,"dmd_eigenvalues.csv"),(SE,"shock_location_errors.csv"),(PR[0],"training_testing_performance.csv")):d.to_csv(OUT/"10_tables"/n,index=False)
 E[(E["rank"]==E["selected_rank"])&E.method.isin(["standard","shock_attached","pod"])].to_csv(OUT/"08_comparison"/"standard_vs_attached.csv",index=False);C.to_csv(OUT/"08_comparison"/"combined_state_dmd.csv",index=False)
 pd.DataFrame({"time_s":t,"x_s_full_m":xs,"x_s_standard_m":S[4]["p"]["shock"],"x_s_attached_m":A[4]["p"]["shock"],"x_s_standard_prediction_m":PR[3],"x_s_attached_prediction_m":PR[4],"attached_frame_prediction_m":PR[5]}).to_csv(OUT/"07_validation"/"shock_positions.csv",index=False)
 pd.DataFrame([["application","detonationFoam"],["fluxScheme","HLLCP"],["ddtScheme","Euler"],["chemistry","off"],["combustionModel","none"],["equationOfState","perfectGas"],["mesh_cells",5000]],columns=["parameter","value"]).to_csv(OUT/"10_tables"/"case_parameters.csv",index=False)
 plots(x,t,Q,xs,S[4],A[4],SV,E,SE,PR[1:]);docs(snap,E,SE,PR[0],log)
 (OUT/"13_reproducibility"/"environment_manifest.json").write_text(json.dumps({"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__,"parameters":{"energy_target":TARGET,"train_end_s":TRAIN}},indent=2))
 print(json.dumps({"status":"complete","snapshots":len(t),"cells":len(x),"standard_pressure_rank":S[4]["p"]["model"].rank,"attached_pressure_rank":A[4]["p"]["model"].rank}))
if __name__=="__main__":main()
