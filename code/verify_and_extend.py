import sys, math, time, json, platform, os, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
sys.path.insert(0,str(Path(__file__).resolve().parent))
import solver as q


def min_angle_deg(mesh):
    V=mesh.p[mesh.t]
    mins=180.0
    for tri in V:
        for k in range(3):
            v1=tri[(k+1)%3]-tri[k]; v2=tri[(k+2)%3]-tri[k]
            cs=np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            ang=np.degrees(np.arccos(np.clip(cs,-1,1)))
            mins=min(mins,ang)
    return mins


def reconstruct_transition(p,sn,lev):
    oldmesh,oldu,*_=sn[lev-1]
    mesh,ud,A,b,I,B,eta=sn[lev]
    marked=q.dorfler_mark(sn[lev-1][-1],.5)
    remesh,parents,_=q.refine_marked(oldmesh,marked)
    assert remesh.nnodes==mesh.nnodes and np.allclose(remesh.p,mesh.p) and np.array_equal(remesh.t,mesh.t)
    u0=q.prolong_midpoint(oldu,mesh,parents,p.g)
    return oldmesh,mesh,u0,ud,A,b,I,B


def main(outdir, run_timing=False, primary_dir=None):
    OUT=Path(outdir); OUT.mkdir(parents=True,exist_ok=True)
    primary=Path(primary_dir) if primary_dir is not None else Path(__file__).resolve().parents[1]/'data'
    validation=[]
    all_snaps={}
    for name,p in q.problems().items():
        n0=8 if p.mesh_kind=='square' else 4
        _,sn=q.afem_sequence(p,n0=n0,levels=7,theta=.5); all_snaps[name]=sn
        mesh=sn[-1][0]
        ma=min_angle_deg(mesh)
        validation.append(dict(Test='minimum_angle',Problem=name,Value=ma,Tolerance=30.0,Pass=ma>=30.0))
        oldmesh,mesh,u0,ud,A,b,I,B=reconstruct_transition(p,sn,7)
        ug,_,_=q.correction_local_assembled(p,mesh,u0,I)
        diff=np.max(np.abs(ug-ud))
        validation.append(dict(Test='global_correction_equals_direct',Problem=name,Value=diff,Tolerance=5e-12,Pass=diff<5e-12))
        act=q.active_nodes(mesh,oldmesh.nnodes,1)
        uf,_=q.correction_restricted(p,mesh,u0,act,A,b)
        ul,_,_=q.correction_local_assembled(p,mesh,u0,act)
        diff2=np.max(np.abs(uf-ul))
        validation.append(dict(Test='local_vs_global_restricted_assembly',Problem=name,Value=diff2,Tolerance=5e-12,Pass=diff2<5e-12))
        e0=ud-u0; d=ul-u0; er=ud-ul
        lhs=float(e0@(A@e0)); rhs=float(er@(A@er)+d@(A@d)); rel=abs(lhs-rhs)/max(abs(lhs),1e-30)
        validation.append(dict(Test='active_subspace_pythagorean',Problem=name,Value=rel,Tolerance=2e-11,Pass=rel<2e-11))
        rv,dg=q.residual_vector_diag(p,mesh,ul); rr=b-A@ul
        rdiff=np.max(np.abs(rv-rr))
        validation.append(dict(Test='matrixfree_residual_equals_sparse',Problem=name,Value=rdiff,Tolerance=1e-11,Pass=rdiff<1e-11))

    lin=q.Problem(
        'P1 patch test','square',
        lambda x:np.ones(x.shape[1]),
        lambda x:np.zeros(x.shape[1]),
        lambda x:np.zeros(x.shape[1]),
        lambda x:x[0]+2*x[1],
        lambda x:np.vstack((np.ones_like(x[0]),2*np.ones_like(x[0]))),
        lambda x:x[0]+2*x[1])
    mesh=q.make_square(8); A,b=q.assemble(lin,mesh); u,*_=q.solve(lin,mesh,A,b)
    patch=np.max(np.abs(u-lin.exact(mesh.p.T)))
    validation.append(dict(Test='P1_linear_patch_test',Problem='P1 patch test',Value=patch,Tolerance=5e-13,Pass=patch<5e-13))

    n=64; h=1/n; m=n-1
    T=diags([-np.ones(m-1),2*np.ones(m),-np.ones(m-1)],[-1,0,1],format='csr')
    L=(kron(eye(m),T)+kron(T,eye(m)))/(h*h)
    x=np.arange(1,n)*h; X,Y=np.meshgrid(x,x,indexing='xy'); f=2*np.pi*np.pi*np.sin(np.pi*X)*np.sin(np.pi*Y)
    ufd=spsolve(L,f.ravel()).reshape(m,m)
    pex=q.problems()['Smooth Poisson']; mf=q.make_square(n); Af,bf=q.assemble(pex,mf); ufe,*_=q.solve(pex,mf,Af,bf)
    ufei=np.array([[ufe[i+j*(n+1)] for i in range(1,n)] for j in range(1,n)])
    fd_fe=np.max(np.abs(ufd-ufei)); fd_exact=np.max(np.abs(ufd-np.sin(np.pi*X)*np.sin(np.pi*Y)))
    validation.append(dict(Test='independent_FD_vs_FEM_n64',Problem='Smooth Poisson',Value=fd_fe,Tolerance=2e-3,Pass=fd_fe<2e-3))
    validation.append(dict(Test='independent_FD_exact_error_n64',Problem='Smooth Poisson',Value=fd_exact,Tolerance=2e-3,Pass=fd_exact<2e-3))

    vdf=pd.DataFrame(validation); vdf.to_csv(OUT/'verification_checks.csv',index=False)

    # Use the requested primary CSV directory for derived audits.
    uni=pd.read_csv(primary/'uniform.csv'); af=pd.read_csv(primary/'afem.csv')
    slopes=[]
    for prob in uni.Problem.unique():
        for method,df in [('Uniform',uni[uni.Problem==prob]),('AFEM-direct',af[af.Problem==prob])]:
            d=df.sort_values('DOF').tail(4)
            for metric in ['Energy','L2']:
                slope=float(np.polyfit(np.log(d.DOF),np.log(d[metric]),1)[0])
                slopes.append(dict(Problem=prob,Method=method,Metric=metric,Slope=slope,DOF_min=int(d.DOF.min()),DOF_max=int(d.DOF.max())))
    pd.DataFrame(slopes).to_csv(OUT/'convergence_slopes.csv',index=False)

    sens=[]
    for name in ['L-shaped singularity','High-contrast interface (1e4)','Localized reaction-diffusion layer']:
        p=q.problems()[name]; n0=8 if p.mesh_kind=='square' else 4
        for theta in [.3,.5,.7]:
            df,sn=q.afem_sequence(p,n0=n0,levels=7,theta=theta)
            row=df.iloc[-1]
            sens.append(dict(Problem=name,Theta=theta,Level=7,DOF=int(row.DOF),Energy=row.Energy,Estimator=row.Estimator,Cumulative_s=row.Cumulative_s,MinAngle=min_angle_deg(sn[-1][0])))
    pd.DataFrame(sens).to_csv(OUT/'dorfler_sensitivity.csv',index=False)

    p0=q.problems()['High-contrast interface (1e4)']
    p_un=q.Problem('Interface-unweighted','square',p0.D,p0.c,p0.f,p0.exact,p0.grad,p0.g)
    # The original ablation deliberately disabled coefficient-aware weighting by changing the name.
    dfu,_=q.afem_sequence(p_un,n0=8,levels=7,theta=.5)
    dfr,_=q.afem_sequence(p0,n0=8,levels=7,theta=.5)
    ab=pd.concat([dfu.assign(EstimatorType='Unweighted residual'),dfr.assign(EstimatorType='Coefficient-weighted residual')],ignore_index=True)
    ab.to_csv(OUT/'interface_estimator_ablation.csv',index=False)

    taurows=[]
    for name,p in q.problems().items():
        sn=all_snaps[name]
        for lev in [5,6,7]:
            oldmesh,mesh,u0,ud,A,b,I,B=reconstruct_transition(p,sn,lev)
            r0=q.scaled_residual_matrixfree(p,mesh,u0,I)
            cands=[]
            for kr in range(5):
                act=q.active_nodes(mesh,oldmesh.nnodes,kr); ul,nt,nnz=q.correction_local_assembled(p,mesh,u0,act)
                rr=q.scaled_residual_matrixfree(p,mesh,ul,I)/(r0 if r0 else 1.0)
                en=q.errors(p,mesh,ul)[2]
                cands.append((kr,act,ul,rr,en,nt,nnz))
            direct=q.errors(p,mesh,ud)[2]
            for tau in [.05,.10,.15,.25,.40]:
                chosen=next((c for c in cands if c[3]<=tau),cands[-1])
                kr,act,ul,rr,en,nt,nnz=chosen
                taurows.append(dict(Problem=name,Level=lev,Tau=tau,Halo=kr,ActiveDOF=len(act),InteriorDOF=len(I),ActiveFraction=len(act)/len(I),ResidualReduction=rr,Energy=en,DirectEnergy=direct,EnergyRatio=en/direct,TouchedCells=nt,LocalNNZ=nnz))
    pd.DataFrame(taurows).to_csv(OUT/'rchc_tau_sensitivity.csv',index=False)

    if run_timing:
        def stats(a):
            a=np.asarray(a,float); return dict(Median_s=float(np.median(a)),Q25_s=float(np.quantile(a,.25)),Q75_s=float(np.quantile(a,.75)),Mean_s=float(a.mean()),SD_s=float(a.std(ddof=1)))
        trows=[]; repeats=25
        for name,p in q.problems().items():
            sn=all_snaps[name]
            for lev in [5,6,7]:
                oldmesh,mesh,u0,ud,Aref,bref,I,B=reconstruct_transition(p,sn,lev)
                for _ in range(3):
                    A,b=q.assemble(p,mesh); q.solve(p,mesh,A,b)
                vals=[]
                for _ in range(repeats):
                    t=time.perf_counter(); A,b=q.assemble(p,mesh); q.solve(p,mesh,A,b); vals.append(time.perf_counter()-t)
                row=dict(Problem=name,Level=lev,Method='Direct assembly+solve',DOF=mesh.nnodes,InteriorDOF=len(I),ActiveDOF=len(I),ActiveFraction=1.0,TouchedCells=mesh.nelems,LocalNNZ=Aref[I][:,I].nnz,Repeats=repeats); row.update(stats(vals)); trows.append(row)
                for kr in [0,1,2]:
                    act=q.active_nodes(mesh,oldmesh.nnodes,kr)
                    for _ in range(3): q.correction_local_assembled(p,mesh,u0,act)
                    vals=[]; nt=nnz=0
                    for _ in range(repeats):
                        t=time.perf_counter(); ul,nt,nnz=q.correction_local_assembled(p,mesh,u0,act); vals.append(time.perf_counter()-t)
                    row=dict(Problem=name,Level=lev,Method=f'HLRC-r{kr} local assembly+solve',DOF=mesh.nnodes,InteriorDOF=len(I),ActiveDOF=len(act),ActiveFraction=len(act)/len(I),TouchedCells=nt,LocalNNZ=nnz,Repeats=repeats); row.update(stats(vals)); trows.append(row)
                vals=[]
                for _ in range(3): q.residual_estimator(p,mesh,ud,robust_diffusion=('High-contrast' in name))
                for _ in range(repeats):
                    t=time.perf_counter(); q.residual_estimator(p,mesh,ud,robust_diffusion=('High-contrast' in name)); vals.append(time.perf_counter()-t)
                row=dict(Problem=name,Level=lev,Method='Residual estimator',DOF=mesh.nnodes,InteriorDOF=len(I),ActiveDOF=0,ActiveFraction=0,TouchedCells=mesh.nelems,LocalNNZ=0,Repeats=repeats); row.update(stats(vals)); trows.append(row)
        pd.DataFrame(trows).to_csv(OUT/'timing_25reps.csv',index=False)

    summary={'all_verification_pass':bool(vdf.Pass.all()),'fd_fem_maxdiff':float(fd_fe),'fd_exact_maxerr':float(fd_exact)}
    (OUT/'extended_summary.json').write_text(json.dumps(summary,indent=2))
    print(vdf.to_string(index=False))
    print('ALL PASS',bool(vdf.Pass.all()))
    print('Wrote',OUT)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir',default=str(Path(__file__).resolve().parents[1]/'regenerated_extended'))
    ap.add_argument('--primary-dir',default=str(Path(__file__).resolve().parents[1]/'data'))
    ap.add_argument('--timing',action='store_true')
    args=ap.parse_args()
    main(args.outdir,args.timing,args.primary_dir)
