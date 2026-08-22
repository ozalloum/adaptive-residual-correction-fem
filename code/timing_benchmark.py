import os, sys, time, json, platform
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import solver as q
from verify_and_extend import reconstruct_transition

REPEATS=25
rows=[]
for name,p in q.problems().items():
    n0=8 if p.mesh_kind=='square' else 4
    _,sn=q.afem_sequence(p,n0=n0,levels=7,theta=.5)
    oldmesh,mesh,u0,ud,Aref,bref,I,B=reconstruct_transition(p,sn,7)
    methods=[('Direct assembly+solve',None)] + [(f'HLRC-r{r} local assembly+solve',r) for r in (0,1,2)]
    for meth,r in methods:
        vals=[]
        if r is None:
            for _ in range(3):
                A,b=q.assemble(p,mesh); q.solve(p,mesh,A,b)
            for _ in range(REPEATS):
                t=time.perf_counter(); A,b=q.assemble(p,mesh); q.solve(p,mesh,A,b); vals.append(time.perf_counter()-t)
            active=len(I); frac=1.0; touched=mesh.nelems; nnz=Aref[I][:,I].nnz
        else:
            act=q.active_nodes(mesh,oldmesh.nnodes,r)
            for _ in range(3): q.correction_local_assembled(p,mesh,u0,act)
            for _ in range(REPEATS):
                t=time.perf_counter(); ul,touched,nnz=q.correction_local_assembled(p,mesh,u0,act); vals.append(time.perf_counter()-t)
            active=len(act); frac=len(act)/len(I)
        a=np.asarray(vals)
        rows.append(dict(Problem=name,Level=7,Method=meth,DOF=mesh.nnodes,InteriorDOF=len(I),ActiveDOF=active,ActiveFraction=frac,TouchedCells=touched,LocalNNZ=nnz,Repeats=REPEATS,Median_s=float(np.median(a)),Q25_s=float(np.quantile(a,.25)),Q75_s=float(np.quantile(a,.75)),Mean_s=float(a.mean()),SD_s=float(a.std(ddof=1))))
out=ROOT/'data'/'timing_final_25reps.csv'
pd.DataFrame(rows).to_csv(out,index=False)
env=dict(purpose='final-level timing audit',python=platform.python_version(),platform=platform.platform(),cpu_count_visible=os.cpu_count(),numpy=np.__version__,scipy=__import__('scipy').__version__,pandas=pd.__version__,repeats=REPEATS,thread_env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
(ROOT/'timing_environment.json').write_text(json.dumps(env,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
