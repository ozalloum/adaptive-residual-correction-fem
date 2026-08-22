import os, math
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
VALIDATION = os.path.join(ROOT, 'validation')
os.makedirs(VALIDATION, exist_ok=True)

def load(name):
    return pd.read_csv(os.path.join(DATA, name))

checks=[]

def record(name, value, tol, ok, note=''):
    checks.append(dict(check=name, value=value, tolerance=tol, pass_=bool(ok), note=note))

# 1. All archived verification checks pass.
v = load('verification_checks.csv')
record('archived_verification_matrix', int(v['Pass'].sum()), len(v), bool(v['Pass'].all()), f"{int(v['Pass'].sum())}/{len(v)} pass")

# 2. Recompute convergence slopes from raw uniform/AFEM data.
u=load('uniform.csv'); a=load('afem.csv'); s=load('convergence_slopes.csv')
max_slope_diff=0.0
for _,r in s.iterrows():
    df = u[u.Problem==r.Problem].copy() if r.Method=='Uniform' else a[a.Problem==r.Problem].copy()
    x = df['DOF'].to_numpy(dtype=float)
    y = df[r.Metric].to_numpy(dtype=float)
    mask=(x>=r.DOF_min)&(x<=r.DOF_max)
    xx=np.log(x[mask]); yy=np.log(y[mask])
    slope=np.polyfit(xx,yy,1)[0]
    max_slope_diff=max(max_slope_diff,abs(slope-r.Slope))
record('recomputed_convergence_slopes', max_slope_diff, 5e-12, max_slope_diff<5e-12)

# 3. Recompute exact energy-recovery identity and work-normalized recovery.
ls=load('locality_spectrum.csv')
rho_diff=0.0; work_diff=0.0
for _,r in ls.iterrows():
    den=r.ProlongedEnergy**2-r.DirectEnergy**2
    if abs(den)>1e-30:
        rho=(r.ProlongedEnergy**2-r.Energy**2)/den
        rho_diff=max(rho_diff,abs(rho-r.RecoveryFraction))
    if r.ActiveFraction>0:
        work=r.RecoveryFraction/r.ActiveFraction
        work_diff=max(work_diff,abs(work-r.WorkNormalizedRecovery))
record('energy_recovery_identity',rho_diff,1e-7,rho_diff<1e-7, 'Tolerance reflects rounded CSV values')
record('work_normalized_recovery_identity',work_diff,5e-11,work_diff<5e-11)

# 4. Recovery must be monotone under nested halo expansion.
viol=0
for (p,l),g in ls.groupby(['Problem','Level']):
    g=g.sort_values('Halo', key=lambda x: x.replace({999:99}))
    vals=g.RecoveryFraction.to_numpy()
    viol += int(np.sum(np.diff(vals)<-1e-7))
record('halo_monotonicity',viol,0,viol==0, 'Roundoff-level decreases below 1e-7 ignored')

# 5. Global halo must reproduce direct AFEM energy to roundoff.
global_rows=ls[ls.Halo==999]
gdiff=float(np.max(np.abs(global_rows.Energy-global_rows.DirectEnergy)))
record('global_halo_equals_direct_energy',gdiff,5e-12,gdiff<5e-12)

# 6. Interface weighted-estimator ablation at final level.
ia=load('interface_estimator_ablation.csv')
uw=ia[ia.Problem=='Interface-unweighted'].sort_values('Level').iloc[-1]
wt=ia[ia.Problem=='High-contrast interface (1e4)'].sort_values('Level').iloc[-1]
record('weighted_interface_energy_better',wt.Energy/uw.Energy,1.0,wt.Energy<uw.Energy)
record('weighted_interface_dof_better',wt.DOF/uw.DOF,1.0,wt.DOF<uw.DOF)

# 7. Default Dörfler theta=0.5 is an interior tradeoff between theta=.3 and .7.
ds=load('dorfler_sensitivity.csv')
tradeoff_ok=True
for p,g in ds.groupby('Problem'):
    g=g.set_index('Theta')
    tradeoff_ok &= g.loc[0.3,'DOF'] < g.loc[0.5,'DOF'] < g.loc[0.7,'DOF']
    tradeoff_ok &= g.loc[0.3,'Energy'] > g.loc[0.5,'Energy'] > g.loc[0.7,'Energy']
record('dorfler_tradeoff_ordering',1 if tradeoff_ok else 0,1,tradeoff_ok)

# 8. At finest level, one halo recovers >=99.4% of global correction for every problem while using <=77% active dofs.
final=[]
for p,g in ls.groupby('Problem'):
    level=g.Level.max(); r=g[(g.Level==level)&(g.Halo==1)].iloc[0]
    final.append((p,r.RecoveryFraction,r.ActiveFraction))
minrho=min(x[1] for x in final); maxphi=max(x[2] for x in final)
record('one_halo_min_recovery_finest',minrho,0.994,minrho>=0.994)
record('one_halo_max_active_fraction_finest',maxphi,0.77,maxphi<=0.77)

out=pd.DataFrame(checks).rename(columns={'pass_':'Pass'})
out.to_csv(os.path.join(VALIDATION,'independent_validation.csv'),index=False)
print(out.to_string(index=False))
if not out.Pass.all():
    raise SystemExit('One or more checks failed')
