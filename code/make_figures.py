import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D=os.path.join(ROOT,'data'); F=os.path.join(ROOT,'figures'); os.makedirs(F,exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'figure.dpi':160,'savefig.bbox':'tight','pdf.fonttype':42,'ps.fonttype':42})
problems=['Smooth Poisson','L-shaped singularity','High-contrast interface (1e4)','Localized reaction-diffusion layer']
short={'Smooth Poisson':'Smooth Poisson','L-shaped singularity':'L-shaped singularity','High-contrast interface (1e4)':'High-contrast interface $10^4$','Localized reaction-diffusion layer':'Localized reaction-diffusion layer'}
colors=['#1f77b4','#d62728','#2ca02c','#9467bd']
markers=['o','s','^','D']
u=pd.read_csv(os.path.join(D,'uniform.csv')); a=pd.read_csv(os.path.join(D,'afem.csv'))
# Fig 1 energy convergence 2x2
fig,axs=plt.subplots(2,2,figsize=(7.2,5.4),sharex=False)
for ax,p in zip(axs.ravel(),problems):
    gu=u[u.Problem==p]; ga=a[a.Problem==p]
    ax.loglog(gu.DOF,gu.Energy,'o--',label='Uniform $P_1$',ms=4,lw=1.2)
    ax.loglog(ga.DOF,ga.Energy,'s-',label='AFEM-direct',ms=4,lw=1.3)
    ax.set_xlabel('Degrees of freedom'); ax.set_ylabel('Energy error'); ax.grid(True,which='both',alpha=.2)
    ax.text(0.02,0.96,f'({chr(97+list(problems).index(p))})',transform=ax.transAxes,ha='left',va='top',fontsize=9,fontweight='bold')
    ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(F,'Fig01_energy_convergence.pdf')); fig.savefig(os.path.join(F,'Fig01_energy_convergence.png'),dpi=220); plt.close(fig)
# Fig 2 L2 convergence
fig,axs=plt.subplots(2,2,figsize=(7.2,5.4))
for ax,p in zip(axs.ravel(),problems):
    gu=u[u.Problem==p]; ga=a[a.Problem==p]
    ax.loglog(gu.DOF,gu.L2,'o--',label='Uniform $P_1$',ms=4,lw=1.2)
    ax.loglog(ga.DOF,ga.L2,'s-',label='AFEM-direct',ms=4,lw=1.3)
    ax.set_xlabel('Degrees of freedom'); ax.set_ylabel('$L^2$ error'); ax.grid(True,which='both',alpha=.2)
    ax.text(0.02,0.96,f'({chr(97+list(problems).index(p))})',transform=ax.transAxes,ha='left',va='top',fontsize=9,fontweight='bold')
    ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(F,'Fig02_l2_convergence.pdf')); fig.savefig(os.path.join(F,'Fig02_l2_convergence.png'),dpi=220); plt.close(fig)
# Fig 3 locality spectrum final level
ls=pd.read_csv(os.path.join(D,'locality_spectrum.csv'))
fig,ax=plt.subplots(figsize=(6.6,4.5))
for p,c,m in zip(problems,colors,markers):
    g=ls[ls.Problem==p]; g=g[g.Level==g.Level.max()].copy(); g=g[g.Halo!=999].sort_values('ActiveFraction')
    ax.plot(g.ActiveFraction,g.RecoveryFraction,marker=m,color=c,lw=1.4,ms=5,label=short[p])
ax.axhline(.99,color='0.45',lw=1,ls=':'); ax.set_xlabel('Active interior-DOF fraction $\\phi(W)$'); ax.set_ylabel('Energy-recovery fraction $\\rho(W)$'); ax.set_ylim(.90,1.005); ax.set_xlim(.38,1.02); ax.grid(True,alpha=.2); ax.legend(frameon=False,loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(F,'Fig03_locality_spectrum.pdf')); fig.savefig(os.path.join(F,'Fig03_locality_spectrum.png'),dpi=220); plt.close(fig)
# Fig 4 work normalized recovery final level
fig,ax=plt.subplots(figsize=(6.6,4.5))
for p,c,m in zip(problems,colors,markers):
    g=ls[ls.Problem==p]; g=g[(g.Level==g.Level.max())&(g.Halo!=999)].sort_values('ActiveFraction')
    ax.plot(g.ActiveFraction,g.WorkNormalizedRecovery,marker=m,color=c,lw=1.4,ms=5,label=short[p])
ax.axhline(1,color='0.45',lw=1,ls=':'); ax.set_xlabel('Active interior-DOF fraction $\\phi(W)$'); ax.set_ylabel('Work-normalized recovery $J(W)=\\rho/\\phi$'); ax.grid(True,alpha=.2); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(F,'Fig04_work_normalized_recovery.pdf')); fig.savefig(os.path.join(F,'Fig04_work_normalized_recovery.png'),dpi=220); plt.close(fig)
# Fig 5 Dörfler sensitivity
s=pd.read_csv(os.path.join(D,'dorfler_sensitivity.csv'))
fig,axs=plt.subplots(1,3,figsize=(8.2,3.2))
for ax,p,c in zip(axs,s.Problem.unique(),colors[1:]):
    g=s[s.Problem==p].sort_values('Theta')
    ax.loglog(g.DOF,g.Energy,'o-',color=c,lw=1.4,ms=5)
    for _,r in g.iterrows(): ax.annotate(f"$\\theta$={r.Theta:.1f}",(r.DOF,r.Energy),textcoords='offset points',xytext=(4,4),fontsize=7)
    ax.set_xlabel('DOF'); ax.set_ylabel('Energy error'); ax.grid(True,which='both',alpha=.2)
    ax.text(0.02,0.96,f'({chr(97+list(s.Problem.unique()).index(p))})',transform=ax.transAxes,ha='left',va='top',fontsize=9,fontweight='bold')
fig.tight_layout(); fig.savefig(os.path.join(F,'Fig05_dorfler_sensitivity.pdf')); fig.savefig(os.path.join(F,'Fig05_dorfler_sensitivity.png'),dpi=220); plt.close(fig)
# Fig 6 RCHC threshold sensitivity final 3 levels aggregated (mean by tau)
r=pd.read_csv(os.path.join(D,'rchc_tau_sensitivity.csv'))
fig,axs=plt.subplots(1,2,figsize=(7.4,3.2))
for p,c,m in zip(problems,colors,markers):
    g=r[r.Problem==p].groupby('Tau',as_index=False).agg(EnergyRatio=('EnergyRatio','mean'),ActiveFraction=('ActiveFraction','mean'))
    axs[0].plot(g.Tau,g.EnergyRatio,marker=m,color=c,lw=1.3,ms=4,label=short[p])
    axs[1].plot(g.Tau,g.ActiveFraction,marker=m,color=c,lw=1.3,ms=4,label=short[p])
axs[0].axhline(1,color='0.45',lw=1,ls=':'); axs[0].set_ylabel('Localized/direct energy-error ratio'); axs[1].set_ylabel('Active interior-DOF fraction')
for ax in axs: ax.set_xlabel('Residual threshold $\\tau$'); ax.grid(True,alpha=.2)
handles,labels=axs[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=7,bbox_to_anchor=(0.5,-0.02)); fig.tight_layout(rect=(0,0.10,1,1)); fig.savefig(os.path.join(F,'Fig06_rchc_sensitivity.pdf')); fig.savefig(os.path.join(F,'Fig06_rchc_sensitivity.png'),dpi=220); plt.close(fig)
# Fig 7 repeated localized workflow tradeoff
lc=pd.read_csv(os.path.join(D,'localized_correction.csv'))
fig,ax=plt.subplots(figsize=(6.6,4.5))
method_mark={'HLRC-r0':'o','HLRC-r1':'s','HLRC-r2':'^','RCHC':'D'}
for pi,p in enumerate(problems):
    ad=a[a.Problem==p].sort_values('Level').iloc[-1]
    for meth,mk in method_mark.items():
        g=lc[(lc.Problem==p)&(lc.Method==meth)].sort_values('Level');
        if g.empty: continue
        rr=g.iloc[-1]; x=rr.Cumulative_s/ad.Cumulative_s; y=rr.Energy/ad.Energy
        ax.scatter(x,y,marker=mk,s=45,color=colors[pi])
        ax.annotate(meth,(x,y),textcoords='offset points',xytext=(3,3),fontsize=6.5)
ax.axvline(1,color='0.45',ls=':',lw=1); ax.axhline(1,color='0.45',ls=':',lw=1)
ax.set_xlabel('Cumulative-time ratio to AFEM-direct'); ax.set_ylabel('Energy-error ratio to AFEM-direct'); ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True,which='both',alpha=.2)
# problem legend
for p,c in zip(problems,colors): ax.scatter([],[],color=c,label=short[p])
ax.legend(frameon=False,fontsize=7,loc='upper right'); fig.tight_layout(); fig.savefig(os.path.join(F,'Fig07_workflow_tradeoff.pdf')); fig.savefig(os.path.join(F,'Fig07_workflow_tradeoff.png'),dpi=220); plt.close(fig)
