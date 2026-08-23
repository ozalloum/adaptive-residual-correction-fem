#!/usr/bin/env python3
"""Regenerate manuscript Figures 1--7 and plot CSVs from archived repository data."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
PLOT_CSV = ROOT / "plot_csv"
PLOT_CSV.mkdir(parents=True, exist_ok=True)
print("Data directory:", DATA)
print("Output directory:", OUT)

# Load the archived result tables.
files = {
    'uniform': 'uniform.csv',
    'afem': 'afem.csv',
    'locality': 'locality_spectrum.csv',
    'localized': 'localized_correction.csv',
    'slopes': 'convergence_slopes.csv',
    'dorfler': 'dorfler_sensitivity.csv',
    'interface': 'interface_estimator_ablation.csv',
    'rchc': 'rchc_tau_sensitivity.csv',
    'timing': 'timing_final_25reps.csv',
    'verification': 'verification_checks.csv',
}
D = {k: pd.read_csv(DATA / v) for k, v in files.items()}
for k, df in D.items():
    print(f'{k:12s}: {len(df):5d} rows')

# Plotting conventions used consistently across the manuscript.
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.28,
    'grid.linestyle': '-',
})

problem_order = [
    'Smooth Poisson',
    'L-shaped singularity',
    'High-contrast interface (1e4)',
    'Localized reaction-diffusion layer',
]
problem_display = {
    'Smooth Poisson': 'Smooth Poisson',
    'L-shaped singularity': 'L-shaped singularity',
    'High-contrast interface (1e4)': r'High-contrast interface $10^4$',
    'Localized reaction-diffusion layer': 'Localized reaction-diffusion layer',
}
problem_styles = {
    'Smooth Poisson': dict(color='tab:blue', marker='o', linestyle='-'),
    'L-shaped singularity': dict(color='tab:red', marker='s', linestyle='--'),
    'High-contrast interface (1e4)': dict(color='tab:green', marker='^', linestyle='-.'),
    'Localized reaction-diffusion layer': dict(color='tab:purple', marker='D', linestyle=':'),
}

def panel_label(ax, label, x=-0.13, y=1.04):
    """Put a panel label outside the plotting area."""
    ax.text(x, y, label, transform=ax.transAxes, ha='left', va='top',
            fontsize=14, fontweight='bold', clip_on=False)

def expand_log_limits(values, frac=0.10):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    lo, hi = np.log10(values.min()), np.log10(values.max())
    pad = frac * (hi - lo if hi > lo else 1.0)
    return 10**(lo - pad), 10**(hi + pad)

def save_figure(fig, stem):
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

uniform, afem = D['uniform'], D['afem']

for metric, stem, ylabel in [
    ('Energy', 'Fig01_energy_convergence', 'Energy error'),
    ('L2', 'Fig02_l2_convergence', r'$L^2$ error'),
]:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), constrained_layout=True)
    for ax, prob, lab in zip(axes.flat, problem_order, ['(a)', '(b)', '(c)', '(d)']):
        du = uniform[uniform['Problem'] == prob].sort_values('DOF')
        da = afem[afem['Problem'] == prob].sort_values('DOF')
        ax.loglog(du['DOF'], du[metric], color='tab:blue', marker='o', linestyle='--',
                  label=r'Uniform $P_1$', linewidth=1.6, markersize=5)
        ax.loglog(da['DOF'], da[metric], color='tab:orange', marker='s', linestyle='-',
                  label='AFEM-direct', linewidth=1.6, markersize=5)
        ax.set_xlabel('Degrees of freedom')
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', frameon=False)
        ax.grid(True, which='both', alpha=0.25)
        panel_label(ax, lab)
    save_figure(fig, stem)

# Machine-readable data used in Figures 1 and 2.
fig01_data = pd.concat([
    uniform[['Problem','Method','DOF','Energy']].copy(),
    afem[['Problem','Method','DOF','Energy']].copy(),
], ignore_index=True)
fig01_data.to_csv(PLOT_CSV / 'Fig01_energy_convergence.csv', index=False)

fig02_data = pd.concat([
    uniform[['Problem','Method','DOF','L2']].copy(),
    afem[['Problem','Method','DOF','L2']].copy(),
], ignore_index=True)
fig02_data.to_csv(PLOT_CSV / 'Fig02_l2_convergence.csv', index=False)

loc = D['locality']
parts = []
for prob in problem_order:
    dp = loc[loc['Problem'] == prob]
    finest = dp['Level'].max()
    parts.append(dp[dp['Level'] == finest].copy())
locf = pd.concat(parts, ignore_index=True)

fig, ax = plt.subplots(figsize=(9.2, 6.3), constrained_layout=True)
for prob in problem_order:
    d = locf[locf['Problem'] == prob].sort_values('ActiveFraction')
    s = problem_styles[prob]
    ax.plot(d['ActiveFraction'], d['RecoveryFraction'], label=problem_display[prob],
            color=s['color'], marker=s['marker'], linestyle=s['linestyle'],
            linewidth=1.8, markersize=7)
ax.axhline(0.99, color='0.45', linestyle=':', linewidth=1.4)
ax.set_xlim(0.38, 1.02)
ax.set_ylim(0.90, 1.005)
ax.set_xlabel(r'Active interior-DOF fraction $\phi(W)$')
ax.set_ylabel(r'Energy-recovery fraction $\rho(W)$')
ax.legend(loc='lower right', frameon=False)
save_figure(fig, 'Fig03_locality_spectrum')

fig, ax = plt.subplots(figsize=(9.2, 6.3), constrained_layout=True)
for prob in problem_order:
    d = locf[locf['Problem'] == prob].sort_values('ActiveFraction')
    s = problem_styles[prob]
    ax.plot(d['ActiveFraction'], d['WorkNormalizedRecovery'], label=problem_display[prob],
            color=s['color'], marker=s['marker'], linestyle=s['linestyle'],
            linewidth=1.8, markersize=7)
ax.axhline(1.0, color='0.45', linestyle=':', linewidth=1.4)
ax.set_xlim(0.40, 1.02)
ax.set_ylim(0.96, 2.22)
ax.set_xlabel(r'Active interior-DOF fraction $\phi(W)$')
ax.set_ylabel(r'Work-normalized recovery $J(W)=\rho/\phi$')
ax.legend(loc='upper right', frameon=False)
save_figure(fig, 'Fig04_work_normalized_recovery')

# Machine-readable data used in Figures 3 and 4 (finest recorded level per benchmark).
locf[['Problem','Level','Halo','DOF','InteriorDOF','ActiveDOF','ActiveFraction',
      'Energy','DirectEnergy','ProlongedEnergy','RecoveryFraction',
      'WorkNormalizedRecovery']].to_csv(PLOT_CSV / 'Fig03_locality_spectrum.csv', index=False)
locf[['Problem','Level','Halo','ActiveFraction','RecoveryFraction',
      'WorkNormalizedRecovery']].to_csv(PLOT_CSV / 'Fig04_work_normalized_recovery.csv', index=False)

sens = D['dorfler']
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), constrained_layout=True)
colors = ['tab:red', 'tab:green', 'tab:purple']
for ax, prob, color, lab in zip(axes, sens['Problem'].unique(), colors, ['(a)', '(b)', '(c)']):
    d = sens[sens['Problem'] == prob].sort_values('DOF')
    ax.loglog(d['DOF'], d['Energy'], color=color, marker='o', linestyle='-',
              linewidth=1.8, markersize=7)
    for _, r in d.iterrows():
        ax.annotate(rf'$\theta$={r["Theta"]:.1f}', (r['DOF'], r['Energy']),
                    xytext=(6, 6), textcoords='offset points', fontsize=10)
    ax.set_xlabel('DOF')
    ax.set_ylabel('Energy error')
    # Deliberately larger horizontal margin so the theta=0.7 annotation stays inside each panel.
    ax.set_xlim(*expand_log_limits(d['DOF'], frac=0.32))
    ax.set_ylim(*expand_log_limits(d['Energy'], frac=0.10))
    ax.grid(True, which='both', alpha=0.25)
    panel_label(ax, lab, x=-0.17, y=1.03)
save_figure(fig, 'Fig05_dorfler_sensitivity')

print(sens[['Problem','Theta','DOF','Energy','Estimator','Cumulative_s']])

# Machine-readable data used in Figure 5.
sens[['Problem','Theta','Level','DOF','Energy','Estimator','Cumulative_s','MinAngle']].to_csv(PLOT_CSV / 'Fig05_dorfler_sensitivity.csv', index=False)

# The manuscript Figure 6 reports averages over the last three recorded levels for each benchmark.
rchc = D['rchc']
avg_parts = []
for prob in problem_order:
    dp = rchc[rchc['Problem'] == prob]
    last3 = sorted(dp['Level'].unique())[-3:]
    q = (dp[dp['Level'].isin(last3)]
         .groupby(['Problem','Tau'], as_index=False)
         .agg(EnergyRatio=('EnergyRatio','mean'), ActiveFraction=('ActiveFraction','mean')))
    avg_parts.append(q)
ravg = pd.concat(avg_parts, ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.3))
for ax, ycol, ylabel, lab in [
    (axes[0], 'EnergyRatio', 'Localized/direct energy-error ratio', '(a)'),
    (axes[1], 'ActiveFraction', 'Active interior-DOF fraction', '(b)'),
]:
    for prob in problem_order:
        d = ravg[ravg['Problem'] == prob].sort_values('Tau')
        s = problem_styles[prob]
        ax.plot(d['Tau'], d[ycol], label=problem_display[prob],
                color=s['color'], marker=s['marker'], linestyle=s['linestyle'],
                linewidth=1.8, markersize=6)
    if ycol == 'EnergyRatio':
        ax.axhline(1.0, color='0.45', linestyle=':', linewidth=1.4)
    ax.set_xlabel(r'Residual threshold $\tau$')
    ax.set_ylabel(ylabel)
    panel_label(ax, lab, x=-0.12, y=1.03)
    ax.grid(True, which='both', alpha=0.25)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.subplots_adjust(bottom=0.20, left=0.08, right=0.98, top=0.96, wspace=0.18)
save_figure(fig, 'Fig06_rchc_sensitivity')

print(ravg)

# Machine-readable averaged data plotted in Figure 6.
ravg[['Problem','Tau','EnergyRatio','ActiveFraction']].to_csv(PLOT_CSV / 'Fig06_rchc_sensitivity.csv', index=False)

localized = D['localized']
# Final recorded state of each direct AFEM benchmark.
direct_final = afem.sort_values('Level').groupby('Problem', as_index=False).tail(1).set_index('Problem')
# Final recorded state of each localized method/benchmark.
local_final = localized.sort_values('Level').groupby(['Problem','Method'], as_index=False).tail(1).copy()
local_final['EnergyRatio'] = [r.Energy / direct_final.loc[r.Problem, 'Energy'] for r in local_final.itertuples()]
local_final['TimeRatio'] = [r.Cumulative_s / direct_final.loc[r.Problem, 'Cumulative_s'] for r in local_final.itertuples()]
local_final['L2Ratio'] = [r.L2 / direct_final.loc[r.Problem, 'L2'] for r in local_final.itertuples()]

method_markers = {'HLRC-r0':'o', 'HLRC-r1':'s', 'HLRC-r2':'^', 'RCHC':'D'}
fig, ax = plt.subplots(figsize=(9.2, 6.3), constrained_layout=True)
for prob in problem_order:
    d = local_final[local_final['Problem'] == prob]
    color = problem_styles[prob]['color']
    for _, r in d.iterrows():
        ax.scatter(r['TimeRatio'], r['EnergyRatio'], s=80, color=color,
                   marker=method_markers[r['Method']], zorder=3)
        ax.annotate(r['Method'], (r['TimeRatio'], r['EnergyRatio']), xytext=(4, 4),
                    textcoords='offset points', fontsize=9)
ax.axvline(1.0, color='0.45', linestyle=':', linewidth=1.4)
ax.axhline(1.0, color='0.45', linestyle=':', linewidth=1.4)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.80, 2.45)
ax.set_ylim(0.95, 3.35)
ax.set_xlabel('Cumulative-time ratio to AFEM-direct')
ax.set_ylabel('Energy-error ratio to AFEM-direct')
legend_handles = [
    Line2D([0],[0], marker='o', linestyle='None', markersize=8,
           markerfacecolor=problem_styles[p]['color'], markeredgecolor=problem_styles[p]['color'],
           label=problem_display[p]) for p in problem_order
]
ax.legend(handles=legend_handles, loc='upper right', frameon=False)
save_figure(fig, 'Fig07_workflow_tradeoff')

print(local_final[['Problem','Method','EnergyRatio','L2Ratio','TimeRatio','ActiveFraction']].sort_values(['Problem','Method']))

# Machine-readable final-state ratios plotted in Figure 7.
local_final[['Problem','Method','Level','DOF','ActiveDOF','ActiveFraction','Energy','L2','Cumulative_s','EnergyRatio','L2Ratio','TimeRatio']].sort_values(['Problem','Method']).to_csv(PLOT_CSV / 'Fig07_workflow_tradeoff.csv', index=False)
