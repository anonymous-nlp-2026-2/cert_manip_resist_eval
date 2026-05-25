import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(filename='/tmp/beta_std_trajectory.log', level=logging.INFO,
                    format='%(asctime)s %(message)s')
log = logging.getLogger()

BASE = Path('/root/cert_manip_resist_eval/artifacts/results/plan001')
OUT = BASE

# --- Load data sources ---
with open(BASE / 'dbei_predictor_shift.json') as f:
    dbei = json.load(f)

# Also load wild cluster bootstrap files for cross-reference
with open(BASE / 'k5_wild_cluster_bootstrap.json') as f:
    wcb_k5 = json.load(f)
with open(BASE / 'k7_wild_bootstrap_results.json') as f:
    wcb_k7 = json.load(f)

# --- Extract per-condition std_beta from dbei_predictor_shift ---
# This file has K=3,5,7 consistently with the same methodology
K_LABELS = ['K3', 'K5', 'K7']
K_VALUES = [3, 5, 7]

conditions_all = list(dbei['conditions_detail']['K3']['conditions'].keys())
log.info(f"Conditions ({len(conditions_all)}): {conditions_all}")

result = {
    'beta_std_keff': {},
    'beta_std_eta': {},
    'pvalues_keff': {},
    'pvalues_eta': {},
    'per_condition': {}
}

for kl, kv in zip(K_LABELS, K_VALUES):
    conds = dbei['conditions_detail'][kl]['conditions']
    keff_betas = []
    eta_betas = []
    keff_ps = []
    eta_ps = []
    
    for cname in conditions_all:
        c = conds[cname]
        kb = c['keff']['std_beta']
        eb = c['eta_in_keff_model']['std_beta']
        kp = c['keff']['final_p']
        ep = c['eta_in_keff_model']['final_p']
        
        keff_betas.append(kb)
        eta_betas.append(eb)
        keff_ps.append(kp)
        eta_ps.append(ep)
        
        # Store per-condition trajectory
        key = cname
        if key not in result['per_condition']:
            result['per_condition'][key] = {'keff_std': {}, 'eta_std': {}, 'keff_p': {}, 'eta_p': {}}
        result['per_condition'][key]['keff_std'][kl] = kb
        result['per_condition'][key]['eta_std'][kl] = eb
        result['per_condition'][key]['keff_p'][kl] = kp
        result['per_condition'][key]['eta_p'][kl] = ep
    
    result['beta_std_keff'][kl] = {
        'mean': float(np.mean(keff_betas)),
        'sd': float(np.std(keff_betas, ddof=1)),
        'median': float(np.median(keff_betas)),
        'values': keff_betas
    }
    result['beta_std_eta'][kl] = {
        'mean': float(np.mean(eta_betas)),
        'sd': float(np.std(eta_betas, ddof=1)),
        'median': float(np.median(eta_betas)),
        'values': eta_betas
    }
    result['pvalues_keff'][kl] = keff_ps
    result['pvalues_eta'][kl] = eta_ps
    
    log.info(f"{kl}: β_std(K_eff) mean={np.mean(keff_betas):.4f}±{np.std(keff_betas,ddof=1):.4f}, "
             f"β_std(η) mean={np.mean(eta_betas):.4f}±{np.std(eta_betas,ddof=1):.4f}")

# Cross-reference with wild cluster bootstrap (K=5, K=7)
cross_ref = {}
for cname in conditions_all:
    cr = {}
    if cname in wcb_k5['per_condition']:
        cr['K5_wcb_keff_std'] = wcb_k5['per_condition'][cname]['keff_ols_std_beta']
        cr['K5_wcb_eta_std'] = wcb_k5['per_condition'][cname]['eta_ols_std_beta']
    if cname in wcb_k7['per_condition']:
        cr['K7_wcb_keff_std'] = wcb_k7['per_condition'][cname]['keff_ols_std_beta']
        cr['K7_wcb_eta_std'] = wcb_k7['per_condition'][cname]['eta_ols_std_beta']
    cross_ref[cname] = cr
result['cross_reference_wcb'] = cross_ref

# Shift magnitude narrative
k3_mean = result['beta_std_keff']['K3']['mean']
k5_mean = result['beta_std_keff']['K5']['mean']
k7_mean = result['beta_std_keff']['K7']['mean']
e3_mean = result['beta_std_eta']['K3']['mean']
e5_mean = result['beta_std_eta']['K5']['mean']
e7_mean = result['beta_std_eta']['K7']['mean']
result['shift_magnitude'] = (
    f"β_std(K_eff) trajectory: K3={k3_mean:.3f} → K5={k5_mean:.3f} → K7={k7_mean:.3f}; "
    f"β_std(η_max) trajectory: K3={e3_mean:.3f} → K5={e5_mean:.3f} → K7={e7_mean:.3f}"
)
result['source'] = 'dbei_predictor_shift.json (model: log(ASR) ~ log(K_eff) + log(eta_max))'
result['bootstrap'] = dbei['bootstrap']

# Save JSON
with open(OUT / 'beta_std_trajectory.json', 'w') as f:
    json.dump(result, f, indent=2)
log.info(f"JSON saved to {OUT / 'beta_std_trajectory.json'}")

# ============================================================
# Figure 1: β_std trajectory across K
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.2,
    'figure.dpi': 300,
})

fig, ax = plt.subplots(figsize=(3.5, 2.8))

x = np.array(K_VALUES)
jitter_w = 0.12

# K_eff line
keff_means = [result['beta_std_keff'][kl]['mean'] for kl in K_LABELS]
keff_sds = [result['beta_std_keff'][kl]['sd'] for kl in K_LABELS]
keff_vals = [result['beta_std_keff'][kl]['values'] for kl in K_LABELS]

# η_max line
eta_means = [result['beta_std_eta'][kl]['mean'] for kl in K_LABELS]
eta_sds = [result['beta_std_eta'][kl]['sd'] for kl in K_LABELS]
eta_vals = [result['beta_std_eta'][kl]['values'] for kl in K_LABELS]

# Mean ± SD lines
ax.errorbar(x - 0.05, keff_means, yerr=keff_sds, fmt='o-', color='#2166ac',
            markersize=5, capsize=3, capthick=0.8, label=r'$\beta_{\mathrm{std}}(K_{\mathrm{eff}})$',
            zorder=5)
ax.errorbar(x + 0.05, eta_means, yerr=eta_sds, fmt='s-', color='#b2182b',
            markersize=5, capsize=3, capthick=0.8, label=r'$\beta_{\mathrm{std}}(\eta_{\max})$',
            zorder=5)

# Jittered individual conditions
rng = np.random.RandomState(42)
for i, kl in enumerate(K_LABELS):
    n = len(keff_vals[i])
    jx_keff = x[i] - 0.05 + rng.uniform(-jitter_w, jitter_w, n)
    jx_eta = x[i] + 0.05 + rng.uniform(-jitter_w, jitter_w, n)
    ax.scatter(jx_keff, keff_vals[i], c='#2166ac', alpha=0.35, s=12, edgecolors='none', zorder=3)
    ax.scatter(jx_eta, eta_vals[i], c='#b2182b', alpha=0.35, s=12, edgecolors='none', zorder=3)

ax.set_xticks(K_VALUES)
ax.set_xlabel('Panel size $K$')
ax.set_ylabel(r'Standardized $\beta$')
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.axhline(0, color='grey', lw=0.4, ls='--', zorder=1)
ax.set_xlim(2.2, 7.8)

fig.tight_layout(pad=0.3)
fig.savefig(OUT / 'fig_beta_std_trajectory.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_beta_std_trajectory.png', bbox_inches='tight', dpi=300)
log.info("Fig 1 saved")
plt.close(fig)

# ============================================================
# Figure 2: p-value distribution (violin + strip)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True)

for ax_idx, (predictor, pkey, color, label) in enumerate([
    ('keff', 'pvalues_keff', '#2166ac', r'$K_{\mathrm{eff}}$'),
    ('eta', 'pvalues_eta', '#b2182b', r'$\eta_{\max}$'),
]):
    ax = axes[ax_idx]
    data_list = [result[pkey][kl] for kl in K_LABELS]
    
    parts = ax.violinplot(data_list, positions=K_VALUES, widths=1.2,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.25)
    
    for i, kl in enumerate(K_LABELS):
        vals = np.array(data_list[i])
        jx = K_VALUES[i] + rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(jx, vals, c=color, alpha=0.6, s=14, edgecolors='white', linewidths=0.3, zorder=4)
        ax.scatter(K_VALUES[i], np.median(vals), c='black', s=25, marker='_', linewidths=1.5, zorder=5)
    
    ax.axhline(0.05, color='grey', lw=0.6, ls='--', zorder=1, alpha=0.6)
    ax.text(7.5, 0.05, '$\\alpha$=.05', fontsize=7, color='grey', va='center')
    ax.set_xticks(K_VALUES)
    ax.set_xlabel('Panel size $K$')
    ax.set_title(label, fontsize=9)
    
axes[0].set_ylabel('Bootstrap $p$-value')
fig.tight_layout(pad=0.3)
fig.savefig(OUT / 'fig_pvalue_distribution.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_pvalue_distribution.png', bbox_inches='tight', dpi=300)
log.info("Fig 2 saved")
plt.close(fig)

# Print summary
print("=== β_std trajectory summary ===")
for kl in K_LABELS:
    keff = result['beta_std_keff'][kl]
    eta = result['beta_std_eta'][kl]
    n_sig_keff = sum(1 for p in result['pvalues_keff'][kl] if p < 0.05)
    n_sig_eta = sum(1 for p in result['pvalues_eta'][kl] if p < 0.05)
    print(f"{kl}: β_std(K_eff)={keff['mean']:.3f}±{keff['sd']:.3f} (sig {n_sig_keff}/12), "
          f"β_std(η)={eta['mean']:.3f}±{eta['sd']:.3f} (sig {n_sig_eta}/12)")

print(f"\n{result['shift_magnitude']}")
print(f"\nOutputs: {OUT / 'beta_std_trajectory.json'}")
print(f"         {OUT / 'fig_beta_std_trajectory.pdf'}")
print(f"         {OUT / 'fig_pvalue_distribution.pdf'}")
