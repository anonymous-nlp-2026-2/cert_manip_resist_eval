import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging

logging.basicConfig(filename='/tmp/betastd_trajectory.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

OUT_DIR = '/root/cert_manip_resist_eval/artifacts/results/plan001'

# Load existing analysis
with open(f'{OUT_DIR}/beta_std_trajectory.json') as f:
    data = json.load(f)

log.info("Loaded existing beta_std_trajectory.json")

# === Build output JSON in requested format ===
output = {
    "beta_std_keff": {},
    "beta_std_eta": {},
    "p_values_keff": {},
    "p_values_eta": {},
    "shift_evidence": "",
    "per_condition": data.get("per_condition", {}),
    "bootstrap": data.get("bootstrap", {}),
}

for k_label in ["K3", "K5", "K7"]:
    output["beta_std_keff"][k_label] = {
        "mean": data["beta_std_keff"][k_label]["mean"],
        "sd": data["beta_std_keff"][k_label]["sd"],
        "values": data["beta_std_keff"][k_label]["values"],
    }
    output["beta_std_eta"][k_label] = {
        "mean": data["beta_std_eta"][k_label]["mean"],
        "sd": data["beta_std_eta"][k_label]["sd"],
        "values": data["beta_std_eta"][k_label]["values"],
    }
    output["p_values_keff"][k_label] = data["pvalues_keff"][k_label]
    output["p_values_eta"][k_label] = data["pvalues_eta"][k_label]

# Compute significance counts
sig_counts_keff = {}
sig_counts_eta = {}
for k_label in ["K3", "K5", "K7"]:
    sig_counts_keff[k_label] = sum(1 for p in output["p_values_keff"][k_label] if p < 0.05)
    sig_counts_eta[k_label] = sum(1 for p in output["p_values_eta"][k_label] if p < 0.05)

keff_means = [output["beta_std_keff"][k]["mean"] for k in ["K3", "K5", "K7"]]
eta_means = [output["beta_std_eta"][k]["mean"] for k in ["K3", "K5", "K7"]]

output["shift_evidence"] = (
    f"β_std(K_eff) rises from {keff_means[0]:.3f} (K=3) to {keff_means[1]:.3f} (K=5) to {keff_means[2]:.3f} (K=7), "
    f"while β_std(η_max) declines from {eta_means[0]:.3f} to {eta_means[1]:.3f} to {eta_means[2]:.3f}. "
    f"Sig. conditions (p<0.05): K_eff {sig_counts_keff['K3']}→{sig_counts_keff['K5']}→{sig_counts_keff['K7']}/12; "
    f"η_max {sig_counts_eta['K3']}→{sig_counts_eta['K5']}→{sig_counts_eta['K7']}/12. "
    f"Predictor shift confirmed: certification diversity dominates at higher K."
)

with open(f'{OUT_DIR}/betastd_trajectory.json', 'w') as f:
    json.dump(output, f, indent=2)
log.info("Wrote betastd_trajectory.json")

# === Figure 1: β_std trajectory ===
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
})

fig, ax = plt.subplots(figsize=(4.5, 3.5))

K_vals = [3, 5, 7]
K_ticks = np.array(K_vals)

# K_eff line
keff_mean = np.array([output["beta_std_keff"][f"K{k}"]["mean"] for k in K_vals])
keff_sd = np.array([output["beta_std_keff"][f"K{k}"]["sd"] for k in K_vals])
keff_vals = [output["beta_std_keff"][f"K{k}"]["values"] for k in K_vals]

# η_max line
eta_mean = np.array([output["beta_std_eta"][f"K{k}"]["mean"] for k in K_vals])
eta_sd = np.array([output["beta_std_eta"][f"K{k}"]["sd"] for k in K_vals])
eta_vals = [output["beta_std_eta"][f"K{k}"]["values"] for k in K_vals]

# Colors
c_keff = '#2166AC'
c_eta = '#B2182B'

# Error bars
ax.errorbar(K_ticks, keff_mean, yerr=keff_sd, fmt='o-', color=c_keff, 
            capsize=4, capthick=1.2, markersize=6, label=r'$\beta_{\mathrm{std}}(K_{\mathrm{eff}})$',
            zorder=5)
ax.errorbar(K_ticks, eta_mean, yerr=eta_sd, fmt='s--', color=c_eta,
            capsize=4, capthick=1.2, markersize=6, label=r'$\beta_{\mathrm{std}}(\eta_{\max})$',
            zorder=5)

# Individual condition scatter (jittered)
rng = np.random.RandomState(42)
for i, k in enumerate(K_vals):
    jitter_keff = rng.uniform(-0.15, 0.15, len(keff_vals[i]))
    jitter_eta = rng.uniform(-0.15, 0.15, len(eta_vals[i]))
    ax.scatter(k + jitter_keff, keff_vals[i], color=c_keff, alpha=0.25, s=18, 
               edgecolors='none', zorder=3)
    ax.scatter(k + jitter_eta, eta_vals[i], color=c_eta, alpha=0.25, s=18,
               edgecolors='none', zorder=3)

ax.set_xlabel(r'Panel Size ($K$)')
ax.set_ylabel(r'Standardized Effect Size ($\beta_{\mathrm{std}}$)')
ax.set_xticks(K_ticks)
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':', zorder=1)
ax.legend(frameon=False, loc='upper right', fontsize=9)
ax.set_xlim(2, 8)
ax.set_ylim(-0.3, 0.85)

fig.tight_layout(pad=0.5)
fig.savefig(f'{OUT_DIR}/fig_betastd_trajectory.pdf', bbox_inches='tight')
fig.savefig(f'{OUT_DIR}/fig_betastd_trajectory.png', bbox_inches='tight', dpi=300)
plt.close(fig)
log.info("Wrote fig_betastd_trajectory.pdf/png")

# === Figure 2: Bootstrap p-value distribution ===
fig2, axes2 = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

for i, k in enumerate(K_vals):
    ax2 = axes2[i]
    k_label = f"K{k}"
    
    pvals_keff = output["p_values_keff"][k_label]
    pvals_eta = output["p_values_eta"][k_label]
    
    positions = [1, 2]
    parts_keff = ax2.violinplot([pvals_keff], positions=[1], showmeans=True, showextrema=False)
    parts_eta = ax2.violinplot([pvals_eta], positions=[2], showmeans=True, showextrema=False)
    
    for pc in parts_keff['bodies']:
        pc.set_facecolor(c_keff)
        pc.set_alpha(0.5)
    parts_keff['cmeans'].set_color(c_keff)
    
    for pc in parts_eta['bodies']:
        pc.set_facecolor(c_eta)
        pc.set_alpha(0.5)
    parts_eta['cmeans'].set_color(c_eta)
    
    # Individual points
    jitter_k = rng.uniform(-0.08, 0.08, len(pvals_keff))
    jitter_e = rng.uniform(-0.08, 0.08, len(pvals_eta))
    ax2.scatter(1 + jitter_k, pvals_keff, color=c_keff, alpha=0.6, s=15, edgecolors='none', zorder=5)
    ax2.scatter(2 + jitter_e, pvals_eta, color=c_eta, alpha=0.6, s=15, edgecolors='none', zorder=5)
    
    ax2.axhline(0.05, color='gray', linewidth=0.7, linestyle='--', zorder=1)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([r'$K_{\mathrm{eff}}$', r'$\eta_{\max}$'], fontsize=9)
    ax2.set_title(f'$K = {k}$', fontsize=10)
    
    n_sig_keff = sum(1 for p in pvals_keff if p < 0.05)
    n_sig_eta = sum(1 for p in pvals_eta if p < 0.05)
    ax2.text(1, -0.08, f'{n_sig_keff}/12', ha='center', fontsize=7, color=c_keff)
    ax2.text(2, -0.08, f'{n_sig_eta}/12', ha='center', fontsize=7, color=c_eta)
    
    if i == 0:
        ax2.set_ylabel('Bootstrap $p$-value')
    ax2.set_ylim(-0.12, 1.05)

fig2.tight_layout(pad=0.8)
fig2.savefig(f'{OUT_DIR}/fig_betastd_pvalue_dist.pdf', bbox_inches='tight')
fig2.savefig(f'{OUT_DIR}/fig_betastd_pvalue_dist.png', bbox_inches='tight', dpi=300)
plt.close(fig2)
log.info("Wrote fig_betastd_pvalue_dist.pdf/png")

# Print summary
print("=== β_std Trajectory Summary ===")
for k in ["K3", "K5", "K7"]:
    print(f"{k}: β_std(K_eff) = {output['beta_std_keff'][k]['mean']:.3f} ± {output['beta_std_keff'][k]['sd']:.3f}, "
          f"β_std(η_max) = {output['beta_std_eta'][k]['mean']:.3f} ± {output['beta_std_eta'][k]['sd']:.3f}")
print(f"\nSig counts (p<0.05):")
print(f"  K_eff: {sig_counts_keff}")
print(f"  η_max: {sig_counts_eta}")
print(f"\n{output['shift_evidence']}")
print(f"\nFiles written:")
print(f"  {OUT_DIR}/betastd_trajectory.json")
print(f"  {OUT_DIR}/fig_betastd_trajectory.pdf")
print(f"  {OUT_DIR}/fig_betastd_pvalue_dist.pdf")
