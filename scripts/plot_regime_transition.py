#!/usr/bin/env python3
"""Plot Figure 2: regime transition (K_eff vs eta_max significance counts)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

K_vals = [3, 5, 7]
keff   = [4, 7, 8]
eta    = [3, 2, 1]

fig, ax = plt.subplots(figsize=(5.5, 4.5))

ax.axvspan(3, 5, alpha=0.10, color="gray", zorder=0)

ax.plot(K_vals, keff, "-o", color="#1f77b4", markersize=10, linewidth=2.2,
        label=r"$K_{\mathrm{eff}}$", zorder=3)
ax.plot(K_vals, eta, "--s", color="#ff7f0e", markersize=10, linewidth=2.8,
        dashes=(5, 3), label=r"$\eta_{\max}$", zorder=3)

offsets_keff = [(-28, 12), (-28, 12), (-28, 12)]
offsets_eta  = [(-28, -22), (12, 8), (12, 8)]

for i, k in enumerate(K_vals):
    ax.annotate(f"{keff[i]}/12", (k, keff[i]),
                textcoords="offset points", xytext=offsets_keff[i],
                fontsize=11, fontweight="bold", color="#1f77b4")
    ax.annotate(f"{eta[i]}/12", (k, eta[i]),
                textcoords="offset points", xytext=offsets_eta[i],
                fontsize=11, fontweight="bold", color="#ff7f0e")

handles, labels = ax.get_legend_handles_labels()
shift_patch = Patch(facecolor="gray", alpha=0.15, edgecolor="gray",
                    linewidth=0.8, label="Predictor shift zone")
ax.legend(handles=[shift_patch] + handles, loc="upper right",
          fontsize=10, framealpha=0.9, edgecolor="#cccccc")

ax.set_xlabel("Panel size $K$", fontsize=12)
ax.set_ylabel("FDR-significant conditions\n(out of 12)", fontsize=12)
ax.set_xticks(K_vals)
ax.set_ylim(-0.5, 12.5)
ax.set_yticks(range(0, 13, 2))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

out_dir = Path("/root/cert_manip_resist_eval/docs/paper/figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_regime_transition.pdf", bbox_inches="tight")
fig.savefig(out_dir / "fig_regime_transition.png", bbox_inches="tight", dpi=200)

out_dir2 = Path("/root/cert_manip_resist_eval/artifacts/_project/chart")
out_dir2.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir2 / "fig_regime_transition.pdf", bbox_inches="tight")
fig.savefig(out_dir2 / "fig_regime_transition.png", bbox_inches="tight", dpi=200)

plt.close()
print("Done.")
