#!/usr/bin/env python3
"""Compare plug-in MI K_eff vs closed-form Gaussian copula K_eff."""

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats

sys.path.insert(0, "/root/cert_manip_resist_eval")
from src.unified_data_loader import load_all_scores, DATASETS

RESULTS_DIR = Path("/root/cert_manip_resist_eval/artifacts/results/plan001")
MI_PATH = RESULTS_DIR / "mi_matrix" / "mi_matrix_15models.json"
OUTPUT_PATH = RESULTS_DIR / "keff_plugin_vs_closedform.json"

# MI matrix model names -> score file model names
MI_TO_SCORE_NAME = {
    "qwen2.5-72b": "qwen2.5-72b",
    "llama3.1-70b": "llama3.1-70b",
    "mistral-large": "mistral-large",
    "qwen2.5-32b": "qwen2.5-32b",
    "qwen2.5-14b": "qwen2.5-14b",
    "llama3.1-8b": "llama3.1-8b",
    "claude-opus-4-6": "claude-opus-4",
    "gpt-5.5": "gpt-5.5",
    "gemini-3.1-pro-preview": "gemini-2.5-pro",
    "gpt-4.1": "gpt-4.1",
    "claude-sonnet-4-6": "claude-sonnet-4",
    "gpt-4.1-nano": "gpt-4o",
    "gemini-3.5-flash": "gemini-2.5-flash",
    "gpt-4.1-mini": "gpt-4o-mini",
    "claude-3-haiku": "claude-3-haiku",
}


def extract_winners(scores):
    mapping = {"A": 0, "B": 1, "tie": 2}
    return np.array([mapping.get(s.get("winner", "tie"), 2) for s in scores])


def pairwise_spearman_matrix(clean_scores, mi_models, ds):
    """Compute pairwise Spearman rho between all model pairs."""
    K = len(mi_models)
    rho_mat = np.eye(K)
    winners = {}

    for i, mi_name in enumerate(mi_models):
        score_name = MI_TO_SCORE_NAME.get(mi_name, mi_name)
        if score_name in clean_scores[ds] and len(clean_scores[ds][score_name]) > 0:
            w = extract_winners(clean_scores[ds][score_name]).astype(float)
            rng = np.random.RandomState(i)
            w = w + rng.normal(0, 1e-6, len(w))
            winners[i] = w

    for i in range(K):
        for j in range(i + 1, K):
            if i in winners and j in winners:
                r, _ = stats.spearmanr(winners[i], winners[j])
                rho_mat[i, j] = r
                rho_mat[j, i] = r

    return rho_mat


def plugin_keff(mi_submatrix):
    """K_eff from plug-in MI: (1/K) sum_i [1 + sum_{j!=i} exp(-2*MI(i,j))]"""
    K = mi_submatrix.shape[0]
    if K <= 1:
        return 1.0
    total = 0.0
    for i in range(K):
        k_eff_i = 1.0
        for j in range(K):
            if j != i:
                k_eff_i += np.exp(-2.0 * mi_submatrix[i, j])
        total += k_eff_i
    return float(total / K)


def closedform_keff(spearman_rho_submatrix):
    """K_eff from Gaussian copula closed-form: K - (1/K) sum_i sum_{j!=i} rho_ij^2"""
    K = spearman_rho_submatrix.shape[0]
    if K <= 1:
        return 1.0
    sum_rho_sq = 0.0
    for i in range(K):
        for j in range(K):
            if j != i:
                sum_rho_sq += spearman_rho_submatrix[i, j] ** 2
    return float(K - sum_rho_sq / K)


def compare_keff_for_panels(mi_full, spearman_full, panel_indices_list):
    """Compute both K_eff for a list of panels."""
    plugin_vals = []
    cf_vals = []
    for indices in panel_indices_list:
        idx = np.array(indices)
        mi_sub = mi_full[np.ix_(idx, idx)]
        rho_sub = spearman_full[np.ix_(idx, idx)]
        plugin_vals.append(plugin_keff(mi_sub))
        cf_vals.append(closedform_keff(rho_sub))
    return np.array(plugin_vals), np.array(cf_vals)


def compute_comparison_stats(plugin_vals, cf_vals):
    if len(plugin_vals) < 3:
        return {"error": "insufficient_data", "n": len(plugin_vals)}

    pearson_r, pearson_p = stats.pearsonr(plugin_vals, cf_vals)
    rmse = float(np.sqrt(np.mean((plugin_vals - cf_vals) ** 2)))
    max_abs_dev = float(np.max(np.abs(plugin_vals - cf_vals)))
    rank_spearman, rank_p = stats.spearmanr(plugin_vals, cf_vals)
    mean_abs_dev = float(np.mean(np.abs(plugin_vals - cf_vals)))

    return {
        "pearson_r": round(float(pearson_r), 6),
        "pearson_p": float(pearson_p),
        "rmse": round(rmse, 6),
        "max_abs_dev": round(max_abs_dev, 6),
        "mean_abs_dev": round(mean_abs_dev, 6),
        "rank_spearman": round(float(rank_spearman), 6),
        "rank_spearman_p": float(rank_p),
        "mean_plugin": round(float(np.mean(plugin_vals)), 4),
        "mean_closedform": round(float(np.mean(cf_vals)), 4),
        "n_panels": len(plugin_vals),
    }


def main():
    print("Loading scores...")
    clean, _ = load_all_scores()

    print("Loading MI matrix...")
    with open(MI_PATH) as f:
        mi_data = json.load(f)

    mi_models = mi_data["models"]
    n_models = len(mi_models)

    # Verify all MI models have corresponding scores
    for ds in DATASETS:
        avail = []
        for mi_name in mi_models:
            score_name = MI_TO_SCORE_NAME.get(mi_name, mi_name)
            if score_name in clean[ds]:
                avail.append(mi_name)
            else:
                print(f"  WARNING: {mi_name} -> {score_name} NOT in {ds} scores")
        print(f"  {ds}: {len(avail)}/{n_models} models")

    # Generate panels for K=3,5,7
    rng = np.random.RandomState(42)
    panel_configs = {}
    for K in [3, 5, 7]:
        all_panels = list(combinations(range(n_models), K))
        if K <= 3:
            panel_configs[K] = all_panels
        else:
            idx = rng.choice(len(all_panels), size=min(500, len(all_panels)), replace=False)
            panel_configs[K] = [all_panels[i] for i in sorted(idx)]
        print(f"  K={K}: {len(panel_configs[K])} panels (of {len(all_panels)} total)")

    results = {"overall": {}, "by_K": {}, "by_benchmark": {}, "n_panels_compared": 0}
    all_plugin = []
    all_cf = []
    by_K = {f"K{K}": ([], []) for K in [3, 5, 7]}
    by_bench = {ds: ([], []) for ds in DATASETS}

    for ds in DATASETS:
        mi_full = np.array(mi_data["mi_matrix"][ds])
        print(f"\nDataset: {ds}")
        spearman_full = pairwise_spearman_matrix(clean, mi_models, ds)

        # Diagnostic: element-wise comparison
        diffs = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                exp_factor = np.exp(-2.0 * mi_full[i, j])  # independence factor from MI
                rho_sq = spearman_full[i, j] ** 2  # dependence from Spearman
                # Under Gaussian copula: exp(-2MI) = 1 - rho^2
                diffs.append(abs(exp_factor - (1 - rho_sq)))
        print(f"  Element-wise |exp(-2MI) - (1-rho^2)|: mean={np.mean(diffs):.4f}, max={np.max(diffs):.4f}")

        for K in [3, 5, 7]:
            panels = panel_configs[K]
            pv, cv = compare_keff_for_panels(mi_full, spearman_full, panels)
            all_plugin.extend(pv)
            all_cf.extend(cv)
            by_K[f"K{K}"][0].extend(pv)
            by_K[f"K{K}"][1].extend(cv)
            by_bench[ds][0].extend(pv)
            by_bench[ds][1].extend(cv)
            print(f"  K={K}: plugin_mean={np.mean(pv):.4f}, cf_mean={np.mean(cv):.4f}, "
                  f"r={np.corrcoef(pv, cv)[0,1]:.4f}")

    all_plugin = np.array(all_plugin)
    all_cf = np.array(all_cf)

    results["overall"] = compute_comparison_stats(all_plugin, all_cf)
    results["n_panels_compared"] = len(all_plugin)

    for K_key, (pv, cv) in by_K.items():
        results["by_K"][K_key] = compute_comparison_stats(np.array(pv), np.array(cv))

    for ds, (pv, cv) in by_bench.items():
        results["by_benchmark"][ds] = compute_comparison_stats(np.array(pv), np.array(cv))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"\n=== RESULTS ===")
    o = results["overall"]
    print(f"Overall ({o['n_panels']} panels): r={o['pearson_r']}, RMSE={o['rmse']}, "
          f"max_dev={o['max_abs_dev']}, rank_rho={o['rank_spearman']}")
    for k, v in results["by_K"].items():
        print(f"  {k}: r={v['pearson_r']}, RMSE={v['rmse']}, rank_rho={v['rank_spearman']}")
    for ds, v in results["by_benchmark"].items():
        print(f"  {ds}: r={v['pearson_r']}, RMSE={v['rmse']}, rank_rho={v['rank_spearman']}")


if __name__ == "__main__":
    main()
