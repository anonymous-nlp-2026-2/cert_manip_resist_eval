#!/usr/bin/env python3
"""Monte Carlo power simulation for CRVE under different panel sizes.

Demonstrates that K=3 has low statistical power to detect a true effect
due to massive SE inflation from judge overlap, explaining 0/12 FDR results.

Output: artifacts/results/plan001/crossed_re_power.json
"""

import json
import itertools
import time
import numpy as np
from scipy import stats as scipy_stats
from pathlib import Path

N_JUDGES = 15
PANEL_SIZES = [3, 5, 7]
N_REPLICATES = 500
ALPHA = 0.05

BETA_STD_VALUES = [0.2, 0.3, 0.4, 0.5]
SIGMA_JUDGE_VALUES = [0.1, 0.3, 0.5]
SIGMA_EPS = 0.3  # residual noise

np.random.seed(42)


def _json_default(obj):
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def generate_mi_matrix(n_judges, seed=123):
    """Generate a realistic MI matrix for judges."""
    rng = np.random.RandomState(seed)
    # Each judge has a latent quality vector
    latent = rng.randn(n_judges, 3)
    mi = np.zeros((n_judges, n_judges))
    for i in range(n_judges):
        for j in range(n_judges):
            if i == j:
                mi[i, j] = 1.0
            else:
                # MI based on similarity of latent vectors
                sim = np.dot(latent[i], latent[j]) / (
                    np.linalg.norm(latent[i]) * np.linalg.norm(latent[j]) + 1e-8
                )
                mi[i, j] = max(0.05, 0.3 + 0.5 * sim)
    return mi


def mi_based_keff(mi_matrix, model_indices):
    """Compute K_eff using MI-based formula (same as production code)."""
    K = len(model_indices)
    if K <= 1:
        return 1.0
    keff = 0.0
    for i_idx in model_indices:
        term = 1.0
        for j_idx in model_indices:
            if i_idx != j_idx:
                term += np.exp(-2 * mi_matrix[i_idx, j_idx])
        keff += term
    return keff / K


def multi_membership_crve(y, X, panel_jids, n_judges):
    """Compute CRVE sandwich SE for multi-membership clustering.
    
    panel_jids: list of tuples, each tuple = judge indices in that panel
    """
    n = len(y)
    p = X.shape[1]
    
    # OLS fit
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    
    # Meat: sum score contributions by judge
    # For each judge, sum X_i * e_i for all panels containing that judge
    score_by_judge = np.zeros((n_judges, p))
    for i in range(n):
        for j in panel_jids[i]:
            score_by_judge[j] += X[i] * resid[i]
    
    # HC1-like correction: G/(G-1) * (n-1)/(n-p)
    G = n_judges
    hc1 = (G / (G - 1)) * ((n - 1) / (n - p))
    
    meat = np.zeros((p, p))
    for j in range(n_judges):
        s = score_by_judge[j].reshape(-1, 1)
        meat += s @ s.T
    meat *= hc1
    
    V_crve = XtX_inv @ meat @ XtX_inv
    se_crve = np.sqrt(np.diag(V_crve))
    
    return beta, se_crve


def run_power_sim(K, panels, keff_values, sigma_judge, beta_std, n_reps, rng):
    """Run Monte Carlo power simulation for one condition.
    
    Returns: power (fraction of replicates rejecting H0)
    """
    n_panels = len(panels)
    
    # Standardize K_eff
    keff_arr = np.array(keff_values)
    keff_mean = keff_arr.mean()
    keff_sd = keff_arr.std()
    if keff_sd < 1e-10:
        return 0.0, 0.0, 0.0  # degenerate
    keff_std = (keff_arr - keff_mean) / keff_sd
    
    # Design matrix: intercept + K_eff_std
    X = np.column_stack([np.ones(n_panels), keff_std])
    
    df = N_JUDGES - 1  # t-distribution df
    t_crit = scipy_stats.t.ppf(1 - ALPHA / 2, df)
    
    reject_count = 0
    beta_estimates = []
    se_estimates = []
    
    for rep in range(n_reps):
        # Generate judge random effects
        judge_re = rng.randn(N_JUDGES) * sigma_judge
        
        # Panel-level judge effect: mean of judge REs in each panel
        panel_re = np.array([np.mean([judge_re[j] for j in panel]) for panel in panels])
        
        # Generate response: Y = 0 + beta_std * keff_std + panel_RE + noise
        noise = rng.randn(n_panels) * SIGMA_EPS
        y = beta_std * keff_std + panel_re + noise
        
        # Fit CRVE
        beta_hat, se_hat = multi_membership_crve(y, X, panels, N_JUDGES)
        
        beta_keff = beta_hat[1]
        se_keff = se_hat[1]
        
        beta_estimates.append(beta_keff)
        se_estimates.append(se_keff)
        
        # t-test
        t_stat = beta_keff / se_keff if se_keff > 1e-10 else 0.0
        if abs(t_stat) > t_crit:
            reject_count += 1
    
    power = reject_count / n_reps
    mean_beta = np.mean(beta_estimates)
    mean_se = np.mean(se_estimates)
    
    return power, mean_beta, mean_se


def main():
    print("=" * 60)
    print("CRVE Monte Carlo Power Simulation")
    print("=" * 60)
    t0 = time.time()
    
    # Generate MI matrix (deterministic)
    mi_matrix = generate_mi_matrix(N_JUDGES)
    
    # Pre-compute panels and K_eff for each panel size
    panel_data = {}
    for K in PANEL_SIZES:
        panels = list(itertools.combinations(range(N_JUDGES), K))
        keff_values = [mi_based_keff(mi_matrix, list(p)) for p in panels]
        panel_data[K] = {
            "panels": panels,
            "keff_values": keff_values,
            "n_panels": len(panels),
            "keff_mean": float(np.mean(keff_values)),
            "keff_std": float(np.std(keff_values)),
            "keff_range": [float(np.min(keff_values)), float(np.max(keff_values))],
        }
        print(f"\nK={K}: {len(panels)} panels, "
              f"K_eff mean={np.mean(keff_values):.3f} "
              f"std={np.std(keff_values):.3f} "
              f"range=[{np.min(keff_values):.3f}, {np.max(keff_values):.3f}]")
    
    # Also compute OLS SE for reference (to show SE inflation)
    print(f"\nRunning {N_REPLICATES} replicates per condition...")
    
    results = []
    rng = np.random.RandomState(2026)
    
    for beta_std in BETA_STD_VALUES:
        for sigma_judge in SIGMA_JUDGE_VALUES:
            row = {"beta_std": beta_std, "sigma_judge": sigma_judge}
            for K in PANEL_SIZES:
                panels = panel_data[K]["panels"]
                keff_values = panel_data[K]["keff_values"]
                
                power, mean_beta, mean_se = run_power_sim(
                    K, panels, keff_values, sigma_judge, beta_std,
                    N_REPLICATES, rng
                )
                
                row[f"K{K}_power"] = power
                row[f"K{K}_mean_beta"] = mean_beta
                row[f"K{K}_mean_se"] = mean_se
                
                print(f"  β={beta_std}, σ_j={sigma_judge}, K={K}: "
                      f"power={power:.3f}, mean_β={mean_beta:.3f}, "
                      f"mean_SE={mean_se:.3f}")
            
            results.append(row)
    
    # Summary table
    print("\n" + "=" * 80)
    print("POWER TABLE (α=0.05, two-sided, df=14)")
    print("=" * 80)
    print(f"{'β_std':>6} {'σ_judge':>8} | {'K=3':>10} {'K=5':>10} {'K=7':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['beta_std']:>6.1f} {r['sigma_judge']:>8.1f} | "
              f"{r['K3_power']:>10.3f} {r['K5_power']:>10.3f} {r['K7_power']:>10.3f}")
    
    # Compute SE inflation factors
    print("\n" + "=" * 80)
    print("MEAN CRVE SE (higher = more inflation from judge overlap)")
    print("=" * 80)
    print(f"{'β_std':>6} {'σ_judge':>8} | {'K=3':>10} {'K=5':>10} {'K=7':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['beta_std']:>6.1f} {r['sigma_judge']:>8.1f} | "
              f"{r['K3_mean_se']:>10.4f} {r['K5_mean_se']:>10.4f} {r['K7_mean_se']:>10.4f}")
    
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Save results
    output = {
        "method": "monte_carlo_power_simulation",
        "description": "CRVE power analysis across panel sizes K=3,5,7",
        "n_judges": N_JUDGES,
        "n_replicates": N_REPLICATES,
        "alpha": ALPHA,
        "sigma_eps": SIGMA_EPS,
        "panel_info": {
            f"K{K}": {
                "n_panels": panel_data[K]["n_panels"],
                "keff_mean": panel_data[K]["keff_mean"],
                "keff_std": panel_data[K]["keff_std"],
                "keff_range": panel_data[K]["keff_range"],
            }
            for K in PANEL_SIZES
        },
        "power_table": results,
        "conclusion": {
            "K3_underpowered": any(
                r["K3_power"] < 0.5 and r["beta_std"] == 0.3
                for r in results
            ),
            "K5_K7_higher_power": all(
                r["K5_power"] > r["K3_power"] and r["K7_power"] > r["K3_power"]
                for r in results
                if r["beta_std"] == 0.3
            ),
        },
        "elapsed_seconds": elapsed,
    }
    
    out_path = Path("/root/cert_manip_resist_eval/artifacts/results/plan001/crossed_re_power.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
