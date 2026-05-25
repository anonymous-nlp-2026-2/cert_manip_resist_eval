#!/usr/bin/env python3
"""Bootstrap 95% CI for Random Fault R² vs Adaptive R².

Compares OLS R² under two regimes:
  1. Random Fault (RF): independent Bernoulli errors based on clean accuracy
     → K_eff theory works → high R²
  2. Adaptive: actual adversarial attack ASR
     → K_eff theory breaks → low R²

Methodology:
  - K=5 panels (C(15,5) = 3003 panels), majority vote
  - OLS: log(ASR + eps) ~ log(K_eff) + log(eta_max + eps)
  - RF: 2 conditions (1 per dataset), averaged
  - Adaptive: 12 conditions (2 datasets × 6 attacks), averaged
  - 1000 bootstrap resamples at panel level
"""

import os
os.environ["HF_HOME"] = "/root/autodl-tmp/.hf_cache"

import sys
sys.path.insert(0, "/root/cert_manip_resist_eval")

import json
import time
import itertools
import numpy as np
import statsmodels.api as sm
from pathlib import Path

from src.unified_data_loader import (
    ALL_MODELS, DATASETS, ATTACKS,
    load_all_scores, load_pairs, load_mi_data,
    _is_valid_score,
)

EPS = 1e-6
N_BOOT = 1000
SEED = 42
K = 5
N_SIM = 10000


def _json_default(obj):
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def majority_vote(votes):
    counts = {"A": 0, "B": 0, "tie": 0}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    mc = max(counts.values())
    winners = [k for k, c in counts.items() if c == mc]
    if len(winners) == 1:
        return winners[0]
    for w in winners:
        if w != "tie":
            return w
    return "tie"


def compute_panel_asr(jids, clean_ds, attacked_ds, pairs_ds):
    n = min(len(pairs_ds), *(len(clean_ds[j]) for j in jids),
            *(len(attacked_ds[j]) for j in jids))
    nf = nc = 0
    for i in range(n):
        if not all(_is_valid_score(clean_ds[j][i]) for j in jids):
            continue
        if not all(_is_valid_score(attacked_ds[j][i]) for j in jids):
            continue
        gt = pairs_ds[i]["ground_truth_winner"]
        cv = majority_vote([clean_ds[j][i]["winner"] for j in jids])
        if cv != gt:
            continue
        nc += 1
        av = majority_vote([attacked_ds[j][i]["winner"] for j in jids])
        if av != gt:
            nf += 1
    return nf / max(nc, 1)


def compute_individual_asr(cl, at, pairs):
    n = min(len(cl), len(at), len(pairs))
    nf = nc = 0
    for i in range(n):
        if not _is_valid_score(cl[i]) or not _is_valid_score(at[i]):
            continue
        gt = pairs[i]["ground_truth_winner"]
        if cl[i]["winner"] == gt:
            nc += 1
            if at[i]["winner"] != gt:
                nf += 1
    return nf / max(nc, 1)


def mi_based_keff(mi_mat, indices):
    K = len(indices)
    if K <= 1:
        return 1.0
    keff = 0.0
    for i in indices:
        term = 1.0
        for j in indices:
            if i != j:
                term += np.exp(-2 * mi_mat[i, j])
        keff += term
    return keff / K


def simulate_rf_asr(error_rates, n_sim, rng):
    """Simulate Random Fault ASR for one panel via Monte Carlo.
    
    error_rates: (K,) array of per-judge error rates (1 - clean_accuracy)
    Returns: mean ASR across n_sim simulations
    """
    K = len(error_rates)
    threshold = (K + 1) // 2  # majority: ≥3 for K=5
    randoms = rng.random((n_sim, K))
    errors = randoms < error_rates[np.newaxis, :]
    majority_wrong = errors.sum(axis=1) >= threshold
    return float(majority_wrong.mean())


def ols_r2(keffs, eta_maxs, asrs):
    """OLS: log(ASR+eps) ~ log(K_eff) + log(eta_max+eps), return R²."""
    y = np.log(np.array(asrs) + EPS)
    x1 = np.log(np.array(keffs))
    x2 = np.log(np.array(eta_maxs) + EPS)
    X = sm.add_constant(np.column_stack([x1, x2]))
    try:
        return float(sm.OLS(y, X).fit().rsquared)
    except Exception:
        return np.nan


def main():
    print("=" * 60)
    print("R² Bootstrap CI: Random Fault vs Adaptive")
    print("=" * 60)

    print("\nLoading data...")
    mi_data = load_mi_data()
    clean, attacked = load_all_scores()
    pairs = load_pairs()
    n_models = len(ALL_MODELS)
    rng = np.random.default_rng(SEED)

    # Precompute clean accuracy per model per dataset
    print("Computing clean accuracy...")
    clean_acc = {}
    for ds in DATASETS:
        for mid in ALL_MODELS:
            if mid in clean[ds]:
                scores = clean[ds][mid]
                p = pairs[ds]
                n = min(len(scores), len(p))
                correct = sum(
                    1 for i in range(n)
                    if _is_valid_score(scores[i]) and scores[i]["winner"] == p[i]["ground_truth_winner"]
                )
                total = sum(1 for i in range(n) if _is_valid_score(scores[i]))
                clean_acc[(ds, mid)] = correct / max(total, 1)

    # Precompute individual ASR
    print("Computing individual ASR...")
    ind_asr = {}
    for ds in DATASETS:
        for atk in ATTACKS:
            for mid in ALL_MODELS:
                if mid in clean[ds] and mid in attacked[ds][atk]:
                    ind_asr[(ds, atk, mid)] = compute_individual_asr(
                        clean[ds][mid], attacked[ds][atk][mid], pairs[ds]
                    )

    # Generate all K=5 panels
    combos = list(itertools.combinations(range(n_models), K))
    n_panels = len(combos)
    print(f"K={K}: {n_panels} panels")

    # Build panel data arrays
    # RF regime: 2 conditions (1 per dataset)
    # Adaptive regime: 12 conditions (2 ds × 6 attacks)
    print("\nBuilding panel data...")

    keff_per_ds = {}
    for ds in DATASETS:
        mi_mat = np.array(mi_data["mi_matrix"][ds])
        keffs = np.array([mi_based_keff(mi_mat, list(c)) for c in combos])
        keff_per_ds[ds] = keffs
        print(f"  K_eff[{ds}]: mean={keffs.mean():.4f}")

    # RF regime: simulate ASR and compute eta_max from error rates
    print("\nSimulating Random Fault regime...")
    rf_asr = {}
    rf_eta = {}
    for ds in DATASETS:
        t0 = time.time()
        asrs = np.zeros(n_panels)
        etas = np.zeros(n_panels)
        for pi, combo in enumerate(combos):
            jids = [ALL_MODELS[i] for i in combo]
            error_rates = np.array([1 - clean_acc.get((ds, mid), 0) for mid in jids])
            asrs[pi] = simulate_rf_asr(error_rates, N_SIM, rng)
            etas[pi] = error_rates.max()
        rf_asr[ds] = asrs
        rf_eta[ds] = etas
        print(f"  {ds}: mean ASR={asrs.mean():.4f}, mean eta={etas.mean():.4f} ({time.time()-t0:.1f}s)")

    # Adaptive regime: compute panel ASR for each condition
    print("\nComputing Adaptive regime panel ASR...")
    adaptive_asr = {}
    adaptive_eta = {}
    conditions = [(ds, atk) for ds in DATASETS for atk in ATTACKS]
    for ds, atk in conditions:
        cond = f"{ds}×{atk}"
        t0 = time.time()
        asrs = np.zeros(n_panels)
        etas = np.zeros(n_panels)
        for pi, combo in enumerate(combos):
            jids = [ALL_MODELS[i] for i in combo]
            if not all(mid in clean[ds] and mid in attacked[ds][atk] for mid in jids):
                asrs[pi] = np.nan
                etas[pi] = np.nan
                continue
            asrs[pi] = compute_panel_asr(jids, clean[ds], attacked[ds][atk], pairs[ds])
            etas[pi] = max(ind_asr.get((ds, atk, mid), 0) for mid in jids)
        adaptive_asr[(ds, atk)] = asrs
        adaptive_eta[(ds, atk)] = etas
        valid = ~np.isnan(asrs)
        print(f"  {cond}: {valid.sum()} panels, mean ASR={asrs[valid].mean():.4f} ({time.time()-t0:.1f}s)")

    # Compute R² for all conditions
    def compute_all_r2(idx=None):
        """Compute RF and Adaptive mean R² for given panel indices."""
        if idx is None:
            idx = np.arange(n_panels)

        # RF R² (2 conditions)
        rf_r2s = []
        for ds in DATASETS:
            k = keff_per_ds[ds][idx]
            e = rf_eta[ds][idx]
            a = rf_asr[ds][idx]
            r2 = ols_r2(k, e, a)
            if not np.isnan(r2):
                rf_r2s.append(r2)

        # Adaptive R² (12 conditions)
        ad_r2s = []
        for ds, atk in conditions:
            a = adaptive_asr[(ds, atk)][idx]
            e = adaptive_eta[(ds, atk)][idx]
            k = keff_per_ds[ds][idx]
            valid = ~np.isnan(a)
            if valid.sum() < 20:
                continue
            r2 = ols_r2(k[valid], e[valid], a[valid])
            if not np.isnan(r2):
                ad_r2s.append(r2)

        rf_mean = float(np.mean(rf_r2s)) if rf_r2s else np.nan
        ad_mean = float(np.mean(ad_r2s)) if ad_r2s else np.nan
        return rf_mean, ad_mean, rf_r2s, ad_r2s

    # Point estimates
    print("\nComputing point estimates...")
    rf_mean, ad_mean, rf_per_cond, ad_per_cond = compute_all_r2()
    delta = rf_mean - ad_mean

    print(f"  RF R² per dataset: {[f'{r:.4f}' for r in rf_per_cond]}")
    print(f"  RF R² mean: {rf_mean:.4f}")
    print(f"  Adaptive R² per condition: {[f'{r:.4f}' for r in ad_per_cond]}")
    print(f"  Adaptive R² mean: {ad_mean:.4f}")
    print(f"  Delta (RF - Adaptive): {delta:.4f}")

    # Bootstrap
    print(f"\nRunning {N_BOOT} bootstrap resamples...")
    boot_rf = np.zeros(N_BOOT)
    boot_ad = np.zeros(N_BOOT)
    boot_delta = np.zeros(N_BOOT)

    t0 = time.time()
    for b in range(N_BOOT):
        if (b + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (b + 1) * (N_BOOT - b - 1)
            print(f"  Bootstrap {b+1}/{N_BOOT} ({elapsed:.0f}s, ~{eta:.0f}s remaining)")

        idx = rng.choice(n_panels, size=n_panels, replace=True)
        r, a, _, _ = compute_all_r2(idx)
        boot_rf[b] = r
        boot_ad[b] = a
        boot_delta[b] = r - a

    # Remove invalid bootstraps
    valid = ~(np.isnan(boot_rf) | np.isnan(boot_ad))
    boot_rf = boot_rf[valid]
    boot_ad = boot_ad[valid]
    boot_delta = boot_delta[valid]
    n_valid = len(boot_rf)

    # Percentile CI
    rf_ci = (float(np.percentile(boot_rf, 2.5)), float(np.percentile(boot_rf, 97.5)))
    ad_ci = (float(np.percentile(boot_ad, 2.5)), float(np.percentile(boot_ad, 97.5)))
    delta_ci = (float(np.percentile(boot_delta, 2.5)), float(np.percentile(boot_delta, 97.5)))
    delta_sig = "significant" if delta_ci[0] > 0 else "not significant"

    total_time = time.time() - t0
    print(f"\nValid bootstraps: {n_valid}/{N_BOOT}")
    print(f"Total bootstrap time: {total_time:.0f}s")

    print("\n" + "=" * 70)
    print(f"RESULTS: R² Bootstrap CI (K={K}, {n_panels} panels)")
    print("=" * 70)
    print(f"  RF (Random Fault) R²:  {rf_mean:.4f}  95% CI [{rf_ci[0]:.4f}, {rf_ci[1]:.4f}]")
    print(f"  Adaptive R²:           {ad_mean:.4f}  95% CI [{ad_ci[0]:.4f}, {ad_ci[1]:.4f}]")
    print(f"  Δ (RF − Adaptive):     {delta:.4f}  95% CI [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]")
    print(f"  Δ CI excludes 0:       {delta_sig}")
    print("=" * 70)

    output = {
        "analysis": "r2_bootstrap_ci",
        "description": "Bootstrap 95% CI for Random Fault vs Adaptive OLS R²",
        "panel_size": K,
        "n_panels": n_panels,
        "n_rf_conditions": len(DATASETS),
        "n_adaptive_conditions": len(conditions),
        "n_bootstrap": N_BOOT,
        "n_valid_bootstrap": n_valid,
        "seed": SEED,
        "eps": EPS,
        "n_sim_rf": N_SIM,
        "regression": "log(ASR+eps) ~ log(K_eff) + log(eta_max+eps)",
        "point_estimates": {
            "rf_r2_mean": rf_mean,
            "adaptive_r2_mean": ad_mean,
            "delta": delta,
            "rf_r2_per_dataset": dict(zip(DATASETS, rf_per_cond)),
            "adaptive_r2_per_condition": dict(zip(
                [f"{ds}×{atk}" for ds, atk in conditions], ad_per_cond
            )),
        },
        "bootstrap_95ci": {
            "rf_r2": {"lower": rf_ci[0], "upper": rf_ci[1]},
            "adaptive_r2": {"lower": ad_ci[0], "upper": ad_ci[1]},
            "delta": {"lower": delta_ci[0], "upper": delta_ci[1]},
        },
        "bootstrap_stats": {
            "rf_r2": {"mean": float(boot_rf.mean()), "std": float(boot_rf.std())},
            "adaptive_r2": {"mean": float(boot_ad.mean()), "std": float(boot_ad.std())},
            "delta": {"mean": float(boot_delta.mean()), "std": float(boot_delta.std())},
        },
        "significance": delta_sig,
    }

    out_path = Path("/root/cert_manip_resist_eval/artifacts/results/plan001/r2_bootstrap_ci.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
