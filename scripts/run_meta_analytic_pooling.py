#!/usr/bin/env python3
"""R6 2A: DerSimonian-Laird random-effects meta-analysis.

Pools β_std for K_eff and η_max across 12 conditions for each K=3,5,7.
Output: artifacts/results/plan001/meta_analytic_pooling.json
"""

import sys
import json
import time
import itertools
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.unified_data_loader import (
    ALL_MODELS, DATASETS, ATTACKS,
    load_all_scores, load_pairs, load_mi_data,
    _is_valid_score,
)

BACKUP_DIR = PROJECT_ROOT / "artifacts" / "server_backup" / "results"
LOCAL_CKPTS = [
    BACKUP_DIR / "_taxonomy_checkpoint.json",
    BACKUP_DIR / "_step2v3_checkpoint.json",
]
LOCAL_SCORES_DIR = BACKUP_DIR / "plan001" / "individual_scores"
LOCAL_MI_PATH = BACKUP_DIR / "plan001" / "mi_matrix" / "mi_matrix_15models.json"

EPS = 1e-6
N_MODELS = len(ALL_MODELS)


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


def compute_individual_asr(clean_list, attacked_list, pairs):
    n = min(len(clean_list), len(attacked_list), len(pairs))
    n_flips = n_correct = 0
    for i in range(n):
        if not _is_valid_score(clean_list[i]) or not _is_valid_score(attacked_list[i]):
            continue
        gt = pairs[i]["ground_truth_winner"]
        if clean_list[i]["winner"] == gt:
            n_correct += 1
            if attacked_list[i]["winner"] != gt:
                n_flips += 1
    return n_flips / max(n_correct, 1)


def compute_panel_asr(jids, clean_ds, attacked_ds_atk, pairs):
    n = min(len(pairs), *(len(clean_ds[j]) for j in jids),
            *(len(attacked_ds_atk[j]) for j in jids))
    n_flips = n_correct = 0
    for i in range(n):
        if not all(_is_valid_score(clean_ds[j][i]) for j in jids):
            continue
        if not all(_is_valid_score(attacked_ds_atk[j][i]) for j in jids):
            continue
        gt = pairs[i]["ground_truth_winner"]
        cv = majority_vote([clean_ds[j][i]["winner"] for j in jids])
        if cv != gt:
            continue
        n_correct += 1
        av = majority_vote([attacked_ds_atk[j][i]["winner"] for j in jids])
        if av != gt:
            n_flips += 1
    return n_flips / max(n_correct, 1)


def mi_based_keff(mi_matrix, model_indices):
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


def dersimonian_laird(betas, variances):
    """DerSimonian-Laird random-effects meta-analysis."""
    betas = np.array(betas, dtype=float)
    variances = np.array(variances, dtype=float)
    k = len(betas)

    w = 1.0 / variances
    w_sum = np.sum(w)
    beta_fe = np.sum(w * betas) / w_sum
    Q = np.sum(w * (betas - beta_fe) ** 2)
    Q_df = k - 1
    Q_p = 1.0 - stats.chi2.cdf(Q, Q_df)
    I2 = max(0, (Q - Q_df) / Q) * 100 if Q > 0 else 0.0

    C = w_sum - np.sum(w ** 2) / w_sum
    tau2 = max(0, (Q - Q_df) / C)

    w_star = 1.0 / (variances + tau2)
    w_star_sum = np.sum(w_star)
    beta_pooled = np.sum(w_star * betas) / w_star_sum
    se_pooled = 1.0 / np.sqrt(w_star_sum)
    ci_lo = beta_pooled - 1.96 * se_pooled
    ci_hi = beta_pooled + 1.96 * se_pooled

    return {
        "pooled_beta_std": float(beta_pooled),
        "se_pooled": float(se_pooled),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "I2_pct": float(I2),
        "tau2": float(tau2),
        "Q": float(Q),
        "Q_df": int(Q_df),
        "Q_p": float(Q_p),
        "k": int(k),
    }


def main():
    t_start = time.time()
    print("Loading data...")
    clean, attacked = load_all_scores(
        checkpoint_files=LOCAL_CKPTS, individual_dir=LOCAL_SCORES_DIR)
    pairs = load_pairs(checkpoint_files=LOCAL_CKPTS)
    mi_data = load_mi_data(mi_path=LOCAL_MI_PATH)
    mi_matrices = mi_data["mi_matrix"]

    models = ALL_MODELS
    cond_keys = [f"{ds}x{atk}" for ds in DATASETS for atk in ATTACKS]

    # Precompute individual ASRs
    ind_asr = {}
    for ds in DATASETS:
        for atk in ATTACKS:
            for mid in models:
                if mid in clean[ds] and mid in attacked[ds][atk]:
                    ind_asr[(ds, atk, mid)] = compute_individual_asr(
                        clean[ds][mid], attacked[ds][atk][mid], pairs[ds])

    meta_results = {}

    for K in [3, 5, 7]:
        print(f"\n{'=' * 60}")
        print(f"K = {K}")
        print(f"{'=' * 60}")

        all_combos = list(itertools.combinations(range(N_MODELS), K))
        n_combos = len(all_combos)
        print(f"Total panels: {n_combos}")

        # Precompute K_eff per (combo, ds)
        keff_cache = {}
        for ds in DATASETS:
            mi_mat = np.array(mi_matrices[ds])
            for combo in all_combos:
                keff_cache[(combo, ds)] = mi_based_keff(mi_mat, list(combo))

        ols_results = {}

        for ci, ck in enumerate(cond_keys):
            ds, atk = ck.split("x", 1)
            t0 = time.time()

            ds_clean = clean.get(ds, {})
            ds_attacked = attacked.get(ds, {}).get(atk, {})
            ds_pairs = pairs.get(ds, [])

            if not ds_clean or not ds_attacked or not ds_pairs:
                ols_results[ck] = {"skipped": True}
                continue

            panel_asrs = []
            panel_keffs = []
            panel_etas = []

            for combo in all_combos:
                jids = tuple(models[i] for i in combo)
                if not all(j in ds_clean and j in ds_attacked for j in jids):
                    continue
                if not all((ds, atk, j) in ind_asr for j in jids):
                    continue

                keff = keff_cache[(combo, ds)]
                eta_max = max(ind_asr[(ds, atk, j)] for j in jids)

                cs = {j: ds_clean[j] for j in jids}
                ats = {j: ds_attacked[j] for j in jids}
                asr = compute_panel_asr(jids, cs, ats, ds_pairs)

                panel_asrs.append(asr)
                panel_keffs.append(keff)
                panel_etas.append(eta_max)

            n_p = len(panel_asrs)
            if n_p < 10:
                ols_results[ck] = {"skipped": True, "n_panels": n_p}
                continue

            y = np.log(np.array(panel_asrs) + EPS)
            log_keff = np.log(np.array(panel_keffs))
            log_eta = np.log(np.array(panel_etas) + EPS)
            sd_y = np.std(y)
            sd_keff = np.std(log_keff)
            sd_eta = np.std(log_eta)

            # OLS: y ~ 1 + log_keff + log_eta
            X = np.column_stack([np.ones(n_p), log_keff, log_eta])
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ (X.T @ y)
            resid = y - X @ beta
            s2 = np.sum(resid ** 2) / (n_p - 3)
            se_raw = np.sqrt(s2 * np.diag(XtX_inv))
            t_vals = beta / se_raw

            scale_keff = sd_keff / sd_y if sd_y > EPS else 0.0
            scale_eta = sd_eta / sd_y if sd_y > EPS else 0.0

            ols_results[ck] = {
                "n_panels": n_p,
                "keff_beta_std": float(beta[1] * scale_keff),
                "keff_se_std": float(se_raw[1] * scale_keff),
                "keff_t": float(t_vals[1]),
                "eta_beta_std": float(beta[2] * scale_eta),
                "eta_se_std": float(se_raw[2] * scale_eta),
                "eta_t": float(t_vals[2]),
                "R2": float(1 - np.sum(resid**2) / np.sum((y - np.mean(y))**2)),
                "skipped": False,
            }

            elapsed = time.time() - t0
            r = ols_results[ck]
            print(f"  [{ci+1}/12] {ck}: keff={r['keff_beta_std']:.4f}±{r['keff_se_std']:.4f}, "
                  f"eta={r['eta_beta_std']:.4f}±{r['eta_se_std']:.4f} ({elapsed:.1f}s)")

        # Meta-analysis
        valid = [ck for ck in cond_keys if not ols_results.get(ck, {}).get("skipped", True)]

        keff_betas = [ols_results[ck]["keff_beta_std"] for ck in valid]
        keff_vars = [ols_results[ck]["keff_se_std"] ** 2 for ck in valid]
        eta_betas = [ols_results[ck]["eta_beta_std"] for ck in valid]
        eta_vars = [ols_results[ck]["eta_se_std"] ** 2 for ck in valid]

        keff_meta = dersimonian_laird(keff_betas, keff_vars)
        eta_meta = dersimonian_laird(eta_betas, eta_vars)

        meta_results[f"K{K}"] = {
            "n_conditions": len(valid),
            "conditions": valid,
            "keff": {
                "meta": keff_meta,
                "per_condition": {
                    ck: {
                        "beta_std": ols_results[ck]["keff_beta_std"],
                        "se_std": ols_results[ck]["keff_se_std"],
                        "n_panels": ols_results[ck]["n_panels"],
                    }
                    for ck in valid
                },
            },
            "eta_max": {
                "meta": eta_meta,
                "per_condition": {
                    ck: {
                        "beta_std": ols_results[ck]["eta_beta_std"],
                        "se_std": ols_results[ck]["eta_se_std"],
                        "n_panels": ols_results[ck]["n_panels"],
                    }
                    for ck in valid
                },
            },
        }

        keff_ci = keff_meta["ci_95"]
        eta_ci = eta_meta["ci_95"]
        print(f"\n  META K={K} ({len(valid)} conditions):")
        print(f"    K_eff: pooled={keff_meta['pooled_beta_std']:.4f} "
              f"[{keff_ci[0]:.4f}, {keff_ci[1]:.4f}], "
              f"I²={keff_meta['I2_pct']:.1f}%, Q p={keff_meta['Q_p']:.4f}")
        print(f"    η_max: pooled={eta_meta['pooled_beta_std']:.4f} "
              f"[{eta_ci[0]:.4f}, {eta_ci[1]:.4f}], "
              f"I²={eta_meta['I2_pct']:.1f}%, Q p={eta_meta['Q_p']:.4f}")

    # Summary table
    print(f"\n{'='*75}")
    print(f"{'K':>3} {'Predictor':>10} {'Pooled β_std':>13} {'95% CI':>22} {'I²':>7} {'Q p':>8}")
    print(f"{'-'*75}")
    for K in [3, 5, 7]:
        for pred in ["keff", "eta_max"]:
            m = meta_results[f"K{K}"][pred]["meta"]
            ci = m["ci_95"]
            label = "K_eff" if pred == "keff" else "η_max"
            print(f"{K:>3} {label:>10} {m['pooled_beta_std']:>13.4f} "
                  f"[{ci[0]:>9.4f}, {ci[1]:>8.4f}] {m['I2_pct']:>6.1f}% {m['Q_p']:>8.4f}")
    print(f"{'='*75}")

    total = time.time() - t_start
    print(f"\nTotal time: {total:.0f}s")

    # Save
    output = {
        "analysis": "meta_analytic_pooling",
        "purpose": "R6 2A: DerSimonian-Laird random-effects pooling of β_std across 12 conditions",
        "method": "DerSimonian-Laird",
        "model_spec": "log(ASR+ε) ~ log(K_eff) + log(η_max+ε)",
        "n_models": N_MODELS,
        "models": ALL_MODELS,
        "results": meta_results,
    }
    out_path = PROJECT_ROOT / "artifacts" / "results" / "plan001" / "meta_analytic_pooling.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default, ensure_ascii=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
