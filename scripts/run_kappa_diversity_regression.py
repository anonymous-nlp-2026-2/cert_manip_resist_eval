#!/usr/bin/env python3
"""κ-diversity vs K_eff regression comparison (R6 Batch 1A).

Runs identical WCB regressions with κ-diversity and K_eff as the diversity
predictor. Both use B=9999, Rademacher weights, K-position max-p clustering,
and BH FDR correction within each K.

Output: artifacts/results/plan001/kappa_diversity_regression.json
"""

import sys
import json
import time
import itertools
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

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
FDR_ALPHA = 0.05
B_BOOT = 9999
SEED = 42
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


def pairwise_kappa(clean_i, clean_j):
    n = min(len(clean_i), len(clean_j))
    categories = ["A", "B", "tie"]
    n_valid = n_agree = 0
    cnt_i = {c: 0 for c in categories}
    cnt_j = {c: 0 for c in categories}
    for idx in range(n):
        if not _is_valid_score(clean_i[idx]) or not _is_valid_score(clean_j[idx]):
            continue
        n_valid += 1
        vi = clean_i[idx]["winner"]
        vj = clean_j[idx]["winner"]
        cnt_i[vi] += 1
        cnt_j[vj] += 1
        if vi == vj:
            n_agree += 1
    if n_valid == 0:
        return 0.0
    p_o = n_agree / n_valid
    p_e = sum(cnt_i[c] * cnt_j[c] for c in categories) / (n_valid * n_valid)
    if p_e >= 1.0:
        return 0.0
    return (p_o - p_e) / (1 - p_e)


def panel_kappa_diversity(jids, clean_ds):
    kappas = []
    for a in range(len(jids)):
        for b in range(a + 1, len(jids)):
            kappas.append(pairwise_kappa(clean_ds[jids[a]], clean_ds[jids[b]]))
    return 1 - np.mean(kappas)


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


def wild_cluster_bootstrap_vec(y, X, cluster_ids, B=B_BOOT, seed=SEED,
                               test_col_idx=1):
    n, p = X.shape
    uq = np.unique(cluster_ids)
    G = len(uq)
    c2i = {c: i for i, c in enumerate(uq)}
    cidx = np.array([c2i[c] for c in cluster_ids])

    XtX = X.T @ X
    Li = np.linalg.inv(XtX)
    b_full = Li @ (X.T @ y)
    e_full = y - X @ b_full
    s2 = np.sum(e_full ** 2) / (n - p)
    t_orig = b_full[test_col_idx] / np.sqrt(s2 * Li[test_col_idx, test_col_idx])

    cr = [i for i in range(p) if i != test_col_idx]
    Xr = X[:, cr]
    br = np.linalg.solve(Xr.T @ Xr, Xr.T @ y)
    yhr = Xr @ br
    er = y - yhr

    Xe = X.T * er[None, :]
    Xtyhr = X.T @ yhr
    yhr_sq = float(np.sum(yhr ** 2))
    er_sq = float(np.sum(er ** 2))
    yhe = yhr * er

    rng = np.random.RandomState(seed)
    Wc = 2.0 * rng.randint(0, 2, size=(B, G)).astype(np.float64) - 1.0
    W = Wc[:, cidx]

    XtWe = Xe @ W.T
    XtY = Xtyhr[:, None] + XtWe
    Beta = Li @ XtY

    YtY = yhr_sq + er_sq + 2.0 * (W @ yhe)
    BXY = np.sum(Beta * XtY, axis=0)
    RSS = YtY - BXY
    S2 = RSS / (n - p)

    SE = np.sqrt(np.maximum(S2, 0) * Li[test_col_idx, test_col_idx])
    T = Beta[test_col_idx, :] / np.maximum(SE, 1e-30)

    pv = (1 + np.sum(np.abs(T) >= np.abs(t_orig))) / (1 + B)
    return float(pv), float(t_orig), int(G)


def bootstrap_K_positions(y, X, panel_jids_list, K, B=B_BOOT, seed=SEED,
                          test_col_idx=1):
    pos_ps = []
    pos_clusters = []
    for pos in range(K):
        cids = np.array([jids[pos] for jids in panel_jids_list])
        pv, _, G = wild_cluster_bootstrap_vec(
            y, X, cids, B=B, seed=seed, test_col_idx=test_col_idx)
        pos_ps.append(pv)
        pos_clusters.append(G)
    return pos_ps, max(pos_ps), pos_clusters


def run_for_K(K, clean, attacked, pairs, mi_data):
    print(f"\n{'=' * 60}")
    print(f"K = {K}")
    print(f"{'=' * 60}")

    models = ALL_MODELS
    all_combos = list(itertools.combinations(range(N_MODELS), K))
    n_combos = len(all_combos)
    print(f"Total panels: {n_combos}")

    mi_matrices = mi_data["mi_matrix"]

    ind_asr = {}
    for ds in DATASETS:
        for atk in ATTACKS:
            for mid in models:
                if mid in clean[ds] and mid in attacked[ds][atk]:
                    ind_asr[(ds, atk, mid)] = compute_individual_asr(
                        clean[ds][mid], attacked[ds][atk][mid], pairs[ds])

    print("Computing κ-diversity...")
    t0 = time.time()
    kd_cache = {}
    for ds in DATASETS:
        for combo in all_combos:
            jids = tuple(models[i] for i in combo)
            if all(j in clean[ds] for j in jids):
                kd_cache[(jids, ds)] = panel_kappa_diversity(jids, clean[ds])
    print(f"  Done: {len(kd_cache)} values in {time.time()-t0:.1f}s")

    print("Computing K_eff from MI matrix...")
    t0 = time.time()
    keff_cache = {}
    for ds in DATASETS:
        mi_mat = np.array(mi_matrices[ds])
        for combo in all_combos:
            jids = tuple(models[i] for i in combo)
            keff_cache[(jids, ds)] = mi_based_keff(mi_mat, list(combo))
    print(f"  Done: {len(keff_cache)} values in {time.time()-t0:.1f}s")

    cond_keys = [f"{ds}x{atk}" for ds in DATASETS for atk in ATTACKS]
    results = {}

    for ci, ck in enumerate(cond_keys):
        ds, atk = ck.split("x", 1)
        t0 = time.time()
        print(f"\n  [{ci+1}/12] {ck}")

        ds_clean = clean.get(ds, {})
        ds_attacked = attacked.get(ds, {}).get(atk, {})
        ds_pairs = pairs.get(ds, [])

        if not ds_clean or not ds_attacked or not ds_pairs:
            print("    SKIP: missing data")
            results[ck] = {"skipped": True}
            continue

        panel_asrs = []
        panel_kds = []
        panel_keffs = []
        panel_etas = []
        jids_list = []

        for combo in all_combos:
            jids = tuple(models[i] for i in combo)
            if not all(j in ds_clean and j in ds_attacked for j in jids):
                continue
            if not all((ds, atk, j) in ind_asr for j in jids):
                continue
            if (jids, ds) not in kd_cache:
                continue

            kd = kd_cache[(jids, ds)]
            keff = keff_cache[(jids, ds)]
            eta_max = max(ind_asr[(ds, atk, j)] for j in jids)

            cs = {j: ds_clean[j] for j in jids}
            ats = {j: ds_attacked[j] for j in jids}
            asr = compute_panel_asr(jids, cs, ats, ds_pairs)

            panel_asrs.append(asr)
            panel_kds.append(kd)
            panel_keffs.append(keff)
            panel_etas.append(eta_max)
            jids_list.append(jids)

        n_p = len(panel_asrs)
        print(f"    Panels: {n_p}")

        if n_p < 10:
            results[ck] = {"skipped": True, "n_panels": n_p}
            continue

        y = np.log(np.array(panel_asrs) + EPS)
        log_kd = np.log(np.array(panel_kds) + EPS)
        log_keff = np.log(np.array(panel_keffs))
        log_eta = np.log(np.array(panel_etas) + EPS)
        sd_y = np.std(y)

        # --- κ-diversity regression ---
        X_kd = np.column_stack([np.ones(n_p), log_kd, log_eta])
        XtX_kd = X_kd.T @ X_kd
        Li_kd = np.linalg.inv(XtX_kd)
        beta_kd = Li_kd @ (X_kd.T @ y)
        e_kd = y - X_kd @ beta_kd
        s2_kd = np.sum(e_kd**2) / (n_p - 3)
        t_vals_kd = beta_kd / np.sqrt(s2_kd * np.diag(Li_kd))

        sb_kd = float(beta_kd[1] * np.std(log_kd) / sd_y) if sd_y > EPS else 0.0
        sb_eta_kd = float(beta_kd[2] * np.std(log_eta) / sd_y) if sd_y > EPS else 0.0

        kd_ps, kd_max_p, kd_cls = bootstrap_K_positions(
            y, X_kd, jids_list, K, B=B_BOOT, seed=SEED, test_col_idx=1)
        eta_kd_ps, eta_kd_max_p, eta_kd_cls = bootstrap_K_positions(
            y, X_kd, jids_list, K, B=B_BOOT, seed=SEED+1, test_col_idx=2)

        # --- K_eff regression ---
        X_ke = np.column_stack([np.ones(n_p), log_keff, log_eta])
        XtX_ke = X_ke.T @ X_ke
        Li_ke = np.linalg.inv(XtX_ke)
        beta_ke = Li_ke @ (X_ke.T @ y)
        e_ke = y - X_ke @ beta_ke
        s2_ke = np.sum(e_ke**2) / (n_p - 3)
        t_vals_ke = beta_ke / np.sqrt(s2_ke * np.diag(Li_ke))

        sb_ke = float(beta_ke[1] * np.std(log_keff) / sd_y) if sd_y > EPS else 0.0
        sb_eta_ke = float(beta_ke[2] * np.std(log_eta) / sd_y) if sd_y > EPS else 0.0

        ke_ps, ke_max_p, ke_cls = bootstrap_K_positions(
            y, X_ke, jids_list, K, B=B_BOOT, seed=SEED, test_col_idx=1)
        eta_ke_ps, eta_ke_max_p, eta_ke_cls = bootstrap_K_positions(
            y, X_ke, jids_list, K, B=B_BOOT, seed=SEED+1, test_col_idx=2)

        # --- Rank correlation ---
        rho, rho_p = spearmanr(panel_kds, panel_keffs)

        elapsed = time.time() - t0
        print(f"    κ_div: std_β={sb_kd:.4f} max_p={kd_max_p:.4f}")
        print(f"    K_eff: std_β={sb_ke:.4f} max_p={ke_max_p:.4f}")
        print(f"    ρ(κ_div, K_eff)={rho:.4f} ({elapsed:.1f}s)")

        results[ck] = {
            "n_panels": n_p,
            "rank_corr_kd_keff": float(rho),
            "rank_corr_p": float(rho_p),
            # κ-diversity results
            "kd_boot_p": [float(p) for p in kd_ps],
            "kd_final_p": float(kd_max_p),
            "kd_ols_std_beta": sb_kd,
            "kd_ols_t": float(t_vals_kd[1]),
            "kd_ols_beta": float(beta_kd[1]),
            "kd_n_clusters": kd_cls,
            "kd_eta_final_p": float(eta_kd_max_p),
            "kd_eta_std_beta": sb_eta_kd,
            # K_eff results
            "ke_boot_p": [float(p) for p in ke_ps],
            "ke_final_p": float(ke_max_p),
            "ke_ols_std_beta": sb_ke,
            "ke_ols_t": float(t_vals_ke[1]),
            "ke_ols_beta": float(beta_ke[1]),
            "ke_n_clusters": ke_cls,
            "ke_eta_final_p": float(eta_ke_max_p),
            "ke_eta_std_beta": sb_eta_ke,
            "skipped": False,
        }

    # FDR correction
    vk = [k for k in cond_keys if not results.get(k, {}).get("skipped", True)]
    if vk:
        kd_fps = [results[k]["kd_final_p"] for k in vk]
        ke_fps = [results[k]["ke_final_p"] for k in vk]
        _, kd_fdr, _, _ = multipletests(kd_fps, alpha=FDR_ALPHA, method="fdr_bh")
        _, ke_fdr, _, _ = multipletests(ke_fps, alpha=FDR_ALPHA, method="fdr_bh")
        for i, k in enumerate(vk):
            results[k]["kd_fdr_p"] = float(kd_fdr[i])
            results[k]["ke_fdr_p"] = float(ke_fdr[i])

    kd_sig = sum(1 for k in vk if results[k]["kd_fdr_p"] < FDR_ALPHA)
    ke_sig = sum(1 for k in vk if results[k]["ke_fdr_p"] < FDR_ALPHA)
    kd_sig_neg = sum(1 for k in vk if results[k]["kd_fdr_p"] < FDR_ALPHA
                     and results[k]["kd_ols_beta"] < 0)
    ke_sig_neg = sum(1 for k in vk if results[k]["ke_fdr_p"] < FDR_ALPHA
                     and results[k]["ke_ols_beta"] < 0)

    mean_rho = np.mean([results[k]["rank_corr_kd_keff"] for k in vk])

    print(f"\n  SUMMARY K={K}:")
    print(f"    κ_div FDR sig: {kd_sig}/12 ({kd_sig_neg} negative)")
    print(f"    K_eff FDR sig: {ke_sig}/12 ({ke_sig_neg} negative)")
    print(f"    Mean ρ(κ_div, K_eff): {mean_rho:.4f}")

    return {
        "K": K,
        "n_panels": n_combos,
        "per_condition": {k: results[k] for k in vk},
        "kd_fdr_sig": kd_sig,
        "ke_fdr_sig": ke_sig,
        "kd_sig_negative": kd_sig_neg,
        "ke_sig_negative": ke_sig_neg,
        "mean_rank_corr": float(mean_rho),
    }


def main():
    t_start = time.time()
    print("Loading data...")
    clean, attacked = load_all_scores(
        checkpoint_files=LOCAL_CKPTS, individual_dir=LOCAL_SCORES_DIR)
    pairs = load_pairs(checkpoint_files=LOCAL_CKPTS)
    mi_data = load_mi_data(mi_path=LOCAL_MI_PATH)

    all_res = {}
    for K in [3, 5, 7]:
        all_res[f"K{K}"] = run_for_K(K, clean, attacked, pairs, mi_data)

    output = {
        "analysis": "kappa_diversity_vs_keff_regression",
        "purpose": "R6 Batch 1A: test whether predictor shift pattern is unique to K_eff",
        "method": "wild_cluster_bootstrap",
        "B": B_BOOT,
        "weights": "rademacher",
        "clustering": "max_p_across_K_positions",
        "fdr_alpha": FDR_ALPHA,
        "seed": SEED,
        "n_models": N_MODELS,
        "models": ALL_MODELS,
    }

    for K in [3, 5, 7]:
        kk = f"K{K}"
        r = all_res[kk]
        output[kk] = {
            "n_panels": r["n_panels"],
            "kappa_div_fdr_sig": f"{r['kd_fdr_sig']}/12",
            "kappa_div_sig_negative": r["kd_sig_negative"],
            "keff_fdr_sig": f"{r['ke_fdr_sig']}/12",
            "keff_sig_negative": r["ke_sig_negative"],
            "mean_rank_corr_kd_keff": r["mean_rank_corr"],
            "per_condition": r["per_condition"],
        }

    output["predictor_shift_comparison"] = {
        "description": "Does significance increase from K=3 to K=5/7 for both predictors?",
        "kappa_diversity": {
            "K3": all_res["K3"]["kd_fdr_sig"],
            "K5": all_res["K5"]["kd_fdr_sig"],
            "K7": all_res["K7"]["kd_fdr_sig"],
        },
        "keff": {
            "K3": all_res["K3"]["ke_fdr_sig"],
            "K5": all_res["K5"]["ke_fdr_sig"],
            "K7": all_res["K7"]["ke_fdr_sig"],
        },
    }

    out_path = PROJECT_ROOT / "artifacts" / "results" / "plan001" / "kappa_diversity_regression.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"FINAL COMPARISON (total: {total_time:.0f}s)")
    print(f"{'=' * 60}")
    print(f"  {'K':<4} {'κ_div sig':>10} {'K_eff sig':>10} {'Mean ρ':>8}")
    for K in [3, 5, 7]:
        kk = f"K{K}"
        r = all_res[kk]
        print(f"  {K:<4} {r['kd_fdr_sig']:>7}/12  {r['ke_fdr_sig']:>7}/12  {r['mean_rank_corr']:.4f}")


if __name__ == "__main__":
    main()
