#!/usr/bin/env python3
"""A2: Quantization dummy regression — test AWQ INT4 fraction as confound."""

import os
os.environ["HF_HOME"] = "/root/autodl-tmp/.hf_cache"

import sys
sys.path.insert(0, "/root/cert_manip_resist_eval")

import json
import time
import logging
import itertools
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils import setup_logging
from src.unified_data_loader import (
    ALL_MODELS, LOCAL_MODELS, DATASETS, ATTACKS,
    load_all_scores, load_pairs, load_mi_data,
    _is_valid_score,
)

logger = setup_logging("quant_dummy_reg")
fh = logging.FileHandler("/tmp/quant_dummy_regression.log", mode="w")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(fh)

EPS = 1e-6
B_BOOT = 499
SEED = 42
K_VALUES = [3, 5, 7]
N_LOCAL = len(LOCAL_MODELS)
OUT_PATH = Path("/root/cert_manip_resist_eval/artifacts/results/plan001/quant_dummy_regression.json")


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


def compute_panel_asr(jids, clean, attacked, pairs):
    n = min(len(pairs), *(len(clean[j]) for j in jids),
            *(len(attacked[j]) for j in jids))
    n_flips = n_correct = 0
    for i in range(n):
        if not all(_is_valid_score(clean[j][i]) for j in jids):
            continue
        if not all(_is_valid_score(attacked[j][i]) for j in jids):
            continue
        gt = pairs[i]["ground_truth_winner"]
        cv = majority_vote([clean[j][i]["winner"] for j in jids])
        if cv != gt:
            continue
        n_correct += 1
        av = majority_vote([attacked[j][i]["winner"] for j in jids])
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


def wild_cluster_bootstrap(y, X_full, cluster_ids, B=B_BOOT, seed=SEED,
                           test_col_idx=1):
    n, p = X_full.shape
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    panel_cluster_idx = np.array([cluster_to_idx[c] for c in cluster_ids])

    fit_full = sm.OLS(y, X_full).fit()
    t_original = fit_full.tvalues[test_col_idx]

    cols_restricted = [i for i in range(p) if i != test_col_idx]
    X_restricted = X_full[:, cols_restricted]
    fit_restricted = sm.OLS(y, X_restricted).fit()
    y_hat_r = fit_restricted.fittedvalues
    e_r = y - y_hat_r

    rng = np.random.RandomState(seed)
    t_boot = np.empty(B)
    for b in range(B):
        w_cluster = rng.choice([-1, 1], size=n_clusters)
        w_panel = w_cluster[panel_cluster_idx]
        y_boot = y_hat_r + w_panel * e_r
        try:
            fit_b = sm.OLS(y_boot, X_full).fit()
            t_boot[b] = fit_b.tvalues[test_col_idx]
        except Exception:
            t_boot[b] = 0.0

    p_value = (1 + np.sum(np.abs(t_boot) >= np.abs(t_original))) / (1 + B)
    return float(p_value)


def bootstrap_K_positions(y, X_full, panel_jids_list, B=B_BOOT, seed=SEED,
                          test_col_idx=1):
    K = len(panel_jids_list[0])
    position_ps = []
    for k in range(K):
        cluster_ids = np.array([jids[k] for jids in panel_jids_list])
        p_val = wild_cluster_bootstrap(
            y, X_full, cluster_ids, B=B, seed=seed, test_col_idx=test_col_idx
        )
        position_ps.append(p_val)
    return position_ps, max(position_ps)


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("A2: Quantization Dummy Regression")
    logger.info("Model: log(ASR+eps) ~ log(K_eff) + log(eta_max+eps) + quant_frac + log(K_eff)*quant_frac")
    logger.info("=" * 60)

    mi_data = load_mi_data()
    pairs = load_pairs()
    clean, attacked = load_all_scores()

    n_models = len(ALL_MODELS)
    logger.info(f"Models: {n_models} total, {N_LOCAL} AWQ INT4: {LOCAL_MODELS}")

    ind_asr = {}
    for ds in DATASETS:
        for atk in ATTACKS:
            for mid in ALL_MODELS:
                if mid in clean[ds] and mid in attacked[ds][atk]:
                    ind_asr[(ds, atk, mid)] = compute_individual_asr(
                        clean[ds][mid], attacked[ds][atk][mid], pairs[ds]
                    )
    logger.info(f"Individual ASR: {len(ind_asr)} values computed")

    all_results = {}

    for K in K_VALUES:
        t_k = time.time()
        logger.info(f"\n{'='*60}\nK = {K}\n{'='*60}")

        all_combos = list(itertools.combinations(range(n_models), K))
        n_total = len(all_combos)
        logger.info(f"C({n_models},{K}) = {n_total} panels")

        k_results = {"K": K, "n_total_panels": n_total, "conditions": {}}
        sig_counts = {"keff": 0, "interaction": 0, "quant_frac": 0, "eta": 0}
        valid_count = 0
        keff_std_betas = []
        interaction_dirs = []

        for ds in DATASETS:
            mi_mat = np.array(mi_data["mi_matrix"][ds])

            combo_keff = {}
            for combo in all_combos:
                combo_keff[combo] = mi_based_keff(mi_mat, list(combo))

            for atk in ATTACKS:
                cond = f"{ds}x{atk}"
                logger.info(f"\n--- {cond} (K={K}) ---")

                p_asrs, p_keffs, p_etas, p_qfracs = [], [], [], []
                p_jids_list = []
                skipped = 0

                for combo in all_combos:
                    jids = tuple(ALL_MODELS[i] for i in combo)
                    if not all((ds, atk, j) in ind_asr for j in jids):
                        skipped += 1
                        continue

                    keff = combo_keff[combo]
                    eta_max = max(ind_asr[(ds, atk, j)] for j in jids)
                    n_local = sum(1 for i in combo if i < N_LOCAL)
                    qfrac = n_local / K

                    cs = {j: clean[ds][j] for j in jids}
                    ats = {j: attacked[ds][atk][j] for j in jids}
                    asr = compute_panel_asr(jids, cs, ats, pairs[ds])

                    p_asrs.append(asr)
                    p_keffs.append(keff)
                    p_etas.append(eta_max)
                    p_qfracs.append(qfrac)
                    p_jids_list.append(jids)

                n_panels = len(p_asrs)
                logger.info(f"  {n_panels} panels, {skipped} skipped")

                if n_panels < 20:
                    k_results["conditions"][cond] = {"n_panels": n_panels, "skipped": True}
                    continue

                y = np.log(np.array(p_asrs) + EPS)
                log_keff = np.log(np.array(p_keffs))
                log_eta = np.log(np.array(p_etas) + EPS)
                qfrac_arr = np.array(p_qfracs)

                if np.std(y) < 1e-10:
                    k_results["conditions"][cond] = {
                        "n_panels": n_panels, "skipped": True, "reason": "constant_asr"
                    }
                    continue

                interaction = log_keff * qfrac_arr
                # [const, log_keff, log_eta, quant_frac, log_keff*quant_frac]
                X = np.column_stack([np.ones(n_panels), log_keff, log_eta,
                                     qfrac_arr, interaction])

                skip_cond = False
                for col_idx in range(1, 5):
                    if np.std(X[:, col_idx]) < 1e-12:
                        logger.warning(f"  Zero variance in col {col_idx}, skipping")
                        k_results["conditions"][cond] = {
                            "n_panels": n_panels, "skipped": True,
                            "reason": "zero_variance_predictor"
                        }
                        skip_cond = True
                        break
                if skip_cond:
                    continue

                fit = sm.OLS(y, X).fit()
                sd_y = np.std(y)

                names = ["keff", "eta_max", "quant_frac", "interaction"]
                std_betas = {}
                ols_res = {
                    "R2": float(fit.rsquared),
                    "R2_adj": float(fit.rsquared_adj),
                    "n": n_panels,
                }
                for idx, name in enumerate(names, 1):
                    ols_res[f"{name}_beta"] = float(fit.params[idx])
                    ols_res[f"{name}_se"] = float(fit.bse[idx])
                    ols_res[f"{name}_t"] = float(fit.tvalues[idx])
                    ols_res[f"{name}_p"] = float(fit.pvalues[idx])
                    sb = float(fit.params[idx] * np.std(X[:, idx]) / sd_y)
                    std_betas[name] = sb
                    ols_res[f"{name}_std_beta"] = sb

                X_no_c = X[:, 1:]
                vif = {}
                for idx, name in enumerate(names):
                    try:
                        vif[name] = float(variance_inflation_factor(X_no_c, idx))
                    except Exception:
                        vif[name] = None

                boot_res = {}
                for test_idx, name in [(1, "keff"), (3, "quant_frac"),
                                       (4, "interaction")]:
                    pos_ps, final_p = bootstrap_K_positions(
                        y, X, p_jids_list, B=B_BOOT, seed=SEED,
                        test_col_idx=test_idx
                    )
                    boot_res[name] = {
                        "position_ps": pos_ps,
                        "final_p": final_p,
                    }

                cond_res = {
                    "n_panels": n_panels,
                    "skipped": False,
                    "ols": ols_res,
                    "vif": vif,
                    "std_betas": std_betas,
                    "bootstrap": boot_res,
                    "quant_frac_stats": {
                        "mean": float(np.mean(qfrac_arr)),
                        "std": float(np.std(qfrac_arr)),
                        "unique": sorted(set(float(x) for x in qfrac_arr)),
                    },
                }
                k_results["conditions"][cond] = cond_res

                valid_count += 1
                bp_keff = boot_res["keff"]["final_p"]
                bp_int = boot_res["interaction"]["final_p"]
                bp_qf = boot_res["quant_frac"]["final_p"]

                if bp_keff < 0.05:
                    sig_counts["keff"] += 1
                if bp_int < 0.05:
                    sig_counts["interaction"] += 1
                if bp_qf < 0.05:
                    sig_counts["quant_frac"] += 1
                if ols_res["eta_max_p"] < 0.05:
                    sig_counts["eta"] += 1

                keff_std_betas.append(std_betas["keff"])
                interaction_dirs.append(
                    "positive" if ols_res["interaction_beta"] > 0 else "negative"
                )

                logger.info(
                    f"  R2={ols_res['R2']:.4f} | "
                    f"keff: std_b={std_betas['keff']:+.4f} boot_p={bp_keff:.4f} | "
                    f"qfrac: std_b={std_betas['quant_frac']:+.4f} boot_p={bp_qf:.4f} | "
                    f"int: b={ols_res['interaction_beta']:+.4f} boot_p={bp_int:.4f}"
                )
                logger.info(
                    f"  VIF: keff={vif['keff']:.1f} eta={vif['eta_max']:.1f} "
                    f"qfrac={vif['quant_frac']:.1f} int={vif['interaction']:.1f}"
                )

        n_pos = sum(1 for d in interaction_dirs if d == "positive")
        n_neg = sum(1 for d in interaction_dirs if d == "negative")
        if n_pos > n_neg * 2:
            int_dir = "positive"
        elif n_neg > n_pos * 2:
            int_dir = "negative"
        else:
            int_dir = "mixed"

        k_results["summary"] = {
            "keff_sig_count": f"{sig_counts['keff']}/{valid_count}",
            "keff_beta_std_mean": float(np.mean(keff_std_betas)) if keff_std_betas else None,
            "interaction_sig_count": f"{sig_counts['interaction']}/{valid_count}",
            "interaction_direction": int_dir,
            "quant_frac_sig_count": f"{sig_counts['quant_frac']}/{valid_count}",
            "eta_sig_count": f"{sig_counts['eta']}/{valid_count}",
        }
        all_results[f"K{K}"] = k_results

        logger.info(f"\nK={K} Summary (elapsed {time.time()-t_k:.1f}s):")
        for k, v in k_results["summary"].items():
            logger.info(f"  {k}: {v}")

    # Predictor shift
    keff_sigs = {}
    for key in ["K3", "K5", "K7"]:
        s = all_results[key]["summary"]["keff_sig_count"]
        keff_sigs[key] = int(s.split("/")[0])

    predictor_shift = (keff_sigs["K3"] <= keff_sigs["K5"] <= keff_sigs["K7"]
                       and keff_sigs["K7"] > keff_sigs["K3"])

    # Interaction interpretation
    int_sigs = [
        int(all_results[k]["summary"]["interaction_sig_count"].split("/")[0])
        for k in ["K3", "K5", "K7"]
    ]
    if max(int_sigs) <= 1:
        int_interp = (
            "Interaction K_eff x quant_frac rarely significant; "
            "quantization composition does not modulate K_eff effect. "
            "K_eff findings are robust to AWQ INT4 mixing."
        )
    elif all(s >= 6 for s in int_sigs):
        int_interp = (
            "Interaction frequently significant; K_eff effect depends on "
            "quantization fraction. Results should be interpreted with caution."
        )
    else:
        int_interp = (
            f"Interaction significant in {int_sigs} conditions (K=3/5/7). "
            "Mixed evidence for quant_frac modulating K_eff effect."
        )

    output = {
        "model": "log(ASR+eps) ~ log(K_eff) + log(eta_max+eps) + quant_frac + log(K_eff)*quant_frac",
        "epsilon": EPS,
        "bootstrap_B": B_BOOT,
        "bootstrap_method": "wild_cluster_bootstrap_max_over_K_positions",
        "n_conditions": 12,
        "n_models": 15,
        "n_local_awq": N_LOCAL,
        "local_models": LOCAL_MODELS,
        "results_by_K": {k: v["summary"] for k, v in all_results.items()},
        "predictor_shift_survives": predictor_shift,
        "interaction_interpretation": int_interp,
        "conditions_detail": {k: v["conditions"] for k, v in all_results.items()},
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=_json_default, ensure_ascii=False)

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.1f}s. Saved: {OUT_PATH}")
    print(f"\nDone in {elapsed:.1f}s. Saved: {OUT_PATH}")
    print(f"\nResults by K:")
    for k in ["K3", "K5", "K7"]:
        s = output["results_by_K"][k]
        print(f"  {k}: keff_sig={s['keff_sig_count']} int_sig={s['interaction_sig_count']} "
              f"dir={s['interaction_direction']} keff_std_mean={s['keff_beta_std_mean']:.4f}")
    print(f"Predictor shift survives: {output['predictor_shift_survives']}")
    print(f"Interpretation: {output['interaction_interpretation']}")


if __name__ == "__main__":
    main()
