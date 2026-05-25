#!/usr/bin/env python3
"""Cross-Benchmark Bootstrap: Bidirectional (MMLU↔ARC) sign consistency.

Extends V2 analysis with reverse direction: ARC→MMLU cross_predict_R2.
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

from src.utils import setup_logging
from src.unified_data_loader import (
    ALL_MODELS, DATASETS, ATTACKS,
    load_all_scores, load_pairs, load_mi_data,
    _is_valid_score,
)

logger = setup_logging("cross_benchmark_bidir")

EPS = 1e-6
B_BOOT = 9999
SEED = 42
K_VALUES = [3, 5, 7]


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


def get_panel_keff(combo_indices, mi_matrix_np, keff_precomputed, ds):
    jids = tuple(ALL_MODELS[i] for i in combo_indices)
    pk = "|".join(jids)
    if keff_precomputed and ds in keff_precomputed and pk in keff_precomputed[ds]:
        return keff_precomputed[ds][pk]["keff"]
    return mi_based_keff(mi_matrix_np, list(combo_indices))


def build_panel_data(K, combos, atk, ds, clean, attacked, pairs, ind_asr,
                     mi_matrix_np, keff_precomputed):
    panel_asrs = []
    panel_keffs = []
    panel_etas = []

    ds_clean = clean.get(ds, {})
    ds_attacked = attacked.get(ds, {}).get(atk, {})
    ds_pairs = pairs.get(ds, [])

    if not ds_clean or not ds_attacked or not ds_pairs:
        return None, None, None

    for combo in combos:
        jids = tuple(ALL_MODELS[i] for i in combo)
        if not all(j in ds_clean and j in ds_attacked for j in jids):
            continue
        if not all((ds, atk, j) in ind_asr for j in jids):
            continue

        keff = get_panel_keff(combo, mi_matrix_np, keff_precomputed, ds)
        eta_max = max(ind_asr[(ds, atk, j)] for j in jids)

        cs = {j: ds_clean[j] for j in jids}
        ats = {j: ds_attacked[j] for j in jids}
        asr = compute_panel_asr(jids, cs, ats, ds_pairs)

        panel_asrs.append(asr)
        panel_keffs.append(keff)
        panel_etas.append(eta_max)

    if len(panel_asrs) < 10:
        return None, None, None

    return np.array(panel_asrs), np.array(panel_keffs), np.array(panel_etas)


def fit_ols(asrs, keffs, etas):
    y = np.log(asrs + EPS)
    x_keff = np.log(keffs)
    x_eta = np.log(etas + EPS)
    X = np.column_stack([np.ones(len(y)), x_keff, x_eta])
    try:
        fit = sm.OLS(y, X).fit()
        return fit.params[1], fit.params[2], fit.rsquared, fit.pvalues[1], fit.pvalues[2]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def _cross_predict_r2_one_dir(src_asrs, src_keffs, src_etas, tgt_asrs, tgt_keffs, tgt_etas):
    """Fit OLS on source, predict target, return R2."""
    y_src = np.log(src_asrs + EPS)
    X_src = np.column_stack([np.ones(len(y_src)), np.log(src_keffs), np.log(src_etas + EPS)])
    try:
        fit = sm.OLS(y_src, X_src).fit()
    except Exception:
        return np.nan

    y_tgt = np.log(tgt_asrs + EPS)
    X_tgt = np.column_stack([np.ones(len(y_tgt)), np.log(tgt_keffs), np.log(tgt_etas + EPS)])
    y_pred = X_tgt @ fit.params
    ss_res = np.sum((y_tgt - y_pred) ** 2)
    ss_tot = np.sum((y_tgt - np.mean(y_tgt)) ** 2)
    if ss_tot < 1e-12:
        return np.nan
    return 1 - ss_res / ss_tot


def bootstrap_bidirectional(mmlu_asrs, mmlu_keffs, mmlu_etas,
                             arc_asrs, arc_keffs, arc_etas,
                             B=B_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n_mmlu = len(mmlu_asrs)
    n_arc = len(arc_asrs)

    boot_mmlu_bk = np.empty(B)
    boot_arc_bk = np.empty(B)
    boot_fwd_r2 = np.empty(B)  # MMLU→ARC
    boot_rev_r2 = np.empty(B)  # ARC→MMLU

    for b in range(B):
        idx_m = rng.randint(0, n_mmlu, size=n_mmlu)
        idx_a = rng.randint(0, n_arc, size=n_arc)

        bk_m, _, _, _, _ = fit_ols(mmlu_asrs[idx_m], mmlu_keffs[idx_m], mmlu_etas[idx_m])
        bk_a, _, _, _, _ = fit_ols(arc_asrs[idx_a], arc_keffs[idx_a], arc_etas[idx_a])

        boot_mmlu_bk[b] = bk_m
        boot_arc_bk[b] = bk_a

        # MMLU→ARC
        boot_fwd_r2[b] = _cross_predict_r2_one_dir(
            mmlu_asrs[idx_m], mmlu_keffs[idx_m], mmlu_etas[idx_m],
            arc_asrs[idx_a], arc_keffs[idx_a], arc_etas[idx_a]
        )
        # ARC→MMLU
        boot_rev_r2[b] = _cross_predict_r2_one_dir(
            arc_asrs[idx_a], arc_keffs[idx_a], arc_etas[idx_a],
            mmlu_asrs[idx_m], mmlu_keffs[idx_m], mmlu_etas[idx_m]
        )

    same_sign = np.sign(boot_mmlu_bk) == np.sign(boot_arc_bk)
    valid = ~(np.isnan(boot_mmlu_bk) | np.isnan(boot_arc_bk))
    sign_rate = float(np.mean(same_sign[valid])) if np.any(valid) else np.nan

    def _ci(arr):
        v = arr[~np.isnan(arr)]
        if len(v) == 0:
            return None, None
        return float(np.nanmean(v)), [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    fwd_mean, fwd_ci = _ci(boot_fwd_r2)
    rev_mean, rev_ci = _ci(boot_rev_r2)

    return {
        "sign_consistency_rate": sign_rate,
        "mmlu_beta_keff_mean": float(np.nanmean(boot_mmlu_bk)),
        "mmlu_beta_keff_ci": [float(np.nanpercentile(boot_mmlu_bk, 2.5)),
                               float(np.nanpercentile(boot_mmlu_bk, 97.5))],
        "arc_beta_keff_mean": float(np.nanmean(boot_arc_bk)),
        "arc_beta_keff_ci": [float(np.nanpercentile(boot_arc_bk, 2.5)),
                              float(np.nanpercentile(boot_arc_bk, 97.5))],
        "fwd_cross_r2_mean": fwd_mean,
        "fwd_cross_r2_ci": fwd_ci,
        "rev_cross_r2_mean": rev_mean,
        "rev_cross_r2_ci": rev_ci,
        "n_valid_boots": int(np.sum(valid)),
    }


def main():
    logger.info("=" * 70)
    logger.info("Cross-Benchmark Bootstrap: Bidirectional MMLU <-> ARC (V2)")
    logger.info("=" * 70)

    mi_data = load_mi_data()
    mi_matrices_raw = mi_data["mi_matrix"]
    keff_precomputed = mi_data.get("keff_per_panel", {})
    pairs = load_pairs()
    clean, attacked = load_all_scores()

    mi_matrices_np = {}
    for ds in DATASETS:
        mi_matrices_np[ds] = np.array(mi_matrices_raw[ds])

    logger.info("Computing individual ASR...")
    ind_asr = {}
    for ds in DATASETS:
        for atk in ATTACKS:
            for mid in ALL_MODELS:
                if mid in clean[ds] and mid in attacked[ds][atk]:
                    ind_asr[(ds, atk, mid)] = compute_individual_asr(
                        clean[ds][mid], attacked[ds][atk][mid], pairs[ds]
                    )
    logger.info(f"  Computed {len(ind_asr)} individual ASR values")

    results = {}
    sign_count = 0
    total = 0

    for K in K_VALUES:
        combos = list(itertools.combinations(range(len(ALL_MODELS)), K))
        logger.info(f"\nK={K}: C(15,{K}) = {len(combos)} panels")

        for atk in ATTACKS:
            cond_key = f"K{K}x{atk}"
            logger.info(f"\n  [{cond_key}]")
            t0 = time.time()

            mmlu_asrs, mmlu_keffs, mmlu_etas = build_panel_data(
                K, combos, atk, "mmlu", clean, attacked, pairs, ind_asr,
                mi_matrices_np["mmlu"], keff_precomputed
            )
            arc_asrs, arc_keffs, arc_etas = build_panel_data(
                K, combos, atk, "arc_challenge", clean, attacked, pairs, ind_asr,
                mi_matrices_np["arc_challenge"], keff_precomputed
            )

            if mmlu_asrs is None or arc_asrs is None:
                logger.info(f"    SKIP: insufficient data")
                results[cond_key] = {"skipped": True}
                continue

            # Point estimates
            mmlu_bk, mmlu_be, mmlu_r2, mmlu_pk, mmlu_pe = fit_ols(mmlu_asrs, mmlu_keffs, mmlu_etas)
            arc_bk, arc_be, arc_r2, arc_pk, arc_pe = fit_ols(arc_asrs, arc_keffs, arc_etas)

            fwd_r2 = _cross_predict_r2_one_dir(mmlu_asrs, mmlu_keffs, mmlu_etas,
                                                 arc_asrs, arc_keffs, arc_etas)
            rev_r2 = _cross_predict_r2_one_dir(arc_asrs, arc_keffs, arc_etas,
                                                 mmlu_asrs, mmlu_keffs, mmlu_etas)

            sc = bool(np.sign(mmlu_bk) == np.sign(arc_bk))
            total += 1
            if sc:
                sign_count += 1

            logger.info(f"    MMLU: β_keff={mmlu_bk:+.4f} (p={mmlu_pk:.2e}), R²={mmlu_r2:.4f}")
            logger.info(f"    ARC:  β_keff={arc_bk:+.4f} (p={arc_pk:.2e}), R²={arc_r2:.4f}")
            logger.info(f"    MMLU→ARC R²={fwd_r2:.4f}, ARC→MMLU R²={rev_r2:.4f}, sign={sc}")

            boot = bootstrap_bidirectional(
                mmlu_asrs, mmlu_keffs, mmlu_etas,
                arc_asrs, arc_keffs, arc_etas,
                B=B_BOOT, seed=SEED
            )

            elapsed = time.time() - t0
            logger.info(f"    Bootstrap: sign_rate={boot['sign_consistency_rate']:.3f}, "
                        f"fwd_R²={boot['fwd_cross_r2_mean']:.4f}, "
                        f"rev_R²={boot['rev_cross_r2_mean']:.4f} ({elapsed:.1f}s)")

            results[cond_key] = {
                "K": K,
                "attack": atk,
                "n_mmlu_panels": int(len(mmlu_asrs)),
                "n_arc_panels": int(len(arc_asrs)),
                "point_estimates": {
                    "mmlu": {
                        "beta_keff": mmlu_bk, "beta_eta": mmlu_be,
                        "p_keff": mmlu_pk, "p_eta": mmlu_pe, "R2": mmlu_r2,
                    },
                    "arc": {
                        "beta_keff": arc_bk, "beta_eta": arc_be,
                        "p_keff": arc_pk, "p_eta": arc_pe, "R2": arc_r2,
                    },
                    "sign_consistent": sc,
                    "mmlu_to_arc_R2": fwd_r2,
                    "arc_to_mmlu_R2": rev_r2,
                },
                "bootstrap": boot,
            }

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"SUMMARY: Sign Consistency = {sign_count}/{total}")
    logger.info(f"{'='*70}")

    per_k = {}
    for K in K_VALUES:
        k_conds = [k for k in results if k.startswith(f"K{K}x") and not results[k].get("skipped")]
        k_sc = sum(1 for k in k_conds if results[k]["point_estimates"]["sign_consistent"])
        per_k[f"K{K}"] = f"{k_sc}/{len(k_conds)}"
        logger.info(f"  K={K}: {k_sc}/{len(k_conds)} sign consistent")

    inconsistent = []
    for cond_key in sorted(results.keys()):
        r = results[cond_key]
        if r.get("skipped"):
            continue
        pe = r["point_estimates"]
        s = "Y" if pe["sign_consistent"] else "X"
        logger.info(
            f"  {cond_key:<30} MMLU β={pe['mmlu']['beta_keff']:+.4f}  "
            f"ARC β={pe['arc']['beta_keff']:+.4f}  sign={s}  "
            f"fwd_R²={pe['mmlu_to_arc_R2']:.4f}  rev_R²={pe['arc_to_mmlu_R2']:.4f}"
        )
        if not pe["sign_consistent"]:
            inconsistent.append(cond_key)

    output = {
        "analysis": "cross_benchmark_bidirectional_v2",
        "data_version": "V2 (_is_valid_score filtering)",
        "n_models": len(ALL_MODELS),
        "models": ALL_MODELS,
        "K_values": K_VALUES,
        "attacks": ATTACKS,
        "B_bootstrap": B_BOOT,
        "seed": SEED,
        "summary": {
            "sign_consistent": sign_count,
            "total_conditions": total,
            "sign_consistency_str": f"{sign_count}/{total}",
            "per_K": per_k,
            "inconsistent_conditions": inconsistent,
        },
        "conditions": results,
    }

    out_dir = Path("/root/cert_manip_resist_eval/artifacts/results/plan001")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cross_benchmark_v2_bidirectional.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default, ensure_ascii=False)
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
