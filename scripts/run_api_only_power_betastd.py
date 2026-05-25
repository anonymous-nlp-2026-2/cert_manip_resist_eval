#!/usr/bin/env python3
"""API-only power analysis + β_std comparison (full pool vs API-only)."""

import json
import sys
import time
import itertools
import logging
import numpy as np
import statsmodels.api as sm
from pathlib import Path

sys.path.insert(0, "/root/cert_manip_resist_eval")

MI_MODELS = [
    "qwen2.5-72b", "llama3.1-70b", "mistral-large",
    "qwen2.5-32b", "qwen2.5-14b", "llama3.1-8b",
    "claude-opus-4-6", "gpt-5.5", "gemini-3.1-pro-preview", "gpt-4.1",
    "claude-sonnet-4-6", "gpt-4.1-nano", "gemini-3.5-flash",
    "gpt-4.1-mini", "claude-3-haiku",
]
LOCAL_MODELS = MI_MODELS[:6]
POOL_A = MI_MODELS[6:]  # 9 API models
POOL_A_INDICES = list(range(6, 15))
ALL_INDICES = list(range(15))

DATASETS = ["mmlu", "arc_challenge"]
ATTACKS = [
    "prompt_injection", "sycophancy", "score_manipulation",
    "verbosity_bias", "position_bias", "authority_bias",
]
EPS = 1e-6

BASE_DIR = Path("/root/cert_manip_resist_eval")
SCORES_DIR = BASE_DIR / "artifacts/results/plan001/individual_scores"
MI_PATH = BASE_DIR / "artifacts/results/plan001/mi_matrix/mi_matrix_15models.json"
CKPT_PATHS = [
    BASE_DIR / "artifacts/results/_taxonomy_checkpoint.json",
    BASE_DIR / "artifacts/results/_step2v3_checkpoint.json",
]
POOL_A_MDE_PATH = BASE_DIR / "artifacts/results/plan001/pool_a_mde.json"
OUT_PATH = BASE_DIR / "artifacts/results/plan001/api_only_power_betastd.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/api_only_power.log", mode="w"),
    ],
)
logger = logging.getLogger("api_betastd")


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


def _invert_winners(scores):
    swap = {"A": "B", "B": "A", "tie": "tie"}
    return [{**s, "winner": swap.get(s.get("winner", "tie"), "tie")} for s in scores]


def load_mi_matrix():
    with open(MI_PATH) as f:
        data = json.load(f)
    return {ds: np.array(data["mi_matrix"][ds]) for ds in DATASETS}


def load_pairs():
    for ckpt_path in CKPT_PATHS:
        with open(ckpt_path) as f:
            data = json.load(f)
        if "pairs" in data:
            return {ds: data["pairs"][ds][:300] for ds in DATASETS}
    raise FileNotFoundError("No pairs found")


def load_all_scores():
    """Load scores for all 15 models from checkpoints + individual files."""
    clean = {ds: {} for ds in DATASETS}
    attacked = {ds: {a: {} for a in ATTACKS} for ds in DATASETS}

    # 1. Load local models from checkpoints
    for ckpt_path in CKPT_PATHS:
        if not ckpt_path.exists():
            continue
        with open(ckpt_path) as f:
            data = json.load(f)
        for ds in DATASETS:
            for mid, scores in data.get("clean_scores", {}).get(ds, {}).items():
                if mid not in clean[ds]:
                    clean[ds][mid] = scores
            for atk in ATTACKS:
                for mid, scores in data.get("attacked_scores", {}).get(ds, {}).get(atk, {}).items():
                    if mid not in attacked[ds][atk]:
                        if atk == "position_bias":
                            scores = _invert_winners(scores)
                        attacked[ds][atk][mid] = scores

    # 2. Load API models from individual score files
    for mid in MI_MODELS:
        for ds in DATASETS:
            fp = SCORES_DIR / f"{mid}__{ds}__clean.json"
            if fp.exists() and mid not in clean[ds]:
                with open(fp) as f:
                    clean[ds][mid] = json.load(f).get("scores", [])
            for atk in ATTACKS:
                fp = SCORES_DIR / f"{mid}__{ds}__{atk}.json"
                if fp.exists() and mid not in attacked[ds][atk]:
                    with open(fp) as f:
                        scores = json.load(f).get("scores", [])
                        if atk == "position_bias":
                            scores = _invert_winners(scores)
                        attacked[ds][atk][mid] = scores

    # Validate coverage
    for ds in DATASETS:
        loaded_clean = [m for m in MI_MODELS if m in clean[ds]]
        logger.info(f"  {ds} clean: {len(loaded_clean)}/15 models")
        for atk in ATTACKS:
            loaded_atk = [m for m in MI_MODELS if m in attacked[ds][atk]]
            if len(loaded_atk) < 15:
                logger.warning(f"  {ds}/{atk}: {len(loaded_atk)}/15 models")

    return clean, attacked


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


def _is_valid(s):
    if s.get("error") or s.get("format_error"):
        return False
    return s.get("winner") in ("A", "B", "tie")


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


def compute_panel_asr(jids, clean, attacked, pairs):
    n = min(len(pairs), *(len(clean[j]) for j in jids),
            *(len(attacked[j]) for j in jids))
    n_flips = n_correct = 0
    for i in range(n):
        if not all(_is_valid(clean[j][i]) for j in jids):
            continue
        if not all(_is_valid(attacked[j][i]) for j in jids):
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


def compute_individual_asr(clean_list, attacked_list, pairs):
    n = min(len(clean_list), len(attacked_list), len(pairs))
    n_flips = n_correct = 0
    for i in range(n):
        if not _is_valid(clean_list[i]) or not _is_valid(attacked_list[i]):
            continue
        gt = pairs[i]["ground_truth_winner"]
        if clean_list[i]["winner"] != gt:
            continue
        n_correct += 1
        if attacked_list[i]["winner"] != gt:
            n_flips += 1
    return n_flips / max(n_correct, 1)


def build_panel_data(K, mi_matrices, clean, attacked, pairs,
                     model_list, model_indices):
    """Build panel-level regression data for a given K and model subset."""
    combos = list(itertools.combinations(range(len(model_list)), K))
    logger.info(f"  K={K}: {len(combos)} candidate panels from {len(model_list)} models")

    data_by_cond = {}
    for ds in DATASETS:
        mi = mi_matrices[ds]
        for atk in ATTACKS:
            cond = f"{ds}x{atk}"
            log_keffs, log_asrs, log_etas, panel_jids = [], [], [], []

            n_skipped = 0
            for combo in combos:
                jids = [model_list[c] for c in combo]
                midx = [model_indices[c] for c in combo]

                if not all(j in clean[ds] and j in attacked[ds][atk] for j in jids):
                    n_skipped += 1
                    continue

                keff = mi_based_keff(mi, midx)
                asr = compute_panel_asr(jids, clean[ds], attacked[ds][atk], pairs[ds])

                eta = max(
                    compute_individual_asr(
                        clean[ds][j], attacked[ds][atk][j], pairs[ds]
                    )
                    for j in jids
                )

                log_keffs.append(np.log(keff + EPS))
                log_asrs.append(np.log(asr + EPS))
                log_etas.append(np.log(eta + EPS))
                panel_jids.append(jids)

            if n_skipped > 0:
                logger.info(f"    {cond}: {len(log_keffs)} panels (skipped {n_skipped} for missing data)")

            if len(log_keffs) < 10:
                logger.warning(f"    {cond}: only {len(log_keffs)} panels, skipping")
                continue

            data_by_cond[cond] = {
                "log_keff": np.array(log_keffs),
                "log_asr": np.array(log_asrs),
                "log_eta": np.array(log_etas),
                "panel_jids": panel_jids,
                "n_panels": len(log_keffs),
            }

    return data_by_cond


def run_ols_betastd(data_by_cond):
    """Run OLS regression per condition, return β_std for K_eff."""
    results = {}
    for cond, d in data_by_cond.items():
        y = d["log_asr"]
        X = sm.add_constant(np.column_stack([d["log_keff"], d["log_eta"]]))

        try:
            model = sm.OLS(y, X).fit()
        except Exception as e:
            logger.warning(f"  OLS failed for {cond}: {e}")
            continue

        beta_raw = model.params[1]
        sd_keff = np.std(d["log_keff"])
        sd_y = np.std(y)
        beta_std = beta_raw * sd_keff / sd_y if sd_y > 1e-10 else 0.0

        results[cond] = {
            "beta_raw": round(float(beta_raw), 4),
            "beta_std": round(float(beta_std), 4),
            "p_value": float(model.pvalues[1]),
            "R2": round(float(model.rsquared), 4),
            "sd_keff": round(float(sd_keff), 4),
            "sd_y": round(float(sd_y), 4),
            "n_panels": d["n_panels"],
            "eta_beta_std": round(float(
                model.params[2] * np.std(d["log_eta"]) / sd_y
            ), 4) if sd_y > 1e-10 else 0.0,
            "eta_p": float(model.pvalues[2]),
        }
    return results


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("API-only β_std comparison analysis")
    logger.info("=" * 60)

    mi_matrices = load_mi_matrix()
    pairs = load_pairs()

    # Load scores for ALL 15 models (checkpoints + individual files)
    logger.info("Loading all 15 models' scores...")
    clean_all, attacked_all = load_all_scores()

    # --- Full pool K=3 ---
    logger.info("\n--- Full Pool (15 models) K=3 ---")
    fp_k3_data = build_panel_data(3, mi_matrices, clean_all, attacked_all,
                                   pairs, MI_MODELS, ALL_INDICES)
    fp_k3_beta = run_ols_betastd(fp_k3_data)

    # --- Full pool K=5 ---
    logger.info("\n--- Full Pool (15 models) K=5 ---")
    fp_k5_data = build_panel_data(5, mi_matrices, clean_all, attacked_all,
                                   pairs, MI_MODELS, ALL_INDICES)
    fp_k5_beta = run_ols_betastd(fp_k5_data)

    # --- API-only K=3 ---
    # Filter clean/attacked to only API models
    clean_api = {ds: {m: s for m, s in v.items() if m in POOL_A}
                 for ds, v in clean_all.items()}
    attacked_api = {ds: {atk: {m: s for m, s in v.items() if m in POOL_A}
                         for atk, v in attacks.items()}
                    for ds, attacks in attacked_all.items()}

    logger.info("\n--- API-only (9 models) K=3 ---")
    api_k3_data = build_panel_data(3, mi_matrices, clean_api, attacked_api,
                                    pairs, POOL_A, POOL_A_INDICES)
    api_k3_beta = run_ols_betastd(api_k3_data)

    # --- API-only K=5 ---
    logger.info("\n--- API-only (9 models) K=5 ---")
    api_k5_data = build_panel_data(5, mi_matrices, clean_api, attacked_api,
                                    pairs, POOL_A, POOL_A_INDICES)
    api_k5_beta = run_ols_betastd(api_k5_data)

    # Cross-validate: check K=5 full pool β_std against existing data
    logger.info("\nCross-validation against k5_wild_cluster_bootstrap.json:")
    try:
        with open(BASE_DIR / "artifacts/results/plan001/k5_wild_cluster_bootstrap.json") as f:
            k5_wcb = json.load(f)
        for cond_key, cond_data in k5_wcb["per_condition"].items():
            our_cond = cond_key.replace("x", "x")
            if our_cond in fp_k5_beta:
                our = fp_k5_beta[our_cond]["beta_std"]
                ref = cond_data["keff_ols_std_beta"]
                logger.info(f"  {our_cond}: ours={our:.4f}, ref={ref:.4f}, diff={abs(our-ref):.4f}")
    except Exception as e:
        logger.warning(f"  Cross-validation failed: {e}")

    # Build comparison
    conditions = sorted(set(list(fp_k3_beta.keys()) + list(api_k3_beta.keys())
                            + list(fp_k5_beta.keys()) + list(api_k5_beta.keys())))

    def make_comparison(fp_beta, api_beta):
        comp = {}
        for cond in conditions:
            fp_val = fp_beta.get(cond, {}).get("beta_std", None)
            api_val = api_beta.get(cond, {}).get("beta_std", None)
            ratio = None
            if fp_val is not None and api_val is not None and abs(fp_val) > 1e-6:
                ratio = round(api_val / fp_val, 3)
            comp[cond] = {
                "full_pool": fp_val,
                "api_only": api_val,
                "ratio": ratio,
                "full_pool_p": fp_beta.get(cond, {}).get("p_value"),
                "api_only_p": api_beta.get(cond, {}).get("p_value"),
            }
        return comp

    k3_comp = make_comparison(fp_k3_beta, api_k3_beta)
    k5_comp = make_comparison(fp_k5_beta, api_k5_beta)

    def agg_stats(comp):
        fp_vals = [v["full_pool"] for v in comp.values() if v["full_pool"] is not None]
        api_vals = [v["api_only"] for v in comp.values() if v["api_only"] is not None]
        ratios = [v["ratio"] for v in comp.values() if v["ratio"] is not None]
        return {
            "full_pool_mean": round(float(np.mean(fp_vals)), 4) if fp_vals else None,
            "full_pool_median": round(float(np.median(fp_vals)), 4) if fp_vals else None,
            "api_only_mean": round(float(np.mean(api_vals)), 4) if api_vals else None,
            "api_only_median": round(float(np.median(api_vals)), 4) if api_vals else None,
            "ratio_mean": round(float(np.mean(ratios)), 3) if ratios else None,
            "ratio_median": round(float(np.median(ratios)), 3) if ratios else None,
        }

    # Load existing power data
    with open(POOL_A_MDE_PATH) as f:
        pool_a_mde = json.load(f)

    # Determine interpretation
    k3_api_betas = [v.get("beta_std") for v in api_k3_beta.values() if v.get("beta_std") is not None]
    k5_api_betas = [v.get("beta_std") for v in api_k5_beta.values() if v.get("beta_std") is not None]
    k3_fp_betas = [v.get("beta_std") for v in fp_k3_beta.values() if v.get("beta_std") is not None]
    k5_fp_betas = [v.get("beta_std") for v in fp_k5_beta.values() if v.get("beta_std") is not None]

    k3_ratio = np.mean(k3_api_betas) / np.mean(k3_fp_betas) if k3_fp_betas and k3_api_betas else None
    k5_ratio = np.mean(k5_api_betas) / np.mean(k5_fp_betas) if k5_fp_betas and k5_api_betas else None

    if k3_ratio and k5_ratio:
        avg_ratio = (k3_ratio + k5_ratio) / 2
        if avg_ratio < 0.5:
            interpretation = "effect_shrinkage"
        elif avg_ratio > 0.8:
            interpretation = "power_issue"
        else:
            interpretation = "both"
    else:
        interpretation = "insufficient_data"

    def count_sig(beta_dict, alpha=0.05):
        n_sig = sum(1 for v in beta_dict.values() if v.get("p_value", 1) < alpha)
        return f"{n_sig}/{len(beta_dict)}"

    elapsed = time.time() - t0

    output = {
        "api_only_models": len(POOL_A),
        "api_only_model_list": POOL_A,
        "full_pool_models": len(MI_MODELS),
        "panels_per_K": {
            "K3": {"full_pool": f"C(15,3)={len(list(itertools.combinations(range(15),3)))}",
                   "api_only": f"C(9,3)={len(list(itertools.combinations(range(9),3)))}"},
            "K5": {"full_pool": f"C(15,5)={len(list(itertools.combinations(range(15),5)))}",
                   "api_only": f"C(9,5)={len(list(itertools.combinations(range(9),5)))}"},
        },
        "beta_std_comparison": {
            "K3": {
                "per_condition": k3_comp,
                "aggregate": agg_stats(k3_comp),
            },
            "K5": {
                "per_condition": k5_comp,
                "aggregate": agg_stats(k5_comp),
            },
        },
        "significance_counts": {
            "K3": {
                "full_pool_keff_sig": count_sig(fp_k3_beta),
                "api_only_keff_sig": count_sig(api_k3_beta),
            },
            "K5": {
                "full_pool_keff_sig": count_sig(fp_k5_beta),
                "api_only_keff_sig": count_sig(api_k5_beta),
            },
        },
        "full_pool_detail": {
            "K3": fp_k3_beta,
            "K5": fp_k5_beta,
        },
        "api_only_detail": {
            "K3": api_k3_beta,
            "K5": api_k5_beta,
        },
        "mde_at_80pct": pool_a_mde.get("mde_at_80pct", {}),
        "power_curve": pool_a_mde.get("power_by_beta", {}),
        "interpretation": interpretation,
        "interpretation_detail": {
            "k3_beta_ratio": round(float(k3_ratio), 3) if k3_ratio else None,
            "k5_beta_ratio": round(float(k5_ratio), 3) if k5_ratio else None,
            "criteria": "ratio<0.5→effect_shrinkage, ratio>0.8→power_issue, else→both",
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=_json_default, ensure_ascii=False)
    logger.info(f"\nSaved: {OUT_PATH}")
    logger.info(f"Total time: {elapsed:.1f}s")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for k_label in ["K3", "K5"]:
        agg = output["beta_std_comparison"][k_label]["aggregate"]
        logger.info(f"\n{k_label}:")
        logger.info(f"  Full pool mean β_std(K_eff): {agg['full_pool_mean']}")
        logger.info(f"  API-only mean β_std(K_eff):  {agg['api_only_mean']}")
        logger.info(f"  Ratio (api/full):            {agg['ratio_mean']}")
        sig = output["significance_counts"][k_label]
        logger.info(f"  Full pool sig: {sig['full_pool_keff_sig']}")
        logger.info(f"  API-only sig:  {sig['api_only_keff_sig']}")
    logger.info(f"\nInterpretation: {interpretation}")
    logger.info(f"MDE@80%: K3={output['mde_at_80pct'].get('K3')}, K5={output['mde_at_80pct'].get('K5')}")


if __name__ == "__main__":
    main()
