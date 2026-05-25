#!/usr/bin/env python3
"""Gaussian Copula Goodness-of-Fit test (Genest & Rémillard 2008 CvM).

Tests whether the Gaussian copula assumption underlying K_eff is valid
for each pair of judges under clean evaluation conditions.
"""

import json
import itertools
import numpy as np
from pathlib import Path
from scipy.stats import rankdata, norm, kendalltau, multivariate_normal

SCORES_DIR = Path("/root/cert_manip_resist_eval/artifacts/results/plan001/individual_scores")
OUTPUT_DIR = Path("/root/cert_manip_resist_eval/artifacts/results/plan001")
DATASETS = ["mmlu", "arc_challenge"]
B_BOOTSTRAP = 500

def discover_models():
    models = set()
    for f in SCORES_DIR.glob("*__mmlu__clean.json"):
        model_id = f.name.replace("__mmlu__clean.json", "")
        models.add(model_id)
    return sorted(models)

def load_clean_scores(model_id, dataset):
    fp = SCORES_DIR / f"{model_id}__{dataset}__clean.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        data = json.load(f)
    scores = []
    for s in data.get("scores", []):
        if isinstance(s, dict) and not s.get("error") and not s.get("format_error"):
            if s.get("winner") in ("A", "B", "tie"):
                sa = s.get("score_a")
                scores.append(float(sa) if sa is not None else None)
            else:
                scores.append(None)
        else:
            scores.append(None)
    return scores

def pseudo_observations(x):
    return rankdata(x, method='average') / (len(x) + 1)

def fit_rho_kendall(u, v):
    tau, _ = kendalltau(u, v)
    return np.clip(np.sin(np.pi * tau / 2), -0.999, 0.999)

def cvm_stat(u, v, rho):
    n = len(u)
    C_n = np.sum((u[:, None] >= u[None, :]) & (v[:, None] >= v[None, :]), axis=1) / n
    z = np.column_stack([
        norm.ppf(np.clip(u, 1e-10, 1 - 1e-10)),
        norm.ppf(np.clip(v, 1e-10, 1 - 1e-10))
    ])
    rv = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
    C_theta = rv.cdf(z)
    return float(np.sum((C_n - C_theta) ** 2))

def bootstrap_pvalue(rho_hat, n, Sn_obs, B=500):
    cov = [[1.0, rho_hat], [rho_hat, 1.0]]
    count = 0
    for _ in range(B):
        z = np.random.multivariate_normal([0, 0], cov, size=n)
        u_star = pseudo_observations(norm.cdf(z[:, 0]))
        v_star = pseudo_observations(norm.cdf(z[:, 1]))
        rho_star = fit_rho_kendall(u_star, v_star)
        Sn_star = cvm_stat(u_star, v_star, rho_star)
        if Sn_star >= Sn_obs:
            count += 1
    return count / B

def run_gof_for_dataset(models, dataset):
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} | Models: {len(models)}")
    print(f"{'='*60}", flush=True)

    score_vectors = {}
    for mid in models:
        scores = load_clean_scores(mid, dataset)
        if scores is not None:
            score_vectors[mid] = scores

    print(f"Loaded {len(score_vectors)} models", flush=True)

    results = []
    pairs = list(itertools.combinations(sorted(score_vectors.keys()), 2))
    print(f"Testing {len(pairs)} pairs (B={B_BOOTSTRAP})...", flush=True)

    for idx, (m_i, m_j) in enumerate(pairs):
        scores_i = score_vectors[m_i]
        scores_j = score_vectors[m_j]

        n = min(len(scores_i), len(scores_j))
        valid_idx = [k for k in range(n)
                     if scores_i[k] is not None and scores_j[k] is not None]

        if len(valid_idx) < 30:
            results.append({"pair": f"{m_i}|{m_j}", "n_valid": len(valid_idx), "skipped": True})
            continue

        x = np.array([scores_i[k] for k in valid_idx])
        y = np.array([scores_j[k] for k in valid_idx])

        u = pseudo_observations(x)
        v = pseudo_observations(y)
        rho_hat = fit_rho_kendall(u, v)
        Sn = cvm_stat(u, v, rho_hat)
        p_value = bootstrap_pvalue(rho_hat, len(valid_idx), Sn, B=B_BOOTSTRAP)

        results.append({
            "pair": f"{m_i}|{m_j}",
            "model_i": m_i,
            "model_j": m_j,
            "n_valid": len(valid_idx),
            "rho_hat": float(rho_hat),
            "Sn": float(Sn),
            "p_value": float(p_value),
            "reject_005": p_value < 0.05,
        })

        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1}/{len(pairs)}] {m_i}|{m_j} p={p_value:.3f} rho={rho_hat:.3f}", flush=True)

    print(f"  [{len(pairs)}/{len(pairs)}] complete", flush=True)
    return results

def main():
    np.random.seed(42)
    models = discover_models()
    print(f"Discovered {len(models)} models: {models}", flush=True)

    all_results = {}
    summary = {}

    for ds in DATASETS:
        results = run_gof_for_dataset(models, ds)
        all_results[ds] = results

        tested = [r for r in results if not r.get("skipped")]
        rejected = [r for r in tested if r["reject_005"]]
        Sn_vals = [r["Sn"] for r in tested]
        p_vals = [r["p_value"] for r in tested]

        summary[ds] = {
            "n_pairs_tested": len(tested),
            "n_pairs_skipped": len(results) - len(tested),
            "n_rejected_005": len(rejected),
            "rejection_rate_005": len(rejected) / len(tested) if tested else 0,
            "Sn_mean": float(np.mean(Sn_vals)) if Sn_vals else None,
            "Sn_median": float(np.median(Sn_vals)) if Sn_vals else None,
            "Sn_max": float(np.max(Sn_vals)) if Sn_vals else None,
            "Sn_min": float(np.min(Sn_vals)) if Sn_vals else None,
            "p_value_mean": float(np.mean(p_vals)) if p_vals else None,
            "p_value_median": float(np.median(p_vals)) if p_vals else None,
        }

        if rejected:
            judge_reject_count = {}
            for r in rejected:
                for m in [r["model_i"], r["model_j"]]:
                    judge_reject_count[m] = judge_reject_count.get(m, 0) + 1
            summary[ds]["judge_rejection_frequency"] = dict(
                sorted(judge_reject_count.items(), key=lambda x: -x[1])
            )
        else:
            summary[ds]["judge_rejection_frequency"] = {}

        print(f"\n--- {ds} Summary ---")
        print(f"  Tested: {len(tested)} pairs")
        print(f"  Rejected (a=0.05): {len(rejected)} ({summary[ds]['rejection_rate_005']:.1%})")
        print(f"  Sn: mean={summary[ds]['Sn_mean']:.4f}, median={summary[ds]['Sn_median']:.4f}, max={summary[ds]['Sn_max']:.4f}")

    output = {
        "method": "Genest-Remillard_2008_CvM",
        "description": "Gaussian copula goodness-of-fit test using Cramer-von Mises statistic with parametric bootstrap",
        "alpha": 0.05,
        "B_bootstrap": B_BOOTSTRAP,
        "n_models": len(models),
        "models": models,
        "variable": "score_a (1-5 Likert scale from judge)",
        "summary": summary,
        "pair_results": all_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "gaussian_copula_gof.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
