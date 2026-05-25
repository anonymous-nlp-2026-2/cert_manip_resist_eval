#!/usr/bin/env python3
"""Gaussian Copula GoF with jittering — fast version.

Adds U(-0.5, 0.5) noise to break ties, R=5 repetitions, B=100 bootstrap.
Tests all 105 pairs × 2 datasets.
"""

import json
import itertools
import numpy as np
from pathlib import Path
from scipy.stats import rankdata, norm, kendalltau, multivariate_normal

SCORES_DIR = Path("/root/cert_manip_resist_eval/artifacts/results/plan001/individual_scores")
OUTPUT_DIR = Path("/root/cert_manip_resist_eval/artifacts/results/plan001")
DATASETS = ["mmlu", "arc_challenge"]
B_BOOTSTRAP = 100
R_JITTER = 5

def discover_models():
    models = set()
    for f in SCORES_DIR.glob("*__mmlu__clean.json"):
        models.add(f.name.replace("__mmlu__clean.json", ""))
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

def bootstrap_pvalue(rho_hat, n, Sn_obs, B=100):
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

def jittered_gof_test(x_raw, y_raw, R=5, B=100):
    p_values = []
    for _ in range(R):
        x_jit = x_raw + np.random.uniform(-0.499, 0.499, len(x_raw))
        y_jit = y_raw + np.random.uniform(-0.499, 0.499, len(y_raw))
        u = pseudo_observations(x_jit)
        v = pseudo_observations(y_jit)
        rho_hat = fit_rho_kendall(u, v)
        Sn = cvm_stat(u, v, rho_hat)
        p = bootstrap_pvalue(rho_hat, len(u), Sn, B=B)
        p_values.append(p)
    return float(np.mean(p_values)), float(np.median(p_values)), [float(x) for x in p_values]

def run_jittered_gof(models, dataset):
    print(f"\n{'='*60}")
    print(f"[JITTERED] Dataset: {dataset} | Models: {len(models)}")
    print(f"{'='*60}", flush=True)

    score_vectors = {}
    for mid in models:
        scores = load_clean_scores(mid, dataset)
        if scores is not None:
            score_vectors[mid] = scores

    print(f"Loaded {len(score_vectors)} models", flush=True)

    results = []
    pairs = list(itertools.combinations(sorted(score_vectors.keys()), 2))
    print(f"Testing {len(pairs)} pairs (R={R_JITTER}, B={B_BOOTSTRAP})...", flush=True)

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

        mean_p, median_p, all_p = jittered_gof_test(x, y, R=R_JITTER, B=B_BOOTSTRAP)

        results.append({
            "pair": f"{m_i}|{m_j}",
            "model_i": m_i,
            "model_j": m_j,
            "n_valid": len(valid_idx),
            "mean_p_value": mean_p,
            "median_p_value": median_p,
            "all_p_values": all_p,
            "reject_005_mean": mean_p < 0.05,
        })

        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1}/{len(pairs)}] {m_i}|{m_j} mean_p={mean_p:.3f}", flush=True)

    print(f"  [{len(pairs)}/{len(pairs)}] complete", flush=True)
    return results

def main():
    np.random.seed(123)
    models = discover_models()
    print(f"Discovered {len(models)} models", flush=True)

    all_results = {}
    summary = {}

    for ds in DATASETS:
        results = run_jittered_gof(models, ds)
        all_results[ds] = results

        tested = [r for r in results if not r.get("skipped")]
        rejected = [r for r in tested if r["reject_005_mean"]]
        mean_ps = [r["mean_p_value"] for r in tested]

        summary[ds] = {
            "n_pairs_tested": len(tested),
            "n_rejected_005": len(rejected),
            "rejection_rate_005": len(rejected) / len(tested) if tested else 0,
            "avg_mean_p_value": float(np.mean(mean_ps)) if mean_ps else None,
            "median_mean_p_value": float(np.median(mean_ps)) if mean_ps else None,
        }

        if rejected:
            jrc = {}
            for r in rejected:
                for m in [r["model_i"], r["model_j"]]:
                    jrc[m] = jrc.get(m, 0) + 1
            summary[ds]["judge_rejection_frequency"] = dict(sorted(jrc.items(), key=lambda x: -x[1]))
        else:
            summary[ds]["judge_rejection_frequency"] = {}

        print(f"\n--- {ds} Jittered Summary ---")
        print(f"  Tested: {len(tested)} pairs")
        print(f"  Rejected (a=0.05): {len(rejected)} ({summary[ds]['rejection_rate_005']:.1%})")
        print(f"  Mean p-value: {summary[ds]['avg_mean_p_value']:.3f}")

    output = {
        "method": "Genest-Remillard_2008_CvM_jittered",
        "description": "Gaussian copula GoF with U[-0.5,0.5] jittering to handle ordinal discreteness",
        "alpha": 0.05,
        "B_bootstrap": B_BOOTSTRAP,
        "R_jitter": R_JITTER,
        "n_models": len(models),
        "models": models,
        "summary": summary,
        "pair_results": all_results,
    }

    out_path = OUTPUT_DIR / "gaussian_copula_gof_jittered.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
