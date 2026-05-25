#!/usr/bin/env python3
"""Synthetic validation of K_eff predictor shift (R6 Batch 1B).

Tests whether "predictor shift" (K_eff significance increasing with panel
size K) is a genuine power effect or a statistical artifact.

Scenario A — True Shift: K_eff causally affects panel ASR (constant effect)
Scenario B — No Shift: ASR depends only on η_max, K_eff is confounded

Monte Carlo: 200 reps × K={3,5,7} × WCB (B=999, Rademacher, position max-p)

Key insight: with constant β_keff, WCB power should NOT increase with K
(fewer clusters: 13→11→9 offset larger sample size). If real data shows
increasing significance, K_eff effect genuinely strengthens at larger K.
"""

import json
import time
import itertools
import numpy as np
import statsmodels.api as sm
from pathlib import Path

N_MODELS = 15
K_VALUES = [3, 5, 7]
N_MC = 200
B_BOOT = 999
ALPHA = 0.05
SEED = 2026

INTERCEPT = 0.20
BETA_KEFF_A = -0.008
BETA_ETA = 0.05
NOISE_STD = 0.08
CONFOUND_SCALE = 0.4


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
    raise TypeError(f"Not serializable: {type(obj)}")


def generate_model_properties(rng, n=N_MODELS):
    eta = rng.beta(2, 5, size=n)
    v_sum = eta[:, None] + eta[None, :]
    v_max = eta.max()
    base_corr = 0.1 + CONFOUND_SCALE * v_sum / (2 * v_max + 1e-10)
    noise = rng.normal(0, 0.08, size=(n, n))
    noise = (noise + noise.T) / 2
    corr = base_corr + noise
    corr = np.clip(corr, 0.02, 0.93)
    np.fill_diagonal(corr, 0)
    mi = -0.5 * np.log(1 - corr ** 2 + 1e-10)
    mi = np.maximum(mi, 0)
    np.fill_diagonal(mi, 0)
    return eta, mi


def compute_keffs_batch(mi_mat, panels_arr):
    _, K = panels_arr.shape
    mi_sub = mi_mat[panels_arr[:, :, None], panels_arr[:, None, :]]
    exp_neg2mi = np.exp(-2 * mi_sub)
    diag_mask = np.eye(K, dtype=bool)
    exp_neg2mi[:, diag_mask] = 0
    terms = 1.0 + np.sum(exp_neg2mi, axis=2)
    return np.mean(terms, axis=1)


def wcb_maxp(y, X, panels_arr, K, B, seed, test_col=1):
    n, p = X.shape
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 1.0, 0.0, 0.0

    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    s2 = np.sum(resid ** 2) / max(n - p, 1)
    se_tc = np.sqrt(s2 * XtX_inv[test_col, test_col])
    if se_tc < 1e-15:
        return 1.0, float(beta[test_col]), 0.0
    t_obs = beta[test_col] / se_tc
    abs_t_obs = abs(t_obs)

    cols_r = [i for i in range(p) if i != test_col]
    X_r = X[:, cols_r]
    try:
        XrXr_inv = np.linalg.inv(X_r.T @ X_r)
    except np.linalg.LinAlgError:
        return 1.0, float(beta[test_col]), float(t_obs)
    beta_r = XrXr_inv @ (X_r.T @ y)
    y_hat_r = X_r @ beta_r
    e_r = y - y_hat_r
    XtX_inv_tc = XtX_inv[test_col, test_col]

    max_pval = 0.0
    for pos in range(K):
        cl = panels_arr[:, pos]
        uniq = np.unique(cl)
        n_cl = len(uniq)
        cl_map = {c: i for i, c in enumerate(uniq)}
        cl_idx = np.array([cl_map[c] for c in cl])

        rng = np.random.RandomState(seed + pos * 7919)
        w_cl = rng.choice([-1, 1], size=(B, n_cl))
        w = w_cl[:, cl_idx]
        Y_b = y_hat_r[None, :] + w * e_r[None, :]
        beta_b = XtX_inv @ (X.T @ Y_b.T)
        resid_b = Y_b.T - X @ beta_b
        s2_b = np.sum(resid_b ** 2, axis=0) / max(n - p, 1)
        se_b = np.sqrt(np.maximum(s2_b * XtX_inv_tc, 1e-30))
        t_b = beta_b[test_col] / se_b
        pval = (1 + np.sum(np.abs(t_b) >= abs_t_obs)) / (1 + B)
        max_pval = max(max_pval, pval)

    return max_pval, float(beta[test_col]), float(t_obs)


def run_trial(rng, scenario, K, eta, mi_mat, panels_arr):
    n_panels = panels_arr.shape[0]
    keffs = compute_keffs_batch(mi_mat, panels_arr)
    eta_maxs = eta[panels_arr].max(axis=1)
    keff_z = (keffs - keffs.mean()) / (keffs.std() + 1e-10)
    eta_z = (eta_maxs - eta_maxs.mean()) / (eta_maxs.std() + 1e-10)

    eps = rng.normal(0, NOISE_STD, n_panels)
    if scenario == "A":
        asr = INTERCEPT + BETA_KEFF_A * keff_z + BETA_ETA * eta_z + eps
    else:
        asr = INTERCEPT + BETA_ETA * eta_z + eps
    asr = np.clip(asr, 0.001, 0.999)

    X = np.column_stack([np.ones(n_panels), keff_z, eta_z])
    seed = rng.randint(0, 10 ** 7)
    pval, beta_keff, t_keff = wcb_maxp(asr, X, panels_arr, K, B_BOOT, seed)
    return {
        "p": pval,
        "beta": beta_keff,
        "t": t_keff,
        "corr": float(np.corrcoef(keff_z, eta_z)[0, 1]),
    }


def main():
    t0 = time.time()
    master_rng = np.random.RandomState(SEED)

    panel_arrays = {}
    for K in K_VALUES:
        combos = list(itertools.combinations(range(N_MODELS), K))
        panel_arrays[K] = np.array(combos)
        n_cl = N_MODELS - K + 1
        print(f"K={K}: {len(combos)} panels, {n_cl} clusters/position")

    results = {}
    for scenario in ["A", "B"]:
        results[scenario] = {}
        for K in K_VALUES:
            pa = panel_arrays[K]
            trials = []
            for i in range(N_MC):
                seed = master_rng.randint(0, 10 ** 7)
                rng = np.random.RandomState(seed)
                eta, mi_mat = generate_model_properties(rng)
                res = run_trial(rng, scenario, K, eta, mi_mat, pa)
                trials.append(res)
                if (i + 1) % 50 == 0:
                    print(f"  Scenario {scenario} K={K}: {i + 1}/{N_MC} "
                          f"({time.time() - t0:.0f}s)")

            pvals = [t["p"] for t in trials]
            betas = [t["beta"] for t in trials]
            reject = sum(1 for p in pvals if p < ALPHA)
            rr = reject / N_MC

            results[scenario][f"K{K}"] = {
                "rejection_rate": rr,
                "reject_count": reject,
                "n_trials": N_MC,
                "mean_keff_beta": float(np.mean(betas)),
                "keff_beta_ci95": [
                    float(np.percentile(betas, 2.5)),
                    float(np.percentile(betas, 97.5)),
                ],
                "mean_p": float(np.mean(pvals)),
                "median_p": float(np.median(pvals)),
                "mean_corr_keff_eta": float(
                    np.mean([t["corr"] for t in trials])
                ),
                "n_panels": int(pa.shape[0]),
            }

            ci = results[scenario][f"K{K}"]["keff_beta_ci95"]
            print(
                f"  Scenario {scenario} K={K}: "
                f"rej={rr:.3f} ({reject}/{N_MC}), "
                f"mean_beta={np.mean(betas):.4f} "
                f"[{ci[0]:.4f},{ci[1]:.4f}]"
            )

    elapsed = round(time.time() - t0, 1)

    rr_A = [results["A"][f"K{K}"]["rejection_rate"] for K in K_VALUES]
    rr_B = [results["B"][f"K{K}"]["rejection_rate"] for K in K_VALUES]
    shift_A = all(rr_A[i] <= rr_A[i + 1] for i in range(len(rr_A) - 1))
    artifact = (rr_B[-1] - rr_B[0]) > 0.10

    output = {
        "method": "synthetic_validation_predictor_shift",
        "dgp": {
            "n_models": N_MODELS,
            "vulnerability_dist": "Beta(2,5)",
            "mi_confounding": (
                f"base_corr=0.1+{CONFOUND_SCALE}*(v_i+v_j)/(2*v_max)+noise"
            ),
            "scenario_A": (
                f"ASR={INTERCEPT}+({BETA_KEFF_A})*K_eff_z"
                f"+({BETA_ETA})*eta_max_z+N(0,{NOISE_STD}^2)"
            ),
            "scenario_B": (
                f"ASR={INTERCEPT}+({BETA_ETA})*eta_max_z+N(0,{NOISE_STD}^2)"
            ),
        },
        "wcb_config": {
            "B": B_BOOT,
            "weights": "rademacher",
            "clustering": "position_based_max_p",
            "alpha": ALPHA,
        },
        "n_mc_trials": N_MC,
        "scenario_A": results["A"],
        "scenario_B": results["B"],
        "diagnostics": {
            "scenario_A_rejection_rates": rr_A,
            "scenario_B_rejection_rates": rr_B,
            "shift_in_A": bool(shift_A),
            "artifact_detected": bool(artifact),
            "expected_pattern": (
                "Under constant beta_keff, WCB power should DECREASE "
                "with K (fewer clusters: 13->11->9). "
                "If real data shows INCREASING significance, the K_eff "
                "effect genuinely strengthens at larger K."
            ),
        },
        "interpretation": {
            "A_decrease_B_flat": (
                "Expected baseline: constant-effect K_eff loses WCB power "
                "at larger K due to fewer clusters. Real-data predictor "
                "shift therefore reflects genuine intensification of K_eff "
                "effect, not statistical artifact."
            ),
            "A_flat_B_flat": (
                "Cluster count reduction and sample size increase roughly "
                "cancel. Real-data shift still implies genuine effect "
                "intensification."
            ),
            "A_increase_B_flat": (
                "WCB power increases with K despite fewer clusters. "
                "Sample size effect dominates. Real-data shift could be "
                "partly explained by power increase."
            ),
            "both_increase": (
                "Predictor shift is a statistical artifact of the "
                "WCB + position clustering methodology."
            ),
        },
        "elapsed_seconds": elapsed,
        "seed": SEED,
    }

    out_path = Path(
        "/root/cert_manip_resist_eval/artifacts/results/plan001/"
        "synthetic_validation.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default,
                  ensure_ascii=False)

    print(f"\n{'=' * 74}")
    print("SYNTHETIC VALIDATION RESULTS")
    print(f"{'=' * 74}")
    print(
        f"{'Sc':<4} {'K':>2} {'Rej Rate':>10} {'Count':>7} "
        f"{'Mean beta':>10} {'95% CI':>22} {'r(K,eta)':>9}"
    )
    print("-" * 74)
    for sc in ["A", "B"]:
        for K in K_VALUES:
            r = results[sc][f"K{K}"]
            ci = r["keff_beta_ci95"]
            print(
                f"  {sc:<2} {K:>2} {r['rejection_rate']:>10.3f} "
                f"{r['reject_count']:>5}/{N_MC} "
                f"{r['mean_keff_beta']:>10.4f} "
                f"[{ci[0]:>8.4f},{ci[1]:>8.4f}] "
                f"{r['mean_corr_keff_eta']:>9.3f}"
            )
        if sc == "A":
            print()
    print(f"{'=' * 74}")
    print(f"Shift in A: {shift_A} | Artifact in B: {artifact}")
    print(f"Clusters/position: K=3->13, K=5->11, K=7->9")
    print(f"Elapsed: {elapsed}s | Saved: {out_path}")


if __name__ == "__main__":
    main()
