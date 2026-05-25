# Certified Manipulation Resistance Evaluation

Code and data for the paper: *"Diversity Hurts: Decomposing Multi-Judge Panel Robustness into Weakest-Link Susceptibility and Effective Independence"* (Anonymous, EMNLP 2026 Submission).

## Overview

This repository provides the full experimental pipeline for evaluating manipulation resistance of multi-judge LLM evaluation panels. We decompose panel attack success rate into:
- **Weakest-link susceptibility** ($\eta_{\max}$): the vulnerability of the most manipulable judge
- **Effective independence** ($K_{\mathrm{eff}}$): the number of truly independent judges in a panel

## Setup

```bash
pip install -r requirements.txt
```

**Key dependencies**: vllm, transformers, datasets, numpy, scipy, pandas, scikit-learn, tqdm

For API-based judge scoring, set the `OPENROUTER_API_KEY` environment variable.

## Project Structure

```
src/                    # Core library
  config.py             # Model/dataset/experiment configuration
  attacks.py            # Attack implementations
  panels.py             # Panel construction and aggregation
  keff.py               # K_eff (effective independence) computation
  correlation.py        # Inter-judge correlation analysis
  scaling.py            # Panel size scaling analysis
  eval_pipeline.py      # End-to-end evaluation pipeline
  unified_data_loader.py # Unified data loading
  utils.py              # Shared utilities

scripts/                # Experiment scripts
  run_step1_correlation.py      # Step 1: Inter-judge correlation
  run_step2_attacks.py          # Step 2: Attack experiments
  run_step3_keff.py             # Step 3: K_eff computation
  run_step4_scaling.py          # Step 4: Panel size scaling
  run_plan001_step1_api_scoring.py  # API-based scoring pipeline
  run_plan001_step4_panel_regression.py  # Panel-level regression
  ...                           # Additional analysis scripts

scoring/                # API-based judge scoring subsystem
  src/                  # Scoring library (attacks, judge service, config)
  scripts/              # Scoring scripts

docs/paper/             # Paper source (LaTeX)
```

## Reproducing Experiments

### 1. Configure Model Paths

Edit `src/config.py` to set model paths in `MODEL_REGISTRY` to point to your local model directories. Models used:
- Qwen2.5-72B/32B/14B-Instruct (AWQ)
- Llama-3.1-70B/8B-Instruct (AWQ)
- Mistral-Large-Instruct-2407 (AWQ)
- API models via OpenRouter (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, etc.)

### 2. Run the Pipeline

```bash
# Step 1: Compute inter-judge correlations
python scripts/run_step1_correlation.py

# Step 2: Run attack experiments
python scripts/run_step2_attacks.py

# Step 3: Compute K_eff
python scripts/run_step3_keff.py

# Step 4: Panel size scaling analysis
python scripts/run_step4_scaling.py

# API-based scoring (requires OPENROUTER_API_KEY)
python scripts/run_plan001_step1_api_scoring.py

# Panel regression analysis
python scripts/run_plan001_step4_panel_regression.py
```

### 3. Statistical Analyses

```bash
# Wild cluster bootstrap
python scripts/run_wild_cluster_bootstrap_v2.py

# Meta-analytic pooling
python scripts/run_meta_analytic_pooling.py

# Monte Carlo size calibration
python scripts/run_mc_size_calibration.py

# Kappa-diversity regression
python scripts/run_kappa_diversity_regression.py

# Cross-benchmark validation
python scripts/run_cross_benchmark_bootstrap.py
```

Results are saved to `artifacts/results/`.

## License

This code is released for academic research purposes.
