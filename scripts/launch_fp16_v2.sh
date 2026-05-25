#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
source /etc/network_turbo 2>/dev/null || true
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export HF_HOME=/root/autodl-tmp/.hf_cache
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0

cd /root/cert_manip_resist_eval
python scripts/run_fp16_baseline_v2.py "$@"
