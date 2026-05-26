#!/usr/bin/env bash
# Burst-vs-step speedup sweep across model sizes (run on the GPU box).
# Each model gets a fresh engine (sequential invocations) to avoid GPU-pool
# accumulation / OOM. max_model_len is per-model (gpt2 caps at 1024).
set -u
cd /root/genlm/genlm-control
export VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
PY=/root/genlm-venv/bin/python

# model:max_model_len  (cached on the box: a 124M -> 7B ladder)
SPECS=(
  "gpt2:1024"
  "Qwen/Qwen2.5-0.5B-Instruct:2048"
  "Qwen/Qwen2.5-1.5B-Instruct:2048"
  "Qwen/Qwen2.5-7B-Instruct:2048"
)

for spec in "${SPECS[@]}"; do
  m="${spec%%:*}"; L="${spec##*:}"
  echo "##### MODEL $m (max_model_len=$L) #####"
  "$PY" benchmarks/bench_burst.py \
      --model "$m" --max-model-len "$L" \
      --n-particles 16 --max-tokens 128 --n-trials 3 --gpu-mem 0.7 \
      2>&1 | grep -vE "WARNING|Loading|Capturing|it/s\]|INFO 0|^\(|Processed prompts|Adding requests"
  echo
done
echo "##### SWEEP DONE #####"
