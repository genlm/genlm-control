#!/usr/bin/env bash
# cProfile every scenario's burst path -> per-scenario gprof2dot call-graph SVG.
# Fresh engine per scenario (separate procs) avoids GPU-pool accumulation.
# Usage: prof_all.sh [model] [mode]   (mode: burst|step, default burst)
set -u
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
MODE="${2:-burst}"
LABEL="${3:-$MODE}"   # output tag: prof__<scenario>__<LABEL>.svg (use a version label)
PY=/root/genlm/genlm-venv/bin/python
cd /root/genlm/genlm-control
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 OMP_NUM_THREADS=1
export PATH=/root/genlm/genlm-venv/bin:$PATH  # so prof_entry finds gprof2dot/dot
mkdir -p benchmarks/results
COMMON="--mode $MODE --n-particles 16 --max-tokens 128 --gpu-mem 0.6 --max-model-len 2048"

prof() { echo "##### prof $* #####"; $PY benchmarks/prof_entry.py "$@" 2>&1 \
         | grep -E "wrote|Error|Traceback|NotAcceleratable|raise"; echo; }

prof --scenario direct --model "$MODEL" --out benchmarks/results/prof__direct__$LABEL $COMMON
prof --scenario awrs   --model "$MODEL" --constraint alpha --out benchmarks/results/prof__awrs__$LABEL $COMMON
prof --scenario set    --model "$MODEL" --constraint alpha --out benchmarks/results/prof__set__$LABEL $COMMON
prof --scenario cot    --model "$MODEL" --out benchmarks/results/prof__cot__$LABEL $COMMON
prof --scenario ds1000 --model "$MODEL" --library Pandas --item 0 --timeout 6 --out benchmarks/results/prof__ds1000__$LABEL $COMMON
# LoRA: Qwen base + adapter, shorter rollout (per-view StepLoop off path is heavy)
prof --scenario lora --out benchmarks/results/prof__lora__$LABEL --mode "$MODE" \
     --n-particles 16 --max-tokens 64 --gpu-mem 0.6 --max-model-len 2048
echo "##### PROF DONE #####"
