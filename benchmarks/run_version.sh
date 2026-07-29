#!/usr/bin/env bash
# Run the full scenario sweep for ONE control version into the shared results store.
# A fresh engine per scenario (separate python procs) avoids GPU-pool accumulation.
# Usage:  run_version.sh <label> [model] [store]
#   label  : version key (main | speedup-old | speedup-now)
#   model  : LM for direct/awrs/set/cot/ds1000 (default Qwen2.5-7B-Instruct)
#   store  : results store name (default bench)
set -u
LABEL="${1:?need a label}"
MODEL="${2:-Qwen/Qwen2.5-7B-Instruct}"
STORE="${3:-bench}"
DRAW="${4:-gumbel_max}"  # token picker: gumbel_max | multinomial | inverse_cdf
PY=/root/genlm/genlm-venv/bin/python
cd /root/genlm/genlm-control
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 OMP_NUM_THREADS=1

COMMON="--label $LABEL --store $STORE --draw $DRAW --n-particles 16 --max-tokens 128 \
        --n-trials 3 --n-warmup 1 --gpu-mem 0.6 --max-model-len 2048"
FILT="grep -vE WARNING|Loading|Capturing|it/s.|INFO|^\(|Processed|Adding|capability|deprecated|Fetching|warnings.warn|SparseCsr"

run() { echo "##### [$LABEL] $* #####"; $PY benchmarks/bench.py "$@" 2>&1 \
        | grep -vE "WARNING|Loading|Capturing|it/s\]|INFO|^\(|Processed|Adding|capability|deprecated|Fetching|warnings\.warn|SparseCsr|^\s*$"; echo; }

run --scenario direct --model "$MODEL" $COMMON
run --scenario awrs   --model "$MODEL" --constraint alpha $COMMON
run --scenario set    --model "$MODEL" --constraint alpha $COMMON
run --scenario cot    --model "$MODEL" $COMMON
run --scenario ds1000 --model "$MODEL" --library Pandas --item 0 --timeout 6 --critic-split $COMMON

# LoRA K=2 multiview exists on the speedup-now line (added after main/6f18790).
if [ "$LABEL" != "main" ] && [ "$LABEL" != "speedup-old" ]; then
  run --scenario lora --label "$LABEL" --store "$STORE" --draw "$DRAW" --n-particles 16 --max-tokens 64 \
      --n-trials 3 --n-warmup 1 --gpu-mem 0.6 --max-model-len 2048
fi
echo "##### [$LABEL] SWEEP DONE #####"
