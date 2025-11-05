#!/bin/bash
# COLLIE Character-Level Evaluation Script
# Runs evaluation on multiple constraint types with 0 resampling

MODEL="meta-llama/Llama-3.1-8B-Instruct"
MAX_INSTANCES=50
MAX_TOKENS=400
ESS_THRESHOLD=0.0
K=5
BASE_DIR="/teamspace/studios/this_studio/genlm-control"
LOG_DIR="${BASE_DIR}/collie_results_char"

# Create log directory
mkdir -p "$LOG_DIR"

# Constraint types to test (c02 to c08)
# Word level: c02, c03
# Sentence level: c04, c05, c06a, c07
# Paragraph level: c08

CONSTRAINTS=(
    "c02"      # Word: exact 12 chars + positions
    "c03"      # Word: 7+ chars + last char constraint
    "c04"      # Sentence: exact character count
    "c05"      # Sentence: exact word count + positions
    "c06a"     # Sentence: word count + all words <= 7 chars
    "c07"      # Sentence: must contain 3 words
    "c08"      # Paragraph: all sentences start with same word
)

echo "=========================================="
echo "COLLIE Character-Level Evaluation"
echo "=========================================="
echo "Model: $MODEL"
echo "Max instances: $MAX_INSTANCES"
echo "Max tokens: $MAX_TOKENS"
echo "ESS threshold: $ESS_THRESHOLD (no resampling)"
echo "Beam K: $K"
echo "Number of constraints: ${#CONSTRAINTS[@]}"
echo "=========================================="
echo ""

for constraint in "${CONSTRAINTS[@]}"; do
    echo "Running constraint: $constraint"
    echo "----------------------------------------"
    
    log_file="${LOG_DIR}/collie_${constraint}_char.log"
    
    python "${BASE_DIR}/run_collie_eval_char.py" \
        --model "$MODEL" \
        --constraint_filter "$constraint" \
        --max_instances "$MAX_INSTANCES" \
        --max_tokens "$MAX_TOKENS" \
        --ess_threshold "$ESS_THRESHOLD" \
        --K "$K" \
        2>&1 | tee "$log_file"
    
    echo ""
    echo "Completed $constraint. Results saved to: $log_file"
    echo ""
done

echo "=========================================="
echo "All evaluations complete!"
echo "Results saved in: $LOG_DIR"
echo "=========================================="

