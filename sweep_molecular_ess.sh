#!/bin/bash
# Sweep ESS threshold on molecular synthesis with character-level model
# Tests: 0.2, 0.4, 0.5, 0.6, 0.8 on 50 instances

MODEL="meta-llama/Llama-3.1-8B-Instruct"
N_INSTANCES=50
N_MOLECULES=10
N_PARTICLES=5
MAX_INSTANCES=50
OUTPUT_DIR="results_molecular_sweep"

# Create output directory
mkdir -p $OUTPUT_DIR

# ESS thresholds to sweep
ESS_THRESHOLDS=(0.0 0.2 0.4 0.5 0.6 0.8)

echo "=========================================="
echo "Molecular Synthesis ESS Threshold Sweep"
echo "=========================================="
echo "Model: $MODEL"
echo "Instances: $N_INSTANCES (max: $MAX_INSTANCES)"
echo "Molecules per instance: $N_MOLECULES"
echo "Particles: $N_PARTICLES"
echo "ESS thresholds: ${ESS_THRESHOLDS[@]}"
echo "=========================================="
echo ""

# Run sweep
for ess in "${ESS_THRESHOLDS[@]}"; do
    echo "Running with ess_threshold=$ess..."
    LOG_FILE="$OUTPUT_DIR/molecular_char_ess${ess}.log"
    
    python run_molecular_eval_char.py \
        --model $MODEL \
        --n_instances $N_INSTANCES \
        --n_molecules $N_MOLECULES \
        --n_particles $N_PARTICLES \
        --ess_threshold $ess \
        --max_instances $MAX_INSTANCES \
        2>&1 | tee $LOG_FILE
    
    # Extract final accuracy
    ACC=$(grep "Average Weighted Accuracy" $LOG_FILE | tail -1 | grep -oP '[\d.]+' | tail -1)
    echo "ESS=$ess -> Accuracy=$ACC"
    echo ""
done

echo "=========================================="
echo "Sweep Complete! Results:"
echo "=========================================="

# Summary
for ess in "${ESS_THRESHOLDS[@]}"; do
    LOG_FILE="$OUTPUT_DIR/molecular_char_ess${ess}.log"
    if [ -f "$LOG_FILE" ]; then
        ACC=$(grep "Average Weighted Accuracy" $LOG_FILE | tail -1 | grep -oP '[\d.]+' | tail -1)
        echo "ESS=$ess -> Accuracy=$ACC"
    fi
done

echo ""
echo "Logs saved in: $OUTPUT_DIR/"

