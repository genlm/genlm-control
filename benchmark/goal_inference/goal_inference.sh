#! /bin/bash

python benchmark/goal_inference/run_inference.py --model_name meta-llama/Meta-Llama-3.1-8B --max_tokens 100 --grammar_dir benchmark/goal_inference/data/grammars --n_particles 1 --ess_threshold 0.9 --output_dir benchmark/goal_inference/results/lm1 --sampler_name lm --overwrite --time_sampler --timeout 1000 --lm_args '{"engine_opts" : {"max_model_len" : 7760, "rope_scaling": {"rope_type": "dynamic", "factor": 8.0}}}'
python benchmark/goal_inference/run_inference.py --model_name meta-llama/Meta-Llama-3.1-8B --max_tokens 100 --grammar_dir benchmark/goal_inference/data/grammars --n_particles 1 --ess_threshold 0. --output_dir benchmark/goal_inference/results/swar_lcd --sampler_name clip --sampler_args '{"proper_weights": false, "top_p1": 0.9, "top_p2": 0.900001}' --time_sampler --timeout 1000 --lm_args '{"engine_opts": {"max_model_len": 7760, "rope_scaling": {"rope_type": "dynamic", "factor": 8.0}}}' --overwrite
python benchmark/goal_inference/run_inference.py --model_name meta-llama/Meta-Llama-3.1-8B --max_tokens 100 --grammar_dir benchmark/goal_inference/data/grammars --n_particles 5 --ess_threshold 0.999 --output_dir benchmark/goal_inference/results/swar5 --sampler_name clip --sampler_args '{"top_p1": 0.9, "top_p2": 0.900001}' --time_sampler --timeout 1000 --lm_args '{"engine_opts": {"max_model_len": 7760, "rope_scaling": {"rope_type": "dynamic", "factor": 8.0}}}' --overwrite
python benchmark/goal_inference/run_inference.py --model_name meta-llama/Meta-Llama-3.1-8B --max_tokens 100 --grammar_dir benchmark/goal_inference/data/grammars --n_particles 10 --ess_threshold 0.95 --output_dir benchmark/goal_inference/results/twisted10 --sampler_name lm --use_critic --time_sampler --timeout 1000 --lm_args '{"engine_opts" : {"max_model_len": 7760, "rope_scaling": {"rope_type": "dynamic", "factor": 8.0}}}' --overwrite