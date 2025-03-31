#! /bin/bash

python benchmark/molecular_synthesis/run_evaluation.py --results_dir benchmark/molecular_synthesis/results/lm1 --output_pkl benchmark/molecular_synthesis/results/lm1/lm1_eval.pkl --n_workers 1
python benchmark/molecular_synthesis/run_evaluation.py --results_dir benchmark/molecular_synthesis/results/swar_lcd --output_pkl benchmark/molecular_synthesis/results/swar_lcd/swar_lcd_eval.pkl --n_workers 1
python benchmark/molecular_synthesis/run_evaluation.py --results_dir benchmark/molecular_synthesis/results/twisted10 --output_pkl benchmark/molecular_synthesis/results/twisted10/twisted10_eval.pkl --n_workers 1
python benchmark/molecular_synthesis/run_evaluation.py --results_dir benchmark/molecular_synthesis/results/swar5 --output_pkl benchmark/molecular_synthesis/results/swar5/swar5_eval.pkl --n_workers 1
