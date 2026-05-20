#!/usr/bin/env bash
set -euo pipefail
python main_GLoMo_humor.py --dataset sarcasm --train_batch_size 64 --layers 3 --max_seq_length 70 --d_l 160 --n_epochs 5 --save_best_by acc2 --experiment_tag mustard_baseline_seed5576 --log_path /data01/lyw/GLoMo/experiments/mustard_baseline_seed5576/train.log
