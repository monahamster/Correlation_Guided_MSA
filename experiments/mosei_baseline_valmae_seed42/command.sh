#!/usr/bin/env bash
set -euo pipefail
python main_cgmsa.py --dataset mosei --max_seq_length 80 --train_batch_size 64 --d_l 192 --layers 3 --VISUAL_DIM 35 --learning_rate 1e-5 --n_epochs 100 --seed 42 --save_best_by valid_mae --experiment_tag mosei_baseline_valmae_seed42 --log_path /data01/lyw/GLoMo/experiments/mosei_baseline_valmae_seed42/train.log
