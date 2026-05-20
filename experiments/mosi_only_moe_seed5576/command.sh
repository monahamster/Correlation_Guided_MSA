#!/usr/bin/env bash
set -euo pipefail
python main_GLoMo.py --dataset mosi --max_seq_length 60 --train_batch_size 240 --d_l 48 --layers 4 --VISUAL_DIM 47 --learning_rate 4e-5 --n_epochs 70 --use_moe_reliability --moe_reliability_lambda 0.05 --save_best_by acc2 --experiment_tag mosi_only_moe_seed5576 --log_path /data01/lyw/GLoMo/experiments/mosi_only_moe_seed5576/train.log
