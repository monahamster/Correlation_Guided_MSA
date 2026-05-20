#!/usr/bin/env bash
set -euo pipefail
python main_GLoMo_humor.py --dataset sarcasm --train_batch_size 64 --layers 3 --max_seq_length 70 --d_l 160 --n_epochs 5 --use_moe_reliability --corr_model_path /data01/lyw/GLoMo/pretrained-model/correlation_glomo_mustard_051907.pt --moe_reliability_lambda 0.05 --save_best_by acc2 --experiment_tag mustard_only_moe_seed5576 --log_path /data01/lyw/GLoMo/experiments/mustard_only_moe_seed5576/train.log
