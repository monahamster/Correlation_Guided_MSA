#!/usr/bin/env bash
set -euo pipefail
python main_GLoMo_humor.py --dataset urfunny --train_batch_size 220 --layers 3 --max_seq_length 80 --d_l 112 --n_epochs 100 --use_fusion_correlation --corr_model_path /data01/lyw/GLoMo/pretrained-model/correlation_glomo_urfunny_051907.pt --corr_alpha 0.2 --save_best_by acc2 --experiment_tag urfunny_only_fusion_seed5576 --log_path /data01/lyw/GLoMo/experiments/urfunny_only_fusion_seed5576/train.log
