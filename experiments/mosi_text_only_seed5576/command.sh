#!/usr/bin/env bash
set -euo pipefail
python main_GLoMo.py --dataset mosi --max_seq_length 60 --train_batch_size 240 --d_l 48 --layers 4 --VISUAL_DIM 47 --learning_rate 4e-5 --n_epochs 70 --use_fusion_correlation --corr_model_path /data01/lyw/GLoMo/pretrained-model/correlation_glomo_mosi_012112.pt --corr_alpha 0.2 --use_moe_reliability --moe_reliability_lambda 0.05 --drop_audio --drop_visual --save_best_by acc2 --experiment_tag mosi_text_only_seed5576 --log_path /data01/lyw/GLoMo/experiments/mosi_text_only_seed5576/train.log
