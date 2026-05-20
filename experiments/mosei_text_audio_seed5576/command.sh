#!/usr/bin/env bash
set -euo pipefail
python main_GLoMo.py --dataset mosei --max_seq_length 80 --train_batch_size 64 --d_l 192 --layers 3 --VISUAL_DIM 35 --learning_rate 1e-5 --n_epochs 100 --use_fusion_correlation --corr_model_path /data01/lyw/GLoMo/pretrained-model/correlation_glomo_mosei_012112.pt --corr_alpha 0.2 --use_moe_reliability --moe_reliability_lambda 0.1 --drop_visual --save_best_by acc2 --experiment_tag mosei_text_audio_seed5576 --log_path /data01/lyw/GLoMo/experiments/mosei_text_audio_seed5576/train.log
