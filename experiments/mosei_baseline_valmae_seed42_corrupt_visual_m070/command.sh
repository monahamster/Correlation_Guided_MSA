#!/usr/bin/env bash
set -euo pipefail
python main_cgmsa.py --dataset mosei --max_seq_length 80 --train_batch_size 64 --d_l 192 --layers 3 --VISUAL_DIM 35 --learning_rate 1e-5 --n_epochs 100 --seed 42 --test 1 --load 1 --model_path /data01/lyw/GLoMo/experiments/mosei_baseline_valmae_seed42/best_model.pt --test_audio_noise_std 0.0 --test_visual_mask_ratio 0.7 --test_corruption_seed 20260831 --experiment_tag mosei_baseline_valmae_seed42_corrupt_visual_m070 --log_path /data01/lyw/GLoMo/experiments/mosei_baseline_valmae_seed42_corrupt_visual_m070/test.log
