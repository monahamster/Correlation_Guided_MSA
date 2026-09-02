#!/usr/bin/env bash
set -euo pipefail
python main_cgmsa.py --dataset mosi --max_seq_length 60 --train_batch_size 240 --d_l 48 --layers 4 --VISUAL_DIM 47 --learning_rate 4e-5 --n_epochs 70 --seed 2026 --test 1 --load 1 --model_path /data01/lyw/GLoMo/experiments/mosi_baseline_valmae_seed2026/best_model.pt --test_audio_noise_std 0.2 --test_visual_mask_ratio 0.0 --test_corruption_seed 20260831 --experiment_tag mosi_baseline_valmae_seed2026_corrupt_audio_n020 --log_path /data01/lyw/GLoMo/experiments/mosi_baseline_valmae_seed2026_corrupt_audio_n020/test.log
