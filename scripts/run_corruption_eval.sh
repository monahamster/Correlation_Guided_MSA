#!/usr/bin/env bash
# Evaluate one validation-selected checkpoint under isolated, reproducible
# audio-noise and visual-frame-masking conditions. No training is performed.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <mosi|mosei> <baseline|ours> <seed> <gpu_id>" >&2
  exit 2
fi

dataset="$1"
variant="$2"
seed="$3"
gpu_id="$4"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$repo_root/Correlation_Guided_MSA"
experiment_root="$repo_root/experiments"
checkpoint="$experiment_root/${dataset}_${variant}_valmae_seed${seed}/best_model.pt"

if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint not found: $checkpoint" >&2
  exit 1
fi

case "$dataset" in
  mosi)
    common_args=(--dataset mosi --max_seq_length 60 --train_batch_size 240 --d_l 48 --layers 4 --VISUAL_DIM 47 --learning_rate 4e-5 --n_epochs 70)
    corr_args=(--corr_model_path "$repo_root/pretrained-model/correlation_glomo_mosi_012112.pt" --corr_alpha 0.2 --moe_reliability_lambda 0.05)
    ;;
  mosei)
    common_args=(--dataset mosei --max_seq_length 80 --train_batch_size 64 --d_l 192 --layers 3 --VISUAL_DIM 35 --learning_rate 1e-5 --n_epochs 100)
    corr_args=(--corr_model_path "$repo_root/pretrained-model/correlation_glomo_mosei_012112.pt" --corr_alpha 0.2 --moe_reliability_lambda 0.1)
    ;;
  *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

case "$variant" in
  baseline) model_args=() ;;
  ours) model_args=(--use_fusion_correlation --use_moe_reliability "${corr_args[@]}") ;;
  *) echo "Unsupported variant: $variant" >&2; exit 2 ;;
esac

# name, relative acoustic Gaussian-noise sigma, visual valid-frame mask ratio
conditions=(
  "clean 0.0 0.0"
  "audio_n010 0.1 0.0"
  "audio_n020 0.2 0.0"
  "audio_n030 0.3 0.0"
  "audio_n050 0.5 0.0"
  "audio_n100 1.0 0.0"
  "visual_m010 0.0 0.1"
  "visual_m030 0.0 0.3"
  "visual_m050 0.0 0.5"
  "visual_m070 0.0 0.7"
  "visual_m100 0.0 1.0"
)

cd "$work_dir"
for condition in "${conditions[@]}"; do
  read -r condition_name audio_noise visual_mask <<< "$condition"
  tag="${dataset}_${variant}_valmae_seed${seed}_corrupt_${condition_name}"
  run_dir="$experiment_root/$tag"
  if [[ -e "$run_dir/metrics.json" ]]; then
    echo "Skipping existing result: $run_dir"
    continue
  fi
  mkdir -p "$run_dir"

  CUDA_VISIBLE_DEVICES="$gpu_id" python main_cgmsa.py \
    "${common_args[@]}" \
    "${model_args[@]}" \
    --seed "$seed" \
    --test 1 \
    --load 1 \
    --model_path "$checkpoint" \
    --test_audio_noise_std "$audio_noise" \
    --test_visual_mask_ratio "$visual_mask" \
    --test_corruption_seed 20260831 \
    --experiment_tag "$tag" \
    --log_path "$run_dir/test.log" \
    | tee "$run_dir/test.log"
done
