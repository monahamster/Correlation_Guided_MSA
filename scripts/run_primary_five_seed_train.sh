#!/usr/bin/env bash
# Train one additional validation-selected primary-comparison run.
# The script is intentionally limited to the GLoMo* baseline and full model;
# it does not overwrite the existing three-seed records.
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

case "$dataset" in
  mosi)
    common_args=(--dataset mosi --max_seq_length 60 --train_batch_size 240 --d_l 48 --layers 4 --VISUAL_DIM 47 --learning_rate 4e-5 --n_epochs 70)
    corr_path="$repo_root/pretrained-model/correlation_glomo_mosi_012112.pt"
    corr_alpha=0.2
    moe_lambda=0.05
    ;;
  mosei)
    common_args=(--dataset mosei --max_seq_length 80 --train_batch_size 64 --d_l 192 --layers 3 --VISUAL_DIM 35 --learning_rate 1e-5 --n_epochs 100)
    corr_path="$repo_root/pretrained-model/correlation_glomo_mosei_012112.pt"
    corr_alpha=0.2
    moe_lambda=0.1
    ;;
  *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

case "$variant" in
  baseline) model_args=() ;;
  ours)
    model_args=(--use_fusion_correlation --use_moe_reliability
      --corr_model_path "$corr_path" --corr_alpha "$corr_alpha"
      --moe_reliability_lambda "$moe_lambda")
    ;;
  *) echo "Unsupported variant: $variant" >&2; exit 2 ;;
esac

tag="${dataset}_${variant}_valmae_seed${seed}"
run_dir="$experiment_root/$tag"
if [[ -e "$run_dir/metrics.json" || -e "$run_dir/best_model.pt" ]]; then
  echo "Refusing to overwrite existing experiment: $run_dir" >&2
  exit 1
fi

mkdir -p "$run_dir"
cd "$work_dir"
CUDA_VISIBLE_DEVICES="$gpu_id" python main_cgmsa.py \
  "${common_args[@]}" \
  "${model_args[@]}" \
  --seed "$seed" \
  --save_best_by valid_mae \
  --experiment_tag "$tag" \
  --log_path "$run_dir/train.log" \
  | tee "$run_dir/train.log"
