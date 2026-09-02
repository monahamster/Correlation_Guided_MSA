#!/usr/bin/env bash
# Evaluate correlation-guidance controls for one trained "ours" checkpoint.
# No optimization is performed; only inference-time scale assignment changes.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <mosi|mosei> <seed> <gpu_id>" >&2
  exit 2
fi

dataset="$1"
seed="$2"
gpu_id="$3"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$repo_root/Correlation_Guided_MSA"
experiment_root="$repo_root/experiments"
checkpoint="$experiment_root/${dataset}_ours_valmae_seed${seed}/best_model.pt"

if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint not found: $checkpoint" >&2
  exit 1
fi

case "$dataset" in
  mosi)
    common_args=(--dataset mosi --max_seq_length 60 --train_batch_size 240 --d_l 48 --layers 4 --VISUAL_DIM 47 --learning_rate 4e-5 --n_epochs 70)
    correlation_args=(--corr_model_path "$repo_root/pretrained-model/correlation_glomo_mosi_012112.pt" --corr_alpha 0.2 --moe_reliability_lambda 0.05)
    ;;
  mosei)
    common_args=(--dataset mosei --max_seq_length 80 --train_batch_size 64 --d_l 192 --layers 3 --VISUAL_DIM 35 --learning_rate 1e-5 --n_epochs 100)
    correlation_args=(--corr_model_path "$repo_root/pretrained-model/correlation_glomo_mosei_012112.pt" --corr_alpha 0.2 --moe_reliability_lambda 0.1)
    ;;
  *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
esac

cd "$work_dir"
for control in neutral shuffle; do
  tag="${dataset}_ours_valmae_seed${seed}_control_${control}"
  run_dir="$experiment_root/$tag"
  if [[ -e "$run_dir/metrics.json" ]]; then
    echo "Skipping existing result: $run_dir"
    continue
  fi
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES="$gpu_id" python main_cgmsa.py \
    "${common_args[@]}" \
    --use_fusion_correlation --use_moe_reliability "${correlation_args[@]}" \
    --seed "$seed" --test 1 --load 1 --model_path "$checkpoint" \
    --correlation_control "$control" \
    --experiment_tag "$tag" --log_path "$run_dir/test.log" \
    | tee "$run_dir/test.log"
done
