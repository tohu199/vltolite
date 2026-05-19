#!/usr/bin/env bash
# Run all four loss ablation settings sequentially.
# Usage (from repo root):
#   bash src/train_ablation.sh
#   bash src/train_ablation.sh data.attributes=6_StanfordCars seed=42

set -euo pipefail

cd "$(dirname "$0")/.."

EXPERIMENTS=(
  ablation_cls_only
  ablation_cls_vis
  ablation_cls_txt
  ablation_full
)

for exp in "${EXPERIMENTS[@]}"; do
  echo "=== experiment=${exp} ==="
  python src/train.py "experiment=${exp}" "$@"
done
