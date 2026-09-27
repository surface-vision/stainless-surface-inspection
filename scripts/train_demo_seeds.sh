#!/bin/zsh
# Train the three demo seeds one after another (parallel runs contend for the GPU and are ~2x slower overall).
cd "$(dirname "$0")/.."
for s in 1 2 3; do
  mkdir -p models/demo_seed$s
  .venv/bin/python scripts/train_demo_seeds.py --seed $s --epochs 120 --workers 8 > models/demo_seed$s/train.log 2>&1
done
echo "all seeds done" > models/demo_seeds_done.txt
