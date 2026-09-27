#!/bin/zsh
# Waits for the seed chain to finish, then runs validation-only selection and the test scorecard.
cd "$(dirname "$0")/.."
while [ ! -f models/demo_seeds_done.txt ]; do sleep 120; done
.venv/bin/python scripts/select_demo_seed.py 2>&1 | grep -vE "Ultralytics|Model summary|Fusing|Class +Images|all +[0-9]|Speed|Results saved|^\s*$|WARNING"
