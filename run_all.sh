#!/usr/bin/env bash
# Reproduces every ABIDE result in notebooks/QTST_ABIDE_demo.ipynb.
# Task: autism (DX_GROUP=1) vs control (DX_GROUP=2), binary classification, AUROC.
# Each job saves a checkpoint after every epoch; re-running this script resumes unfinished seeds.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
COMMON="--data-dir data/nilearn_data --seq-len 116 --qubits 8 --seeds 2024 2025 2026 --epochs 10"
mkdir -p logs

for SETTING in n100 all; do
  if [ "$SETTING" = n100 ]; then NSUB=100; else NSUB=0; fi
  $PY qtst/run_abide.py $COMMON --n-subjects $NSUB --model quantum --threads 8 \
      --out-dir results/$SETTING/quantum > logs/${SETTING}_quantum.log 2>&1 &
  $PY qtst/run_abide.py $COMMON --n-subjects $NSUB --model small_transformer --match-quantum --threads 1 \
      --out-dir results/$SETTING/small_transformer > logs/${SETTING}_small_transformer.log 2>&1 &
  $PY qtst/run_fc_baseline.py --data-dir data/nilearn_data --seq-len 116 --n-subjects $NSUB \
      --seeds 2024 2025 2026 --out-dir results/$SETTING/fc_logreg > logs/${SETTING}_fc_logreg.log 2>&1 &
done
wait
grep -h "^SUMMARY" logs/*.log
