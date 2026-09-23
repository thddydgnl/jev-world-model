#!/bin/bash
# Wait for the ablation on GPU 0 to finish, then take that GPU for seed 3.
cd ~/jev-wm
while ! grep -q "=====" ablation.log 2>/dev/null; do sleep 30; done
sleep 10
echo "[queue] ablation done at $(date +%H:%M:%S), starting seed 1234 on GPU0" >> queue.log
HF_HOME=$HOME/hf CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u \
  scripts/mvp_c_headroom.py --trap --cap 30 --arms validity,oracle,A \
  --policy-seed 1234 --worlds 12 --roots 4 --out artifacts/seed1234 \
  > seed1234.log 2>&1
echo "[queue] seed 1234 finished at $(date +%H:%M:%S)" >> queue.log
