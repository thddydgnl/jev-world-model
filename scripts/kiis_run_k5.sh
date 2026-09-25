#!/bin/bash
# K5 on the GPU server: build the 800 test rollouts once, then every world model
# evaluates the same file. GPU0: D0 then B0; GPU1: D then B; JEV (no GPU) with
# its two repeat passes alongside. Refuses to overwrite an existing run.
set -uo pipefail
cd ~/jev-wm
export HF_HOME=~/hf
PY=".venv/bin/python -u scripts/kiis_k5_rollout.py"
O=artifacts/kiis_k5
R=data/kiis/k5_rollouts.json
if [ -e "$O" ] || [ -e "$R" ]; then echo "refusing to overwrite $O / $R"; exit 1; fi
mkdir -p $O

CUDA_VISIBLE_DEVICES=0 $PY build --split test --out $R > $O/build.log 2>&1 || { echo build failed > $O/FAILED; exit 1; }

( CUDA_VISIBLE_DEVICES=0 $PY eval --model D0_gen --rollouts $R > $O/D0_gen.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY eval --model B0_typed --rollouts $R > $O/B0_typed.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 $PY eval --model D_gen --adapter artifacts/kiis_k3/D_gen --rollouts $R > $O/D_gen.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY eval --model B_typed --adapter artifacts/kiis_k3/B_typed --rollouts $R > $O/B_typed.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES= $PY eval --model A_jev --budget 2 --rollouts $R > $O/A_jev.log 2>&1
  CUDA_VISIBLE_DEVICES= $PY eval --model A_jev --budget 0.5 --repeat 1 --rollouts $R > $O/A_jev.rep1.log 2>&1
  CUDA_VISIBLE_DEVICES= $PY eval --model A_jev --budget 0.5 --repeat 2 --rollouts $R > $O/A_jev.rep2.log 2>&1 ) &
wait

$PY report --rollouts $R > $O/report.log 2>&1
n=$(ls $O/*.manifest.json 2>/dev/null | wc -l)
if [ "$n" = "7" ]; then echo complete > $O/DONE; else echo "only $n of 7 manifests" > $O/FAILED; fi
