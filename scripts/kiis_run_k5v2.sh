#!/bin/bash
# K5v2 and X1 on the GPU server (kiis2026f/실험계획.md §4 V5, §9 X1). Each world
# set's rollouts are built once, then every world model evaluates the same file.
#   v2:test  the frozen v2 task on the test names, 400 starts x 2 sequences
#   v2:x1    the same test names on a structure no model was trained on, 200 x 2
# GPU0: D0 then B0, GPU1: D then B, each on both sets; JEV (no GPU) alongside,
# with its two repeat passes on v2:test. Refuses to overwrite an existing run.
#
# SMOKE=1 builds from the dev-name sets instead (v2:dev, v2:x1dev), 1 world x 2
# starts, and ADAPTERS may point at other adapters for a pipeline check.
set -uo pipefail
cd ~/jev-wm
export HF_HOME=~/hf
PY=".venv/bin/python -u scripts/kiis_k5_rollout.py"
ADAPTERS=${ADAPTERS:-artifacts/kiis_k3v2}
if [ "${SMOKE:-0}" = "1" ]; then
  MAIN_SET=v2:dev; X1_SET=v2:x1dev; TAG=_smoke
  MAIN_BUILD="--worlds 1 --starts 2"; X1_BUILD="--worlds 1 --starts 2"
else
  MAIN_SET=v2:test; X1_SET=v2:x1; TAG=""
  MAIN_BUILD="--starts-total 400"; X1_BUILD="--starts-total 200"
fi
O=artifacts/kiis_k5v2$TAG; X=artifacts/kiis_x1$TAG
RO=data/kiis/k5v2_rollouts$TAG.json; RX=data/kiis/x1_rollouts$TAG.json
for p in $O $X $RO $RX; do
  if [ -e "$p" ]; then echo "refusing to overwrite $p"; exit 1; fi
done
mkdir -p $O $X

CUDA_VISIBLE_DEVICES=0 $PY build --world-set $MAIN_SET $MAIN_BUILD --out $RO > $O/build.log 2>&1 &
b0=$!
CUDA_VISIBLE_DEVICES=1 $PY build --world-set $X1_SET $X1_BUILD --out $RX > $X/build.log 2>&1 &
b1=$!
wait $b0 || { echo "build $MAIN_SET failed" > $O/FAILED; exit 1; }
wait $b1 || { echo "build $X1_SET failed" > $X/FAILED; exit 1; }

ev() {   # ev <gpu or empty> <model> <rollouts> <out> [extra args]
  local gpu=$1 model=$2 ro=$3 out=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu $PY eval --model $model --rollouts $ro --out $out "$@" \
    > $out/$model$(echo "$@" | grep -o 'repeat [0-9]' | sed 's/repeat /.rep/').log 2>&1
}
( ev 0 D0_gen $RO $O; ev 0 D0_gen $RX $X; ev 0 B0_typed $RO $O; ev 0 B0_typed $RX $X ) &
( ev 1 D_gen $RO $O --adapter $ADAPTERS/D_gen; ev 1 D_gen $RX $X --adapter $ADAPTERS/D_gen
  ev 1 B_typed $RO $O --adapter $ADAPTERS/B_typed; ev 1 B_typed $RX $X --adapter $ADAPTERS/B_typed ) &
( ev "" A_jev $RO $O --budget 2; ev "" A_jev $RO $O --budget 0.5 --repeat 1
  ev "" A_jev $RO $O --budget 0.5 --repeat 2; ev "" A_jev $RX $X --budget 1 ) &
wait

$PY report --rollouts $RO --out $O > $O/report.log 2>&1
$PY report --rollouts $RX --out $X > $X/report.log 2>&1
$PY compare --base $O --x1 $X --out $X/compare_h7.json > $X/compare_h7.log 2>&1
n=$(ls $O/*.manifest.json 2>/dev/null | wc -l); m=$(ls $X/*.manifest.json 2>/dev/null | wc -l)
if [ "$n" = "7" ] && [ "$m" = "5" ]; then
  echo complete > $O/DONE; echo complete > $X/DONE
else
  echo "manifests: k5v2 $n of 7, x1 $m of 5" | tee $O/FAILED > $X/FAILED
fi
