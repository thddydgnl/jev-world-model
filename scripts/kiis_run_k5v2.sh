#!/bin/bash
# K5v2 and X1 on the GPU server (kiis2026f/실험계획.md §4 V5, §9 X1). Each world
# set's rollouts are built once, then every world model evaluates the same file.
#   v2:test  the frozen v2 task on the test names, 400 starts x 2 sequences
#   v2:x1    the same test names on a structure no model was trained on, 200 x 2
# GPU0: D0 then B0, GPU1: D then B, each on both sets; JEV (no GPU) alongside,
# with its two repeat passes on v2:test. Every model also runs teacher-forced
# (every step from the true previous state) and the typed Qwen with beam 1, for
# H8 (error accumulation). Refuses to overwrite an existing run.
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

ev() {   # ev <gpu or empty> <run name> <rollouts> <out> <eval args...>
  local gpu=$1 name=$2 ro=$3 out=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu $PY eval --rollouts $ro --out $out "$@" > $out/$name.log 2>&1
}
both() { # both <gpu or empty> <run name> <eval args...>: the v2 set, then X1
  local gpu=$1 name=$2; shift 2
  ev "$gpu" $name $RO $O "$@"; ev "$gpu" $name $RX $X "$@"
}
# free-running (fr), teacher-forced (tf, H8) and, for the typed Qwen, beam 1 (H8 control)
( both 0 D0_gen --model D0_gen; both 0 D0_gen.tf --model D0_gen --mode tf
  both 0 B0_typed --model B0_typed; both 0 B0_typed.tf --model B0_typed --mode tf
  both 0 B0_typed.beam1 --model B0_typed --beam 1 ) &
( both 1 D_gen --model D_gen --adapter $ADAPTERS/D_gen
  both 1 D_gen.tf --model D_gen --adapter $ADAPTERS/D_gen --mode tf
  both 1 B_typed --model B_typed --adapter $ADAPTERS/B_typed
  both 1 B_typed.tf --model B_typed --adapter $ADAPTERS/B_typed --mode tf
  both 1 B_typed.beam1 --model B_typed --adapter $ADAPTERS/B_typed --beam 1 ) &
( ev "" A_jev $RO $O --model A_jev --budget 2
  ev "" A_jev.rep1 $RO $O --model A_jev --budget 0.5 --repeat 1
  ev "" A_jev.rep2 $RO $O --model A_jev --budget 0.5 --repeat 2
  ev "" A_jev $RX $X --model A_jev --budget 1
  both "" A_jev.tf --model A_jev --mode tf --budget 1 ) &
wait

$PY report --rollouts $RO --out $O > $O/report.log 2>&1
$PY report --rollouts $RX --out $X > $X/report.log 2>&1
$PY compare --base $O --x1 $X --out $X/compare_h7.json > $X/compare_h7.log 2>&1
$PY accumulation --out $O > $O/accumulation_h8.log 2>&1
$PY accumulation --out $X > $X/accumulation_h8.log 2>&1
n=$(ls $O/*.manifest.json 2>/dev/null | wc -l); m=$(ls $X/*.manifest.json 2>/dev/null | wc -l)
if [ "$n" = "14" ] && [ "$m" = "12" ]; then
  echo complete > $O/DONE; echo complete > $X/DONE
else
  echo "manifests: k5v2 $n of 14, x1 $m of 12" | tee $O/FAILED > $X/FAILED
fi
