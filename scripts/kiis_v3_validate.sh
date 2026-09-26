#!/bin/bash
# V3 (kiis2026f/실험계획.md §4 V3): validate one retrained adapter on the GPU server.
#   bash scripts/kiis_v3_validate.sh <B_typed|D_gen> <gpu>
# Started by hand after that model's training has finished (not chained).
#  1. 1-step validation on the same 200 val transitions the training selected
#     its epoch on (kiis_validate_wm.py; base vs epochs vs selected, loss, gates)
#  2. a dev smoke under condition F2 with the selected adapter; with D_gen also
#     the B0_typed smoke that anchors the step-0 candidate comparison
#  3. SHA-256 of the validation inputs (source_hashes.json) and of every adapter
#     file to be copied home (sync_manifest.json), read by
#     scripts/kiis_check_k3.py --version v2
set -uo pipefail
cd ~/jev-wm
export HF_HOME=~/hf
arm=$1; gpu=$2
B=artifacts/kiis_k3v2; V=$B/validation
mkdir -p $V
F2="--trap --v2 --levers T3 --policy-hint-carry-key --policy-samples 2 --split dev --worlds 1 --roots 1 --cap 30 --policy-seed 20260921"

CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u scripts/kiis_validate_wm.py --adapter $B/$arm \
  --out $V/$arm.json > $V/$arm.log 2>&1
echo $? > $V/$arm.exit
CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u scripts/mvp_c_headroom.py $F2 --arms $arm \
  --adapter $B/$arm --out $V/smoke_$arm > $V/smoke_$arm.log 2>&1
if [ "$arm" = "D_gen" ]; then
  CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python -u scripts/mvp_c_headroom.py $F2 --arms B0_typed \
    --out $V/smoke_B0_typed > $V/smoke_B0_typed.log 2>&1
fi

.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

base = Path("artifacts/kiis_k3v2")
inputs = ["scripts/kiis_train_wm.py", "scripts/kiis_validate_wm.py", "scripts/mvp_c_headroom.py",
          "src/runinfo.py", "src/agent/policy.py", "src/wm/llm_backends.py",
          "src/wm/recursive_forecaster.py", "data/kiis/transitions_v2/train.jsonl",
          "data/kiis/transitions_v2/val.jsonl"]
(base / "validation" / "source_hashes.json").write_text(
    json.dumps({p: sha(p) for p in inputs}, indent=2) + "\n")
files = sorted(p for arm in ("B_typed", "D_gen") if (base / arm).is_dir()
               for p in (base / arm).rglob("*") if p.is_file())
(base / "sync_manifest.json").write_text(json.dumps(
    {"source": "kiis-mvf-gpu:~/jev-wm/artifacts/kiis_k3v2",
     "files": {str(p.relative_to(base)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in files}},
    indent=2) + "\n")
print(f"hashed {len(inputs)} inputs, {len(files)} adapter files")
PY
echo done > $V/$arm.DONE
