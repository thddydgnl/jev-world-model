#!/bin/bash
# W2 checks on the GPU server (kiis2026f/실험계획.md §4 W2), started by hand
# once V3 (kiis_run_k4.py) and F3 (kiis_k5_rollout.py) hold the calibration's
# choice and the code is synced. Not chained to anything; K4v3 is started
# separately after these are read.
#  1. reuse check of the v1 K3 adapters on v3 val (kiis_validate_wm.py --val):
#     B_typed on GPU0 and D_gen on GPU1 side by side
#  2. K4v3 dev smoke, 9 arms x 1 episode (kiis_run_k4.py --version v3 --mode smoke)
#  3. K5v3 and X1 dev smoke (SMOKE=1 kiis_run_k5v3.sh)
# Needs both GPUs free. Writes artifacts/kiis_k3/reuse_v3/ and the smoke dirs.
set -uo pipefail
cd ~/jev-wm
export HF_HOME=~/hf
V=artifacts/kiis_k3/reuse_v3
if [ -e $V ]; then echo "refusing to overwrite $V"; exit 1; fi
mkdir -p $V
date -u +%FT%TZ > $V/started

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/kiis_validate_wm.py --adapter artifacts/kiis_k3/B_typed \
  --val data/kiis/transitions_v3/val.jsonl --out $V/B_typed.json > $V/B_typed.log 2>&1 &
b=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/kiis_validate_wm.py --adapter artifacts/kiis_k3/D_gen \
  --val data/kiis/transitions_v3/val.jsonl --out $V/D_gen.json > $V/D_gen.log 2>&1 &
d=$!
wait $b; echo $? > $V/B_typed.exit
wait $d; echo $? > $V/D_gen.exit

.venv/bin/python -u scripts/kiis_run_k4.py --version v3 --mode smoke > $V/k4v3_smoke.log 2>&1
echo $? > $V/k4v3_smoke.exit
SMOKE=1 bash scripts/kiis_run_k5v3.sh > $V/k5v3_smoke.log 2>&1
echo $? > $V/k5v3_smoke.exit

.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

V = Path("artifacts/kiis_k3/reuse_v3")
out = {"adapters": {}, "exits": {}}
for arm in ("B_typed", "D_gen"):
    r = json.loads((V / f"{arm}.json").read_text()) if (V / f"{arm}.json").exists() else {}
    res = r.get("results", {})
    out["adapters"][arm] = {
        "base_acc": res.get("base", {}).get("inference", {}).get("transition_acc"),
        "selected_acc": res.get("selected", {}).get("inference", {}).get("transition_acc"),
        "selected_weights": res.get("selected", {}).get("weights_sha256"),
        "gates": r.get("gates"),
        "files": {str(p.relative_to("artifacts/kiis_k3")): sha(p)
                  for p in sorted(Path(f"artifacts/kiis_k3/{arm}").rglob("*")) if p.is_file()}}
for name in ("B_typed", "D_gen", "k4v3_smoke", "k5v3_smoke"):
    f = V / f"{name}.exit"
    out["exits"][name] = int(f.read_text()) if f.exists() else None
out["val_sha256"] = sha("data/kiis/transitions_v3/val.jsonl")
(V / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "adapters"}))
for arm, a in out["adapters"].items():
    print(arm, a["base_acc"], "->", a["selected_acc"], a["gates"])
PY
date -u +%FT%TZ > $V/finished
