"""Write the one-step transition sets for the LLM world models (K2).

    train  150 worlds  -> B and D training (N is drawn from here)
    val     20 worlds  -> checkpoint selection
    test    24 worlds  -> held out; K5 reads its own rollouts, this is for reference

Each world is checked against kiis2026f/worlds_manifest.json first, so these
are the worlds fixed in K0 and nothing else. Output is engine truth only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import EnvInfos

from agent.task import state_schema
from env.transitions import world_transitions
from env.worlds import build_trap_world, world_fingerprint

INFOS = EnvInfos(facts=True, typed_entities=True, possible_admissible_commands=True,
                 admissible_commands=True)
COUNTS = {"train": 150, "val": 20, "test": 24}
OUT = Path("data/kiis/transitions")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of worlds (smoke)")
    args = ap.parse_args()

    ref = json.loads(Path("kiis2026f/worlds_manifest.json").read_text())["worlds"]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"note": "written by scripts/kiis_transitions.py; engine truth only", "splits": {}}
    for split in args.splits.split(","):
        n_worlds = max(1, int(COUNTS[split] * args.scale))
        fp_ref = {r["world_id"]: r["fingerprint"] for r in ref[split]}
        records, kinds, execs, changed = [], Counter(), Counter(), Counter()
        with tempfile.TemporaryDirectory() as d:
            for i in range(n_worlds):
                game, path, meta = build_trap_world(i, Path(d), random.Random(0), split=split)
                if world_fingerprint(game) != fp_ref[meta["world_id"]]:
                    raise SystemExit(f"{meta['world_id']} differs from worlds_manifest.json")
                schema = state_schema(game, meta)
                env = textworld.start(str(path), request_infos=INFOS)
                env.reset()
                rng = random.Random(f"kiis-transitions-{split}-{i}")
                for rec in world_transitions(env, game, meta, schema, rng):
                    rec["split"] = split
                    rec["schema"] = schema
                    records.append(rec)
                    kinds[rec["kind"]] += 1
                    execs[rec["executes"]] += 1
                    changed[len(rec["changed"])] += 1
                env.close()
        path = OUT / f"{split}.jsonl"
        blob = "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records) + "\n"
        path.write_text(blob)
        manifest["splits"][split] = {
            "worlds": n_worlds, "transitions": len(records),
            "sha256": hashlib.sha256(blob.encode()).hexdigest()[:16],
            "kinds": dict(kinds), "executes": {str(k): v for k, v in execs.items()},
            "changed_vars": {str(k): v for k, v in sorted(changed.items())}}
        print(f"{split:5s} worlds={n_worlds:3d} transitions={len(records):6d} "
              f"executes={execs[True]/len(records):.0%} kinds={dict(kinds)} "
              f"changed={dict(sorted(changed.items()))}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
