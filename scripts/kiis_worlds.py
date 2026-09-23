"""Fix the KIIS worlds before any result exists (kiis2026f/실험계획.md §3).

Builds every world of every split once and writes, for each, its names and its
fingerprint — the hash of its initial facts and entities (see
env.worlds.world_fingerprint for why not a file hash). The file is committed
before the main runs, so K6 can check that each run's run_manifest.json used
exactly these worlds, and anyone can check that no test name appears in train.
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from env.worlds import TRAP_VOCAB, build_trap_world, vocab_split, world_fingerprint

COUNTS = {"dev": 12, "train": 150, "val": 20, "test": 24}
OUT = Path("kiis2026f/worlds_manifest.json")


def main() -> int:
    manifest = {"note": "written by scripts/kiis_worlds.py; do not edit by hand",
                "vocab_split": {slot: vocab_split(slot) for slot in TRAP_VOCAB},
                "worlds": {}}
    names_by_split: dict[str, set[str]] = {}
    with tempfile.TemporaryDirectory() as d:
        for split, n in COUNTS.items():
            rows, seen = [], set()
            for i in range(n):
                game, _, meta = build_trap_world(i, Path(d), random.Random(0), split=split)
                names = [meta[k] for k in ("box", "key", "table", "door",
                                           "apple", "pear", "room_a", "room_b")]
                seen.update(names)
                rows.append({"world_id": meta["world_id"], "names": names,
                             "fingerprint": world_fingerprint(game)})
            manifest["worlds"][split] = rows
            names_by_split[split] = seen
            print(f"{split:5s} {n:4d} worlds, {len(seen):3d} distinct names")

    leak = names_by_split["test"] & (names_by_split["train"] | names_by_split["val"])
    if leak:
        print(f"test names also used in train/val: {sorted(leak)}")
        return 1
    OUT.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"no test name in train/val -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
