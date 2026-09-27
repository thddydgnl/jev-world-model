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

from env.worlds import (LEVERS, TEST3_VOCAB, TRAP_VOCAB, build_trap_world,
                        build_trap_world_v2, vocab_split, world_fingerprint)

COUNTS = {"dev": 12, "train": 150, "val": 20, "test": 24}
OUT = Path("kiis2026f/worlds_manifest.json")

# v2 (kiis2026f/실험계획.md §4 V2): the calibrated task (T3) on every split, and
# X1 — the test names on a structure no model was trained on (T1+T2+T3).
V2_LEVERS = ("T3",)
V2_SETS = {"dev": ("dev", 12, V2_LEVERS), "train": ("train", 150, V2_LEVERS),
           "val": ("val", 20, V2_LEVERS), "test": ("test", 24, V2_LEVERS),
           "x1": ("test", 24, LEVERS),
           "x1dev": ("dev", 12, LEVERS)}          # X1 structure on dev names, smoke only
OUT_V2 = Path("kiis2026f/worlds_manifest_v2.json")
# v3 (kiis2026f/실험계획.md §4 W): no task lever (the v1 structure), the v2 test
# names moved to calibration (cal_test), and a new test vocabulary (test3) for
# the test and X1 sets.
V3_SETS = {"dev": ("dev", 12, ()), "train": ("train", 150, ()), "val": ("val", 20, ()),
           "cal_test": ("test", 24, ()),
           "test": ("test3", 24, ()),
           "x1": ("test3", 24, LEVERS),
           "x1dev": ("dev", 12, LEVERS)}
OUT_V3 = Path("kiis2026f/worlds_manifest_v3.json")
NAME_KEYS = ("box", "key", "table", "door", "apple", "pear", "room_a", "room_b",
             "box2", "key2", "door2", "room_c")


def main_v2(version: str = "v2") -> int:
    sets, out = (V2_SETS, OUT_V2) if version == "v2" else (V3_SETS, OUT_V3)
    manifest = {"note": f"written by scripts/kiis_worlds.py --{version}; do not edit by hand",
                "vocab_split": {slot: vocab_split(slot) for slot in TRAP_VOCAB},
                "sets": {}}
    if version == "v3":
        manifest["test3_vocab"] = TEST3_VOCAB
    names_by_set: dict[str, set[str]] = {}
    with tempfile.TemporaryDirectory() as d:
        for name, (split, n, levers) in sets.items():
            rows, seen = [], set()
            for i in range(n):
                game, _, meta = build_trap_world_v2(i, Path(d), split, levers)
                names = {k: meta[k] for k in NAME_KEYS if meta.get(k)}
                seen.update(names.values())
                rows.append({"world_id": meta["world_id"], "names": names,
                             "fingerprint": world_fingerprint(game)})
            manifest["sets"][name] = {"split": split, "levers": list(levers), "worlds": rows}
            names_by_set[name] = seen
            print(f"{name:5s} {split:5s} levers={','.join(levers):9s} {n:4d} worlds, "
                  f"{len(seen):3d} distinct names")
    seen_in_training = names_by_set["train"] | names_by_set["val"]
    for name in ("test", "x1"):
        leak = names_by_set[name] & seen_in_training
        if leak:
            print(f"{name} names also used in train/val: {sorted(leak)}")
            return 1
    if version == "v3":
        # stricter: no WORD of a v3 test name occurs in any other set's names
        others = set().union(*(v for k, v in names_by_set.items() if k not in ("test", "x1")))
        words = {w for n in others for w in n.split()} - {"key", "door"}
        for name in ("test", "x1"):
            shared = {w for n in names_by_set[name] for w in n.split()} & words
            if shared:
                print(f"{name} shares words with other sets: {sorted(shared)}")
                return 1
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"no test or X1 name in train/val -> {out}")
    return 0


def main() -> int:
    if "--v2" in sys.argv[1:]:
        return main_v2("v2")
    if "--v3" in sys.argv[1:]:
        return main_v2("v3")
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
