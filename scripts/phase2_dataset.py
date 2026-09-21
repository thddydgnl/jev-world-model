"""Phase 2: the transition dataset arm B trains on (설계.md §15, §30).

Scales MVP-A up and adds the split that MVP-A did not have: **world-level**
train/validation/test. §15 is explicit that the split is fixed at the world
level first and that different action branches of one world must never straddle
it, because roots from the same world share topology and entity names.

Uses only the engine's ground truth. No JEV output is read, written or used for
checkpoint selection (설계.md §16.1), so this dataset carries no Output.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import EnvInfos

from env.serialize import canonical_state, assert_clean, state_hash, facts_hash
from env.worlds import build_world, build_query_catalog, label

INFOS = EnvInfos(facts=True, typed_entities=True, possible_admissible_commands=True,
                 last_action=True, admissible_commands=True)

# 설계.md §30 Train-small. World ranges are disjoint by construction.
SPLITS = {
    "train": (0, 200),
    "val":   (200, 240),
    "test":  (240, 320),
}
ROOTS_PER_WORLD = 6
HORIZONS = (1, 2, 3)      # §30: h=3 is needed if H=4 is ever evaluated
SEED = 20260921


def sequences(meta: dict, rng: random.Random) -> list[tuple[str, list[str]]]:
    """Mixed collection policy (§15.2): contrastive pairs plus random branches,
    so the model also sees the wrong plans a policy would propose."""
    box, key, door = meta["box"], meta["key"], meta["door"]
    table, dis = meta["table"], meta["distractor"]
    fixed = [
        ("open_then_take", [f"open {box}", f"take {key} from {box}", "go east"]),
        ("take_then_open", [f"take {key} from {box}", f"open {box}", "go east"]),
        ("take_then_unlock", [f"take {key} from {box}",
                              f"unlock {door} with {key}", f"open {door}"]),
        ("null_examine", [f"examine {table}", f"examine {box}", "look"]),
        ("open_then_go", [f"open {box}", "go east", "go west"]),
        ("distractor", [f"take {dis}", f"insert {dis} into {box}", "go east"]),
    ]
    return fixed


def collect(split: str, lo: int, hi: int, out: Path, rng: random.Random) -> dict:
    wdir = out / "worlds" / split
    wdir.mkdir(parents=True, exist_ok=True)
    records, support_fail, sufficiency = [], 0, {}
    world_ids = []

    for w in range(lo, hi):
        game, path, meta = build_world(w, wdir, rng)
        catalog = build_query_catalog(game, meta)
        world_ids.append(meta["world_id"])
        env = textworld.start(str(path), request_infos=INFOS)
        env.reset()

        for r in range(ROOTS_PER_WORLD):
            root_env = env.copy()
            root_prefix = []
            for _ in range(rng.randint(0, 4)):
                adm = [c for c in root_env.state["admissible_commands"]
                       if not c.startswith(("look", "inventory", "examine"))]
                if not adm:
                    break
                c = rng.choice(adm)
                root_env.step(c)
                root_prefix.append(c)

            root_facts = list(root_env.state["_facts"])
            root_state = canonical_state(root_facts, game, attempt_index=len(root_prefix))
            assert_clean(root_state)
            root_id = f"{meta['world_id']}_r{r}"
            root_labels = {q["id"]: label(root_facts, q) for q in catalog}

            # one extra random branch per root, drawn from the static catalog
            static = sorted(set(game.possible_admissible_commands))
            seqs = sequences(meta, rng) + [
                ("random", [rng.choice(static) for _ in range(max(HORIZONS))])]

            for seq_name, actions in seqs:
                for h in HORIZONS:
                    if h > len(actions):
                        continue
                    branch = root_env.copy()
                    n_inv = 0
                    for a in actions[:h]:
                        if a not in branch.state["admissible_commands"]:
                            n_inv += 1
                        branch.step(a)
                    end_facts = list(branch.state["_facts"])
                    branch.close()

                    labels, changed = {}, {}
                    for q in catalog:
                        y = label(end_facts, q)
                        if y not in q["options"]:
                            support_fail += 1
                        labels[q["id"]] = y
                        changed[q["id"]] = (y != root_labels[q["id"]])

                    suff = (state_hash(root_state), tuple(actions[:h]))
                    sufficiency.setdefault(suff, set()).add(facts_hash(end_facts))

                    records.append({
                        "split": split, "world_id": meta["world_id"],
                        "world_group": f"template_{w % 4}", "root_id": root_id,
                        "root_prefix": root_prefix, "seq_name": seq_name,
                        "actions": actions[:h], "horizon": h,
                        "n_invalid": n_inv,
                        "current_state": root_state, "root_labels": root_labels,
                        "labels": labels, "changed": changed, "meta": meta,
                        "catalog": [{"id": q["id"], "label": q["label"],
                                     "kind": q["kind"], "options": q["options"]}
                                    for q in catalog],
                    })
            root_env.close()
        env.close()

    path = out / f"{split}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n")
    collisions = sum(1 for v in sufficiency.values() if len(v) > 1)
    ch = sum(sum(r["changed"].values()) for r in records)
    tot = sum(len(r["changed"]) for r in records)
    print(f"  {split:6s} worlds={hi-lo:4d} roots={(hi-lo)*ROOTS_PER_WORLD:5d} "
          f"prefixes={len(records):6d} labels={tot:7d}  "
          f"changed={ch/tot:5.1%}  support위반={support_fail} 충돌={collisions}")
    return {"split": split, "worlds": world_ids, "n_prefixes": len(records),
            "n_labels": tot, "support_fail": support_fail, "collisions": collisions}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/phase2")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of each split's world range (smoke with 0.05)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    print(f"Phase 2 dataset -> {out}  (scale={args.scale})")
    manifest = []
    for split, (lo, hi) in SPLITS.items():
        n = max(1, int((hi - lo) * args.scale))
        manifest.append(collect(split, lo, lo + n, out, rng))

    ids = {m["split"]: set(m["worlds"]) for m in manifest}
    overlap = [(a, b, ids[a] & ids[b]) for a in ids for b in ids if a < b]
    bad = [(a, b, o) for a, b, o in overlap if o]
    (out / "manifest.json").write_text(json.dumps(
        {"seed": SEED, "roots_per_world": ROOTS_PER_WORLD,
         "horizons": list(HORIZONS), "splits": manifest}, indent=2))

    print(f"\nworld 교차: {'없음' if not bad else bad}")
    ok = not bad and all(m["support_fail"] == 0 and m["collisions"] == 0 for m in manifest)
    print(f"Phase 2 {'PASS' if ok else 'FAIL'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
