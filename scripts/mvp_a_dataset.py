"""MVP-A: build the labelled counterfactual dataset and audit it (설계.md §25).

Gate conditions, all of which must hold before MVP-B means anything:
  - target support   : GT inside the option set 100% of the time
  - state sufficiency: identical (canonical state, action prefix) -> identical endpoint
  - contrast supply  : enough pairs where two sequences have DIFFERENT ground truth
"""
from __future__ import annotations

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

N_WORLDS = 8
ROOTS_PER_WORLD = 4
SEED = 20260921
OUT = Path("data/mvp_a")


def sequences(meta: dict) -> list[tuple[str, list[str]]]:
    box, key, door, table = meta["box"], meta["key"], meta["door"], meta["table"]
    dis = meta["distractor"]
    return [
        ("take_distractor", [f"take {dis}", "go east"]),
        ("open_then_take", [f"open {box}", f"take {key} from {box}"]),
        ("take_then_open", [f"take {key} from {box}", f"open {box}"]),   # order swap
        ("take_then_unlock", [f"take {key} from {box}", f"unlock {door} with {key}"]),
        ("null_examine", [f"examine {table}", f"examine {box}"]),
        ("open_then_go", [f"open {box}", "go east"]),
    ]


def main() -> int:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    worlds_dir = OUT / "worlds"; worlds_dir.mkdir(exist_ok=True)

    records, support_fail, sufficiency = [], 0, {}
    n_prefix = 0

    for w in range(N_WORLDS):
        game, path, meta = build_world(w, worlds_dir, rng)
        catalog = build_query_catalog(game, meta)
        env = textworld.start(str(path), request_infos=INFOS)
        env.reset()

        for r in range(ROOTS_PER_WORLD):
            # --- collect a root by applying 0..3 random admissible commands
            root_env = env.copy()
            n_steps = rng.randint(0, 3)
            root_prefix = []
            for _ in range(n_steps):
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

            for seq_name, actions in sequences(meta):
                for h in (1, 2):
                    branch = root_env.copy()
                    for a in actions[:h]:
                        branch.step(a)
                    end_facts = list(branch.state["_facts"])
                    branch.close()
                    n_prefix += 1

                    labels, changed = {}, {}
                    for q in catalog:
                        y = label(end_facts, q)
                        if y not in q["options"]:
                            support_fail += 1
                        labels[q["id"]] = y
                        changed[q["id"]] = (y != root_labels[q["id"]])

                    # state sufficiency: same canonical state + same prefix -> same endpoint
                    suff_key = (state_hash(root_state), tuple(actions[:h]))
                    end_hash = facts_hash(end_facts)
                    sufficiency.setdefault(suff_key, set()).add(end_hash)

                    records.append({
                        "world_id": meta["world_id"], "root_id": root_id,
                        "root_prefix": root_prefix, "seq_name": seq_name,
                        "actions": actions[:h], "horizon": h,
                        "current_state": root_state, "root_labels": root_labels,
                        "labels": labels, "changed": changed,
                        "meta": meta,
                        "catalog": [{"id": q["id"], "label": q["label"], "kind": q["kind"],
                                     "options": q["options"]} for q in catalog],
                    })
            root_env.close()
        env.close()

    (OUT / "prefixes.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records))

    # ------------------------------------------------------------ audit
    collisions = sum(1 for v in sufficiency.values() if len(v) > 1)
    changed_counts = Counter()
    for rec in records:
        for qid, ch in rec["changed"].items():
            changed_counts[(qid, ch)] += 1

    # contrastive pairs: same root+horizon+query, two sequences with different GT
    by_key = defaultdict(list)
    for rec in records:
        for qid, y in rec["labels"].items():
            by_key[(rec["root_id"], rec["horizon"], qid)].append((rec["seq_name"], y))
    diff_pairs = null_pairs = 0
    for entries in by_key.values():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if entries[i][1] != entries[j][1]:
                    diff_pairs += 1
                else:
                    null_pairs += 1

    print(f"worlds={N_WORLDS}  roots={N_WORLDS*ROOTS_PER_WORLD}  prefixes={n_prefix}  "
          f"labels={n_prefix*5}")
    print(f"\n[gate] target support 위반 : {support_fail}  "
          f"({'PASS' if support_fail == 0 else 'FAIL'})")
    print(f"[gate] state 충돌          : {collisions}  "
          f"({'PASS' if collisions == 0 else 'FAIL'})")
    print(f"[gate] 대조 pair(정답 다름) : {diff_pairs}  "
          f"({'PASS' if diff_pairs >= 50 else 'FAIL — 너무 적음'})")
    print(f"       null pair(정답 같음) : {null_pairs}")

    print("\n질문별 변화 비율 (changed / total):")
    for q in ("q_key_parent", "q_box_mode", "q_door_mode",
              "q_distractor_parent", "q_player_room"):
        ch, un = changed_counts[(q, True)], changed_counts[(q, False)]
        print(f"  {q:22s} {ch:4d}/{ch+un:4d}  ({ch/(ch+un)*100:5.1f}%)")

    ok = support_fail == 0 and collisions == 0 and diff_pairs >= 50
    print(f"\n{'='*58}\nMVP-A {'PASS' if ok else 'FAIL'} -> data/mvp_a/prefixes.jsonl")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
