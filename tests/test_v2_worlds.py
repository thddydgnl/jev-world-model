"""v2 worlds (kiis2026f/실험계획.md §4 V0) keep every property the v1 pipeline
relies on, for every combination of levers:

1. the v2 solution path runs command by command and reaches the goal;
2. the state schema still covers everything mutable: rebuilding a state from
   the variable readings alone equals the engine's state after random actions
   (the round trip recursion needs), and every true value is an answer option;
3. the levers do what they claim: the decoy key opens nothing, the side room is
   reachable through its own door, the goal food starts in the second container;
4. names: v2 with no levers is the v1 world; within a world no word is shared
   except the head nouns key/door; test names come from the test vocabulary;
   no v2 test world repeats a v1 test world's names.
"""
import itertools, json, random, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import textworld
from textworld import EnvInfos
from agent.task import (goal_satisfied, project_state, read_schema_values,
                        state_schema, static_catalog, trap_goal_spec)
from env.serialize import canonical_state
from env.transitions import solution_path_v2
from env.worlds import (LEVERS, SHARED_HEADS, build_trap_world, build_trap_world_v2,
                        vocab_split, world_fingerprint)
from test_recursive_roundtrip import read_vars

INFOS = EnvInfos(facts=True, admissible_commands=True, possible_admissible_commands=True)
COMBOS = [c for r in range(len(LEVERS) + 1) for c in itertools.combinations(LEVERS, r)]


def main() -> int:
    d = Path(tempfile.mkdtemp())
    rng = random.Random(7)
    fails, checks = [], 0

    def check(ok, msg):
        nonlocal checks
        checks += 1
        if not ok:
            fails.append(msg)

    # v2 with no levers is the v1 world
    for i in range(4):
        g1, _, _ = build_trap_world(i, d, random.Random(0), split="dev")
        g2, _, _ = build_trap_world_v2(i, d, "dev", ())
        check(world_fingerprint(g1) == world_fingerprint(g2), f"dev {i}: v2() != v1")

    for split, levers, i in itertools.product(("dev", "test"), COMBOS, range(3)):
        tag = f"{split}/{i}/{','.join(levers) or '-'}"
        game, path, meta = build_trap_world_v2(i, d, split, levers)
        schema = state_schema(game, meta)
        goal = trap_goal_spec(game, meta)
        catalog = static_catalog(game)
        env = textworld.start(str(path), request_infos=INFOS)
        env.reset()

        # 1. solution path
        e = env.copy()
        for a in solution_path_v2(meta, random.Random(i)):
            check(a in e.state["admissible_commands"], f"{tag}: solution step not admissible: {a}")
            e.step(a)
        check(goal_satisfied(list(e.state["_facts"]), goal), f"{tag}: solution misses goal")
        e.close()

        # 3. levers
        adm0 = set(env.state["admissible_commands"])
        if "T2" in levers:
            e = env.copy()
            e.step(f"take {meta['key2']} from {meta['table']}")
            adm = set(e.state["admissible_commands"])
            check(f"unlock {meta['door']} with {meta['key2']}" not in adm,
                  f"{tag}: decoy key unlocks the door")
            check(any(meta["key2"] in c and "unlock" in c for c in catalog),
                  f"{tag}: decoy unlock not even in the catalog")
            e.close()
        if "T1" in levers:
            check(f"open {meta['door2']}" in adm0, f"{tag}: side door not openable")
            e = env.copy()
            e.step(f"open {meta['door2']}")
            e.step(f"go {meta['side']}")
            room = next(f.arguments[1].name for f in e.state["_facts"]
                        if f.name == "at" and f.arguments[0].name == "P")
            names = {v: info.name for v, info in game.infos.items()}
            check(names.get(room) == meta["room_c"], f"{tag}: side exit leads to {names.get(room)}")
            e.close()
        if "T3" in levers:
            check(f"take {meta['apple']} from {meta['table']}" not in catalog or
                  f"take {meta['apple']} from {meta['table']}" not in adm0,
                  f"{tag}: goal food still on the table")

        # 2. round trip and support on random states and actions
        for _ in range(8):
            root = env.copy()
            for _ in range(rng.randint(0, 6)):
                adm = [c for c in root.state["admissible_commands"]
                       if not c.startswith(("look", "inventory", "examine"))]
                if adm:
                    root.step(rng.choice(adm))
            canon = canonical_state(list(root.state["_facts"]), game)
            for a in rng.sample(catalog, 4):
                b = root.copy(); b.step(a)
                facts = list(b.state["_facts"])
                truth = canonical_state(facts, game)
                vals = read_vars(facts, schema)
                rebuilt = project_state(canon, vals, schema)
                check(rebuilt["dynamic_facts"] == truth["dynamic_facts"],
                      f"{tag}: round trip after {a!r}")
                check(read_schema_values(truth, schema) == vals, f"{tag}: readers disagree")
                for q in schema:
                    check(vals[q["id"]] in q["options"], f"{tag}: {q['id']}={vals[q['id']]} outside options")
                b.close()
            root.close()
        env.close()

        # 4. names within the world
        names = [meta[k] for k in meta if k in ("box", "key", "table", "door", "apple", "pear",
                                                 "room_a", "room_b", "box2", "key2", "door2",
                                                 "room_c") and meta[k]]
        toks = [t for n in names for t in n.split() if t not in SHARED_HEADS]
        check(len(toks) == len(set(toks)), f"{tag}: shared word in {names}")
        if split == "test":
            test_words = {w for slot in ("box", "key", "table", "door", "apple", "pear", "room")
                          for w in vocab_split(slot)["test"]}
            check(all(n in test_words for n in names), f"{tag}: non-test name in {names}")

    # 4. v2 test worlds do not repeat v1 test worlds
    ref = json.loads((ROOT / "kiis2026f/worlds_manifest.json").read_text())["worlds"]["test"]
    v1 = {tuple(r["names"]) for r in ref}
    for i in range(24):
        _, _, meta = build_trap_world_v2(i, d, "test", LEVERS)
        base = tuple(meta[k] for k in ("box", "key", "table", "door", "apple", "pear",
                                       "room_a", "room_b"))
        check(base not in v1, f"test {i}: same names as a v1 test world")

    print(f"{checks - len(fails)}/{checks} checks passed")
    for f in fails[:20]:
        print("  FAIL", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
