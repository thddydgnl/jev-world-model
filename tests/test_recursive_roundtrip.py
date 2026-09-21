"""The reconstruction must round-trip, or recursion cannot work.

Take a real state, apply an action in the engine, read every state variable
with the labeler, and rebuild a state from those readings alone. The rebuilt
state must equal the engine's own canonical state. If it does not, feeding a
prediction back in would drift for reasons that have nothing to do with JEV.

Also checks the perfect-forecaster reduction: a recursive rollout driven by
engine truth must reach the same endpoint as executing the prefix directly.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import EnvInfos
from agent.task import (state_schema, project_state, goal_from_rows,
                        trap_goal_spec, static_catalog, goal_satisfied)
from env.serialize import canonical_state
from env.worlds import build_trap_world

INFOS = EnvInfos(facts=True, admissible_commands=True, possible_admissible_commands=True)


def read_vars(facts, schema):
    """Ground-truth value of each schema variable, from engine facts."""
    rows = {tuple([f.name] + [a.name for a in f.arguments]) for f in facts}
    out = {}
    for q in schema:
        if q["kind"] == "room":
            out[q["id"]] = next(r[2] for r in rows if r[0] == "at" and r[1] == "P")
        elif q["kind"] == "parent":
            o = q["target"]
            v = None
            if ("eaten", o) in rows:
                v = "gone"
            else:
                for r in rows:
                    if len(r) == 3 and r[1] == o and r[0] in ("in", "on", "at"):
                        v = "inventory" if r[2] == "I" else f"{r[0]}:{r[2]}"
            out[q["id"]] = v
        else:
            x = q["target"]
            for pred, val in (("open", "open"), ("closed", "closed_unlocked"),
                              ("locked", "locked")):
                if (pred, x) in rows:
                    out[q["id"]] = val
    return out


def main():
    rng = random.Random(11)
    d = Path("/tmp/rt"); d.mkdir(exist_ok=True)
    rt_bad = roll_bad = checked = 0

    for w in range(4):
        game, path, meta = build_trap_world(w, d, rng)
        schema = state_schema(game, meta)
        goal = trap_goal_spec(game, meta)
        catalog = static_catalog(game)
        env = textworld.start(str(path), request_infos=INFOS)
        env.reset()

        for trial in range(6):
            root = env.copy()
            for _ in range(rng.randint(0, 4)):
                adm = [c for c in root.state["admissible_commands"]
                       if not c.startswith(("look", "inventory", "examine"))]
                if adm:
                    root.step(rng.choice(adm))

            canon = canonical_state(list(root.state["_facts"]), game)
            actions = [rng.choice(catalog) for _ in range(2)]

            # --- 1. round trip of one step
            b = root.copy(); b.step(actions[0])
            truth = canonical_state(list(b.state["_facts"]), game)
            rebuilt = project_state(canon, read_vars(list(b.state["_facts"]), schema), schema)
            checked += 1
            if rebuilt["dynamic_facts"] != truth["dynamic_facts"]:
                rt_bad += 1
                if rt_bad <= 2:
                    only_t = [r for r in truth["dynamic_facts"] if r not in rebuilt["dynamic_facts"]]
                    only_r = [r for r in rebuilt["dynamic_facts"] if r not in truth["dynamic_facts"]]
                    print(f"  ROUNDTRIP {meta['world_id']} '{actions[0]}'")
                    print(f"    엔진에만: {only_t}")
                    print(f"    재구성에만: {only_r}")
            b.close()

            # --- 2. two-step recursion vs direct execution
            cur, e = canon, root.copy()
            for a in actions:
                e.step(a)
                cur = project_state(cur, read_vars(list(e.state["_facts"]), schema), schema)
            end_truth = canonical_state(list(e.state["_facts"]), game)
            if cur["dynamic_facts"] != end_truth["dynamic_facts"]:
                roll_bad += 1
            # goal scored off the reconstruction must match the engine
            sat_rec, _ = goal_from_rows(cur["dynamic_facts"], goal)
            if sat_rec != goal_satisfied(list(e.state["_facts"]), goal):
                roll_bad += 1
            e.close(); root.close()
        env.close()

    print(f"\n1-step 왕복      : {checked-rt_bad}/{checked} 일치")
    print(f"2-step 재귀+목표 : {checked-roll_bad}/{checked} 일치")
    cov = coverage_check()
    ok = rt_bad == 0 and roll_bad == 0 and cov == 0
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1




def coverage_check():
    """The carry-forward assumption is only sound if every fact the schema does
    NOT cover is in fact immutable. Check that against many real transitions."""
    from agent.task import schema_covers
    rng = random.Random(23)
    d = Path("/tmp/rt"); d.mkdir(exist_ok=True)
    violations, transitions = [], 0
    for w in range(4):
        game, path, meta = build_trap_world(w, d, rng)
        schema = state_schema(game, meta)
        catalog = static_catalog(game)
        env = textworld.start(str(path), request_infos=INFOS); env.reset()
        for _ in range(40):
            b = env.copy()
            for _ in range(rng.randint(0, 5)):
                b.step(rng.choice(catalog))
            before = canonical_state(list(b.state["_facts"]), game)["dynamic_facts"]
            b.step(rng.choice(catalog))
            after = canonical_state(list(b.state["_facts"]), game)["dynamic_facts"]
            b.close()
            transitions += 1
            un_b = {tuple(r) for r in before if not schema_covers(schema, list(r))}
            un_a = {tuple(r) for r in after if not schema_covers(schema, list(r))}
            if un_b != un_a:
                violations.append((meta["world_id"], sorted(un_b ^ un_a)))
        env.close()
    print(f"\n스키마 커버리지  : {transitions-len(violations)}/{transitions} "
          f"전이에서 미커버 사실 불변")
    for v in violations[:3]:
        print(f"  위반: {v}")
    return len(violations)


if __name__ == "__main__":
    raise SystemExit(main())
