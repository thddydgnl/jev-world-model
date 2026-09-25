"""Why do capped episodes fail? Replay K4 episodes from their roots and ask, at
every step, whether the policy's candidate plans offered a useful action.

Descriptive only: reads saved K4 logs, calls no model, changes nothing about the
agents. Roots are rebuilt with the runner's own RNG sequence (worlds and roots do
not depend on the policy seed), and every replayed step must reproduce the
recorded validity, or the script stops.

A "useful" action is one that is admissible now and belongs to the trap-world
solution: open the box (and, in v2 with T3, the second container), take the key,
unlock and open the door, take the goal object, go east — or head back to room
A from the far room (without the object) or from the v2 side room.
A capped episode is then
  trap      the goal object was eaten (no longer solvable)
  coverage  no candidate plan started with a useful action in the last 10 steps
  planner   a useful first action was offered in the last 10 steps, not taken
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import textworld
from textworld import EnvInfos

from agent.task import goal_satisfied, trap_goal_spec
from env.worlds import OPPOSITE, build_trap_world, build_trap_world_v2, world_fingerprint

INFOS = EnvInfos(facts=True, typed_entities=True, possible_admissible_commands=True,
                 last_action=True, admissible_commands=True)
SEED = 20260921          # mvp_c_headroom.SEED: worlds and roots, not the policy
TAIL = 10


def roots(split: str, n_worlds: int, n_roots: int, tmp: Path, fingerprints: dict,
          levers: tuple | None = None):
    """(world_id, root) -> (root env, meta, game), consuming the RNG exactly as
    the runner. `levers` is None for v1 worlds, the v2 levers otherwise."""
    rng = random.Random(SEED)
    out = {}
    for w in range(n_worlds):
        if levers is None:
            game, path, meta = build_trap_world(w, tmp, rng, split=split)
        else:
            game, path, meta = build_trap_world_v2(w, tmp, split, levers)
        goal = trap_goal_spec(game, meta)
        fp = world_fingerprint(game)
        if fingerprints.get(meta["world_id"]) not in (None, fp):
            raise SystemExit(f"{meta['world_id']}: fingerprint {fp} != manifest")
        env = textworld.start(str(path), request_infos=INFOS); env.reset()
        for r in range(n_roots):
            root = env.copy()
            for _ in range(rng.randint(0, 2)):
                adm = [c for c in root.state["admissible_commands"]
                       if not c.startswith(("look", "inventory", "examine"))]
                if not adm:
                    break
                root.step(rng.choice(adm))
            if goal_satisfied(list(root.state["_facts"]), goal):
                root.close(); continue
            out[(meta["world_id"], r)] = (root, meta, game)
    return out


def useful(env, game, meta: dict) -> set[str]:
    adm = env.state["admissible_commands"]
    rid = next(f.arguments[1].name for f in env.state["_facts"]
               if f.name == "at" and f.arguments[0].name == "P")
    room = game.infos[rid].name
    if room == meta["room_b"]:
        wanted = {"go west"}
    elif meta.get("room_c") and room == meta["room_c"]:
        wanted = {f"go {OPPOSITE[meta['side']]}", f"open {meta['door2']}"}
    else:
        wanted = {f"open {meta['box']}", f"unlock {meta['door']} with {meta['key']}",
                  f"open {meta['door']}", "go east"}
        if meta.get("box2"):
            wanted.add(f"open {meta['box2']}")
    return {a for a in adm
            if a in wanted or a.startswith((f"take {meta['key']} ", f"take {meta['apple']} "))
            or a in (f"take {meta['key']}", f"take {meta['apple']}")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories (each with run_manifest/episodes/steps)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    per_ep = []
    cache: dict[tuple, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for run in args.runs:
            run = Path(run)
            man = json.loads((run / "run_manifest.json").read_text())
            r = man["run"]
            v2 = man["condition"].get("v2")
            levers = tuple(v2["levers"]) if v2 else None
            key = (r["split"], r["worlds"], r["roots"], levers)
            if key not in cache:
                d = Path(tmp) / f"{len(cache)}"
                d.mkdir()
                cache[key] = roots(*key[:3], d, man["world_fingerprints"], levers)
            base = cache[key]
            eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
            steps = collections.defaultdict(list)
            for l in open(run / "steps.jsonl"):
                s = json.loads(l)
                steps[(s["arm"], s["world"], s["root"])].append(s)
            for e in eps:
                root, meta, game = base[(e["world_id"], e["root"])]
                env = root.copy()
                offered = []
                for s in steps[(e["arm"], e["world_id"], e["root"])]:
                    adm = env.state["admissible_commands"]
                    if (s["action"] in adm) != s["valid"]:
                        raise SystemExit(f"replay mismatch {run} {e['arm']} "
                                         f"{e['world_id']}/{e['root']} step {s['step']}")
                    u = useful(env, game, meta)
                    firsts = {p[0] for p in s["plans"] if p}
                    offered.append((bool(firsts & u), s["action"] in u))
                    env.step(s["action"])
                env.close()
                tail = offered[-TAIL:]
                if e["success"]:
                    kind = "success"
                elif e["traps"]:
                    kind = "trap"
                elif not any(o for o, _ in tail):
                    kind = "coverage"
                else:
                    kind = "planner"
                per_ep.append({"run": str(run.relative_to(ROOT)) if run.is_absolute() else str(run),
                               "split": r["split"], "seed": r["policy_seed"],
                               "arm": e["arm"], "world": e["world_id"], "root": e["root"],
                               "kind": kind, "steps": len(offered),
                               "steps_offered": sum(o for o, _ in offered),
                               "steps_taken": sum(t for _, t in offered)})
        for root, _, _ in (v for c in cache.values() for v in c.values()):
            root.close()

    summary = collections.defaultdict(collections.Counter)
    offered_rate = collections.defaultdict(lambda: [0, 0])
    for p in per_ep:
        summary[(p["split"], p["arm"])][p["kind"]] += 1
        offered_rate[(p["split"], p["arm"])][0] += p["steps_offered"]
        offered_rate[(p["split"], p["arm"])][1] += p["steps"]
    print(f"{'split':<6}{'arm':<10}{'n':>5}{'success':>9}{'coverage':>10}{'planner':>9}"
          f"{'trap':>6}{'useful offered/step':>21}")
    rows = []
    for (split, arm), c in sorted(summary.items()):
        n = sum(c.values())
        o, t = offered_rate[(split, arm)]
        print(f"{split:<6}{arm:<10}{n:>5}{c['success']:>9}{c['coverage']:>10}"
              f"{c['planner']:>9}{c['trap']:>6}{o / t:>21.1%}")
        rows.append({"split": split, "arm": arm, "n": n, **c,
                     "useful_offered_per_step": round(o / t, 4)})
    zero = collections.defaultdict(lambda: [0, 0])
    for p in per_ep:
        if p["arm"] == "oracle":
            zero[(p["split"], p["world"])][0] += p["kind"] == "success"
            zero[(p["split"], p["world"])][1] += 1
    print("oracle 0-success worlds:",
          [f"{s}:{w}" for (s, w), (k, n) in sorted(zero.items()) if k == 0])
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"tail": TAIL, "summary": rows, "episodes": per_ep}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
