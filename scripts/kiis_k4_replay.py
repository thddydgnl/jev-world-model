"""Why an arm fails in closed loop: replay K4 episodes in the engine
(kiis2026f/실험계획.md §4 W3, "D0가 낮은 원인").

For every root the zero-shot unit finished, the B0, D0 and C_fm episodes are
replayed action by action from the same root the runner built. At each step
the replay recomputes what C_fm's rule would have picked from the logged plans
(the first plan whose first action has not already failed from this state), so
a world-model arm's step is an "override" when it chose something else. The
logged validity and success of every episode must come out the same, or the
script stops.

    python scripts/kiis_k4_replay.py --logs artifacts/kiis_k4v3 --seeds 20260921,1234,777
    python scripts/kiis_k4_replay.py --k5 artifacts/kiis_k5v3 --rollouts data/kiis/k5v3_rollouts.json

The second form tabulates the K5 teacher-forced rows by action kind and truth
(p_exec, parse failures, exact state). Diagnostic only: nothing here feeds a
planning decision or a hypothesis test.
"""
import argparse
import json
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ARMS = ("C_fm", "B0_typed", "D0_gen")


def kind(action):
    v = action.split()[0]
    if v == "take":
        return "take key" if " key" in action.split(" from ")[0] else "take food"
    return v


def load(p):
    rows = []
    for line in Path(p).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass           # a file caught mid-write
    return rows


def build_roots():
    """Worlds and roots exactly as mvp_c_headroom.main builds them for the
    v3 test split with no levers."""
    import textworld
    import mvp_c_headroom as runner
    from agent.task import goal_satisfied, trap_goal_spec
    from env.worlds import build_trap_world_v2
    wdir = Path(tempfile.mkdtemp())
    rng = random.Random(runner.SEED)
    worlds, roots = {}, {}
    for w in range(24):
        game, path, meta = build_trap_world_v2(w, wdir, "test3", ())
        goal = trap_goal_spec(game, meta)
        env = textworld.start(str(path), request_infos=runner.INFOS); env.reset()
        for r in range(4):
            root = env.copy()
            for _ in range(rng.randint(0, 2)):
                adm = [c for c in root.state["admissible_commands"]
                       if not c.startswith(("look", "inventory", "examine"))]
                if not adm:
                    break
                root.step(rng.choice(adm))
            if goal_satisfied(list(root.state["_facts"]), goal):
                root.close(); continue
            roots[(meta["world_id"], r)] = root
        worlds[meta["world_id"]] = (game, goal)
    return worlds, roots


def atoms(facts, goal):
    idx = {tuple([f.name] + [a.name for a in f.arguments]) for f in facts}
    return "".join("1" if a in idx else "0" for a in goal["atoms"])


def replay(worlds, roots, world, root_idx, recs):
    import mvp_c_headroom as runner
    from agent.task import goal_satisfied
    from env.serialize import canonical_state
    game, goal = worlds[world]
    env = roots[(world, root_idx)].copy()
    failed, rows = set(), []
    for n, rec in enumerate(recs):
        facts = list(env.state["_facts"])
        state = canonical_state(facts, game, attempt_index=n)
        fp = "|".join(",".join(r) for r in state["dynamic_facts"])
        adm = set(env.state["admissible_commands"])
        default = runner.c_fm_prefix(rec["plans"], state, failed)[0]
        chosen = rec["action"]
        if (chosen in adm) != rec["valid"]:
            raise SystemExit(f"replay mismatch {world} r{root_idx} step {n}: {chosen}")
        row = {"chosen": chosen, "default": default, "override": chosen != default,
               "c_valid": chosen in adm, "d_valid": default in adm, "atoms0": atoms(facts, goal)}
        if not row["c_valid"]:
            failed.add((fp, chosen))
        env.step(chosen)
        row["c_atoms"] = atoms(list(env.state["_facts"]), goal)
        rows.append(row)
    ok = goal_satisfied(list(env.state["_facts"]), goal)
    env.close()
    return rows, ok


def replay_all(logs, seeds):
    worlds, roots = build_roots()
    out = []
    for seed in seeds:
        units = {}
        for unit in ("zero_shot", "core_a"):
            eps = load(Path(logs) / seed / unit / "episodes.jsonl")
            steps = defaultdict(list)
            for s in load(Path(logs) / seed / unit / "steps.jsonl"):
                steps[(s["arm"], s["world"], s["root"])].append(s)
            for e in eps:
                k = (e["arm"], e["world_id"], e["root"])
                st = sorted(steps.get(k, []), key=lambda s: s["step"])
                if e["arm"] in ARMS and len(st) == e["steps"]:
                    units[k] = (e, st)
        done = {(w, r) for (a, w, r) in units if a == "D0_gen"} & \
               {(w, r) for (a, w, r) in units if a == "B0_typed"}
        for (arm, w, r), (e, st) in sorted(units.items()):
            if (w, r) not in done:
                continue
            rows, ok = replay(worlds, roots, w, r, st)
            if ok != e["success"]:
                raise SystemExit(f"success mismatch {seed} {arm} {w} r{r}")
            out.append({"seed": seed, "arm": arm, "world": w, "root": r,
                        "episode": e, "rows": rows})
    return out


def summarize(R) -> dict:
    """The numbers report() prints, for kiis_report.py."""
    out = {"episodes": len(R), "roots": len({(x["seed"], x["world"], x["root"]) for x in R}),
           "goal_room_without_food": {}, "overrides": {}, "back_in_start_room": {}}
    for arm in ARMS:
        xs = [x for x in R if x["arm"] == arm]
        hit = [x for x in xs if any(r["c_atoms"][0] == "0" and r["c_atoms"][2] == "1" for r in x["rows"])]
        miss = [x for x in xs if x not in hit]
        sr = lambda v: sum(x["episode"]["success"] for x in v)
        out["goal_room_without_food"][arm] = {
            "n": len(xs), "success": sr(xs), "entered": len(hit), "success_after": sr(hit),
            "never_entered": len(miss), "success_never_entered": sr(miss)}
    for arm in ("B0_typed", "D0_gen"):
        n, o = Counter(), Counter()
        rows = [r for x in R if x["arm"] == arm for r in x["rows"]]
        for r in rows:
            n[kind(r["default"])] += 1
            o[kind(r["default"])] += r["override"]
        ov = [r for r in rows if r["override"]]
        out["overrides"][arm] = {"steps": len(rows), "overrides": len(ov),
                                 "replaced_valid": sum(r["d_valid"] for r in ov),
                                 "chosen_valid": sum(r["c_valid"] for r in ov),
                                 "by_policy_choice": {k: [o[k], n[k]] for k in
                                                      ("take food", "take key", "open", "unlock", "go") if n[k]}}
    for arm in ARMS:
        steps = food = 0
        for x in R:
            if x["arm"] != arm or x["episode"]["success"]:
                continue
            rows = x["rows"]
            back = [i for i, r in enumerate(rows) if r["atoms0"] == "010"
                    and any(q["c_atoms"][2] == "1" for q in rows[:i])]
            steps += len(back)
            food += sum(kind(rows[i]["default"]) == "take food" for i in back)
        out["back_in_start_room"][arm] = {"steps": steps, "food_first": food}
    why = Counter()
    for x in R:
        for ex in x["episode"].get("parse_failure_examples", []):
            if kind(ex["action"]) != "take food":
                continue
            locs = Counter(m.group(2) for m in re.finditer(r"\b(in|on|at)\((f_\d+)\s*,", ex["reply"]))
            why["same food in two places" if any(v > 1 for v in locs.values()) else "other"] += 1
    out["take_food_parse_failures"] = dict(why)
    return out


def report(R):
    s = summarize(R)
    print(f"replayed {s['episodes']} episodes on {s['roots']} roots; validity and success match the logs")
    print("\nentered the goal room without the food (door open, food not carried)")
    for arm, g in s["goal_room_without_food"].items():
        print(f"  {arm:9s} success {g['success']}/{g['n']}  entered {g['entered']} ({100 * g['entered'] / g['n']:.0f}%)"
              f"  success after {g['success_after']}/{g['entered']}  never entered: success "
              f"{g['success_never_entered']}/{g['never_entered']}")
    print("\nsteps where the arm replaced C_fm's choice, by that choice")
    for arm, g in s["overrides"].items():
        print(f"  {arm:9s} {g['overrides']}/{g['steps']} steps; replaced choice was valid in "
              f"{g['replaced_valid']}, the arm's choice in {g['chosen_valid']}")
        print("    " + "  ".join(f"{k} {a}/{b}" for k, (a, b) in g["by_policy_choice"].items()))
    print("\nafter returning to the start room without the food (state 010): policy's first choice is the food")
    for arm, g in s["back_in_start_room"].items():
        print(f"  {arm:9s} {g['food_first']}/{g['steps']} steps")
    print(f"\nD0 take-food parse failures (logged examples): {s['take_food_parse_failures']}")


def k5_table(k5_dir, rollouts):
    R = {r["id"]: r for r in json.load(open(rollouts))["rollouts"]}
    for arm in ("D0_gen", "B0_typed", "A_jev"):
        agg = defaultdict(Counter)
        for r in load(Path(k5_dir) / f"{arm}.tf.jsonl"):
            g = agg[(kind(R[r["id"]]["actions"][r["k"] - 1]), r["executes"])]
            g["n"] += 1
            g["p_exec"] += r["p_exec"]
            g["parse"] += r.get("parse_failed", False)
            g["exact"] += r["exact"]
        print(f"== {arm} (teacher-forced)\n  {'action':10s} {'truth':6s} {'n':>5s} {'p_exec':>7s} {'parse':>6s} {'exact':>6s}")
        for (c, ex) in sorted(agg, key=lambda k: (k[0], not k[1])):
            g = agg[(c, ex)]
            if g["n"] >= 15:
                print(f"  {c:10s} {'runs' if ex else 'fails':6s} {g['n']:5d} {100 * g['p_exec'] / g['n']:6.0f}% "
                      f"{100 * g['parse'] / g['n']:5.0f}% {100 * g['exact'] / g['n']:5.0f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="artifacts/kiis_k4v3")
    ap.add_argument("--seeds", default="20260921,1234,777")
    ap.add_argument("--out", default=None, help="write the replayed rows here (json)")
    ap.add_argument("--k5", default=None, help="K5 output dir: tabulate its teacher-forced rows instead")
    ap.add_argument("--rollouts", default="data/kiis/k5v3_rollouts.json")
    args = ap.parse_args()
    if args.k5:
        k5_table(args.k5, args.rollouts)
        return 0
    R = replay_all(args.logs, [s for s in args.seeds.split(",") if s])
    if args.out:
        Path(args.out).write_text(json.dumps(R))
    report(R)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
