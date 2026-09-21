"""MVP-C: is there room for a world model to help at all? (설계.md §25)

Two closed-loop arms over identical worlds, goals, seeds and candidate sets:

  C       follow the frozen policy's own top preference
  oracle  score the SAME candidate prefixes by their TRUE endpoint and pick
          the argmax under the common utility

The gap is the headroom a learned forecaster could conceivably capture. It is
NOT an upper bound on achievable performance: the utility is myopic and the
candidate set comes from the same policy (설계.md §22.4).

Gate (설계.md §24 "oracle headroom"): >= ~10 points.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import EnvInfos

from agent.task import (goal_spec, goal_satisfied, goal_progress, static_catalog,
                        utility, unique_prefixes, plan_prior)
from env.serialize import canonical_state
from env.worlds import build_world
from forecast import render_facts

INFOS = EnvInfos(facts=True, typed_entities=True, possible_admissible_commands=True,
                 last_action=True, admissible_commands=True)

N_WORLDS = 12          # all locked-door variants -> genuine 5-step dependency
ROOTS_PER_WORLD = 4
STEP_CAP = 15          # MVP scale; 설계.md §3.3 uses 30 for the main experiment
K, H = 8, 2
SEED = 20260921
OUT = Path("artifacts/mvp_c")


def run_episode(arm, env_root, game, meta, goal, policy, catalog, log):
    env = env_root.copy()
    steps = invalid = 0
    status = "cap"
    history: list[tuple[str, bool]] = []
    for step in range(STEP_CAP):
        facts = list(env.state["_facts"])
        if goal_satisfied(facts, goal):
            status = "success"
            break
        state = canonical_state(facts, game, attempt_index=steps)
        plans, pstatus = policy.plans(render_facts(state), goal["text"], catalog,
                                      history, K, H)

        if arm == "C":
            action = plans[0][0]
            chosen = tuple(plans[0][:1])
        else:
            best, chosen = None, None
            for prefix in unique_prefixes(plans, H):
                branch = env.copy()
                n_inv = 0
                for a in prefix:
                    if a not in branch.state["admissible_commands"]:
                        n_inv += 1
                    branch.step(a)
                end = list(branch.state["_facts"])
                branch.close()
                u = utility(conj=float(goal_satisfied(end, goal)),
                            progress=goal_progress(end, goal),
                            n_invalid=n_inv, h=len(prefix),
                            prior=plan_prior(plans, prefix))
                if best is None or u > best:
                    best, chosen = u, prefix
            action = chosen[0]

        was_valid = action in env.state["admissible_commands"]
        env.step(action)
        history.append((action, was_valid))
        steps += 1
        invalid += (not was_valid)
        log.append({"arm": arm, "world": meta["world_id"], "step": step,
                    "action": action, "valid": was_valid,
                    "policy_status": pstatus, "prefix": list(chosen)})
    else:
        if goal_satisfied(list(env.state["_facts"]), goal):
            status = "success"
    progress = goal_progress(list(env.state["_facts"]), goal)
    env.close()
    return {"arm": arm, "status": status, "success": status == "success",
            "steps": steps, "invalid": invalid, "progress": progress}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--worlds", type=int, default=N_WORLDS)
    ap.add_argument("--roots", type=int, default=ROOTS_PER_WORLD)
    args = ap.parse_args()

    from agent.policy import Policy
    print(f"loading {args.model} on {args.device} ...", flush=True)
    policy = Policy(args.model, args.device, seed=SEED)
    print("loaded.", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    wdir = OUT / "worlds"; wdir.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    episodes, log = [], []

    for w in range(args.worlds):
        idx = w * 3                      # idx % 3 == 0 -> locked door
        game, path, meta = build_world(idx, wdir, rng)
        goal = goal_spec(game, meta)
        catalog = static_catalog(game)
        env = textworld.start(str(path), request_infos=INFOS); env.reset()

        for r in range(args.roots):
            root = env.copy()
            for _ in range(rng.randint(0, 2)):
                adm = [c for c in root.state["admissible_commands"]
                       if not c.startswith(("look", "inventory", "examine"))]
                if not adm:
                    break
                root.step(rng.choice(adm))
            if goal_satisfied(list(root.state["_facts"]), goal):
                root.close(); continue          # already solved; skip
            for arm in ("C", "oracle"):
                res = run_episode(arm, root, game, meta, goal,
                                  policy, catalog, log)
                res.update({"world_id": meta["world_id"], "root": r,
                            "catalog_size": len(catalog)})
                episodes.append(res)
            root.close()
            done = len(episodes)
            print(f"  {meta['world_id']} r{r}  "
                  f"C={'O' if episodes[-2]['success'] else 'X'}({episodes[-2]['steps']})  "
                  f"oracle={'O' if episodes[-1]['success'] else 'X'}({episodes[-1]['steps']})  "
                  f"[{done//2} pairs]", flush=True)
        env.close()

    (OUT / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e) for e in episodes) + "\n")
    (OUT / "steps.jsonl").write_text(
        "\n".join(json.dumps(s) for s in log) + "\n")

    def agg(arm):
        v = [e for e in episodes if e["arm"] == arm]
        n = len(v)
        return {"n": n,
                "success": sum(e["success"] for e in v) / n,
                "steps": sum(e["steps"] for e in v) / n,
                "invalid": sum(e["invalid"] for e in v) / max(sum(e["steps"] for e in v), 1),
                "progress": sum(e["progress"] for e in v) / n}

    c, o = agg("C"), agg("oracle")
    gap = o["success"] - c["success"]
    print("\n" + "=" * 62)
    print(f"{'arm':<10}{'n':>5}{'success':>10}{'steps':>9}{'invalid':>10}{'progress':>10}")
    for name, a in (("C", c), ("oracle", o)):
        print(f"{name:<10}{a['n']:>5}{a['success']:>10.1%}{a['steps']:>9.1f}"
              f"{a['invalid']:>10.1%}{a['progress']:>10.1%}")
    print(f"\noracle headroom = {gap:+.1%}  (gate: >= +10%p)")
    print(f"policy calls={policy.calls} repairs={policy.repairs} "
          f"fallbacks={policy.fallbacks} catalog={episodes[0]['catalog_size']}")
    print("=" * 62)
    print(f"MVP-C {'PASS — WM이 기여할 여지 있음' if gap >= 0.10 else 'FAIL — 여지 부족'}")
    return 0 if gap >= 0.10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
