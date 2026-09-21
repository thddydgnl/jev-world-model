"""MVP-C: is there room for a world model to help at all? (설계.md §25)

Two closed-loop arms over identical worlds, goals, seeds and candidate sets:

  C         follow the frozen policy's own top preference
  validity  score by the TRUE invalid-command count and the policy prior only
            (설계.md §20 "validity-only"); the goal terms are zeroed
  oracle    score the SAME candidate prefixes by their FULL true endpoint

validity sits between the other two on information, not on mechanism: it reads
the same engine branch as oracle but is allowed to use only whether the
commands execute. The oracle-minus-validity gap is the part of the headroom
that needs an actual goal-directed forecast rather than a validity filter.

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

from concurrent.futures import ThreadPoolExecutor

from agent.task import (goal_spec, trap_goal_spec, goal_satisfied, goal_progress,
                        static_catalog, utility, unique_prefixes, plan_prior,
                        goal_queries)
from env.serialize import canonical_state, state_hash
from env.worlds import build_world, build_trap_world
from forecast import render_facts

INFOS = EnvInfos(facts=True, typed_entities=True, possible_admissible_commands=True,
                 last_action=True, admissible_commands=True)

N_WORLDS = 12          # all locked-door variants -> genuine 5-step dependency
ROOTS_PER_WORLD = 4
STEP_CAP = 15          # MVP scale; 설계.md §3.3 uses 30 for the main experiment
K, H = 8, 2
CAP = [15]          # mutable so --cap can override
ALL_ARMS = ("C", "validity", "oracle", "A")
ARMS = ["C", "validity", "oracle"]
SEED = 20260921
OUT = Path("artifacts/mvp_c")


def score_arm_a(fc, state, prefixes, plans):
    """Arm A: identical planner, goal and utility; JEV supplies the terms."""
    firsts = sorted({p[0] for p in prefixes})
    p_first = fc.validity(state, firsts, [])
    key = state_hash(state)

    def one(prefix):
        terms, n_inv = fc.score(state, key, prefix, p_first)
        return prefix, utility(terms.conj, terms.progress, n_inv,
                               len(prefix), plan_prior(plans, prefix))

    with ThreadPoolExecutor(max_workers=8) as pool:
        scored = list(pool.map(one, prefixes))
    return max(scored, key=lambda kv: kv[1])[0]


def run_episode(arm, env_root, game, meta, goal, policy, catalog, log, fc=None,
                root_idx=0):
    env = env_root.copy()
    steps = invalid = traps = decoys = 0
    status = "cap"
    history: list[tuple[str, bool]] = []
    for step in range(CAP[0]):
        facts = list(env.state["_facts"])
        if goal_satisfied(facts, goal):
            status = "success"
            break
        state = canonical_state(facts, game, attempt_index=steps)
        policy.set_context(meta["world_id"], root_idx, arm, step)
        plans, pstatus = policy.plans(render_facts(state), goal["text"], catalog,
                                      history, K, H)

        if arm == "C":
            action = plans[0][0]
            chosen = tuple(plans[0][:1])
        elif arm == "A":
            fc._endpoint_cache.clear()
            chosen = score_arm_a(fc, state, unique_prefixes(plans, H), plans)
            action = chosen[0]
        else:
            goal_blind = (arm == "validity")
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
                u = utility(
                    conj=0.0 if goal_blind else float(goal_satisfied(end, goal)),
                    progress=0.0 if goal_blind else goal_progress(end, goal),
                    n_invalid=n_inv, h=len(prefix),
                    prior=plan_prior(plans, prefix))
                if best is None or u > best:
                    best, chosen = u, prefix
            action = chosen[0]

        was_valid = action in env.state["admissible_commands"]
        is_trap = action in goal.get("trap_commands", [])
        is_decoy = action in goal.get("decoy_commands", [])
        traps += is_trap
        decoys += is_decoy
        env.step(action)
        history.append((action, was_valid))
        steps += 1
        invalid += (not was_valid)
        log.append({"arm": arm, "world": meta["world_id"], "step": step,
                    "action": action, "valid": was_valid,
                    "policy_status": pstatus, "prefix": list(chosen),
                    "trap": bool(is_trap), "decoy": bool(is_decoy)})
    else:
        if goal_satisfied(list(env.state["_facts"]), goal):
            status = "success"
    progress = goal_progress(list(env.state["_facts"]), goal)
    env.close()
    return {"arm": arm, "status": status, "success": status == "success",
            "steps": steps, "invalid": invalid, "progress": progress,
            "traps": traps, "decoys": decoys}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--worlds", type=int, default=N_WORLDS)
    ap.add_argument("--roots", type=int, default=ROOTS_PER_WORLD)
    ap.add_argument("--trap", action="store_true",
                    help="irreversible-trap worlds (eat the goal object)")
    ap.add_argument("--cap", type=int, default=STEP_CAP)
    ap.add_argument("--out", default=None)
    ap.add_argument("--arms", default="C,validity,oracle",
                    help="comma list from C,validity,oracle,A")
    args = ap.parse_args()

    global ARMS
    ARMS = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = set(ARMS) - set(ALL_ARMS)
    if bad:
        raise SystemExit(f"unknown arms: {sorted(bad)}")
    use_jev = "A" in ARMS
    jev = stack = None
    if use_jev:
        from jev_client import JevClient
        from wm.jev_forecaster import JevForecaster
        jev = JevClient(save_raw=True)
        print("arm A enabled: JEV forecaster")

    from agent.policy import Policy
    print(f"loading {args.model} on {args.device} ...", flush=True)
    policy = Policy(args.model, args.device, seed=SEED)
    print("loaded.", flush=True)

    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    wdir = out / "worlds"; wdir.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    episodes, log = [], []

    CAP[0] = args.cap
    for w in range(args.worlds):
        if args.trap:
            game, path, meta = build_trap_world(w, wdir, rng)
            goal = trap_goal_spec(game, meta)
        else:
            idx = w * 3                  # idx % 3 == 0 -> locked door
            game, path, meta = build_world(idx, wdir, rng)
            goal = goal_spec(game, meta)
        gqs, conj_q = goal_queries(game, meta, goal) if use_jev else (None, None)
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
            for arm in ARMS:
                fc = None
                if arm == "A":
                    from wm.jev_forecaster import JevForecaster
                    fc = JevForecaster(jev, gqs, conj_q)
                res = run_episode(arm, root, game, meta, goal,
                                  policy, catalog, log, fc, root_idx=r)
                if fc is not None:
                    res["jev_requests"] = fc.stats.requests
                    res["bound_violations"] = fc.stats.bound_violations
                res.update({"world_id": meta["world_id"], "root": r,
                            "catalog_size": len(catalog)})
                episodes.append(res)
            root.close()
            (out / "episodes.jsonl").write_text(
                "\n".join(json.dumps(e) for e in episodes) + "\n")
            (out / "steps.jsonl").write_text(
                "\n".join(json.dumps(s) for s in log) + "\n")
            recent = episodes[-len(ARMS):]
            marks = "  ".join(
                f"{e['arm']}={'O' if e['success'] else 'X'}({e['steps']})" for e in recent)
            print(f"  {meta['world_id']} r{r}  {marks}  "
                  f"[{len(episodes)//len(ARMS)} sets]", flush=True)
        env.close()

    (out / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e) for e in episodes) + "\n")
    (out / "steps.jsonl").write_text(
        "\n".join(json.dumps(s) for s in log) + "\n")

    def agg(arm):
        v = [e for e in episodes if e["arm"] == arm]
        n = len(v)
        return {"n": n,
                "success": sum(e["success"] for e in v) / n,
                "steps": sum(e["steps"] for e in v) / n,
                "invalid": sum(e["invalid"] for e in v) / max(sum(e["steps"] for e in v), 1),
                "progress": sum(e["progress"] for e in v) / n,
                "traps": sum(e["traps"] for e in v),
                "decoys": sum(e["decoys"] for e in v)}

    stats = {a: agg(a) for a in ARMS}
    c, v, o = stats["C"], stats["validity"], stats["oracle"]
    print("\n" + "=" * 66)
    print(f"{'arm':<11}{'n':>5}{'success':>10}{'steps':>9}{'invalid':>10}"
          f"{'progress':>10}{'trap':>8}{'decoy':>8}")
    for name in ARMS:
        a = stats[name]
        print(f"{name:<11}{a['n']:>5}{a['success']:>10.1%}{a['steps']:>9.1f}"
              f"{a['invalid']:>10.1%}{a['progress']:>10.1%}"
              f"{a['traps']:>8}{a['decoys']:>8}")

    total = o["success"] - c["success"]
    by_validity = v["success"] - c["success"]
    needs_goal = o["success"] - v["success"]
    share = by_validity / total if total else float("nan")
    print(f"\n전체 headroom      oracle - C        = {total:+.1%}")
    print(f"  유효성으로 설명   validity - C      = {by_validity:+.1%}  "
          f"(전체의 {share:.0%})")
    print(f"  목표예측이 필요   oracle - validity = {needs_goal:+.1%}")
    if use_jev:
        print(f"\nJEV: calls={jev.calls} tokens={jev.input_tokens:,} "
              f"cost=${jev.input_tokens/1e6*0.042:.3f}")
        jev.close()
    print(f"\npolicy calls={policy.calls} repairs={policy.repairs} "
          f"fallbacks={policy.fallbacks} catalog={episodes[0]['catalog_size']}")
    print("=" * 66)
    print(f"MVP-C {'PASS' if total >= 0.10 else 'FAIL'} (headroom gate >= +10%p)")
    print("AB3: " + ("유효성 필터만으로 대부분 설명됨 -> 풍부한 예측의 한계효용 낮음"
                     if share >= 0.70 else
                     "목표 지향 예측이 유효성 너머의 기여를 함"))
    return 0 if total >= 0.10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
