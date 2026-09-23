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
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import EnvInfos

from concurrent.futures import ThreadPoolExecutor

from agent.task import (goal_spec, trap_goal_spec, goal_satisfied, goal_progress,
                        static_catalog, utility, unique_prefixes, plan_prior,
                        goal_queries, state_schema)
from env.serialize import canonical_state, state_hash
from env.worlds import SPLITS, build_world, build_trap_world, world_fingerprint
from forecast import render_facts
import runinfo

INFOS = EnvInfos(facts=True, typed_entities=True, possible_admissible_commands=True,
                 last_action=True, admissible_commands=True)

N_WORLDS = 12          # all locked-door variants -> genuine 5-step dependency
ROOTS_PER_WORLD = 4
STEP_CAP = 15          # MVP scale; 설계.md §3.3 uses 30 for the main experiment
K = 8
H = 2
CAP = [15]
HORIZON = [2]      # mutable so --horizon can override          # mutable so --cap can override
ALL_ARMS = ("C", "validity", "oracle", "A", "Aend", "Aval", "R", "A_jev")
JEV_ARMS = ("A", "Aend", "Aval", "R", "A_jev")
JEV_MODE = {"A": "full", "Aend": "endpoint", "Aval": "validity"}
# A_jev is the recursive arm under its KIIS name (kiis2026f/실험계획.md §1); R is
# kept so older run directories still mean what they meant.
RECURSIVE_ARMS = ("R", "A_jev")
ARMS = ["C", "validity", "oracle"]
SEED = 20260921
OUT = Path("artifacts/mvp_c")


def true_trajectory(env, game, prefix):
    """Real canonical state after each action. DIAGNOSTIC ONLY — handed to the
    recursive arm's drift counters, never to its scoring."""
    b = env.copy()
    out = []
    for a in prefix:
        b.step(a)
        out.append(canonical_state(list(b.state["_facts"]), game))
    b.close()
    return out


def score_arm_a(fc, state, prefixes, plans, env=None, game=None, failed=None):
    """JEV-backed arms: identical planner, goal and utility.

    `failed` holds (state fingerprint, action) pairs this episode already
    ATTEMPTED and watched fail. That is observed history, which 설계.md §3.5
    lists as allowed online information — the policy already receives it — and
    without it a confidently wrong validity judgement (0.91 on a command that
    cannot run) loops forever, because the failed action is a no-op so the
    state, and therefore the forecast, never changes.
    """
    # The endpoint ablation must not issue validity requests at all.
    fingerprint = "|".join(",".join(r) for r in state["dynamic_facts"])
    # The recursive arm asks validity inside its own rollout; the endpoint
    # ablation must not ask it at all.
    p_first = ({} if fc.mode in ("endpoint", "recursive")
               else fc.validity(state, sorted({p[0] for p in prefixes}), []))
    key = state_hash(state)

    def one(prefix):
        if truths is not None:
            terms, n_inv = fc.score(state, key, prefix, p_first,
                                    truth=truths[prefix])
        else:
            terms, n_inv = fc.score(state, key, prefix, p_first)
        return prefix, utility(terms.conj, terms.progress, n_inv,
                               len(prefix), plan_prior(plans, prefix))

    live = [p for p in prefixes
            if failed is None or (fingerprint, p[0]) not in failed]
    if not live:                       # everything tried and failed; keep going
        live = list(prefixes)

    # env.copy() deep-copies shared engine state, so it cannot run inside the
    # thread pool: eight workers cloning the same env raced and crashed with
    # "dictionary changed size during iteration". Build the diagnostic
    # trajectories serially first.
    truths = None
    if fc.mode == "recursive" and env is not None:
        truths = {p: true_trajectory(env, game, p) for p in live}
    with ThreadPoolExecutor(max_workers=8) as pool:
        scored = list(pool.map(one, live))
    return max(scored, key=lambda kv: kv[1])[0]


def run_episode(arm, env_root, game, meta, goal, policy, catalog, log, fc=None,
                root_idx=0):
    env = env_root.copy()
    steps = invalid = traps = decoys = 0
    status = "cap"
    history: list[tuple[str, bool]] = []
    failed: set[tuple[str, str]] = set()      # (state fingerprint, action)
    for step in range(CAP[0]):
        facts = list(env.state["_facts"])
        if goal_satisfied(facts, goal):
            status = "success"
            break
        state = canonical_state(facts, game, attempt_index=steps)
        policy.set_context(meta["world_id"], root_idx, step)
        plans, pstatus = policy.plans(render_facts(state), goal["text"], catalog,
                                      history, K, HORIZON[0])

        if arm == "C":
            action = plans[0][0]
            chosen = tuple(plans[0][:1])
        elif arm in JEV_ARMS:
            fc.reset_step_cache()
            chosen = score_arm_a(fc, state, unique_prefixes(plans, HORIZON[0]), plans,
                                 env=env, game=game, failed=failed)
            action = chosen[0]
        else:
            goal_blind = (arm == "validity")
            best, chosen = None, None
            for prefix in unique_prefixes(plans, HORIZON[0]):
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
        if not was_valid:
            failed.add(("|".join(",".join(r) for r in state["dynamic_facts"]), action))
        env.step(action)
        history.append((action, was_valid))
        steps += 1
        invalid += (not was_valid)
        log.append({"arm": arm, "world": meta["world_id"], "root": root_idx,
                    "step": step, "action": action, "valid": was_valid,
                    "policy_status": pstatus, "plans": plans, "prefix": list(chosen),
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
    ap.add_argument("--horizon", type=int, default=H,
                    help="planning horizon H; prefixes run 1..H")
    ap.add_argument("--budget", type=float, default=3.0,
                    help="stop before exceeding this many USD of JEV input tokens "
                         "(this process only; the API returned 402 mid-run twice on 9/21)")
    ap.add_argument("--policy-seed", type=int, default=SEED,
                    help="varies policy sampling while worlds/roots stay fixed")
    ap.add_argument("--arms", default="C,validity,oracle",
                    help=f"comma list from {','.join(ALL_ARMS)}")
    ap.add_argument("--split", default="dev", choices=SPLITS,
                    help="trap-world names: dev = t000.. (developed on), "
                         "test = held-out names (kiis2026f/실험계획.md §3)")
    args = ap.parse_args()
    if args.split != "dev" and not args.trap:
        raise SystemExit("--split applies to trap worlds only; add --trap")

    global ARMS
    ARMS = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = set(ARMS) - set(ALL_ARMS)
    if bad:
        raise SystemExit(f"unknown arms: {sorted(bad)}")
    use_jev = any(a in JEV_ARMS for a in ARMS)
    jev = stack = None
    if use_jev:
        from jev_client import JevClient
        from wm.jev_forecaster import JevForecaster
        jev = JevClient(save_raw=True, budget_usd=args.budget)
        print("arm A enabled: JEV forecaster")

    from agent.policy import Policy
    print(f"loading {args.model} on {args.device} ...", flush=True)
    policy = Policy(args.model, args.device, seed=args.policy_seed)
    print("loaded.", flush=True)

    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    wdir = out / "worlds"; wdir.mkdir(exist_ok=True)
    rng = random.Random(SEED)
    episodes, log = [], []

    CAP[0] = args.cap
    HORIZON[0] = args.horizon

    # What this run is, written before the first episode so a crash still
    # leaves it behind. Every episode carries config_hash; analyze_arms.py
    # refuses to pool two of them.
    cond = runinfo.condition(model=args.model, max_new_tokens=policy.max_new_tokens,
                             k=K, h=HORIZON[0], cap=CAP[0], trap=args.trap)
    config_hash = runinfo.digest(cond)
    arm_hashes = {a: runinfo.arm_hash(a) for a in ARMS}
    manifest = {"config_hash": config_hash, "condition": cond,
                "run": {"arms": ARMS, "arm_hashes": arm_hashes, "split": args.split,
                        "worlds": args.worlds, "roots": args.roots,
                        "policy_seed": args.policy_seed, "budget_usd": args.budget,
                        "host": socket.gethostname(),
                        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                "revision": runinfo.revision(), "world_fingerprints": {}}

    def write_manifest():
        (out / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    write_manifest()
    print(f"condition {config_hash}  split={args.split}  arms={ARMS}", flush=True)

    for w in range(args.worlds):
        if args.trap:
            game, path, meta = build_trap_world(w, wdir, rng, split=args.split)
            goal = trap_goal_spec(game, meta)
        else:
            idx = w * 3                  # idx % 3 == 0 -> locked door
            game, path, meta = build_world(idx, wdir, rng)
            goal = goal_spec(game, meta)
        gqs, conj_q = goal_queries(game, meta, goal) if use_jev else (None, None)
        schema = state_schema(game, meta) if use_jev else None
        catalog = static_catalog(game)
        env = textworld.start(str(path), request_infos=INFOS); env.reset()
        manifest["world_fingerprints"][meta["world_id"]] = world_fingerprint(game)
        write_manifest()

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
                if arm in RECURSIVE_ARMS:
                    from wm.recursive_forecaster import RecursiveForecaster
                    fc = RecursiveForecaster(jev, schema, goal)
                elif arm in JEV_ARMS:
                    from wm.jev_forecaster import JevForecaster
                    fc = JevForecaster(jev, gqs, conj_q, mode=JEV_MODE[arm])
                res = run_episode(arm, root, game, meta, goal,
                                  policy, catalog, log, fc, root_idx=r)
                if fc is not None:
                    res["jev_requests"] = fc.stats.requests
                    res["bound_violations"] = getattr(fc.stats, "bound_violations", 0)
                    if arm in RECURSIVE_ARMS:
                        s = fc.stats
                        res["drift"] = {
                            "n": s.depth_n, "exact": s.depth_exact,
                            "vars_ok": s.depth_vars_ok, "vars_tot": s.depth_vars_tot,
                            "in_beam": s.depth_in_beam,
                            "contradictions": s.contradictions,
                            "cache_hits": s.cache_hits}
                res.update({"world_id": meta["world_id"], "root": r,
                            "catalog_size": len(catalog), "split": args.split,
                            "policy_seed": args.policy_seed,
                            "config_hash": config_hash, "arm_hash": arm_hashes[arm]})
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
    print("\n" + "=" * 74)
    print(f"{'arm':<11}{'n':>5}{'success':>10}{'steps':>9}{'invalid':>10}"
          f"{'progress':>10}{'trap':>7}{'decoy':>7}")
    for name in ARMS:
        a = stats[name]
        print(f"{name:<11}{a['n']:>5}{a['success']:>10.1%}{a['steps']:>9.1f}"
              f"{a['invalid']:>10.1%}{a['progress']:>10.1%}"
              f"{a['traps']:>7}{a['decoys']:>7}")

    def gap(hi, lo, label):
        if hi in stats and lo in stats:
            d = stats[hi]["success"] - stats[lo]["success"]
            print(f"  {label:<34}{d:>+8.1%}")
            return d
        return None

    print("\n분해")
    total = gap("oracle", "C", "전체 headroom   oracle - C")
    by_val = gap("validity", "C", "유효성이 설명   validity - C")
    needs = gap("oracle", "validity", "예측이 필요     oracle - validity")
    a_gain = gap("A", "C", "arm A 이득      A - C")
    a_vs_v = gap("A", "validity", "arm A vs 유효성 A - validity")
    a_vs_o = gap("A", "oracle", "arm A vs 완벽   A - oracle")

    if a_gain is not None and total:
        print(f"\n  arm A 가 회수한 headroom 비율: {a_gain/total:.0%}")
    if a_vs_v is not None and needs is not None and needs > 0:
        print(f"  예측 구간({needs:+.1%}) 중 arm A 몫: {a_vs_v/needs:.0%}")

    if use_jev:
        eps_a = [e for e in episodes if e["arm"] in JEV_ARMS]
        req = sum(e.get("jev_requests", 0) for e in eps_a)
        bv = sum(e.get("bound_violations", 0) for e in eps_a)
        print(f"\nJEV: 요청 {req}  에피소드당 {req/max(len(eps_a),1):.0f}  "
              f"conjunction bound 위반 {bv}")
        drifts = [e["drift"] for e in eps_a if "drift" in e]
        if drifts:
            print("\n오류 누적 (재귀 arm, 예측 상태 vs 실제 상태)")
            print(f"  {'depth':>6}{'n':>8}{'상태 완전일치':>14}{'변수 정확도':>13}{'beam 포함':>11}")
            depths = sorted({int(k) for d in drifts for k in d["n"]})
            key = lambda d, sub, dep: d[sub].get(dep, d[sub].get(str(dep), 0))
            for dep in depths:
                n = sum(key(d, "n", dep) for d in drifts)
                ex = sum(key(d, "exact", dep) for d in drifts)
                vo = sum(key(d, "vars_ok", dep) for d in drifts)
                vt = sum(key(d, "vars_tot", dep) for d in drifts)
                ib = sum(key(d, "in_beam", dep) for d in drifts)
                print(f"  {dep:>6}{n:>8}{ex/max(n,1):>14.1%}"
                      f"{vo/max(vt,1):>13.1%}{ib/max(n,1):>11.1%}")
            print(f"  모순 감지 {sum(d['contradictions'] for d in drifts)}  "
                  f"캐시 적중 {sum(d['cache_hits'] for d in drifts)}")
        if jev is not None:
            print(f"     calls={jev.calls} tokens={jev.input_tokens:,} "
                  f"cost=${jev.input_tokens/1e6*0.042:.3f}")
            jev.close()

    print(f"\npolicy calls={policy.calls} repairs={policy.repairs} "
          f"fallbacks={policy.fallbacks} catalog={episodes[0]['catalog_size']}")
    print("=" * 74)

    manifest["run"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["counts"] = {"episodes": len(episodes), "policy_calls": policy.calls,
                          "policy_repairs": policy.repairs,
                          "policy_fallbacks": policy.fallbacks,
                          "jev_calls": jev.calls if jev else 0,
                          "jev_input_tokens": jev.input_tokens if jev else 0,
                          "jev_usd": round(jev.spent_usd, 4) if jev else 0.0}
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
