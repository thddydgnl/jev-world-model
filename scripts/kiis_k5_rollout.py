"""K5 — k-step rollout accuracy of every world model (kiis2026f/실험계획.md K5, §6.2).

    build   test worlds -> 400 start states x 2 action sequences of length 4
            (the policy's own H=4 plan, and a random 75% valid / 25% invalid one),
            with the engine's true state after every action. Engine truth only.
    eval    one world model rolls every sequence out with the planner's own
            recursive beam (RecursiveForecaster), and each depth is compared with
            the truth. Nothing here feeds back into any agent.
    report  state exact match / variable accuracy / executes accuracy at k=1..4,
            world-clustered bootstrap CIs, a persistence floor, JEV repeats.

Start states are stored as the action path from reset, so every process rebuilds
them from the fixed worlds and checks the stored truth before using it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import textworld
from textworld import EnvInfos

import runinfo
from agent.task import read_schema_values, state_schema, static_catalog, trap_goal_spec
from env.serialize import canonical_state
from env.transitions import OBSERVE, solution_path
from env.worlds import build_trap_world, world_fingerprint

INFOS = EnvInfos(facts=True, typed_entities=True, possible_admissible_commands=True,
                 admissible_commands=True)
ROLLOUTS = ROOT / "data/kiis/k5_rollouts.json"
OUT = ROOT / "artifacts/kiis_k5"
DEPTH = 4
N_STARTS = 400
P_VALID = 0.75
POLICY_K, POLICY_H = 8, 4
POLICY_SEED = 20260921
MODELS = ("A_jev", "B0_typed", "B_typed", "D0_gen", "D_gen")
REPEAT_EVERY = 8          # rollouts with index % 8 == 0 (100 of 800) for JEV repeats


# ------------------------------------------------------------------ worlds

def worlds(split: str, n: int, tmp: Path):
    ref = {w["world_id"]: w["fingerprint"] for w in
           json.loads((ROOT / "kiis2026f/worlds_manifest.json").read_text())["worlds"][split]}
    for i in range(n):
        game, path, meta = build_trap_world(i, tmp, random.Random(0), split=split)
        if world_fingerprint(game) != ref[meta["world_id"]]:
            raise SystemExit(f"{meta['world_id']} differs from worlds_manifest.json")
        env = textworld.start(str(path), request_infos=INFOS)
        env.reset()
        yield game, meta, env


def replay(env, path: list[str]):
    e = env.copy()
    for a in path:
        e.step(a)
    return e


def truth_of(env, game, schema, actions: list[str]) -> list[dict]:
    b = env.copy()
    out = []
    for a in actions:
        runs = a in b.state["admissible_commands"]
        b.step(a)
        canon = canonical_state(list(b.state["_facts"]), game)
        out.append({"executes": runs, "values": read_schema_values(canon, schema),
                    "dynamic_facts": canon["dynamic_facts"]})
    b.close()
    return out


def random_actions(env, catalog: list[str], rng: random.Random, n: int,
                   prefix: list[str] = ()) -> list[str]:
    """Continue `prefix` to length n: each new action runs with probability
    P_VALID (a state-changing command) or is one that does not run, judged on
    the true state reached so far."""
    b = env.copy()
    for a in prefix:
        b.step(a)
    out = list(prefix)
    while len(out) < n:
        adm = set(b.state["admissible_commands"])
        valid = [c for c in catalog if c in adm and not c.startswith(OBSERVE)]
        invalid = [c for c in catalog if c not in adm]
        pool = valid if (rng.random() < P_VALID and valid) or not invalid else invalid
        a = rng.choice(pool)
        b.step(a)
        out.append(a)
    b.close()
    return out


def start_paths(env, meta: dict, rng: random.Random, n: int) -> list[tuple[str, list[str]]]:
    """Walks and solution branches, as in env.transitions.sample_states, but the
    actions are kept so the state can be rebuilt. Distinct states only."""
    out, seen = [], set()
    for attempt in range(n * 10):
        if len(out) == n:
            break
        path: list[str] = []
        if attempt % 2 == 0:
            source, walk = "walk", rng.randint(0, 8)
        else:
            source = "branch"
            sol = solution_path(meta, rng)
            path = sol[:rng.randint(0, len(sol))]
            walk = rng.randint(0, 3)
        e = replay(env, path)
        for _ in range(walk):
            adm = [c for c in e.state["admissible_commands"] if not c.startswith(OBSERVE)]
            if not adm:
                break
            a = rng.choice(adm)
            e.step(a)
            path.append(a)
        fp = tuple(sorted(str(p) for p in e.state["_facts"]))
        e.close()
        if fp in seen:
            continue
        seen.add(fp)
        out.append((source, path))
    return out


def build(args) -> int:
    from agent.policy import Policy
    from forecast import render_facts
    rev = runinfo.revision()      # at start: the code this process loaded
    policy = Policy(args.model, args.device, seed=POLICY_SEED)
    n_worlds = 24 if args.split == "test" else args.worlds
    per_world = [N_STARTS // n_worlds + (1 if i < N_STARTS % n_worlds else 0)
                 for i in range(n_worlds)]
    rollouts = []
    with tempfile.TemporaryDirectory() as d:
        for i, (game, meta, env) in enumerate(worlds(args.split, n_worlds, Path(d))):
            schema = state_schema(game, meta)
            goal = trap_goal_spec(game, meta)
            catalog = static_catalog(game)
            rng = random.Random(f"kiis-k5-{meta['world_id']}")
            n = per_world[i] if args.split == "test" else args.starts
            for j, (source, path) in enumerate(start_paths(env, meta, rng, n)):
                start = replay(env, path)
                canon = canonical_state(list(start.state["_facts"]), game)
                policy.set_context(meta["world_id"], f"k5-{j}", 0)
                plans, status = policy.plans(render_facts(canon), goal["text"], catalog,
                                             [], POLICY_K, POLICY_H)
                plan = next((p for p in plans if len(p) == DEPTH), plans[0])
                seqs = {"policy": random_actions(start, catalog, random.Random(
                            f"k5-pad-{meta['world_id']}-{j}"), DEPTH, plan[:DEPTH]),
                        "random": random_actions(start, catalog, random.Random(
                            f"k5-rand-{meta['world_id']}-{j}"), DEPTH)}
                for kind, actions in seqs.items():
                    rollouts.append({
                        "id": f"{meta['world_id']}-{j:02d}-{kind}", "world": meta["world_id"],
                        "start": j, "source": source, "path": path, "kind": kind,
                        "actions": actions,
                        "policy_status": status if kind == "policy" else None,
                        "padded": DEPTH - len(plan[:DEPTH]) if kind == "policy" else None,
                        "start_values": read_schema_values(canon, schema),
                        "truth": truth_of(start, game, schema, actions)})
                start.close()
            env.close()
            print(f"{meta['world_id']}: {sum(1 for r in rollouts if r['world'] == meta['world_id'])} rollouts",
                  flush=True)
    out = Path(args.out) if args.out else ROLLOUTS
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_out = {"note": "scripts/kiis_k5_rollout.py build; engine truth and policy plans only",
                "split": args.split, "depth": DEPTH, "p_valid": P_VALID,
                "policy": {"model": args.model, "k": POLICY_K, "h": POLICY_H, "seed": POLICY_SEED,
                           "calls": policy.calls, "repairs": policy.repairs,
                           "fallbacks": policy.fallbacks},
                "revision": rev, "software": runinfo.software()}
    out.write_text(json.dumps({"meta": meta_out, "rollouts": rollouts}) + "\n")
    print(f"{len(rollouts)} rollouts -> {out}  policy repairs={policy.repairs} "
          f"fallbacks={policy.fallbacks}")
    return 0


# ------------------------------------------------------------------ eval

class Threaded:
    """TypedStep over a remote client: one depth's requests in parallel, as the
    closed-loop JEV arm scores its prefixes."""

    def __init__(self, step, workers: int = 8) -> None:
        self.step, self.workers = step, workers

    def predict_batch(self, pairs):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda ca: self.step(*ca), pairs))


def make_step(model: str, schema, ctx: dict):
    from wm.recursive_forecaster import TypedStep
    if model == "A_jev":
        return Threaded(TypedStep(ctx["jev"], schema))
    from wm.llm_backends import GenerativeStep, LlmJudgeClient
    if model.startswith("B"):
        return TypedStep(LlmJudgeClient(ctx["scorer"]), schema)
    return GenerativeStep(ctx["scorer"], schema)


def evaluate(args) -> int:
    from wm.recursive_forecaster import RecursiveForecaster
    rev = runinfo.revision()      # at start: the code this process loaded
    data = json.loads(Path(args.rollouts).read_text())
    rolls = data["rollouts"]
    if args.repeat:
        rolls = [r for i, r in enumerate(rolls) if i % REPEAT_EVERY == 0]
    if args.limit:
        rolls = rolls[:args.limit]
    by_world = defaultdict(list)
    for r in rolls:
        by_world[r["world"]].append(r)

    ctx: dict = {}
    adapter = None
    if args.model == "A_jev":
        from jev_client import JevClient
        ctx["jev"] = JevClient(save_raw=True, budget_usd=args.budget)
    else:
        from agent.policy import Policy
        from wm.llm_backends import QwenScorer
        policy = Policy(args.base, args.device, seed=POLICY_SEED)
        wm_model = policy.model
        if args.model in runinfo.ADAPTER_ARMS:
            if not args.adapter:
                raise SystemExit(f"{args.model} needs --adapter")
            from peft import PeftModel
            wm_model = PeftModel.from_pretrained(policy.model, args.adapter)
            wm_model.eval()
            adapter = runinfo.adapter_info(args.adapter)
        elif args.adapter:
            raise SystemExit(f"{args.model} takes no adapter")
        ctx["scorer"] = QwenScorer(wm_model, policy.tok, args.device, batch=args.wm_batch)

    name = args.model + (f".rep{args.repeat}" if args.repeat else "")
    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    stats = defaultdict(int)
    with tempfile.TemporaryDirectory() as d:
        n_worlds = 1 + max(int("".join(c for c in w if c.isdigit())) for w in by_world)
        for game, meta, env in worlds(data["meta"]["split"], n_worlds, Path(d)):
            todo = by_world.get(meta["world_id"])
            if not todo:
                env.close(); continue
            schema = state_schema(game, meta)
            goal = trap_goal_spec(game, meta)
            fc = RecursiveForecaster(None, schema, goal, step=make_step(args.model, schema, ctx))
            beams, starts = {}, {}
            for r in todo:
                s = replay(env, r["path"])
                canon = canonical_state(list(s.state["_facts"]), game)
                if (read_schema_values(canon, schema) != r["start_values"]
                        or truth_of(s, game, schema, r["actions"]) != r["truth"]):
                    raise SystemExit(f"{r['id']}: stored truth does not replay")
                s.close()
                beams[r["id"]] = [(canon, 1.0)]
                starts[r["id"]] = canon
            fp = fc._fingerprint
            for depth in range(1, DEPTH + 1):
                pending: dict[tuple, tuple[dict, str]] = {}
                for r in todo:
                    a = r["actions"][depth - 1]
                    for state, _ in beams[r["id"]]:
                        key = (fp(state), a)
                        if key not in fc._cache and key not in pending:
                            pending[key] = (state, a)
                if pending:
                    pairs = list(pending.values())
                    for key, (state, _), pred in zip(pending, pairs,
                                                     fc.step.predict_batch(pairs)):
                        fc._store(key, state, pred)
                for r in todo:
                    a = r["actions"][depth - 1]
                    prev_top = max(beams[r["id"]], key=lambda kv: kv[1])[0]
                    p_exec = fc._cache[(fp(prev_top), a)][1]
                    beams[r["id"]], _ = fc._advance_beam(beams[r["id"]], a)
                    top = max(beams[r["id"]], key=lambda kv: kv[1])[0]
                    tr = r["truth"][depth - 1]
                    pv = read_schema_values(top, schema)
                    rows.append({
                        "id": r["id"], "world": r["world"], "kind": r["kind"], "k": depth,
                        "exact": pv == tr["values"],
                        "vars_ok": sum(pv.get(k) == v for k, v in tr["values"].items()),
                        "vars": len(tr["values"]),
                        "facts_exact": top["dynamic_facts"] == tr["dynamic_facts"],
                        "in_beam": any(read_schema_values(s, schema) == tr["values"]
                                       for s, _ in beams[r["id"]]),
                        "executes": tr["executes"], "p_exec": round(float(p_exec), 4)})
            for k in ("requests", "parse_failures", "cache_hits", "contradictions"):
                stats[k] += getattr(fc.stats, k)
            env.close()
            print(f"{name} {meta['world_id']}: {len(todo)} rollouts  "
                  f"{time.time() - t0:.0f}s  requests={stats['requests']}", flush=True)

    (out / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    manifest = {"model": args.model, "repeat": args.repeat, "rollouts": len(rolls),
                "rollouts_file": str(Path(args.rollouts)), "rows": len(rows),
                "arm_hash": runinfo.arm_hash(args.model, adapter),
                "arm_config": runinfo.arm_config(args.model, adapter),
                "stats": dict(stats), "seconds": round(time.time() - t0, 1),
                "revision": rev, "software": runinfo.software()}
    if "jev" in ctx:
        manifest["jev_usd"] = round(ctx["jev"].spent_usd, 4)
    (out / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"done {name}: {len(rows)} rows, {manifest['seconds']}s, {dict(stats)}")
    return 0


# ------------------------------------------------------------------ report

def balanced(pairs: list[tuple[bool, float]]) -> float | None:
    pos = [p >= 0.5 for t, p in pairs if t]
    neg = [p < 0.5 for t, p in pairs if not t]
    if not pos or not neg:
        return None
    return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2


def boot(rows: list[dict], stat, rng: random.Random, n: int = 4000) -> tuple[float, float]:
    by_w = defaultdict(list)
    for r in rows:
        by_w[r["world"]].append(r)
    ws = sorted(by_w)
    vals = []
    for _ in range(n):
        sample = [r for w in (rng.choice(ws) for _ in ws) for r in by_w[w]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def report(args) -> int:
    data = json.loads(Path(args.rollouts).read_text())
    out = Path(args.out) if args.out else OUT
    persist = []
    for r in data["rollouts"]:
        for k, tr in enumerate(r["truth"], start=1):
            persist.append({"id": r["id"], "world": r["world"], "kind": r["kind"], "k": k,
                            "exact": r["start_values"] == tr["values"],
                            "vars_ok": sum(r["start_values"].get(x) == v
                                           for x, v in tr["values"].items()),
                            "vars": len(tr["values"]), "facts_exact": None,
                            "in_beam": r["start_values"] == tr["values"],
                            "executes": tr["executes"], "p_exec": 0.0})
    models = {"persistence": persist}
    for m in MODELS:
        p = out / f"{m}.jsonl"
        if p.exists():
            models[m] = [json.loads(l) for l in open(p) if l.strip()]
    rng = random.Random(0)
    exact = lambda rs: sum(r["exact"] for r in rs) / len(rs) if rs else None
    summary = {"n_rollouts": len(data["rollouts"]), "models": {}}
    print(f"{'model':<12}{'kind':<8}" + "".join(f"{'k=' + str(k):>20}" for k in range(1, DEPTH + 1)))
    for m, rows in models.items():
        summary["models"][m] = {}
        for kind in ("all", "policy", "random"):
            sel = [r for r in rows if kind == "all" or r["kind"] == kind]
            cells, line = {}, f"{m:<12}{kind:<8}"
            for k in range(1, DEPTH + 1):
                rk = [r for r in sel if r["k"] == k]
                lo, hi = boot(rk, exact, rng) if kind == "all" else (None, None)
                cells[k] = {"n": len(rk), "exact": exact(rk), "ci": [lo, hi],
                            "var_acc": sum(r["vars_ok"] for r in rk) / sum(r["vars"] for r in rk),
                            "in_beam": sum(r["in_beam"] for r in rk) / len(rk),
                            "exec_balanced": balanced([(r["executes"], r["p_exec"]) for r in rk])}
                line += f"{cells[k]['exact']:>9.1%}" + (f" [{lo:.0%},{hi:.0%}]" if lo is not None
                                                        else " " * 10)
            summary["models"][m][kind] = cells
            print(line)
    print("\nvariable accuracy / executes balanced accuracy (all rollouts)")
    for m in models:
        c = summary["models"][m]["all"]
        print(f"{m:<12}" + "".join(
            f"   k={k} {c[k]['var_acc']:.1%} / " +
            (f"{c[k]['exec_balanced']:.1%}" if c[k]["exec_balanced"] is not None else "-")
            for k in range(1, DEPTH + 1)))
    # JEV repeats: the same 100 rollouts, three independent passes
    reps = [out / "A_jev.jsonl"] + sorted(out.glob("A_jev.rep*.jsonl"))
    if len(reps) > 1:
        runs = []
        for p in reps:
            keep = {(r["id"], r["k"]): r["exact"] for r in (json.loads(l) for l in open(p) if l.strip())}
            runs.append(keep)
        common = set.intersection(*(set(r) for r in runs))
        flips = sum(len({run[key] for run in runs}) > 1 for key in common)
        per_run = {k: [sum(run[key] for key in common if key[1] == k) /
                       max(1, sum(1 for key in common if key[1] == k)) for run in runs]
                   for k in range(1, DEPTH + 1)}
        summary["jev_repeats"] = {"passes": len(runs), "cells": len(common),
                                  "flip_rate": flips / len(common) if common else None,
                                  "exact_by_pass": per_run}
        print(f"\nJEV repeats: {len(runs)} passes, {len(common)} (rollout, k) cells, "
              f"outcome differs across passes in {flips} ({flips / max(1, len(common)):.1%})")
        for k, v in per_run.items():
            print(f"   k={k}: " + ", ".join(f"{x:.1%}" for x in v))
    for m in MODELS:
        p = out / f"{m}.manifest.json"
        if p.exists():
            man = json.loads(p.read_text())
            summary["models"][m]["cost"] = {"seconds": man["seconds"], "stats": man["stats"],
                                            "jev_usd": man.get("jev_usd"),
                                            "arm_hash": man["arm_hash"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", default="Qwen/Qwen3-4B")
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--split", default="test", choices=("test", "dev"))
    b.add_argument("--worlds", type=int, default=1, help="dev smoke only")
    b.add_argument("--starts", type=int, default=2, help="dev smoke only: starts per world")
    b.add_argument("--out", default=None)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True, choices=MODELS)
    e.add_argument("--base", default="Qwen/Qwen3-4B")
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--adapter", default=None)
    e.add_argument("--wm-batch", type=int, default=8)
    e.add_argument("--budget", type=float, default=2.0)
    e.add_argument("--repeat", type=int, default=0, help="JEV repeat pass number (1, 2)")
    e.add_argument("--limit", type=int, default=0, help="smoke: first N rollouts")
    e.add_argument("--rollouts", default=str(ROLLOUTS))
    e.add_argument("--out", default=None)
    r = sub.add_parser("report")
    r.add_argument("--rollouts", default=str(ROLLOUTS))
    r.add_argument("--out", default=None)
    args = ap.parse_args()
    return {"build": build, "eval": evaluate, "report": report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
