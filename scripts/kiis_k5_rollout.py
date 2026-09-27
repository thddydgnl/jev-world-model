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
from env.transitions import OBSERVE, solution_path_v2
from env.worlds import build_trap_world, build_trap_world_v2, world_fingerprint

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
# v2 (kiis2026f/실험계획.md §4 V2): world sets from worlds_manifest_v2.json, named
# "v2:<set>" (v2:test, v2:x1, v2:dev). The policy that writes the H=4 sequences and
# the probe behind every arm hash are condition F2's.
F2 = {"levers": ("T3",), "hint": 2, "samples": 2}
# v3 (§4 W): "v3:<set>" (v3:test, v3:x1, v3:dev, v3:x1dev) from worlds_manifest_v3.json,
# condition F3. `retry` (P3) is filled in at W2 from the calibration's choice; P3
# never fires here (no history), but it is part of F3's condition hash.
F3 = {"levers": (), "hint": 2, "samples": 2, "retry": None}


def cond_of(world_set: str) -> dict | None:
    """The versioned condition of a world set, or None for v1."""
    if world_set.startswith("v3:"):
        if F3["retry"] is None:
            raise SystemExit("v3 is not frozen yet (kiis2026f/실험계획.md §4 W2)")
        return F3
    return F2 if world_set.startswith("v2:") else None


# ------------------------------------------------------------------ worlds

def world_set_info(world_set: str) -> tuple[str, tuple | None, dict]:
    """(split, levers or None for v1, world_id -> fingerprint) of a world set."""
    if world_set.startswith(("v2:", "v3:")):
        s = json.loads((ROOT / f"kiis2026f/worlds_manifest_{world_set[:2]}.json").read_text()
                       )["sets"][world_set[3:]]
        return s["split"], tuple(s["levers"]), {w["world_id"]: w["fingerprint"] for w in s["worlds"]}
    ref = json.loads((ROOT / "kiis2026f/worlds_manifest.json").read_text())["worlds"][world_set]
    return world_set, None, {w["world_id"]: w["fingerprint"] for w in ref}


def worlds(world_set: str, n: int, tmp: Path):
    split, levers, ref = world_set_info(world_set)
    for i in range(n):
        if levers is None:
            game, path, meta = build_trap_world(i, tmp, random.Random(0), split=split)
        else:
            game, path, meta = build_trap_world_v2(i, tmp, split, levers)
        if world_fingerprint(game) != ref[meta["world_id"]]:
            raise SystemExit(f"{meta['world_id']} differs from the worlds manifest")
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
            sol = solution_path_v2(meta, rng)
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
    world_set = args.world_set or args.split
    cond = cond_of(world_set)
    v2 = cond is not None                                   # v2 or v3
    smoke = world_set.split(":")[-1].endswith("dev")       # v2:dev, v2:x1dev, v1 dev
    policy = Policy(args.model, args.device, seed=POLICY_SEED)
    if v2:
        policy.hint, policy.samples = cond["hint"], cond["samples"]
    n_worlds = args.worlds if smoke else len(world_set_info(world_set)[2])
    total = args.starts_total
    per_world = [total // n_worlds + (1 if i < total % n_worlds else 0) for i in range(n_worlds)]
    rollouts = []
    with tempfile.TemporaryDirectory() as d:
        for i, (game, meta, env) in enumerate(worlds(world_set, n_worlds, Path(d))):
            schema = state_schema(game, meta)
            goal = trap_goal_spec(game, meta)
            catalog = static_catalog(game)
            rng = random.Random(f"kiis-k5-{world_set}-{meta['world_id']}" if v2
                                else f"kiis-k5-{meta['world_id']}")
            n = args.starts if smoke else per_world[i]
            for j, (source, path) in enumerate(start_paths(env, meta, rng, n)):
                start = replay(env, path)
                canon = canonical_state(list(start.state["_facts"]), game)
                policy.set_context(meta["world_id"], f"k5-{world_set}-{j}" if v2 else f"k5-{j}", 0)
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
    split, levers, _ = world_set_info(world_set)
    meta_out = {"note": "scripts/kiis_k5_rollout.py build; engine truth and policy plans only",
                "split": split, "world_set": world_set, "levers": list(levers) if levers is not None else None,
                "depth": DEPTH, "p_valid": P_VALID,
                "policy": {"model": args.model, "k": POLICY_K, "h": POLICY_H, "seed": POLICY_SEED,
                           "hint": policy.hint, "samples": policy.samples,
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


def checked_start(env, game, schema, r: dict):
    """Rebuild a rollout's start from its path and check the stored truth.
    Returns an engine copy at the start (caller closes it) and its canonical state."""
    s = replay(env, r["path"])
    canon = canonical_state(list(s.state["_facts"]), game)
    if (read_schema_values(canon, schema) != r["start_values"]
            or truth_of(s, game, schema, r["actions"]) != r["truth"]):
        raise SystemExit(f"{r['id']}: stored truth does not replay")
    return s, canon


def _predict(fc, pending: dict, failed_keys: set) -> None:
    """Ask the step backend for every pending (state, action) in one batch."""
    if pending:
        pairs = list(pending.values())
        for key, (state, _), pred in zip(pending, pairs, fc.step.predict_batch(pairs)):
            fc._store(key, state, pred)
            if pred.parse_failed:
                failed_keys.add(key)


def _row(r: dict, k: int, top: dict, beam_states: list, p_exec: float, bad: bool,
         schema) -> dict:
    tr = r["truth"][k - 1]
    pv = read_schema_values(top, schema)
    return {"id": r["id"], "world": r["world"], "kind": r["kind"], "k": k,
            "exact": pv == tr["values"] and not bad,
            "vars_ok": 0 if bad else sum(pv.get(x) == v for x, v in tr["values"].items()),
            "vars": len(tr["values"]),
            "facts_exact": top["dynamic_facts"] == tr["dynamic_facts"] and not bad,
            "in_beam": not bad and any(read_schema_values(b, schema) == tr["values"]
                                       for b in beam_states),
            "executes": tr["executes"], "p_exec": round(float(p_exec), 4),
            "exec_ok": (p_exec >= 0.5) == tr["executes"] and not bad,
            "parse_failed": bad}


def free_running_rows(fc, env, game, schema, todo: list) -> list[dict]:
    """The model rolls each sequence out on its own predictions (the planner's
    recursive beam), compared with the truth at every depth.

    §6.2: a generative reply that did not parse counts as wrong. The fallback
    (state unchanged, not executed) would otherwise score as a correct
    prediction whenever the true state did not change. From the step whose
    reply failed, the rest of that rollout counts as wrong."""
    beams = {}
    for r in todo:
        s, canon = checked_start(env, game, schema, r)
        s.close()
        beams[r["id"]] = [(canon, 1.0)]
    fp = fc._fingerprint
    failed_keys: set[tuple] = set()
    broken = {r["id"]: False for r in todo}
    rows = []
    for depth in range(1, DEPTH + 1):
        pending: dict[tuple, tuple[dict, str]] = {}
        for r in todo:
            a = r["actions"][depth - 1]
            for state, _ in beams[r["id"]]:
                key = (fp(state), a)
                if key not in fc._cache and key not in pending:
                    pending[key] = (state, a)
        _predict(fc, pending, failed_keys)
        for r in todo:
            a = r["actions"][depth - 1]
            prev_top = max(beams[r["id"]], key=lambda kv: kv[1])[0]
            used = (fp(prev_top), a)
            p_exec = fc._cache[used][1]
            broken[r["id"]] |= used in failed_keys
            beams[r["id"]], _ = fc._advance_beam(beams[r["id"]], a)
            top = max(beams[r["id"]], key=lambda kv: kv[1])[0]
            rows.append(_row(r, depth, top, [b for b, _ in beams[r["id"]]], p_exec,
                             broken[r["id"]], schema))
    return rows


def teacher_forced_rows(fc, env, game, schema, todo: list) -> list[dict]:
    """Every step predicted from the TRUE state before it (engine replay), one
    step at a time: what each depth would score if the model's own earlier
    predictions had always been right. The gap to the free-running rollout at
    the same depth is the error the rollout adds by feeding on itself (H8).

    The top of a one-step beam is the executed branch when p >= 0.5, else the
    unchanged state, as in _advance_beam. A reply that failed to parse is wrong
    for its own step only."""
    from wm.recursive_forecaster import PRUNE_W
    fp = fc._fingerprint
    targets, pending, failed_keys = [], {}, set()
    for r in todo:
        s, canon = checked_start(env, game, schema, r)
        for k, a in enumerate(r["actions"], start=1):
            key = (fp(canon), a)
            if key not in fc._cache and key not in pending:
                pending[key] = (canon, a)
            targets.append((r, k, key, canon))
            s.step(a)
            canon = canonical_state(list(s.state["_facts"]), game)
        s.close()
    _predict(fc, pending, failed_keys)
    rows = []
    for r, k, key, prev in targets:
        nxt, p_exec = fc._cache[key]
        top = nxt if p_exec >= 0.5 else prev
        beam_states = [b for b, w in ((nxt, p_exec), (prev, 1 - p_exec)) if w > PRUNE_W] or [top]
        rows.append(_row(r, k, top, beam_states, p_exec, key in failed_keys, schema))
    return rows


def evaluate(args) -> int:
    from wm.recursive_forecaster import RecursiveForecaster
    rev = runinfo.revision()      # at start: the code this process loaded
    data = json.loads(Path(args.rollouts).read_text())
    world_set = data["meta"].get("world_set", data["meta"]["split"])
    cond = cond_of(world_set)
    if cond is not None:
        runinfo.configure_v2(cond["levers"], cond["hint"], cond["samples"], bool(cond.get("retry")))
    if args.mode == "tf" and (args.repeat or args.beam != 4):
        raise SystemExit("teacher-forced takes neither --repeat nor --beam")
    if args.beam != 4:
        if args.model not in ("A_jev", "B0_typed", "B_typed"):
            raise SystemExit("--beam applies to the typed models; generative rollouts have one branch")
        import wm.recursive_forecaster as rf
        rf.BEAM_K = args.beam          # read at every _advance_beam call
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

    name = (args.model + (".tf" if args.mode == "tf" else "")
            + (f".beam{args.beam}" if args.beam != 4 else "")
            + (f".rep{args.repeat}" if args.repeat else ""))
    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    stats = defaultdict(int)
    with tempfile.TemporaryDirectory() as d:
        n_worlds = 1 + max(int("".join(c for c in w if c.isdigit())) for w in by_world)
        for game, meta, env in worlds(world_set, n_worlds, Path(d)):
            todo = by_world.get(meta["world_id"])
            if not todo:
                env.close(); continue
            schema = state_schema(game, meta)
            goal = trap_goal_spec(game, meta)
            fc = RecursiveForecaster(None, schema, goal, step=make_step(args.model, schema, ctx))
            walk = teacher_forced_rows if args.mode == "tf" else free_running_rows
            rows += walk(fc, env, game, schema, todo)
            for k in ("requests", "parse_failures", "cache_hits", "contradictions"):
                stats[k] += getattr(fc.stats, k)
            env.close()
            print(f"{name} {meta['world_id']}: {len(todo)} rollouts  "
                  f"{time.time() - t0:.0f}s  requests={stats['requests']}", flush=True)

    (out / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    manifest = {"model": args.model, "mode": args.mode, "beam": args.beam,
                "repeat": args.repeat, "rollouts": len(rolls),
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

def exec_ok(r: dict) -> bool:
    """Executes judgement right? Rows written before the parse-failure rule
    carry no exec_ok and are read from p_exec."""
    return r["exec_ok"] if "exec_ok" in r else (r["p_exec"] >= 0.5) == r["executes"]


def balanced(rows: list[dict]) -> float | None:
    pos = [exec_ok(r) for r in rows if r["executes"]]
    neg = [exec_ok(r) for r in rows if not r["executes"]]
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
        for suffix in ("", ".beam1", ".tf"):      # free-running, beam-1 control, teacher-forced
            p = out / f"{m}{suffix}.jsonl"
            if p.exists():
                models[m + suffix] = [json.loads(l) for l in open(p) if l.strip()]
    rng = random.Random(0)
    exact = lambda rs: sum(r["exact"] for r in rs) / len(rs) if rs else None
    summary = {"n_rollouts": len(data["rollouts"]), "models": {}}
    print(f"{'model':<16}{'kind':<8}" + "".join(f"{'k=' + str(k):>20}" for k in range(1, DEPTH + 1)))
    for m, rows in models.items():
        summary["models"][m] = {}
        for kind in ("all", "policy", "random"):
            sel = [r for r in rows if kind == "all" or r["kind"] == kind]
            cells, line = {}, f"{m:<16}{kind:<8}"
            for k in range(1, DEPTH + 1):
                rk = [r for r in sel if r["k"] == k]
                lo, hi = boot(rk, exact, rng) if kind == "all" else (None, None)
                cells[k] = {"n": len(rk), "exact": exact(rk), "ci": [lo, hi],
                            "var_acc": sum(r["vars_ok"] for r in rk) / sum(r["vars"] for r in rk),
                            "in_beam": sum(r["in_beam"] for r in rk) / len(rk),
                            "exec_balanced": balanced(rk),
                            "parse_broken": sum(r.get("parse_failed", False) for r in rk) / len(rk)}
                line += f"{cells[k]['exact']:>9.1%}" + (f" [{lo:.0%},{hi:.0%}]" if lo is not None
                                                        else " " * 10)
            summary["models"][m][kind] = cells
            print(line)
    print("\nvariable accuracy / executes balanced accuracy (all rollouts)")
    for m in models:
        c = summary["models"][m]["all"]
        print(f"{m:<16}" + "".join(
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


def compare(args) -> int:
    """H7 (kiis2026f/실험계획.md §5): does a model trained on the task's structure
    lose more than frozen JEV on a structure it never saw? Per model and world,
    the drop in mean exact match over k=1..4 from the in-distribution set (K5v2)
    to X1; the worlds are the same test names, so drops pair by world. Reports
    each model's mean drop and, against JEV, the paired difference with a
    world-bootstrap 95% CI."""
    def per_world(d: Path, m: str) -> dict[str, float]:
        rows = [json.loads(l) for l in open(d / f"{m}.jsonl") if l.strip()]
        acc = defaultdict(list)
        for r in rows:
            acc[r["world"]].append(r["exact"])
        return {w: sum(v) / len(v) for w, v in acc.items()}
    base, x1 = Path(args.base), Path(args.x1)
    drops = {}
    for m in MODELS:
        if (base / f"{m}.jsonl").exists() and (x1 / f"{m}.jsonl").exists():
            a, b = per_world(base, m), per_world(x1, m)
            drops[m] = {w: a[w] - b[w] for w in sorted(set(a) & set(b))}
    rng = random.Random(0)
    res = {"models": {}, "vs_A_jev": {}}
    for m, d in drops.items():
        res["models"][m] = {"mean_drop": sum(d.values()) / len(d), "worlds": len(d)}
    if "A_jev" in drops:
        for m in (x for x in drops if x != "A_jev"):
            ws = sorted(set(drops[m]) & set(drops["A_jev"]))
            diff = {w: drops[m][w] - drops["A_jev"][w] for w in ws}
            boots = sorted(sum(diff[rng.choice(ws)] for _ in ws) / len(ws) for _ in range(4000))
            res["vs_A_jev"][m] = {"diff": sum(diff.values()) / len(ws),
                                  "ci": [boots[100], boots[3899]], "worlds": len(ws)}
    for m, v in res["models"].items():
        extra = res["vs_A_jev"].get(m)
        print(f"{m:<10} drop {v['mean_drop']:+.1%}" + (
            f"   − A_jev drop: {extra['diff']:+.1%} [{extra['ci'][0]:+.1%}, {extra['ci'][1]:+.1%}]"
            if extra else ""))
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


def accumulation(args) -> int:
    """H8 (kiis2026f/실험계획.md §5): does the typed style lose less to its own
    errors than the generative one, on the same Qwen?

    For each model run with a teacher-forced twin (m.tf.jsonl) in --out:
    FR_k and TF_k (state exact match, free-running vs teacher-forced), the gap
    TF_k - FR_k, and the primary score G = mean of the gap over k = 2..4 (k = 1
    has no earlier prediction to feed on, so its gap is 0 by construction and
    is printed as a check). Also, in the free run, P(right at k | right at
    k-1) and FR_4 / FR_1; and G again over rollouts with no parse failure.
    Pairs (generative - typed): D0 - B0, D - B, and the same against the
    typed runs with beam 1 (typed format without its probability beam).
    CIs: world bootstrap (4,000), both models resampled together."""
    out = Path(args.out)
    def load(name):
        p = out / f"{name}.jsonl"
        return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else None
    variants = {}
    for m in MODELS:
        tf = load(f"{m}.tf")
        for suffix in ("", ".beam1"):
            fr = load(m + suffix)
            if fr is not None and tf is not None:
                variants[m + suffix] = (fr, tf)

    def g_score(fr, tf, content_only=False):
        """Mean over k >= 2 of (teacher-forced exact - free-running exact);
        rows may repeat (bootstrap), and each repeat counts."""
        broken = ({r["id"] for r in fr + tf if r.get("parse_failed")} if content_only else set())
        tfx = {(r["id"], r["k"]): r["exact"] for r in tf}
        gaps = [tfx[(r["id"], r["k"])] - r["exact"] for r in fr
                if r["k"] >= 2 and r["id"] not in broken]
        return sum(gaps) / len(gaps) if gaps else None

    rng = random.Random(0)
    res = {"models": {}, "pairs": {}}
    print(f"{'model':<16}{'FR k1..4':>30}  {'TF k1..4':>30}{'gap k1':>8}{'G':>8}{'G content':>11}"
          f"{'P(k|k-1) 2..4':>18}{'FR4/FR1':>9}")
    for name, (fr, tf) in variants.items():
        frx = defaultdict(dict)
        for r in fr:
            frx[r["id"]][r["k"]] = r["exact"]
        FR = {k: sum(v[k] for v in frx.values()) / len(frx) for k in range(1, DEPTH + 1)}
        tfd = defaultdict(dict)
        for r in tf:
            tfd[r["id"]][r["k"]] = r["exact"]
        TF = {k: sum(v[k] for v in tfd.values()) / len(tfd) for k in range(1, DEPTH + 1)}
        cond = [sum(v[k] for v in frx.values() if v[k - 1]) / max(1, sum(v[k - 1] for v in frx.values()))
                for k in range(2, DEPTH + 1)]
        g, gc = g_score(fr, tf), g_score(fr, tf, content_only=True)
        res["models"][name] = {"FR": FR, "TF": TF, "gap_k1": TF[1] - FR[1], "G": g,
                               "G_content_only": gc, "cond_2_4": cond,
                               "retention": FR[DEPTH] / FR[1] if FR[1] else None}
        print(f"{name:<16}{'  '.join(f'{FR[k]:.1%}' for k in FR):>30}  {'  '.join(f'{TF[k]:.1%}' for k in TF):>30}"
              f"{TF[1] - FR[1]:>+8.1%}{g:>+8.1%}" + (f"{gc:>+11.1%}" if gc is not None else f"{'-':>11}")
              + f"{'  '.join(f'{c:.2f}' for c in cond):>18}"
              + (f"{FR[DEPTH] / FR[1]:>9.2f}" if FR[1] else f"{'-':>9}"))
    pairs = [("D0_gen", "B0_typed"), ("D_gen", "B_typed"),
             ("D0_gen", "B0_typed.beam1"), ("D_gen", "B_typed.beam1")]
    print("\nG(generative) - G(typed), world bootstrap 95% CI (> 0: typed accumulates less)")
    for gen, typ in pairs:
        if gen not in variants or typ not in variants:
            continue
        (fg, tg), (ft, tt) = variants[gen], variants[typ]
        ws = sorted({r["world"] for r in fg} & {r["world"] for r in ft})
        by = lambda rows: {w: [r for r in rows if r["world"] == w] for w in ws}
        fgw, tgw, ftw, ttw = by(fg), by(tg), by(ft), by(tt)
        def diff(sample, content_only=False):
            a = g_score([r for w in sample for r in fgw[w]], [r for w in sample for r in tgw[w]],
                        content_only=content_only)
            b = g_score([r for w in sample for r in ftw[w]], [r for w in sample for r in ttw[w]],
                        content_only=content_only)
            return None if a is None or b is None else a - b
        for label, co in (("all", False), ("content only", True)):
            point = diff(ws, co)
            boots = sorted(x for x in (diff([rng.choice(ws) for _ in ws], co) for _ in range(4000))
                           if x is not None)
            ci = [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]]
            res["pairs"][f"{gen} - {typ} ({label})"] = {"diff": point, "ci": ci, "worlds": len(ws)}
            print(f"  {gen} - {typ:<16} {label:<13} {point:+.1%}  [{ci[0]:+.1%}, {ci[1]:+.1%}]")
    (out / "accumulation_h8.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", default="Qwen/Qwen3-4B")
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--split", default="test", choices=("test", "dev"), help="v1 world sets")
    b.add_argument("--world-set", default=None,
                   help="overrides --split: v2:test, v2:x1, v2:dev (worlds_manifest_v2.json) or the same in v3")
    b.add_argument("--starts-total", type=int, default=N_STARTS,
                   help="start states over all worlds (not for dev smoke)")
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
    e.add_argument("--mode", choices=("fr", "tf"), default="fr",
                   help="fr: free-running rollout (default); tf: teacher-forced, every step "
                        "from the true previous state (H8)")
    e.add_argument("--beam", type=int, default=4,
                   help="rollout beam width for the typed models (1 = one branch, H8 control)")
    e.add_argument("--rollouts", default=str(ROLLOUTS))
    e.add_argument("--out", default=None)
    r = sub.add_parser("report")
    r.add_argument("--rollouts", default=str(ROLLOUTS))
    r.add_argument("--out", default=None)
    h = sub.add_parser("accumulation", help="H8: free-running vs teacher-forced error accumulation")
    h.add_argument("--out", required=True, help="an eval output directory with *.tf.jsonl twins")
    c = sub.add_parser("compare", help="H7: in-distribution vs X1 accuracy drops")
    c.add_argument("--base", required=True, help="K5v2 output directory")
    c.add_argument("--x1", required=True, help="X1 output directory")
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    return {"build": build, "eval": evaluate, "report": report,
            "compare": compare, "accumulation": accumulation}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
