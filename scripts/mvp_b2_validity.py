"""Can JEV predict ACTION VALIDITY directly?

MVP-B asked for the endpoint of a sequence and found JEV assumes every command
succeeds. But preconditions were only ever implicit in those questions. Here we
ask about them head-on, which is a judgement about the CURRENT state — the thing
JEV scored 160/160 on in the 0-step condition.

This matters because AB3 showed 97% of the available headroom is action
validity. If JEV can answer this, arm A recovers the part that counts.

  V0  "Would `a` execute right now?"                    (current-state judgement)
  V1  "After attempting `a1`, would `a2` execute?"      (one step of lookahead)

Ground truth is TextWorld's own admissible_commands at the relevant state.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import EnvInfos

from env.serialize import canonical_state
from env.worlds import build_world
from forecast import render_facts
from jev_client import JevClient

INFOS = EnvInfos(facts=True, typed_entities=True, possible_admissible_commands=True,
                 last_action=True, admissible_commands=True)
N_WORLDS, ROOTS_PER_WORLD, SEED = 8, 4, 20260921
PER_ROOT = 10          # actions probed per root, balanced valid/invalid
OUT = Path("artifacts/mvp_b2")


def pick_actions(env, catalog, rng):
    """Balanced probe set: half executable, half not, drawn from the same
    static catalog so surface form cannot give the answer away."""
    adm = set(env.state["admissible_commands"])
    ok = sorted(a for a in catalog if a in adm)
    bad = sorted(a for a in catalog if a not in adm)
    half = PER_ROOT // 2
    return (rng.sample(ok, min(half, len(ok)))
            + rng.sample(bad, min(PER_ROOT - half, len(bad))))


def question(cmd: str, after: list[str] | None) -> dict:
    if after:
        frame = (f"Starting from `current_state`, first attempt these commands "
                 f"in order: {' then '.join(repr(a) for a in after)}. "
                 f"A command that cannot run fails and changes nothing. "
                 f"THEN consider the command below.")
    else:
        frame = "Consider the command below against `current_state` as it is now."
    return {
        "type": "choice",
        "instructions": (
            f"{frame} Command: `{cmd}`. Would this command actually execute, "
            f"or would it fail because its requirements are not met in that "
            f"state? Answer about whether it RUNS, not whether it is useful."),
        "criteria": {
            "executes": "The command runs and takes effect.",
            "fails": "The command cannot run; its preconditions are unmet, so "
                     "nothing changes.",
        },
    }


def main() -> int:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    wdir = Path("data/mvp_a/worlds")
    jobs = []

    for w in range(N_WORLDS):
        game, path, meta = build_world(w, wdir, rng)
        catalog = sorted(set(game.possible_admissible_commands))
        env = textworld.start(str(path), request_infos=INFOS)
        env.reset()
        for r in range(ROOTS_PER_WORLD):
            root = env.copy()
            for _ in range(rng.randint(0, 3)):
                adm = [c for c in root.state["admissible_commands"]
                       if not c.startswith(("look", "inventory", "examine"))]
                if not adm:
                    break
                root.step(rng.choice(adm))

            facts = list(root.state["_facts"])
            state = canonical_state(facts, game)
            root_id = f"{meta['world_id']}_r{r}"

            # --- V0: validity now
            probes = pick_actions(root, catalog, rng)
            jobs.append({"cond": "V0", "root_id": root_id, "state": state,
                         "after": [], "probes": probes,
                         "gt": {a: a in root.state["admissible_commands"] for a in probes}})

            # --- V1: validity after one command
            first = rng.choice([a for a in catalog if a in root.state["admissible_commands"]])
            nxt = root.copy()
            nxt.step(first)
            probes2 = pick_actions(nxt, catalog, rng)
            jobs.append({"cond": "V1", "root_id": root_id, "state": state,
                         "after": [first], "probes": probes2,
                         "gt": {a: a in nxt.state["admissible_commands"] for a in probes2}})
            nxt.close()
            root.close()
        env.close()

    print(f"roots={N_WORLDS*ROOTS_PER_WORLD}  requests={len(jobs)}  "
          f"questions={sum(len(j['probes']) for j in jobs)}")

    rows = []
    with JevClient(save_raw=True) as jev:
        def run(job):
            st = {"current_state": {"player_location": job["state"]["player_room"],
                                    "entities": job["state"]["entities"],
                                    "facts_readable": render_facts(job["state"])},
                  "pending_commands": job["after"]}
            qs = {f"q{i}": question(a, job["after"]) for i, a in enumerate(job["probes"])}
            out = jev.ask(st, qs, tag=job["cond"])
            for i, a in enumerate(job["probes"]):
                ans = out["answers"][f"q{i}"]
                rows.append({"cond": job["cond"], "root_id": job["root_id"],
                             "action": a, "gt": bool(job["gt"][a]),
                             "pred": ans["choice"] == "executes",
                             "p_exec": float(ans["probabilities"]["executes"])})
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(run, jobs))
        calls, toks = jev.calls, jev.input_tokens

    (OUT / "validity.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")

    print(f"calls={calls} tokens={toks:,} cost=${toks/1e6*0.042:.4f}\n")
    print("=" * 62)
    for cond in ("V0", "V1"):
        v = [r for r in rows if r["cond"] == cond]
        if not v:
            continue
        acc = sum(r["pred"] == r["gt"] for r in v) / len(v)
        tp = sum(r["pred"] and r["gt"] for r in v)
        fp = sum(r["pred"] and not r["gt"] for r in v)
        fn = sum((not r["pred"]) and r["gt"] for r in v)
        tn = sum((not r["pred"]) and not r["gt"] for r in v)
        base = max(tp + fn, tn + fp) / len(v)
        name = {"V0": "지금 실행 가능한가", "V1": "한 행동 뒤 실행 가능한가"}[cond]
        print(f"[{cond}] {name}   n={len(v)}")
        print(f"     정확도 {acc:.1%}   (다수 class 기준선 {base:.1%})")
        print(f"     실제 가능한 것 맞힘  {tp}/{tp+fn} = {tp/max(tp+fn,1):.1%}")
        print(f"     실제 불가한 것 맞힘  {tn}/{tn+fp} = {tn/max(tn+fp,1):.1%}")
        print(f"     오경보(불가한데 가능하다고 함) {fp}")
        print()
    print("=" * 62)
    v0 = [r for r in rows if r["cond"] == "V0"]
    if v0:
        a0 = sum(r["pred"] == r["gt"] for r in v0) / len(v0)
        print("판정: " + ("유효성을 직접 물으면 JEV가 답할 수 있다 "
                         "-> arm A 가 headroom 의 큰 몫을 회수할 수 있음"
                         if a0 >= 0.90 else
                         "직접 물어도 어렵다 -> precondition 맹점은 질문 형식의 문제가 아님"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
