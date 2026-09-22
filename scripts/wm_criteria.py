"""Do the recursive predictions actually meet the definition of a world model?

아이디어.md §4 sets three operational conditions. Condition 1 is already
measured (changed-fact accuracy against persistence, action-absent control).
The other two have only ever been satisfied *by construction*:

  2. order sensitivity — the pre-registered probe in 설계.md §9 failed 0/18 on
     the direct-endpoint formulation. It has never been run on the recursive
     one, where each step is conditioned on a reconstructed state.

  3. goal independence — the recursion predicts the whole state, so one rollout
     should score against any goal with no further calls. Being able to is not
     the same as having shown it.

This runs both.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import EnvInfos

from agent.task import (state_schema, goal_from_rows, trap_goal_spec,
                        read_schema_values, static_catalog)
from env.serialize import canonical_state
from env.worlds import build_world, build_trap_world
from jev_client import JevClient
from wm.recursive_forecaster import RecursiveForecaster

INFOS = EnvInfos(facts=True, typed_entities=True, possible_admissible_commands=True,
                 last_action=True, admissible_commands=True)
OUT = Path("artifacts/wm_criteria")


# ---------------------------------------------------------------- condition 2

def order_sensitivity(jev) -> dict:
    """Re-run 설계.md §9's probe through the recursive forecaster.

    The pairs are MVP-A's: same root, `open box`+`take key` in both orders, kept
    only where the engine's own endpoints differ. Direct-endpoint questions
    scored 0/18 on these.
    """
    recs = [json.loads(l) for l in
            Path("data/mvp_a/prefixes.jsonl").read_text().splitlines()]
    by = defaultdict(dict)
    for r in recs:
        if r["horizon"] == 2 and r["seq_name"] in ("open_then_take", "take_then_open"):
            by[(r["world_id"], r["root_id"])][r["seq_name"]] = r

    rng = random.Random(20260921)
    games = {}
    for w in range(8):
        g, _, meta = build_world(w, Path("data/mvp_a/worlds"), rng)
        games[meta["world_id"]] = (g, meta)

    pairs, exact, single = 0, 0, 0
    rows = []
    for (wid, _rid), d in sorted(by.items()):
        if len(d) != 2:
            continue
        a, b = d["open_then_take"], d["take_then_open"]
        gt_a, gt_b = a["labels"]["q_key_parent"], b["labels"]["q_key_parent"]
        if gt_a == gt_b:                      # not a discriminating pair
            continue
        game, meta = games[wid]
        schema = state_schema(game, meta)
        fc = RecursiveForecaster(jev, schema, {"atoms": []})
        key_var = next(q["id"] for q in schema
                       if q["kind"] == "parent" and meta["key"] in q["ask"])

        got = {}
        for name, rec in (("open_then_take", a), ("take_then_open", b)):
            beam, _ = fc.rollout(rec["current_state"], tuple(rec["actions"]))
            top = max(beam, key=lambda kv: kv[1])[0]
            got[name] = read_schema_values(top, schema)[key_var]

        pairs += 1
        ok_a = got["open_then_take"] == gt_a
        ok_b = got["take_then_open"] == gt_b
        exact += ok_a and ok_b
        single += ok_a or ok_b
        rows.append({"world": wid, "gt": [gt_a, gt_b],
                     "pred": [got["open_then_take"], got["take_then_open"]],
                     "both": bool(ok_a and ok_b)})

    print(f"\n{'='*66}\n조건 2 — 순서 민감도 (설계.md §9 사전 등록 검사)\n{'='*66}")
    print(f"  판별 pair {pairs}개")
    print(f"  둘 다 정답 (pair-exact)  {exact}/{pairs} = {exact/max(pairs,1):.1%}")
    print(f"  적어도 하나 정답         {single}/{pairs} = {single/max(pairs,1):.1%}")
    print(f"\n  같은 검사, direct-endpoint 방식: 0/18 = 0.0%   (MVP-B)")
    for r in rows[:4]:
        m = "OK " if r["both"] else "X  "
        print(f"    {m}{r['world']}  정답 {r['gt']}  예측 {r['pred']}")
    return {"pairs": pairs, "exact": exact, "single": single, "rows": rows}


# ---------------------------------------------------------------- condition 3

def goal_reuse(jev) -> dict:
    """One rollout, two goals, zero extra calls.

    Arm A would need a fresh set of questions and a fresh round of requests for
    the second goal, because it only ever asks about the goal's own atoms.
    """
    rng = random.Random(4242)
    rows, calls_before_total, extra_total = [], 0, 0
    agree_1 = agree_2 = n = 0

    for w in range(6):
        Path("/tmp/wmcrit").mkdir(parents=True, exist_ok=True)
        game, path, meta = build_trap_world(w, Path("/tmp/wmcrit"), rng)
        schema = state_schema(game, meta)
        ids = {i.name: v for v, i in game.infos.items() if i.name}
        goal1 = trap_goal_spec(game, meta)
        # A different goal over the SAME state variables.
        goal2 = {"id": "stash_key", "text": f"the {meta['key']} is in the {meta['box']}",
                 "atoms": [("in", ids[meta["key"]], ids[meta["box"]]),
                           ("open", ids[meta["box"]])]}
        catalog = static_catalog(game)
        env = textworld.start(str(path), request_infos=INFOS); env.reset()

        for _ in range(3):
            root = env.copy()
            for _ in range(rng.randint(0, 3)):
                adm = [c for c in root.state["admissible_commands"]
                       if not c.startswith(("look", "inventory", "examine"))]
                if adm:
                    root.step(rng.choice(adm))
            canon = canonical_state(list(root.state["_facts"]), game)
            actions = tuple(rng.choice(catalog) for _ in range(2))

            fc = RecursiveForecaster(jev, schema, goal1)
            before = jev.calls
            beam, _ = fc.rollout(canon, actions)
            spent = jev.calls - before

            # Score the SAME beam against both goals. No model call happens here.
            mark = jev.calls
            s1 = sum(w_ * goal_from_rows(s["dynamic_facts"], goal1)[0] for s, w_ in beam)
            s2 = sum(w_ * goal_from_rows(s["dynamic_facts"], goal2)[0] for s, w_ in beam)
            extra = jev.calls - mark

            b = root.copy()
            for a in actions:
                b.step(a)
            truth = canonical_state(list(b.state["_facts"]), game)
            t1 = goal_from_rows(truth["dynamic_facts"], goal1)[0]
            t2 = goal_from_rows(truth["dynamic_facts"], goal2)[0]
            b.close(); root.close()

            n += 1
            agree_1 += (s1 >= 0.5) == t1
            agree_2 += (s2 >= 0.5) == t2
            calls_before_total += spent
            extra_total += extra
            rows.append({"world": meta["world_id"], "calls": spent, "extra": extra,
                         "goal1": [round(s1, 2), t1], "goal2": [round(s2, 2), t2]})
        env.close()

    print(f"\n{'='*66}\n조건 3 — 목표 독립성 (아이디어.md §4, §18 목표 재가중)\n{'='*66}")
    print(f"  rollout {n}개   목표 1개당 호출 {calls_before_total/max(n,1):.1f}")
    print(f"  **두 번째 목표를 채점하는 데 든 추가 호출: {extra_total}**")
    print(f"  목표 1 판정이 엔진과 일치  {agree_1}/{n} = {agree_1/max(n,1):.1%}")
    print(f"  목표 2 판정이 엔진과 일치  {agree_2}/{n} = {agree_2/max(n,1):.1%}")
    print(f"\n  arm A 는 목표가 바뀌면 질문을 새로 만들고 전량 재호출해야 한다.")
    return {"n": n, "calls": calls_before_total, "extra": extra_total,
            "agree_1": agree_1, "agree_2": agree_2}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with JevClient(save_raw=True, budget_usd=0.60) as jev:
        c2 = order_sensitivity(jev)
        c3 = goal_reuse(jev)
        print(f"\n{'='*66}")
        print(f"JEV: {jev.calls} calls, {jev.input_tokens:,} tokens, "
              f"${jev.spent_usd:.3f}")
    (OUT / "results.json").write_text(json.dumps({"condition2": c2, "condition3": c3},
                                                 indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
