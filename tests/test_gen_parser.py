"""The generative format must round-trip, and no prompt may carry the answer.

1. Round trip. Write the engine's own next state in the generative reply format
   and parse it back: every schema variable and the execute flag must come out
   exactly (K2 gate: 100%). If this fails, a generative arm could be marked
   wrong for a correct reply.
2. Robustness. Replies that are missing the result line, or name two places for
   one object, must be rejected rather than guessed.
3. Leakage. For transitions that change the state, no fact that exists only
   after the command may appear in either prompt, and the typed prompt's state
   must be the current state alone.
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import EnvInfos
from agent.task import state_schema
from env.transitions import world_transitions
from env.worlds import build_trap_world
from forecast import render_facts
from wm.llm_backends import (fact_str, gen_messages, gen_target, parse_generation,
                             typed_messages)
from wm.recursive_forecaster import TypedStep

INFOS = EnvInfos(facts=True, typed_entities=True, possible_admissible_commands=True,
                 admissible_commands=True)


def main():
    d = Path("/tmp/gen_parser"); d.mkdir(exist_ok=True)
    rt_bad = rt_n = leak = leak_n = 0
    for split, idx in [("dev", 3), ("test", 0), ("test", 5), ("train", 11)]:
        game, path, meta = build_trap_world(idx, d, random.Random(0), split=split)
        schema = state_schema(game, meta)
        env = textworld.start(str(path), request_infos=INFOS); env.reset()
        recs = world_transitions(env, game, meta, schema, random.Random(idx), 4, 4)
        env.close()
        for r in recs:
            # 1. round trip
            text = gen_target(r["executes"], r["next_facts"])
            values, p = parse_generation(text, schema, r["canon"])
            want = r["values_next"] if r["executes"] else r["values_now"]
            rt_n += 1
            if values != want or p != (1.0 if r["executes"] else 0.0):
                rt_bad += 1
                print(f"  ROUNDTRIP {meta['world_id']} {r['action']!r}: {values} vs {want}")
            # 3. leakage
            if r["executes"] and r["changed"]:
                leak_n += 1
                cur = r["canon"]["dynamic_facts"] + r["canon"]["static_facts"]
                new_rows = [x for x in r["next_facts"] if x not in cur]
                # every form a fact takes in a prompt: JSON array, predicate
                # text, and the readable sentence render_facts writes
                nxt_canon = dict(r["canon"], dynamic_facts=[x for x in r["next_facts"]
                                                            if x not in r["canon"]["static_facts"]])
                now_sent = set(render_facts(r["canon"]))
                forms = ([json.dumps(x, ensure_ascii=False) for x in new_rows]
                         + [fact_str(x) for x in new_rows]
                         + [s for s in render_facts(nxt_canon) if s not in now_sent])
                gen_prompt = gen_messages(r["canon"], schema, r["action"])[-1]["content"]
                typed_prompts = [typed_messages(s, q)[0][1]["content"]
                                 for s, qs, _ in TypedStep(None, schema).requests(r["canon"], r["action"])
                                 for q in qs.values()]
                hits = [f for f in forms if f in gen_prompt or any(f in t for t in typed_prompts)]
                if hits:
                    leak += 1
                    print(f"  LEAK {meta['world_id']} {r['action']!r}: {hits}")

    # 2. robustness
    game, path, meta = build_trap_world(0, d, random.Random(0), split="test")
    schema = state_schema(game, meta)
    env = textworld.start(str(path), request_infos=INFOS); env.reset()
    r = world_transitions(env, game, meta, schema, random.Random(1), 2, 0)[0]
    env.close()
    good = gen_target(True, r["next_facts"])
    body = good.split("\n", 1)[1]
    bad_cases = {
        "no result line": body,
        "empty": "",
        "two places for the player": good + "\nat(P, r_1)\nat(P, r_0)",
    }
    robust_bad = 0
    for name, text in bad_cases.items():
        if parse_generation(text, schema, r["canon"])[0] is not None:
            robust_bad += 1
            print(f"  ACCEPTED BAD REPLY: {name}")
    as_json = "result: executes\n" + "\n".join(json.dumps(x) for x in r["next_facts"])
    if parse_generation(as_json, schema, r["canon"])[0] != r["values_next"]:
        robust_bad += 1
        print("  REJECTED fact lines written as JSON arrays")

    print(f"\n왕복 {rt_n - rt_bad}/{rt_n}, 나쁜 답 거부 {len(bad_cases) + 1 - robust_bad}/{len(bad_cases) + 1}, "
          f"누출 없음 {leak_n - leak}/{leak_n}")
    ok = rt_bad == 0 and robust_bad == 0 and leak == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
