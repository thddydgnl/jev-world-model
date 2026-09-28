"""v3 (kiis2026f/실험계획.md §4 W): new test vocabulary, P3, and hashes.

1. vocabulary: no word of a test3 name occurs in any other name the builders
   know (TRAP_VOCAB, TRAP_NAMES, NAME_POOL, V2_DEV_EXTRA), apart from the head
   nouns key/door; every test3 world, with no lever and with all of them, runs
   its solution path to the goal and shares no word between its own names.
2. P3 in the prompt: without `stuck` the prompt is byte for byte the old one;
   with it, one note naming the failed commands sits before the instructions.
3. P3 in the runner, with a scripted policy on a real dev world: it is asked
   only when every candidate's first command already failed from this state,
   never for C, and then the new command is tried (C_fm and oracle); with P3
   off the same arms loop as in the v2 pilot.
4. hashes: F2 (c7a604152a04) and the V1b no-lever setting (83c831589532) are
   unchanged by the new option; turning P3 on changes the condition hash.
"""
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import random
import textworld

from agent.policy import Policy
from agent.task import goal_satisfied, static_catalog, trap_goal_spec
from env.transitions import solution_path_v2
from env.worlds import (LEVERS, NAME_POOL, SHARED_HEADS, TEST3_VOCAB, TRAP_NAMES,
                        TRAP_VOCAB, V2_DEV_EXTRA, build_trap_world_v2)
import mvp_c_headroom as runner
import runinfo


class ScriptedPolicy:
    """Stands in for the Qwen policy: fixed plans, a fixed P3 answer, and a
    record of when P3 was asked."""

    def __init__(self, plans, stuck_plans):
        self._plans, self._stuck = plans, stuck_plans
        self.stuck_calls = []
        self.step = None

    def set_context(self, world_id, root, step):
        self.step = step

    def plans(self, facts, goal, catalog, history, k, h):
        return [list(p) for p in self._plans], "ok"

    def plans_stuck(self, facts, goal, catalog, history, stuck, k, h):
        self.stuck_calls.append((self.step, tuple(stuck)))
        return [list(p) for p in self._stuck], "ok"


def main() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            print("FAIL " + msg)
        ok &= bool(cond)
        return cond

    n = 0
    # 1. vocabulary
    other = [w for ws in TRAP_VOCAB.values() for w in ws]
    other += [w for names in TRAP_NAMES for w in names]
    other += [w for names in NAME_POOL for w in names]
    other += [w for names in V2_DEV_EXTRA for w in names]
    other_words = {t for w in other for t in w.split()} - SHARED_HEADS
    for slot, words in TEST3_VOCAB.items():
        for w in words:
            shared = set(w.split()) & other_words
            check(not shared, f"test3 {slot} {w!r} shares {sorted(shared)}"); n += 1
        check(len(words) >= 8, f"test3 {slot}: only {len(words)} words"); n += 1
    t3 = {w for ws in TEST3_VOCAB.values() for w in ws}
    d = Path(tempfile.mkdtemp())
    for levers in ((), LEVERS):
        for i in range(24):
            game, path, meta = build_trap_world_v2(i, d, "test3", levers)
            tag = f"test3/{i}/{','.join(levers) or '-'}"
            check(meta["world_id"] == f"t3v{i:03d}", f"{tag}: id {meta['world_id']}"); n += 1
            names = [meta[k] for k in ("box", "key", "table", "door", "apple", "pear",
                                       "room_a", "room_b", "box2", "key2", "door2", "room_c")
                     if meta.get(k)]
            check(all(x in t3 for x in names), f"{tag}: name outside test3 {names}"); n += 1
            toks = [t for x in names for t in x.split() if t not in SHARED_HEADS]
            check(len(toks) == len(set(toks)), f"{tag}: shared word {names}"); n += 1
            env = textworld.start(str(path), request_infos=runner.INFOS); env.reset()
            goal = trap_goal_spec(game, meta)
            for a in solution_path_v2(meta, random.Random(i)):
                check(a in env.state["admissible_commands"], f"{tag}: {a} not admissible"); n += 1
                env.step(a)
            check(goal_satisfied(list(env.state["_facts"]), goal), f"{tag}: goal missed"); n += 1
            env.close()

    # 2. P3 in the prompt
    facts, goal_txt, cat = ["at(P, room)"], "carry x", ["go east", "open box"]
    hist = [("go east", False)]
    base = Policy.render_prompt(facts, goal_txt, cat, hist, 8, 2, hint=2)
    check(Policy.render_prompt(facts, goal_txt, cat, hist, 8, 2, hint=2, stuck=None) == base,
          "stuck=None changes the prompt"); n += 1
    st = Policy.render_prompt(facts, goal_txt, cat, hist, 8, 2, hint=2, stuck=["go east"])
    check(st.count("STUCK:") == 1 and '"go east"' in st.split("STUCK:")[1].split("\n\n")[0],
          "stuck note missing or without the command"); n += 1
    check(st.index("STUCK:") < st.index("Return 8 distinct"), "stuck note after the instructions"); n += 1
    check(st.replace(st[st.index("STUCK:"):st.index("Return 8 distinct")], "") == base,
          "stuck note changes anything else"); n += 1

    # 3. P3 in the runner
    game, path, meta = build_trap_world_v2(0, d, "dev", ())
    goal = trap_goal_spec(game, meta)
    catalog = static_catalog(game)
    env = textworld.start(str(path), request_infos=runner.INFOS); env.reset()
    runner.CAP[0], runner.HORIZON[0] = 6, 2
    stuck_first = [["go east"], [f"unlock {meta['door']} with {meta['key']}"]]
    open_box = f"open {meta['box']}"

    def run(arm, p3, plans=stuck_first):
        runner.STUCK_RETRY[0] = p3
        pol = ScriptedPolicy(plans, [[open_box]])
        log = []
        res = runner.run_episode(arm, env, game, meta, goal, pol, catalog, log)
        return [r["action"] for r in log], pol.stuck_calls, res

    for arm in ("C_fm", "oracle", "validity"):
        acts, calls, res = run(arm, False)
        check(open_box not in acts, f"{arm} P3 off: tried {open_box} without being told"); n += 1
        check(calls == [] and "stuck_retries" not in res, f"{arm} P3 off: P3 asked or logged"); n += 1
        acts, calls, res = run(arm, True)
        check(calls and calls[0][0] == 2, f"{arm} P3 on: first ask at step {calls[:1]}, want 2"); n += 1
        check(set(calls[0][1]) == {p[0] for p in stuck_first},
              f"{arm} P3 on: named {calls[0][1] if calls else None}"); n += 1
        check(acts[:3].count(open_box) == 1 and acts[2] == open_box,
              f"{arm} P3 on: actions {acts[:3]}"); n += 1
        check(res["stuck_retries"] == len(calls), f"{arm}: stuck_retries {res['stuck_retries']}"); n += 1
        # not stuck: never asked
        acts, calls, res = run(arm, True, plans=[[open_box], ["go east"]])
        check(calls == [] or all(c[0] >= 1 for c in calls), f"{arm}: asked while not stuck"); n += 1
        check(acts[0] == open_box, f"{arm}: first action {acts[0]}"); n += 1
    acts, calls, res = run("C", True)
    check(calls == [] and "go east" == acts[0] and acts.count("go east") == len(acts),
          f"C with P3 on: {acts} {calls}"); n += 1
    runner.STUCK_RETRY[0] = False
    env.close()

    # 5. K5 eval finds a world's position from its set's manifest, not from the
    #    digits of its id (t3v023 once read as 3023; 9/28)
    import kiis_k5_rollout as k5
    for ws, last in (("v3:test", "t3v023"), ("v3:x1", "t3v023"), ("v2:test", "tev023"), ("test", "te023")):
        order = list(k5.world_set_info(ws)[2])
        check(len(order) == 24 and order.index(last) == 23, f"{ws}: {last} at {order.index(last)} of {len(order)}"); n += 1

    # 4. hashes
    def chash(levers, hint, samples, retry=False):
        runinfo.configure_v2(levers, hint, samples, retry)
        cond = runinfo.condition(model="Qwen/Qwen3-4B", max_new_tokens=320, k=8, h=2,
                                 cap=30, trap=True)
        return runinfo.digest(cond)
    check(chash(("T3",), 2, 2) == "c7a604152a04", "F2 hash changed"); n += 1
    check(chash((), 2, 2) == "83c831589532", "V1b P1'+P2 hash changed"); n += 1
    h3 = chash((), 2, 2, True)
    check(h3 not in ("83c831589532", "c7a604152a04"), "P3 does not change the hash"); n += 1
    print(f"P3 on, no levers: condition {h3}")

    print(("PASS" if ok else "FAIL") + f" ({n} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
