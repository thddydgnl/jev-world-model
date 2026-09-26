"""v2 candidate levers in the policy, checked without loading a model.

P1: the hint appears only when asked, and the default prompt is the v1 prompt.
P2: sample 0 is exactly the v1 call (same seed, same plans first); extra samples
use their own seeds, add only new plans, and the situation seed is restored;
the fallback fires only when no sample produced a plan.
"""
import json, sys, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.policy import AFFORDANCE_HINT, HINT_CARRY_KEY, PROMPT, Policy

CATALOG = ["open box", "take key from box", "go east", "look", "unlock door with key"]


def fake(replies):
    """A Policy whose _generate answers from `replies`, keyed by the seed in use."""
    p = object.__new__(Policy)
    p.seed, p.calls, p.repairs, p.fallbacks = 20260921, 0, 0, 0
    p.hint, p.samples = False, 1
    p.seen = []

    def gen(prompt):
        p.seen.append(p._ctx_seed)
        p.calls += 1
        return replies.get(p._ctx_seed, "no json")
    p._generate = gen
    return p


def main() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    facts, goal = ["at(P, r_0)"], "be east"
    v1 = PROMPT.format(facts="- at(P, r_0)", goal=goal,
                       catalog="\n".join(f"- {c}" for c in CATALOG),
                       history="- (nothing attempted yet)", k=8, h=2)
    check(Policy.render_prompt(facts, goal, CATALOG) == v1, "default prompt is the v1 prompt")
    hinted = Policy.render_prompt(facts, goal, CATALOG, hint=True)
    check(AFFORDANCE_HINT in hinted and AFFORDANCE_HINT not in v1, "hint only when asked")
    check(hinted.replace(AFFORDANCE_HINT + "\n\n", "") == v1, "hint adds nothing else")
    carry = Policy.render_prompt(facts, goal, CATALOG, hint=2)
    check(HINT_CARRY_KEY not in hinted, "P1 has no carry-the-key rule")
    check(carry.replace(AFFORDANCE_HINT + "\n" + HINT_CARRY_KEY + "\n\n", "") == v1,
          "P1' is P1 plus the carry-the-key rule, nothing else")

    key = "tv000|0|3"
    s0 = 20260921 + (zlib.crc32(key.encode()) & 0x7FFFFFFF)
    s1 = 20260921 + (zlib.crc32(f"{key}#1".encode()) & 0x7FFFFFFF)
    r0 = json.dumps({"plans": [["open box"], ["go east", "look"], ["bogus"]]})
    r1 = json.dumps({"plans": [["go east", "look"], ["take key from box"]]})

    p = fake({s0: r0, s1: r1})
    p.set_context("tv000", 0, 3)
    single, st = p.plans(facts, goal, CATALOG)
    check(single == [["open box"], ["go east", "look"]] and st == "ok", "samples=1 as v1")

    p = fake({s0: r0, s1: r1})
    p.samples = 2
    p.set_context("tv000", 0, 3)
    merged, st = p.plans(facts, goal, CATALOG)
    check(p.seen == [s0, s1], "sample seeds: situation, then situation#1")
    check(merged[:2] == single, "sample 0's plans come first, unchanged")
    check(merged == single + [["take key from box"]], "only new plans are added")
    check(p._ctx_seed == s0, "situation seed restored")

    p = fake({s1: r1})               # sample 0 unusable (repaired, still nothing)
    p.samples = 2
    p.set_context("tv000", 0, 3)
    merged, st = p.plans(facts, goal, CATALOG)
    check(merged == [["go east", "look"], ["take key from box"]] and p.fallbacks == 0,
          "one usable sample is enough, no fallback")

    p = fake({})
    p.samples = 2
    p.set_context("tv000", 0, 3)
    merged, st = p.plans(facts, goal, CATALOG)
    check(st == "fallback" and p.fallbacks == 1 and merged == [[c] for c in CATALOG[:8]],
          "fallback only when no sample gave a plan")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
