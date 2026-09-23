"""Batched scoring must decide exactly as per-prefix scoring does.

The LLM world-model arms score all candidate prefixes with
RecursiveForecaster.score_many, one rollout depth at a time, so that a local
model sees one batch per depth. The JEV arm scores prefix by prefix with
`score`. If the two ever disagreed on a beam, a weight or an expected invalid
count, the arms would not share one planner. A deterministic fake backend with
uncertain answers (so the beam really branches) makes the comparison exact.
"""
import json, random, sys, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import EnvInfos
from agent.task import state_schema, trap_goal_spec, static_catalog, unique_prefixes
from env.serialize import canonical_state
from env.worlds import build_trap_world
from wm.recursive_forecaster import RecursiveForecaster, TypedStep


class FakeClient:
    """JEV-shaped answers derived from a hash of the request: repeatable, and
    spread over the options so that execution branches really split."""
    def ask(self, state, questions, tag=""):
        blob = json.dumps(state, sort_keys=True)
        ans = {}
        for qid, q in questions.items():
            opts = list(q["criteria"])
            h = zlib.crc32((blob + qid + json.dumps(q, sort_keys=True)).encode())
            w = [((h >> (3 * i)) & 7) + 1 for i in range(len(opts))]
            ans[qid] = {"probabilities": {o: x / sum(w) for o, x in zip(opts, w)}}
        return {"answers": ans}


class BatchingClient(FakeClient):
    """Same answers, but through ask_many, as the local LLM client does."""
    def __init__(self):
        self.batches = []
    def ask_many(self, requests):
        self.batches.append(len(requests))
        return [self.ask(s, q, t) for s, q, t in requests]


def main():
    rng = random.Random(5)
    d = Path("/tmp/batched_rollout"); d.mkdir(exist_ok=True)
    bad = checked = 0
    for split, idx in [("dev", 2), ("dev", 7), ("test", 3)]:
        game, path, meta = build_trap_world(idx, d, rng, split=split)
        schema, goal = state_schema(game, meta), trap_goal_spec(game, meta)
        catalog = static_catalog(game)
        env = textworld.start(str(path), request_infos=EnvInfos(facts=True, admissible_commands=True))
        env.reset()
        for trial in range(6):
            canon = canonical_state(list(env.state["_facts"]), game)
            plans = [[rng.choice(catalog) for _ in range(rng.randint(1, 3))] for _ in range(8)]
            prefixes = unique_prefixes(plans, 3)

            one = RecursiveForecaster(FakeClient(), schema, goal)
            ref = [one.score(canon, "k", p, {}) for p in prefixes]

            client = BatchingClient()
            many = RecursiveForecaster(None, schema, goal, step=TypedStep(client, schema))
            got = many.score_many(canon, prefixes)

            for p, (t1, n1), (t2, n2) in zip(prefixes, ref, got):
                checked += 1
                if (abs(t1.conj - t2.conj) > 1e-12 or abs(t1.progress - t2.progress) > 1e-12
                        or abs(n1 - n2) > 1e-12):
                    bad += 1
                    print(f"  MISMATCH {meta['world_id']} {p}: {t1} {n1} vs {t2} {n2}")
            if one.stats.requests != many.stats.requests:
                print(f"  note: requests {one.stats.requests} vs {many.stats.requests}")
        env.close()
    print(f"\nscore_many = score: {checked - bad}/{checked} "
          f"({'PASS' if bad == 0 else 'FAIL'})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
