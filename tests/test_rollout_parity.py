"""A perfect step backend must make the recursive rollout reproduce the engine.

Every world-model arm shares RecursiveForecaster's rollout, beam and goal
scoring and differs only in its step backend. So plug in a backend that answers
from the engine: the rollout's top state must equal the state reached by really
executing the prefix, its expected invalid count must equal the real one, and
the goal read off it must match the engine's. Any difference is a wiring bug in
the shared rollout, which would contaminate every arm the same way.

Runs on dev worlds and on test-split worlds, so the new names are exercised too.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import EnvInfos
from agent.task import (state_schema, trap_goal_spec, static_catalog,
                        goal_satisfied, goal_progress, read_schema_values)
from env.serialize import canonical_state
from env.worlds import build_trap_world
from wm.recursive_forecaster import RecursiveForecaster, StepPrediction

INFOS = EnvInfos(facts=True, admissible_commands=True, possible_admissible_commands=True)


class EngineStep:
    """Step backend that answers from the engine instead of a model.

    Keeps a real environment for every state it has produced, keyed the same
    way the forecaster keys its cache, so a predicted state can be advanced by
    the engine exactly as a model would be asked about it.
    """

    def __init__(self, root_env, game, schema):
        self.game, self.schema = game, schema
        self.envs = {self._fp(canonical_state(list(root_env.state["_facts"]), game)): root_env}

    @staticmethod
    def _fp(canon):
        return RecursiveForecaster._fingerprint(canon)

    def __call__(self, canon, action):
        b = self.envs[self._fp(canon)].copy()
        runs = action in b.state["admissible_commands"]
        b.step(action)
        nxt = canonical_state(list(b.state["_facts"]), self.game)
        self.envs.setdefault(self._fp(nxt), b)
        values = read_schema_values(nxt, self.schema)
        return StepPrediction(values, {k: 1.0 for k in values}, 1.0 if runs else 0.0)


def main():
    rng = random.Random(41)
    d = Path("/tmp/rollout_parity"); d.mkdir(exist_ok=True)
    bad = checked = 0

    for split, idx in [("dev", 0), ("dev", 1), ("dev", 5), ("dev", 10),
                       ("test", 0), ("test", 7)]:
        game, path, meta = build_trap_world(idx, d, rng, split=split)
        schema, goal = state_schema(game, meta), trap_goal_spec(game, meta)
        catalog = static_catalog(game)
        env = textworld.start(str(path), request_infos=INFOS)
        env.reset()

        for trial in range(8):
            root = env.copy()
            for _ in range(rng.randint(0, 4)):
                adm = [c for c in root.state["admissible_commands"]
                       if not c.startswith(("look", "inventory", "examine"))]
                if adm:
                    root.step(rng.choice(adm))
            canon = canonical_state(list(root.state["_facts"]), game)
            prefix = tuple(rng.choice(catalog) for _ in range(rng.randint(1, 4)))

            # truth: execute the prefix
            e = root.copy(); n_inv = 0
            for a in prefix:
                n_inv += a not in e.state["admissible_commands"]
                e.step(a)
            end = list(e.state["_facts"])
            truth = canonical_state(end, game)

            fc = RecursiveForecaster(None, schema, goal,
                                     step=EngineStep(root, game, schema))
            terms, e_inv = fc.score(canon, "k", prefix, {})
            beam = fc._last_beam
            top = max(beam, key=lambda kv: kv[1])[0]

            ok = (top["dynamic_facts"] == truth["dynamic_facts"]
                  and abs(beam[0][1] - 1.0) < 1e-9 and len(beam) == 1
                  and abs(e_inv - n_inv) < 1e-9
                  and terms.conj == float(goal_satisfied(end, goal))
                  and abs(terms.progress - goal_progress(end, goal)) < 1e-9)
            checked += 1
            if not ok:
                bad += 1
                print(f"  MISMATCH {meta['world_id']} {prefix}: "
                      f"invalid {e_inv} vs {n_inv}, beam {len(beam)}")
            e.close(); root.close()
        env.close()

    print(f"\n엔진 정답 backend의 rollout = 실제 실행: {checked - bad}/{checked} "
          f"({'PASS' if bad == 0 else 'FAIL'})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
