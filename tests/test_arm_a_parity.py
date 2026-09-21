"""Arm A must reduce to the oracle arm when the forecaster is perfect.

If JEV returned the true endpoint as a one-hot distribution and validity as 0/1,
arm A's utility must equal the oracle's and it must select the same prefix.
Any difference is a wiring bug in the mixture or the utility, not a model result.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import EnvInfos
from agent.task import (trap_goal_spec, goal_queries, goal_satisfied, goal_progress,
                        static_catalog, utility, unique_prefixes, plan_prior)
from env.serialize import canonical_state, state_hash
from env.worlds import build_trap_world
from wm.jev_forecaster import JevForecaster, GoalTerms

INFOS = EnvInfos(facts=True, admissible_commands=True, possible_admissible_commands=True)


class PerfectForecaster(JevForecaster):
    """Same code path; the two JEV calls are replaced by engine truth."""

    def __init__(self, env, game, goal, gqs, conj):
        super().__init__(client=None, goal_queries=gqs, conj_query=conj)
        self.env, self.game, self.goal = env, game, goal

    def _branch(self, actions):
        b = self.env.copy()
        for a in actions:
            b.step(a)
        facts = list(b.state["_facts"])
        b.close()
        return facts

    def validity(self, canon, cmds, after):
        b = self.env.copy()
        for a in after:
            b.step(a)
        adm = set(b.state["admissible_commands"])
        b.close()
        return {c: 1.0 if c in adm else 0.0 for c in cmds}

    def endpoint(self, canon, state_key, actions):
        facts = self._branch(actions)
        return GoalTerms(conj=float(goal_satisfied(facts, self.goal)),
                         progress=goal_progress(facts, self.goal))


def main():
    rng = random.Random(7)
    d = Path("/tmp/parity"); d.mkdir(exist_ok=True)
    mismatches = checked = 0

    for w in range(4):
        game, path, meta = build_trap_world(w, d, rng)
        goal = trap_goal_spec(game, meta)
        gqs, conj = goal_queries(game, meta, goal)
        catalog = static_catalog(game)
        env = textworld.start(str(path), request_infos=INFOS)
        env.reset()

        for trial in range(3):
            root = env.copy()
            for _ in range(rng.randint(0, 3)):
                adm = [c for c in root.state["admissible_commands"]
                       if not c.startswith(("look", "inventory", "examine"))]
                if adm:
                    root.step(rng.choice(adm))
            canon = canonical_state(list(root.state["_facts"]), game)
            plans = [[rng.choice(catalog), rng.choice(catalog)] for _ in range(8)]
            prefixes = unique_prefixes(plans, 2)

            # --- oracle scoring (the existing arm)
            oracle = {}
            for pre in prefixes:
                b = root.copy(); n_inv = 0
                for a in pre:
                    if a not in b.state["admissible_commands"]:
                        n_inv += 1
                    b.step(a)
                end = list(b.state["_facts"]); b.close()
                oracle[pre] = utility(float(goal_satisfied(end, goal)),
                                      goal_progress(end, goal), n_inv, len(pre),
                                      plan_prior(plans, pre))

            # --- arm A scoring with a perfect forecaster
            fc = PerfectForecaster(root, game, goal, gqs, conj)
            p_first = fc.validity(canon, sorted({p[0] for p in prefixes}), [])
            key = state_hash(canon)
            arm_a = {}
            for pre in prefixes:
                fc._endpoint_cache.clear()
                terms, n_inv = fc.score(canon, key, pre, p_first)
                arm_a[pre] = utility(terms.conj, terms.progress, n_inv, len(pre),
                                     plan_prior(plans, pre))

            best_o = max(oracle, key=lambda k: oracle[k])
            best_a = max(arm_a, key=lambda k: arm_a[k])
            checked += 1
            same_action = best_o[0] == best_a[0]
            max_gap = max(abs(oracle[p] - arm_a[p]) for p in prefixes)
            if not same_action or max_gap > 1e-6:
                mismatches += 1
                print(f"  MISMATCH {meta['world_id']} t{trial}: "
                      f"oracle={best_o} A={best_a} maxΔu={max_gap:.4f}")
            root.close()
        env.close()

    print(f"\n{checked - mismatches}/{checked} 일치  "
          f"({'PASS' if mismatches == 0 else 'FAIL'})")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
