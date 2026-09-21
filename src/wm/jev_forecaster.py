"""Arm A: JEV as the forecaster behind the common planner.

Implements docs/arm_a_design.md. The planner, candidate set, goal and utility
are identical to the oracle arm; the only difference is that the goal terms and
the expected invalid count come from JEV instead of from the engine.

The chain never enters a JEV question. Each candidate prefix is split into
execution patterns, every pattern reduces to a subsequence in which every
command runs, and JEV is asked only:

  - per-command validity          (91.6% / 92.2% in MVP-B2)
  - endpoint of a clean sequence  (100% on 845 items in MVP-B)

It is never asked what happens when a command fails, which is the one thing it
measurably gets wrong (0/18 on order swaps).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forecast import ROLLOUT_CONVENTION, render_facts

PRUNE_P = 0.95          # drop the failure branch when a command is this certain


def validity_question(cmd: str, after: list[str]) -> dict:
    """Exact wording that measured 91.6% (V0) / 92.2% (V1) in MVP-B2."""
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


def endpoint_questions(goal_queries: list[dict], actions: list[str]) -> dict:
    """Exact framing that measured 100% on cleanly-executing sequences."""
    if actions:
        listing = " then ".join(f"`{a}`" for a in actions)
        frame = (f"Starting from `current_state`, attempt exactly these "
                 f"{len(actions)} command(s) in this order: {listing}. "
                 f"Apply `rollout_convention`. Report the resulting state AFTER "
                 f"the sequence, not the current state.")
    else:
        frame = ("Report the CURRENT state described in `current_state`. "
                 "No commands are attempted.")
    qs = {}
    for q in goal_queries:
        qs[q["id"]] = {"type": "choice",
                       "instructions": f"{frame} Question: {q['ask']}",
                       "criteria": dict(q["options"])}
    return qs


def build_state(canon: dict, goal_queries: list[dict], actions: list[str],
                horizon: int) -> dict[str, Any]:
    return {
        "current_state": {
            "player_location": canon["player_room"],
            "entities": canon["entities"],
            "facts_canonical": canon["dynamic_facts"] + canon["static_facts"],
            "facts_readable": render_facts(canon),
        },
        "action_sequence": actions,
        "horizon": horizon,
        "rollout_convention": ROLLOUT_CONVENTION,
        "query_catalog": [{"id": q["id"], "about": q["ask"], "options": q["options"]}
                          for q in goal_queries],
    }


@dataclass
class GoalTerms:
    conj: float
    progress: float


@dataclass
class Stats:
    requests: int = 0
    pruned: int = 0
    patterns: int = 0
    bound_violations: int = 0


class JevForecaster:
    """One instance per episode; caches within a planning step."""

    def __init__(self, client, goal_queries: list[dict], conj_query: dict) -> None:
        self.client = client
        self.goal_queries = goal_queries
        self.conj_query = conj_query
        self._endpoint_cache: dict[tuple, GoalTerms] = {}
        self.stats = Stats()

    # ------------------------------------------------------------- validity
    def validity(self, canon: dict, cmds: list[str], after: list[str]) -> dict[str, float]:
        """One request for many commands; they share the state and are
        evaluated independently (confirmed in the API docs)."""
        if not cmds:
            return {}
        st = build_state(canon, self.goal_queries, after, len(after))
        qs = {f"v{i}": validity_question(c, after) for i, c in enumerate(cmds)}
        out = self.client.ask(st, qs, tag="A_valid")
        self.stats.requests += 1
        return {c: float(out["answers"][f"v{i}"]["probabilities"]["executes"])
                for i, c in enumerate(cmds)}

    # ------------------------------------------------------------- endpoint
    def endpoint(self, canon: dict, state_key: str, actions: tuple[str, ...]) -> GoalTerms:
        key = (state_key, actions)
        if key in self._endpoint_cache:
            return self._endpoint_cache[key]

        st = build_state(canon, self.goal_queries, list(actions), len(actions))
        qs = endpoint_questions(self.goal_queries, list(actions))
        qs["q_goal_conj"] = {
            "type": "choice",
            "instructions": endpoint_questions(
                [{"id": "x", "ask": self.conj_query["ask"], "options": {}}],
                list(actions))["x"]["instructions"],
            "criteria": dict(self.conj_query["options"]),
        }
        out = self.client.ask(st, qs, tag="A_end")
        self.stats.requests += 1

        atoms = []
        for q in self.goal_queries:
            probs = out["answers"][q["id"]]["probabilities"]
            atoms.append(sum(probs.get(o, 0.0) for o in q["true_options"]))
        conj = float(out["answers"]["q_goal_conj"]["probabilities"]["yes"])

        # 설계.md §10.3: conjunction must lie within the Fréchet bounds of the
        # marginals. Record violations; never silently repair them.
        lo = max(0.0, sum(atoms) - (len(atoms) - 1))
        hi = min(atoms) if atoms else 1.0
        if not (lo - 1e-6 <= conj <= hi + 1e-6):
            self.stats.bound_violations += 1

        terms = GoalTerms(conj=conj, progress=sum(atoms) / len(atoms))
        self._endpoint_cache[key] = terms
        return terms

    # -------------------------------------------------------------- scoring
    def score(self, canon: dict, state_key: str, prefix: tuple[str, ...],
              p_first: dict[str, float]) -> tuple[GoalTerms, float]:
        """Mix goal terms over execution patterns; return also E[invalid]."""
        p1 = p_first[prefix[0]]
        probs = [p1]

        if len(prefix) == 1:
            branches = [(p1, prefix), (1.0 - p1, ())]
        else:
            # p2 is asked conditional on the earlier commands being attempted.
            p2 = self.validity(canon, [prefix[1]], [prefix[0]])[prefix[1]]
            probs.append(p2)
            if p1 >= PRUNE_P:
                self.stats.pruned += 1
                branches = [(p2, prefix), (1.0 - p2, (prefix[0],))]
            else:
                p2_alt = self.validity(canon, [prefix[1]], [])[prefix[1]]
                branches = [
                    (p1 * p2, prefix),
                    (p1 * (1.0 - p2), (prefix[0],)),
                    ((1.0 - p1) * p2_alt, (prefix[1],)),
                    ((1.0 - p1) * (1.0 - p2_alt), ()),
                ]

        conj = progress = 0.0
        for w, seq in branches:
            if w <= 1e-4:
                continue
            self.stats.patterns += 1
            t = self.endpoint(canon, state_key, seq)
            conj += w * t.conj
            progress += w * t.progress
        n_invalid = sum(1.0 - p for p in probs)
        return GoalTerms(conj, progress), n_invalid
