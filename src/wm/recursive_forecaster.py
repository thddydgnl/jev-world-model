"""Candidate A2: recursive one-step simulation (아이디어.md §7).

The direct-horizon forecaster asks about the endpoint of a whole prefix and
about the goal's own atoms. Two consequences: the questions get harder as h
grows, and the predictions are goal-shaped, so a new goal means new questions
and a whole new round of calls. By 아이디어.md §4 that is a goal-conditioned
scorer rather than a world model.

Here JEV is only ever asked about ONE action from ONE state — the regime where
it measured 100% on cleanly executing sequences — and the answers rebuild a
state in the same canonical form as a real one, which is then fed back in. The
goal is scored against the reconstructed state locally, at no extra call, and
any other goal could be scored against the same rollout.

아이디어.md flagged the danger in this candidate: repairing the reconstruction
with transition rules would mean the researcher's simulator produces the
performance. So `project_state` applies only the exclusivity already declared
in the question design — one value per variable — and nothing else. There is no
"a closed box blocks take" anywhere in this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.task import goal_from_rows, project_state
from forecast import ROLLOUT_CONVENTION, render_facts
from wm.jev_forecaster import INCLUDE_QUERY_CATALOG, validity_question

PRUNE_W = 0.02      # drop beam branches below this weight
BEAM_K = 4          # states carried forward per step


def step_questions(schema: list[dict], action: str) -> dict:
    """One action, one state, every mutable variable. h=1 throughout."""
    frame = (f"Starting from `current_state`, attempt exactly this one command: "
             f"`{action}`. Apply `rollout_convention`. Report the resulting "
             f"state AFTER the command, not the current state.")
    qs = {q["id"]: {"type": "choice",
                    "instructions": f"{frame} Question: {q['ask']}",
                    "criteria": dict(q["options"])}
          for q in schema}
    qs["v_exec"] = validity_question(action, [])
    return qs


def build_state(canon: dict, schema: list[dict], action: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "current_state": {
            "player_location": canon["player_room"],
            "entities": canon["entities"],
            "facts_canonical": canon["dynamic_facts"] + canon["static_facts"],
            "facts_readable": render_facts(canon),
        },
        "action_sequence": [action],
        "horizon": 1,
        "rollout_convention": ROLLOUT_CONVENTION,
    }
    if INCLUDE_QUERY_CATALOG:
        state["query_catalog"] = [{"id": q["id"], "about": q["ask"],
                                   "options": q["options"]} for q in schema]
    return state


@dataclass
class Stats:
    requests: int = 0
    steps: int = 0
    branches: int = 0
    cache_hits: int = 0
    contradictions: int = 0


class RecursiveForecaster:
    """One instance per episode. Caches on (state fingerprint, action)."""

    mode = "recursive"

    def __init__(self, client, schema: list[dict], goal: dict) -> None:
        self.client = client
        self.schema = schema
        self.goal = goal
        self._cache: dict[tuple, tuple[dict, float]] = {}
        self.stats = Stats()

    @staticmethod
    def _fingerprint(canon: dict) -> str:
        return "|".join(",".join(r) for r in canon["dynamic_facts"])

    def _advance(self, canon: dict, action: str) -> tuple[dict, float]:
        """Predicted next state and the probability the command executes."""
        key = (self._fingerprint(canon), action)
        if key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]

        out = self.client.ask(build_state(canon, self.schema, action),
                              step_questions(self.schema, action), tag="R_step")
        self.stats.requests += 1

        values, top = {}, {}
        for q in self.schema:
            probs = out["answers"][q["id"]]["probabilities"]
            best = max(probs, key=probs.get)
            values[q["id"]] = best
            top[q["id"]] = probs[best]
        p_exec = float(out["answers"]["v_exec"]["probabilities"]["executes"])

        nxt = project_state(canon, values, self.schema)
        # A variable whose argmax carries little mass means the marginals did not
        # agree on one state. Counted, never patched.
        if any(v < 0.5 for v in top.values()):
            self.stats.contradictions += 1

        self._cache[key] = (nxt, p_exec)
        return nxt, p_exec

    def rollout(self, canon: dict, actions: tuple[str, ...]) -> tuple[list, float]:
        """Beam over execution branches. Returns [(state, weight)] and E[invalid]."""
        beam: list[tuple[dict, float]] = [(canon, 1.0)]
        n_invalid = 0.0
        for a in actions:
            self.stats.steps += 1
            nxt: list[tuple[dict, float]] = []
            fail_mass = 0.0
            for state, w in beam:
                s2, p = self._advance(state, a)
                fail_mass += w * (1.0 - p)
                if w * p > PRUNE_W:
                    nxt.append((s2, w * p))
                if w * (1.0 - p) > PRUNE_W:
                    nxt.append((state, w * (1.0 - p)))
            n_invalid += fail_mass
            merged: dict[str, tuple[dict, float]] = {}
            for state, w in nxt:
                fp = self._fingerprint(state)
                if fp in merged:
                    merged[fp] = (state, merged[fp][1] + w)
                else:
                    merged[fp] = (state, w)
            beam = sorted(merged.values(), key=lambda kv: -kv[1])[:BEAM_K]
            total = sum(w for _, w in beam) or 1.0
            beam = [(s, w / total) for s, w in beam]
            self.stats.branches += len(beam)
        return beam, n_invalid

    def score(self, canon: dict, state_key: str, prefix: tuple[str, ...],
              p_first: dict[str, float]) -> tuple[Any, float]:
        """Same signature as JevForecaster.score so the planner is unchanged."""
        from wm.jev_forecaster import GoalTerms
        beam, n_invalid = self.rollout(canon, prefix)
        self._last_beam = beam
        conj = progress = 0.0
        for state, w in beam:
            sat, prog = goal_from_rows(state["dynamic_facts"], self.goal)
            conj += w * float(sat)
            progress += w * prog
        return GoalTerms(conj, progress), n_invalid
