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

from agent.task import goal_from_rows, project_state, read_schema_values
from forecast import ROLLOUT_CONVENTION, render_facts
from wm.jev_forecaster import INCLUDE_QUERY_CATALOG, validity_question

PRUNE_W = 0.02      # drop beam branches below this weight
BEAM_K = 4          # states carried forward per step


def step_questions(schema: list[dict], action: str) -> dict:
    """One action, one state, every mutable variable. h=1 throughout.

    Validity is NOT asked here. Measured on a case JEV should reject (take from
    a closed container), the same validity question averaged 0.285 asked alone
    and 0.576 bundled with these six — 10 samples each, difference +0.291
    against a standard error of 0.028. Questions in one request are evaluated
    independently, but they share the `state`, and a state carrying
    `action_sequence` alongside six questions that all presuppose the command
    ran frames the validity judgement toward "it ran".
    """
    frame = (f"Starting from `current_state`, attempt exactly this one command: "
             f"`{action}`. Apply `rollout_convention`. Report the resulting "
             f"state AFTER the command, not the current state.")
    return {q["id"]: {"type": "choice",
                      "instructions": f"{frame} Question: {q['ask']}",
                      "criteria": dict(q["options"])}
            for q in schema}


def validity_state(canon: dict) -> dict[str, Any]:
    """Payload for the validity question alone: the current state and nothing
    about any action being simulated."""
    return {
        "current_state": {
            "player_location": canon["player_room"],
            "entities": canon["entities"],
            "facts_canonical": canon["dynamic_facts"] + canon["static_facts"],
            "facts_readable": render_facts(canon),
        },
    }


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
class StepPrediction:
    """What one step backend says about (state, action)."""
    values: dict[str, str]      # predicted value of every schema variable
    top: dict[str, float]       # probability of that value; 1.0 when the backend gives none
    p_exec: float               # probability the command executes
    requests: int = 0           # model requests this prediction cost
    parse_failed: bool = False  # generative backends: no usable state came back


class TypedStep:
    """One step asked as typed questions — the JEV style.

    Validity goes in a request of its own, then every schema variable is asked
    as a Choice over all its type-possible values and read by argmax. Any client
    whose `ask(state, questions, tag)` returns the JEV response shape works here,
    so a local model can stand in for the API without touching the rollout. A
    client that also has `ask_many` gets a whole depth of the rollout at once.
    """

    def __init__(self, client, schema: list[dict]) -> None:
        self.client = client
        self.schema = schema

    def requests(self, canon: dict, action: str) -> list[tuple[dict, dict, str]]:
        """The two requests, as (state, questions, tag), in the order they are sent."""
        return [(build_state(canon, self.schema, action),
                 step_questions(self.schema, action), "R_step"),
                (validity_state(canon),
                 {"v_exec": validity_question(action, [])}, "R_valid")]

    def assemble(self, out: dict, vout: dict) -> StepPrediction:
        values, top = {}, {}
        for q in self.schema:
            probs = out["answers"][q["id"]]["probabilities"]
            best = max(probs, key=probs.get)
            values[q["id"]] = best
            top[q["id"]] = probs[best]
        p_exec = float(vout["answers"]["v_exec"]["probabilities"]["executes"])
        return StepPrediction(values, top, p_exec, requests=2)

    def __call__(self, canon: dict, action: str) -> StepPrediction:
        (s1, q1, t1), (s2, q2, t2) = self.requests(canon, action)
        out = self.client.ask(s1, q1, tag=t1)
        vout = self.client.ask(s2, q2, tag=t2)
        return self.assemble(out, vout)

    def predict_batch(self, pairs: list[tuple[dict, str]]) -> list[StepPrediction]:
        if not hasattr(self.client, "ask_many"):
            return [self(c, a) for c, a in pairs]
        outs = self.client.ask_many([r for c, a in pairs for r in self.requests(c, a)])
        return [self.assemble(outs[2 * i], outs[2 * i + 1]) for i in range(len(pairs))]


@dataclass
class Stats:
    requests: int = 0
    steps: int = 0
    branches: int = 0
    cache_hits: int = 0
    contradictions: int = 0
    parse_failures: int = 0
    # Drift diagnostics, keyed by rollout depth (1-based). Written only when the
    # runner supplies the true trajectory, read only by the analysis — never by
    # `score`, so the truth cannot reach a planning decision.
    depth_n: dict = field(default_factory=dict)
    depth_exact: dict = field(default_factory=dict)
    depth_vars_ok: dict = field(default_factory=dict)
    depth_vars_tot: dict = field(default_factory=dict)
    depth_in_beam: dict = field(default_factory=dict)
    valid_n: int = 0
    valid_ok: int = 0


class RecursiveForecaster:
    """One instance per episode. Caches on (state fingerprint, action).

    The rollout, beam and goal scoring live here once. How a single step is
    predicted is the `step` backend: typed questions by default (JEV or any
    client with the same response shape), or anything else that returns a
    StepPrediction. Swapping it is the only thing that differs between the
    world-model arms.
    """

    mode = "recursive"

    def __init__(self, client, schema: list[dict], goal: dict, step=None) -> None:
        self.step = step if step is not None else TypedStep(client, schema)
        self.schema = schema
        self.goal = goal
        self._cache: dict[tuple, tuple[dict, float]] = {}
        self.stats = Stats()

    def reset_step_cache(self) -> None:
        """Kept across planning steps on purpose: the key is (state fingerprint,
        action), so a repeat of the same situation has the same answer and does
        not need paying for twice."""
        return

    @staticmethod
    def _fingerprint(canon: dict) -> str:
        return "|".join(",".join(r) for r in canon["dynamic_facts"])

    def _store(self, key: tuple, canon: dict, pred: StepPrediction) -> tuple[dict, float]:
        self.stats.requests += pred.requests
        self.stats.parse_failures += pred.parse_failed
        nxt = project_state(canon, pred.values, self.schema)
        # A variable whose argmax carries little mass means the marginals did not
        # agree on one state. Counted, never patched.
        if any(v < 0.5 for v in pred.top.values()):
            self.stats.contradictions += 1
        self._cache[key] = (nxt, pred.p_exec)
        return nxt, pred.p_exec

    def _advance(self, canon: dict, action: str) -> tuple[dict, float]:
        """Predicted next state and the probability the command executes."""
        key = (self._fingerprint(canon), action)
        if key in self._cache:
            self.stats.cache_hits += 1
            return self._cache[key]
        return self._store(key, canon, self.step(canon, action))

    def _advance_beam(self, beam: list, action: str) -> tuple[list, float]:
        """One depth of the rollout: every beam state tries `action`; it runs
        with probability p or leaves the state as it was. Returns the new beam
        and the expected number of failures at this depth."""
        self.stats.steps += 1
        nxt: list[tuple[dict, float]] = []
        fail_mass = 0.0
        for state, w in beam:
            s2, p = self._advance(state, action)
            fail_mass += w * (1.0 - p)
            if w * p > PRUNE_W:
                nxt.append((s2, w * p))
            if w * (1.0 - p) > PRUNE_W:
                nxt.append((state, w * (1.0 - p)))
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
        return beam, fail_mass

    def _record_drift(self, depth: int, beam: list, truth: dict) -> None:
        """Diagnostic only. Compares the rollout with the real trajectory."""
        st = self.stats
        st.depth_n[depth] = st.depth_n.get(depth, 0) + 1
        top = max(beam, key=lambda kv: kv[1])[0]
        if top["dynamic_facts"] == truth["dynamic_facts"]:
            st.depth_exact[depth] = st.depth_exact.get(depth, 0) + 1
        pv = read_schema_values(top, self.schema)
        tv = read_schema_values(truth, self.schema)
        ok = sum(1 for k in tv if pv.get(k) == tv[k])
        st.depth_vars_ok[depth] = st.depth_vars_ok.get(depth, 0) + ok
        st.depth_vars_tot[depth] = st.depth_vars_tot.get(depth, 0) + len(tv)
        if any(s["dynamic_facts"] == truth["dynamic_facts"] for s, _ in beam):
            st.depth_in_beam[depth] = st.depth_in_beam.get(depth, 0) + 1

    def rollout(self, canon: dict, actions: tuple[str, ...],
                truth: list[dict] | None = None) -> tuple[list, float]:
        """Beam over execution branches. Returns [(state, weight)] and E[invalid].

        `truth` is the real state after each action, supplied by the runner for
        drift measurement. It is written to stats and never used for scoring.
        """
        beam: list[tuple[dict, float]] = [(canon, 1.0)]
        n_invalid = 0.0
        for depth, a in enumerate(actions, start=1):
            beam, fail_mass = self._advance_beam(beam, a)
            n_invalid += fail_mass
            if truth is not None and depth <= len(truth):
                self._record_drift(depth, beam, truth[depth - 1])
        return beam, n_invalid

    def _goal_terms(self, beam: list):
        from wm.jev_forecaster import GoalTerms
        conj = progress = 0.0
        for state, w in beam:
            sat, prog = goal_from_rows(state["dynamic_facts"], self.goal)
            conj += w * float(sat)
            progress += w * prog
        return GoalTerms(conj, progress)

    def score(self, canon: dict, state_key: str, prefix: tuple[str, ...],
              p_first: dict[str, float],
              truth: list[dict] | None = None) -> tuple[Any, float]:
        """Same signature as JevForecaster.score so the planner is unchanged."""
        beam, n_invalid = self.rollout(canon, prefix, truth)
        self._last_beam = beam
        return self._goal_terms(beam), n_invalid

    def score_many(self, canon: dict, prefixes: list[tuple[str, ...]],
                   truths: dict | None = None) -> list[tuple[Any, float]]:
        """`score` for every prefix at once, one rollout depth at a time.

        All (state, action) pairs a depth needs are predicted together, which
        is what makes a local model usable: one batched forward instead of one
        per beam state. Each prefix then advances through `_advance_beam`
        exactly as `rollout` would, so results match per-prefix `score`
        (tests/test_batched_rollout.py). Returned in the order of `prefixes`.
        """
        beams = {p: [(canon, 1.0)] for p in prefixes}
        n_inv = {p: 0.0 for p in prefixes}
        for depth in range(1, max(len(p) for p in prefixes) + 1):
            todo: dict[tuple, tuple[dict, str]] = {}
            for p in prefixes:
                if depth > len(p):
                    continue
                for state, _ in beams[p]:
                    key = (self._fingerprint(state), p[depth - 1])
                    if key not in self._cache and key not in todo:
                        todo[key] = (state, p[depth - 1])
            if todo:
                pairs = list(todo.values())
                if hasattr(self.step, "predict_batch"):
                    preds = self.step.predict_batch(pairs)
                else:
                    preds = [self.step(c, a) for c, a in pairs]
                for key, (state, _), pred in zip(todo, pairs, preds):
                    self._store(key, state, pred)
            for p in prefixes:
                if depth > len(p):
                    continue
                beams[p], fail_mass = self._advance_beam(beams[p], p[depth - 1])
                n_inv[p] += fail_mass
                truth = truths.get(p) if truths else None
                if truth is not None and depth <= len(truth):
                    self._record_drift(depth, beams[p], truth[depth - 1])
        return [(self._goal_terms(beams[p]), n_inv[p]) for p in prefixes]
