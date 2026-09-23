"""One-step transitions labelled by the engine, for training and scoring the
LLM world models (kiis2026f/실험계획.md K2).

A transition is (state, action) -> (does it execute, value of every schema
variable afterwards). States come from two sources so that both early and late
stages of the task are covered: plain random walks, which mostly stay near the
start, and branches off a solution path, which reach the open door and the far
room. Actions are drawn from the static catalog — the same list the policy
chooses from — balanced between commands that run and commands that do not,
because a world model has to judge both.

Only engine ground truth goes in. No model output is read (MCA 2.3(b)).
"""
from __future__ import annotations

import random
from typing import Any

from agent.task import read_schema_values, static_catalog
from env.serialize import canonical_state

OBSERVE = ("look", "inventory", "examine")


def solution_path(meta: dict, rng: random.Random) -> list[str]:
    """A valid way to finish the trap-world task. The goal object can be picked
    up at any point before leaving, so its position is drawn."""
    steps = [f"open {meta['box']}", f"take {meta['key']} from {meta['box']}",
             f"unlock {meta['door']} with {meta['key']}", f"open {meta['door']}"]
    steps.insert(rng.randint(0, len(steps)), f"take {meta['apple']} from {meta['table']}")
    return steps + ["go east"]


def _walk(env, n: int, rng: random.Random) -> None:
    for _ in range(n):
        adm = [c for c in env.state["admissible_commands"] if not c.startswith(OBSERVE)]
        if not adm:
            return
        env.step(rng.choice(adm))


def sample_states(env, meta: dict, rng: random.Random, n_walk: int, n_branch: int) -> list:
    """Engine copies at sampled states. `env` is a fresh, reset environment."""
    out = []
    for _ in range(n_walk):
        e = env.copy()
        _walk(e, rng.randint(0, 8), rng)
        out.append(("walk", e))
    for _ in range(n_branch):
        e = env.copy()
        path = solution_path(meta, rng)
        for a in path[:rng.randint(0, len(path))]:
            e.step(a)
        _walk(e, rng.randint(0, 3), rng)
        out.append(("branch", e))
    return out


def sample_actions(env, catalog: list[str], rng: random.Random,
                   n_change: int = 3, n_observe: int = 1, n_invalid: int = 4) -> list[str]:
    """Commands to ask about in this state: some that run and change things,
    one that runs and changes nothing, some that do not run."""
    adm = set(env.state["admissible_commands"])
    valid = [c for c in catalog if c in adm]
    change = [c for c in valid if not c.startswith(OBSERVE)]
    observe = [c for c in valid if c.startswith(OBSERVE)]
    invalid = [c for c in catalog if c not in adm]
    pick = (rng.sample(change, min(n_change, len(change)))
            + rng.sample(observe, min(n_observe, len(observe)))
            + rng.sample(invalid, min(n_invalid, len(invalid))))
    rng.shuffle(pick)
    return pick


def label(env, game: Any, schema: list[dict], action: str) -> dict:
    """Engine truth for one (state, action)."""
    canon = canonical_state(list(env.state["_facts"]), game)
    runs = action in env.state["admissible_commands"]
    b = env.copy()
    b.step(action)
    nxt = canonical_state(list(b.state["_facts"]), game)
    b.close()
    now = read_schema_values(canon, schema)
    after = read_schema_values(nxt, schema)
    return {"canon": canon, "action": action, "executes": runs,
            "values_now": now, "values_next": after,
            "changed": sorted(k for k in after if after[k] != now[k]),
            "next_facts": nxt["dynamic_facts"] + nxt["static_facts"]}


def world_transitions(env, game: Any, meta: dict, schema: list[dict],
                      rng: random.Random, n_walk: int = 8, n_branch: int = 8) -> list[dict]:
    """Every sampled (state, action) of one world, de-duplicated."""
    catalog = static_catalog(game)
    seen, out = set(), []
    for kind, e in sample_states(env, meta, rng, n_walk, n_branch):
        for a in sample_actions(e, catalog, rng):
            rec = label(e, game, schema, a)
            key = (tuple(map(tuple, rec["canon"]["dynamic_facts"])), a)
            if key in seen:
                continue
            seen.add(key)
            rec.update({"world_id": meta["world_id"], "kind": kind})
            out.append(rec)
        e.close()
    return out
