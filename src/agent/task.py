"""Goals, the static command catalog, and the common utility scorer.

The engine carries no quests (설계.md §3.1), so goals live here. Both arms
share this module exactly: the only thing that differs between arm C and the
oracle arm is WHICH candidate prefix gets selected, never the goal, the
candidate set, or the scoring function.
"""
from __future__ import annotations

from typing import Any

# 설계.md §11 initial coefficients. Not tuned; identical for every arm.
ALPHA, LAMBDA, COST, BETA = 0.25, 0.10, 0.01, 0.05


def goal_spec(game: Any, meta: dict) -> dict[str, Any]:
    """Reach the far room, expressed as a 3-atom conjunction along the
    dependency chain: open box -> take key -> unlock door -> open door -> go east.

    A single terminal atom would make `goal_progress` flat for every 2-step
    prefix, leaving the receding-horizon scorer with no gradient until the goal
    is already within reach. 설계.md §17 calls for 1-3 explicit predicates and
    their conjunction; three is what makes H=2 lookahead informative here.
    """
    ids = {info.name: vid for vid, info in game.infos.items() if info.name}
    key, door, room_b = ids[meta["key"]], ids[meta["door"]], ids[meta["room_b"]]
    return {
        "id": "reach_far_room",
        "text": (f"Be located in the {meta['room_b']}, "
                 f"carrying the {meta['key']}, with the {meta['door']} open."),
        "atoms": [("in", key, "I"), ("open", door), ("at", "P", room_b)],
    }


def _index(facts: list[Any]) -> set[tuple[str, ...]]:
    return {tuple([f.name] + [a.name for a in f.arguments]) for f in facts}


def goal_satisfied(facts: list[Any], goal: dict) -> bool:
    idx = _index(facts)
    return all(a in idx for a in goal["atoms"])


def goal_progress(facts: list[Any], goal: dict) -> float:
    idx = _index(facts)
    return sum(a in idx for a in goal["atoms"]) / len(goal["atoms"])


def static_catalog(game: Any) -> list[str]:
    """Type-correct commands, independent of the current state. This is NOT
    the set of commands that will succeed now (설계.md §6)."""
    return sorted(set(game.possible_admissible_commands))


def utility(conj: float, progress: float, n_invalid: float, h: int,
            prior: float) -> float:
    """설계.md §11. `conj` and `progress` are probabilities in [0,1]; for the
    oracle arm they are one-hot readings of the true endpoint."""
    return (conj
            + ALPHA * progress
            - LAMBDA * (n_invalid / max(h, 1))
            - COST * h
            + BETA * prior)


def unique_prefixes(plans: list[list[str]], max_h: int) -> list[tuple[str, ...]]:
    """All prefixes of length 1..max_h, de-duplicated, order preserved."""
    seen: dict[tuple[str, ...], None] = {}
    for plan in plans:
        for h in range(1, min(max_h, len(plan)) + 1):
            seen.setdefault(tuple(plan[:h]), None)
    return list(seen)


def plan_prior(plans: list[list[str]], prefix: tuple[str, ...]) -> float:
    """Best (lowest) rank among plans carrying this prefix, mapped to [0,1]."""
    best = None
    for rank, plan in enumerate(plans):
        if tuple(plan[:len(prefix)]) == prefix:
            best = rank if best is None else min(best, rank)
    if best is None or len(plans) <= 1:
        return 0.0
    return 1.0 - best / (len(plans) - 1)
