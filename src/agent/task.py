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


def trap_goal_spec(game: Any, meta: dict) -> dict[str, Any]:
    """Carry the apple through the door. `eat {apple}` stays executable and
    permanently unsatisfiable-makes the first atom."""
    ids = {info.name: vid for vid, info in game.infos.items() if info.name}
    apple, door, room_b = ids[meta["apple"]], ids[meta["door"]], ids[meta["room_b"]]
    return {
        "id": "carry_apple_through",
        "text": (f"Be located in the {meta['room_b']}, carrying the "
                 f"{meta['apple']}, with the {meta['door']} open."),
        "atoms": [("in", apple, "I"), ("open", door), ("at", "P", room_b)],
        "trap_commands": [f"eat {meta['apple']}"],
        "decoy_commands": [f"eat {meta['pear']}"],
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
    """설계.md §11, with one change: the invalid term is a COUNT, not a rate.

    §11 wrote it as -lambda * N_invalid / h. Dividing by the prefix length turns
    it into a failure rate, which inverts the ranking the planner needs: a
    4-step prefix containing 2 unexecutable commands scores -lambda*0.5 while a
    1-step prefix containing 1 scores -lambda*1.0, so the prefix with more
    failures looks better. Since only the first action is ever executed, what
    matters is how much of the lookahead is expected to fail at all.

    At H=4 the rate form let long prefixes win on the discount alone: arm R
    dropped to 39.6% success with 22.7 steps per episode while its state
    predictions stayed intact (beam containment 99.2% at depth 4).

    `conj` and `progress` are probabilities in [0,1]; the oracle arm passes
    one-hot readings of the true endpoint.
    """
    return (conj
            + ALPHA * progress
            - LAMBDA * n_invalid
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


# ------------------------------------------------- goal queries for arm A

def goal_queries(game: Any, meta: dict, goal: dict) -> tuple[list[dict], dict]:
    """Turn the goal atoms into typed questions with COMPLETE answer support.

    Each atom becomes one Choice over every type-possible value, and
    `true_options` marks which values satisfy it. Asking in this form reuses the
    exact question family MVP-A verified at 0 support violations and MVP-B
    scored 100% on for cleanly-executing sequences.
    """
    name_of = {vid: info.name for vid, info in game.infos.items() if info.name}
    rooms = [vid for vid, info in game.infos.items() if info.type == "r"]
    containers = [vid for vid, info in game.infos.items() if info.type == "c"]
    supporters = [vid for vid, info in game.infos.items() if info.type == "s"]

    def parent_options(label: str) -> dict[str, str]:
        opts = {"inventory": f"The {label} is carried by the player."}
        for r in rooms:
            opts[f"at:{r}"] = f"The {label} is on the floor of the {name_of[r]}."
        for c in containers:
            opts[f"in:{c}"] = f"The {label} is inside the {name_of[c]}."
        for s in supporters:
            opts[f"on:{s}"] = f"The {label} is on top of the {name_of[s]}."
        return opts

    qs: list[dict] = []
    for i, atom in enumerate(goal["atoms"]):
        pred = atom[0]
        if pred == "in" and atom[2] == "I":
            label = name_of[atom[1]]
            qs.append({"id": f"g{i}", "kind": "parent",
                       "ask": (f"Where is the {label} located? Give its DIRECT "
                               f"container, supporter, room floor, or the "
                               f"player's inventory."),
                       "options": parent_options(label),
                       "true_options": ["inventory"]})
        elif pred in ("open", "closed", "locked"):
            label = name_of[atom[1]]
            qs.append({"id": f"g{i}", "kind": "mode",
                       "ask": f"Is the {label} open, closed but unlocked, or locked?",
                       "options": {"open": f"The {label} is open.",
                                   "closed_unlocked": f"The {label} is closed but not locked.",
                                   "locked": f"The {label} is locked."},
                       "true_options": [pred if pred != "closed" else "closed_unlocked"]})
        elif pred == "at" and atom[1] == "P":
            qs.append({"id": f"g{i}", "kind": "room",
                       "ask": "Which room is the player in?",
                       "options": {r: f"The player is in the {name_of[r]}." for r in rooms},
                       "true_options": [atom[2]]})
        else:
            raise ValueError(f"No question form for goal atom {atom}")

    conj = {"ask": (f"Is ALL of the following true at once? {goal['text']}"),
            "options": {"yes": "Every listed condition holds.",
                        "no": "At least one listed condition does not hold."}}
    return qs, conj


# ------------------------------------------- full mutable state schema (A2)

def state_schema(game: Any, meta: dict) -> list[dict]:
    """Every mutable state variable, independent of any goal.

    This is what separates a world model from a goal-conditioned scorer:
    the questions describe the world, so the same predictions can be scored
    against a different goal without re-querying (아이디어.md §4, criterion 3).
    """
    name_of = {vid: info.name for vid, info in game.infos.items() if info.name}
    rooms = [v for v, i in game.infos.items() if i.type == "r"]
    containers = [v for v, i in game.infos.items() if i.type == "c"]
    supporters = [v for v, i in game.infos.items() if i.type == "s"]
    portable = [v for v, i in game.infos.items() if i.type in ("o", "k", "f")]
    edible = {v for v, i in game.infos.items() if i.type == "f"}
    openable = containers + [v for v, i in game.infos.items() if i.type == "d"]

    schema: list[dict] = [{
        "id": "v_player", "kind": "room", "target": "P",
        "ask": "Which room is the player in?",
        "options": {r: f"The player is in the {name_of[r]}." for r in rooms},
    }]
    for o in portable:
        label = name_of[o]
        opts = {"inventory": f"The {label} is carried by the player."}
        for r in rooms:
            opts[f"at:{r}"] = f"The {label} is on the floor of the {name_of[r]}."
        for c in containers:
            opts[f"in:{c}"] = f"The {label} is inside the {name_of[c]}."
        for s in supporters:
            opts[f"on:{s}"] = f"The {label} is on top of the {name_of[s]}."
        if o in edible:
            # Eating removes the object from the world entirely. Without this
            # option the ground truth falls outside the answer support, which
            # 설계.md §24 requires to be 100%.
            opts["gone"] = f"The {label} has been eaten and no longer exists."
        schema.append({"id": f"v_par_{o}", "kind": "parent", "target": o,
                       "ask": (f"Where is the {label} located? Give its DIRECT "
                               f"container, supporter, room floor, or the "
                               f"player's inventory."),
                       "options": opts})
    for x in openable:
        label = name_of[x]
        schema.append({"id": f"v_mode_{x}", "kind": "mode", "target": x,
                       "ask": f"Is the {label} open, closed but unlocked, or locked?",
                       "options": {"open": f"The {label} is open.",
                                   "closed_unlocked": f"The {label} is closed but not locked.",
                                   "locked": f"The {label} is locked."}})
    return schema


def schema_covers(schema: list[dict], row: list[str]) -> bool:
    """Does some schema variable determine this fact?"""
    parents = {q["target"] for q in schema if q["kind"] == "parent"}
    modes = {q["target"] for q in schema if q["kind"] == "mode"}
    if row[0] == "at" and row[1] == "P":
        return True
    if row[0] in ("in", "on", "at") and len(row) == 3 and row[1] in parents:
        return True
    if row[0] == "eaten" and row[1] in parents:
        return True
    if row[0] in ("open", "closed", "locked") and row[1] in modes:
        return True
    return False


def project_state(base: dict, values: dict[str, str], schema: list[dict]) -> dict:
    """Rebuild a canonical state from per-variable answers.

    Facts the schema determines are replaced by the answers; facts it does not
    cover are carried forward unchanged. That carry-forward is an assumption —
    "what we did not ask about did not change" — and it is only sound while the
    schema covers everything mutable, which tests/test_recursive_roundtrip.py
    checks against the engine.

    Beyond that, only the exclusivity already declared in the question design is
    applied: one value per variable. No transition rule is used here — inserting
    one would mean the researcher's simulator is producing the performance
    (아이디어.md §7, candidate A2).
    """
    rows: list[list[str]] = [list(r) for r in base["dynamic_facts"]
                             if not schema_covers(schema, list(r))]
    player_room = base["player_room"]
    for q in schema:
        v = values.get(q["id"])
        if v is None:
            continue
        if q["kind"] == "room":
            player_room = v
            rows.append(["at", "P", v])
        elif q["kind"] == "parent":
            o = q["target"]
            if v == "gone":
                rows.append(["eaten", o])
            elif v == "inventory":
                rows.append(["in", o, "I"])
            else:
                rel, _, host = v.partition(":")
                rows.append([rel, o, host])
        else:
            pred = {"open": "open", "closed_unlocked": "closed", "locked": "locked"}[v]
            rows.append([pred, q["target"]])
    return {**base, "player_room": player_room, "dynamic_facts": sorted(rows)}


def read_schema_values(canon: dict, schema: list[dict]) -> dict[str, str]:
    """Inverse of `project_state`: read each variable off a canonical state.
    Used to compare a predicted state with the real one variable by variable."""
    rows = {tuple(r) for r in canon["dynamic_facts"]}
    out: dict[str, str] = {}
    for q in schema:
        if q["kind"] == "room":
            v = next((r[2] for r in rows if r[0] == "at" and r[1] == "P"), None)
        elif q["kind"] == "parent":
            o = q["target"]
            if ("eaten", o) in rows:
                v = "gone"
            else:
                v = None
                for r in rows:
                    if len(r) == 3 and r[1] == o and r[0] in ("in", "on", "at"):
                        v = "inventory" if r[2] == "I" else f"{r[0]}:{r[2]}"
        else:
            v = None
            for pred, val in (("open", "open"), ("closed", "closed_unlocked"),
                              ("locked", "locked")):
                if (pred, q["target"]) in rows:
                    v = val
        out[q["id"]] = v
    return out


def goal_from_rows(rows: list[list[str]], goal: dict) -> tuple[bool, float]:
    """Score a goal against a reconstructed state — no extra model call."""
    have = {tuple(r) for r in rows}
    hits = [tuple(a) in have for a in goal["atoms"]]
    return all(hits), sum(hits) / len(hits)
