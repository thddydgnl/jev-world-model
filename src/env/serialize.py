"""Canonical state serialization (설계.md §5) and the online information
whitelist (§3.5).

The raw GameState carries `game`, `_game_progression`, `_valid_actions`,
`_winning_policy` and the current `admissible_commands` regardless of
request_infos. None of those may reach a model, so we never pass the
GameState through: we build a fresh plain dict from an explicit whitelist.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = "tw_fo_cf_v1"

# Anything in the raw GameState that must never reach an online model.
FORBIDDEN_KEYS = frozenset({
    "game", "_game_progression", "_facts", "_valid_actions", "_valid_commands",
    "admissible_commands", "_winning_policy", "policy_commands", "_last_action",
    "objective", "max_score", "won", "lost", "extra.walkthrough",
})

# Predicates that describe the mutable physical configuration.
DYNAMIC_PREDICATES = frozenset({
    "at", "in", "on", "open", "closed", "locked", "match", "free", "link",
})
# Predicates that never change within an episode.
STATIC_PREDICATES = frozenset({"match", "link", "north_of", "south_of",
                               "east_of", "west_of", "free"})


def entity_table(game: Any) -> dict[str, dict[str, str]]:
    """var id -> {id, name, type}. Display names are kept separate from ids."""
    table: dict[str, dict[str, str]] = {}
    builtin = {"P": "player", "I": "inventory"}
    for var_id, info in game.infos.items():
        name = info.name or builtin.get(var_id) or var_id
        table[var_id] = {"id": var_id, "name": name, "type": info.type}
    for var_id, name in builtin.items():
        table.setdefault(var_id, {"id": var_id, "name": name, "type": var_id})
    return table


def _prop_to_list(prop: Any) -> list[str]:
    return [prop.name] + [a.name for a in prop.arguments]


def canonical_state(facts: list[Any], game: Any, attempt_index: int = 0) -> dict[str, Any]:
    """Build the online-visible state. Input is the raw fact list; output is
    plain JSON containing only whitelisted information."""
    table = entity_table(game)
    dynamic: list[list[str]] = []
    static: list[list[str]] = []
    player_room = None

    for prop in facts:
        row = _prop_to_list(prop)
        if prop.name == "at" and prop.arguments[0].name == "P":
            player_room = prop.arguments[1].name
        if prop.name in STATIC_PREDICATES:
            static.append(row)
        elif prop.name in DYNAMIC_PREDICATES:
            dynamic.append(row)
        else:
            dynamic.append(row)

    referenced = {a for row in dynamic + static for a in row[1:]}
    entities = [table[v] for v in sorted(referenced) if v in table]

    return {
        "schema_version": SCHEMA_VERSION,
        "player_room": player_room,
        "entities": entities,
        "dynamic_facts": sorted(dynamic),
        "static_facts": sorted(static),
        "attempt_index": attempt_index,
    }


def assert_clean(state: dict[str, Any]) -> None:
    """Fail loudly if a forbidden key ever appears in an online payload."""
    leaked = FORBIDDEN_KEYS & set(state)
    if leaked:
        raise ValueError(f"Forbidden keys in online state: {sorted(leaked)}")
    blob = json.dumps(state, ensure_ascii=False)
    for marker in ("_winning_policy", "policy_commands", "GameProgression", "_valid_actions"):
        if marker in blob:
            raise ValueError(f"Forbidden marker {marker!r} found in serialized state")


def state_hash(state: dict[str, Any]) -> str:
    """Hash of the physical configuration only; attempt_index is excluded so
    the same physical state at different times hashes identically."""
    core = {k: v for k, v in state.items() if k != "attempt_index"}
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def facts_hash(facts: list[Any]) -> str:
    """Hash straight off the engine's fact set, independent of our serializer."""
    rows = sorted("|".join(_prop_to_list(p)) for p in facts)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]
