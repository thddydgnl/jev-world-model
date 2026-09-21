"""World family + query compiler + deterministic endpoint labeler.

The world template is built so that ONE action pair (`open box` /
`take key from box`) is order-sensitive in some roots and order-INsensitive in
others. That gives us discrimination pairs and null-control pairs from the same
surface text, which is what separates "tracks state-dependent consequences"
from "pattern-matches the action string".
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from textworld import GameMaker

NAME_POOL = [
    ("wooden box", "brass key", "oak table", "red book", "steel door", "workroom", "cellar"),
    ("iron chest", "silver key", "pine shelf", "blue mug", "birch door", "study", "pantry"),
    ("tin crate", "copper key", "stone bench", "green vase", "elm door", "atelier", "vault"),
    ("glass case", "bronze key", "marble stand", "white lamp", "cedar door", "gallery", "annex"),
]

MODES = ("open", "closed_unlocked", "locked")


def build_world(idx: int, out_dir: Path, rng: random.Random) -> tuple[Any, Path, dict]:
    box_n, key_n, table_n, distract_n, door_n, room_a_n, room_b_n = NAME_POOL[idx % len(NAME_POOL)]
    suffix = f" {idx // len(NAME_POOL) + 1}" if idx >= len(NAME_POOL) else ""
    box_n, key_n, table_n, distract_n = (n + suffix for n in (box_n, key_n, table_n, distract_n))

    m = GameMaker()
    room_a = m.new_room(room_a_n + suffix)
    room_b = m.new_room(room_b_n + suffix)
    path = m.connect(room_a.east, room_b.west)
    door = m.new_door(path, name=door_n + suffix)
    # Vary the door so `go east` succeeds in some worlds and fails in others;
    # otherwise player_room never changes and that query carries no signal.
    door_mode = ("locked", "closed", "open")[idx % 3]
    door.add_property(door_mode)

    box = m.new(type="c", name=box_n)
    box.add_property("closed")
    table = m.new(type="s", name=table_n)
    key = m.new(type="k", name=key_n)
    distract = m.new(type="o", name=distract_n)

    m.add_fact("match", key, door)
    room_a.add(box, table)
    box.add(key)
    # The distractor starts on the supporter in half the worlds, on the floor otherwise.
    (table if idx % 2 == 0 else room_a).add(distract)
    m.set_player(room_a)
    m.quests = []

    game = m.build()
    p = out_dir / f"world_{idx:03d}.json"
    game.save(str(p))

    meta = {
        "world_id": f"w{idx:03d}",
        "box": box_n, "key": key_n, "table": table_n,
        "distractor": distract_n, "door": door_n + suffix, "door_mode": door_mode,
        "room_a": room_a_n + suffix, "room_b": room_b_n + suffix,
    }
    return game, p, meta


# ---------------------------------------------------------------- queries

def _index(facts: list[Any]) -> dict[str, list[list[str]]]:
    idx: dict[str, list[list[str]]] = {}
    for f in facts:
        idx.setdefault(f.name, []).append([a.name for a in f.arguments])
    return idx


def parent_of(facts: list[Any], obj_id: str) -> str:
    """Direct parent of a portable object, as a canonical option key."""
    idx = _index(facts)
    for a, b in idx.get("in", []):
        if a == obj_id:
            return "inventory" if b == "I" else f"in:{b}"
    for a, b in idx.get("on", []):
        if a == obj_id:
            return f"on:{b}"
    for a, b in idx.get("at", []):
        if a == obj_id:
            return f"at:{b}"
    return "unknown"


def mode_of(facts: list[Any], ent_id: str) -> str:
    idx = _index(facts)
    for name, key in (("locked", "locked"), ("closed", "closed_unlocked"), ("open", "open")):
        for args in idx.get(name, []):
            if args and args[0] == ent_id:
                return key
    return "unknown"


def player_room(facts: list[Any]) -> str:
    for f in facts:
        if f.name == "at" and f.arguments[0].name == "P":
            return f.arguments[1].name
    return "unknown"


def build_query_catalog(game: Any, meta: dict) -> list[dict]:
    """One entry per question. Options enumerate EVERY type-possible value so
    the ground truth is always inside the support (설계.md §24 target support)."""
    ids = {info.name: vid for vid, info in game.infos.items() if info.name}
    rooms = [vid for vid, info in game.infos.items() if info.type == "r"]
    containers = [vid for vid, info in game.infos.items() if info.type == "c"]
    supporters = [vid for vid, info in game.infos.items() if info.type == "s"]
    name_of = {vid: info.name for vid, info in game.infos.items() if info.name}

    def parent_options(label: str) -> dict[str, str]:
        opts = {"inventory": f"The {label} is carried by the player."}
        for r in rooms:
            opts[f"at:{r}"] = f"The {label} is on the floor of the {name_of[r]}."
        for c in containers:
            opts[f"in:{c}"] = f"The {label} is inside the {name_of[c]}."
        for s in supporters:
            opts[f"on:{s}"] = f"The {label} is on top of the {name_of[s]}."
        return opts

    mode_options = lambda label: {
        "open": f"The {label} is open.",
        "closed_unlocked": f"The {label} is closed but not locked.",
        "locked": f"The {label} is locked.",
    }

    key_id, box_id = ids[meta["key"]], ids[meta["box"]]
    dis_id, door_id = ids[meta["distractor"]], ids[meta["door"]]

    return [
        {"id": "q_key_parent", "kind": "parent", "target": key_id,
         "label": meta["key"], "options": parent_options(meta["key"])},
        {"id": "q_box_mode", "kind": "mode", "target": box_id,
         "label": meta["box"], "options": mode_options(meta["box"])},
        {"id": "q_door_mode", "kind": "mode", "target": door_id,
         "label": meta["door"], "options": mode_options(meta["door"])},
        {"id": "q_distractor_parent", "kind": "parent", "target": dis_id,
         "label": meta["distractor"], "options": parent_options(meta["distractor"])},
        {"id": "q_player_room", "kind": "room", "target": "P", "label": "player",
         "options": {r: f"The player is in the {name_of[r]}." for r in rooms}},
    ]


def label(facts: list[Any], query: dict) -> str:
    if query["kind"] == "parent":
        return parent_of(facts, query["target"])
    if query["kind"] == "mode":
        return mode_of(facts, query["target"])
    return player_room(facts)


# ---------------------------------------------------------------- trap worlds

TRAP_NAMES = [
    ("wooden box", "brass key", "oak table", "steel door", "red apple", "ripe pear", "workroom", "cellar"),
    ("iron chest", "silver key", "pine shelf", "birch door", "green apple", "soft plum", "study", "pantry"),
    ("tin crate", "copper key", "stone bench", "elm door", "gold apple", "dry fig", "atelier", "vault"),
    ("glass case", "bronze key", "marble stand", "cedar door", "pink apple", "small date", "gallery", "annex"),
]


def build_trap_world(idx: int, out_dir: Path, rng: random.Random) -> tuple[Any, Path, dict]:
    """Same skeleton as build_world, plus an IRREVERSIBLE trap.

    `eat {food}` is admissible whenever the food is carried and permanently
    removes it from the world (`eaten(f)`). The goal requires carrying the
    apple, so eating it destroys the episode while remaining a perfectly
    executable command. A validity filter cannot see the difference; only a
    forecast of the endpoint can. The pear is a harmless decoy, so "never eat"
    and "do not eat the goal object" are distinguishable.
    """
    box_n, key_n, table_n, door_n, apple_n, pear_n, ra_n, rb_n = TRAP_NAMES[idx % len(TRAP_NAMES)]
    suffix = f" {idx // len(TRAP_NAMES) + 1}" if idx >= len(TRAP_NAMES) else ""
    # Suffix every name here, including the door, so no use site has to
    # remember to. Forgetting it for one name is how this broke twice.
    box_n, key_n, table_n, door_n, apple_n, pear_n, ra_n, rb_n = (
        n + suffix for n in (box_n, key_n, table_n, door_n, apple_n, pear_n, ra_n, rb_n))

    m = GameMaker()
    room_a = m.new_room(ra_n)
    room_b = m.new_room(rb_n)
    door = m.new_door(m.connect(room_a.east, room_b.west), name=door_n)
    door.add_property("locked")

    box = m.new(type="c", name=box_n); box.add_property("closed")
    table = m.new(type="s", name=table_n)
    key = m.new(type="k", name=key_n)
    apple = m.new(type="f", name=apple_n)
    pear = m.new(type="f", name=pear_n)

    m.add_fact("match", key, door)
    room_a.add(box, table, pear)
    box.add(key)
    table.add(apple)
    m.set_player(room_a)
    m.quests = []

    game = m.build()
    p = out_dir / f"trap_{idx:03d}.json"
    game.save(str(p))
    meta = {"world_id": f"t{idx:03d}", "box": box_n, "key": key_n, "table": table_n,
            "door": door_n, "apple": apple_n, "pear": pear_n,
            "room_a": ra_n, "room_b": rb_n, "trap": True}
    return game, p, meta
