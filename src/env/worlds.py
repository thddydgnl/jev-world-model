"""World family + query compiler + deterministic endpoint labeler.

The world template is built so that ONE action pair (`open box` /
`take key from box`) is order-sensitive in some roots and order-INsensitive in
others. That gives us discrimination pairs and null-control pairs from the same
surface text, which is what separates "tracks state-dependent consequences"
from "pattern-matches the action string".
"""
from __future__ import annotations

import hashlib
import json
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


# ---------------------------------------------------------------- KIIS splits

# The dev worlds t000–t011 are TRAP_NAMES with numeric suffixes: four name sets,
# all of them looked at while the pipeline was being fixed. Everything a result
# is reported on uses the vocabulary below instead, split per slot into
# train / val / test so that no test name is ever seen in training
# (kiis2026f/실험계획.md §3). No word here occurs in TRAP_NAMES or NAME_POOL, so
# the dev worlds share no vocabulary with any split either.
TRAP_VOCAB: dict[str, list[str]] = {
    "box": [  # container
        "leather trunk", "walnut cabinet", "wicker hamper", "plastic bin",
        "clay urn", "velvet coffer", "maple cupboard", "chrome locker",
        "rattan basket", "ceramic jar", "lacquered dresser", "enamel canister",
        "teak drawer", "black strongbox", "grey safe", "dented footlocker",
        "carved bureau", "tall armoire", "zinc tub", "painted hutch",
        "vinyl hatbox", "felt toybox", "rosewood casket", "cardboard carton"],
    "key": [
        "nickel key", "pewter key", "rusty key", "tiny key", "ornate key",
        "skeleton key", "notched key", "crooked key", "twisted key", "slender key",
        "tarnished key", "square key", "hollow key", "jagged key", "polished key",
        "antique key", "spare key", "master key", "flat key", "bent key",
        "stubby key", "chunky key", "gilded key", "platinum key"],
    "table": [  # supporter
        "writing desk", "kitchen counter", "granite pedestal", "window ledge",
        "mahogany sideboard", "tiled worktop", "concrete plinth", "metal rack",
        "folding trestle", "oval podium", "slate altar", "brick mantel",
        "artist easel", "bar stool", "serving cart", "low platform",
        "acrylic console", "bedside nightstand", "drafting board", "butcher block",
        "round ottoman", "garden trolley", "reading lectern", "flat dais"],
    "door": [
        "arched door", "heavy door", "narrow door", "sliding door", "rusted door",
        "frosted door", "studded door", "swinging door", "creaky door", "bolted door",
        "reinforced door", "latticed door", "crimson door", "ivory door", "olive door",
        "scarlet door", "amber door", "violet door", "indigo door", "charcoal door",
        "screen door", "barn door", "garden door", "saloon door"],
    "apple": [  # goal food
        "juicy mango", "fresh peach", "sour lemon", "round melon", "dark cherry",
        "sweet orange", "tart lime", "fuzzy kiwi", "plump grape", "crisp radish",
        "yellow banana", "wild berry", "purple beet", "vine tomato", "seedless guava",
        "spotted papaya", "bright apricot", "golden quince", "raw carrot",
        "crunchy cucumber", "sticky lychee", "creamy avocado", "tangy grapefruit",
        "baby turnip"],
    "pear": [  # decoy food
        "stale bread", "hard cheese", "salted nut", "toasted muffin", "sugar cookie",
        "plain cracker", "brown egg", "cold sausage", "rye loaf", "honey cake",
        "corn cob", "bean pod", "salty pretzel", "jelly donut", "rice ball",
        "meat pie", "fish stick", "raisin bun", "chocolate truffle", "tofu cube",
        "mint candy", "ham slice", "butter waffle", "garlic knot"],
    "room": [
        "kitchen", "bedroom", "library", "hallway", "parlor", "garage", "attic",
        "basement", "workshop", "laundry", "nursery", "office", "studio", "foyer",
        "lounge", "scullery", "cloakroom", "chapel", "conservatory", "armory",
        "bakery", "greenhouse", "storeroom", "boathouse", "washroom", "den", "loft",
        "porch", "sunroom", "playroom", "darkroom", "mudroom", "pavilion", "cabin",
        "refectory", "observatory"],
}
VOCAB_SEED = 20260923
SPLITS = ("dev", "train", "val", "test")
SPLIT_PREFIX = {"dev": "t", "train": "tr", "val": "va", "test": "te"}


def vocab_split(slot: str) -> dict[str, list[str]]:
    """One slot's words, shuffled once and cut 60 / 10 / 30."""
    words = list(TRAP_VOCAB[slot])
    random.Random(f"kiis-vocab-{VOCAB_SEED}-{slot}").shuffle(words)
    n_train = round(0.6 * len(words))
    n_val = max(1, round(0.1 * len(words)))
    return {"train": words[:n_train],
            "val": words[n_train:n_train + n_val],
            "test": words[n_train + n_val:]}


def trap_names(split: str, idx: int) -> tuple[str, ...]:
    """(box, key, table, door, apple, pear, room_a, room_b) for one world.

    Drawn from the split's share of each slot, redrawn until no word is shared
    between two names, so that no command can refer to two entities at once and
    no name gives away another.
    """
    if split == "dev":
        names = TRAP_NAMES[idx % len(TRAP_NAMES)]
        # Suffix every name here, including the door, so no use site has to
        # remember to. Forgetting it for one name is how this broke twice.
        suffix = f" {idx // len(TRAP_NAMES) + 1}" if idx >= len(TRAP_NAMES) else ""
        return tuple(n + suffix for n in names)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    pools = {slot: vocab_split(slot)[split] for slot in TRAP_VOCAB}
    rng = random.Random(f"kiis-names-{VOCAB_SEED}-{split}-{idx}")
    for _ in range(1000):
        rooms = rng.sample(pools["room"], 2)
        names = tuple(rng.choice(pools[s]) for s in
                      ("box", "key", "table", "door", "apple", "pear")) + tuple(rooms)
        tokens = [t for n in names for t in n.split()]
        if len(tokens) == len(set(tokens)):
            return names
    raise RuntimeError(f"no clash-free names for {split} {idx}")


def world_fingerprint(game: Any) -> str:
    """Hash of what the engine and the models actually use: the initial facts
    and every entity's id, name and type.

    Not a file hash. TextWorld writes random flavour text (`desc`, `room_type`)
    into the game file on every build, so two builds of the same world differ
    byte for byte. None of that text reaches a model or changes a transition.
    """
    facts = sorted(str(f) for f in game.world.facts)
    ents = sorted((vid, info.name or "", info.type or "") for vid, info in game.infos.items())
    blob = json.dumps([facts, ents], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_trap_world(idx: int, out_dir: Path, rng: random.Random,
                     split: str = "dev") -> tuple[Any, Path, dict]:
    """Same skeleton as build_world, plus an IRREVERSIBLE trap.

    `eat {food}` is admissible whenever the food is carried and permanently
    removes it from the world (`eaten(f)`). The goal requires carrying the
    apple, so eating it destroys the episode while remaining a perfectly
    executable command. A validity filter cannot see the difference; only a
    forecast of the endpoint can. The pear is a harmless decoy, so "never eat"
    and "do not eat the goal object" are distinguishable.

    `split` picks the names only; the structure is the same in every split.
    """
    box_n, key_n, table_n, door_n, apple_n, pear_n, ra_n, rb_n = trap_names(split, idx)

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
    stem = f"trap_{idx:03d}" if split == "dev" else f"trap_{split}_{idx:03d}"
    p = out_dir / f"{stem}.json"
    game.save(str(p))
    meta = {"world_id": f"{SPLIT_PREFIX[split]}{idx:03d}", "box": box_n, "key": key_n,
            "table": table_n, "door": door_n, "apple": apple_n, "pear": pear_n,
            "room_a": ra_n, "room_b": rb_n, "trap": True, "split": split}
    return game, p, meta


# ------------------------------------------------------------ v2 (KIIS V0)
#
# kiis2026f/실험계획.md §4 V. In v1 every world model reached the oracle's
# ceiling: the world model's job was almost only "will this command run", and
# the ceiling itself was set by the policy's candidates. v2 adds commands that
# run but do not help, switched on one lever at a time during calibration:
#   T1  a side room C behind a closed, unlocked second door; the decoy food moves there
#   T2  a decoy key on the supporter that matches nothing
#   T3  the goal food starts inside a second, closed container
# The v1 builder above is untouched, so v1 worlds and their fingerprints stay valid.

LEVERS = ("T1", "T2", "T3")
V2_PREFIX = {"dev": "tv", "train": "trv", "val": "vav", "test": "tev"}
# Extra dev names, one set per TRAP_NAMES entry: (box2, key2, door2, room_c).
# None of these words is in TRAP_VOCAB or in the other dev names, apart from the
# head nouns "key" and "door".
V2_DEV_EXTRA = [
    ("cork barrel", "dull key", "wicket gate", "porchway"),
    ("canvas sack", "long key", "plank door", "landing"),
    ("cloth bundle", "worn key", "mesh gate", "larder"),
    ("linen bag", "odd key", "reed curtain", "lobby"),
]
# Words two names in one v2 world may share: two keys, possibly two doors.
SHARED_HEADS = {"key", "door"}
SIDE_EXITS = ("north", "south", "west")   # room C's side of room A; B is always east
OPPOSITE = {"north": "south", "south": "north", "west": "east", "east": "west"}


def parse_levers(text: str | None) -> tuple[str, ...]:
    """'T1,T3' -> ('T1', 'T3'), in LEVERS order. Empty or None -> ()."""
    got = {t.strip() for t in (text or "").split(",") if t.strip()}
    bad = got - set(LEVERS)
    if bad:
        raise ValueError(f"unknown levers {sorted(bad)}")
    return tuple(t for t in LEVERS if t in got)


def trap_names_v2(split: str, idx: int) -> dict[str, str]:
    """All names a v2 world may need, whichever levers are on, so switching a
    lever never renames the rest of the world."""
    if split == "dev":
        base = trap_names("dev", idx)
        suffix = f" {idx // len(TRAP_NAMES) + 1}" if idx >= len(TRAP_NAMES) else ""
        extra = tuple(n + suffix for n in V2_DEV_EXTRA[idx % len(V2_DEV_EXTRA)])
    else:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        pools = {slot: vocab_split(slot)[split] for slot in TRAP_VOCAB}
        rng = random.Random(f"kiis-names-v2-{VOCAB_SEED}-{split}-{idx}")
        for _ in range(5000):
            rooms = rng.sample(pools["room"], 3)
            boxes = rng.sample(pools["box"], 2)
            keys = rng.sample(pools["key"], 2)
            doors = rng.sample(pools["door"], 2)
            base = (boxes[0], keys[0], rng.choice(pools["table"]), doors[0],
                    rng.choice(pools["apple"]), rng.choice(pools["pear"]), rooms[0], rooms[1])
            extra = (boxes[1], keys[1], doors[1], rooms[2])
            tokens = [t for n in base + extra for t in n.split() if t not in SHARED_HEADS]
            if len(tokens) == len(set(tokens)):
                break
        else:
            raise RuntimeError(f"no clash-free v2 names for {split} {idx}")
    keys = ("box", "key", "table", "door", "apple", "pear", "room_a", "room_b",
            "box2", "key2", "door2", "room_c")
    return dict(zip(keys, base + extra))


def build_trap_world_v2(idx: int, out_dir: Path, split: str = "dev",
                        levers: tuple[str, ...] = LEVERS) -> tuple[Any, Path, dict]:
    """The v1 trap world plus the chosen levers. Goal and trap are unchanged:
    carry the goal food into room B through the opened door; eating it is
    executable and irreversible."""
    n = trap_names_v2(split, idx)
    side = SIDE_EXITS[idx % len(SIDE_EXITS)]

    m = GameMaker()
    room_a = m.new_room(n["room_a"])
    room_b = m.new_room(n["room_b"])
    door = m.new_door(m.connect(room_a.east, room_b.west), name=n["door"])
    door.add_property("locked")

    box = m.new(type="c", name=n["box"]); box.add_property("closed")
    table = m.new(type="s", name=n["table"])
    key = m.new(type="k", name=n["key"])
    apple = m.new(type="f", name=n["apple"])
    pear = m.new(type="f", name=n["pear"])
    m.add_fact("match", key, door)
    room_a.add(box, table)
    box.add(key)

    if "T1" in levers:
        room_c = m.new_room(n["room_c"])
        door2 = m.new_door(m.connect(getattr(room_a, side), getattr(room_c, OPPOSITE[side])),
                           name=n["door2"])
        door2.add_property("closed")
        room_c.add(pear)
    else:
        room_a.add(pear)
    if "T2" in levers:
        table.add(m.new(type="k", name=n["key2"]))
    if "T3" in levers:
        box2 = m.new(type="c", name=n["box2"]); box2.add_property("closed")
        room_a.add(box2)
        box2.add(apple)
    else:
        table.add(apple)
    m.set_player(room_a)
    m.quests = []

    game = m.build()
    tag = "".join(t[1] for t in levers) or "0"
    p = out_dir / f"trapv2_{split}_{idx:03d}_L{tag}.json"
    game.save(str(p))
    meta = {"world_id": f"{V2_PREFIX[split]}{idx:03d}", **{k: n[k] for k in (
                "box", "key", "table", "door", "apple", "pear", "room_a", "room_b")},
            "trap": True, "split": split, "version": 2, "levers": list(levers),
            "side": side if "T1" in levers else None}
    for lever, keys in (("T1", ("door2", "room_c")), ("T2", ("key2",)), ("T3", ("box2",))):
        for k in keys:
            meta[k] = n[k] if lever in levers else None
    return game, p, meta
