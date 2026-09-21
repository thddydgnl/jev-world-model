"""Build the JEV forecast request (설계.md §7).

The state carries canonical ids AND a deterministic human-readable rendering of
the same facts. The rendering adds no new facts and no transition rules — it
only restates what is already there, so that a low score cannot be blamed on
opaque variable names. That matters because the 0-step condition has to isolate
"can it read our input" from "can it predict consequences".
"""
from __future__ import annotations

from typing import Any

ROLLOUT_CONVENTION = (
    "Attempt each command in `action_sequence` in order, starting from "
    "`current_state`. A command that cannot be executed in the state reached so "
    "far fails and changes nothing; continue with the next command. No new "
    "observations arrive during the sequence. Report the state after the whole "
    "sequence has been attempted."
)

PRED_TEMPLATES = {
    "at": "{0} is on the floor of {1}.",
    "in": "{0} is inside {1}.",
    "on": "{0} is on top of {1}.",
    "open": "{0} is open.",
    "closed": "{0} is closed (not locked).",
    "locked": "{0} is locked.",
    "match": "{0} is the key that fits {1}.",
    "link": "{1} connects {0} and {2}.",
    "free": "the passage between {0} and {1} is unobstructed.",
    "north_of": "{0} is north of {1}.",
    "south_of": "{0} is south of {1}.",
    "east_of": "{0} is east of {1}.",
    "west_of": "{0} is west of {1}.",
}


def _display(entities: list[dict], vid: str) -> str:
    for e in entities:
        if e["id"] == vid:
            return f"the {e['name']} ({vid})"
    if vid == "P":
        return "the player (P)"
    if vid == "I":
        return "the player's inventory (I)"
    return vid


def render_facts(state: dict[str, Any]) -> list[str]:
    ents = state["entities"]
    out = []
    for row in state["dynamic_facts"] + state["static_facts"]:
        pred, args = row[0], row[1:]
        tpl = PRED_TEMPLATES.get(pred)
        shown = [_display(ents, a) for a in args]
        if tpl:
            try:
                s = tpl.format(*shown)
                out.append(s[:1].upper() + s[1:])  # keep ids' case intact
                continue
            except IndexError:
                pass
        out.append(f"{pred}({', '.join(shown)})")
    return sorted(out)


def build_state(current_state: dict, catalog: list[dict], actions: list[str],
                horizon: int) -> dict[str, Any]:
    return {
        "current_state": {
            "player_location": current_state["player_room"],
            "entities": current_state["entities"],
            "facts_canonical": current_state["dynamic_facts"] + current_state["static_facts"],
            "facts_readable": render_facts(current_state),
        },
        "action_sequence": actions,
        "horizon": horizon,
        "rollout_convention": ROLLOUT_CONVENTION,
        "query_catalog": [
            {"id": q["id"], "about": q["label"], "options": q["options"]}
            for q in catalog
        ],
    }


def build_questions(catalog: list[dict], actions: list[str], horizon: int) -> dict[str, Any]:
    if horizon == 0:
        frame = (
            "Report the CURRENT state described in `current_state`. "
            "No commands are attempted."
        )
    else:
        listing = " then ".join(f"`{a}`" for a in actions) if actions else "(no commands)"
        frame = (
            f"Starting from `current_state`, attempt exactly these {horizon} "
            f"command(s) in this order: {listing}. Apply `rollout_convention`. "
            f"Report the resulting state AFTER the sequence, not the current state."
        )
    asks = {
        "parent": "Where is the {} located? Give its DIRECT container, supporter, "
                  "room floor, or the player's inventory.",
        "mode": "Is the {} open, closed but unlocked, or locked?",
        "room": "Which room is the {} in?",
    }
    questions = {}
    for q in catalog:
        ask = asks[q["kind"]].format(q["label"])
        questions[q["id"]] = {
            "type": "choice",
            "instructions": f"{frame} Question: {ask}",
            "criteria": dict(q["options"]),
        }
    return questions
