"""What a run was: the condition it ran under, the code, and the worlds.

Numbers from different conditions once ended up in one table because nothing
recorded which condition a run had used (docs/protocol.md A.4). Every run now
writes its condition down and hashes it, every episode carries that hash, and
scripts/analyze_arms.py refuses to pool two hashes (kiis2026f/실험계획.md §2).

The hash covers what shapes a decision or a model input: the policy prompt and
the typed questions rendered on a fixed probe world, decoding, K, H, the step
cap, the utility, the rollout parameters and the JEV model. It is built from
those values, not from source files, so adding an arm elsewhere in the runner
does not change the condition of arms that have already run. The flip side is
that a change to decision logic must be reflected in the labels below by hand.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import textworld
from textworld import EnvInfos

_ROOT = Path(__file__).resolve().parent.parent


def digest(obj: Any) -> str:
    blob = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def revision() -> dict[str, Any]:
    """Commit the code came from. The GPU server is not a git checkout, so
    scripts/sync_server.sh leaves a REVISION file next to the code."""
    try:
        commit = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(_ROOT), "status", "--porcelain",
                                "--", "src", "scripts", "tests"],
                               capture_output=True, text=True, check=True).stdout
        return {"commit": commit, "dirty_files": len(dirty.splitlines()), "source": "git"}
    except (OSError, subprocess.CalledProcessError):
        pass
    f = _ROOT / "REVISION"
    if f.exists():
        return {"line": f.read_text().strip(), "source": "REVISION"}
    return {"source": "unknown"}


def _probe_texts(k: int, h: int) -> dict[str, str]:
    """Render every model-facing text once, on dev world t000's first state."""
    from agent.policy import Policy
    from agent.task import state_schema, static_catalog, trap_goal_spec
    from env.serialize import canonical_state
    from env.worlds import build_trap_world
    from forecast import render_facts
    from wm.jev_forecaster import validity_question
    from wm.recursive_forecaster import (build_state, step_questions,
                                         validity_state)

    with tempfile.TemporaryDirectory() as d:
        game, path, meta = build_trap_world(0, Path(d), random.Random(0))
        env = textworld.start(str(path), request_infos=EnvInfos(facts=True))
        env.reset()
        canon = canonical_state(list(env.state["_facts"]), game)
        env.close()
    schema, goal = state_schema(game, meta), trap_goal_spec(game, meta)
    catalog = static_catalog(game)
    action = f"take {meta['key']} from {meta['box']}"
    facts = render_facts(canon)
    return {
        "state_render": json.dumps(facts, ensure_ascii=False),
        "policy_prompt": Policy.render_prompt(facts, goal["text"], catalog,
                                              [(action, False)], k, h),
        "typed_step": json.dumps([build_state(canon, schema, action),
                                  step_questions(schema, action)],
                                 sort_keys=True, ensure_ascii=False),
        "typed_validity": json.dumps([validity_state(canon),
                                      validity_question(action, [])],
                                     sort_keys=True, ensure_ascii=False),
        "goal": json.dumps(goal, sort_keys=True, ensure_ascii=False),
    }


def condition(*, model: str, max_new_tokens: int, k: int, h: int,
              cap: int, trap: bool) -> dict[str, Any]:
    """Condition F as data. Everything here is shared by every arm."""
    from agent import task
    from agent.policy import Policy
    from jev_client import PINNED_MODEL
    from wm import recursive_forecaster as rf
    from wm.jev_forecaster import INCLUDE_QUERY_CATALOG

    texts = _probe_texts(k, h)
    return {
        "env": {"textworld": textworld.__version__,
                "structure": "trap" if trap else "basic",
                "step_cap": cap,
                "root_prefix": "0-2 random admissible commands, not look/inventory/examine"},
        "policy": {"model": model, "dtype": "bfloat16",
                   "temperature": Policy.TEMPERATURE, "top_p": Policy.TOP_P,
                   "max_new_tokens": max_new_tokens, "history": Policy.HISTORY,
                   "schema_repairs": 1, "seed_key": "world|root|step",
                   "prompt": digest(texts["policy_prompt"])},
        "candidates": {"K": k, "H": h, "prefixes": "unique, lengths 1..H"},
        "utility": {"alpha": task.ALPHA, "lambda": task.LAMBDA, "cost": task.COST,
                    "beta": task.BETA, "invalid_term": "count"},
        "failure_memory": "drop prefixes whose first action already failed from the same state",
        "rollout": {"beam": rf.BEAM_K, "prune_w": rf.PRUNE_W,
                    "cache": "state fingerprint + action, whole episode"},
        "typed_questions": {"step": digest(texts["typed_step"]),
                            "validity": digest(texts["typed_validity"]),
                            "validity_request": "separate",
                            "query_catalog_in_state": INCLUDE_QUERY_CATALOG},
        "state_render": digest(texts["state_render"]),
        "goal": digest(texts["goal"]),
        "jev_model": PINNED_MODEL,
    }


# What each arm adds on top of the shared condition. A change here changes that
# arm's hash only.
ARM_CONFIG: dict[str, dict[str, Any]] = {
    "C": {"selector": "first action of the policy's top plan"},
    "validity": {"selector": "utility with engine validity, goal terms zeroed"},
    "oracle": {"selector": "utility with the engine's true endpoint"},
    "A_jev": {"world_model": "recursive one-step, typed", "backend": "jev"},
    "R": {"world_model": "recursive one-step, typed", "backend": "jev"},
    "A": {"world_model": "direct-horizon, execution-pattern mixture", "backend": "jev"},
    "Aend": {"world_model": "direct-horizon, endpoint only", "backend": "jev"},
    "Aval": {"world_model": "direct-horizon, validity only", "backend": "jev"},
}


def arm_hash(arm: str) -> str:
    return digest(ARM_CONFIG.get(arm, {"unregistered": arm}))
