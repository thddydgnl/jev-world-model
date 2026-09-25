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


_PROBE: tuple | None = None
# v2 settings (kiis2026f/실험계획.md §4 V): None is v1. When set, the probe world
# is the v2 dev world t000 with these levers, the policy prompt is rendered with
# the hint if on, and the settings join the condition — so v1 hashes are untouched.
_V2: dict | None = None


def configure_v2(levers: tuple[str, ...], hint: bool, samples: int) -> None:
    global _V2, _PROBE
    _V2 = {"levers": list(levers), "policy_hint": bool(hint), "policy_samples": int(samples)}
    _PROBE = None


def _probe_world() -> tuple:
    """Dev world t000's first state: the fixed input every probe renders on."""
    global _PROBE
    if _PROBE is None:
        from agent.task import state_schema, static_catalog, trap_goal_spec
        from env.serialize import canonical_state
        from env.worlds import build_trap_world, build_trap_world_v2
        with tempfile.TemporaryDirectory() as d:
            if _V2 is not None:
                game, path, meta = build_trap_world_v2(0, Path(d), "dev", tuple(_V2["levers"]))
            else:
                game, path, meta = build_trap_world(0, Path(d), random.Random(0))
            env = textworld.start(str(path), request_infos=EnvInfos(facts=True))
            env.reset()
            canon = canonical_state(list(env.state["_facts"]), game)
            env.close()
        _PROBE = (canon, state_schema(game, meta), trap_goal_spec(game, meta),
                  static_catalog(game), meta)
    return _PROBE


def _probe_texts(k: int, h: int) -> dict[str, str]:
    """Render every model-facing text once, on dev world t000's first state."""
    from agent.policy import Policy
    from forecast import render_facts
    from wm.jev_forecaster import validity_question
    from wm.recursive_forecaster import (build_state, step_questions,
                                         validity_state)

    canon, schema, goal, catalog, meta = _probe_world()
    action = f"take {meta['key']} from {meta['box']}"
    facts = render_facts(canon)
    return {
        "state_render": json.dumps(facts, ensure_ascii=False),
        "policy_prompt": Policy.render_prompt(facts, goal["text"], catalog,
                                              [(action, False)], k, h,
                                              hint=bool(_V2 and _V2["policy_hint"])),
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
    cond = {
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
    if _V2 is not None:
        cond["env"]["structure"] = "trap_v2"
        cond["v2"] = dict(_V2)
    return cond


# What each arm adds on top of the shared condition. A change here changes that
# arm's hash only.
ARM_CONFIG: dict[str, dict[str, Any]] = {
    "C": {"selector": "first action of the policy's top plan"},
    # The planner's failure memory (condition["failure_memory"]) reached these
    # two only after K1 found them without it; the flag makes their hashes
    # differ from the runs made before (kiis2026f/실험계획.md §11).
    "validity": {"selector": "utility with engine validity, goal terms zeroed",
                 "failure_memory": True},
    "oracle": {"selector": "utility with the engine's true endpoint",
               "failure_memory": True},
    "A_jev": {"world_model": "recursive one-step, typed", "backend": "jev"},
    "R": {"world_model": "recursive one-step, typed", "backend": "jev"},
    "A": {"world_model": "direct-horizon, execution-pattern mixture", "backend": "jev"},
    "Aend": {"world_model": "direct-horizon, endpoint only", "backend": "jev"},
    "Aval": {"world_model": "direct-horizon, validity only", "backend": "jev"},
}


LLM_ARMS = ("B0_typed", "B_typed", "D0_gen", "D_gen")
ADAPTER_ARMS = ("B_typed", "D_gen")


def _llm_config(arm: str) -> dict[str, Any]:
    from wm import llm_backends as lb
    canon, schema, _goal, _catalog, meta = _probe_world()
    prompts = lb.probe_prompts(canon, schema, f"take {meta['key']} from {meta['box']}")
    if arm.startswith("B"):
        return {"world_model": "recursive one-step, typed", "backend": "qwen, policy weights",
                "readout": "next-token probability of the option letters",
                "prompt": digest(prompts["typed"])}
    return {"world_model": "recursive one-step, generative", "backend": "qwen, policy weights",
            "decoding": "greedy", "max_new_tokens": lb.GEN_MAX_NEW_TOKENS,
            "retry": "one sample, T=0.7 top_p=0.9, seeded by state and action",
            "on_parse_failure": "counted; treated as not executed",
            "parser": lb.PARSER_VERSION, "prompt": digest(prompts["generative"])}


def arm_config(arm: str, adapter: dict | None = None) -> dict[str, Any]:
    """What an arm adds to the shared condition. LLM arms include their
    rendered prompt and, for B and D, the adapter they were given."""
    if arm in LLM_ARMS:
        cfg = _llm_config(arm)
        if arm in ADAPTER_ARMS:
            if adapter is None:
                raise ValueError(f"{arm} needs an adapter")
            cfg["adapter"] = adapter
        return cfg
    return ARM_CONFIG.get(arm, {"unregistered": arm})


def arm_hash(arm: str, adapter: dict | None = None) -> str:
    return digest(arm_config(arm, adapter))


def adapter_info(path: str) -> dict[str, Any]:
    """Identity of a trained LoRA adapter: its weights' hash and how it was made."""
    p = Path(path)
    weights = p / "adapter_model.safetensors"
    meta = json.loads((p / "train_meta.json").read_text()) if (p / "train_meta.json").exists() else {}
    return {"weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()[:16],
            "n_transitions": meta.get("n_transitions"), "data_sha": meta.get("data_sha"),
            "lora": meta.get("lora"), "epochs": len(meta.get("epochs", []))}


def software() -> dict[str, str | None]:
    out: dict[str, str | None] = {"textworld": textworld.__version__}
    for name in ("torch", "transformers", "peft"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out
