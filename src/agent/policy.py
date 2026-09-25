"""Frozen common policy pi (설계.md §4).

Emits ONLY action-id sequences. It is never asked for predicted states,
success probabilities, or explanations, and it never sees the current
admissible-command list — only the static type-correct catalog.

The same frozen weights, prompt, seed and decoding serve every arm.
"""
from __future__ import annotations

import contextlib
import json
import re
import zlib
from typing import Any

PROMPT = """You control a deterministic text environment.
You are given the complete CURRENT physical facts, a goal, and a static
catalog of commands. A catalog entry is syntactically available but
may fail in the current state.

CURRENT FACTS:
{facts}

GOAL: {goal}

COMMAND CATALOG:
{catalog}

WHAT YOU ALREADY TRIED (oldest first):
{history}
A command marked FAILED did nothing: its preconditions were not met in the
state it was attempted from. Repeating it from an unchanged state will fail
again.

Return {k} distinct candidate command sequences, each of length {h}.
Order them by your preference for achieving the goal, best first.
Use only commands from the catalog, copied exactly.
Do not output predicted states, probabilities, explanations, or commands
outside the catalog. Do not assume a command succeeds merely because it is
listed.

Output JSON only: {{"plans": [["cmd", "cmd"], ...]}}"""


# v2 candidate lever P1 (kiis2026f/실험계획.md §4 V). General physics of the
# environment, true in every world; no world's names or solution. Off in v1.
AFFORDANCE_HINT = """NOTE: Closed containers and doors can be opened. A locked door or
container must first be unlocked with the key that matches it. Objects can be
inside closed containers and cannot be taken until the container is open."""


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Policy:
    TEMPERATURE = 0.7
    TOP_P = 0.9
    HISTORY = 10          # most recent attempts shown in the prompt

    def __init__(self, model_id: str = "Qwen/Qwen3-4B", device: str = "cuda:0",
                 seed: int = 20260921, max_new_tokens: int = 320) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device)
        self.model.eval()
        self.device = device
        self.calls = 0
        self.repairs = 0
        self.fallbacks = 0
        # An LLM world model may put a LoRA adapter on these same weights; it
        # sets this to the adapter's disable_adapter() so the policy always
        # generates from the base model and every arm sees the same candidates.
        self.adapter_off = contextlib.nullcontext
        # v2 candidate levers, set by the runner; the defaults are v1.
        self.hint = False          # P1: AFFORDANCE_HINT in the prompt
        self.samples = 1           # P2: independent samples merged per step

    def set_context(self, world_id: str, root: int, step: int) -> None:
        """Pin the sampling seed to WHERE we are, not to how many calls have
        happened. A global counter makes an arm's samples depend on which other
        arms ran first, which showed up as ~7pp swings between otherwise
        identical runs. Keyed this way, every arm sees the same randomness at
        the same (world, root, step) and arms can be split across processes.

        The arm is deliberately not part of the key. It used to be, which gave
        each arm different candidates even at step 0, where every arm faces the
        same state and the same prompt — noise in exactly the comparison the
        pairing is meant to make clean."""
        key = f"{world_id}|{root}|{step}"
        self._ctx_key = key
        self._ctx_seed = self.seed + (zlib.crc32(key.encode()) & 0x7FFFFFFF)

    def _generate(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        self.torch.manual_seed(getattr(self, "_ctx_seed", self.seed + self.calls))
        with self.torch.no_grad(), self.adapter_off():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=self.TEMPERATURE, top_p=self.TOP_P,
                pad_token_id=self.tok.eos_token_id)
        self.calls += 1
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)

    @classmethod
    def render_prompt(cls, facts: list[str], goal: str, catalog: list[str],
                      history: list[tuple[str, bool]] | None = None,
                      k: int = 8, h: int = 2, hint: bool = False) -> str:
        """The exact text the policy sees. Needs no model, so a run can record
        what its prompt was before any weights are loaded."""
        hist = history or []
        hist_txt = "\n".join(
            f'- "{a}" -> {"ok" if ok else "FAILED"}' for a, ok in hist[-cls.HISTORY:]
        ) or "- (nothing attempted yet)"
        text = PROMPT.format(facts="\n".join(f"- {f}" for f in facts),
                             goal=goal, catalog="\n".join(f"- {c}" for c in catalog),
                             history=hist_txt, k=k, h=h)
        if hint:
            text = text.replace("\nCURRENT FACTS:", f"\n{AFFORDANCE_HINT}\n\nCURRENT FACTS:", 1)
        return text

    def plans(self, facts: list[str], goal: str, catalog: list[str],
              history: list[tuple[str, bool]] | None = None,
              k: int = 8, h: int = 2) -> tuple[list[list[str]], str]:
        """`history` is the episode's own past attempts and whether the engine
        accepted them — allowed online information per 설계.md §3.5, and
        supplied identically to every arm.

        With `samples` > 1 (v2 lever P2) the same prompt is sampled again, each
        extra sample seeded by the situation key and its index, and new plans
        are appended after the first sample's. Sample 0 is exactly the
        single-sample call, so v1 candidates are a prefix of the merged list."""
        prompt = self.render_prompt(facts, goal, catalog, history, k, h, hint=self.hint)
        plans, status = self._sample(prompt, catalog, k, h)
        if self.samples > 1:
            key = getattr(self, "_ctx_key", None)
            seed0 = getattr(self, "_ctx_seed", None)
            for i in range(1, self.samples):
                if key is not None:
                    self._ctx_seed = self.seed + (zlib.crc32(f"{key}#{i}".encode()) & 0x7FFFFFFF)
                more, st = self._sample(prompt, catalog, k, h)
                if status == "none":
                    status = st
                plans += [p for p in more if p not in plans]
            if seed0 is not None:
                self._ctx_seed = seed0

        if not plans:
            # Deterministic fallback; counted in the error rate, never dropped.
            self.fallbacks += 1
            status = "fallback"
            plans = [[c] for c in catalog[:k]]
        return plans[:k * self.samples], status

    def _sample(self, prompt: str, catalog: list[str], k: int,
                h: int) -> tuple[list[list[str]], str]:
        """One sample: plans parsed from the reply (at most k), after at most one
        schema repair. Status "none" when nothing usable came back."""
        status = "ok"
        raw = self._generate(prompt)
        data = _extract_json(raw)
        if data is None or not isinstance(data.get("plans"), list):
            # One schema repair, allowed for every arm alike (설계.md §4).
            self.repairs += 1
            status = "repaired"
            raw = self._generate(prompt + '\n\nRespond with JSON only, '
                                          'starting with {"plans":')
            data = _extract_json(raw)

        plans: list[list[str]] = []
        allowed = set(catalog)
        if data and isinstance(data.get("plans"), list):
            for p in data["plans"]:
                if not isinstance(p, list):
                    continue
                seq = [str(c).strip() for c in p[:h] if str(c).strip() in allowed]
                if seq and seq not in plans:
                    plans.append(seq)
        return plans[:k], (status if plans else "none")
