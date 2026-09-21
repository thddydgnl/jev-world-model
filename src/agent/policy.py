"""Frozen common policy pi (설계.md §4).

Emits ONLY action-id sequences. It is never asked for predicted states,
success probabilities, or explanations, and it never sees the current
admissible-command list — only the static type-correct catalog.

The same frozen weights, prompt, seed and decoding serve every arm.
"""
from __future__ import annotations

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


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Policy:
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

    def set_context(self, world_id: str, root: int, arm: str, step: int) -> None:
        """Pin the sampling seed to WHERE we are, not to how many calls have
        happened. A global counter makes an arm's samples depend on which other
        arms ran first, which showed up as ~7pp swings between otherwise
        identical runs. Keyed this way, every arm sees the same randomness at
        the same (world, root, step) and arms can be split across processes."""
        key = f"{world_id}|{root}|{arm}|{step}"
        self._ctx_seed = self.seed + (zlib.crc32(key.encode()) & 0x7FFFFFFF)

    def _generate(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        self.torch.manual_seed(getattr(self, "_ctx_seed", self.seed + self.calls))
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=self.tok.eos_token_id)
        self.calls += 1
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True)

    def plans(self, facts: list[str], goal: str, catalog: list[str],
              history: list[tuple[str, bool]] | None = None,
              k: int = 8, h: int = 2) -> tuple[list[list[str]], str]:
        """`history` is the episode's own past attempts and whether the engine
        accepted them — allowed online information per 설계.md §3.5, and
        supplied identically to every arm."""
        hist = history or []
        hist_txt = "\n".join(
            f'- "{a}" -> {"ok" if ok else "FAILED"}' for a, ok in hist[-10:]
        ) or "- (nothing attempted yet)"
        prompt = PROMPT.format(facts="\n".join(f"- {f}" for f in facts),
                               goal=goal, catalog="\n".join(f"- {c}" for c in catalog),
                               history=hist_txt, k=k, h=h)
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

        if not plans:
            # Deterministic fallback; counted in the error rate, never dropped.
            self.fallbacks += 1
            status = "fallback"
            plans = [[c] for c in catalog[:k]]
        return plans[:k], status
