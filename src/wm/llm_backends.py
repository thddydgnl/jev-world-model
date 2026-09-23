"""Qwen as a one-step world model, in the two query styles
(kiis2026f/실험계획.md §1.2–1.3).

Typed — the JEV style. The model gets the very payload and questions JEV gets,
one prompt per question, the options listed as letters. An option's probability
is the next-token probability of its letter, renormalised over the letters on
offer. LlmJudgeClient answers in JEV's response shape, so TypedStep and the
rollout run unchanged on top of it.

Generative. The model gets the same payload without questions and writes the
next state itself: a result line and the fact list. A parser reads the schema
variables back out. Nothing to choose from, no probabilities: one prediction,
so the rollout's beam stays one state wide.

Both run on the policy's own weights. A LoRA adapter, when one is loaded, is
switched off around the policy's generation (Policy.adapter_off), so every arm
still sees the same candidates.
"""
from __future__ import annotations

import json
import random
import re
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

from agent.task import read_schema_values
from wm.jev_forecaster import validity_question
from wm.recursive_forecaster import (StepPrediction, build_state, step_questions,
                                     validity_state)

LETTERS = "ABCDEFGHIJKLMNOP"

TYPED_SYSTEM = ("You answer questions about a deterministic text environment. "
                "Exactly one option is correct.")

GEN_SYSTEM = "You simulate a deterministic text environment."
GEN_INSTRUCTIONS = (
    "Attempt the command in `action_sequence` once, starting from `current_state`, "
    "as `rollout_convention` describes. Reply in exactly this format and nothing else:\n"
    "result: executes    (or: result: fails)\n"
    "then every fact of the resulting state, one per line, written as "
    "predicate(argument, argument) with the ids used in `facts_canonical`.")
GEN_MAX_NEW_TOKENS = 320
PARSER_VERSION = "v1"


# ------------------------------------------------------------------ prompts

def typed_messages(state: dict, question: dict,
                   order: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """Chat messages for one typed question, and the option keys in the order
    shown. Keys stay visible next to their text, as they are in JEV's request."""
    keys = list(question["criteria"]) if order is None else list(order)
    opts = "\n".join(f"{LETTERS[i]}. [{k}] {question['criteria'][k]}"
                     for i, k in enumerate(keys))
    user = (f"STATE:\n{json.dumps(state, ensure_ascii=False)}\n\n"
            f"QUESTION:\n{question['instructions']}\n\n"
            f"OPTIONS:\n{opts}\n\n"
            "Answer with the letter of the correct option only.")
    return [{"role": "system", "content": TYPED_SYSTEM},
            {"role": "user", "content": user}], keys


def fact_str(row: list[str]) -> str:
    return f"{row[0]}({', '.join(row[1:])})"


def gen_user(canon: dict, schema: list[dict], action: str) -> str:
    """The typed step request's state, without the questions."""
    state = build_state(canon, schema, action)
    return f"STATE:\n{json.dumps(state, ensure_ascii=False)}\n\n{GEN_INSTRUCTIONS}"


def gen_target(executes: bool, next_facts: list[list[str]]) -> str:
    return "\n".join(["result: executes" if executes else "result: fails"]
                     + [fact_str(r) for r in next_facts])


_EXAMPLE: tuple[str, str] | None = None


def format_example() -> tuple[str, str]:
    """The one worked example the generative prompt carries: dev world t000,
    first state, `open` on the box. Dev names never occur in train or test, and
    the example is fixed here once (K2) and shared by D0 and D."""
    global _EXAMPLE
    if _EXAMPLE is None:
        import textworld
        from textworld import EnvInfos
        from agent.task import state_schema
        from env.serialize import canonical_state
        from env.worlds import build_trap_world
        with tempfile.TemporaryDirectory() as d:
            game, path, meta = build_trap_world(0, Path(d), random.Random(0))
            env = textworld.start(str(path), request_infos=EnvInfos(
                facts=True, admissible_commands=True))
            env.reset()
            canon = canonical_state(list(env.state["_facts"]), game)
            action = f"open {meta['box']}"
            runs = action in env.state["admissible_commands"]
            env.step(action)
            nxt = canonical_state(list(env.state["_facts"]), game)
            env.close()
        schema = state_schema(game, meta)
        _EXAMPLE = (gen_user(canon, schema, action),
                    gen_target(runs, nxt["dynamic_facts"] + nxt["static_facts"]))
    return _EXAMPLE


def gen_messages(canon: dict, schema: list[dict], action: str) -> list[dict]:
    ex_user, ex_answer = format_example()
    return [{"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": ex_user},
            {"role": "assistant", "content": ex_answer},
            {"role": "user", "content": gen_user(canon, schema, action)}]


# ------------------------------------------------------------------ parser

_RESULT = re.compile(r"^\s*result\s*:\s*(executes|fails)\b", re.I)
_FACT = re.compile(r"^\s*[-*]?\s*([A-Za-z_]+)\s*\(([^()]*)\)\s*[,.]?\s*$")


def _rows(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        m = _FACT.match(line)
        if m:
            rows.append([m.group(1)] + [a.strip().strip("'\"") for a in m.group(2).split(",")])
        elif line.strip().startswith("["):
            try:
                row = json.loads(line.strip().rstrip(","))
                if isinstance(row, list) and all(isinstance(x, str) for x in row):
                    rows.append(row)
            except json.JSONDecodeError:
                pass
    return rows


def _candidates(q: dict, rows: list[list[str]]) -> set[str]:
    if q["kind"] == "room":
        return {r[2] for r in rows if len(r) == 3 and r[0] == "at" and r[1] == "P"}
    if q["kind"] == "parent":
        o = q["target"]
        out = {("inventory" if r[2] == "I" else f"{r[0]}:{r[2]}")
               for r in rows if len(r) == 3 and r[0] in ("in", "on", "at") and r[1] == o}
        if ["eaten", o] in rows:
            out.add("gone")
        return out
    x = q["target"]
    return {val for pred, val in (("open", "open"), ("closed", "closed_unlocked"),
                                  ("locked", "locked")) if [pred, x] in rows}


def parse_generation(text: str, schema: list[dict],
                     canon: dict) -> tuple[dict | None, float | None]:
    """(values, p_exec) from a generated reply, or (None, None) if unusable.

    `fails` means the state is unchanged, whatever facts follow. `executes`
    needs exactly one legal value for every schema variable; a missing or
    doubled variable is a failure, not a guess.
    """
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None, None
    m = _RESULT.match(lines[0])
    if not m:
        return None, None
    if m.group(1).lower() == "fails":
        return read_schema_values(canon, schema), 0.0
    rows = _rows(lines[1:])
    values = {}
    for q in schema:
        c = _candidates(q, rows)
        if len(c) != 1:
            return None, None
        v = c.pop()
        if v not in q["options"]:
            return None, None
        values[q["id"]] = v
    return values, 1.0


# ------------------------------------------------------------------ model

def _body_and_head(model):
    """The transformer body and the LM head of a (possibly LoRA-wrapped)
    causal LM, so a score reads one hidden state instead of full logits."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return inner.model, inner.lm_head


class QwenScorer:
    """Batched scoring and generation on one causal LM."""

    def __init__(self, model, tok, device: str, batch: int = 8) -> None:
        import torch
        self.torch = torch
        self.model, self.tok, self.device, self.batch = model, tok, device, batch
        self.letter_ids = []
        for c in LETTERS:
            ids = tok.encode(c, add_special_tokens=False)
            assert len(ids) == 1, f"letter {c!r} is not one token"
            self.letter_ids.append(ids[0])
        self.eos_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.eos_id
        self.seconds = 0.0          # time spent in this scorer, for the cost column

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def option_probs(self, prompts: list[list[dict]], n_options: list[int]) -> list[list[float]]:
        """For each prompt, the distribution over its first n letters."""
        torch = self.torch
        t0 = time.time()
        ids = [self.prompt_ids(p) for p in prompts]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out: list[list[float] | None] = [None] * len(ids)
        body, head = _body_and_head(self.model)
        with torch.no_grad():
            for s in range(0, len(order), self.batch):
                chunk = order[s:s + self.batch]
                L = max(len(ids[i]) for i in chunk)
                x = torch.full((len(chunk), L), self.pad_id, dtype=torch.long)
                att = torch.zeros((len(chunk), L), dtype=torch.long)
                for j, i in enumerate(chunk):          # right padding: the last real
                    x[j, :len(ids[i])] = torch.tensor(ids[i])   # token keeps its position
                    att[j, :len(ids[i])] = 1
                x, att = x.to(self.device), att.to(self.device)
                h = body(input_ids=x, attention_mask=att, use_cache=False).last_hidden_state
                last = torch.tensor([len(ids[i]) - 1 for i in chunk], device=self.device)
                logits = head(h[torch.arange(len(chunk), device=self.device), last]).float()
                for j, i in enumerate(chunk):
                    sel = logits[j, self.letter_ids[:n_options[i]]]
                    out[i] = torch.softmax(sel, dim=-1).tolist()
        self.seconds += time.time() - t0
        return out  # type: ignore[return-value]

    def generate(self, prompts: list[list[dict]], max_new_tokens: int,
                 sample: bool = False, seed: int | None = None) -> list[str]:
        torch = self.torch
        t0 = time.time()
        texts = [self.tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False) for p in prompts]
        outs: list[str] = []
        side = self.tok.padding_side
        self.tok.padding_side = "left"
        try:
            for s in range(0, len(texts), self.batch):
                enc = self.tok(texts[s:s + self.batch], return_tensors="pt", padding=True,
                               add_special_tokens=False).to(self.device)
                kw: dict[str, Any] = dict(max_new_tokens=max_new_tokens,
                                          pad_token_id=self.pad_id, eos_token_id=self.eos_id)
                if sample:
                    if seed is not None:
                        torch.manual_seed(seed)
                    kw.update(do_sample=True, temperature=0.7, top_p=0.9)
                else:
                    kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
                with torch.no_grad():
                    gen = self.model.generate(**enc, **kw)
                for row in gen[:, enc["input_ids"].shape[1]:]:
                    outs.append(self.tok.decode(row, skip_special_tokens=True))
        finally:
            self.tok.padding_side = side
        self.seconds += time.time() - t0
        return outs


class LlmJudgeClient:
    """JEV's `ask` interface, answered by a local model's option probabilities."""

    def __init__(self, scorer: QwenScorer) -> None:
        self.scorer = scorer
        self.calls = 0

    def ask(self, state: dict, questions: dict, tag: str = "") -> dict:
        return self.ask_many([(state, questions, tag)])[0]

    def ask_many(self, requests: list[tuple[dict, dict, str]]) -> list[dict]:
        flat, index = [], []
        for r, (state, questions, _tag) in enumerate(requests):
            for qid, q in questions.items():
                msgs, keys = typed_messages(state, q)
                flat.append(msgs)
                index.append((r, qid, keys))
        probs = self.scorer.option_probs(flat, [len(k) for _, _, k in index])
        out: list[dict] = [{"answers": {}} for _ in requests]
        for (r, qid, keys), p in zip(index, probs):
            out[r]["answers"][qid] = {"probabilities": dict(zip(keys, p))}
        self.calls += len(requests)
        return out


class GenerativeStep:
    """Step backend for the generative arms (D0, D)."""

    def __init__(self, scorer: QwenScorer, schema: list[dict],
                 max_new_tokens: int = GEN_MAX_NEW_TOKENS) -> None:
        self.scorer, self.schema, self.max_new_tokens = scorer, schema, max_new_tokens
        self.replies: list[str] = []      # last batch, for inspection
        self.failed: list[dict] = []      # replies that could not be parsed, kept for the error analysis

    def __call__(self, canon: dict, action: str) -> StepPrediction:
        return self.predict_batch([(canon, action)])[0]

    def predict_batch(self, pairs: list[tuple[dict, str]]) -> list[StepPrediction]:
        msgs = [gen_messages(c, self.schema, a) for c, a in pairs]
        texts = self.scorer.generate(msgs, self.max_new_tokens)
        self.replies = texts
        preds: list[StepPrediction | None] = [None] * len(pairs)
        for i, t in enumerate(texts):
            values, p = parse_generation(t, self.schema, pairs[i][0])
            if values is not None:
                preds[i] = StepPrediction(values, {k: 1.0 for k in values}, p, requests=1)
                continue
            # One more try, sampled with a seed fixed by the situation — the same
            # allowance the policy gets for a malformed reply.
            fp = "|".join(",".join(r) for r in pairs[i][0]["dynamic_facts"])
            seed = zlib.crc32(f"{fp}#{pairs[i][1]}".encode()) & 0x7FFFFFFF
            t2 = self.scorer.generate([msgs[i]], self.max_new_tokens, sample=True, seed=seed)[0]
            values, p = parse_generation(t2, self.schema, pairs[i][0])
            if values is not None:
                preds[i] = StepPrediction(values, {k: 1.0 for k in values}, p, requests=2)
            else:
                now = read_schema_values(pairs[i][0], self.schema)
                preds[i] = StepPrediction(now, {k: 1.0 for k in now}, 0.0,
                                          requests=2, parse_failed=True)
                self.failed.append({"action": pairs[i][1], "reply": t[:600], "retry": t2[:600]})
        return preds  # type: ignore[return-value]


def probe_prompts(canon: dict, schema: list[dict], action: str) -> dict[str, str]:
    """Rendered prompts of both styles, for the arm hash."""
    st = build_state(canon, schema, action)
    q = step_questions(schema, action)
    first = next(iter(q.values()))
    typed, _ = typed_messages(st, first)
    valid, _ = typed_messages(validity_state(canon), validity_question(action, []))
    return {"typed": json.dumps(typed + valid, ensure_ascii=False),
            "generative": json.dumps(gen_messages(canon, schema, action), ensure_ascii=False)}
