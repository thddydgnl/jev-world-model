"""JEV API client. Implements the contract in 설계.md §8.

Pins the model explicitly; never silently substitutes. Validates every
response against the requested criteria before returning.
"""
from __future__ import annotations

import json
import math
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from config import get_api_key

PINNED_MODEL = "jev-1.13.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"

_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _ROOT / "artifacts" / "jev_raw"


PRICE_PER_INPUT_TOKEN = 0.042 / 1e6      # docs.typesafe.ai/models, 2026-09-20


class JevError(RuntimeError):
    """Response could not be validated. Distinct from transport failure."""


class BudgetExceeded(RuntimeError):
    """Our own spend cap, not the provider's. Raised BEFORE a request so a run
    stops at a known point instead of being cut mid-episode by a 402."""


def _retry_delay(resp: httpx.Response, attempt: int, cap: float = 16.0) -> float:
    """Honour Retry-After in both numeric-seconds and HTTP-date forms."""
    raw = resp.headers.get("retry-after", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(raw)
            import datetime as _dt

            now = _dt.datetime.now(_dt.timezone.utc)
            if when.tzinfo is None:
                when = when.replace(tzinfo=_dt.timezone.utc)
            return max(0.0, (when - now).total_seconds())
        except (TypeError, ValueError):
            pass
    return min(2.0**attempt, cap)


def _validate(payload: dict[str, Any], out: dict[str, Any]) -> None:
    if out.get("model") != PINNED_MODEL:
        raise JevError(f"Model drift: requested {PINNED_MODEL}, got {out.get('model')!r}")
    answers = out.get("answers")
    if not isinstance(answers, dict):
        raise JevError("Response has no 'answers' object")
    for qid, spec in payload["questions"].items():
        if qid not in answers:
            raise JevError(f"Missing answer for question {qid!r}")
        ans = answers[qid]
        if spec["type"] == "noul":
            p = ans.get("noul")
            if not (isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1):
                raise JevError(f"Invalid noul probability for {qid!r}: {p!r}")
            continue
        probs = ans.get("probabilities")
        if not isinstance(probs, dict):
            raise JevError(f"No probabilities for {qid!r}")
        keys = list(spec["criteria"])
        if set(probs) != set(keys):
            raise JevError(
                f"Option mismatch for {qid!r}: "
                f"sent {sorted(keys)}, got {sorted(probs)}"
            )
        values = [float(probs[k]) for k in keys]
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in values):
            raise JevError(f"Invalid probability for {qid!r}: {values}")
        total = sum(values)
        # The API reports probabilities rounded to 2 decimals, so a sum of e.g.
        # 0.99 is display precision, not a malformed response. Anything beyond
        # that is a real failure.
        if not (0.97 <= total <= 1.03):
            raise JevError(f"Unnormalized distribution for {qid!r}: sum={total}")
        if total > 0 and abs(total - 1.0) > 1e-9:
            ans["probabilities"] = {k: float(probs[k]) / total for k in keys}


class JevClient:
    def __init__(self, save_raw: bool = True, max_attempts: int = 4,
                 budget_usd: float | None = None, outage_wait_s: float = 3600.0) -> None:
        self._key = get_api_key()
        self.budget_usd = budget_usd
        # Transient failures (timeouts, 429/529, 5xx) are retried at least
        # max_attempts times and then, with backoff up to a minute, until the
        # request has been failing for outage_wait_s. A 503 outage on 9/28
        # outlasted the four quick retries and killed two K4v3 units; waiting
        # changes nothing about the requests or their answers.
        self.outage_wait_s = outage_wait_s
        self._client = httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0))
        self._save_raw = save_raw
        self._max_attempts = max_attempts
        self.calls = 0
        self.input_tokens = 0
        if save_raw:
            _RAW_DIR.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def spent_usd(self) -> float:
        return self.input_tokens * PRICE_PER_INPUT_TOKEN

    def ask(self, state: Any, questions: dict[str, Any], tag: str = "") -> dict[str, Any]:
        if self.budget_usd is not None and self.spent_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"spend cap reached: ${self.spent_usd:.3f} of ${self.budget_usd:.2f} "
                f"after {self.calls} calls")
        payload = {"model": PINNED_MODEL, "state": state, "questions": questions}
        first_fail = None
        attempt = 0
        while True:
            try:
                r = self._client.post(
                    ENDPOINT,
                    headers={"Authorization": f"Bearer {self._key}"},
                    json=payload,
                )
            except (httpx.TimeoutException, httpx.TransportError):
                first_fail = time.monotonic() if first_fail is None else first_fail
                if not self._may_retry(first_fail, attempt):
                    raise
                self._wait(min(2.0**attempt, 60.0), attempt, "transport error")
                attempt += 1
                continue
            if r.status_code in (429, 529) or 500 <= r.status_code < 600:
                first_fail = time.monotonic() if first_fail is None else first_fail
                if not self._may_retry(first_fail, attempt):
                    r.raise_for_status()
                self._wait(_retry_delay(r, attempt, cap=60.0), attempt, f"HTTP {r.status_code}")
                attempt += 1
                continue
            r.raise_for_status()
            out = r.json()
            _validate(payload, out)
            self.calls += 1
            self.input_tokens += int(out.get("usage", {}).get("input_tokens", 0))
            if self._save_raw:
                stamp = f"{int(time.time() * 1000)}_{self.calls:05d}"
                name = f"{tag}_{stamp}.json" if tag else f"{stamp}.json"
                (_RAW_DIR / name).write_text(
                    json.dumps({"request": payload, "response": out}, ensure_ascii=False)
                )
            return out

    def _may_retry(self, first_fail: float, attempt: int) -> bool:
        return (attempt + 1 < self._max_attempts
                or time.monotonic() - first_fail < self.outage_wait_s)

    @staticmethod
    def _wait(seconds: float, attempt: int, why: str) -> None:
        if attempt >= 4 and attempt % 5 == 4:
            import sys
            print(f"[jev] {why}; retry {attempt + 1}, waiting {seconds:.0f}s", file=sys.stderr, flush=True)
        time.sleep(seconds)


def choice(instructions: str, criteria: dict[str, str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}
