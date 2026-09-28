"""JEV client retries (9/28): transient failures are waited out up to the
outage window, other errors are not retried, and a success after a long
outage returns the answer unchanged. No network: the HTTP client, the clock
and sleep are replaced."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import httpx
import jev_client as jc


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


class FakeHttp:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def post(self, url, headers=None, json=None):
        self.calls += 1
        code = self.script.pop(0) if self.script else self.last
        self.last = code
        req = httpx.Request("POST", url)
        if code == "timeout":
            raise httpx.ReadTimeout("timeout", request=req)
        body = {"model": jc.PINNED_MODEL, "answers": {"q": {"noul": 0.7}},
                "usage": {"input_tokens": 100}} if code == 200 else {}
        return httpx.Response(code, json=body, request=req)

    def close(self):
        pass


def client(script, clock, **kw):
    jc.get_api_key = lambda: "test-key"
    jc.time.monotonic = clock.monotonic
    jc.time.sleep = clock.sleep
    c = jc.JevClient(save_raw=False, **kw)
    c._client = FakeHttp(script)
    return c


def main() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    q = {"q": {"type": "noul", "text": "does it run?"}}
    jc._validate = lambda payload, out: None      # answer shape is not what is tested here

    clk = Clock()
    c = client([503] * 20 + [200], clk)
    out = c.ask({"s": 1}, q)
    check(out["answers"]["q"]["noul"] == 0.7 and c._client.calls == 21,
          f"20 x 503 then 200: answered after {c._client.calls} calls, {clk.t:.0f}s waited")
    check(max(clk.sleeps) <= 60.0, f"backoff capped at 60s (max {max(clk.sleeps):.0f})")
    check(c.input_tokens == 100, "tokens counted once, for the answered request")

    clk = Clock()
    c = client([503], clk)            # 503 forever
    try:
        c.ask({"s": 1}, q)
        check(False, "endless 503 must raise")
    except httpx.HTTPStatusError:
        check(3600 <= clk.t < 3600 + 61, f"endless 503 raises after the outage window ({clk.t:.0f}s)")

    clk = Clock()
    c = client([400], clk)
    try:
        c.ask({"s": 1}, q)
        check(False, "400 must raise")
    except httpx.HTTPStatusError:
        check(c._client.calls == 1 and clk.t == 0, "400 is not retried")

    clk = Clock()
    c = client(["timeout"] * 6 + [200], clk)
    out = c.ask({"s": 1}, q)
    check(c._client.calls == 7, f"6 timeouts then 200: answered after {c._client.calls} calls")

    clk = Clock()
    c = client([503, 503, 200], clk, outage_wait_s=0)
    out = c.ask({"s": 1}, q)
    check(c._client.calls == 3, "short blips still get the max_attempts quick retries")

    clk = Clock()
    c = client([503] * 10, clk, outage_wait_s=0)
    try:
        c.ask({"s": 1}, q)
        check(False, "outage_wait_s=0 must give up after max_attempts")
    except httpx.HTTPStatusError:
        check(c._client.calls == 4, f"outage_wait_s=0 gives up after max_attempts ({c._client.calls} calls)")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
