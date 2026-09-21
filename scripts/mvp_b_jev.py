"""MVP-B: does JEV's typed judgement carry action-conditioned future information?
(설계.md §25, gates in §24)

Four conditions:
  h0     0-step current-state extraction  -> tests OUR INPUT, not JEV's ability
  h1     1-step forecast                  -> the core measurement
  h2     2-step forecast (incl. order swaps)
  noact  future framing, EMPTY action list -> AB1 "action absent" control

Read h0 first. If h0 is low, nothing below it is interpretable.
"""
from __future__ import annotations

import json
import math
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecast import build_state, build_questions
from jev_client import JevClient

DATA = Path("data/mvp_a/prefixes.jsonl")
OUT = Path("artifacts/mvp_b")
CONCURRENCY = 8
EPS = 1e-6


def jobs(recs: list[dict]) -> list[dict]:
    out, seen_root = [], set()
    for i, r in enumerate(recs):
        out.append({"cond": f"h{r['horizon']}", "idx": i,
                    "actions": r["actions"], "horizon": r["horizon"]})
        if r["root_id"] not in seen_root:
            seen_root.add(r["root_id"])
            out.append({"cond": "h0", "idx": i, "actions": [], "horizon": 0})
            out.append({"cond": "noact", "idx": i, "actions": [], "horizon": 2})
    return out


def main() -> int:
    recs = [json.loads(l) for l in DATA.read_text().splitlines()]
    todo = jobs(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"records={len(recs)}  requests={len(todo)}  concurrency={CONCURRENCY}")

    lock = threading.Lock()
    results, failures = [], []

    with JevClient(save_raw=True) as jev:
        def run(job):
            r = recs[job["idx"]]
            st = build_state(r["current_state"], r["catalog"], job["actions"], job["horizon"])
            qs = build_questions(r["catalog"], job["actions"], job["horizon"])
            try:
                out = jev.ask(st, qs, tag=job["cond"])
            except Exception as e:
                with lock:
                    failures.append((job["cond"], job["idx"], repr(e)[:120]))
                return
            row = {"cond": job["cond"], "idx": job["idx"], "root_id": r["root_id"],
                   "world_id": r["world_id"], "seq_name": r["seq_name"],
                   "horizon": job["horizon"], "actions": job["actions"], "answers": {}}
            for qid, a in out["answers"].items():
                # For h0/noact the correct target is the ROOT state, not the endpoint.
                gt = r["root_labels"][qid] if job["cond"] in ("h0", "noact") else r["labels"][qid]
                row["answers"][qid] = {
                    "pred": a["choice"], "probs": a["probabilities"], "gt": gt,
                    "changed": (r["changed"][qid] if job["cond"] not in ("h0", "noact") else False),
                }
            with lock:
                results.append(row)
                if len(results) % 50 == 0:
                    print(f"  ... {len(results)}/{len(todo)}")

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            list(pool.map(run, todo))
        calls, toks = jev.calls, jev.input_tokens

    (OUT / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results))
    print(f"\ncalls={calls}  input_tokens={toks:,}  cost=${toks/1e6*0.042:.4f}  "
          f"failures={len(failures)}")
    for f in failures[:5]:
        print("  FAIL", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
