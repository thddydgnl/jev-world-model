"""Manual audit: cross-check the labeler against TextWorld's own human-readable
facts (an independent derivation), then eyeball the order-swap design claim."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

recs = [json.loads(l) for l in Path("data/mvp_a/prefixes.jsonl").read_text().splitlines()]

# --- 1. cross-check: derive key location from dynamic_facts independently
def parent_from_facts(facts, target):
    for row in facts:
        if row[0] in ("in", "on", "at") and len(row) == 3 and row[1] == target:
            return "inventory" if row[2] == "I" else f"{row[0]}:{row[2]}"
    return "unknown"

mismatch = 0
for r in recs:
    # root_labels must match an independent read of the root state
    kid = next(q for q in r["catalog"] if q["id"] == "q_key_parent")
    tgt = [e["id"] for e in r["current_state"]["entities"]
           if e["name"] == r["meta"]["key"]]
    if not tgt: continue
    indep = parent_from_facts(r["current_state"]["dynamic_facts"], tgt[0])
    if indep != r["root_labels"]["q_key_parent"]:
        mismatch += 1
print(f"[audit] root label 독립 교차검증 불일치: {mismatch}/{len(recs)}  "
      f"({'PASS' if mismatch == 0 else 'FAIL'})")

# --- 2. design claim: order-swap differs iff the box starts closed
by_root = {}
for r in recs:
    by_root.setdefault((r["root_id"], r["horizon"]), {})[r["seq_name"]] = r

closed_diff = closed_same = open_diff = open_same = 0
for (root, h), seqs in by_root.items():
    if h != 2 or "open_then_take" not in seqs or "take_then_open" not in seqs: continue
    a, b = seqs["open_then_take"], seqs["take_then_open"]
    box_start = a["root_labels"]["q_box_mode"]
    differs = a["labels"]["q_key_parent"] != b["labels"]["q_key_parent"]
    if box_start == "open":   open_diff += differs;   open_same += (not differs)
    else:                     closed_diff += differs; closed_same += (not differs)

print(f"\n[audit] 순서 교환 pair (open→take vs take→open), h=2, q_key_parent")
print(f"  상자가 닫힌 root : 결과 다름 {closed_diff} / 같음 {closed_same}   <- 판별 pair여야 함")
print(f"  상자가 열린 root : 결과 다름 {open_diff} / 같음 {open_same}   <- null pair여야 함")
design_ok = closed_diff > 0 and open_same > 0 and open_diff == 0
print(f"  설계 의도 충족: {'PASS' if design_ok else 'FAIL'}")

# --- 3. eyeball three concrete cases
print("\n[audit] 실제 사례 3건")
shown = 0
for (root, h), seqs in by_root.items():
    if h != 2 or shown >= 3: continue
    if "open_then_take" not in seqs or "take_then_open" not in seqs: continue
    a, b = seqs["open_then_take"], seqs["take_then_open"]
    if a["labels"]["q_key_parent"] == b["labels"]["q_key_parent"]: continue
    print(f"\n  root={root}  상자 초기={a['root_labels']['q_box_mode']}  "
          f"열쇠 초기={a['root_labels']['q_key_parent']}")
    for r in (a, b):
        print(f"    {r['actions']}")
        print(f"      -> key={r['labels']['q_key_parent']}  box={r['labels']['q_box_mode']}")
    shown += 1

print(f"\n[audit] 정답 분포 (q_key_parent)")
for v, c in Counter(r["labels"]["q_key_parent"] for r in recs).most_common():
    print(f"  {v:16s} {c:4d}")
