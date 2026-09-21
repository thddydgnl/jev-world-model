"""MVP-B analysis and gate decision (설계.md §21, §24, §25)."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PRED = Path("artifacts/mvp_b/predictions.jsonl")
DATA = Path("data/mvp_a/prefixes.jsonl")
EPS = 1e-6
QIDS = ["q_key_parent", "q_box_mode", "q_door_mode", "q_distractor_parent", "q_player_room"]


def nll(p: float) -> float:
    return -math.log(max(p, EPS))


def brier(probs: dict, gt: str) -> float:
    return sum((v - (1.0 if k == gt else 0.0)) ** 2 for k, v in probs.items())


def bar(x: float, w: int = 22) -> str:
    n = int(round(x * w))
    return "█" * n + "·" * (w - n)


def main() -> int:
    rows = [json.loads(l) for l in PRED.read_text().splitlines()]
    recs = [json.loads(l) for l in DATA.read_text().splitlines()]
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["cond"]].append(r)

    # ---------------------------------------------------------- gate 1: h0
    print("=" * 66)
    print("GATE 1 — 0-step 현재 상태 추출 (JEV 능력이 아니라 '내 입력'을 검사)")
    print("=" * 66)
    h0 = [(qid, a) for r in by_cond["h0"] for qid, a in r["answers"].items()]
    h0_acc = sum(a["pred"] == a["gt"] for _, a in h0) / len(h0)
    print(f"  전체 정확도  {h0_acc:6.1%}  {bar(h0_acc)}   n={len(h0)}  (목표 >95%)")
    per_q = defaultdict(list)
    for qid, a in h0:
        per_q[qid].append(a["pred"] == a["gt"])
    for qid in QIDS:
        v = per_q[qid]
        print(f"    {qid:22s} {sum(v)/len(v):6.1%}  n={len(v)}")
    gate1 = h0_acc > 0.95
    print(f"  -> {'PASS' if gate1 else 'FAIL — 아래 결과는 해석 불가'}")

    # ------------------------------------------- gate 2: forecast vs persistence
    print()
    print("=" * 66)
    print("GATE 2 — 행동 이후 예측 vs persistence (변화한 사실만이 핵심)")
    print("=" * 66)
    print(f"  {'조건':<7}{'구분':<10}{'n':>6}{'JEV':>9}{'persist':>10}{'차이':>9}{'NLL':>8}{'Brier':>8}")
    gate2_margin = None
    for cond in ("h1", "h2"):
        strata = {"changed": [], "unchanged": [], "all": []}
        for r in by_cond[cond]:
            rec = recs[r["idx"]]
            for qid, a in r["answers"].items():
                item = (a["pred"] == a["gt"], rec["root_labels"][qid] == a["gt"],
                        nll(a["probs"].get(a["gt"], 0.0)), brier(a["probs"], a["gt"]))
                strata["all"].append(item)
                strata["changed" if a["changed"] else "unchanged"].append(item)
        for name in ("all", "changed", "unchanged"):
            v = strata[name]
            if not v:
                continue
            acc = sum(x[0] for x in v) / len(v)
            per = sum(x[1] for x in v) / len(v)
            m_nll = sum(x[2] for x in v) / len(v)
            m_br = sum(x[3] for x in v) / len(v)
            print(f"  {cond:<7}{name:<10}{len(v):>6}{acc:>9.1%}{per:>10.1%}"
                  f"{acc-per:>+9.1%}{m_nll:>8.3f}{m_br:>8.3f}")
            if name == "changed":
                gate2_margin = acc - per if gate2_margin is None else min(gate2_margin, acc - per)
        print()
    gate2 = gate2_margin is not None and gate2_margin >= 0.10
    print(f"  -> changed-fact에서 persistence 대비 최소 마진 {gate2_margin:+.1%}  "
          f"({'PASS' if gate2 else 'FAIL'}, 기준 +10%p)")

    # ------------------------------------------------- gate 3: contrastive pairs
    print()
    print("=" * 66)
    print("GATE 3 — 반사실 pair 판별력 (같은 root, 다른 행동열, 다른 정답)")
    print("=" * 66)
    by_key = defaultdict(dict)
    for r in rows:
        if r["cond"] not in ("h1", "h2"):
            continue
        for qid, a in r["answers"].items():
            by_key[(r["root_id"], r["horizon"], qid)][r["seq_name"]] = a
    pair_exact = pair_n = null_same = null_n = 0
    for entries in by_key.values():
        names = sorted(entries)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = entries[names[i]], entries[names[j]]
                if a["gt"] != b["gt"]:
                    pair_n += 1
                    pair_exact += (a["pred"] == a["gt"] and b["pred"] == b["gt"])
                else:
                    null_n += 1
                    null_same += (a["pred"] == b["pred"])
    print(f"  판별 pair (정답 다름)  pair-exact  {pair_exact/pair_n:6.1%}  "
          f"{bar(pair_exact/pair_n)}  n={pair_n}")
    print(f"  null  pair (정답 같음)  예측 일치  {null_same/null_n:6.1%}  "
          f"{bar(null_same/null_n)}  n={null_n}")
    gate3 = pair_exact / pair_n >= 0.50

    # order swap specifically
    print("\n  순서 교환만 따로 (open→take vs take→open, q_key_parent):")
    ok = tot = 0
    for (root, h, qid), e in by_key.items():
        if qid != "q_key_parent" or h != 2:
            continue
        if "open_then_take" not in e or "take_then_open" not in e:
            continue
        a, b = e["open_then_take"], e["take_then_open"]
        if a["gt"] == b["gt"]:
            continue
        tot += 1
        ok += (a["pred"] == a["gt"] and b["pred"] == b["gt"])
    print(f"    둘 다 정답  {ok}/{tot}  ({ok/tot:.1%})" if tot else "    (해당 pair 없음)")
    print(f"  -> {'PASS' if gate3 else 'FAIL'} (기준 pair-exact ≥50%)")

    # ------------------------------------------------ gate 4: action-absent control
    print()
    print("=" * 66)
    print("GATE 4 — 행동 제거 대조 (AB1): 행동을 빼면 예측이 실제로 달라지는가")
    print("=" * 66)
    noact = {}
    for r in by_cond["noact"]:
        for qid, a in r["answers"].items():
            noact[(r["root_id"], qid)] = a
    should_move = did_move = stay = false_move = 0
    for r in by_cond["h2"]:
        for qid, a in r["answers"].items():
            base = noact.get((r["root_id"], qid))
            if base is None:
                continue
            if a["changed"]:
                should_move += 1
                did_move += (a["pred"] != base["pred"])
            else:
                stay += 1
                false_move += (a["pred"] != base["pred"])
    print(f"  정답이 바뀌어야 할 때  예측도 바뀜   {did_move}/{should_move}  "
          f"({did_move/should_move:.1%})  {bar(did_move/should_move)}")
    print(f"  정답이 그대로일 때    예측이 바뀜   {false_move}/{stay}  "
          f"({false_move/stay:.1%})  {bar(false_move/stay)}  <- 낮을수록 좋음")
    gate4 = (did_move / should_move) - (false_move / stay) >= 0.30
    print(f"  -> 순 반응성 {did_move/should_move - false_move/stay:+.1%}  "
          f"({'PASS' if gate4 else 'FAIL'}, 기준 +30%p)")

    # ---------------------------------------------------------------- verdict
    print()
    print("=" * 66)
    gates = [("0-step 입력 가독성", gate1), ("changed-fact > persistence", gate2),
             ("반사실 pair 판별", gate3), ("행동 조건부 반응", gate4)]
    for name, g in gates:
        print(f"  [{'PASS' if g else 'FAIL'}] {name}")
    allp = all(g for _, g in gates)
    print("=" * 66)
    print(f"MVP-B {'PASS — 본 실험 진행 근거 있음' if allp else 'FAIL — 설계 재검토 필요'}")
    return 0 if allp else 1


if __name__ == "__main__":
    raise SystemExit(main())
