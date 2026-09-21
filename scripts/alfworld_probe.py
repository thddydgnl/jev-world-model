"""Ask ALFWorld the question that disqualified TextWorld: does action validity
dominate, or does choosing among valid actions actually require foresight?

No policy and no GPU. ALFWorld ships a PDDL planner, so `policy_commands` gives
the optimal plan from ANY state and its length is the true distance to the goal.
For every admissible action we branch once and classify the result:

  progress   d decreases   — the action advances the goal
  detour     d increases   — valid, reversible, wasted moves
  dead       no plan       — valid and IRREVERSIBLE: the goal is now unreachable

A world with many `detour` and any `dead` actions is one where a validity filter
cannot substitute for a forecast. That is exactly what the TextWorld suite
lacked.
"""
from __future__ import annotations

import argparse
import glob
import time
from collections import Counter

import textworld
from textworld import EnvInfos
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler

INFOS = EnvInfos(facts=True, admissible_commands=True, won=True, policy_commands=True)
DATA = "/home/yonghwi/alfworld_data/json_2.1.1/train/**/game.tw-pddl"


def open_game(f):
    return textworld.start(f, request_infos=INFOS,
                           wrappers=[AlfredDemangler(shuffle=False)])


def distance(state):
    """Optimal remaining plan length, or None when the goal is unreachable."""
    if state["won"]:
        return 0
    pc = state.get("policy_commands")
    return len(pc) if pc else None


def replay(f, prefix):
    env = open_game(f)
    env.reset()
    for a in prefix:
        env.step(a)
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--states", type=int, default=4, help="states probed per game")
    args = ap.parse_args()

    files = sorted(glob.glob(DATA, recursive=True))
    print(f"게임 풀: {len(files)}개, 표본 {args.games}개\n")
    step = max(1, len(files) // args.games)
    sample = files[::step][:args.games]

    totals = Counter()
    per_state = []
    t_start = time.time()

    for gi, f in enumerate(sample):
        task = f.split("/")[-3]
        env = open_game(f)
        s = env.reset()
        d0 = distance(s)
        if d0 is None:
            print(f"[{gi}] {task[:48]}  계획 없음 — 건너뜀")
            env.close()
            continue
        print(f"[{gi}] {task[:48]}  최적 {d0} step, admissible {len(s['admissible_commands'])}")

        prefix = []
        for si in range(args.states):
            cur = replay(f, prefix)
            st = cur.state
            d = distance(st)
            adm = list(st["admissible_commands"])
            if d is None or d == 0 or not adm:
                cur.close()
                break

            counts = Counter()
            dead_actions = []
            for a in adm:
                br = replay(f, prefix + [a])
                d2 = distance(br.state)
                br.close()
                if d2 is None:
                    counts["dead"] += 1
                    dead_actions.append(a)
                elif d2 < d:
                    counts["progress"] += 1
                elif d2 == d:
                    counts["neutral"] += 1
                else:
                    counts["detour"] += 1
            n = len(adm)
            per_state.append((task, si, d, n, counts))
            totals.update(counts)
            totals["states"] += 1
            totals["actions"] += n
            print(f"     state{si} d={d:2d} n={n:3d} | "
                  f"progress {counts['progress']:3d} neutral {counts['neutral']:3d} "
                  f"detour {counts['detour']:3d} dead {counts['dead']:3d}"
                  + (f"  <-- {dead_actions[:2]}" if dead_actions else ""))

            nxt = st["policy_commands"][0]
            prefix.append(nxt)
            cur.close()
        env.close()

    print(f"\n{'='*70}")
    a = totals["actions"] or 1
    print(f"상태 {totals['states']}개 / 유효행동 {a}개 / {time.time()-t_start:.0f}초")
    print(f"  progress  {totals['progress']:5d}  ({totals['progress']/a:6.1%})  목표를 전진시킴")
    print(f"  neutral   {totals['neutral']:5d}  ({totals['neutral']/a:6.1%})  거리 불변")
    print(f"  detour    {totals['detour']:5d}  ({totals['detour']/a:6.1%})  유효하지만 멀어짐")
    print(f"  dead      {totals['dead']:5d}  ({totals['dead']/a:6.1%})  유효하지만 목표 파괴 (비가역)")
    useless = totals["neutral"] + totals["detour"] + totals["dead"]
    print(f"\n  유효하지만 전진 아님: {useless}/{a} = {useless/a:.1%}")
    print(f"  -> 무작위로 유효 행동을 고르면 전진 확률 {totals['progress']/a:.1%}")
    print(f"{'='*70}")
    if totals["dead"]:
        print("비가역 행동 존재 -> 유효성 필터로는 회피 불가. 예측이 필요함.")
    else:
        print("비가역 행동 없음 -> TextWorld 와 같은 약점 가능성.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
