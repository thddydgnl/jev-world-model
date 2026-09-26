"""Multi-seed analysis with world-clustered bootstrap (설계.md §22.2).

Three things the point estimates we have been reading cannot tell us:

  1. how much of an arm gap survives resampling WORLDS, which is the
     independent unit — episodes from one world share topology and names;
  2. how much moves when only the policy seed changes, which today showed a
     6.2pp swing between otherwise identical runs;
  3. whether a paired difference is distinguishable from zero at all.

Episodes are paired within (seed, world, root) so every arm sees the same
situation, and the bootstrap resamples worlds, never episodes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# result dir -> policy seed used for that run
DEFAULT_SOURCES = {
    "artifacts/main_A": 20260921,
    "artifacts/main_VO": 20260921,
    "artifacts/ablation": 20260921,
    "artifacts/seed777": 777,
    "artifacts/seed1234": 1234,
}
B = 4000
ORDER = ["C", "C_fm", "validity", "Aval", "Aend", "A", "R",
         "A_jev", "B0_typed", "D0_gen", "B_typed", "D_gen", "oracle"]


def load(sources: dict[str, int]) -> list[dict]:
    eps = []
    for d, seed in sources.items():
        p = Path(d) / "episodes.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            # Runs since the KIIS runner record their own seed; the dir=seed
            # mapping is only needed for older directories.
            e["seed"] = e.get("policy_seed", seed)
            eps.append(e)
    return eps


def by_world(eps: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for e in eps:
        out[e["world_id"]].append(e)
    return out


def rate(eps: list[dict], arm: str) -> float | None:
    v = [e for e in eps if e["arm"] == arm]
    return sum(e["success"] for e in v) / len(v) if v else None


def boot_ci(worlds: dict[str, list[dict]], stat, rng: random.Random,
            b: int = B) -> tuple[float, float, float] | None:
    keys = list(worlds)
    point = stat([e for k in keys for e in worlds[k]])
    if point is None:
        return None
    draws = []
    for _ in range(b):
        pick = [rng.choice(keys) for _ in keys]
        sample = [e for k in pick for e in worlds[k]]
        s = stat(sample)
        if s is not None:
            draws.append(s)
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws))]
    return point, lo, hi


def paired_diff(eps: list[dict], a: str, b: str) -> float | None:
    """Difference over situations where BOTH arms ran."""
    idx = defaultdict(dict)
    for e in eps:
        idx[(e["seed"], e["world_id"], e["root"])][e["arm"]] = e["success"]
    both = [v for v in idx.values() if a in v and b in v]
    if not both:
        return None
    return sum(v[a] for v in both) / len(both) - sum(v[b] for v in both) / len(both)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=None,
                    help="dir=seed pairs, comma separated; default is the known runs")
    ap.add_argument("--allow-mixed-config", action="store_true",
                    help="pool runs whose condition hashes differ (never for a results table)")
    args = ap.parse_args()
    sources = DEFAULT_SOURCES
    if args.sources:
        sources = {}
        for part in args.sources.split(","):
            d, _, s = part.partition("=")
            sources[d] = int(s)

    eps = load(sources)
    if not eps:
        print("결과 디렉터리가 없습니다.")
        return 1
    # Runs made before the condition hash existed carry none ("-"). Pooling
    # them with hashed runs, or pooling two hashes, is how numbers from
    # different conditions ended up in one table before (docs/protocol.md A.4).
    hashes = sorted({e.get("config_hash") or "-" for e in eps})
    print(f"조건 해시: {', '.join(hashes)}")
    if len(hashes) > 1 and not args.allow_mixed_config:
        print("조건 해시가 서로 다른 실행을 합칠 수 없습니다. "
              "조건별로 나눠 분석하거나 --allow-mixed-config 를 명시하세요.")
        return 2
    # Same rule per arm: an arm fixed mid-way (e.g. the K1 failure-memory fix
    # for validity/oracle) must not be pooled with its own earlier runs.
    per_arm = defaultdict(set)
    for e in eps:
        per_arm[e["arm"]].add(e.get("arm_hash") or "-")
    mixed = {a: sorted(h) for a, h in per_arm.items() if len(h) > 1}
    if mixed and not args.allow_mixed_config:
        print(f"같은 arm 안에서 arm 해시가 섞여 있습니다: {mixed}")
        return 2
    rng = random.Random(20260921)
    arms = [a for a in ORDER if any(e["arm"] == a for e in eps)]
    seeds = sorted({e["seed"] for e in eps})
    worlds = by_world(eps)
    print(f"에피소드 {len(eps)}  arms={arms}  seeds={seeds}  worlds={len(worlds)}\n")

    # ---------------------------------------------------- per seed
    print("=" * 72)
    print("arm별 성공률 — seed 별 / 전체 (world-clustered 95% CI)")
    print("=" * 72)
    hdr = "  {:<10}".format("arm") + "".join(f"{('seed ' + str(s)):>12}" for s in seeds)
    print(hdr + f"{'pooled [95% CI]':>26}")
    for a in arms:
        row = f"  {a:<10}"
        for s in seeds:
            r = rate([e for e in eps if e["seed"] == s], a)
            row += f"{'—':>12}" if r is None else f"{r:>12.1%}"
        ci = boot_ci(worlds, lambda v, a=a: rate(v, a), rng)
        row += (f"{ci[0]:>13.1%} [{ci[1]:.0%},{ci[2]:.0%}]" if ci else f"{'—':>26}")
        print(row)

    # seed spread
    print("\n  seed 간 변동폭 (같은 arm, 정책 샘플링만 다름)")
    for a in arms:
        rs = [rate([e for e in eps if e["seed"] == s], a) for s in seeds]
        rs = [r for r in rs if r is not None]
        if len(rs) > 1:
            print(f"    {a:<10} {min(rs):.1%} ~ {max(rs):.1%}   폭 {max(rs)-min(rs):+.1%}p")

    # ---------------------------------------------------- paired gaps
    print("\n" + "=" * 72)
    print("짝지은 차이 (같은 seed·world·root 에서 둘 다 실행된 경우만)")
    print("=" * 72)
    pairs = [("A", "validity", "arm A − 완벽한 유효성 필터"),
             ("A", "oracle", "arm A − 완벽한 예측기"),
             ("oracle", "validity", "예측 구간 (상한)"),
             ("A", "Aend", "질문 분리의 기여"),
             ("A", "Aval", "endpoint 질문의 기여"),
             ("Aval", "validity", "JEV 유효성 오차의 비용"),
             ("A", "C", "arm A − No-WM"),
             # KIIS 2026 fall (kiis2026f/실험계획.md §5)
             ("A_jev", "C", "JEV 세계모델 − 기본 에이전트"),
             ("B0_typed", "C", "타입화 Qwen − 기본 에이전트"),
             ("D0_gen", "C", "생성형 Qwen − 기본 에이전트"),
             ("B_typed", "C", "학습 타입화 − 기본 에이전트"),
             ("D_gen", "C", "학습 생성형 − 기본 에이전트"),
             ("B0_typed", "D0_gen", "타입화 − 생성형 (학습 없음)"),
             ("B_typed", "D_gen", "타입화 − 생성형 (학습)"),
             ("B_typed", "A_jev", "학습 타입화 LLM − JEV"),
             ("A_jev", "B0_typed", "JEV − 학습 없는 Qwen (타입화)"),
             ("A_jev", "validity", "JEV − 완벽한 유효성 필터"),
             ("A_jev", "oracle", "JEV − 완벽한 예측기"),
             # V2: the no-world-model baseline with failure memory (C_fm)
             ("C_fm", "C", "실패 기억 − 기본 에이전트"),
             ("A_jev", "C_fm", "JEV 세계모델 − 실패 기억 기본"),
             ("B0_typed", "C_fm", "타입화 Qwen − 실패 기억 기본"),
             ("D0_gen", "C_fm", "생성형 Qwen − 실패 기억 기본"),
             ("B_typed", "C_fm", "학습 타입화 − 실패 기억 기본"),
             ("D_gen", "C_fm", "학습 생성형 − 실패 기억 기본"),
             ("validity", "C_fm", "완벽한 유효성 필터 − 실패 기억 기본")]
    for hi, lo, label in pairs:
        if hi not in arms or lo not in arms:
            continue
        ci = boot_ci(worlds, lambda v, h=hi, l=lo: paired_diff(v, h, l), rng)
        if ci is None:
            continue
        sig = "" if ci[1] <= 0 <= ci[2] else "  *"
        print(f"  {label:<30} {ci[0]:>+7.1%}  [{ci[1]:>+6.1%}, {ci[2]:>+6.1%}]{sig}")
    print("\n  * = 95% CI 가 0 을 포함하지 않음")

    # ---------------------------------------------------- discordance
    print("\n" + "=" * 72)
    print("불일치 (한쪽만 성공한 에피소드 수)")
    print("=" * 72)
    idx = defaultdict(dict)
    for e in eps:
        idx[(e["seed"], e["world_id"], e["root"])][e["arm"]] = e["success"]
    for hi, lo, label in pairs:
        if hi not in arms or lo not in arms:
            continue
        both = [v for v in idx.values() if hi in v and lo in v]
        h_only = sum(1 for v in both if v[hi] and not v[lo])
        l_only = sum(1 for v in both if v[lo] and not v[hi])
        print(f"  {label:<30} {hi} 만 {h_only:3d} / {lo} 만 {l_only:3d}  (n={len(both)})")

    # ---------------------------------------------------- efficiency
    print("\n" + "=" * 72)
    print("보조 지표 (평균)")
    print("=" * 72)
    print(f"  {'arm':<10}{'steps':>9}{'invalid':>10}{'progress':>10}{'n':>7}")
    for a in arms:
        v = [e for e in eps if e["arm"] == a]
        st = sum(e["steps"] for e in v)
        print(f"  {a:<10}{st/len(v):>9.1f}"
              f"{sum(e['invalid'] for e in v)/max(st,1):>10.1%}"
              f"{sum(e['progress'] for e in v)/len(v):>10.1%}{len(v):>7}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
