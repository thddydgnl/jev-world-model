"""K4 closed-loop verdicts for the pre-registered expectations
(kiis2026f/실험계획.md §5: H1, H1b, H2, H3, H4, H5) and every arm against the
oracle, with analyze_arms.py's world-clustered paired bootstrap.

    python scripts/kiis_k4_hypotheses.py --logs artifacts/kiis_k4v3 \
        --out artifacts/kiis_k4v3/hypotheses.json

Each test is a contrast over the situations (seed, world, root) where every arm
in it ran: the sum of coef * success. H3 is the difference in differences
(B - D) - (B0 - D0). Verdicts follow §5: a CI that contains 0 is "not
distinguishable", never "equal". Each contrast gets its own RNG seeded the same
way, so a verdict does not depend on which other contrasts were computed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_arms as aa

SEEDS = (20260921, 777, 1234)
UNITS = ("core_a", "core_b", "zero_shot", "generative", "typed")
RNG_SEED = 20260921

# (name, coefficients, direction the expectation predicts: +1 CI above 0, -1 below 0, 0 none)
TESTS = [
    ("H1  A - C", {"A_jev": 1, "C": -1}, 1),
    ("H1  B - C", {"B_typed": 1, "C": -1}, 1),
    ("H1  D - C", {"D_gen": 1, "C": -1}, 1),
    ("H1b A - C_fm", {"A_jev": 1, "C_fm": -1}, 1),
    ("H1b B - C_fm", {"B_typed": 1, "C_fm": -1}, 1),
    ("H1b D - C_fm", {"D_gen": 1, "C_fm": -1}, 1),
    ("H2  B0 - D0", {"B0_typed": 1, "D0_gen": -1}, 1),
    ("H3  (B - D) - (B0 - D0)", {"B_typed": 1, "D_gen": -1, "B0_typed": -1, "D0_gen": 1}, -1),
    ("H4  B - A", {"B_typed": 1, "A_jev": -1}, 1),
    ("H5  A - B0", {"A_jev": 1, "B0_typed": -1}, 1),
    ("    B - D", {"B_typed": 1, "D_gen": -1}, 0),
    ("    B - B0", {"B_typed": 1, "B0_typed": -1}, 0),
    ("    D - D0", {"D_gen": 1, "D0_gen": -1}, 0),
    ("    B0 - C_fm", {"B0_typed": 1, "C_fm": -1}, 0),
    ("    D0 - C_fm", {"D0_gen": 1, "C_fm": -1}, 0),
    ("    oracle - validity", {"oracle": 1, "validity": -1}, 0),
] + [(f"    {a} - oracle", {a: 1, "oracle": -1}, 0)
     for a in ("A_jev", "B_typed", "D_gen", "B0_typed", "D0_gen", "validity", "C_fm")]


def contrast_by_world(eps: list[dict], coef: dict[str, int]) -> tuple[dict, int, int]:
    """Per world, the contrast in each situation where all its arms ran; also
    how many situations came out positive and negative."""
    idx = defaultdict(dict)
    for e in eps:
        idx[(e["seed"], e["world_id"], e["root"])][e["arm"]] = int(e["success"])
    out, pos, neg = defaultdict(list), 0, 0
    for (_seed, world, _root), v in idx.items():
        if all(a in v for a in coef):
            x = sum(c * v[a] for a, c in coef.items())
            out[world].append(x)
            pos += x > 0
            neg += x < 0
    return out, pos, neg


def verdict(lo: float, hi: float, direction: int) -> str:
    if lo > 0:
        side = "above 0"
    elif hi < 0:
        side = "below 0"
    else:
        return "not distinguishable"
    if direction == 0:
        return side
    return "holds" if (side == "above 0") == (direction > 0) else "opposite"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="artifacts/kiis_k4v3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sources = {str(Path(args.logs) / str(s) / u): s for s in SEEDS for u in UNITS}
    eps = aa.load(sources)
    hashes = sorted({e.get("config_hash") or "-" for e in eps})
    per_arm = defaultdict(set)
    for e in eps:
        per_arm[e["arm"]].add(e.get("arm_hash") or "-")
    mixed = {a: sorted(h) for a, h in per_arm.items() if len(h) > 1}
    if len(hashes) != 1 or mixed:
        raise SystemExit(f"refusing to pool: condition hashes {hashes}, mixed arm hashes {mixed}")
    worlds = aa.by_world(eps)
    res = {"config_hash": hashes[0], "episodes": len(eps), "worlds": len(worlds),
           "arms": {}, "tests": {}}
    print(f"condition {hashes[0]}  episodes {len(eps)}  worlds {len(worlds)}\n")
    for arm in aa.ORDER:
        n = sum(e["arm"] == arm for e in eps)
        if not n:
            continue
        point, lo, hi = aa.boot_ci(worlds, lambda v, a=arm: aa.rate(v, a), random.Random(RNG_SEED))
        per_seed = {s: sum(e["success"] for e in eps if e["arm"] == arm and e["seed"] == s)
                    for s in SEEDS}
        res["arms"][arm] = {"n": n, "success": point, "ci": [lo, hi], "per_seed": per_seed}
        print(f"  {arm:<10} n={n:3d}  {point:6.1%} [{lo:5.1%}, {hi:5.1%}]  per seed {per_seed}")
    print()
    for name, coef, direction in TESTS:
        if not all(a in res["arms"] for a in coef):
            continue
        pw, pos, neg = contrast_by_world(eps, coef)
        point, lo, hi = aa.boot_paired(pw, random.Random(RNG_SEED))
        n = sum(len(v) for v in pw.values())
        v = verdict(lo, hi, direction)
        res["tests"][name.strip()] = {"coef": coef, "n": n, "diff": point, "ci": [lo, hi],
                                      "positive": pos, "negative": neg, "verdict": v}
        print(f"  {name:<26} {point:+7.1%} [{lo:+6.1%}, {hi:+6.1%}]  n={n}  (+{pos} / -{neg})  {v}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
