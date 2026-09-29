"""analyze_arms.py paired bootstrap (9/30): a world drawn twice counts twice.

The old paired statistic re-paired a resampled episode list by (seed, world,
root), so duplicated worlds collapsed into one copy and the intervals were too
narrow. Checked here on small hand-made data, no files.
"""
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_arms as aa


def ep(world, root, arm, success, seed=1):
    return {"seed": seed, "world_id": world, "root": root, "arm": arm, "success": success}


def main() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    # pairing: only situations where both arms ran, grouped by world
    eps = [ep("w1", 0, "a", True), ep("w1", 0, "b", False),
           ep("w1", 1, "a", True),                               # b missing: not paired
           ep("w2", 0, "a", False), ep("w2", 0, "b", False),
           ep("w2", 0, "a", True, seed=2), ep("w2", 0, "b", False, seed=2)]
    pw = aa.paired_by_world(eps, "a", "b")
    check(dict(pw) == {"w1": [1], "w2": [0, 1]}, f"paired_by_world {dict(pw)}")

    # multiplicity: with a scripted draw sequence the mean must weight a world
    # by how often it was drawn
    class Scripted(random.Random):
        def __init__(self, seq):
            super().__init__(0)
            self.seq = list(seq)

        def choice(self, keys):
            return self.seq.pop(0)

    per_world = {"A": [1], "B": [0, 0, 0], "C": [0]}
    draws = ["A", "A", "B"] * 4          # four identical draws of (A, A, B)
    point, lo, hi = aa.boot_paired(per_world, Scripted(draws), b=4)
    check(abs(point - 0.2) < 1e-12, f"point estimate {point}")
    check(abs(lo - 0.4) < 1e-12 and abs(hi - 0.4) < 1e-12,
          f"(A, A, B) gives (1+1+0)/(1+1+3) = 0.4, got [{lo}, {hi}]")

    # against a direct cluster bootstrap on random data
    rng = random.Random(3)
    per_world = {f"w{i}": [rng.choice((-1, 0, 0, 1)) for _ in range(rng.randint(2, 8))]
                 for i in range(24)}
    _, lo, hi = aa.boot_paired(per_world, random.Random(5), b=2000)
    keys = sorted(per_world)
    r = random.Random(5)
    ref = []
    for _ in range(2000):
        s = [x for k in (r.choice(keys) for _ in keys) for x in per_world[k]]
        ref.append(sum(s) / len(s))
    ref.sort()
    check(abs(lo - ref[50]) < 1e-12 and abs(hi - ref[1950]) < 1e-12,
          f"matches a direct cluster bootstrap [{lo:.4f}, {hi:.4f}] vs [{ref[50]:.4f}, {ref[1950]:.4f}]")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
