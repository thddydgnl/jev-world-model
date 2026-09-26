"""Arm C_fm (kiis2026f/실험계획.md §4 V2) is C plus failure memory, nothing more:
with no failures it is exactly C (the top plan's first action); a first action
that already failed from the same state is skipped for the next plan in the
policy's order; a failure in another state does not count; if every first
action has failed, it falls back to the top plan, as the planner arms do."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mvp_c_headroom import c_fm_prefix

S1 = {"dynamic_facts": [["at", "P", "r_0"], ["closed", "c_0"]]}
S2 = {"dynamic_facts": [["at", "P", "r_0"], ["open", "c_0"]]}
FP1 = "|".join(",".join(r) for r in S1["dynamic_facts"])
PLANS = [["go east", "open box"], ["open box", "take key"], ["go east"], ["take key"]]


def main() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    check(c_fm_prefix(PLANS, S1, set()) == ("go east",), "no failures: same as C")
    check(c_fm_prefix(PLANS, S1, {(FP1, "go east")}) == ("open box",),
          "failed first action skipped, next plan in policy order")
    check(c_fm_prefix(PLANS, S2, {(FP1, "go east")}) == ("go east",),
          "a failure in another state does not count")
    check(c_fm_prefix(PLANS, S1, {(FP1, a) for a in ("go east", "open box", "take key")})
          == ("go east",), "all failed: top plan, as the planner arms do")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
