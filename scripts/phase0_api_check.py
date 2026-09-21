"""Phase 0: confirm API access, pinned model, and response schema."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_client import JevClient, choice, PINNED_MODEL

def main() -> int:
    print(f"endpoint  : api.typesafe.ai/v1/systemone")
    print(f"model     : {PINNED_MODEL} (pinned)\n")
    with JevClient(save_raw=True) as jev:
        # Deliberately trivial: if this is wrong, the problem is access, not ability.
        t0 = time.time()
        out = jev.ask(
            state={"box": "closed", "key": "inside the box", "player": "holding nothing"},
            questions={
                "q_trivial": choice(
                    "Is the box currently open or closed? Report the CURRENT state.",
                    {"open": "The box is open.", "closed": "The box is closed."},
                ),
            },
            tag="phase0",
        )
        dt = (time.time() - t0) * 1000

    ans = out["answers"]["q_trivial"]
    print(f"returned model : {out['model']}")
    print(f"latency        : {dt:.0f} ms")
    print(f"usage          : {out.get('usage')}")
    print(f"choice         : {ans['choice']}")
    print(f"probabilities  : {ans['probabilities']}")
    print(f"confidence     : {ans.get('confidence')}")

    ok = ans["choice"] == "closed"
    print(f"\n{'PASS' if ok else 'FAIL'}: schema validated, "
          f"{'trivial answer correct' if ok else 'trivial answer WRONG -> check serialization'}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
