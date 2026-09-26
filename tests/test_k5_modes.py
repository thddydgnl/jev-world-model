"""K5 scoring modes (kiis2026f/실험계획.md §4 V5, H8) checked without a model.

A table backend answers from the engine along each rollout's true trajectory:
  perfect      every answer right -> free-running and teacher-forced both 100%,
               accumulation G = 0, beam 1 identical to beam 4
  wrong at k=2 one wrong answer on the true state before step 2 -> teacher-forced
               wrong at k=2 only; free-running wrong from k=2 on (it then asks
               about states off the true path and answers "nothing happens"),
               so G > 0
  parse at k=2 the k=2 reply "does not parse" -> teacher-forced wrong at k=2
               only, free-running wrong from k=2 on (the §6.2 rule)
and `accumulation` turns such runs into G and a generative - typed difference.
"""
import json, sys, tempfile, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import kiis_k5_rollout as k5
import wm.recursive_forecaster as rf
from agent.task import read_schema_values, state_schema, trap_goal_spec
from env.serialize import canonical_state
from wm.recursive_forecaster import RecursiveForecaster, StepPrediction

ROLLOUTS = ROOT / "artifacts/kiis_k5v2_smoke/k5v2_rollouts_smoke.json"


class TableStep:
    """Engine truth for (true state, action) pairs; `spoil` picks keys to get wrong."""

    def __init__(self, table, schema, spoil=(), parse_fail=()):
        self.table, self.schema = table, schema
        self.spoil, self.parse_fail = set(spoil), set(parse_fail)

    def __call__(self, canon, action):
        key = (RecursiveForecaster._fingerprint(canon), action)
        now = read_schema_values(canon, self.schema)
        if key in self.parse_fail:
            return StepPrediction(now, {k: 1.0 for k in now}, 0.0, parse_failed=True)
        if key not in self.table:          # off the true path: "nothing happens"
            return StepPrediction(now, {k: 1.0 for k in now}, 0.0)
        values, runs = self.table[key]
        if key in self.spoil:              # wrong: claim the command fails
            return StepPrediction(now, {k: 1.0 for k in now}, 0.0 if runs else 1.0)
        return StepPrediction(values, {k: 1.0 for k in values}, 1.0 if runs else 0.0)

    def predict_batch(self, pairs):
        return [self(c, a) for c, a in pairs]


_KEYS: dict = {}


def _keys(r, only_k=None):
    """(state fingerprint, action) of each true step of a rollout, filled by main()."""
    return {key for k, key in _KEYS[r["id"]] if only_k is None or k == only_k}


def main() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + msg)
        ok &= bool(cond)

    data = json.loads(ROLLOUTS.read_text())
    rolls = data["rollouts"]
    with tempfile.TemporaryDirectory() as d:
        game, meta, env = next(k5.worlds(data["meta"]["world_set"], 1, Path(d)))
        schema = state_schema(game, meta)
        goal = trap_goal_spec(game, meta)
        table, k2_keys = {}, set()
        for r in rolls:
            s = k5.replay(env, r["path"])
            _KEYS[r["id"]] = []
            for k, a in enumerate(r["actions"], start=1):
                canon = canonical_state(list(s.state["_facts"]), game)
                runs = a in s.state["admissible_commands"]
                s.step(a)
                nxt = canonical_state(list(s.state["_facts"]), game)
                key = (RecursiveForecaster._fingerprint(canon), a)
                table[key] = (read_schema_values(nxt, schema), runs)
                _KEYS[r["id"]].append((k, key))
                if k == 2:
                    k2_keys.add(key)
            s.close()
        # a k=2 key must not also occur at another depth of some rollout, or the
        # expectations below would not be clean
        other = set()
        for r in rolls:
            s = k5.replay(env, r["path"])
            for k, a in enumerate(r["actions"], start=1):
                key = (RecursiveForecaster._fingerprint(canonical_state(list(s.state["_facts"]), game)), a)
                if k != 2:
                    other.add(key)
                s.step(a)
            s.close()
        k2_only = k2_keys - other
        clean = []                  # rollouts whose k=2 situation occurs nowhere else
        for r in rolls:
            s = k5.replay(env, r["path"])
            s.step(r["actions"][0])
            key = (RecursiveForecaster._fingerprint(canonical_state(list(s.state["_facts"]), game)),
                   r["actions"][1])
            s.close()
            if key in k2_only and sum(1 for x in rolls if x is not r and key in _keys(x)) == 0:
                clean.append(r)
        check(len(clean) >= 2, f"at least two rollouts with their own k=2 situation ({len(clean)}/{len(rolls)})")
        spoil = set()
        for r in clean:
            spoil |= {k for k in _keys(r, only_k=2)}

        def run(mode, step, beam=4):
            rf.BEAM_K = beam
            fc = RecursiveForecaster(None, schema, goal, step=step)
            walk = k5.teacher_forced_rows if mode == "tf" else k5.free_running_rows
            rows = walk(fc, env, game, schema, rolls)
            rf.BEAM_K = 4
            return {(r["id"], r["k"]): r for r in rows}

        perfect = TableStep(table, schema)
        fr, tf, fr1 = run("fr", perfect), run("tf", perfect), run("fr", perfect, beam=1)
        check(all(r["exact"] and r["exec_ok"] for r in fr.values()), "perfect: free-running all right")
        check(all(r["exact"] and r["exec_ok"] for r in tf.values()), "perfect: teacher-forced all right")
        check({k: r["exact"] for k, r in fr1.items()} == {k: r["exact"] for k, r in fr.items()},
              "perfect: beam 1 = beam 4")

        wrong = TableStep(table, schema, spoil=spoil)
        fr, tf = run("fr", wrong), run("tf", wrong)
        check(all(tf[(r["id"], k)]["exact"] == (k != 2) for r in clean for k in range(1, 5)),
              "wrong at k=2: teacher-forced wrong at k=2 only")
        check(all(fr[(r["id"], 1)]["exact"] for r in clean), "wrong at k=2: free-running right at k=1")
        check(all(not fr[(r["id"], 2)]["exact"] for r in clean), "wrong at k=2: free-running wrong at k=2")
        gap_later = sum(tf[(r["id"], k)]["exact"] - fr[(r["id"], k)]["exact"] for r in clean for k in (3, 4))
        check(gap_later > 0, f"wrong at k=2: free-running falls behind teacher-forced after k=2 ({gap_later} cells)")

        pf = TableStep(table, schema, parse_fail=spoil)
        fr, tf = run("fr", pf), run("tf", pf)
        check(all(tf[(r["id"], k)]["parse_failed"] == (k == 2) and tf[(r["id"], k)]["exact"] == (k != 2)
                  for r in clean for k in range(1, 5)), "parse failure at k=2: teacher-forced wrong at k=2 only")
        check(all(fr[(r["id"], k)]["parse_failed"] == (k >= 2) and fr[(r["id"], k)]["exact"] == (k < 2)
                  for r in clean for k in range(1, 5)), "parse failure at k=2: free-running wrong from k=2 on")

        # accumulation: typed perfect, generative wrong at k=2 -> G(gen) > G(typed) = 0
        out = Path(d) / "acc"
        out.mkdir()
        def dump(name, rows):
            (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows.values()) + "\n")
        dump("B0_typed", run("fr", perfect)); dump("B0_typed.tf", run("tf", perfect))
        dump("D0_gen", run("fr", wrong)); dump("D0_gen.tf", run("tf", wrong))   # wrong on `clean` at k=2
        k5.accumulation(types.SimpleNamespace(out=str(out)))
        res = json.loads((out / "accumulation_h8.json").read_text())
        check(res["models"]["B0_typed"]["G"] == 0 and res["models"]["B0_typed"]["gap_k1"] == 0,
              "accumulation: perfect typed G = 0, gap at k=1 = 0")
        check(res["models"]["D0_gen"]["G"] > 0 and res["models"]["D0_gen"]["gap_k1"] == 0,
              "accumulation: G > 0 for the model wrong at k=2, gap at k=1 = 0")
        check(res["pairs"]["D0_gen - B0_typed (all)"]["diff"] > 0, "accumulation: generative - typed > 0")
        env.close()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
