"""Phase 0 smoke test (설계.md §3.4, §24).

Three checks, each of which fails SILENTLY in production if wrong:
  1. no-quest physics   -> wrong => every episode ends at step 0 (loud)
  2. clone isolation    -> wrong => labels silently corrupted (deadly)
  3. whitelist          -> wrong => answers leak, results look GOOD (deadly)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import textworld
from textworld import GameMaker, EnvInfos
from env.serialize import canonical_state, assert_clean, facts_hash, state_hash, FORBIDDEN_KEYS

INFOS = EnvInfos(facts=True, typed_entities=True, possible_admissible_commands=True,
                 last_action=True, admissible_commands=True)
results = []

def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))

def build(path):
    maker = GameMaker()
    room = maker.new_room("workroom")
    box = maker.new(type="c", name="wooden box"); box.add_property("closed")
    key = maker.new(type="k", name="brass key")
    room.add(box); box.add(key)
    maker.set_player(room)
    maker.quests = []          # goals live in our wrapper, not the engine
    game = maker.build()
    game.save(str(path))
    return game

def main():
    root = Path("data/smoke"); root.mkdir(parents=True, exist_ok=True)
    path = root / "physical_world.json"
    game = build(path)

    print("\n[1] no-quest physics")
    env = textworld.start(str(path), request_infos=INFOS)
    s0 = env.reset()
    check("reset 직후 미종료", not s0["won"] and not s0["lost"],
          f"won={s0['won']} lost={s0['lost']}")
    h_before = facts_hash(s0["_facts"])
    s1, score, done = env.step("open wooden box")
    h_after = facts_hash(s1["_facts"])
    check("행동 뒤 물리 상태 변화", h_before != h_after, f"{h_before} -> {h_after}")
    check("no-quest 게임이 자동 종료되지 않음", not done, f"done={done}, score={score}")
    invalid_before = facts_hash(s1["_facts"])
    s2, _, _ = env.step("frobnicate the gizmo")
    check("파서 실패는 no-op", facts_hash(s2["_facts"]) == invalid_before,
          f"feedback={s2.feedback!r}")

    print("\n[2] clone isolation")
    env2 = textworld.start(str(path), request_infos=INFOS)
    root_state = env2.reset()
    root_hash = facts_hash(root_state["_facts"])
    branch = env2.copy()
    branch.step("open wooden box")
    branch.step("take brass key from wooden box")
    after_branch = facts_hash(env2.state["_facts"])
    check("branch 실행 후 원본 불변", root_hash == after_branch,
          f"원본 {root_hash} / branch 후 원본 {after_branch}")
    branch_end = facts_hash(branch.state["_facts"])
    check("branch는 실제로 전진함", branch_end != root_hash, f"branch {branch_end}")
    # replay determinism: a fresh clone from the same root must land identically
    branch2 = env2.copy()
    branch2.step("open wooden box")
    branch2.step("take brass key from wooden box")
    check("동일 prefix replay 결정론", facts_hash(branch2.state["_facts"]) == branch_end)
    branch.close(); branch2.close()

    print("\n[3] whitelist serializer")
    st = canonical_state(list(root_state["_facts"]), game, attempt_index=0)
    try:
        assert_clean(st); clean = True; why = ""
    except ValueError as e:
        clean = False; why = str(e)
    check("금지 키/마커 0건", clean, why)
    present = FORBIDDEN_KEYS & set(root_state.keys())
    check("원시 GameState에는 금지 키가 실제로 존재 (테스트가 유효함)", len(present) > 0,
          f"{len(present)}개: {sorted(present)[:4]}...")
    check("canonical state는 평범한 JSON", isinstance(st, dict) and "dynamic_facts" in st,
          f"keys={sorted(st)}")
    print(f"        state_hash={state_hash(st)}  facts={st['dynamic_facts']}")

    print("\n[4] state sufficiency (§5.3)")
    seen = {}
    collision = False
    for cmds in (["open wooden box"], ["open wooden box"], ["examine wooden box"]):
        b = env2.copy()
        for c in cmds: b.step(c)
        key_ = (state_hash(canonical_state(list(env2.state["_facts"]), game)), tuple(cmds))
        end = facts_hash(b.state["_facts"])
        if key_ in seen and seen[key_] != end: collision = True
        seen[key_] = end
        b.close()
    check("동일 state+action -> 동일 endpoint", not collision)

    env.close(); env2.close()

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{'='*58}\n{len(results)-n_fail}/{len(results)} passed")
    return 1 if n_fail else 0

if __name__ == "__main__":
    raise SystemExit(main())
