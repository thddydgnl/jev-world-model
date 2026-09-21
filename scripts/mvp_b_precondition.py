"""Is the order-swap failure a general precondition-blindness?

Re-replays every prefix and records how many of its commands the engine
actually REFUSED, then splits JEV accuracy by that count.
"""
import json, sys, random
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import textworld
from textworld import EnvInfos
from env.worlds import build_world, build_query_catalog
from env.serialize import canonical_state

sys.path.insert(0, "scripts")
from mvp_a_dataset import sequences, N_WORLDS, ROOTS_PER_WORLD, SEED, INFOS

rng = random.Random(SEED)
invalid_of = {}   # (root_id, seq_name, horizon) -> n_invalid
wdir = Path("data/mvp_a/worlds")
for w in range(N_WORLDS):
    game, path, meta = build_world(w, wdir, rng)
    env = textworld.start(str(path), request_infos=INFOS); env.reset()
    for r in range(ROOTS_PER_WORLD):
        root_env = env.copy()
        for _ in range(rng.randint(0, 3)):
            adm=[c for c in root_env.state["admissible_commands"]
                 if not c.startswith(("look","inventory","examine"))]
            if not adm: break
            root_env.step(rng.choice(adm))
        root_id = f"{meta['world_id']}_r{r}"
        for seq_name, actions in sequences(meta):
            for h in (1,2):
                b = root_env.copy(); n_inv = 0
                for a in actions[:h]:
                    if a not in b.state["admissible_commands"]: n_inv += 1
                    b.step(a)
                invalid_of[(root_id, seq_name, h)] = n_inv
                b.close()
        root_env.close()
    env.close()

rows=[json.loads(l) for l in Path("artifacts/mvp_b/predictions.jsonl").read_text().splitlines()]
buckets=defaultdict(lambda:[0,0])
for r in rows:
    if r["cond"] not in ("h1","h2"): continue
    k=(r["root_id"], r["seq_name"], r["horizon"])
    if k not in invalid_of: continue
    n_inv=invalid_of[k]
    for qid,a in r["answers"].items():
        b=buckets[(r["horizon"], n_inv)]
        b[1]+=1; b[0]+= (a["pred"]==a["gt"])

print("JEV 정확도 — prefix에 포함된 '엔진이 거부한 명령' 수별")
print(f"  {'h':>2} {'거부된 명령':>10} {'n':>6} {'정확도':>9}")
for (h,n_inv) in sorted(buckets):
    ok,tot=buckets[(h,n_inv)]
    print(f"  {h:>2} {n_inv:>10} {tot:>6} {ok/tot:>9.1%}")

# changed-fact만
print("\n변화한 사실만 (persistence가 0%인 항목)")
b2=defaultdict(lambda:[0,0])
for r in rows:
    if r["cond"] not in ("h1","h2"): continue
    k=(r["root_id"], r["seq_name"], r["horizon"])
    if k not in invalid_of: continue
    for qid,a in r["answers"].items():
        if not a["changed"]: continue
        c=b2[(r["horizon"], invalid_of[k])]; c[1]+=1; c[0]+=(a["pred"]==a["gt"])
for (h,n_inv) in sorted(b2):
    ok,tot=b2[(h,n_inv)]
    print(f"  h={h} 거부 {n_inv}개: {ok}/{tot} = {ok/tot:.1%}")

# 핵심: 거부된 명령이 결과를 바꾸는 항목에서의 정확도
print("\n엔진이 거부한 명령이 1개 이상인 prefix의 질문별 정확도 (h=2)")
q=defaultdict(lambda:[0,0])
for r in rows:
    if r["cond"]!="h2": continue
    k=(r["root_id"], r["seq_name"], r["horizon"])
    if invalid_of.get(k,0)==0: continue
    for qid,a in r["answers"].items():
        c=q[qid]; c[1]+=1; c[0]+=(a["pred"]==a["gt"])
for qid in sorted(q):
    ok,tot=q[qid]; print(f"  {qid:22s} {ok:3d}/{tot:3d} = {ok/tot:6.1%}")
