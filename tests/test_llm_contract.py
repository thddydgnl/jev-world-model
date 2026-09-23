"""The local model must answer in JEV's shape, and fail safe.

No weights are loaded: a stub stands in for the model's scores and replies, so
this checks the wiring, not the model.

1. Typed contract. Replay real JEV requests saved during K1 (artifacts/jev_raw,
   R_step and R_valid) through LlmJudgeClient: every answer must carry the same
   question ids and the same option keys as JEV's response, with probabilities
   that sum to one. Option letters must map back to the right keys.
2. Generative fail-safe. A reply that cannot be parsed, even after the retry,
   must come back as "did not execute, state unchanged" and be counted.
"""
import glob, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wm.llm_backends import GenerativeStep, LlmJudgeClient, typed_messages

ROOT = Path(__file__).resolve().parent.parent


class StubScorer:
    """Deterministic 'model': one-hot on an option picked from the prompt text."""
    def __init__(self, reply="nonsense"):
        self.reply, self.seconds, self.picked = reply, 0.0, []

    def option_probs(self, prompts, n_options):
        out = []
        for p, n in zip(prompts, n_options):
            k = sum(map(ord, p[-1]["content"])) % n
            self.picked.append(k)
            out.append([1.0 if i == k else 0.0 for i in range(n)])
        return out

    def generate(self, prompts, max_new_tokens, sample=False, seed=None):
        return [self.reply for _ in prompts]


def main():
    bad = 0
    files = sorted(glob.glob(str(ROOT / "artifacts/jev_raw/R_step_*.json")))[:40] + \
        sorted(glob.glob(str(ROOT / "artifacts/jev_raw/R_valid_*.json")))[:20]
    if not files:
        print("SKIP 1: no saved JEV R_step/R_valid responses in artifacts/jev_raw")
    for f in files:
        rec = json.loads(Path(f).read_text())
        req, resp = rec["request"], rec["response"]
        stub = StubScorer()
        got = LlmJudgeClient(stub).ask(req["state"], req["questions"])
        if set(got["answers"]) != set(resp["answers"]):
            bad += 1; print(f"  question ids differ: {f}")
            continue
        for qid, ans in got["answers"].items():
            if set(ans["probabilities"]) != set(resp["answers"][qid]["probabilities"]):
                bad += 1; print(f"  option keys differ: {f} {qid}")
            if abs(sum(ans["probabilities"].values()) - 1.0) > 1e-9:
                bad += 1; print(f"  not normalised: {f} {qid}")
        # letters map back to keys: the stub's pick k must be the key shown k-th
        for (qid, q), k in zip(req["questions"].items(), stub.picked):
            _, keys = typed_messages(req["state"], q)
            top = max(got["answers"][qid]["probabilities"],
                      key=got["answers"][qid]["probabilities"].get)
            if top != keys[k]:
                bad += 1; print(f"  letter {k} -> {top}, shown as {keys[k]}")
    print(f"1. JEV 응답과 같은 모양: {len(files)}개 요청, 불일치 {bad}")

    # 2. generative fail-safe
    import textworld
    from textworld import EnvInfos
    from agent.task import state_schema, read_schema_values
    from env.serialize import canonical_state
    from env.worlds import build_trap_world
    d = Path("/tmp/llm_contract"); d.mkdir(exist_ok=True)
    game, path, meta = build_trap_world(1, d, random.Random(0), split="test")
    schema = state_schema(game, meta)
    env = textworld.start(str(path), request_infos=EnvInfos(facts=True)); env.reset()
    canon = canonical_state(list(env.state["_facts"]), game); env.close()
    pred = GenerativeStep(StubScorer("I think the box opens."), schema)(canon, f"open {meta['box']}")
    fs_ok = (pred.parse_failed and pred.p_exec == 0.0 and pred.requests == 2
             and pred.values == read_schema_values(canon, schema))
    print(f"2. 파싱 실패 -> 실행 안 됨, 상태 유지, 재시도 1회: {'OK' if fs_ok else 'WRONG'}")
    ok = bad == 0 and fs_ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
