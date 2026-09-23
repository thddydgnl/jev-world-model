"""Does batched (left-padded) greedy generation give the same text as one-at-a-time?

Greedy decoding is deterministic, so for the same prompt the reply must not depend
on which other prompts share the batch. If it does, the padding is corrupting the
longer-padded rows, and every batched generative number is suspect.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "src")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wm.llm_backends import QwenScorer, gen_messages, parse_generation

recs = [json.loads(l) for l in Path("data/kiis/transitions/val.jsonl").read_text().splitlines()[:16]]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
prompts = [gen_messages(r["canon"], r["schema"], r["action"]) for r in recs]
lens = [len(QwenScorer(model, tok, "cuda:0").prompt_ids(p)) for p in prompts]
print("prompt lengths", min(lens), max(lens))

one = QwenScorer(model, tok, "cuda:0", batch=1)
t = time.time(); single = one.generate(prompts, 320); t1 = time.time() - t
many = QwenScorer(model, tok, "cuda:0", batch=8)
t = time.time(); batched = many.generate(prompts, 320); t2 = time.time() - t
same = sum(a == b for a, b in zip(single, batched))
print(f"identical replies batch=1 vs batch=8: {same}/{len(prompts)}  (time {t1:.0f}s vs {t2:.0f}s)")

def score(texts):
    ok = parsed = 0
    for r, x in zip(recs, texts):
        v, p = parse_generation(x, r["schema"], r["canon"])
        if v is None:
            continue
        parsed += 1
        want = r["values_next"] if r["executes"] else r["values_now"]
        ok += (p == (1.0 if r["executes"] else 0.0)) and v == want
    return parsed, ok
print("batch=1: parsed %d, correct %d" % score(single))
print("batch=8: parsed %d, correct %d" % score(batched))
for i, (a, b) in enumerate(zip(single, batched)):
    if a != b:
        print(f"--- #{i} len {lens[i]}\n[single]\n{a[:300]}\n[batched]\n{b[:300]}")
        break
