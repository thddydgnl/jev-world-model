"""Tiny-overfit gate for B, measured as written: accuracy on the prompts trained on.

Rebuilds the exact 224 training examples of artifacts/kiis_k2/overfit_typed (same
32 transitions, same option shuffles: same seeds as the run) and reads the saved
adapter's answer on each. Also reports the canonical-order accuracy again.
"""
import random, sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from kiis_train_wm import encode, evaluate, examples, load, train_prompt_acc

train = load("data/kiis/transitions/train.jsonl", 32, 0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
im_end = tok.convert_tokens_to_ids("<|im_end|>")
rng = random.Random("kiis-train-options-0")
encoded = [encode(tok, m, t, "typed", im_end) + (n,)
           for r in train for m, t, n in examples(r, "typed", rng)]
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="cuda:0")
model = PeftModel.from_pretrained(base, "artifacts/kiis_k2/overfit_typed")
model.eval()
acc = train_prompt_acc(model, tok, encoded, "cuda:0")
print(f"typed: accuracy on the {len(encoded)} training prompts (their shuffled order): {acc:.4f}")
ev = evaluate(model, tok, train, "typed", "cuda:0")
print(f"typed: canonical-order transition accuracy {ev['transition_acc']:.4f}  {ev}")
