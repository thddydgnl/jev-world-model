"""Train the LLM world models B (typed) and D (generative): K2 tiny overfit, K3.

Same transitions for both styles; only the answer format differs
(kiis2026f/실험계획.md §1.4):

  typed  one example per question — the prompt LlmJudgeClient sends, options in
         a fresh random order each time, target = the letter of the true option
  gen    one example per transition — the prompt GenerativeStep sends, target =
         the result line and the true next-state facts

Loss is cross-entropy on the target tokens only, read from the hidden states at
those positions: full-vocabulary logits at every position do not fit.
Evaluation runs through the inference code the agent uses (TypedStep or
GenerativeStep on a QwenScorer), so an adapter is judged exactly as it will run.
Engine truth only; no JEV output is read (MCA 2.3(b)).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wm.llm_backends import (LETTERS, GenerativeStep, LlmJudgeClient, QwenScorer,
                             _body_and_head, gen_messages, gen_target, typed_messages)
from wm.recursive_forecaster import TypedStep

LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"]}


def load(path: str, n: int | None, seed: int) -> list[dict]:
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    random.Random(f"kiis-train-subset-{seed}").shuffle(recs)
    return recs if n is None else recs[:n]


def examples(rec: dict, style: str,
             rng: random.Random) -> list[tuple[list[dict], str, int | None]]:
    """(messages, target, number of options) — options only for typed."""
    if style == "gen":
        return [(gen_messages(rec["canon"], rec["schema"], rec["action"]),
                 gen_target(rec["executes"], rec["next_facts"]), None)]
    (s1, q1, _), (s2, q2, _) = TypedStep(None, rec["schema"]).requests(rec["canon"], rec["action"])
    out = []
    for qid, q in list(q1.items()) + [("v_exec", q2["v_exec"])]:
        truth = ("executes" if rec["executes"] else "fails") if qid == "v_exec" \
            else rec["values_next"][qid]
        order = list(q["criteria"]); rng.shuffle(order)
        msgs, keys = typed_messages(s2 if qid == "v_exec" else s1, q, order=order)
        out.append((msgs, LETTERS[keys.index(truth)], len(keys)))
    return out


def train_prompt_acc(model, tok, encoded, device: str, batch: int = 8) -> float:
    """Typed only: accuracy on the exact prompts trained on (their shuffled
    option order), read the way inference reads a choice — argmax over the
    letters of that question's options. This is what the tiny-overfit gate
    asks; `evaluate` asks the same questions in canonical order, as the agent
    will, which is a slightly different input."""
    import torch
    letter_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
    body, head = _body_and_head(model)
    model.eval()
    ok = 0
    with torch.no_grad():
        for s in range(0, len(encoded), batch):
            chunk = encoded[s:s + batch]
            prompts = [ids[:-1] for ids, _, _ in chunk]        # drop the answer letter
            L = max(len(p) for p in prompts)
            x = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, p in enumerate(prompts):
                x[j, :len(p)] = torch.tensor(p); att[j, :len(p)] = 1
            h = body(input_ids=x.to(device), attention_mask=att.to(device),
                     use_cache=False).last_hidden_state
            last = torch.tensor([len(p) - 1 for p in prompts], device=device)
            logits = head(h[torch.arange(len(chunk), device=device), last]).float()
            for j, (ids, _, n_opts) in enumerate(chunk):
                sel = logits[j, letter_ids[:n_opts]]
                ok += int(letter_ids[int(sel.argmax())] == ids[-1])
    model.train()
    return ok / max(len(encoded), 1)


def encode(tok, msgs, target: str, style: str, im_end: int) -> tuple[list[int], list[int]]:
    prompt = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False),
                 add_special_tokens=False)["input_ids"]
    tgt = tok(target, add_special_tokens=False)["input_ids"]
    if style == "typed":
        assert len(tgt) == 1, target
    else:
        tgt = tgt + [im_end]
    return prompt + tgt, [-100] * len(prompt) + tgt


def batches(encoded, micro_tokens: int, rng: random.Random) -> list[list[int]]:
    """Length-bucketed micro-batches under a padded-token budget, shuffled."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    out, cur, longest = [], [], 0
    for i in order:
        L = max(longest, len(encoded[i][0]))
        if cur and (L * (len(cur) + 1) > micro_tokens or len(cur) >= 16):
            out.append(cur); cur, L = [], len(encoded[i][0])
        cur.append(i); longest = L
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def evaluate(model, tok, recs, style, device) -> dict:
    """Transition accuracy as the agent would see it (kiis2026f/실험계획.md §6.2)."""
    model.eval()
    scorer = QwenScorer(model, tok, device, batch=8)
    by_world = defaultdict(list)
    for r in recs:
        by_world[r["world_id"]].append(r)
    n = ok = ex_n = ex_ok = fail_n = fail_ok = var_ok = var_n = parse_fail = 0
    for rs in by_world.values():
        schema = rs[0]["schema"]
        step = (TypedStep(LlmJudgeClient(scorer), schema) if style == "typed"
                else GenerativeStep(scorer, schema))
        preds = step.predict_batch([(r["canon"], r["action"]) for r in rs])
        for r, p in zip(rs, preds):
            says_runs = p.p_exec >= 0.5
            parse_fail += p.parse_failed
            right_valid = says_runs == r["executes"]
            if r["executes"]:
                ex_n += 1; ex_ok += right_valid
                good = sum(p.values[k] == r["values_next"][k] for k in r["values_next"])
                var_ok += good; var_n += len(r["values_next"])
                all_vars = good == len(r["values_next"])
            else:
                fail_n += 1; fail_ok += right_valid
                all_vars = True
            n += 1; ok += right_valid and all_vars
    model.train()
    return {"transitions": n, "transition_acc": ok / max(n, 1),
            "validity_balanced_acc": 0.5 * (ex_ok / max(ex_n, 1) + fail_ok / max(fail_n, 1)),
            "var_acc_when_executes": var_ok / max(var_n, 1),
            "parse_fail_rate": parse_fail / max(n, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=("typed", "gen"), required=True)
    ap.add_argument("--train", default="data/kiis/transitions/train.jsonl")
    ap.add_argument("--val", default="data/kiis/transitions/val.jsonl")
    ap.add_argument("--n", type=int, default=None, help="train transitions (default all)")
    ap.add_argument("--overfit", action="store_true",
                    help="evaluate on the training transitions (tiny-overfit check)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--micro-tokens", type=int, default=8192)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (throughput runs)")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate every k epochs (generation is slow)")
    ap.add_argument("--stop-at", type=float, default=None,
                    help="stop once transition accuracy reaches this (overfit runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train = load(args.train, args.n, args.seed)
    held = train if args.overfit else load(args.val, args.eval_n, args.seed)[:args.eval_n]
    data_sha = hashlib.sha256(Path(args.train).read_bytes()).hexdigest()[:16]

    tok = AutoTokenizer.from_pretrained(args.model)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    rng = random.Random(f"kiis-train-options-{args.seed}")
    encoded = [encode(tok, m, t, args.style, im_end) + (n,)
               for r in train for m, t, n in examples(r, args.style, rng)]
    lens = [len(x) for x, _, _ in encoded]
    print(f"{args.style}: {len(train)} transitions -> {len(encoded)} examples, "
          f"tokens mean {sum(lens)/len(lens):.0f} max {max(lens)} total {sum(lens):,}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                device_map=args.device)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **LORA))
    model.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    body, head = _body_and_head(model)

    brng = random.Random(f"kiis-train-batches-{args.seed}")
    per_epoch = math.ceil(len(batches(encoded, args.micro_tokens, random.Random(0))) / args.accum)
    total = per_epoch * args.epochs if args.max_steps is None else min(args.max_steps, per_epoch * args.epochs)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - max(0, s - warm) / max(1, total - warm)))

    import runinfo
    log = {"args": vars(args), "lora": LORA, "data_sha": data_sha,
           "n_transitions": len(train), "n_examples": len(encoded),
           "tokens_per_epoch": sum(lens), "epochs": [],
           "revision": runinfo.revision(), "versions": runinfo.software()}
    step = 0
    t_start = time.time()
    real_tokens = 0
    done = False
    for epoch in range(1, args.epochs + 1):
        mbs = batches(encoded, args.micro_tokens, brng)
        run_loss = 0.0; seen = 0
        for k, mb in enumerate(mbs):
            L = max(len(encoded[i][0]) for i in mb)
            x = torch.full((len(mb), L), tok.pad_token_id, dtype=torch.long)
            y = torch.full((len(mb), L), -100, dtype=torch.long)
            att = torch.zeros((len(mb), L), dtype=torch.long)
            for j, i in enumerate(mb):
                ids, lab, _ = encoded[i]
                x[j, :len(ids)] = torch.tensor(ids); y[j, :len(lab)] = torch.tensor(lab)
                att[j, :len(ids)] = 1
                real_tokens += len(ids)
            x, y, att = x.to(args.device), y.to(args.device), att.to(args.device)
            h = body(input_ids=x, attention_mask=att, use_cache=False).last_hidden_state
            lab = y[:, 1:]; m = lab != -100
            loss = F.cross_entropy(head(h[:, :-1][m]).float(), lab[m])
            (loss / args.accum).backward()
            run_loss += loss.item(); seen += 1
            if (k + 1) % args.accum == 0 or k + 1 == len(mbs):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0 or step == 1:
                    el = time.time() - t_start
                    print(f"  step {step}/{total} loss {run_loss/seen:.4f} "
                          f"tok/s {real_tokens/el:,.0f} "
                          f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
                if args.max_steps and step >= args.max_steps:
                    done = True
                    break
        el = time.time() - t_start
        rec = {"epoch": epoch, "steps": step, "train_loss": run_loss / max(seen, 1),
               "seconds": round(el, 1), "tokens_per_s": round(real_tokens / el, 1)}
        if not args.max_steps and (epoch % args.eval_every == 0 or epoch == args.epochs):
            rec["eval"] = evaluate(model, tok, held, args.style, args.device)
            if args.overfit and args.style == "typed":
                rec["eval"]["train_prompt_acc"] = train_prompt_acc(model, tok, encoded, args.device)
            if not args.overfit:
                # Keep every evaluated epoch; the one with the best validation
                # transition accuracy becomes the adapter (plan K3: chosen on val).
                model.save_pretrained(str(out / f"epoch_{epoch}"))
        log["epochs"].append(rec)
        print(f"epoch {epoch}: {json.dumps(rec)}", flush=True)
        if done:
            break
        if args.stop_at and rec.get("eval", {}).get("transition_acc", 0) >= args.stop_at:
            print(f"reached {args.stop_at:.0%} — stopping", flush=True)
            break

    el = time.time() - t_start
    log.update({"gpu_hours": round(el / 3600, 3), "tokens_per_s": round(real_tokens / el, 1),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "gpu": torch.cuda.get_device_name(0)})
    if not args.max_steps:
        scored = [e for e in log["epochs"] if "eval" in e]
        if args.overfit or not scored:
            model.save_pretrained(str(out))
            log["selected_epoch"] = log["epochs"][-1]["epoch"]
        else:
            best = max(scored, key=lambda e: e["eval"]["transition_acc"])
            import shutil
            src = out / f"epoch_{best['epoch']}"
            for f in src.iterdir():
                shutil.copy2(f, out / f.name)
            log["selected_epoch"] = best["epoch"]
            print(f"selected epoch {best['epoch']} "
                  f"(val transition acc {best['eval']['transition_acc']:.3f})", flush=True)
    (out / "train_meta.json").write_text(json.dumps(log, indent=1) + "\n")
    print(f"done: {el/60:.1f} min, {log['tokens_per_s']:,.0f} tok/s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
