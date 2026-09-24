"""K3 post-training gate on the same validation subset used for selection.

Never reads test data or trains a model. Reports inference accuracy for base
and selected adapter, and token-weighted target CE for base and both epochs.
Loss prompts/options are fixed across checkpoints; losses across styles are
not comparable. Checkpoint selection remains the recorded training decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from kiis_train_wm import load, examples, encode, batches, evaluate, _body_and_head


def validation_loss(model, tok, encoded, device):
    import torch
    import torch.nn.functional as F
    model.eval()
    body, head = _body_and_head(model)
    total, tokens = 0.0, 0
    with torch.no_grad():
        for mb in batches(encoded, 8192, random.Random(0)):
            length = max(len(encoded[i][0]) for i in mb)
            x = torch.full((len(mb), length), tok.pad_token_id, dtype=torch.long)
            y = torch.full_like(x, -100)
            att = torch.zeros_like(x)
            for j, i in enumerate(mb):
                ids, labels, _ = encoded[i]
                x[j, :len(ids)] = torch.tensor(ids)
                y[j, :len(labels)] = torch.tensor(labels)
                att[j, :len(ids)] = 1
            x, y, att = x.to(device), y.to(device), att.to(device)
            hidden = body(input_ids=x, attention_mask=att, use_cache=False).last_hidden_state
            labels = y[:, 1:]
            mask = labels != -100
            total += F.cross_entropy(head(hidden[:, :-1][mask]).float(),
                                     labels[mask], reduction="sum").item()
            tokens += int(mask.sum())
    return {"target_token_ce": total / tokens, "target_tokens": tokens}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import runinfo

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    adapter = Path(args.adapter)
    meta = json.loads((adapter / "train_meta.json").read_text())
    cfg = meta["args"]
    assert not cfg["overfit"] and cfg["val"] == "data/kiis/transitions/val.jsonl"
    held = load(cfg["val"], cfg["eval_n"], cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    rng = random.Random("kiis-validation-options-0")
    encoded = [encode(tok, m, t, cfg["style"], im_end) + (n,)
               for r in held for m, t, n in examples(r, cfg["style"], rng)]
    report = {"style": cfg["style"], "adapter": str(adapter),
              "selected_epoch": meta["selected_epoch"], "transitions": len(held),
              "val_sha256": hashlib.sha256(Path(cfg["val"]).read_bytes()).hexdigest(),
              "subset_sha256": hashlib.sha256(json.dumps(held, sort_keys=True).encode()).hexdigest(),
              "loss_definition": "token-weighted target CE; fixed options per validation example",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "revision": runinfo.revision(), "software": runinfo.software(), "results": {}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16,
                                               device_map=args.device)
    model = base
    for label in ["base", "epoch_1", "epoch_2", "selected"]:
        if label != "base":
            if model is not base:
                base = model.unload()
            path = adapter if label == "selected" else adapter / label
            model = PeftModel.from_pretrained(base, str(path))
        metrics = validation_loss(model, tok, encoded, args.device)
        if label in ("base", "selected"):
            metrics["inference"] = evaluate(model, tok, held, cfg["style"], args.device)
        if label != "base":
            metrics["weights_sha256"] = runinfo.adapter_info(str(path))["weights_sha256"]
        report["results"][label] = metrics
        print(label, json.dumps(metrics), flush=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    r = report["results"]
    report["gates"] = {
        "accuracy_improves": r["selected"]["inference"]["transition_acc"] > r["base"]["inference"]["transition_acc"],
        "val_loss_decreases": r["epoch_2"]["target_token_ce"] < r["epoch_1"]["target_token_ce"] < r["base"]["target_token_ce"],
        "selected_weights_match": r["selected"]["weights_sha256"] == r[f"epoch_{meta['selected_epoch']}"]["weights_sha256"],
        "selection_accuracy_reproduced": r["selected"]["inference"] == meta["epochs"][meta["selected_epoch"] - 1]["eval"],
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("GATES", json.dumps(report["gates"]), flush=True)
    return 0 if all(report["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
