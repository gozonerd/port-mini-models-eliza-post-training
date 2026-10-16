"""Train a LoRA adapter on one framing, evaluate at fixed points, record everything.

    ~/.venvs/eval-stack/bin/python tools/eliza_train_adapter.py --config configs/<arm>.json

The config is the prespecification's per-arm record (model, revision, framing, steps, learning rate,
batch, sequence length, LoRA rank/alpha/targets, seed, evaluation points). Nothing is chosen here that
the config does not state; a missing key is an error, not a default.

The base weights stay frozen; only the adapter trains. The loss is next-token cross-entropy on the reply
tokens only (the prompt tokens are masked out). The prompt format, decoding and scoring are the floor's
(tools/eliza_measure_floor.py) so trained arms and floors are comparable.

Outputs under runs/: <run-id>.config.json, <run-id>.loss.tsv (step, loss), <run-id>.evals.tsv (one row
per evaluation point: step, split, framing, n, score), the final adapter in runs/adapters/<run-id>/,
and rows in runs/eliza_runs.tsv (kind "train-eval"). Optional Weights & Biases logging with --wandb.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eliza_measure_floor import build_prompt, normalize, PROMPT_FORMAT_ID, GREETING  # noqa: E402

REQUIRED = ["model", "revision", "framing", "steps", "learning_rate", "batch_size", "grad_accum",
            "max_seq_len", "lora_r", "lora_alpha", "lora_dropout", "target_modules", "seed",
            "eval_points", "eval_dev_items", "final_eval_split", "device"]


def load_rows(split: str, framing: str) -> list[dict]:
    path = os.path.join(ROOT, "data", f"{split}.{framing}.jsonl")
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        sys.exit(f"config is missing {missing}; the prespecification must state every value")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = cfg["device"]
    # GPU precision (PRESPEC v04, D2 ruling "fix before", K86): the base model is held in fp16 and the forward pass
    # runs under fp16 autocast with dynamic loss scaling; the LoRA weights stay fp32 (peft casts adapters up when the
    # base is half precision). A T4 has no native bfloat16, so the earlier bfloat16 path is replaced. CPU stays fp32.
    dtype = torch.float32 if device == "cpu" else torch.float16
    use_amp = device != "cpu"
    precision = "fp16-amp" if use_amp else "fp32"
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=dtype).to(device)
    for p in base.parameters():
        p.requires_grad_(False)
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    framing = cfg["framing"]
    train_rows = load_rows("train", framing)
    dev_rows = load_rows("dev", framing)[: cfg["eval_dev_items"]]
    run_id = f"train_{cfg['model'].split('/')[-1]}_{framing}_s{cfg['seed']}_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    os.makedirs(a.out, exist_ok=True)
    cfg_record = dict(cfg, run_id=run_id, prompt_format=PROMPT_FORMAT_ID, greeting=GREETING,
                      trainable_params=n_train, total_params=n_total, dtype=str(dtype), precision=precision,
                      data_manifest_sha256=hashlib.sha256(open(os.path.join(ROOT, "data", "manifest.json"), "rb").read()).hexdigest()[:12],
                      machine=platform.node(), started_utc=dt.datetime.now(dt.UTC).isoformat())
    json.dump(cfg_record, open(os.path.join(a.out, f"{run_id}.config.json"), "w"), indent=2)
    print(f"{run_id}: trainable {n_train:,} of {n_total:,} parameters; {len(train_rows)} training items", flush=True)

    wb = None
    if a.wandb:
        import wandb
        wb = wandb.init(project="port-mini-models-eliza", name=run_id, config=cfg_record)

    def encode(rows: list[dict]):
        """prompt tokens masked (-100), reply tokens + newline scored."""
        ids, labels = [], []
        for r in rows:
            p = tok(build_prompt(r.get("history", []), r["user"]), add_special_tokens=False)["input_ids"]
            t = tok(r["reply"] + "\n", add_special_tokens=False)["input_ids"]
            seq = (p + t)[-cfg["max_seq_len"]:]
            lab = ([-100] * len(p) + t)[-cfg["max_seq_len"]:]
            ids.append(seq); labels.append(lab)
        L = max(len(s) for s in ids)
        pad = tok.pad_token_id
        ids_t = torch.tensor([s + [pad] * (L - len(s)) for s in ids])
        lab_t = torch.tensor([l + [-100] * (L - len(l)) for l in labels])
        att = (ids_t != pad).long()
        for i, s in enumerate(ids):           # a real pad token inside the sequence must stay attended
            att[i, : len(s)] = 1
        return ids_t.to(device), att.to(device), lab_t.to(device)

    def evaluate(rows: list[dict], label: str, step: int) -> float:
        model.eval()
        tok.padding_side = "left"
        n_ok = 0
        bs = 8
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                batch = rows[i: i + bs]
                enc = tok([build_prompt(r.get("history", []), r["user"]) for r in batch], return_tensors="pt", padding=True).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for r, t in zip(batch, texts):
                    g = normalize(next((ln for ln in t.split("\n") if ln.strip()), ""))
                    ok = g in {normalize(x) for x in r["allowed"]} if framing == "A" else g == normalize(r["reply"])
                    n_ok += ok
        tok.padding_side = "right"
        model.train()
        score = n_ok / max(1, len(rows))
        with open(os.path.join(a.out, f"{run_id}.evals.tsv"), "a") as f:
            if f.tell() == 0:
                f.write("step\tsplit\tframing\tn\tcorrect\tscore\n")
            f.write(f"{step}\t{label}\t{framing}\t{len(rows)}\t{n_ok}\t{score:.4f}\n")
        if wb:
            wb.log({f"eval/{label}_{framing}": score, "step": step})
        print(f"eval step {step} {label}: {n_ok}/{len(rows)} = {score:.4f}", flush=True)
        return score

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"], weight_decay=0.0)
    steps = cfg["steps"]
    eval_steps = sorted({int(round(steps * f)) for f in cfg["eval_points"]})
    model.train()
    order = list(range(len(train_rows)))
    random.shuffle(order)
    pos = 0
    t0 = time.time()
    loss_f = open(os.path.join(a.out, f"{run_id}.loss.tsv"), "w")
    loss_f.write("step\tloss\telapsed_s\n")
    if 0 in eval_steps:
        evaluate(dev_rows, "dev", 0)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(cfg["grad_accum"]):
            if pos + cfg["batch_size"] > len(order):
                random.shuffle(order); pos = 0
            batch = [train_rows[j] for j in order[pos: pos + cfg["batch_size"]]]
            pos += cfg["batch_size"]
            ids, att, lab = encode(batch)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / cfg["grad_accum"]
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt)
        scaler.update()
        loss_f.write(f"{step}\t{total:.4f}\t{time.time() - t0:.0f}\n"); loss_f.flush()
        if wb:
            wb.log({"train/loss": total, "step": step})
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{steps} loss {total:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step in eval_steps:
            evaluate(dev_rows, "dev", step)
    loss_f.close()

    adapter_dir = os.path.join(a.out, "adapters", run_id)
    model.save_pretrained(adapter_dir)
    final_rows = load_rows(cfg["final_eval_split"], framing)
    score = evaluate(final_rows, cfg["final_eval_split"], steps)

    runs_path = os.path.join(a.out, "eliza_runs.tsv")
    new = not os.path.exists(runs_path)
    with open(runs_path, "a", encoding="utf-8") as f:
        if new:
            f.write("run_id\tkind\tmodel\trevision\tadapter\tframing\tsplit\tn\tcorrect\tscore\tconfig_hash\t"
                    "data_manifest_sha256\tstarted_utc\tfinished_utc\tmachine\tstatus\tnote\n")
        chash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
        f.write("\t".join([run_id, "train-eval", cfg["model"], cfg["revision"], os.path.relpath(adapter_dir, ROOT), framing,
                           cfg["final_eval_split"], str(len(final_rows)), str(int(round(score * len(final_rows)))),
                           f"{score:.4f}", chash, cfg_record["data_manifest_sha256"], cfg_record["started_utc"],
                           dt.datetime.now(dt.UTC).isoformat(), platform.node(), "ok", a.note]) + "\n")
    if wb:
        wb.finish()
    print("saved", adapter_dir)


if __name__ == "__main__":
    main()
