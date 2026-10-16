"""Zero-shot floor: score an untrained base model on the held-out test set, both framings.

    ~/.venvs/eval-stack/bin/python tools/measure_floor.py --model EleutherAI/pythia-70m --revision <sha> \
        [--split test] [--limit N] [--out runs]

Harness choices are fixed in FLOOR_PROTOCOL.md and recorded with every result: the transcript prompt
format (greeting line, then alternating user line and reply line, reply lines in capitals), greedy
decoding, at most 48 new tokens, the reply taken as the first non-empty generated line. Scoring is exact match
after the script's own normalization for framing B, and membership in the allowed set for framing A.

Outputs: <out>/<run-id>.items.tsv (one row per item: id, framing, expected, got, match) and a row
appended to <out>/runs.tsv (run id, model, revision, framing, split, n, score, config hash, timestamps,
machine). Nothing here trains anything.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREETING = "HOW DO YOU DO. PLEASE TELL ME YOUR PROBLEM"
PROMPT_FORMAT_ID = "transcript-v1"  # greeting line, then "user\nREPLY\n" pairs, ending with "user\n"


def build_prompt(history: list[list[str]], user: str) -> str:
    lines = [GREETING]
    for u, r in history:
        lines += [u, r]
    lines.append(user)
    return "\n".join(lines) + "\n"


def normalize(s: str) -> str:
    return " ".join(s.strip().upper().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True, help="Hugging Face commit hash, pinned")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N items of each framing (0 = all)")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if a.device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=dtype).to(a.device).eval()

    config = {"prompt_format": PROMPT_FORMAT_ID, "greeting": GREETING, "decoding": "greedy",
              "max_new_tokens": a.max_new_tokens, "dtype": str(dtype), "device": a.device,
              "normalization": "first non-empty line, upper, whitespace-collapsed"}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    run_id = f"floor_{a.model.split('/')[-1]}_{a.split}_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    started = dt.datetime.now(dt.UTC).isoformat()

    def generate(prompts: list[str]) -> list[str]:
        enc = tok(prompts, return_tensors="pt", padding=True).to(a.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        # the reply is the first non-empty line of the continuation (a model may open with a newline)
        return [next((ln for ln in t.split("\n") if ln.strip()), "") for t in texts]

    results = {}
    items_path = os.path.join(a.out, f"{run_id}.items.tsv")
    with open(items_path, "w", encoding="utf-8") as fi:
        fi.write("id\tframing\texpected\tgot\tmatch\n")
        for framing in ("A", "B"):
            path = os.path.join(ROOT, "data", f"{a.split}.{framing}.jsonl")
            rows = [json.loads(l) for l in open(path, encoding="utf-8")]
            if a.limit:
                rows = rows[: a.limit]
            n_ok = 0
            t0 = time.time()
            for i in range(0, len(rows), a.batch):
                batch = rows[i: i + a.batch]
                prompts = [build_prompt(r.get("history", []), r["user"]) for r in batch]
                gots = generate(prompts)
                for r, got in zip(batch, gots):
                    g = normalize(got)
                    if framing == "A":
                        ok = g in {normalize(x) for x in r["allowed"]}
                    else:
                        ok = g == normalize(r["reply"])
                    n_ok += ok
                    fi.write(f"{r['id']}\t{framing}\t{normalize(r['reply'])}\t{g}\t{int(ok)}\n")
                if (i // a.batch) % 20 == 0:
                    print(f"{framing} {i + len(batch)}/{len(rows)} running score {n_ok / (i + len(batch)):.3f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            results[framing] = {"n": len(rows), "correct": n_ok, "score": n_ok / max(1, len(rows))}
            print(f"{framing}: {n_ok}/{len(rows)} = {results[framing]['score']:.4f}", flush=True)

    runs_path = os.path.join(a.out, "runs.tsv")
    new = not os.path.exists(runs_path)
    with open(runs_path, "a", encoding="utf-8") as f:
        if new:
            f.write("run_id\tkind\tmodel\trevision\tadapter\tframing\tsplit\tn\tcorrect\tscore\tconfig_hash\t"
                    "data_manifest_sha256\tstarted_utc\tfinished_utc\tmachine\tstatus\tnote\n")
        manifest_sha = hashlib.sha256(open(os.path.join(ROOT, "data", "manifest.json"), "rb").read()).hexdigest()[:12]
        for framing, r in results.items():
            f.write("\t".join([run_id, "floor", a.model, a.revision, "none", framing, a.split, str(r["n"]),
                               str(r["correct"]), f"{r['score']:.4f}", config_hash, manifest_sha, started,
                               dt.datetime.now(dt.UTC).isoformat(), platform.node(), "ok", a.note]) + "\n")
    json.dump({"run_id": run_id, "config": config, "results": results, "model": a.model, "revision": a.revision},
              open(os.path.join(a.out, f"{run_id}.json"), "w"), indent=2)
    print("wrote", items_path)


if __name__ == "__main__":
    main()
