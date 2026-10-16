"""Contamination probe (REPTME.E3): how much of the 1966 DOCTOR script does an untrained base model already carry?

    python tools/eliza_probe_contamination.py --model EleutherAI/pythia-70m --revision <sha> [--device cuda] [--out runs]

Three measurements per model, none of which trains anything. Fixed before any run; recorded with every result.

1. CACM recital. The article's own conversation (tests/test_cacm.py: the greeting and 15 exchanges) is the most
   reproduced ELIZA text there is. For each exchange k, the prompt is the transcript-v1 format (FLOOR_PROTOCOL.md)
   of the article's conversation up to user line k, and the model's greedy reply is compared to the article's
   reply after the floor's normalization. A base model that recites the article has read it. Score: matches / 15.

2. Fixed-line memorization margin. Every slot-free reply line of the script (reassembly rules with no
   constituent slot, excluding NEWKEY redirects), plus the four NOMATCH messages and the greeting, is scored
   by mean per-token negative log-likelihood (NLL, nats) in three forms: verbatim; word-shuffled (mean of
   --shuffles seeded shuffles of the same words); and lowercase. The margin for a line is
   NLL(shuffled) - NLL(verbatim). A fluent line has a positive margin because grammar is predictable; a
   memorized line has a much larger one. Reported per line and as the median over lines.

3. Reference margin. The same statistic over --n-reference reply lines sampled (seeded) from the test split's
   conversations whose text is NOT one of the fixed lines, i.e. templates filled with held-out filler words.
   They share the script's template structure but cannot have been memorized, because the filler words were
   generated here. The contamination signal is median fixed-line margin minus median reference margin.

Scoring is done bare: BOS (or nothing, if the tokenizer has no BOS) followed by the line and a newline; the
NLL is averaged over the line's tokens only. Outputs: <out>/<run-id>.lines.tsv (one row per line and form),
<out>/<run-id>.cacm.tsv (one row per exchange), <out>/<run-id>.json, and a row per measurement appended to
<out>/eliza_probe_runs.tsv.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from eliza import load_script  # noqa: E402
from eliza.eliza_engine import NOMATCH_MSGS  # noqa: E402
from test_cacm import EXCHANGES  # noqa: E402

SCRIPT = os.path.join(ROOT, "third-party", "ELIZA-script-DOCTOR-original-1966-CACM-appendix.txt")
GREETING = "HOW DO YOU DO. PLEASE TELL ME YOUR PROBLEM"
PROMPT_FORMAT_ID = "transcript-v1"
PROBE_VERSION = "probe-v1"


def normalize(s: str) -> str:
    return " ".join(s.strip().upper().split())


def build_prompt(history, user):
    lines = [GREETING]
    for u, r in history:
        lines += [u, r]
    lines.append(user)
    return "\n".join(lines) + "\n"


def fixed_lines(script) -> list[str]:
    out = set()
    for rule in script.rules.values():
        for t in rule.trans:
            for rr in t.reassembly_rules:
                if not rr or rr[0] == "=" or rr[0] == "PRE":
                    continue
                if any(tok.isdigit() for tok in rr):
                    continue
                out.add(" ".join(rr))
    out.update(NOMATCH_MSGS)
    out.add(GREETING)
    return sorted(out)


def reference_lines(fixed: set[str], n: int, seed: int) -> list[str]:
    path = os.path.join(ROOT, "data", "test.conversations.jsonl")
    cands = []
    for line in open(path, encoding="utf-8"):
        for t in json.loads(line)["turns"]:
            r = t["reply"]
            if t.get("kind") == "reply" and r not in fixed and len(r.split()) >= 3:
                cands.append(r)
    cands = sorted(set(cands))
    rng = random.Random(seed)
    rng.shuffle(cands)
    return cands[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--n-reference", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N fixed lines (smoke tests)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if a.device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=dtype).to(a.device).eval()

    script = load_script(SCRIPT)
    fixed = fixed_lines(script)
    if a.limit:
        fixed = fixed[: a.limit]
    ref = reference_lines(set(fixed_lines(script)), a.n_reference, a.seed)

    config = {"probe_version": PROBE_VERSION, "prompt_format": PROMPT_FORMAT_ID, "greeting": GREETING,
              "scoring": "mean per-token NLL (nats) over the line's tokens, bare context: BOS + line + newline",
              "shuffles": a.shuffles, "n_reference": a.n_reference, "seed": a.seed, "dtype": str(dtype),
              "device": a.device, "n_fixed_lines": len(fixed), "cacm_exchanges": len(EXCHANGES),
              "script_sha256": hashlib.sha256(open(SCRIPT, "rb").read()).hexdigest()}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    run_id = f"probe_{a.model.split('/')[-1]}_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}"
    started = dt.datetime.now(dt.UTC).isoformat()
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []

    @torch.no_grad()
    def nll_per_token(text: str) -> tuple[float, int]:
        ids = tok(text + "\n", add_special_tokens=False)["input_ids"]
        full = torch.tensor([bos + ids], device=a.device)
        logits = model(full).logits[0, :-1].float()
        targets = full[0, 1:]
        lp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])[:, 0]
        n_line = len(ids)                      # includes the newline token; excludes BOS
        line_lp = lp[-n_line:] if bos else lp  # without BOS the first token has no prediction
        return float(-line_lp.mean()), int(line_lp.numel())

    def margin_rows(lines: list[str], kind: str, rng: random.Random, fo):
        margins = []
        for line in lines:
            v, n = nll_per_token(line)
            shuf = []
            words = line.split()
            for _ in range(a.shuffles):
                w = words[:]
                rng.shuffle(w)
                shuf.append(nll_per_token(" ".join(w))[0])
            s = statistics.fmean(shuf)
            lo, _ = nll_per_token(line.lower())
            m = s - v
            margins.append(m)
            fo.write(f"{kind}\t{line}\t{n}\t{v:.4f}\t{s:.4f}\t{lo:.4f}\t{m:.4f}\n")
        return margins

    rng = random.Random(a.seed)
    lines_path = os.path.join(a.out, f"{run_id}.lines.tsv")
    t0 = time.time()
    with open(lines_path, "w", encoding="utf-8") as fo:
        fo.write("kind\tline\tn_tokens\tnll_verbatim\tnll_shuffled_mean\tnll_lowercase\tmargin\n")
        m_fixed = margin_rows(fixed, "fixed", rng, fo)
        print(f"fixed lines: {len(m_fixed)} scored, median margin {statistics.median(m_fixed):.3f} ({time.time()-t0:.0f}s)", flush=True)
        m_ref = margin_rows(ref, "reference", rng, fo)
        print(f"reference lines: {len(m_ref)} scored, median margin {statistics.median(m_ref):.3f} ({time.time()-t0:.0f}s)", flush=True)

    # CACM recital
    cacm_path = os.path.join(a.out, f"{run_id}.cacm.tsv")
    hits = 0
    with open(cacm_path, "w", encoding="utf-8") as fo:
        fo.write("k\tuser\texpected\tgot\tmatch\n")
        history = []
        for k, (u, expected) in enumerate(EXCHANGES):
            enc = tok(build_prompt(history, u), return_tensors="pt").to(a.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            got = normalize(next((ln for ln in text.split("\n") if ln.strip()), ""))
            ok = got == normalize(expected)
            hits += ok
            fo.write(f"{k}\t{u}\t{normalize(expected)}\t{got}\t{int(ok)}\n")
            history.append([u, expected])   # the article's own reply goes into the history, as printed
    print(f"CACM recital: {hits}/{len(EXCHANGES)} ({time.time()-t0:.0f}s)", flush=True)

    results = {
        "cacm_recital": {"n": len(EXCHANGES), "matches": hits, "score": hits / len(EXCHANGES)},
        "fixed_margin": {"n": len(m_fixed), "median": statistics.median(m_fixed), "mean": statistics.fmean(m_fixed)},
        "reference_margin": {"n": len(m_ref), "median": statistics.median(m_ref), "mean": statistics.fmean(m_ref)},
    }
    results["contamination_signal"] = results["fixed_margin"]["median"] - results["reference_margin"]["median"]
    finished = dt.datetime.now(dt.UTC).isoformat()
    runs_path = os.path.join(a.out, "eliza_probe_runs.tsv")
    new = not os.path.exists(runs_path)
    with open(runs_path, "a", encoding="utf-8") as f:
        if new:
            f.write("run_id\tkind\tmodel\trevision\tmeasure\tn\tvalue\tconfig_hash\tstarted_utc\tfinished_utc\tmachine\tdtype\tnote\n")
        rows = [("cacm_recital", len(EXCHANGES), results["cacm_recital"]["score"]),
                ("fixed_margin_median", len(m_fixed), results["fixed_margin"]["median"]),
                ("reference_margin_median", len(m_ref), results["reference_margin"]["median"]),
                ("contamination_signal", len(m_fixed), results["contamination_signal"])]
        for measure, n, value in rows:
            f.write("\t".join([run_id, "probe", a.model, a.revision, measure, str(n), f"{value:.4f}", config_hash,
                               started, finished, platform.node(), str(dtype), a.note]) + "\n")
    json.dump({"run_id": run_id, "config": config, "results": results, "model": a.model, "revision": a.revision,
               "started_utc": started, "finished_utc": finished},
              open(os.path.join(a.out, f"{run_id}.json"), "w"), indent=2)
    print(json.dumps(results, indent=2))
    print("wrote", lines_path, cacm_path)


if __name__ == "__main__":
    main()
