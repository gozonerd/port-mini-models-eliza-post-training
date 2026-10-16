"""Mechanical data generator for Wave 01 (ruling K60, option 1).

Every user sentence is built from the 1966 script's own decomposition patterns, with slots filled
from the script's word lists and from a public word list (SCOWL, via Debian's wamerican package,
permissive licence). No language model writes anything here. The engine labels every turn, so
every reply token is one of Weizenbaum's templates.

    python3 tools/eliza_generate_data.py --out data --seed 20261015 --train 3000 --dev 400 --test 400

Outputs, per split: <split>.conversations.jsonl (the canonical record, one conversation per line
with per-turn traces), <split>.A.jsonl (framing A: single exchange, allowed set), <split>.B.jsonl
(framing B: history, exact next reply); plus manifest.json and coverage.tsv.

Splits are by conversation. The filler vocabulary is partitioned three ways with the seed, so a
held-out sentence contains filler words the training set never showed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from eliza import Eliza, load_script, ENGINE_PATH  # noqa: E402
from eliza import engine as _engine  # noqa: E402
esplit = _engine.split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "third-party", "ELIZA-script-DOCTOR-original-1966-CACM-appendix.txt")
WORDLIST = "/usr/share/dict/american-english"

# How a user types the word the script will see after its substitutions (the inverse of the
# script's (X = Y) rules). The script's keyword scan runs on the substituted form, so a pattern
# written with YOU is matched when the user typed I or ME; we type the common form.
INVERSE = {
    "YOU": "I", "YOUR": "my", "I": "you", "MY": "your", "ARE": "am", "WAS": "was",
    "YOURSELF": "myself", "MYSELF": "yourself", "YOU'RE": "I'm", "I'M": "you're",
    "DREAMT": "dreamed", "DREAM": "dreams", "DON'T": "don't", "CAN'T": "can't", "WON'T": "won't",
}


def inverse_word(tok: str, prev_user: str) -> str:
    if tok == "ARE":
        return "am" if prev_user.lower() == "i" else "are"
    if tok in INVERSE:
        return INVERSE[tok]
    return tok.lower()


class Generator:
    def __init__(self, script, rng: random.Random, fillers: list[str]):
        self.script = script
        self.rng = rng
        self.fillers = fillers
        self.tags = {}
        for r in script.rules.values():
            for t in r.tags:
                self.tags.setdefault(t.lstrip("/"), []).append(r.keyword)
        # (keyword, pattern index) pairs that can complete with a plain template
        self.targets = []
        for kw, r in script.rules.items():
            if kw == "zNONE":
                continue
            for i, t in enumerate(r.trans):
                self.targets.append((kw, i))
        # keywords whose rule links elsewhere with no patterns of its own (e.g. ALIKE 10 (=DIT))
        self.link_only = [kw for kw, r in script.rules.items() if not r.trans and r.link_keyword]

    def filler(self, n: int) -> list[str]:
        return [self.rng.choice(self.fillers) for _ in range(n)]

    def sentence_for(self, kw: str, pat_index: int) -> str:
        """A user sentence that, after substitution, matches the pattern (fillers may still redirect it;
        the engine's label is what counts)."""
        pattern = self.script.rules[kw].trans[pat_index].decomposition
        words: list[str] = []
        for tok in pattern:
            if tok.isdigit():
                n = int(tok)
                words += self.filler(self.rng.randint(0, 3) if n == 0 else n)
            elif tok.startswith("(*"):
                words.append(self.rng.choice(esplit(tok[2:-1])).lower())
            elif tok.startswith("(/"):
                tagged = []
                for tag in esplit(tok[2:-1]):
                    tagged += self.tags.get(tag, [])
                words.append(self.rng.choice(tagged).lower() if tagged else self.rng.choice(self.fillers))
            else:
                words.append(inverse_word(tok, words[-1] if words else ""))
        # make sure the keyword itself is present in user form when the pattern carries no literal for it
        inv_kw = INVERSE.get(kw, kw.lower())
        if inv_kw.upper() not in [w.upper() for w in words] and kw.upper() not in [w.upper() for w in words]:
            words.insert(self.rng.randint(0, len(words)), inv_kw)
        s = " ".join(words)
        if self.rng.random() < 0.5:
            s = s[0].upper() + s[1:]
        if self.rng.random() < 0.6:
            s += self.rng.choice([".", ".", "?", "!"])
        return s

    def sentence_link_only(self, kw: str) -> str:
        words = self.filler(self.rng.randint(0, 3)) + [kw.lower()] + self.filler(self.rng.randint(0, 3))
        return " ".join(words)

    def sentence_none(self) -> str:
        return " ".join(self.filler(self.rng.randint(1, 6)))

    def conversation(self) -> list[str]:
        n = self.rng.randint(2, 12)
        used: list[tuple[str, int]] = []
        out = []
        if self.rng.random() < 0.10:
            # a cycle drill: one pattern fired more times than it has templates, so every
            # position of its reply cycle, and the wrap-around, appear in one conversation
            kw, pi = self.rng.choice(self.targets)
            k = len(self.script.rules[kw].trans[pi].reassembly_rules) + 1
            return [self.sentence_for(kw, pi) for _ in range(k)]
        for _ in range(n):
            r = self.rng.random()
            if r < 0.12:
                out.append(self.sentence_none())
            elif r < 0.20 and self.link_only:
                out.append(self.sentence_link_only(self.rng.choice(self.link_only)))
            elif r < 0.50 and used:
                kw, pi = self.rng.choice(used)          # repeat: pushes the cycle position
                out.append(self.sentence_for(kw, pi))
            else:
                kw, pi = self.rng.choice(self.targets)
                used.append((kw, pi))
                out.append(self.sentence_for(kw, pi))
        return out


def label(script, conv: list[str]) -> list[dict]:
    e = Eliza(script)
    turns = []
    for u in conv:
        reply = e.response(u)
        allowed = e.allowed_replies() or [reply]
        if reply not in allowed:
            allowed = [reply] + allowed
        last = e.last
        path = last.get("path", [])
        fired = path[-1] if path else {}
        turns.append({
            "user": u, "reply": reply, "allowed": allowed, "kind": last.get("kind"),
            "keyword": fired.get("keyword"), "pattern": fired.get("pattern"),
            "template": fired.get("template"), "n_templates": fired.get("n_templates"),
            "limit": last.get("limit"),
        })
    return turns


def sha256_file(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--dev", type=int, default=400)
    ap.add_argument("--test", type=int, default=400)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    script = load_script(SCRIPT)
    keywords = {kw.upper() for kw in script.rules} | {"BUT"}
    raw = [w.strip() for w in open(WORDLIST, encoding="utf-8")]
    vocab = sorted({w for w in raw if re.fullmatch(r"[a-z]{3,8}", w) and w.upper() not in keywords})
    rng = random.Random(a.seed)
    rng.shuffle(vocab)
    third = len(vocab) // 3
    fillers = {"train": vocab[:third], "dev": vocab[third:2 * third], "test": vocab[2 * third:]}

    manifest = {
        "seed": a.seed, "script": os.path.relpath(SCRIPT, ROOT), "script_sha256": sha256_file(SCRIPT),
        "wordlist": WORDLIST, "wordlist_sha256": sha256_file(WORDLIST),
        "wordlist_licence": "SCOWL (Kevin Atkinson), permissive; Debian package wamerican",
        "generator_sha256": sha256_file(os.path.abspath(__file__)),
        "engine_sha256": sha256_file(ENGINE_PATH),
        "filler_vocabulary_sizes": {k: len(v) for k, v in fillers.items()},
        "splits": {},
    }
    coverage = collections.Counter()
    for split, count in (("train", a.train), ("dev", a.dev), ("test", a.test)):
        g = Generator(script, random.Random(f"{a.seed}-{split}"), fillers[split])
        convs = [label(script, g.conversation()) for _ in range(count)]
        with open(os.path.join(a.out, f"{split}.conversations.jsonl"), "w", encoding="utf-8") as fc, \
             open(os.path.join(a.out, f"{split}.A.jsonl"), "w", encoding="utf-8") as fa, \
             open(os.path.join(a.out, f"{split}.B.jsonl"), "w", encoding="utf-8") as fb:
            for ci, turns in enumerate(convs):
                cid = f"{split}-{ci:05d}"
                fc.write(json.dumps({"id": cid, "turns": turns}, ensure_ascii=False) + "\n")
                history = []
                for ti, t in enumerate(turns):
                    fa.write(json.dumps({"id": f"{cid}-{ti}", "user": t["user"], "allowed": t["allowed"],
                                         "reply": t["reply"]}, ensure_ascii=False) + "\n")
                    fb.write(json.dumps({"id": f"{cid}-{ti}", "history": list(history), "user": t["user"],
                                         "reply": t["reply"]}, ensure_ascii=False) + "\n")
                    history.append([t["user"], t["reply"]])
                    coverage[(split, t["kind"], t["keyword"], t["pattern"], t["template"])] += 1
        n_turns = sum(len(c) for c in convs)
        kinds = collections.Counter(t["kind"] for c in convs for t in c)
        manifest["splits"][split] = {"conversations": count, "turns": n_turns, "kinds": dict(kinds)}
        print(f"{split}: {count} conversations, {n_turns} turns, kinds={dict(kinds)}")
    with open(os.path.join(a.out, "coverage.tsv"), "w") as f:
        f.write("split\tkind\tkeyword\tpattern\ttemplate\tcount\n")
        for (s, k, kw, pi, ti), c in sorted(coverage.items(), key=lambda x: (x[0][0], str(x[0][2]), x[0][3] or -1, x[0][4] or -1)):
            f.write(f"{s}\t{k}\t{kw}\t{pi}\t{ti}\t{c}\n")
    manifest["files_sha256"] = {f: sha256_file(os.path.join(a.out, f)) for f in sorted(os.listdir(a.out))
                                if f.endswith(".jsonl") or f == "coverage.tsv"}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    # coverage summary: which (keyword, pattern) pairs are missing in each split, and cycle depth reached
    pairs = {(kw, i) for kw, r in script.rules.items() for i in range(len(r.trans))}
    for split in ("train", "dev", "test"):
        seen = {(kw, pi) for (s, k, kw, pi, ti) in coverage if s == split and k == "reply"}
        missing = sorted(p for p in pairs if p not in seen)
        depth = collections.Counter(ti for (s, k, kw, pi, ti) in coverage if s == split and k == "reply")
        print(f"{split}: patterns fired as plain replies {len(seen)}/{len(pairs)}; missing {missing[:12]}{'...' if len(missing) > 12 else ''}; cycle positions reached {dict(sorted(depth.items()))}")


if __name__ == "__main__":
    main()
