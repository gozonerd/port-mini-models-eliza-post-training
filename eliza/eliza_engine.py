"""A Python port of Joseph Weizenbaum's 1966 ELIZA, run with the DOCTOR script.

The behaviour follows Anthony Hay's C++ implementation (github.com/anthay/ELIZA, CC0),
which was checked against the rediscovered 1965 MAD-SLIP source: the keystack and
NEWKEY, the three delimiters (comma, period, BUT), the LIMIT counter that gates memory
recall, the mid-square HASH that selects the memory template from the Hollerith encoding
of the last chunk of the last word, the four built-in NOMACH messages, and the cycling
of reassembly rules per decomposition rule. Where this port and Hay's binary disagree,
this port is wrong: tools/eliza_verify_against_hay.py is the check.

The script is data. Nothing in this file produces a word that is not in the script.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SPECIAL_RULE_NONE = "zNONE"
NOMATCH_MSGS = ("PLEASE CONTINUE", "HMMM", "GO ON , PLEASE", "I SEE")

# ---------------------------------------------------------------- text helpers

_UPPER_MAP = {
    0x2019: "'",
    0x2018: " ", 0x0060: " ", 0x0022: " ", 0x00AB: " ", 0x00BB: " ", 0x201A: " ",
    0x201B: " ", 0x201C: " ", 0x201D: " ", 0x201E: " ", 0x201F: " ", 0x2039: " ", 0x203A: " ",
    0x0021: ".", 0x003F: ".",
    0x00A1: " ", 0x00BF: " ",
    0x003A: ",", 0x003B: ",", 0x2013: ",", 0x2014: ",",
    0x00DF: "SS", 0xFB00: "FF", 0xFB01: "FI", 0xFB02: "FL", 0xFB03: "FFI", 0xFB04: "FFL",
    0xFB05: "ST", 0xFB06: "ST",
}


def eliza_uppercase(s: str) -> str:
    """Hay's reinterpretation of punctuation plus uppercasing, character by character."""
    out = []
    for ch in s:
        m = _UPPER_MAP.get(ord(ch))
        out.append(m if m is not None else ch.upper())
    return "".join(out)


def split(s: str) -> List[str]:
    return [w for w in s.split(" ") if w]


def join(words: List[str]) -> str:
    return " ".join(w for w in words if w)


def split_user_input(s: str, punctuation: str) -> List[str]:
    """Split on spaces; each punctuation character becomes its own word."""
    result: List[str] = []
    word = ""
    for ch in s:
        if ch == " " or ch in punctuation:
            if word:
                result.append(word)
                word = ""
            if ch != " ":
                result.append(ch)
        else:
            word += ch
    if word:
        result.append(word)
    return result


def to_int(s: str) -> int:
    return int(s) if s.isdigit() else -1


# ------------------------------------------------------------- Hollerith + HASH

_BCD = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', None, '=', "'", None, None, None,
    '+', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', None, '.', ')', None, None, None,
    '-', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', None, '$', '*', None, None, None,
    ' ', '/', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', None, ',', '(', None, None, None,
]
HOLLERITH = {c: i for i, c in enumerate(_BCD) if c is not None}


def last_chunk_as_bcd(s: str) -> int:
    """36-bit Hollerith encoding of the last 6-character chunk of s, space padded."""
    result = 0
    count = 0
    if s:
        start = ((len(s) - 1) // 6) * 6
        for c in s[start:]:
            result = (result << 6) | HOLLERITH.get(c, ord(c) & 0x3F)
            count += 1
    while count < 6:
        result = (result << 6) | HOLLERITH[' ']
        count += 1
    return result & ((1 << 64) - 1)


def hash_(d: int, n: int) -> int:
    """The SLIP HASH(D,N): the middle n bits of the 35-bit datum squared (64-bit truncated)."""
    d &= 0x7FFFFFFFF
    d = (d * d) & ((1 << 64) - 1)
    d >>= 35 - n // 2
    return d & ((1 << n) - 1)


# ------------------------------------------------------------------- matching

TagMap = Dict[str, List[str]]


def inlist(word: str, wordlist: str, tags: TagMap) -> bool:
    wl = wordlist[:-1] if wordlist.endswith(")") else wordlist
    if wl.startswith("("):
        wl = wl[1:]
    wl = wl.lstrip(" ")
    if wl.startswith("*"):
        return word in split(wl[1:])
    if wl.startswith("/"):
        for tag in split(wl[1:]):
            if word in tags.get(tag, []):
                return True
    return False


def _xmatch(tags, pat, words, p_begin, p_end, w_begin, fixed_len, result):
    """Match pat[p_begin:p_end] to words[w_begin:]; returns w_end or None. Port of Hay's xmatch."""
    if len(words) - w_begin < fixed_len:
        return None
    has_wildcard = to_int(pat[p_begin]) == 0
    if has_wildcard:
        if p_end == len(pat):
            wildcard_len = len(words) - w_begin - fixed_len
            wildcard_end = wildcard_len
        else:
            wildcard_len = 0
            wildcard_end = len(words) - w_begin - fixed_len
    else:
        wildcard_len = 0
        wildcard_end = 0
    while True:
        p = p_begin + (1 if has_wildcard else 0)
        w = w_begin + wildcard_len
        while p < p_end:
            n = to_int(pat[p])
            if n > 0:
                result[p] = join(words[w:w + n])
                w += n
            else:
                if pat[p].startswith("("):
                    if inlist(words[w], pat[p], tags):
                        result[p] = words[w]
                        w += 1
                    else:
                        break
                elif pat[p] == words[w]:
                    result[p] = words[w]
                    w += 1
                else:
                    break
            p += 1
        if p == p_end:
            if has_wildcard:
                result[p_begin] = join(words[w_begin:w_begin + wildcard_len])
            return w
        if wildcard_len == wildcard_end:
            return None
        wildcard_len += 1


def match(tags: TagMap, pattern: List[str], words: List[str]) -> Optional[List[str]]:
    """Return the matched components (one per pattern element) or None. Port of Hay's YMATCH-style match."""
    pat = list(pattern)
    result = [""] * len(pat)
    w = 0
    p_seg_end = 0
    while p_seg_end < len(pat):
        fixed_len = 0
        p = p_seg_end
        while p_seg_end < len(pat):
            n = to_int(pat[p_seg_end])
            if n == 0:
                if p_seg_end > p:
                    break
            elif n > 0:
                fixed_len += n
            else:
                fixed_len += 1
            p_seg_end += 1
        w_end = _xmatch(tags, pat, words, p, p_seg_end, w, fixed_len, result)
        if w_end is None:
            return None
        w = w_end
    if w < len(words):
        return None
    return result


def reassemble(rule: List[str], components: List[str]) -> List[str]:
    out: List[str] = []
    for r in rule:
        n = to_int(r)
        if n < 0:
            out.append(r)
        elif n == 0 or n > len(components):
            out.append("HMMM")
        else:
            out.extend(split(components[n - 1]))
    return out


# ----------------------------------------------------------------------- rules

@dataclass
class Transform:
    decomposition: List[str]
    reassembly_rules: List[List[str]]
    next_reassembly_rule: int = 0


@dataclass
class KeywordRule:
    keyword: str
    substitution: str = ""
    precedence: int = 0
    tags: List[str] = field(default_factory=list)
    link_keyword: str = ""
    trans: List[Transform] = field(default_factory=list)

    def has_transformation(self) -> bool:
        return bool(self.trans) or bool(self.link_keyword)

    def word_substitute(self, word: str) -> str:
        if not self.substitution or word != self.keyword:
            return word
        return self.substitution

    def apply_transformation(self, words: List[str], tags: TagMap) -> Tuple[str, List[str], str, dict]:
        """Returns (action, words, link_keyword, trace); action in complete|inapplicable|newkey|linkkey."""
        rule = None
        constituents = None
        pat_index = -1
        for i, t in enumerate(self.trans):
            constituents = match(tags, t.decomposition, words)
            if constituents is not None:
                rule = t
                pat_index = i
                break
        if rule is None:
            if not self.link_keyword:
                return "inapplicable", words, "", {}
            return "linkkey", words, self.link_keyword, {}
        rr = rule.reassembly_rules[rule.next_reassembly_rule]
        trace = {"keyword": self.keyword, "pattern": pat_index, "template": rule.next_reassembly_rule,
                 "n_templates": len(rule.reassembly_rules), "constituents": list(constituents)}
        rule.next_reassembly_rule += 1
        if rule.next_reassembly_rule == len(rule.reassembly_rules):
            rule.next_reassembly_rule = 0
        if len(rr) == 1 and rr[0] == "NEWKEY":
            return "newkey", words, "", trace
        if len(rr) == 2 and rr[0] == "=":
            return "linkkey", words, rr[1], trace
        if rr and rr[0] == "(":
            # (PRE (reassembly) (=reference)) stored as ["(", "PRE", "(", ..., ")", "(", "=", REF, ")", ")"]
            i = 3
            reassembly = []
            while rr[i] != ")":
                reassembly.append(rr[i])
                i += 1
            words = reassemble(reassembly, constituents)
            i += 3
            return "linkkey", words, rr[i], trace
        return "complete", reassemble(rr, constituents), "", trace

    def all_plain_replies(self, pat_index: int, constituents: List[str]) -> List[str]:
        """Every plain (non-special) template of a pattern, instantiated: framing A's allowed set."""
        out = []
        for rr in self.trans[pat_index].reassembly_rules:
            if (len(rr) == 1 and rr[0] == "NEWKEY") or (len(rr) == 2 and rr[0] == "=") or (rr and rr[0] == "("):
                continue
            out.append(join(reassemble(rr, constituents)))
        return out


@dataclass
class MemoryRule:
    keyword: str
    trans: List[Transform] = field(default_factory=list)
    memories: List[str] = field(default_factory=list)

    def create_memory(self, keyword: str, words: List[str], tags: TagMap) -> None:
        if keyword != self.keyword:
            return
        t = self.trans[hash_(last_chunk_as_bcd(words[-1]), 2)]
        constituents = match(tags, t.decomposition, words)
        if constituents is None:
            return
        self.memories.append(join(reassemble(t.reassembly_rules[0], constituents)))

    def memory_exists(self) -> bool:
        return bool(self.memories)

    def recall_memory(self) -> str:
        return self.memories.pop(0) if self.memories else ""


# ---------------------------------------------------------------- script reader

class _Tok:
    OPEN, CLOSE, SYMBOL, NUMBER, EOF = range(5)

    def __init__(self, typ, value=""):
        self.typ, self.value = typ, value

    def open(self): return self.typ == self.OPEN
    def close(self): return self.typ == self.CLOSE
    def eof(self): return self.typ == self.EOF
    def number(self): return self.typ == self.NUMBER
    def symbol(self, v=None): return self.typ == self.SYMBOL and (v is None or self.value == v)


class _Tokenizer:
    def __init__(self, text: str):
        self.text = text
        self.i = 0
        self.peeked: Optional[_Tok] = None
        self.line = 1

    def _is_ws(self, ch): return ord(ch) <= 0x20 or ord(ch) == 0x7F

    def _read(self) -> _Tok:
        n = len(self.text)
        while True:
            while self.i < n and self._is_ws(self.text[self.i]):
                if self.text[self.i] == "\n":
                    self.line += 1
                self.i += 1
            if self.i >= n:
                return _Tok(_Tok.EOF)
            if self.text[self.i] != ";":
                break
            while self.i < n and self.text[self.i] not in "\n\r\x0b\x0c":
                self.i += 1
        ch = self.text[self.i]
        self.i += 1
        if ch == "(":
            return _Tok(_Tok.OPEN)
        if ch == ")":
            return _Tok(_Tok.CLOSE)
        if ch == "=":
            return _Tok(_Tok.SYMBOL, "=")
        if ch.isdigit():
            v = ch
            while self.i < n and self.text[self.i].isdigit():
                v += self.text[self.i]
                self.i += 1
            return _Tok(_Tok.NUMBER, v)
        v = ch
        while self.i < n:
            c = self.text[self.i]
            if c in "();=" or self._is_ws(c):
                break
            v += c
            self.i += 1
        return _Tok(_Tok.SYMBOL, eliza_uppercase(v))

    def peek(self) -> _Tok:
        if self.peeked is None:
            self.peeked = self._read()
        return self.peeked

    def next(self) -> _Tok:
        if self.peeked is not None:
            t, self.peeked = self.peeked, None
            return t
        return self._read()


@dataclass
class Script:
    hello: List[str]
    rules: Dict[str, KeywordRule]
    mem_rule: MemoryRule


def _rdlist(tok: _Tokenizer, prior: bool = True) -> List[str]:
    s: List[str] = []
    t = tok.next()
    if prior:
        if not t.open():
            raise ValueError(f"line {tok.line}: expected '('")
        t = tok.next()
    while not t.close():
        if t.symbol() or t.number():
            s.append(t.value)
        elif t.open():
            sub = []
            t = tok.next()
            while not t.close():
                if not t.symbol():
                    raise ValueError(f"line {tok.line}: expected symbol")
                sub.append(t.value)
                t = tok.next()
            s.append("(" + " ".join(sub) + ")")
        else:
            raise ValueError(f"line {tok.line}: expected ')'")
        t = tok.next()
    return s


def _read_reassembly(tok: _Tokenizer) -> List[str]:
    if not tok.next().open():
        raise ValueError(f"line {tok.line}: expected '('")
    if not tok.peek().symbol("PRE"):
        return _rdlist(tok, prior=False)
    tok.next()
    pre = ["(", "PRE"]
    reconstruct = _rdlist(tok)
    reference = _rdlist(tok)
    if len(reference) != 2 or reference[0] != "=":
        raise ValueError(f"line {tok.line}: expected '(=reference)' in PRE rule")
    pre += ["("] + reconstruct + [")", "("] + reference + [")", ")"]
    if not tok.next().close():
        raise ValueError(f"line {tok.line}: expected ')'")
    return pre


def parse_script(text: str) -> Script:
    tok = _Tokenizer(text)
    hello = _rdlist(tok)
    if tok.peek().symbol("START"):
        tok.next()
    rules: Dict[str, KeywordRule] = {}
    mem_rule: Optional[MemoryRule] = None
    while True:
        t = tok.next()
        if t.eof():
            break
        if not t.open():
            raise ValueError(f"line {tok.line}: expected '('")
        t = tok.peek()
        if t.close():
            tok.next()
            continue
        if not t.symbol():
            raise ValueError(f"line {tok.line}: expected keyword")
        if t.symbol("MEMORY"):
            tok.next()
            kw = tok.next()
            mem_rule = MemoryRule(kw.value)
            for _ in range(4):
                if not tok.next().open():
                    raise ValueError("MEMORY: expected '('")
                decomposition = []
                t = tok.next()
                while not t.symbol("=") and not t.eof():
                    decomposition.append(t.value)
                    t = tok.next()
                reassembly = []
                t = tok.next()
                while not t.close() and not t.eof():
                    reassembly.append(t.value)
                    t = tok.next()
                mem_rule.trans.append(Transform(decomposition, [reassembly]))
            if not tok.next().close():
                raise ValueError("MEMORY: expected ')'")
            continue
        # keyword rule
        t = tok.next()
        keyword = SPECIAL_RULE_NONE if t.value == "NONE" else t.value
        if keyword in rules:
            raise ValueError(f"duplicate keyword {t.value}")
        rule = KeywordRule(keyword)
        t = tok.next()
        while not t.close():
            if t.symbol("="):
                rule.substitution = tok.next().value
            elif t.number():
                rule.precedence = int(t.value)
            elif t.symbol("DLIST"):
                rule.tags = _rdlist(tok)
            elif t.open():
                if tok.peek().symbol("="):
                    tok.next()
                    rule.link_keyword = tok.next().value
                    if not tok.next().close():
                        raise ValueError("expected ')'")
                else:
                    decomposition = _rdlist(tok)
                    reassembly_rules = [_read_reassembly(tok)]
                    while tok.peek().open():
                        reassembly_rules.append(_read_reassembly(tok))
                    if not tok.next().close():
                        raise ValueError("expected ')'")
                    rule.trans.append(Transform(decomposition, reassembly_rules))
            else:
                raise ValueError(f"line {tok.line}: malformed rule")
            t = tok.next()
        rules[keyword] = rule
    if SPECIAL_RULE_NONE not in rules:
        raise ValueError("no NONE rule")
    if mem_rule is None:
        raise ValueError("no MEMORY rule")
    return Script(hello, rules, mem_rule)


def collect_tags(rules: Dict[str, KeywordRule]) -> TagMap:
    tags: TagMap = {}
    for r in rules.values():
        for t in r.tags:
            if t == "/":
                continue
            if len(t) > 1 and t.startswith("/"):
                t = t[1:]
            tags.setdefault(t, []).append(r.keyword)
    return tags


# ---------------------------------------------------------------------- ELIZA

class Eliza:
    """One conversation's worth of state: reply cycles, the LIMIT counter and the memory queue."""

    DELIMITERS = (",", ".", "BUT")

    def __init__(self, script: Script):
        import copy
        self.script = copy.deepcopy(script)   # cycles and memories are per conversation
        self.rules = self.script.rules
        self.mem_rule = self.script.mem_rule
        self.tags = collect_tags(self.rules)
        self.limit = 1
        self.last: dict = {}
        self.punctuation = "".join(d for d in self.DELIMITERS if len(d) == 1 and d in "='+.)-$*/,(")

    def hello(self) -> str:
        return join(self.script.hello)

    def response(self, user_input: str) -> str:
        words = split_user_input(eliza_uppercase(user_input), self.punctuation)
        self.limit = self.limit % 4 + 1
        keystack: List[str] = []
        top_rank = 0
        i = 0
        while i < len(words):
            w = words[i]
            if w in self.DELIMITERS:
                if not keystack:
                    del words[: i + 1]
                    i = 0
                    continue
                del words[i:]
                break
            r = self.rules.get(w)
            if r is not None:
                if r.has_transformation():
                    if r.precedence > top_rank:
                        keystack.insert(0, w)
                        top_rank = r.precedence
                    else:
                        keystack.append(w)
                words[i] = r.word_substitute(w)
            i += 1
        self.last = {"kind": "none", "limit": self.limit, "path": []}
        if not keystack:
            if self.limit == 4 and self.mem_rule.memory_exists():
                self.last["kind"] = "memory"
                return self.mem_rule.recall_memory()
        while keystack:
            top = keystack.pop(0)
            rule = self.rules.get(top)
            if rule is None:
                self.last["kind"] = "nomatch"
                return NOMATCH_MSGS[self.limit - 1]
            self.mem_rule.create_memory(top, words, self.tags)
            act, words, link, trace = rule.apply_transformation(words, self.tags)
            if trace:
                self.last["path"].append(dict(trace, action=act))
            if act == "complete":
                self.last["kind"] = "reply"
                return join(words)
            if act == "inapplicable":
                self.last["kind"] = "nomatch"
                return NOMATCH_MSGS[self.limit - 1]
            if act == "linkkey":
                keystack.insert(0, link)
            elif not keystack:
                break   # NEWKEY with an empty keystack: fall through to NONE
        none_rule = self.rules[SPECIAL_RULE_NONE]
        act, words, _, trace = none_rule.apply_transformation(words, self.tags)
        self.last["path"].append(dict(trace, action=act))
        self.last["kind"] = "none"
        return join(words)

    def allowed_replies(self) -> List[str]:
        """Framing A's allowed set for the reply just produced: every plain template of the pattern that
        completed, instantiated with the same constituents. For memory, nomatch or a reply that went
        through a special template, the set is the single actual reply (filled in by the caller)."""
        if self.last.get("kind") != "reply" or not self.last.get("path"):
            return []
        t = self.last["path"][-1]
        if t.get("action") != "complete":
            return []
        return self.rules[t["keyword"]].all_plain_replies(t["pattern"], t["constituents"])


def load_script(path: str) -> Script:
    with open(path, encoding="utf-8") as f:
        return parse_script(f.read())
