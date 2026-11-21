"""Automatic evidence spans from a token-level diff: the largest differing span in each sentence (verbatim substrings)."""
from __future__ import annotations
import difflib

def diff_evidence(original: str, simplified: str) -> tuple[str, str]:
    a, b = original.split(), simplified.split()
    ops = [o for o in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if o[0] != "equal"]
    if not ops:
        return "(none)", "(none)"
    tag, i1, i2, j1, j2 = max(ops, key=lambda o: (o[2] - o[1]) + (o[4] - o[3]))
    ev_o = " ".join(a[i1:i2]) if i2 > i1 else "(none)"
    ev_s = " ".join(b[j1:j2]) if j2 > j1 else "(absent)"
    return ev_o, ev_s
