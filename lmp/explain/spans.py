"""Textual diff and span location (character offsets) for evidence-linked explanations.
`word_diff` is a purely TEXTUAL alignment (what changed on the surface). It is separate from the inferred legal-semantic findings."""
from __future__ import annotations
import difflib, re

_TOK = re.compile(r"\S+")

def word_diff(original: str, simplified: str) -> list[dict]:
    """Token alignment with char offsets: [{op: replace|delete|insert, original, simplified, orig_span, simp_span}]."""
    to, ts = list(_TOK.finditer(original)), list(_TOK.finditer(simplified))
    a, b = [m.group(0).strip(".,;:।") .lower() for m in to], [m.group(0).strip(".,;:।").lower() for m in ts]
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        os_ = (to[i1].start(), to[i2 - 1].end()) if i2 > i1 else None
        ss_ = (ts[j1].start(), ts[j2 - 1].end()) if j2 > j1 else None
        out.append({"op": tag, "original": original[os_[0]:os_[1]] if os_ else "", "simplified": simplified[ss_[0]:ss_[1]] if ss_ else "",
                    "orig_span": os_, "simp_span": ss_})
    return out

def locate(text: str, expr: str, lang: str) -> list[tuple[int, int]]:
    """Character spans of each comma-separated expression of a finding inside `text` (case-insensitive; skips '(absent)'-style placeholders)."""
    spans = []
    for part in [p.strip() for p in re.split(r",| -> ", str(expr))]:
        if not part or part.startswith("("):
            continue
        for m in re.finditer(re.escape(part), text, re.I if lang == "en" else 0):
            spans.append((m.start(), m.end()))
            break
    return spans
