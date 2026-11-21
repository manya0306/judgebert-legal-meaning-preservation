"""HTML span highlighting for findings (escaped; offsets come from the analyzer)."""
from __future__ import annotations
import html

def highlight(text: str, spans: list[tuple[int, int]], css: str = "background:#ffe08a;padding:0 2px;border-radius:3px") -> str:
    spans = sorted({(int(a), int(b)) for a, b in spans if 0 <= a < b <= len(text)})
    out, pos = [], 0
    for a, b in spans:
        if a < pos: continue
        out += [html.escape(text[pos:a]), f'<mark style="{css}">{html.escape(text[a:b])}</mark>']; pos = b
    out.append(html.escape(text[pos:])); return "".join(out)
