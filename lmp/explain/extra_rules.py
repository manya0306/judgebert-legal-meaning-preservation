"""Additional legal-semantic detectors (v2). Each returns Finding objects; all remain HEURISTIC and carry a hand-set confidence.
Covers: conjunction/disjunction flips in conditions ('and' -> 'or'), comparator direction flips (lower/higher, at least/at most),
statute/section reference changes, party-order (actor swap), and quantity-phrase direction ('at least' vs 'up to')."""
from __future__ import annotations
import re
from . import lexicon as L

_AND = {"en": re.compile(r"(?<![\w'])and(?![\w])", re.I), "hi": re.compile(r"(?<![\u0900-\u097F])और(?![\u0900-\u097F])")}
_OR = {"en": re.compile(r"(?<![\w'])or(?![\w])", re.I), "hi": re.compile(r"(?<![\u0900-\u097F])(?:या|अथवा)(?![\u0900-\u097F])")}
_LOW = {"en": re.compile(r"(?<![\w'])(?:lower|less|fewer|below|under|minimum|smaller)(?![\w])", re.I),
        "hi": re.compile(r"(?<![\u0900-\u097F])(?:कम|न्यूनतम|नीचे)(?![\u0900-\u097F])")}
_HIGH = {"en": re.compile(r"(?<![\w'])(?:higher|more|greater|above|over|maximum|larger)(?![\w])", re.I),
         "hi": re.compile(r"(?<![\u0900-\u097F])(?:अधिक|ज़्यादा|ज्यादा|अधिकतम|ऊपर)(?![\u0900-\u097F])")}
_ATLEAST = {"en": re.compile(r"at least|not less than|minimum of", re.I), "hi": re.compile(r"कम से कम")}
_ATMOST = {"en": re.compile(r"at most|not more than|up to|not exceed|maximum of|no more than", re.I), "hi": re.compile(r"अधिकतम|से अधिक नहीं|तक सीमित")}
_REF = {"en": re.compile(r"\b(?:section|sections|clause|article|rule|schedule|sub-section)\s+\(?\w+\)?(?:\(\w+\))?", re.I),
        "hi": re.compile(r"(?:धारा|खंड|अनुच्छेद|नियम|अनुसूची|उप-धारा)\s+\(?[\w\u0966-\u096F]+\)?")}

def _hits(rx, text): return [m.group(0) for m in rx.finditer(text)]

def run(original: str, simplified: str, lang: str, fo: dict, fs: dict, Finding) -> list:
    out = []
    # 1. conjunction <-> disjunction among conditions (only when the original actually has a condition/exception structure)
    ao, bo = len(_AND[lang].findall(original)), len(_OR[lang].findall(original))
    as_, bs = len(_AND[lang].findall(simplified)), len(_OR[lang].findall(simplified))
    if (fo["conditions"] or fo["exceptions"] or fs["conditions"] or fs["exceptions"]) or "unless" in original.lower():
        if bs > bo and as_ < ao:
            out.append(Finding("altered_condition", "condition logic", "and", "or", "A conjunction ('and') in the original became a disjunction ('or') in the simplified text.",
                               "Cumulative conditions may have become alternatives, which can widen or narrow the rule's reach.", 0.6, "and-or-flip"))
        elif as_ > ao and bs < bo:
            out.append(Finding("altered_condition", "condition logic", "or", "and", "A disjunction ('or') in the original became a conjunction ('and') in the simplified text.",
                               "Alternative conditions may have become cumulative.", 0.6, "or-and-flip"))
    # 2. comparator direction flips (lower<->higher)
    lo_o, hi_o, lo_s, hi_s = (_hits(_LOW[lang], original), _hits(_HIGH[lang], original), _hits(_LOW[lang], simplified), _hits(_HIGH[lang], simplified))
    if (lo_o and not lo_s and hi_s and not hi_o) or (hi_o and not hi_s and lo_s and not lo_o):
        a, b = (lo_o or hi_o), (lo_s or hi_s)
        out.append(Finding("contradiction", "comparison direction", ", ".join(dict.fromkeys(a)), ", ".join(dict.fromkeys(b)),
                           f"Comparator changed from {a[0]!r} to {b[0]!r}.", "The direction of a comparison (e.g. lower vs higher) may be reversed.", 0.6, "comparator-flip"))
    # 3. 'at least' vs 'at most' bounds
    if (_ATLEAST[lang].search(original) and _ATMOST[lang].search(simplified) and not _ATMOST[lang].search(original)) or \
       (_ATMOST[lang].search(original) and _ATLEAST[lang].search(simplified) and not _ATLEAST[lang].search(original)):
        out.append(Finding("scope_change", "bound direction", (_ATLEAST[lang].search(original) or _ATMOST[lang].search(original)).group(0),
                           (_ATLEAST[lang].search(simplified) or _ATMOST[lang].search(simplified)).group(0), "A minimum bound became a maximum bound (or vice versa).",
                           "The permitted range of the quantity may have changed.", 0.65, "bound-flip"))
    # 4. statute / section references
    ro, rs = sorted(x.lower() for x in _hits(_REF[lang], original)), sorted(x.lower() for x in _hits(_REF[lang], simplified))
    if ro != rs and (ro or rs):
        out.append(Finding("altered_reference", "legal reference", ", ".join(ro) or "(none)", ", ".join(rs) or "(absent)",
                           f"Reference(s) differ: original {ro or 'none'}, simplified {rs or 'none'}.",
                           "A cited provision may have been changed or dropped; the rule may point to a different source.", 0.8 if ro and rs else 0.6, "reference-mismatch"))
    # 5. party order (actor swap): same role set but different order of first mentions, with modal present
    def order(text, roles):
        pos = []
        for r in roles:
            m = re.search((r"(?<![\w'])%ss?(?![\w])" % re.escape(r)) if lang == "en" else re.escape(r), text, re.I if lang == "en" else 0)
            if m: pos.append((m.start(), r))
        return [r for _, r in sorted(pos)]
    common = fo["roles"] & fs["roles"]
    if len(common) >= 2:
        oo, os_ = order(original, common), order(simplified, common)
        if oo != os_:
            out.append(Finding("altered_entities", "party order", " -> ".join(oo), " -> ".join(os_), "The order in which the parties are mentioned differs.",
                               "The roles of actor and beneficiary may have been swapped (who owes/gets what).", 0.5, "party-order-differs"))
    return out
