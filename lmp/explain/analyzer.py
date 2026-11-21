"""Rule-based legal-difference detection (English + Hindi), with optional model-supported checks.

Two clearly separated layers in every Finding:
  * `detected_change`   - a purely TEXTUAL observation (what expression is present in one sentence but not the other)
  * `interpretation`    - a heuristic reading of what the change COULD mean legally (never legal advice)
`confidence` is a hand-set heuristic reliability score for the rule (NOT a calibrated probability).
`model_signal` (optional) is the change in a supplied similarity/LMP scorer when the missing expression is re-inserted
into the simplified sentence ("restoration test"); it is a weak, model-derived hint, not a legal finding.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict, field
from . import lexicon as L
from ..langid import identify_language
from ..preprocess import normalize
from .spans import word_diff, locate
from . import extra_rules

DISCLAIMER = ("This is an automated research prototype. Its output is not legal advice and may be wrong; "
              "consult a qualified legal professional for any legal decision.")

@dataclass
class Finding:
    category: str            # one of the 10 ERROR_LABELS
    element: str             # e.g. "obligation/permission", "number"
    original_expression: str
    simplified_expression: str
    detected_change: str
    interpretation: str
    confidence: float
    rule: str
    model_signal: float | None = None
    orig_spans: list = field(default_factory=list)       # [(start, end)] char offsets in the ORIGINAL text
    simp_spans: list = field(default_factory=list)       # [(start, end)] char offsets in the SIMPLIFIED text
    layer: str = "rule_based"                            # rule_based | model_signal (never 'validated')
    uncertainty: str = ""                                # low | medium | high (derived from the heuristic confidence)
    def to_dict(self): return asdict(self)

@dataclass
class Analysis:
    language: str
    findings: list[Finding] = field(default_factory=list)
    features_original: dict = field(default_factory=dict)
    features_simplified: dict = field(default_factory=dict)
    disclaimer: str = DISCLAIMER
    diff: list = field(default_factory=list)             # textual token alignment (NOT a legal finding)
    no_findings_note: str = ""
    @property
    def labels(self) -> set[str]: return {f.category for f in self.findings}

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

def _surfaces(rx, text):  return [m.group(0) for m in rx.finditer(text)]

def main_clause(text: str, lang: str) -> str:
    t = L.LEADING_COND[lang].sub("", text)
    return L.TRAILING_EXC[lang].sub("", t)

def modal_class(text: str, lang: str):
    """Return (class, matched_expression, negated). Classes: obligation, recommendation, permission, prohibition,
    conditional_prohibition (prohibition + 'without ...', functionally an obligation to obtain something), or None."""
    mc = main_clause(text, lang)
    M = L.MODALS[lang]
    negs = _surfaces(L.NEGATION[lang], mc)
    negated = bool(negs)
    without = bool(L.WITHOUT[lang].search(mc))
    hit = None
    if M["prohibited"].search(mc):
        hit = ("prohibition", M["prohibited"].search(mc).group(0))
    else:
        for kind in ("permission", "obligation", "recommendation", "future"):
            if kind in M and (m := M[kind].search(mc)):
                hit = (kind, m.group(0)); break
        if hit and hit[0] == "future": hit = ("obligation", hit[1])
        if hit and negated and hit[0] in ("permission", "obligation", "recommendation"):
            hit = ("prohibition", (hit[1] + " " + negs[0]) if lang == "en" else (negs[0] + " " + hit[1]))
    if hit is None:
        return None, "", negated
    cls, expr = hit
    if cls == "prohibition" and without:
        cls = "conditional_prohibition"
    return cls, expr, negated

def _numbers_and_dates(text: str, lang: str):
    """Return (numbers, dates) as lists of (canonical, surface)."""
    t = text.translate(_DEV_DIGITS)
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)
    nums, dates = [], []
    for rx, canon in L.MULTIPLIERS[lang]:
        for m in rx.finditer(t):
            nums.append((canon, m.group(0)))
        t = rx.sub(" ", t)
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(st|nd|rd|th)?(?![\w])", t):
        val, ordinal = m.group(1), m.group(2)
        after = t[m.end(): m.end() + 14].lower()
        before = t[max(0, m.start() - 14): m.start()].lower()
        near_month = any(mo in after or mo in before for mo in L.MONTHS)
        is_year = len(val) == 4 and 1900 <= int(float(val)) <= 2100
        if ordinal or near_month or is_year or "तारीख" in after:
            dates.append((("year:" if is_year else "day:") + val, m.group(0)))
        else:
            nums.append((val.rstrip("0").rstrip(".") if "." in val else val, m.group(0)))
    for w, v in L.NUMBER_WORDS[lang].items():
        pat = (r"(?<![\w'])%s(?![\w])" % w) if lang == "en" else (r"(?<![\u0900-\u097F])%s(?![\u0900-\u097F])" % w)
        for m in re.finditer(pat, t, re.I):
            nums.append((str(v), m.group(0)))
    for mo, canon in L.MONTHS.items():
        if mo == "may":   # the modal verb 'may' must not be read as the month
            pat = r"(?:\d\s*(?:st|nd|rd|th)?\s+(?:of\s+)?May\b|\bMay\s+\d)"
            for m in re.finditer(pat, t): dates.append(("month:may_m", m.group(0)))
            continue
        for m in re.finditer((r"(?<![\w])%s(?![\w])" % mo) if lang == "en" else mo, t, re.I):
            dates.append(("month:" + canon, m.group(0)))
    return nums, dates

def extract_features(text: str, lang: str) -> dict:
    text = normalize(text)
    cls, expr, neg = modal_class(text, lang)
    nums, dates = _numbers_and_dates(text, lang)
    exc_spans = list(L.EXCEPTION[lang].finditer(text))
    masked = L.EXCEPTION[lang].sub(" ", text)
    low = main_clause(text, lang).lower()
    roles = set()
    for r in L.ROLES[lang]:
        if (lang == "en" and re.search(r"(?<![\w])%ss?(?![\w])" % r, low)) or (lang == "hi" and r in main_clause(text, lang)):
            roles.add(r)
    if lang == "hi" and "मकान मालिक" in main_clause(text, lang): roles.discard("मालिक")
    return {
        "modal_class": cls, "modal_expr": expr, "negated": neg,
        "numbers": nums, "dates": dates,
        "conditions": [m.group(0) for m in L.CONDITION[lang].finditer(masked)],
        "exceptions": [m.group(0) for m in exc_spans],
        "time_rel": {k: _surfaces(rx, text) for k, rx in L.TIME_REL[lang].items() if rx.search(text)},
        "scope": {k: _surfaces(rx, text) for k, rx in L.SCOPE[lang].items() if rx.search(text)},
        "vague_time": _surfaces(L.VAGUE_TIME[lang], text),
        "salient": {k: _surfaces(rx, text) for k, rx in L.SALIENT[lang].items() if rx.search(text)},
        "roles": roles,
    }

def _j(items): return ", ".join(dict.fromkeys(items)) if items else "(none)"

_COMPAT = [{"obligation", "conditional_prohibition"}]
def _compatible(a, b): return a == b or any({a, b} <= g for g in _COMPAT)

def analyze(original: str, simplified: str, lang: str | None = None, scorer=None, min_confidence: float = 0.35) -> Analysis:
    """scorer: optional callable (originals:list, simplified:list)->array of scores, used only for `model_signal`."""
    original, simplified = normalize(original), normalize(simplified)
    lang = lang or identify_language(original)
    if lang not in ("en", "hi"): lang = "en"
    fo, fs = extract_features(original, lang), extract_features(simplified, lang)
    out: list[Finding] = []

    # 1) obligation / permission / prohibition / polarity -------------------------------------------------
    co, cs = fo["modal_class"], fs["modal_class"]
    modal_done = False
    if co and cs and not _compatible(co, cs):
        a, b = fo["modal_expr"], fs["modal_expr"]
        pair = (co, cs)
        if "prohibition" in pair and (set(pair) & {"permission", "obligation"}) or pair in {("conditional_prohibition", "permission")}:
            cat, conf = "contradiction", 0.85
            interp = f"A {co.replace('_', ' ')} became a {cs.replace('_', ' ')}; the two may have opposite legal effect."
        elif co == "obligation" and cs in ("permission", "recommendation"):
            cat, conf = "changed_obligation", 0.9 if cs == "permission" else 0.75
            interp = ("A mandatory duty appears to have become " + ("optional (permission)." if cs == "permission" else "a recommendation (weaker duty)."))
        elif co == "permission" and cs in ("obligation", "recommendation"):
            cat, conf = "changed_permission", 0.8
            interp = "An option/permission appears to have become a requirement or recommendation."
        elif co == "recommendation" and cs == "permission" or co == "recommendation" and cs == "obligation":
            cat, conf = "changed_obligation", 0.5
            interp = "The strength of the normative expression changed (recommendation vs requirement/permission)."
        else:
            cat, conf = "changed_obligation", 0.5
            interp = f"The normative force changed from {co} to {cs}."
        out.append(Finding(cat, "obligation/permission/prohibition", a, b,
                           f'Modal expression "{a}" ({co}) in the original vs "{b}" ({cs}) in the simplified text.', interp, conf, "modal-class-mismatch"))
        modal_done = True
    elif co and not cs and not fs["negated"] and co in ("obligation", "permission", "prohibition"):
        out.append(Finding("changed_obligation" if co == "obligation" else "changed_permission", "obligation/permission",
                           fo["modal_expr"], "(absent)", f'Original modal "{fo["modal_expr"]}" has no counterpart in the simplified text.',
                           "The normative force of the clause may have been dropped or left implicit.", 0.4, "modal-dropped"))
        modal_done = True
    if not modal_done and fo["negated"] != fs["negated"] and not (co == "conditional_prohibition" or cs == "conditional_prohibition"):
        neg_o = _surfaces(L.NEGATION[lang], main_clause(original, lang)); neg_s = _surfaces(L.NEGATION[lang], main_clause(simplified, lang))
        out.append(Finding("contradiction", "negation", _j(neg_o) if neg_o else "(no negation)", _j(neg_s) if neg_s else "(no negation)",
                           "Negation is present in only one of the two main clauses.", "The polarity of the statement may be reversed.", 0.8, "negation-flip"))

    # 2) conditions / exceptions -------------------------------------------------------------------------
    miss_exc = [e for e in fo["exceptions"] if e.lower() not in [x.lower() for x in fs["exceptions"]]]
    if miss_exc and not fs["exceptions"]:
        out.append(Finding("missing_exception", "exception", _j(miss_exc), "(absent)", f'Exception marker(s) {_j(miss_exc)} appear only in the original.',
                           "A carve-out or exception may have been removed, widening the rule.", 0.8, "exception-dropped"))
    miss_cond = [c for c in fo["conditions"] if c.lower() not in [x.lower() for x in fs["conditions"]]]
    if miss_cond and not fs["conditions"]:
        out.append(Finding("altered_condition", "condition", _j(miss_cond), "(absent)", f'Condition marker(s) {_j(miss_cond)} appear only in the original.',
                           "A precondition may have been dropped so the rule now applies unconditionally.", 0.75, "condition-dropped"))
    elif fs["conditions"] and not fo["conditions"]:
        out.append(Finding("altered_condition", "condition", "(absent)", _j(fs["conditions"]), f'Condition marker(s) {_j(fs["conditions"])} appear only in the simplified text.',
                           "A new condition may have been introduced.", 0.6, "condition-added"))

    # 3) numbers / dates ----------------------------------------------------------------------------------
    def multiset_diff(a, b):
        b_left = [c for c, _ in b]; miss = []
        for c, s in a:
            if c in b_left: b_left.remove(c)
            else: miss.append((c, s))
        extra = [(c, s) for c, s in b if c in b_left and (b_left.remove(c) or True)] if False else None
        left_c = [c for c, _ in b]
        for c, _ in a:
            if c in left_c: left_c.remove(c)
        extra = [(c, s) for c, s in b if c in left_c]
        return miss, extra
    for kind, cat in (("numbers", "altered_numbers"), ("dates", "altered_dates")):
        miss, extra = multiset_diff(fo[kind], fs[kind])
        if miss or extra:
            vague = fs["vague_time"] or fs["scope"].get("vague")
            if miss and not extra:
                chg = f'{kind[:-1].capitalize()} {_j([s for _, s in miss])} appears only in the original.'
                interp = ("A specific figure/deadline was dropped" + (f" (replaced by a vague term: {_j(vague)})" if vague else "") + "; the obligation may now be less precise or unenforceable as written.")
                conf = 0.85
            elif extra and not miss:
                chg = f'{kind[:-1].capitalize()} {_j([s for _, s in extra])} appears only in the simplified text.'
                interp, conf = "A figure/date was introduced that is not in the original.", 0.7
            else:
                chg = f'Original {kind}: {_j([s for _, s in miss])}; simplified {kind}: {_j([s for _, s in extra])}.'
                interp, conf = f"A {kind[:-1]} appears to have been replaced by a different one.", 0.9
            out.append(Finding(cat, kind[:-1], _j([s for _, s in miss]), _j([s for _, s in extra]) if extra else "(absent)", chg, interp, conf, f"{kind}-mismatch"))

    # 4) time relations / scope ------------------------------------------------------------------------------
    ro, rs = set(fo["time_rel"]), set(fs["time_rel"])
    only_o = ro - rs
    if only_o:
        if rs & ro:
            out.append(Finding("scope_change", "temporal scope", _j([x for k in only_o for x in fo["time_rel"][k]]), "(absent)",
                               f"Time relation(s) {sorted(only_o)} appear only in the original.", "The period the rule covers may have been narrowed.", 0.55, "time-scope-narrowed"))
        elif rs:
            out.append(Finding("altered_dates", "deadline relation", _j([x for k in only_o for x in fo["time_rel"][k]]), _j([x for k in rs for x in fs["time_rel"][k]]),
                               f"Time relation changed from {sorted(ro)} to {sorted(rs)}.", "A deadline or time reference may have shifted (e.g. 'within' vs 'after').", 0.6, "time-relation-changed"))
    so, ss = set(fo["scope"]), set(fs["scope"])
    if so != ss:
        a = _j([x for k in so - ss for x in fo["scope"][k]]); b = _j([x for k in ss - so for x in fs["scope"][k]])
        swapped = bool({"universal", "restrictive"} & (so ^ ss)) and not ({"universal"} == (so - ss) and not (ss - so))
        conf = 0.65 if ("restrictive" in (so ^ ss) or ((so - ss) and (ss - so))) else 0.3   # a lone dropped 'all/any' is weak evidence
        out.append(Finding("scope_change", "scope quantifier", a, b, f'Scope term(s) differ: original {sorted(so)} vs simplified {sorted(ss)}.',
                           "The set of people/things/situations covered may have been widened or narrowed.", conf, "scope-quantifier"))
    # entities / parties
    lost, gained = fo["roles"] - fs["roles"], fs["roles"] - fo["roles"]
    if lost or gained:
        out.append(Finding("altered_entities", "party/entity", _j(sorted(lost)) if lost else "(none lost)", _j(sorted(gained)) if gained else "(none added)",
                           f"Party/entity terms differ: only in original {sorted(lost)}; only in simplified {sorted(gained)}.",
                           "A person, role or thing named in the rule may have been removed or added.", 0.55 if lost else 0.4, "entity-set-differs"))

    # 5) omission of legally salient notions -----------------------------------------------------------------
    for notion, surf in fo["salient"].items():
        if notion not in fs["salient"]:
            out.append(Finding("omission", notion, _j(surf), "(absent)", f'Salient notion "{notion}" ({_j(surf)}) has no counterpart in the simplified text.',
                               f"A formal requirement ({notion}) may have been dropped.", 0.65, "salient-notion-dropped"))

    extra = extra_rules.run(original, simplified, lang, fo, fs, Finding)
    if any(f.category == "altered_reference" for f in extra):        # a changed section number is a reference change, not a generic number change
        ref_txt = " ".join(f.original_expression + " " + f.simplified_expression for f in extra if f.category == "altered_reference")
        out = [f for f in out if not (f.category == "altered_numbers" and all(t.strip() in ref_txt for t in (f.original_expression + "," + f.simplified_expression).split(",") if t.strip() and not t.strip().startswith("(")))]
    out += extra

    # optional model-supported check -------------------------------------------------------------------------
    if scorer is not None:
        for f in out:
            if f.original_expression and not f.original_expression.startswith("("):
                try:
                    base = float(scorer([original], [simplified])[0])
                    restored = float(scorer([original], [simplified + " " + f.original_expression])[0])
                    f.model_signal = round(restored - base, 3)
                except Exception:
                    f.model_signal = None
    # order: most confident first
    out = [f for f in out if f.confidence >= min_confidence]   # weak heuristics are hidden by default (min_confidence=0 shows all)
    out.sort(key=lambda f: -f.confidence)
    for f in out:
        f.orig_spans, f.simp_spans = locate(original, f.original_expression, lang), locate(simplified, f.simplified_expression, lang)
        f.uncertainty = "low" if f.confidence >= 0.75 else ("medium" if f.confidence >= 0.55 else "high")
    note = "" if out else "No rule fired. This does NOT show that the legal meaning is preserved: the rules only cover the listed phenomena and cannot read paraphrase."
    return Analysis(lang, out, {k: v for k, v in fo.items() if k != "numbers"}, {k: v for k, v in fs.items() if k != "numbers"}, diff=word_diff(original, simplified), no_findings_note=note)
