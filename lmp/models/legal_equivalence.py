"""Conservative exact equivalence after narrowly scoped normalization."""
import re
import unicodedata

def canonical(text, lang):
    s = unicodedata.normalize('NFC', text).strip()
    s = re.sub(r'\s+', ' ', s)
    # Only affirmative, simple English obligations; abstain on quotations,
    # negation, conditions, exceptions, or ambiguous modal contexts.
    if lang == 'en' and not re.search(r'\b(?:not|never|unless|except|if|may|might|could|would|should|whether|without|no)\b|["“”\'‘’;:?!]', s, re.I):
        m = re.fullmatch(r'(The (?:tenant|employee|employer|landlord|supplier|buyer|seller|contractor|licensee|applicant|borrower|lender)) (shall|must) (.+)', s, re.I)
        if m and re.match(r'(?:pay|submit|provide|deliver|notify|maintain|retain|return|file|register|report|complete|keep|obtain|deposit|renew|sign)\b', m[3], re.I):
            s = m[1] + ' MUST ' + m[3]
    return s

def equivalence(a,b,lang):
    if not a.strip() or not b.strip(): return None
    if unicodedata.normalize('NFC',a.strip()) == unicodedata.normalize('NFC',b.strip()):
        return 'Identical text'
    if canonical(a,lang)==canonical(b,lang):
        return 'Equivalent under restricted obligation/whitespace normalization'
    return None
