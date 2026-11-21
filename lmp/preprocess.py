import re, unicodedata

def normalize(text: str) -> str:
    """Unicode NFC normalisation, quote unification and whitespace collapsing. Case is kept
    (legal text can be case-sensitive); lowercasing is done by individual models when needed."""
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()

_TOKEN = re.compile(r"[\w\u0900-\u097F]+(?:'[\w]+)?", re.UNICODE)

def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text).lower())
