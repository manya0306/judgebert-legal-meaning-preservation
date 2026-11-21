"""Lightweight script-based language identification for English vs Hindi.
Hindi is written in Devanagari (U+0900-U+097F); English in Latin script. Romanised Hindi is NOT supported."""
import re
_DEV = re.compile(r"[\u0900-\u097F]")
_LAT = re.compile(r"[A-Za-z]")

def identify_language(text: str) -> str:
    d, l = len(_DEV.findall(text)), len(_LAT.findall(text))
    if d == 0 and l == 0:
        return "unknown"
    return "hi" if d >= l else "en"
