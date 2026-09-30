"""Text normalisation used before computing WER."""
import re
import unicodedata

from num2words import num2words


def _spell_numbers_en(text: str) -> str:
    """Turn digits into the words a speaker would say, so "20th" matches "twentieth".

    FLEURS references are written in spoken form. Applied to hypotheses only.
    """
    text = re.sub(r"(\d)%", r"\1 percent", text)
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: num2words(int(m.group(1)), to="ordinal"), text)

    def cardinal(m):
        n = int(m.group(0).replace(",", ""))
        if 1100 <= n <= 1999 and "," not in m.group(0):
            return num2words(n, to="year")
        return num2words(n)

    text = re.sub(r"\d[\d,]*", cardinal, text)
    return text.replace("-", " ")


_INDIC_DIGITS = {ord(d): str(i) for base in (0x0966, 0x0BE6) for i, d in enumerate(chr(base + k) for k in range(10))}


def _unify_indic(text: str) -> str:
    """Spelling variants that Indic ASR scoring conventionally treats as equal.

    Removes the nukta (तूफ़ान = तूफान), maps chandrabindu to anusvara (हूँ = हूं), drops zero-width
    joiners, and writes Devanagari/Tamil digits as ASCII digits. Applied to reference and hypothesis alike.
    """
    text = unicodedata.normalize("NFD", text)
    text = text.replace("़", "").replace("ँ", "ं").replace("‌", "").replace("‍", "")
    return unicodedata.normalize("NFC", text).translate(_INDIC_DIGITS)


def normalize(text: str, spell_numbers: bool = False, indic: bool = False) -> str:
    """Lowercase, NFC, drop punctuation/symbols (including the danda), collapse spaces.

    Apostrophes are kept so English contractions still match the reference.
    spell_numbers is for English; indic is for Hindi and Tamil. Use the same flags on both sides.
    """
    text = unicodedata.normalize("NFC", text).lower().replace("’", "'")
    if indic:
        text = _unify_indic(text)
    if spell_numbers:
        text = _spell_numbers_en(text)
    out = []
    for ch in text:
        if ch == "'":
            out.append(ch)
        elif unicodedata.category(ch)[0] in ("P", "S"):
            out.append(" ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())
