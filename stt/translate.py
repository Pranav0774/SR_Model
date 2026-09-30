"""Hindi/Tamil to English text translation with NLLB-200."""
import re

CODES = {"hi": "hin_Deva", "ta": "tam_Taml"}
MAX_WORDS = 45


def split_text(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Break a long passage into sentence-sized pieces; NLLB is weaker on very long input."""
    out = []
    for sent in re.split(r"(?<=[।.?!])\s+", text.strip()):
        words = sent.split()
        for i in range(0, len(words), max_words):
            piece = " ".join(words[i:i + max_words])
            if piece:
                out.append(piece)
    return out


class Translator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None
        self.tok = None

    def load(self):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        repo = self.cfg.get("repo", "facebook/nllb-200-distilled-600M")
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(repo).eval()

    def ensure_loaded(self):
        if self.model is None:
            self.load()

    def translate(self, texts: list[str], lang: str) -> list[str]:
        """Translate each text to English. Empty strings stay empty."""
        import torch
        if lang not in CODES:
            raise ValueError(f"Translation from '{lang}' is not supported.")
        self.ensure_loaded()
        self.tok.src_lang = CODES[lang]
        eng = self.tok.convert_tokens_to_ids("eng_Latn")

        pieces, owner = [], []
        for i, t in enumerate(texts):
            for p in split_text(t):
                pieces.append(p)
                owner.append(i)

        results = [""] * len(texts)
        step = self.cfg.get("batch_size", 4)
        for s in range(0, len(pieces), step):
            batch = pieces[s:s + step]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=200)
            with torch.no_grad():
                gen = self.model.generate(**enc, forced_bos_token_id=eng,
                                          num_beams=self.cfg.get("num_beams", 4), max_new_tokens=200)
            for j, out in enumerate(self.tok.batch_decode(gen, skip_special_tokens=True)):
                i = owner[s + j]
                results[i] = (results[i] + " " + out.strip()).strip()
        return results
