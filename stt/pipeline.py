"""Routes audio to the right engine for each language."""
import json
import threading
import time
from pathlib import Path

import numpy as np

from .audio import SR, load_audio, split_chunks
from .engines import ENGINES
from .translate import Translator

SUPPORTED = ("en", "hi", "ta")
ROOT = Path(__file__).resolve().parent.parent


class Transcriber:
    def __init__(self, config_path: str | Path = ROOT / "config.json"):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._engines = {}
        self._lid = None
        self._lock = threading.Lock()
        self.translator = Translator(self.config.get("translation", {}))
        self.ready = False
        self.loading = ""
        self.warm_error = None

    def engine(self, name: str):
        if name not in self._engines:
            self._engines[name] = ENGINES[name](self.config["engines"][name])
        eng = self._engines[name]
        eng.ensure_loaded()
        return eng

    def _load_lid(self):
        if self._lid is None:
            from faster_whisper import WhisperModel
            c = self.config["language_id"]
            self._lid = WhisperModel(c["model"], device=c["device"], compute_type=c["compute_type"])

    def warm_up(self):
        """Load the language detector and every engine the routes use, so the first request is fast."""
        try:
            self.loading = "language detection"
            with self._lock:
                self._load_lid()
            for name in dict.fromkeys(self.config["routes"].values()):
                self.loading = name
                with self._lock:
                    self.engine(name)
            if self.config.get("translation", {}).get("preload", True):
                self.loading = "translation"
                with self._lock:
                    self.translator.ensure_loaded()
            self.ready = True
        except Exception as e:  # missing weights, gated repo, low memory...
            self.warm_error = f"{type(e).__name__}: {e}"
        finally:
            self.loading = ""

    def detect_language(self, wav: np.ndarray) -> str:
        """Pick English, Hindi or Tamil from the first 30 s using a small Whisper model."""
        self._load_lid()
        _, _, probs = self._lid.detect_language(audio=wav[: 30 * SR])
        scores = {lang: p for lang, p in probs if lang in SUPPORTED}
        return max(scores, key=scores.get)

    def transcribe_array(self, wav: np.ndarray, language: str, engine: str | None = None,
                         translate: bool = False) -> dict:
        with self._lock:
            t0 = time.time()
            if language == "auto":
                language = self.detect_language(wav)
            name = engine or self.config["routes"][language]
            eng = self.engine(name)
            if language not in eng.languages:
                raise ValueError(f"Engine '{name}' does not support language '{language}'.")
            pieces = [wav] if eng.max_chunk_s is None else split_chunks(wav, eng.max_chunk_s)
            parts = [eng.transcribe(p, language) for p in pieces]
            text = " ".join(t for t in parts if t)
            result = {
                "text": text,
                "language": language,
                "engine": name,
                "audio_seconds": round(len(wav) / SR, 2),
                "translation": None,
            }
            # translate each audio piece separately: the cuts fall on pauses, so pieces are sentence-like
            if translate and language != "en" and text:
                result["translation"] = " ".join(t for t in self.translator.translate(parts, language) if t)
            result["elapsed_seconds"] = round(time.time() - t0, 2)
            return result

    def transcribe_file(self, path: str, language: str = "auto", engine: str | None = None,
                        translate: bool = False) -> dict:
        return self.transcribe_array(load_audio(path), language, engine, translate)
