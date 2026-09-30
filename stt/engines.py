"""Speech recognition back ends. Each one loads lazily and exposes transcribe(wav, lang)."""
import warnings
from pathlib import Path

import numpy as np

from .audio import SR

ROOT = Path(__file__).resolve().parent.parent

warnings.filterwarnings("ignore", message=".*torch.jit.load.*")


class Engine:
    name = ""
    languages: tuple = ()
    max_chunk_s: float | None = 28.0  # None = the engine handles long audio itself

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None

    def load(self):
        raise NotImplementedError

    def transcribe(self, wav: np.ndarray, lang: str) -> str:
        raise NotImplementedError

    def ensure_loaded(self):
        if self.model is None:
            self.load()


class R2T2Engine(Engine):
    """Confucius4-R2T2 weights, run through the qwen_asr transformers backend."""
    name = "r2t2"
    languages = ("en", "hi")
    _names = {"en": "English", "hi": "Hindi"}

    def load(self):
        import torch
        from qwen_asr import Qwen3ASRModel
        dtype = getattr(torch, self.cfg.get("dtype", "float32"))
        self.model = Qwen3ASRModel.from_pretrained(
            str((ROOT / self.cfg["path"]).resolve()),
            dtype=dtype,
            device_map=self.cfg.get("device", "cpu"),
            max_inference_batch_size=1,
            max_new_tokens=self.cfg.get("max_new_tokens", 512),
        )

    def transcribe(self, wav, lang):
        res = self.model.transcribe(audio=[(wav, SR)], language=[self._names[lang]])
        return res[0].text.strip()


class IndicConformerEngine(Engine):
    """AI4Bharat IndicConformer 600M (Tamil, Hindi and 20 other Indian languages)."""
    name = "indic"
    languages = ("hi", "ta")
    max_chunk_s = 25.0

    def load(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            self.cfg.get("repo", "ai4bharat/indic-conformer-600m-multilingual"),
            trust_remote_code=True,
        )

    def transcribe(self, wav, lang):
        import torch
        out = self.model(torch.from_numpy(wav.copy()).unsqueeze(0), lang, self.cfg.get("decoder", "rnnt"))
        if isinstance(out, (list, tuple)):
            out = out[0]
        return str(out).strip()


class WhisperEngine(Engine):
    """faster-whisper. Not gated, covers all three languages."""
    name = "whisper"
    languages = ("en", "hi", "ta")
    max_chunk_s = None

    def load(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            self.cfg.get("model", "large-v3"),
            device=self.cfg.get("device", "auto"),
            compute_type=self.cfg.get("compute_type", "int8"),
        )

    def transcribe(self, wav, lang):
        segments, _ = self.model.transcribe(
            wav, language=lang, beam_size=5, vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


ENGINES = {c.name: c for c in (R2T2Engine, IndicConformerEngine, WhisperEngine)}
