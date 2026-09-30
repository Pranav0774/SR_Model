"""Audio loading and chunking."""
import shutil
import subprocess

import numpy as np

SR = 16000


def load_audio(path: str) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32 using ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH; install it to read audio files.")
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise ValueError("That file could not be read as audio. Try WAV, MP3, M4A, FLAC, OGG or WebM.")
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    if wav.size == 0:
        raise ValueError("The audio file is empty.")
    return wav


def split_chunks(wav: np.ndarray, max_s: float = 28.0, min_s: float = 6.0) -> list[np.ndarray]:
    """Split long audio into pieces of at most max_s seconds.

    Cuts are placed at the quietest 100 ms window in the last part of each piece,
    so words are less likely to be cut in half.
    """
    max_n, min_n = int(max_s * SR), int(min_s * SR)
    if len(wav) <= max_n:
        return [wav]
    win = int(0.1 * SR)
    chunks, start = [], 0
    while len(wav) - start > max_n:
        lo, hi = start + min_n, start + max_n
        seg = np.abs(wav[lo:hi])
        usable = (len(seg) // win) * win
        energy = seg[:usable].reshape(-1, win).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * win + win // 2
        chunks.append(wav[start:cut])
        start = cut
    chunks.append(wav[start:])
    return [c for c in chunks if len(c) > int(0.2 * SR)]
