"""Download the Tamil fine-tuned Whisper model and convert it for fast CPU inference.

    python prepare_tamil_model.py

Source: vasista22/whisper-tamil-large-v2 (IIT Madras Speech Lab, Apache-2.0, not gated).
Downloads about 6.2 GB, converts it to an int8 CTranslate2 model (about 1.6 GB) in
models/whisper-tamil-large-v2-ct2, then the 6 GB original can be deleted.
"""
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
REPO = "vasista22/whisper-tamil-large-v2"
OUT = ROOT / "models" / "whisper-tamil-large-v2-ct2"


def main():
    if (OUT / "model.bin").exists():
        print("Already converted:", OUT)
        return
    print("Downloading", REPO, "(about 6.2 GB) ...")
    src = snapshot_download(REPO, allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])
    print("Converting to int8 CTranslate2 ...")
    converter = Path(sys.executable).parent / ("ct2-transformers-converter.exe" if sys.platform == "win32"
                                              else "ct2-transformers-converter")
    subprocess.run([str(converter), "--model", src, "--output_dir", str(OUT),
                    "--quantization", "int8", "--copy_files", "preprocessor_config.json"],
                   check=True)
    # faster-whisper reads tokenizer.json from the model folder; this repo only ships vocab.json/merges.txt
    from transformers import WhisperTokenizerFast
    WhisperTokenizerFast.from_pretrained(src).backend_tokenizer.save(str(OUT / "tokenizer.json"))
    print("Done:", OUT)


if __name__ == "__main__":
    main()
