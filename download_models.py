"""Download every model the project uses. Safe to re-run; finished files are skipped.

    python download_models.py            # R2T2 + IndicConformer + Whisper small + NLLB
    python download_models.py --whisper-large   # also Whisper large-v3 (fallback engine)
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper-large", action="store_true")
    args = ap.parse_args()

    print("1/4  R2T2 (English, Hindi) ...")
    snapshot_download("netease-youdao/Confucius4-R2T2", local_dir=ROOT / "models" / "r2t2")

    print("2/4  IndicConformer (Tamil, Hindi) ...")
    try:
        snapshot_download("ai4bharat/indic-conformer-600m-multilingual")
    except Exception as e:
        sys.exit(
            f"\nCould not download IndicConformer: {type(e).__name__}\n"
            "This repository is gated. Do this once:\n"
            "  1. Open https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual and accept the terms.\n"
            "  2. Create a read token at https://huggingface.co/settings/tokens\n"
            "  3. Run:  huggingface-cli login\n"
            "Then run this script again."
        )

    print("3/4  Whisper (language detection) ...")
    from faster_whisper import download_model
    download_model("small")
    if args.whisper_large:
        download_model("large-v3")

    print("4/4  NLLB-200 translation model (Hindi/Tamil to English, about 2.5 GB, licence CC-BY-NC 4.0) ...")
    snapshot_download("facebook/nllb-200-distilled-600M",
                      allow_patterns=["*.json", "*.model", "pytorch_model.bin"])
    print("Done.")


if __name__ == "__main__":
    main()
